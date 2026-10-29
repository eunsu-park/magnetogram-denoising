"""Data, normalization and metrics.

Data layout under `--data-root` (see README):

    manifests/frames.csv   split, hour_utc, frame_index, file   one row per 45 s frame
    manifests/stacks.csv   split, hour_utc, file                one row per hourly group
    <file> of a frame      npz, key `data`: (512, 512) float16, G
    <file> of a stack      npz, keys `input` (the single frame) and `stack` (the 21-frame
                           stack): (512, 512) float32, G

The datasets draw every random number in the same order as the code that trained the published
checkpoints; changing that order changes the trained weights.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import fourier_shift
from scipy.signal import convolve2d
from torch.utils.data import Dataset


# --- Manifests and files -----------------------------------------------------------------------

def _rows(data_root, name, split):
    with open(Path(data_root, "manifests", name), newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def frame_groups(data_root, split):
    """Hourly groups of a split, in time order; each group is its frame files in time order."""
    groups = {}
    for r in _rows(data_root, "frames.csv", split):
        groups.setdefault(r["hour_utc"], []).append((int(r["frame_index"]), r["file"]))
    return [[str(Path(data_root, f)) for _, f in sorted(groups[h])] for h in sorted(groups)]


def stack_files(data_root, split):
    """Stack files of a split, in time order."""
    rows = sorted(_rows(data_root, "stacks.csv", split), key=lambda r: r["hour_utc"])
    return [str(Path(data_root, r["file"])) for r in rows]


def load_frame(path):
    """One frame (float16, G)."""
    with np.load(path) as z:
        return z["data"]


def load_stack(path):
    """(input, stack) of one hourly group (float32, G)."""
    with np.load(path) as z:
        return np.asarray(z["input"], np.float32), np.asarray(z["stack"], np.float32)


def seed_worker(worker_id):
    """Give each DataLoader worker its own NumPy generator, seeded by PyTorch."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


def band_limited_shift(a, offset):
    """Shift `a` by (dy, dx) px with a Fourier phase ramp (keeps the power spectrum).
    The outermost ceil(|offset|) pixels wrap around; callers trim them."""
    return np.fft.ifft2(fourier_shift(np.fft.fft2(a), offset)).real


def align_and_stack(frames, ref_idx):
    """Shift every frame as a whole onto frames[ref_idx] (phase correlation, 1/10 px; Fourier
    shift) and average: the stacked reference. No scaling or distortion is applied."""
    from skimage.registration import phase_cross_correlation
    ref = frames[ref_idx]
    acc = np.zeros_like(ref, dtype=np.float64)
    for f in frames:
        offset, _, _ = phase_cross_correlation(ref, f, upsample_factor=10)
        acc += band_limited_shift(f, offset)
    return (acc / len(frames)).astype(np.float32)


# --- Training datasets -------------------------------------------------------------------------

class FramePairs(Dataset):
    """Noise2Noise: (frame A, frame B) crops from one hourly group, `pair_gap` frames apart.

    Mixed-sigma augmentation adds independent Gaussian noise of one random sigma to A and B.
    With `align`, B is co-aligned to A (phase correlation, Fourier shift) on a crop with a
    24 px margin that is trimmed afterwards.
    """

    def __init__(self, groups, length, seed, crop=256, scale=1000.0, pair_gap=1, align=False,
                 noise_aug_max=0.0, margin=24):
        self.groups, self.length, self.crop, self.scale = groups, length, crop, scale
        self.pair_gap, self.align, self.noise_aug_max = max(1, pair_gap), align, noise_aug_max
        self.margin = margin if align else 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = self.rng
        group = self.groups[rng.integers(0, len(self.groups))]
        n = len(group)
        ia = int(rng.integers(0, n))
        cands = [k for k in (ia + self.pair_gap, ia - self.pair_gap) if 0 <= k < n]
        if not cands:
            cands = [k for k in (ia + 1, ia - 1) if 0 <= k < n]
        ib = int(rng.choice(cands)) if cands else ia
        a_full, b_full = load_frame(group[ia]), load_frame(group[ib])
        h, w = a_full.shape
        c = min(self.crop, h, w)
        m = self.margin if (self.align and h - c > 2 * self.margin) else 0
        top = int(rng.integers(m, max(m + 1, h - c - m)))
        left = int(rng.integers(m, max(m + 1, w - c - m)))
        if m:
            from skimage.registration import phase_cross_correlation
            aw = np.asarray(a_full[top - m:top + c + m, left - m:left + c + m], dtype=np.float32)
            bw = np.asarray(b_full[top - m:top + c + m, left - m:left + c + m], dtype=np.float32)
            bw = band_limited_shift(bw, phase_cross_correlation(aw, bw, upsample_factor=10)[0])
            a = aw[m:m + c, m:m + c] / self.scale
            b = bw[m:m + c, m:m + c] / self.scale
        else:
            # The crop position is drawn a second time here (the first draw is unused); kept
            # so that the published checkpoints are reproduced.
            top = int(rng.integers(0, max(1, h - c)))
            left = int(rng.integers(0, max(1, w - c)))
            a = np.asarray(a_full[top:top + c, left:left + c], dtype=np.float32) / self.scale
            b = np.asarray(b_full[top:top + c, left:left + c], dtype=np.float32) / self.scale
        if self.noise_aug_max > 0:
            sigma = rng.uniform(0.0, self.noise_aug_max) / self.scale
            a = a + rng.normal(0.0, sigma, a.shape)
            b = b + rng.normal(0.0, sigma, b.shape)
        return (torch.from_numpy(a.astype(np.float32))[None],
                torch.from_numpy(b.astype(np.float32))[None])


class FrameMasked(Dataset):
    """Noise2Void: one frame crop with blind spots, each replaced by a random neighbour."""

    def __init__(self, groups, length, seed, crop=256, scale=1000.0, mask_frac=0.005, window=5):
        self.groups, self.length, self.crop, self.scale = groups, length, crop, scale
        self.mask_frac, self.window = mask_frac, window
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = self.rng
        group = self.groups[rng.integers(0, len(self.groups))]
        frame = load_frame(group[int(rng.integers(0, len(group)))])
        h, w = frame.shape
        c = min(self.crop, h, w)
        top = int(rng.integers(0, max(1, h - c)))
        left = int(rng.integers(0, max(1, w - c)))
        target = np.asarray(frame[top:top + c, left:left + c], dtype=np.float32) / self.scale
        n = max(1, int(round(self.mask_frac * c * c)))
        ys, xs = np.unravel_index(rng.choice(c * c, size=n, replace=False), (c, c))
        half = self.window // 2
        masked, mask = target.copy(), np.zeros((c, c), dtype=bool)
        for y, x in zip(ys.tolist(), xs.tolist()):
            while True:  # a non-zero offset inside the window
                dy, dx = int(rng.integers(-half, half + 1)), int(rng.integers(-half, half + 1))
                if dy or dx:
                    break
            masked[y, x] = target[min(max(y + dy, 0), c - 1), min(max(x + dx, 0), c - 1)]
            mask[y, x] = True
        return (torch.from_numpy(masked.astype(np.float32))[None],
                torch.from_numpy(target.astype(np.float32))[None], torch.from_numpy(mask)[None])


class StackPairs(Dataset):
    """Supervised: (input frame, 21-frame stack) crops.

    `norm="scale"` divides by `scale` (supervised U-Net; optional mixed-sigma noise on the input);
    `norm="clip"` clips to +/- `clip` G and maps to [-1, 1] (Park 2020).
    """

    def __init__(self, files, length, seed, crop=256, norm="scale", scale=1000.0, clip=100.0,
                 noise_aug_max=0.0):
        self.files, self.length, self.crop = files, length, crop
        self.norm, self.scale, self.clip, self.noise_aug_max = norm, scale, clip, noise_aug_max
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = self.rng
        x, y = load_stack(self.files[rng.integers(0, len(self.files))])
        h, w = x.shape
        c = min(self.crop, h, w)
        top = int(rng.integers(0, max(1, h - c + 1)))
        left = int(rng.integers(0, max(1, w - c + 1)))
        x, y = x[top:top + c, left:left + c], y[top:top + c, left:left + c]
        if self.norm == "clip":
            x, y = to_clip(x, self.clip), to_clip(y, self.clip)
        else:
            x, y = x / self.scale, y / self.scale
            if self.noise_aug_max > 0:
                sigma = rng.uniform(0.0, self.noise_aug_max) / self.scale
                x = x + rng.normal(0.0, sigma, x.shape)
        return (torch.from_numpy(x.astype(np.float32))[None],
                torch.from_numpy(y.astype(np.float32))[None])


# --- Park 2020 normalization -------------------------------------------------------------------

def to_clip(b, clip=100.0):
    """G -> [-1, 1]: clip to +/- clip G."""
    return np.clip(b, -clip, clip) / clip


def restore_strong(denoised, original, lo=80.0, hi=120.0):
    """Take pixels with |original| > hi from the original, blending linearly over [lo, hi]
    (the clipped network has no valid output there)."""
    w = np.clip((np.abs(original) - lo) / (hi - lo), 0.0, 1.0)
    return (1.0 - w) * denoised + w * original


# --- Metrics -----------------------------------------------------------------------------------

def robust_sigma(x):
    """1.4826 * median absolute deviation from the median."""
    a = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def psnr(image, reference, data_range):
    """Peak signal-to-noise ratio (dB)."""
    diff = image.astype(np.float64) - reference.astype(np.float64)
    return float(10.0 * np.log10(data_range * data_range / float((diff * diff).mean())))


def ssim(image, reference, data_range, win_size=11, sigma=1.5):
    """Single-scale SSIM (Wang et al. 2004): Gaussian 11x11 window, 'valid' convolution."""
    x, y = image.astype(np.float64), reference.astype(np.float64)
    g = np.exp(-(np.arange(win_size, dtype=np.float64) - (win_size - 1) / 2.0) ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    win = np.outer(g, g)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = convolve2d(x, win, mode="valid"), convolve2d(y, win, mode="valid")
    sxx = convolve2d(x * x, win, mode="valid") - mx * mx
    syy = convolve2d(y * y, win, mode="valid") - my * my
    sxy = convolve2d(x * y, win, mode="valid") - mx * my
    lum = (2 * mx * my + c1) / (mx * mx + my * my + c1)
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    return float((lum * cs).mean())
