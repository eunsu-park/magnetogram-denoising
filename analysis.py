"""The numbers of the paper that test.py does not print, the paper's tables, and the figure data.

    python analysis.py --data-root <data> --ckpt-dir <checkpoints> --out-dir results --part table1
    python analysis.py ... --part table1 table2 signal noise figures

Each part prints its numbers and writes, under --out-dir:
  tables/table<N>.csv and tables/table<N>.txt   the paper's tables (machine-readable and as laid out)
  numbers/<part>.txt                            every other number of the part, one per line
  outputs/<method>/<split>/<file>.npz           the outputs it scored (central 256 px; key `output`)
  figures/data/fig-*.npz|csv                    the figure data (part `figures`), read by figures.py

--ckpt-dir holds one folder per checkpoint (CKPTS below), each with model.pt. Methods that are not
trained: adaptive TV (scikit-image), BM3D (the `bm3d` package, licensed for non-commercial use only
and therefore optional: without it the BM3D rows are read from <data-root>/outputs/bm3d when
present, else left out), spatiotemporal noise gating (DeForest 2017), Gaussian blur.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import pipeline
import test

# Checkpoints of the paper (folder names under --ckpt-dir).
CKPTS = {
    "n2n": "n2n_noaug",                       # N2N, no augmentation
    "n2n_mixsigma": "n2n_mixsigma_s1",        # the model of the paper
    **{f"n2n_mixsigma_s{i}": f"n2n_mixsigma_s{i}" for i in range(2, 6)},
    "gap2": "n2n_gap2", "gap3": "n2n_gap3",
    **{f"gap6_s{i}": f"n2n_gap6_s{i}" for i in range(1, 4)},
    "depth1": "n2n_depth1", "depth3": "n2n_depth3", "w16": "n2n_w16", "w64": "n2n_w64",
    "n2v": "n2v",
    "supervised": "supervised",
    "supervised_mixsigma": "supervised_mixsigma",
    "park2020": "park2020",
}
LABELS = {"input": "Noisy input", "n2n": "N2N", "n2n_mixsigma": "N2N, mixed-sigma",
          "park2020": "Park et al. (2020), supervised", "supervised": "Supervised U-Net, MSE",
          "supervised_mixsigma": "Supervised U-Net, MSE, mixed-sigma", "tv": "Adaptive TV",
          "bm3d": "BM3D", "noise_gate": "Noise gating, 16-frame input",
          "stack-3": "3-frame stack", "stack-5": "5-frame stack", "stack-7": "7-frame stack"}


# --- Methods that are not trained --------------------------------------------------------------

def tv(b, sigma):
    """Adaptive TV: TV-Chambolle (weight 0.1) on the frame normalised to a noise level of 0.1."""
    from skimage.restoration import denoise_tv_chambolle
    scale = (sigma / 0.1) or 1.0
    return np.asarray(denoise_tv_chambolle((b / scale).astype(np.float32), weight=0.1), np.float32) * scale


def bm3d(b, sigma):
    import bm3d as _bm3d
    return _bm3d.bm3d(b, sigma_psd=sigma).astype(np.float32)


def has_bm3d():
    try:
        import bm3d  # noqa: F401
        return True
    except ImportError:
        return False


def noise_gate(cube, gamma=3.0, chunk=(12, 16, 16)):
    """Spatiotemporal Fourier noise gating (DeForest 2017): Hann-apodised chunks with 50% overlap,
    components below gamma times the white-noise level removed (noise sigma from adjacent-frame
    differences), overlap-add normalised by the summed squared window."""
    sigma = pipeline.robust_sigma(np.diff(cube, axis=0)) / np.sqrt(2.0)
    ct, cy, cx = chunk
    w = np.ones(chunk, np.float32)
    for ax, n in enumerate(chunk):
        w *= np.hanning(n).astype(np.float32).reshape([-1 if a == ax else 1 for a in range(3)])
    thresh = gamma * sigma * np.sqrt(float((w ** 2).sum()))
    out, norm = np.zeros_like(cube, np.float32), np.zeros_like(cube, np.float32)
    T, H, W = cube.shape
    for t0 in list(range(0, max(T - ct, 0) + 1, ct // 2)) or [0]:
        for y0 in range(0, H - cy + 1, cy // 2):
            for x0 in range(0, W - cx + 1, cx // 2):
                F = np.fft.fftn(cube[t0:t0 + ct, y0:y0 + cy, x0:x0 + cx] * w)
                mask = np.abs(F) > thresh
                mask[0, 0, 0] = True
                g = np.fft.ifftn(F * mask).real.astype(np.float32)
                out[t0:t0 + ct, y0:y0 + cy, x0:x0 + cx] += g * w
                norm[t0:t0 + ct, y0:y0 + cy, x0:x0 + cx] += w * w
    return out / np.maximum(norm, 1e-6)


def acf1(x):
    """Lag-1 spatial autocorrelation (mean of the two axes)."""
    x = x - x.mean()
    v = float((x * x).mean())
    return float(((x[:, :-1] * x[:, 1:]).mean() + (x[:-1, :] * x[1:, :]).mean()) / (2 * v))


# --- Helpers -----------------------------------------------------------------------------------

def load_models(ckpt_dir, names, dev):
    out = {}
    for n in names:
        p = Path(ckpt_dir, CKPTS[n], "model.pt")
        if p.exists():
            out[n] = test.load_denoiser(p, dev)
        else:
            print(f"  ({n}: {p} not found, skipped)")
    return out


def group_mean(rows, method, key, **where):
    v = [r[key] for r in rows if r["method"] == method and all(r.get(k) == x for k, x in where.items())]
    return float(np.mean(v)) if v else float("nan")


def write_table(out_dir, name, header, rows, title):
    """tables/<name>.csv (one row per line of the table) and tables/<name>.txt (as laid out)."""
    d = Path(out_dir, "tables")
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    width = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    lines = [title, "  ".join(h.ljust(n) for h, n in zip(header, width))]
    for r in rows:
        lines.append("  ".join((str(x).ljust(n) if i == 0 else str(x).rjust(n)) for i, (x, n) in enumerate(zip(r, width))))
    (d / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


class Numbers:
    """Collects 'name: value' lines and writes them to numbers/<part>.txt."""

    def __init__(self, out_dir, part):
        self.path = Path(out_dir, "numbers", f"{part}.txt")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = []

    def __call__(self, line):
        print(line, flush=True)
        self.lines.append(line)
        self.path.write_text("\n".join(self.lines) + "\n")


# --- Part: Table 1 and the HMI numbers ---------------------------------------------------------

def part_table1(args, dev):
    num = Numbers(args.out_dir, "table1")
    root, outs = args.data_root, Path(args.out_dir, "outputs")
    models = load_models(args.ckpt_dir, ["n2n", "n2n_mixsigma", "park2020", "supervised", "supervised_mixsigma"], dev)
    classical = {"tv": lambda r: tv(r, 0.7 * pipeline.robust_sigma(r - r.mean()))}   # factor 0.7 from validation
    if has_bm3d():
        classical["bm3d"] = lambda r: bm3d(r, pipeline.robust_sigma(r))
    methods = {"input": lambda b: b, **models, **classical}

    # (a) 21-frame stack reference: test (2013 December) and test_xcycle (2019-2020)
    rows_a = test.table_1a(methods, root, ["test", "test_xcycle"], save_dir=outs)
    # noise gating on the 21 frames of each test group; the input frame is scored
    for group, f in zip(pipeline.frame_groups(root, "test"), pipeline.stack_files(root, "test")):
        cube = np.stack([np.asarray(pipeline.load_frame(p), np.float32) for p in group])
        out = test.center(noise_gate(cube, 3.0)[len(group) // 2], 256)
        y = test.center(pipeline.load_stack(f)[1], 256)
        rows_a.append({"split": "test", "file": Path(f).name, "method": "noise_gate", **test.quality(out, y)})
        d = outs / "noise_gate" / "test"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / Path(f).name, output=out)

    # (b), (c): the frame-level references; TV and BM3D get the noise of input - reference
    ref_methods = {"tv": lambda x, ref: tv(x, pipeline.robust_sigma(x - ref))}
    if has_bm3d():
        ref_methods["bm3d"] = lambda x, ref: bm3d(x, pipeline.robust_sigma(x - ref))
    rows_bc = {k: test.table_1bc({"input": lambda b: b, **models}, root, k, ref_methods=ref_methods)
               for k in ("1b", "1c")}

    order = ["input", "n2n", "n2n_mixsigma", "park2020", "supervised", "supervised_mixsigma", "tv", "bm3d"]
    table = []
    for block, rows, extra in (("(a) 21-frame stack reference", [r for r in rows_a if r["split"] == "test"], ["noise_gate"]),
                               ("(b) disjoint even-frame reference", rows_bc["1b"], ["stack-3", "stack-5", "stack-7"]),
                               ("(c) temporally separated reference, >= 270 s", rows_bc["1c"], ["stack-3", "stack-5", "stack-7"])):
        for m in order + extra:
            if any(r["method"] == m for r in rows):
                table.append([block, LABELS[m], f"{group_mean(rows, m, 'psnr'):.2f}",
                              f"{group_mean(rows, m, 'ssim'):.3f}", f"{group_mean(rows, m, 'resid_sigma'):.2f}"])
    write_table(args.out_dir, "table1", ["block", "method", "PSNR_dB", "SSIM", "residual_sigma_G"], table,
                "Table 1. Denoising performance on the HMI test set (N = 672 hours)")
    xc = [r for r in rows_a if r["split"] == "test_xcycle"]
    num("2019-2020 test set (1,947 hours), PSNR dB: " + ", ".join(
        f"{LABELS[m]} {group_mean(xc, m, 'psnr'):.2f}" for m in order if any(r["method"] == m for r in xc)))

    # PSNR margins of the mixed-sigma model: 24-hour block bootstrap (5,000 resamples, seed 0) for
    # (a) and the 2019-2020 set; paired bootstrap over hours (10,000, seed 0) for (b) and (c).
    rng = np.random.default_rng(0)
    for split in ("test", "test_xcycle"):
        sub = [r for r in rows_a if r["split"] == split]
        p = {m: np.array([r["psnr"] for r in sub if r["method"] == m]) for m in ("n2n_mixsigma", "bm3d", "tv", "park2020")}
        for base in ("bm3d", "tv", "park2020"):
            if len(p[base]) == 0:
                continue
            d = p["n2n_mixsigma"] - p[base]
            n = len(d)
            [d[rng.integers(0, n, n)].mean() for _ in range(5000)]          # the i.i.d. bootstrap drawn first
            blocks = [d[i:i + 24] for i in range(0, n, 24)]
            blk = [np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))]).mean() for _ in range(5000)]
            num(f"{split}: mixed-sigma - {LABELS[base]} PSNR {d.mean():+.2f} dB, 24-h block bootstrap 95% CI "
                f"[{np.percentile(blk, 2.5):+.2f}, {np.percentile(blk, 97.5):+.2f}]")
    for k in ("1b", "1c"):
        rows = rows_bc[k]
        for base in ("tv", "bm3d"):
            a = np.array([r["psnr"] for r in rows if r["method"] == "n2n_mixsigma"])
            b = np.array([r["psnr"] for r in rows if r["method"] == base])
            if len(b) == 0:
                continue
            d = a - b
            boots = np.random.default_rng(0).choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
            num(f"Table {k}: mixed-sigma - {LABELS[base]} PSNR {d.mean():+.3f} dB, paired bootstrap 95% CI "
                f"[{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")

    # lag-1 autocorrelation of the removed component (input - output), test set
    stacks = pipeline.stack_files(root, "test")
    acc = {}
    for f in stacks:
        x, y = (test.center(a, 256) for a in pipeline.load_stack(f))
        for m in ("n2n_mixsigma", "n2n", "tv", "bm3d", "park2020"):
            p = outs / m / "test" / Path(f).name
            if p.exists():
                with np.load(p) as z:
                    acc.setdefault(m, []).append(acf1(x - z["output"]))
        acc.setdefault("input - stack", []).append(acf1(x - y))
    num("lag-1 autocorrelation of the removed component: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in acc.items()))

    # the raw-frame noise estimate vs the noise measured against the stack (48 test hours)
    sub = stacks[::max(1, len(stacks) // 48)][:48]
    s_raw, s_true = [], []
    for f in sub:
        x, y = (test.center(a, 256) for a in pipeline.load_stack(f))
        s_raw.append(pipeline.robust_sigma(x))
        s_true.append(pipeline.robust_sigma(x - y))
    num(f"noise estimate, {len(sub)} test hours: robust sigma of the frame {np.mean(s_raw):.2f} G vs measured "
        f"against the stack {np.mean(s_true):.2f} G ({100 * (np.mean(s_raw) / np.mean(s_true) - 1):.0f}% larger)")

    # the TV / BM3D noise factor and the noise-gating gamma, chosen on the validation month
    val = pipeline.stack_files(root, "val")
    idx = np.unique(np.linspace(0, len(val) - 1, min(48, len(val))).astype(int))
    factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    sweep = {(m, f): [] for m in ("tv", "bm3d") for f in factors}
    for i in idx:
        x, y = (test.center(a, 256) for a in pipeline.load_stack(val[i]))
        dr = float(np.percentile(y, 99.5) - np.percentile(y, 0.5)) or 1.0
        s0 = pipeline.robust_sigma(x)
        for f in factors:
            sweep[("tv", f)].append(pipeline.psnr(tv(x, f * s0), y, dr))
            if has_bm3d():
                sweep[("bm3d", f)].append(pipeline.psnr(bm3d(x, f * s0), y, dr))
    for m in ("tv", "bm3d"):
        if sweep[(m, 1.0)]:
            num(f"validation ({len(idx)} hours), {LABELS[m]} PSNR by noise factor: " +
                ", ".join(f"{f:g} {np.mean(sweep[(m, f)]):.2f}" for f in factors) +
                f"; best {max(factors, key=lambda f: np.mean(sweep[(m, f)])):g}")
    groups = pipeline.frame_groups(root, "val")
    vstacks = pipeline.stack_files(root, "val")
    gi = np.unique(np.linspace(0, len(groups) - 1, 16).astype(int))
    gsweep = {g: [] for g in (2.0, 2.5, 3.0, 3.5, 4.0)}
    for i in gi:
        cube = np.stack([np.asarray(pipeline.load_frame(p), np.float32) for p in groups[i]])
        y = test.center(pipeline.load_stack(vstacks[i])[1], 256)
        for g in gsweep:
            gsweep[g].append(test.quality(test.center(noise_gate(cube, g)[len(groups[i]) // 2], 256), y))
    num(f"validation ({len(gi)} hours), noise gating by gamma: " + ", ".join(
        f"{g:g} {np.mean([q['psnr'] for q in v]):.2f} dB / {np.mean([q['resid_sigma'] for q in v]):.3f} G" for g, v in gsweep.items()))

    # the five training runs and the 270 s models (Table 1a protocol); the ablations (full 512 px)
    seeds = load_models(args.ckpt_dir, ["n2n_mixsigma"] + [f"n2n_mixsigma_s{i}" for i in range(2, 6)]
                        + [f"gap6_s{i}" for i in range(1, 4)], dev)
    rs = test.table_1a(seeds, root, ["test"])
    for m in seeds:
        num(f"Table 1a protocol, {m}: PSNR {group_mean(rs, m, 'psnr'):.2f} dB, SSIM {group_mean(rs, m, 'ssim'):.3f}, "
            f"residual sigma {group_mean(rs, m, 'resid_sigma'):.2f} G")
    five = ["n2n_mixsigma"] + [f"n2n_mixsigma_s{i}" for i in range(2, 6)]
    for key, fmt in (("psnr", ".2f"), ("ssim", ".3f"), ("resid_sigma", ".2f")):
        v = [group_mean(rs, m, key) for m in five if m in seeds]
        num(f"five training runs, {key}: {min(v):{fmt}}-{max(v):{fmt}}")
    abl = load_models(args.ckpt_dir, ["n2n_mixsigma", "gap2", "gap3", "depth1", "depth3", "w16", "w64", "n2n", "n2v"], dev)
    ra = test.table_1a(abl, root, ["test", "val"], crop=512)
    for split in ("test", "val"):
        num(f"ablations, residual sigma over the full 512 px ({split}): " + ", ".join(
            f"{m} {group_mean(ra, m, 'resid_sigma', split=split):.3f} G" for m in abl))


# --- Part: Table 2 and the cross-instrument numbers --------------------------------------------

def blur(a, s):
    """Gaussian blur that keeps NaN where the input has none."""
    from scipy.ndimage import gaussian_filter
    out = gaussian_filter(np.nan_to_num(a), s)
    out[~np.isfinite(a)] = np.nan
    return out


def noise_acf1(a, w):
    """Lag-1 autocorrelation (along rows) of the high-pass (a - 5 px Gaussian) over the mask."""
    from scipy.ndimage import gaussian_filter
    x = np.nan_to_num(a) - gaussian_filter(np.nan_to_num(a), 5.0)
    x = np.where(w, x, 0.0)
    v = float((x * x)[w].mean())
    return float((x[:, :-1] * x[:, 1:])[w[:, :-1] & w[:, 1:]].mean() / v)


def boot_reduction(before, after, rng, n_boot=10000):
    """Reduction 1 - mean(after)/mean(before) with a percentile interval over resampled pairs."""
    i = rng.integers(0, len(before), (n_boot, len(before)))
    red = 100 * (1 - after[i].mean(1) / before[i].mean(1))
    return 100 * (1 - after.mean() / before.mean()), np.percentile(red, 2.5), np.percentile(red, 97.5)


def part_table2(args, dev):
    num = Numbers(args.out_dir, "table2")
    root = args.data_root
    dis, weak = test.disagree, test.weak
    panel = ["n2n_mixsigma"] + [f"n2n_mixsigma_s{i}" for i in range(2, 6)] + [f"gap6_s{i}" for i in range(1, 4)] \
        + ["n2n", "supervised", "supervised_mixsigma"]
    models = load_models(args.ckpt_dir, panel, dev)
    den = models["n2n_mixsigma"]
    bm = has_bm3d()
    rng = np.random.default_rng(0)

    # ---- HMI-MDI on the MDI grid
    mdi, panel_mdi = [], {m: [] for m in models}
    for f in sorted(Path(root, "pairs", "mdi").glob("*.npz")):
        with np.load(f) as z:
            h, m, disk = z["hmi"], z["mdi"], z["disk"]
            h2 = z["hmi_ref"] if "hmi_ref" in z.files else None
            meta = json.loads(str(z["meta"]))
        w = disk & (np.abs(h) < 30) & (np.abs(m) < 30)
        hd, md = den(h), den(m)
        r = {"in40": meta["in_40_pair_subset"], "interval": meta["mdi_interval_s"],
             "before": dis(h, m, w), "after": dis(hd, md, w), "mdi_noise": pipeline.robust_sigma(m[w])}
        if h2 is not None:
            w2 = disk & (np.abs(h) < 30) & (np.abs(h2) < 30)
            r.update(ref_before=dis(h, h2, w2), ref_after=dis(hd, den(h2), w2))
        if r["in40"]:
            s_m, s_h = disk & (np.abs(m) > 70), disk & (np.abs(h) > 70)
            fr = lambda a: test.flux_ratio(a, h, s_m)                         # noqa: E731
            r.update(flux_model=fr(hd), flux_model_on_mdi=test.flux_ratio(md, m, s_h),
                     rescaled_before=dis(h, m / 1.40, w), rescaled_after=dis(hd, md / 1.40, w))
            for t in (20, 50):
                wt = disk & (np.abs(h) < t) & (np.abs(m) < t)
                r[f"before_{t}"], r[f"after_{t}"] = dis(h, m, wt), dis(hd, md, wt)
            for s in (2, 3):
                bh, bmdi = blur(h, s), blur(m, s)
                r.update({f"blur{s}": dis(bh, bmdi, w), f"blur{s}_hmi_only": dis(bh, m, w),
                          f"blur{s}_mdi_only": dis(h, bmdi, w), f"flux_blur{s}": fr(bh)})
            for tag, sel in (("disk", disk), ("weak", w)):
                sh, sm = pipeline.robust_sigma(h[sel]), pipeline.robust_sigma(m[sel])
                th = tv(h, sh)
                r[f"tv_{tag}"], r[f"flux_tv_{tag}"] = dis(th, tv(m, sm), w), fr(th)
                if bm:
                    bh_ = bm3d(h, sh)
                    r[f"bm3d_{tag}"], r[f"flux_bm3d_{tag}"] = dis(bh_, bm3d(m, sm), w), fr(bh_)
            for name, fn in models.items():
                panel_mdi[name].append(dis(fn(h), fn(m), w))
        mdi.append(r)
    sel40 = [r for r in mdi if r["in40"]]
    M = lambda rows, k: float(np.nanmean([r[k] for r in rows if k in r]))   # noqa: E731
    b_all, a_all = np.array([r["before"] for r in mdi]), np.array([r["after"] for r in mdi])
    b40, a40 = np.array([r["before"] for r in sel40]), np.array([r["after"] for r in sel40])
    red, lo, hi = boot_reduction(b_all, a_all, rng)
    num(f"HMI-MDI, all {len(mdi)} pairs: {b_all.mean():.2f} -> {a_all.mean():.2f} G, reduction {red:.1f}% [{lo:.1f}, {hi:.1f}]")
    red40, lo40, hi40 = boot_reduction(b40, a40, rng)
    num(f"HMI-MDI, the 40-pair subset: {b40.mean():.2f} -> {a40.mean():.2f} G, reduction {red40:.1f}% [{lo40:.1f}, {hi40:.1f}]")
    for iv, name in ((30, "one-minute"), (300, "five-minute")):
        s = [r for r in mdi if r["interval"] == iv]
        b, a = M(s, "before"), M(s, "after")
        num(f"  {name} MDI magnetograms: n={len(s)}  {b:.2f} -> {a:.2f} G ({100 * (1 - a / b):.0f}%)")
    num(f"MDI noise: median over the pairs of the robust sigma of the weak-field MDI pixels {np.median([r['mdi_noise'] for r in mdi]):.1f} G")
    ref = [r for r in mdi if "ref_before" in r]
    rb, ra = M(ref, "ref_before"), M(ref, "ref_after")
    same = [r for r in sel40 if "ref_before" in r]
    num(f"HMI vs HMI at each pair's offset ({len(ref)} pairs): {rb:.2f} -> {ra:.2f} G ({100 * (1 - ra / rb):.0f}%); the same pairs "
        f"HMI-MDI {M(same, 'before'):.2f} -> {M(same, 'after'):.2f} G; ratio {M(same, 'after') / ra:.2f}")
    num(f"MDI / 1.40: {M(sel40, 'rescaled_before'):.2f} -> {M(sel40, 'rescaled_after'):.2f} G "
        f"({100 * (1 - M(sel40, 'rescaled_after') / M(sel40, 'rescaled_before')):.0f}%)")
    for t in (20, 50):
        b, a = M(sel40, f"before_{t}"), M(sel40, f"after_{t}")
        num(f"weak-field threshold {t} G: {b:.2f} -> {a:.2f} G ({100 * (1 - a / b):.0f}%)")
    for s in (2, 3):
        num(f"blur {s} px: both {M(sel40, f'blur{s}'):.2f} G, HMI only {M(sel40, f'blur{s}_hmi_only'):.2f}, MDI only {M(sel40, f'blur{s}_mdi_only'):.2f} G")
    num(f"strong-field flux retained (|B_MDI| > 70 G): model {M(sel40, 'flux_model'):.3f}, blur 2 px {M(sel40, 'flux_blur2'):.3f}; "
        f"on MDI (|B_HMI| > 70 G): model {M(sel40, 'flux_model_on_mdi'):.3f}")
    for tag in ("disk", "weak"):
        line = f"noise estimate over the {tag} pixels: TV {M(sel40, f'tv_{tag}'):.2f} G, flux {M(sel40, f'flux_tv_{tag}'):.3f}"
        if bm:
            line += f"; BM3D {M(sel40, f'bm3d_{tag}'):.2f} G, flux {M(sel40, f'flux_bm3d_{tag}'):.3f}"
        num(line)
    mdi_red = {n: 100 * (1 - np.mean(v) / b40.mean()) for n, v in panel_mdi.items()}
    num("HMI-MDI reduction, 40 pairs, by model: " + ", ".join(f"{n} {v:.1f}%" for n, v in mdi_red.items()))

    # ---- HMI-PHI
    phi, panel_phi = [], {m: [] for m in models}
    for f in sorted(Path(root, "pairs", "phi").glob("*.npz")):
        z = dict(np.load(f))
        meta = json.loads(str(z.pop("meta")))
        H, P, ph, hh, hu = z["hmi"], z["phi_on_hmi"], z["phi"], z["hmi_on_hrt"], z["hmi_on_hrt_wcs"]
        w = weak(H, P)
        Hd, Pd = den(H), den(P)
        strong = np.isfinite(H) & np.isfinite(P) & (np.abs(np.nan_to_num(P)) > 70)
        r = {**{k: meta[k] for k in ("sep_deg", "d_au", "fov_r", "shift_arcsec", "corr_before", "corr_after",
                                       "ear_tdel_s", "offset_s", "stray_light_corrected", "phi_date_avg_utc")},
             "before": dis(H, P, w), "after": dis(Hd, Pd, w), "flux_model": test.flux_ratio(Hd, H, strong, 100),
             "hmi_only": dis(Hd, P, w), "phi_only": dis(H, Pd, w), "acf_native": noise_acf1(H, w)}
        for s in (2, 3):
            r[f"blur{s}"], r[f"flux_blur{s}"] = dis(blur(H, s), blur(P, s), w), test.flux_ratio(blur(H, s), H, strong, 100)
        for t in (20, 50):
            wt = weak(H, P, t)
            r[f"before_{t}"], r[f"after_{t}"] = dis(H, P, wt), dis(Hd, Pd, wt)
        h0, p0 = np.nan_to_num(H), np.nan_to_num(P)
        fov = np.isfinite(H) & np.isfinite(P)
        for tag, sel in (("weak", w), ("fov", fov)):
            sh, sp = pipeline.robust_sigma(h0[sel]), pipeline.robust_sigma(p0[sel])
            th = tv(h0, sh)
            r[f"tv_{tag}"], r[f"flux_tv_{tag}"] = dis(th, tv(p0, sp), w), test.flux_ratio(th, H, strong, 100)
            if bm:
                bh_ = bm3d(h0, sh)
                r[f"bm3d_{tag}"], r[f"flux_bm3d_{tag}"] = dis(bh_, bm3d(p0, sp), w), test.flux_ratio(bh_, H, strong, 100)
        # flux retained measured on PHI: strong field selected on HMI (both finite / as in the first export)
        r["flux_model_on_phi"] = test.flux_ratio(Pd, P, fov & (np.abs(h0) > 70))
        r["flux_model_on_phi_nan0"] = test.flux_ratio(Pd, p0, np.abs(h0) > 70)
        w_h = weak(hh, ph)
        r.update(hrt_before=dis(hh, ph, w_h), hrt_blur2=dis(blur(hh, 2), blur(ph, 2), w_h),
                 acf_hrt=noise_acf1(hh, w_h))
        w_u = weak(hu, ph)
        r.update(wcs_before=dis(hu, ph, w_u), wcs_after=dis(den(hu), den(ph), w_u))
        inphi = np.isfinite(P)
        for name, fn in models.items():
            O = z["hmi_270s"]
            wo = weak(H, O) & np.isfinite(O) & inphi
            panel_phi[name].append((dis(fn(H), fn(P), w), dis(fn(H), fn(O), wo)))
        phi.append(r)
    b, a = np.array([r["before"] for r in phi]), np.array([r["after"] for r in phi])
    red, lo, hi = boot_reduction(b, a, rng)
    num(f"HMI-PHI, {len(phi)} epochs: {b.mean():.2f} -> {a.mean():.2f} G, reduction {red:.1f}% [{lo:.1f}, {hi:.1f}]")
    ref270 = np.array([v[1] for v in panel_phi["n2n_mixsigma"]])
    i = rng.integers(0, len(a), (10000, len(a)))
    ratio = a[i].mean(1) / ref270[i].mean(1)
    num(f"ratio to the 270 s reference: {a.mean() / ref270.mean():.3f} [{np.percentile(ratio, 2.5):.3f}, "
        f"{np.percentile(ratio, 97.5):.3f}]; per pair {np.min(a / ref270):.2f}-{np.max(a / ref270):.2f}")
    g = lambda k: np.array([r[k] for r in phi], dtype=float)                 # noqa: E731
    num(f"light-travel time {g('ear_tdel_s').min():.0f}-{g('ear_tdel_s').max():.0f} s; pair offset max {g('offset_s').max():.1f} s")
    num(f"co-alignment shift {g('shift_arcsec').min():.0f}-{g('shift_arcsec').max():.0f} arcsec, median {np.median(g('shift_arcsec')):.0f}; "
        f"correlation {g('corr_before').mean():.2f} -> {g('corr_after').mean():.2f}")
    num(f"separation {g('sep_deg').min():.1f}-{g('sep_deg').max():.1f} deg, median {np.median(g('sep_deg')):.1f}; distance "
        f"{g('d_au').min():.2f}-{g('d_au').max():.2f} AU (pixel ratio {1 / g('d_au').max():.1f}-{1 / g('d_au').min():.1f})")
    stray = np.array([r["stray_light_corrected"] for r in phi])
    year = np.array([r["phi_date_avg_utc"][:4] for r in phi])
    for name, s in [(y, year == y) for y in ("2022", "2023", "2024")] + [
            ("without limb (r < 0.9)", g("fov_r") < 0.9), ("stray-light corrected", stray), ("not corrected", ~stray)]:
        num(f"  {name:24s} n={s.sum():2d}  {b[s].mean():.2f} -> {a[s].mean():.2f} G ({100 * (1 - a[s].mean() / b[s].mean()):.1f}%)")
    num(f"one-sided: HMI only {g('hmi_only').mean():.2f} G, PHI only {g('phi_only').mean():.2f} G")
    for t in (20, 50):
        num(f"weak-field threshold {t} G: {g(f'before_{t}').mean():.2f} -> {g(f'after_{t}').mean():.2f} G")
    num(f"blur 2 px {g('blur2').mean():.2f} G, flux {np.nanmean(g('flux_blur2')):.3f}; model flux {np.nanmean(g('flux_model')):.3f}; "
        f"flux on PHI {g('flux_model_on_phi').mean():.3f} (NaN read as 0: {g('flux_model_on_phi_nan0').mean():.3f})")
    for tag in ("fov", "weak"):
        line = f"noise estimate over the {tag} pixels: TV {g(f'tv_{tag}').mean():.2f} G, flux {np.nanmean(g(f'flux_tv_{tag}')):.3f}"
        if bm:
            line += f"; BM3D {g(f'bm3d_{tag}').mean():.2f} G, flux {np.nanmean(g(f'flux_bm3d_{tag}')):.3f}"
        num(line)
    for x in ("sep_deg", "d_au", "fov_r"):
        num(f"Pearson r(residual, {x}) = {np.corrcoef(g(x), a)[0, 1]:+.2f}; r(residual / reference, {x}) = "
            f"{np.corrcoef(g(x), a / ref270)[0, 1]:+.2f}")
    num(f"HRT grid, WCS only: {g('wcs_before').mean():.2f} -> {g('wcs_after').mean():.2f} G "
        f"({100 * (1 - g('wcs_after').mean() / g('wcs_before').mean()):.0f}%); blur 2 px {g('hrt_blur2').mean():.2f} G "
        f"({100 * (1 - g('hrt_blur2').mean() / g('hrt_before').mean()):.0f}%)")
    num(f"lag-1 autocorrelation of the HMI noise: native {g('acf_native').mean():.2f}, on the HRT grid {g('acf_hrt').mean():.2f}")
    phi_red = {}
    for n, v in panel_phi.items():
        aft, r270 = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        phi_red[n] = (100 * (1 - aft.mean() / b.mean()), aft.mean() / r270.mean())
    num("HMI-PHI by model (reduction, ratio to its own 270 s reference): " +
        ", ".join(f"{n} {v[0]:.1f}% {v[1]:.2f}" for n, v in phi_red.items()))
    five = [n for n in ["n2n_mixsigma"] + [f"n2n_mixsigma_s{i}" for i in range(2, 6)] if n in models]
    gap = [n for n in (f"gap6_s{i}" for i in range(1, 4)) if n in models]
    num(f"five training runs: MDI {min(mdi_red[n] for n in five):.1f}-{max(mdi_red[n] for n in five):.1f}%, "
        f"PHI {min(phi_red[n][0] for n in five):.1f}-{max(phi_red[n][0] for n in five):.1f}%")
    if gap:
        num(f"270 s pairing ({len(gap)} runs): MDI {np.mean([mdi_red[n] for n in gap]):.1f} +/- {np.std([mdi_red[n] for n in gap], ddof=1):.1f}%, "
            f"PHI {np.mean([phi_red[n][0] for n in gap]):.1f} +/- {np.std([phi_red[n][0] for n in gap], ddof=1):.1f}% (sample s.d.)")
    ratios = [v[1] for v in phi_red.values()]
    num(f"ratio to the own 270 s reference over the {len(ratios)} models: {min(ratios):.2f}-{max(ratios):.2f}")
    num("training target x augmentation (MDI / PHI reduction): " + ", ".join(
        f"{n} {mdi_red[n]:.0f}/{phi_red[n][0]:.0f}%" for n in ("n2n", "n2n_mixsigma", "supervised", "supervised_mixsigma") if n in models))

    # ---- the table as printed
    rr = test.table_2bc({"n2n_mixsigma": den}, root)     # the 45 s / 270 s references and the HRT grid

    def row(block, label, before, after, base, flux=""):
        return [block, label, f"{before:.2f}" if before else "", f"{after:.2f}", f"{100 * (1 - after / base):.0f}%", flux]
    A, B, C = ("(a) HMI-MDI, MDI grid (40 pairs)", "(b) HMI-PHI, HMI grid (14 epochs)", "(c) HMI-PHI, HRT grid (14 epochs)")
    b40m, bm_, hb = b40.mean(), b.mean(), g("hrt_before").mean()
    rows = [row(A, "N2N, mixed-sigma", b40m, a40.mean(), b40m, f"{M(sel40, 'flux_model'):.2f}"),
            row(A, "Adaptive TV", None, M(sel40, "tv_disk"), b40m, f"{M(sel40, 'flux_tv_disk'):.2f}")]
    if bm:
        rows.append(row(A, "BM3D", None, M(sel40, "bm3d_disk"), b40m, f"{M(sel40, 'flux_bm3d_disk'):.2f}"))
    rows += [row(A, "Gaussian blur, 2 px", None, M(sel40, "blur2"), b40m, f"{M(sel40, 'flux_blur2'):.2f}"),
             row(A, "HMI vs. HMI, matched offsets", rb, ra, rb),
             row(B, "N2N, mixed-sigma", bm_, a.mean(), bm_, f"{np.nanmean(g('flux_model')):.2f}"),
             row(B, "Adaptive TV", None, g("tv_fov").mean(), bm_, f"{np.nanmean(g('flux_tv_fov')):.2f}")]
    if bm:
        rows.append(row(B, "BM3D", None, g("bm3d_fov").mean(), bm_, f"{np.nanmean(g('flux_bm3d_fov')):.2f}"))
    rows.append(row(B, "Gaussian blur, 2 px", None, g("blur2").mean(), bm_, f"{np.nanmean(g('flux_blur2')):.2f}"))
    for key, label in (("hmi_45s", "HMI vs. HMI, 45 s"), ("hmi_270s", "HMI vs. HMI, 270 s")):
        rb_ = test.mean(rr, f"{key}_before")
        rows.append(row(B, label, rb_, test.mean(rr, f"{key}_after"), rb_))
    rows += [row(C, "N2N, reproject then denoise", hb, test.mean(rr, "hrt_reproject_then_denoise"), hb),
             row(C, "N2N, denoise then reproject", None, test.mean(rr, "hrt_denoise_then_reproject"), hb),
             row(C, "Gaussian blur, 2 px", None, g("hrt_blur2").mean(), hb)]
    write_table(args.out_dir, "table2", ["block", "treatment", "before_G", "after_G", "reduction", "flux_retained"], rows,
                "Table 2. Cross-instrument weak-field disagreement before and after denoising")

    # ---- Fig. 4 data: per-pair disagreement (40 MDI pairs, 14 PHI epochs) and one example per row
    mdi_files = sorted(Path(root, "pairs", "mdi").glob("*.npz"))
    in40 = [f for f, r in zip(mdi_files, mdi) if r["in40"]]
    wf = []
    for f in in40:
        with np.load(f) as z:
            wf.append(float((z["disk"] & (np.abs(z["hmi"]) < 30) & (np.abs(z["mdi"]) < 30)).mean()))
    with np.load(in40[int(np.argmax(wf))]) as z:                       # the quietest pair
        h, m, disk = z["hmi"], z["mdi"], z["disk"]
    ex_mdi = (h, m, den(h), den(m), disk & (np.abs(h) < 30) & (np.abs(m) < 30), in40[int(np.argmax(wf))].stem, 1.98)
    phi_files = sorted(Path(root, "pairs", "phi").glob("*.npz"))
    k = min(range(len(phi)), key=lambda i: (0 if phi[i]["corr_after"] >= 0.95 else 1, phi[i]["fov_r"]))
    with np.load(phi_files[k]) as z:                                  # well co-aligned, nearest disk centre
        H, P = z["hmi"], z["phi_on_hmi"]
    fin = np.isfinite(H) & np.isfinite(P)
    ex_phi = (np.where(fin, H, np.nan), np.where(fin, P, np.nan), np.where(fin, den(H), np.nan),
              np.where(fin, den(P), np.nan), weak(H, P), phi_files[k].stem, 0.5)
    d = Path(args.out_dir, "figures", "data")
    d.mkdir(parents=True, exist_ok=True)
    for other, (H, O, Hd, Od, w, stem, cdelt), (bb, aa) in (("mdi", ex_mdi, (b40, a40)), ("phi", ex_phi, (b, a))):
        c, (hh_, ww_) = 256, w.shape
        best, t0, l0, best_fin, tf, lf = -1.0, 0, 0, -1.0, 0, 0
        fin = np.isfinite(H) & np.isfinite(O)
        for t in range(0, hh_ - c + 1, 32):                            # the quietest fully covered 256 px window
            for l in range(0, ww_ - c + 1, 32):
                ff = float(fin[t:t + c, l:l + c].mean())
                if ff > best_fin:
                    best_fin, tf, lf = ff, t, l
                if ff == 1.0 and float(w[t:t + c, l:l + c].mean()) > best:
                    best, t0, l0 = float(w[t:t + c, l:l + c].mean()), t, l
        if best < 0:
            t0, l0 = tf, lf
        crop = lambda im: im[t0:t0 + c, l0:l0 + c]                       # noqa: E731
        ext = [(l0 - ww_ / 2) * cdelt, (l0 + c - ww_ / 2) * cdelt, (t0 - hh_ / 2) * cdelt, (t0 + c - hh_ / 2) * cdelt]
        np.savez_compressed(d / f"fig-cross-instrument-{other}_data.npz",
                            hmi_raw=crop(np.nan_to_num(H)), other_raw=crop(np.nan_to_num(O)),
                            hmi_denoised=crop(np.nan_to_num(Hd)), other_denoised=crop(np.nan_to_num(Od)),
                            weak_mask=crop(w), crop_origin=np.array([t0, l0]), example=stem,
                            extent_arcsec=np.array(ext), before_G=np.asarray(bb), after_G=np.asarray(aa))


# --- Part: signal preservation (Section 4.4, Fig. 5) -------------------------------------------

def blob(shape, cy, cx, peak, sig):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return peak * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sig ** 2))


def positions(clean, k, rng, margin=24, min_dist=32, quiet=20.0):
    """Up to k positions at least min_dist apart, with |B| < quiet within +/-8 px."""
    ok, pos = np.abs(clean) < quiet, []
    for _ in range(4000):
        cy, cx = rng.integers(margin, clean.shape[0] - margin), rng.integers(margin, clean.shape[1] - margin)
        if not ok[cy - 8:cy + 9, cx - 8:cx + 9].all():
            continue
        if all((cy - p[0]) ** 2 + (cx - p[1]) ** 2 >= min_dist ** 2 for p in pos):
            pos.append((int(cy), int(cx)))
            if len(pos) == k:
                break
    return pos


def aperture(img, cy, cx, rad):
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    return float(img[(y - cy) ** 2 + (x - cx) ** 2 <= rad ** 2].sum())


def centroid_error(img, cy, cx, rad):
    """Offset of the |img|-weighted centroid (weights below 20% of the box peak set to 0)."""
    y0, y1, x0, x1 = cy - rad, cy + rad + 1, cx - rad, cx + rad + 1
    w = np.abs(img[y0:y1, x0:x1]).copy()
    w[w < 0.2 * w.max()] = 0.0
    if w.sum() <= 0:
        return np.inf
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float(np.hypot((w * yy).sum() / w.sum() - cy, (w * xx).sum() / w.sum() - cx))


def injection(denoise, stacks, n_hours=24, seed=0):
    """Synthetic weak elements (peaks 10/20/30/50 G x widths 1/2/4 px, one each per hour) injected
    into the stacks of 24 test hours, with the measured noise (input - stack) of the hour 7 later."""
    rng = np.random.default_rng(seed)
    configs = [(a, s) for a in (10.0, 20.0, 30.0, 50.0) for s in (1.0, 2.0, 4.0)]
    rows = []
    for i in np.unique(np.linspace(0, len(stacks) - 1, n_hours).astype(int)):
        clean = pipeline.load_stack(stacks[i])[1]
        x, y = pipeline.load_stack(stacks[(i + 7) % len(stacks)])
        noise = x - y
        pos = positions(clean, len(configs), rng)
        if len(pos) < len(configs):
            continue
        inj = clean.copy()
        for (peak, sig), (cy, cx) in zip(configs, pos):
            inj += blob(clean.shape, cy, cx, peak, sig).astype(np.float32)
        noisy_inj, noisy_ctl = inj + noise, clean + noise
        den_inj, den_ctl = denoise(noisy_inj), denoise(noisy_ctl)
        rec = den_inj - den_ctl
        empty = positions(clean, 12, rng)
        radii = sorted({max(6, int(3 * s)) for _, s in configs})
        s_den = {r: np.std([aperture(den_ctl - np.median(den_ctl), cy, cx, r) for cy, cx in empty]) for r in radii}
        s_raw = {r: np.std([aperture(noisy_ctl - np.median(noisy_ctl), cy, cx, r) for cy, cx in empty]) for r in radii}
        for (peak, sig), (cy, cx) in zip(configs, pos):
            rad = max(6, int(3 * sig))
            f_inj = float(blob((4 * rad + 1, 4 * rad + 1), 2 * rad, 2 * rad, peak, sig).sum())
            rows.append({"hour": Path(stacks[i]).stem, "peak_G": peak, "sigma_px": sig,
                         "flux_recovery": round(aperture(rec, cy, cx, rad) / f_inj, 4),
                         "centroid_err_px": round(centroid_error(rec, cy, cx, rad), 3),
                         "det_denoised": int(aperture(den_inj - np.median(den_ctl), cy, cx, rad) > 3 * s_den[rad]),
                         "det_raw": int(aperture(noisy_inj - np.median(noisy_ctl), cy, cx, rad) > 3 * s_raw[rad]),
                         "false_denoised": int(aperture(den_ctl - np.median(den_ctl), cy, cx, rad) > 3 * s_den[rad]),
                         "false_raw": int(aperture(noisy_ctl - np.median(noisy_ctl), cy, cx, rad) > 3 * s_raw[rad])})
    return rows


def psd_azimuthal(a):
    """Azimuthally averaged power spectrum of the Hann-windowed, mean-subtracted image."""
    a = a - a.mean()
    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(a * w))) ** 2
    h, wd = p.shape
    y, x = np.ogrid[:h, :wd]
    r = np.hypot(y - h // 2, x - wd // 2).astype(int)
    return (np.bincount(r.ravel(), weights=p.ravel()) / np.maximum(np.bincount(r.ravel()), 1))[:h // 2]


def part_signal(args, dev):
    from scipy.ndimage import gaussian_filter
    num = Numbers(args.out_dir, "signal")
    root, outs = args.data_root, Path(args.out_dir, "outputs")
    fig = []

    # unsigned flux against the stack by |B_stack| (outputs of part table1)
    edges = [0.0, 5, 10, 20, 30, 50, 100, np.inf]
    bins = [f"bin_{lo:g}_{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    for split in ("test_xcycle", "test"):
        methods = [m for m in ("input", "n2n", "n2n_mixsigma", "park2020", "tv", "bm3d") if (outs / m / split).is_dir()]
        fields = ["strong", "weak", "all"] + bins
        pred = {m: dict.fromkeys(fields, 0.0) for m in methods}
        res = {m: dict.fromkeys(fields, 0.0) for m in methods}
        gt_abs, npix = dict.fromkeys(fields, 0.0), dict.fromkeys(fields, 0)
        for f in pipeline.stack_files(root, split):
            y = test.center(pipeline.load_stack(f)[1], 256).astype(np.float64)
            strong = np.abs(y) > 50
            masks = [("strong", strong), ("weak", ~strong), ("all", np.ones_like(strong))] + [
                (b_, (np.abs(y) >= lo) & (np.abs(y) < hi)) for b_, lo, hi in zip(bins, edges[:-1], edges[1:])]
            for fn, mk in masks:
                npix[fn] += int(mk.sum())
                gt_abs[fn] += float(np.abs(y)[mk].sum())
            for m in methods:
                with np.load(outs / m / split / Path(f).name) as z:
                    o = z["output"].astype(np.float64)
                for fn, mk in masks:
                    pred[m][fn] += float(np.abs(o)[mk].sum())
                    res[m][fn] += float((o - y)[mk].sum())
        for m in methods:
            r = {fn: pred[m][fn] / gt_abs[fn] for fn in fields}
            num(f"{split} {m}: unsigned flux / stack: all {r['all']:.3f}, |B| < 50 G {r['weak']:.3f}, |B| > 50 G {r['strong']:.4f}; "
                + ", ".join(f"{b_[4:]} {r[b_]:.2f}" for b_ in bins) + f"; signed bias {res[m]['all'] / npix['all']:+.3f} G")
            if split == "test_xcycle" and m in ("n2n_mixsigma", "tv", "bm3d"):
                fig += [{"panel": "flux", "method": m, "x": b_, "y": r[b_], "se": ""} for b_ in bins]

    # power spectra, test set
    keys = [k for k in ("input", "n2n_mixsigma", "bm3d", "tv") if (outs / k / "test").is_dir()]
    acc, n = {}, 0
    for f in pipeline.stack_files(root, "test"):
        x, y = (test.center(a, 256) for a in pipeline.load_stack(f))
        acc["stack"] = acc.get("stack", 0) + psd_azimuthal(y)
        for k in keys:
            with np.load(outs / k / "test" / Path(f).name) as z:
                acc[k] = acc.get(k, 0) + psd_azimuthal(z["output"])
        n += 1
    lam = 256 / np.arange(1, 128)
    ps = {k: v[1:] / n for k, v in acc.items()}
    for lo, hi in ((32, 256), (16, 32), (8, 16), (4, 8), (2, 4)):
        mk = (lam >= lo) & (lam < hi)
        num(f"power ratio to the stack, {lo}-{hi} px: " + ", ".join(f"{k} {ps[k][mk].mean() / ps['stack'][mk].mean():.3f}" for k in keys))
    for k in ("input", "n2n_mixsigma", "tv", "bm3d"):
        if k in ps:
            fig += [{"panel": "psd", "method": k, "x": round(float(w), 2), "y": float(v), "se": ""}
                    for w, v in zip(lam, ps[k] / ps["stack"])]

    # synthetic injection: the same positions and noise for every method (seed 0)
    stacks = pipeline.stack_files(root, "test")
    models = load_models(args.ckpt_dir, ["n2n_mixsigma", "gap6_s1", "gap6_s2", "gap6_s3"], dev)
    methods = {"model": models["n2n_mixsigma"], **{k: v for k, v in models.items() if k.startswith("gap6")},
               "tv": lambda r: tv(r, 0.7 * pipeline.robust_sigma(r - r.mean())),
               "blur2": lambda r: gaussian_filter(r, 2.0)}
    if has_bm3d():
        methods["bm3d"] = lambda r: bm3d(r, pipeline.robust_sigma(r))
    d = Path(args.out_dir, "injection")
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in methods.items():
        rows = injection(fn, stacks)
        with open(d / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        rec = {}
        for r in rows:
            rec.setdefault((r["peak_G"], r["sigma_px"]), []).append(r["flux_recovery"])
        weak10 = [np.mean(v) for (pk, _), v in rec.items() if pk == 10]
        marg = [r for r in rows if r["peak_G"] in (20, 30) and r["sigma_px"] in (2, 4)]
        num(f"injection {name}: 10 G recovery {min(weak10):.2f}-{max(weak10):.2f}; detection 20-30 G (2, 4 px) "
            f"{np.mean([r['det_denoised'] for r in marg]):.2f} (input {np.mean([r['det_raw'] for r in marg]):.2f}); "
            f"false detection {100 * np.mean([r['false_denoised'] for r in rows]):.2f}%; "
            f"median displacement {np.median([r['centroid_err_px'] for r in rows]):.2f} px")
        if name in ("model", "tv", "bm3d", "blur2"):
            m_ = "n2n_mixsigma" if name == "model" else name
            for sig in (1.0, 2.0, 4.0):
                for pk in (10.0, 20.0, 30.0, 50.0):
                    v = np.array([r["flux_recovery"] for r in rows if r["sigma_px"] == sig and r["peak_G"] == pk])
                    fig.append({"panel": f"injection_sigma{sig:g}", "method": m_, "x": pk, "y": float(v.mean()),
                                "se": float(v.std(ddof=1) / np.sqrt(len(v)))})
    out = Path(args.out_dir, "figures", "data")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fig-signal-preservation_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "method", "x", "y", "se"])
        w.writeheader()
        w.writerows(fig)


# --- Part: noise properties (Section 3.1, Fig. 2) and noise level (Fig. 3) ---------------------

def part_noise(args, dev):
    import datetime as dt
    num = Numbers(args.out_dir, "noise")
    root = args.data_root
    out = Path(args.out_dir, "figures", "data")
    out.mkdir(parents=True, exist_ok=True)
    from skimage.registration import phase_cross_correlation

    # noise structure of the single frames: residuals of the aligned frames from their mean
    tbins, sigmas, radials, example = {}, [], [], None
    for group in pipeline.frame_groups(root, "test"):
        if len(group) < 10:
            continue
        frames = [test.center(np.asarray(pipeline.load_frame(p), np.float32), 256) for p in group]
        times = []
        for p in group:
            with np.load(p) as z:
                times.append(dt.datetime.strptime(json.loads(str(z["meta"]))["t_rec"], "%Y%m%d_%H%M%S"))
        ref = frames[len(frames) // 2]
        aligned = np.stack([pipeline.band_limited_shift(f, phase_cross_correlation(ref, f, upsample_factor=10)[0])
                            for f in frames]).astype(np.float32)
        resid = aligned - aligned.mean(axis=0)
        flat = resid.reshape(len(resid), -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(flat, axis=1) + 1e-12
        for i in range(len(flat)):
            for j in range(i, len(flat)):
                lag = int(round(abs((times[i] - times[j]).total_seconds()) / 45.0) * 45)
                tbins.setdefault(lag, []).append(float(flat[i] @ flat[j] / (norm[i] * norm[j])))
        sigmas.append(pipeline.robust_sigma(resid))
        acc = np.zeros(resid.shape[1:])
        for r in resid:
            r = r - r.mean()
            f = np.fft.fft2(r)
            acc += np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
        acc /= acc.max()
        cy, cx = acc.shape[0] // 2, acc.shape[1] // 2
        rr = np.round(np.hypot(*(np.indices(acc.shape) - np.array([cy, cx])[:, None, None]))).astype(int)
        prof = np.array([acc[rr == k].mean() for k in range(min(cy, cx))])
        radials.append(prof / prof[0])
        if example is None:
            example = resid[len(frames) // 2]
    lags = sorted(tbins)
    cmean = np.array([np.mean(tbins[t]) for t in lags])
    radial = np.mean(radials, axis=0)
    sig = float(np.mean(sigmas))
    np.savez_compressed(out / "fig-noise-structure_data.npz", example_residual=example, radial_autocorr=radial,
                        lag_minutes=np.array([t / 60.0 for t in lags]), temporal_corr=cmean,
                        white_baseline=-1.0 / 20, sigma_G=sig)
    rho = cmean[1:5]
    inflate = 1 + 2 * sum((1 - (k + 1) / 21) * r for k, r in enumerate(rho))
    num(f"single-frame noise (residual from the aligned mean, {len(sigmas)} hours): {sig:.1f} G; spatial lag-1 "
        f"autocorrelation {radial[1]:.2f}; temporal correlation at 45 s {cmean[1]:+.2f} (white-noise baseline -0.05)")
    num(f"21-frame stack: variance inflation {inflate:.2f}, effective frames {21 / inflate:.1f}, stack noise "
        f"{7.9 / np.sqrt(21 / inflate):.1f} G (independent frames {7.9 / np.sqrt(21):.1f} G)")

    # Fig. 2b: the example hour and the per-hour scores (512 px, TV with the measured noise)
    model = load_models(args.ckpt_dir, ["n2n_mixsigma", "n2n"], dev)
    den, fix = model["n2n_mixsigma"], model["n2n"]
    rows, ex = [], None
    for f in pipeline.stack_files(root, "test"):
        x, y = pipeline.load_stack(f)
        dr = float(np.percentile(y, 99.5) - np.percentile(y, 0.5)) or 1.0
        s = pipeline.robust_sigma(x - y)
        d_, t_ = den(x), tv(x, s)
        rows.append({"hour": Path(f).stem, "psnr_input": pipeline.psnr(x, y, dr), "psnr_n2n": pipeline.psnr(d_, y, dr),
                     "psnr_tv": pipeline.psnr(t_, y, dr), "ssim_n2n": pipeline.ssim(d_, y, dr), "ssim_tv": pipeline.ssim(t_, y, dr)})
        if ex is None:
            ex = dict(stacked=y, single=x, n2n_mixsigma=d_, adaptive_tv=t_, sigma_G=s, data_range=dr, hour=Path(f).stem)
    np.savez_compressed(out / "fig-baseline-example_data.npz", **ex)
    with open(out / "fig-baseline-example_data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Fig. 3: added Gaussian noise of 0, 7, 14, 21 G (seed 0), the same models without retraining
    rng = np.random.default_rng(0)
    levels = (0, 7, 14, 21)
    agg = {L: {} for L in levels}
    for f in pipeline.stack_files(root, "test"):
        base, y = (test.center(a, 256) for a in pipeline.load_stack(f))
        dr = float(np.percentile(y, 99.5) - np.percentile(y, 0.5)) or 1.0
        for L in levels:
            noisy = base if L == 0 else base + rng.normal(0, L, base.shape).astype(np.float32)
            s = pipeline.robust_sigma(noisy - y)
            o = {"in": noisy, "n2n": den(noisy), "fix": fix(noisy), "tv": tv(noisy, s)}
            a = agg[L]
            a.setdefault("sig", []).append(s)
            for k, v in o.items():
                a.setdefault(f"{k}_p", []).append(pipeline.psnr(v, y, dr))
                a.setdefault(f"{k}_s", []).append(pipeline.ssim(v, y, dr))
                if k != "in":
                    a.setdefault(f"{k}_r", []).append(pipeline.robust_sigma(v - y))
    cols = ["in_p", "n2n_p", "tv_p", "fix_p", "in_s", "n2n_s", "tv_s", "fix_s", "n2n_r", "tv_r", "fix_r"]
    with open(out / "fig-noise-level_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_add_G", "effective_sigma_G"] + cols)
        for L in levels:
            w.writerow([L, np.mean(agg[L]["sig"])] + [np.mean(agg[L][c]) for c in cols])
            num(f"added noise {L} G (effective {np.mean(agg[L]['sig']):.1f} G): PSNR mixed-sigma {np.mean(agg[L]['n2n_p']):.2f}, "
                f"TV {np.mean(agg[L]['tv_p']):.2f}, no augmentation {np.mean(agg[L]['fix_p']):.2f} dB; residual sigma "
                f"{np.mean(agg[L]['n2n_r']):.1f}, {np.mean(agg[L]['tv_r']):.1f}, {np.mean(agg[L]['fix_r']):.1f} G")
    n_par = sum(p.numel() for p in torch.load(Path(args.ckpt_dir, CKPTS["n2n_mixsigma"], "model.pt"),
                                                map_location="cpu", weights_only=False)["model"].values())
    num(f"U-Net parameters: {n_par:,}")


PARTS = {"table1": part_table1, "table2": part_table2, "signal": part_signal, "noise": part_noise}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--part", nargs="+", default=list(PARTS), choices=list(PARTS))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = torch.device(args.device)
    for part in args.part:
        PARTS[part](args, dev)


if __name__ == "__main__":
    main()
