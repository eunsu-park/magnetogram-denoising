"""Score checkpoints on the test data: the model rows of Tables 1 and 2 of the paper.

    python test.py --ckpt runs/n2n/model.pt --data-root <data> --table 1a
    python test.py --ckpt a.pt b.pt --table 1a 1b 1c 2a 2b 2c --out-dir results/

Tables (G = gauss):
  1a  HMI, 21-frame stack reference: the central 256 px of the input frame and its stack
      (test: 2013 December; test_xcycle: 2019-11 to 2020-01, with --split).
  1b  HMI, even-frame reference: the central odd frame scored against the stack of the even frames.
  1c  HMI, reference at least 270 s away: the central frame against the stack of the frames at
      least 6 frames from it on both sides. 1b and 1c also score the 3-, 5- and 7-frame stacks.
      Frames are cut to the central 256 px before alignment.
  2a  HMI-MDI on the MDI grid: weak-field disagreement (|B| < 30 G in both, r < 0.85 R_sun),
      strong-field flux retained (|B_MDI| > 70 G), and the HMI-vs-HMI reference of 13 pairs.
  2b  HMI-PHI on the HMI grid: weak-field disagreement, strong-field flux retained (|B_PHI| > 70 G),
      and the HMI-vs-HMI references 45 s and 270 s apart, on the pixels PHI covers.
  2c  HMI-PHI on the HRT grid: reproject then denoise, and denoise then reproject (the stored
      HRT-grid coordinates of the bilinear reprojection).

Metrics: PSNR and SSIM with data range = 0.5-99.5 percentile range of the reference, residual
sigma = robust sigma of output - reference; disagreement = robust sigma of the difference of the
two magnetograms over the weak-field mask. Means over groups or pairs are printed; --out-dir
writes one csv per table with one row per group or pair and model.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import pipeline
from network import Generator, unet


# --- Models ------------------------------------------------------------------------------------

def load_denoiser(path, dev):
    """A checkpoint as a function: Gauss array -> denoised Gauss array (NaN read as 0)."""
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    if "generator" in ckpt or cfg.get("model") == "park2020":   # Park 2020
        g = Generator().to(dev)
        g.load_state_dict(ckpt.get("generator", ckpt.get("model")))
        g.eval()
        clip = float(cfg.get("clip", cfg.get("data_range", 100.0)))

        @torch.no_grad()
        def park(b):
            b = np.nan_to_num(b)
            x = torch.from_numpy(pipeline.to_clip(b, clip).astype(np.float32))[None, None].to(dev)
            den = g(x).cpu().numpy()[0, 0] * clip
            return pipeline.restore_strong(den, b)
        return park
    model = unet(int(cfg.get("base", 32)), int(cfg.get("levels", 0))).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scale = float(cfg.get("scale", 1000.0))

    @torch.no_grad()
    def run(b):
        b = np.nan_to_num(b)
        h, w = b.shape
        x = torch.from_numpy((b / scale).astype(np.float32))[None, None].to(dev)
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="reflect")
        return (model(x)[..., :h, :w][0, 0].cpu().numpy() * scale).astype(np.float32)
    return run


def center(a, c):
    h, w = a.shape[-2:]
    t, l = (h - c) // 2, (w - c) // 2
    return a[..., t:t + c, l:l + c]


def quality(out, ref):
    dr = float(np.percentile(ref, 99.5) - np.percentile(ref, 0.5)) or 1.0
    return {"psnr": pipeline.psnr(out, ref, dr), "ssim": pipeline.ssim(out, ref, dr),
            "resid_sigma": pipeline.robust_sigma(out - ref)}


def weak(a, b, t=30.0):
    """Weak-field mask on the raw arrays: both finite and |B| < t."""
    return np.isfinite(a) & np.isfinite(b) & (np.abs(np.nan_to_num(a)) < t) & (np.abs(np.nan_to_num(b)) < t)


def disagree(a, b, w):
    return pipeline.robust_sigma((np.nan_to_num(a) - np.nan_to_num(b))[w])


def flux_ratio(after, before, sel, min_px=0):
    if sel.sum() <= min_px or not np.abs(np.nan_to_num(before))[sel].sum():
        return float("nan")
    return float(np.abs(np.nan_to_num(after))[sel].sum() / np.abs(np.nan_to_num(before))[sel].sum())


# --- Tables ------------------------------------------------------------------------------------

def table_1a(methods, root, splits, crop=256):
    rows = []
    for split in splits:
        for f in pipeline.stack_files(root, split):
            x, y = (center(a, crop) for a in pipeline.load_stack(f))
            for name, fn in methods.items():
                out = fn(x)
                rows.append({"split": split, "file": Path(f).name, "method": name, **quality(out, y),
                             "flux_bias": float(np.mean(out - y)),
                             "unsigned_flux_ratio": float(np.abs(out).sum()) / (float(np.abs(y).sum()) or 1.0)})
    return rows


def frame_reference(frames, kind):
    """(input frame, reference, the frames the k-frame stacks are built from, centre index) of one
    group, or None if the group is too short."""
    if kind == "1b":
        evens, odds = frames[0::2], frames[1::2]
        if len(evens) < 6 or len(odds) < 7:
            return None
        return odds[len(odds) // 2], pipeline.align_and_stack(evens, len(evens) // 2), odds, len(odds) // 2
    c, gap = len(frames) // 2, 6
    far = frames[:c - gap + 1] + frames[c + gap:]
    if len(frames[1::2]) < 7 or len(frames[0::2]) < 6 or len(far) < 8:
        return None
    # aligned to the central frame, which is then removed from the average
    ref = (pipeline.align_and_stack([frames[c]] + far, 0) * (len(far) + 1) - frames[c]) / len(far)
    return frames[c], ref, frames, c


def k_stack(frames, c, k):
    lo = max(0, min(c - k // 2, len(frames) - k))
    return pipeline.align_and_stack(frames[lo:lo + k], c - lo)


def table_1bc(methods, root, kind, crop=256, stacks=(3, 5, 7)):
    rows = []
    for group in pipeline.frame_groups(root, "test"):
        frames = list(center(np.stack([np.asarray(pipeline.load_frame(p), np.float32) for p in group]), crop))
        r = frame_reference(frames, kind)
        if r is None:
            continue
        x, ref, base, c = r
        outs = {name: fn(x) for name, fn in methods.items()}
        outs.update({f"stack-{k}": k_stack(base, c, k) for k in stacks})
        for name, out in outs.items():
            rows.append({"group": Path(group[0]).name, "method": name, **quality(out, ref)})
    return rows


def table_2a(methods, root):
    rows = []
    for f in sorted(glob.glob(f"{root}/pairs/mdi/*.npz")):
        with np.load(f) as z:
            h, m, disk = z["hmi"], z["mdi"], z["disk"]
            h2 = z["hmi_ref"] if "hmi_ref" in z.files else None
            meta = json.loads(str(z["meta"]))
        w = disk & (np.abs(h) < 30) & (np.abs(m) < 30)
        strong = disk & (np.abs(m) > 70)
        base = {"file": Path(f).name, "in_40_pair_subset": meta["in_40_pair_subset"],
                "mdi_interval_s": meta["mdi_interval_s"], "before": disagree(h, m, w)}
        for name, fn in methods.items():
            hd, md = fn(h), fn(m)
            row = {**base, "method": name, "after": disagree(hd, md, w),
                   "flux_retained_hmi": flux_ratio(hd, h, strong)}
            if h2 is not None:
                w2 = disk & (np.abs(h) < 30) & (np.abs(h2) < 30)
                row.update(ref_before=disagree(h, h2, w2), ref_after=disagree(hd, fn(h2), w2))
            rows.append(row)
    return rows


def table_2bc(methods, root):
    rows = []
    for f in sorted(glob.glob(f"{root}/pairs/phi/*.npz")):
        z = dict(np.load(f))
        H, P, phi, hh = z["hmi"], z["phi_on_hmi"], z["phi"], z["hmi_on_hrt"]
        w, in_phi = weak(H, P), np.isfinite(P)
        strong = np.isfinite(H) & np.isfinite(P) & (np.abs(np.nan_to_num(P)) > 70)
        w_h = weak(hh, phi)
        base = {"file": Path(f).name, "hmi_before": disagree(H, P, w), "hrt_before": disagree(hh, phi, w_h)}
        for name, fn in methods.items():
            Hd = fn(H)
            row = {**base, "method": name, "hmi_after": disagree(Hd, fn(P), w),
                   "flux_retained_hmi": flux_ratio(Hd, H, strong, min_px=100)}
            for key in ("hmi_45s", "hmi_270s"):
                O = z[key]
                wo = weak(H, O) & np.isfinite(O) & in_phi
                row[f"{key}_before"], row[f"{key}_after"] = disagree(H, O, wo), disagree(Hd, fn(O), wo)
            phi_d = fn(phi)
            row["hrt_reproject_then_denoise"] = disagree(fn(hh), phi_d, w_h)
            hd_on_hrt = pipeline.bilinear(fn(z["hmi_foot"]), z["coords_hrt"])
            row["hrt_denoise_then_reproject"] = disagree(hd_on_hrt, phi_d, w_h & np.isfinite(hd_on_hrt))
            rows.append(row)
    return rows


# --- Summaries ---------------------------------------------------------------------------------

def mean(rows, key):
    """Mean over the rows that have the value (NaN where a pair has no strong-field pixel)."""
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def reduction(before, after):
    return 100.0 * (1.0 - after / before)


def summarize(table, rows, methods):
    names = list(dict.fromkeys(r["method"] for r in rows))
    if table in ("1a", "1b", "1c"):
        for split in dict.fromkeys(r.get("split", "test") for r in rows):
            sub = [r for r in rows if r.get("split", "test") == split]
            print(f"\nTable {table} ({split}): mean over {len(sub) // len(names)} groups")
            print(f"{'method':26}{'PSNR':>8}{'SSIM':>8}{'resid G':>9}")
            for n in names:
                r = [q for q in sub if q["method"] == n]
                print(f"{n:26}{mean(r, 'psnr'):8.2f}{mean(r, 'ssim'):8.3f}{mean(r, 'resid_sigma'):9.2f}")
    elif table == "2a":
        for subset, sel in (("all pairs", lambda q: True), ("40-pair subset", lambda q: q["in_40_pair_subset"])):
            print(f"\nTable 2a, HMI-MDI, {subset}")
            for n in names:
                r = [q for q in rows if q["method"] == n and sel(q)]
                b, a = mean(r, "before"), mean(r, "after")
                print(f"  {n:24} {len(r):5d} pairs  {b:6.2f} -> {a:5.2f} G ({reduction(b, a):.1f}%)"
                      f"  flux retained {mean(r, 'flux_retained_hmi'):.3f}")
        for n in names:
            r = [q for q in rows if q["method"] == n and "ref_before" in q]
            b, a = mean(r, "ref_before"), mean(r, "ref_after")
            print(f"  HMI vs HMI ({n}, {len(r)} pairs): {b:.2f} -> {a:.2f} G ({reduction(b, a):.1f}%)")
    else:
        print(f"\nTable 2b/2c, HMI-PHI, {len(rows) // len(names)} epochs")
        for n in names:
            r = [q for q in rows if q["method"] == n]
            b, a = mean(r, "hmi_before"), mean(r, "hmi_after")
            print(f"  {n:24} HMI grid {b:.2f} -> {a:.2f} G ({reduction(b, a):.1f}%), flux retained "
                  f"{mean(r, 'flux_retained_hmi'):.3f}")
            for key, label in (("hmi_45s", "HMI vs HMI 45 s"), ("hmi_270s", "HMI vs HMI 270 s")):
                b2, a2 = mean(r, f"{key}_before"), mean(r, f"{key}_after")
                print(f"  {'':24} {label} {b2:.2f} -> {a2:.2f} G ({reduction(b2, a2):.1f}%)")
            hb = mean(r, "hrt_before")
            for key, label in (("hrt_reproject_then_denoise", "HRT grid, reproject then denoise"),
                               ("hrt_denoise_then_reproject", "HRT grid, denoise then reproject")):
                print(f"  {'':24} {label} {hb:.2f} -> {mean(r, key):.2f} G ({reduction(hb, mean(r, key)):.1f}%)")


def run(tables, methods, root, splits=("test",), out_dir=None):
    """Compute the requested tables; returns {table: rows} and writes <out_dir>/table<t>_rows.csv.
    Tables 2b and 2c come from the same computation and are returned together as '2bc'."""
    models = {k: v for k, v in methods.items() if k != "input"}
    tables = list(dict.fromkeys("2bc" if t in ("2b", "2c") else t for t in tables))
    result = {}
    for t in tables:
        if t == "1a":
            rows = table_1a(methods, root, splits)
        elif t in ("1b", "1c"):
            rows = table_1bc(methods, root, t)
        elif t == "2a":
            rows = table_2a(models, root)
        else:
            rows = table_2bc(models, root)
        summarize(t, rows, methods)
        result[t] = rows
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(Path(out_dir, f"table{t}_rows.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    return result


def main(default_split="test"):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--table", nargs="+", default=["1a"], choices=["1a", "1b", "1c", "2a", "2b", "2c"])
    p.add_argument("--split", nargs="+", default=[default_split], help="Table 1a only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", help="write table<t>_rows.csv here")
    args = p.parse_args()
    dev = torch.device(args.device)
    methods = {"input": lambda b: np.nan_to_num(b)}
    methods.update({Path(c).parent.name or Path(c).stem: load_denoiser(c, dev) for c in args.ckpt})
    run(args.table, methods, args.data_root, args.split, args.out_dir)


if __name__ == "__main__":
    main()
