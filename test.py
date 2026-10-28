"""Score checkpoints on the HMI test stacks (Table 1a and the 2019-2020 test).

    python test.py --ckpt runs/n2n/model.pt --data-root <data> --split test
    python test.py --ckpt a.pt b.pt --split test test_xcycle --out-csv scores.csv

Each hourly group gives one input frame and its 21-frame stack. The central 256 px of both are
used; the model denoises the input crop and is scored against the stack in G:
PSNR and SSIM (data range = 0.5-99.5 percentile range of the stack), residual sigma
(robust sigma of output - stack), signed-flux bias (mean of output - stack) and unsigned-flux
ratio (sum |output| / sum |stack|).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import pipeline
from network import Generator, unet


def load_denoiser(path, dev):
    """A checkpoint as a function: Gauss array -> denoised Gauss array."""
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    if "generator" in ckpt or cfg.get("model") == "park2020":   # Park 2020
        g = Generator().to(dev)
        g.load_state_dict(ckpt.get("generator", ckpt.get("model")))
        g.eval()
        clip = float(cfg.get("clip", cfg.get("data_range", 100.0)))

        @torch.no_grad()
        def park(b):
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
        h, w = b.shape
        x = torch.from_numpy((b / scale).astype(np.float32))[None, None].to(dev)
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="reflect")
        return (model(x)[..., :h, :w][0, 0].cpu().numpy() * scale).astype(np.float32)
    return run


def center(a, c):
    h, w = a.shape
    t, l = (h - c) // 2, (w - c) // 2
    return a[t:t + c, l:l + c]


def scores(out, stack):
    dr = float(np.percentile(stack, 99.5) - np.percentile(stack, 0.5)) or 1.0
    return {"psnr": pipeline.psnr(out, stack, dr), "ssim": pipeline.ssim(out, stack, dr),
            "resid_sigma": pipeline.robust_sigma(out - stack),
            "flux_bias": float(np.mean(out - stack)),
            "unsigned_flux_ratio": float(np.abs(out).sum()) / (float(np.abs(stack).sum()) or 1.0)}


def main(default_split="test"):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", nargs="+", default=[default_split])
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-csv", help="per-group scores")
    p.add_argument("--save-dir", help="save each output as <save-dir>/<ckpt>/<split>/<file>.npz")
    args = p.parse_args()
    dev = torch.device(args.device)
    methods = {"input": lambda b: b}
    methods.update({Path(c).parent.name or Path(c).stem: load_denoiser(c, dev) for c in args.ckpt})
    rows = []
    for split in args.split:
        for f in pipeline.stack_files(args.data_root, split):
            x, y = pipeline.load_stack(f)
            x, y = center(x, args.crop), center(y, args.crop)
            for name, fn in methods.items():
                out = fn(x)
                rows.append({"split": split, "file": Path(f).name, "method": name, **scores(out, y)})
                if args.save_dir and name != "input":
                    d = Path(args.save_dir, name, split)
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(d / Path(f).name, output=out.astype(np.float32))
        print(f"\n{split}: mean over {len(rows) // len(methods) if rows else 0} groups")
        print(f"{'method':24}{'PSNR':>8}{'SSIM':>8}{'resid G':>9}{'bias G':>9}{'|flux|':>8}")
        for name in methods:
            r = [q for q in rows if q["split"] == split and q["method"] == name]
            m = {k: np.mean([q[k] for q in r]) for k in
                 ("psnr", "ssim", "resid_sigma", "flux_bias", "unsigned_flux_ratio")}
            print(f"{name:24}{m['psnr']:8.2f}{m['ssim']:8.3f}{m['resid_sigma']:9.2f}"
                  f"{m['flux_bias']:+9.3f}{m['unsigned_flux_ratio']:8.3f}")
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
