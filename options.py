"""Command-line options. The defaults reproduce the models of the paper.

Every model shares one option set; options that a model does not use are ignored. Options left
unset take the per-model default below, so `python train.py --model n2n` trains the paper's model.
"""
import argparse

# Per-model defaults: the settings of the paper's checkpoints.
DEFAULTS = {
    # Noise2Noise, mixed-sigma augmentation: the model of the paper (first of five seeds).
    "n2n": dict(batch_size=8, lr=1e-3, noise_aug_max=21.0, seed=1, num_workers=4, amp=True,
                val_every=500, log_every=50),
    # Noise2Void with the same U-Net.
    "n2v": dict(batch_size=8, lr=1e-3, noise_aug_max=0.0, seed=0, num_workers=0, amp=True,
                val_every=500, log_every=50),
    # Supervised U-Net (frame -> 21-frame stack), MSE.
    "supervised": dict(batch_size=8, lr=1e-3, noise_aug_max=0.0, seed=0, num_workers=4,
                       amp=True, val_every=0, log_every=500),
    # Park et al. (2020) pix2pix, supervised (frame -> 21-frame stack).
    "park2020": dict(batch_size=4, lr=2e-4, noise_aug_max=0.0, seed=0, num_workers=4, amp=False,
                     val_every=2000, log_every=100),
}


def parse(argv=None) -> argparse.Namespace:
    """Parse the command line and fill the per-model defaults."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(DEFAULTS), default="n2n")
    p.add_argument("--data-root", required=True, help="folder holding manifests/, frames/, stacks/")
    p.add_argument("--out-dir", required=True, help="where the checkpoint and the logs go")
    p.add_argument("--device", default="cuda")
    # Optimization.
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--amp", type=int, choices=[0, 1], help="mixed precision on CUDA")
    p.add_argument("--deterministic", type=int, choices=[0, 1], default=1,
                   help="deterministic CUDA algorithms (needed for bitwise reproducibility)")
    # Data.
    p.add_argument("--crop", type=int, default=256, help="training crop (px)")
    p.add_argument("--scale", type=float, default=1000.0, help="G per network unit (U-Net models)")
    p.add_argument("--noise-aug-max", type=float,
                   help="mixed-sigma augmentation: extra Gaussian noise, sigma ~ U(0, max) G")
    p.add_argument("--pair-gap", type=int, default=1,
                   help="N2N: frames between the two frames of a pair (1 = 45 s)")
    p.add_argument("--align", type=int, choices=[0, 1], default=0,
                   help="N2N: co-align the second frame to the first (used with --pair-gap > 1)")
    p.add_argument("--mask-frac", type=float, default=0.005, help="N2V: fraction of blind spots")
    p.add_argument("--window", type=int, default=5, help="N2V: neighbour window (px)")
    # Networks.
    p.add_argument("--base", type=int, default=32, help="U-Net base width")
    p.add_argument("--levels", type=int, default=0,
                   help="U-Net depth; 0 = the two-level U-Net of the paper")
    p.add_argument("--lambda-l1", type=float, default=100.0, help="Park 2020: weight of L1")
    p.add_argument("--clip", type=float, default=100.0,
                   help="Park 2020: input/target clipped to +/- clip G")
    # Logging.
    p.add_argument("--val-every", type=int)
    p.add_argument("--log-every", type=int)
    args = p.parse_args(argv)
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.amp, args.align, args.deterministic = bool(args.amp), bool(args.align), bool(args.deterministic)
    return args
