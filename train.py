"""Train a model of the paper.

    python train.py --model n2n --data-root <data> --out-dir runs/n2n            # the paper's model
    python train.py --model n2n --seed 2 ...                                     # another seed
    python train.py --model n2n --noise-aug-max 0 --seed 0 ...                   # no augmentation
    python train.py --model n2n --pair-gap 6 --align 1 --seed 0 ...              # 270 s pairs
    python train.py --model n2v | supervised | park2020 ...

With the same software, GPU type and seed, the checkpoint is bitwise identical to the published
one (see README). Writes `model.pt` (+ `model_best.pt` for Park 2020) and `metrics.jsonl`.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import options
import pipeline
import validation
from network import Discriminator, Generator, unet


def set_deterministic(on):
    """Deterministic CUDA kernels; must run before CUDA is initialized."""
    if on:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log(out_dir, record):
    print(json.dumps(record), flush=True)
    with open(Path(out_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def save(path, state, args):
    """Weights + training options; paths and device are left out so that the file, not only
    the weights, is identical wherever it is trained."""
    config = {k: v for k, v in vars(args).items() if k not in ("data_root", "out_dir", "device")}
    torch.save({"model": state, "config": config}, path)


def datasets(args):
    """(train, val) datasets of the chosen model; val is None when the model logs none."""
    n = args.steps * args.batch_size
    if args.model in ("n2n", "n2v"):
        tr, va = (pipeline.frame_groups(args.data_root, s) for s in ("train", "val"))
        if args.model == "n2n":
            kw = dict(crop=args.crop, scale=args.scale, pair_gap=args.pair_gap, align=args.align,
                      noise_aug_max=args.noise_aug_max)
            make = pipeline.FramePairs
        else:
            kw = dict(crop=args.crop, scale=args.scale, mask_frac=args.mask_frac, window=args.window)
            make = pipeline.FrameMasked
    else:
        tr, va = (pipeline.stack_files(args.data_root, s) for s in ("train", "val"))
        kw = (dict(crop=args.crop, norm="clip", clip=args.clip) if args.model == "park2020" else
              dict(crop=args.crop, norm="scale", scale=args.scale, noise_aug_max=args.noise_aug_max))
        make = pipeline.StackPairs
    train = make(tr, n, args.seed, **kw)
    val = make(va, args.batch_size * 20, args.seed + 1, **kw) if args.val_every else None
    return train, val


def train_unet(args, dev):
    """N2N, N2V and the supervised U-Net: one network, MSE (N2V: on the blind spots only)."""
    train_ds, val_ds = datasets(args)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=dev.type == "cuda", worker_init_fn=pipeline.seed_worker)
    train_dl = DataLoader(train_ds, **kw)
    val_dl = DataLoader(val_ds, **kw) if val_ds else None
    model = unet(args.base, args.levels).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    amp = args.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    model.train()
    for step, batch in enumerate(train_dl, 1):
        batch = [t.to(dev) for t in batch]
        opt.zero_grad()
        with torch.autocast(device_type=dev.type, enabled=amp):
            loss = validation.unet_loss(model, batch, args.model)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            log(args.out_dir, {"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        if val_dl and step % args.val_every == 0:
            log(args.out_dir, {"step": step, "val": validation.val_loss(model, val_dl, args.model, dev)})
        if step >= args.steps:
            break
    save(Path(args.out_dir, "model.pt"), model.state_dict(), args)


def train_park2020(args, dev):
    """Park et al. (2020): cGAN (BCE) + lambda * L1; the discriminator steps first.
    `model_best.pt` keeps the generator with the lowest validation L1."""
    train_ds, val_ds = datasets(args)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                          drop_last=True, pin_memory=dev.type == "cuda",
                          worker_init_fn=pipeline.seed_worker)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    g, d = Generator().to(dev), Discriminator().to(dev)
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce, l1 = torch.nn.BCELoss(), torch.nn.L1Loss()
    best = float("inf")
    g.train(); d.train()
    for step, (x, y) in enumerate(train_dl, 1):
        x, y = x.to(dev), y.to(dev)
        fake = g(x)
        opt_d.zero_grad(set_to_none=True)
        p_real, p_fake = d(torch.cat([x, y], 1)), d(torch.cat([x, fake.detach()], 1))
        loss_d = 0.5 * (bce(p_real, torch.ones_like(p_real)) + bce(p_fake, torch.zeros_like(p_fake)))
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad(set_to_none=True)
        p = d(torch.cat([x, fake], 1))
        loss_g = bce(p, torch.ones_like(p)) + args.lambda_l1 * l1(fake, y)
        loss_g.backward()
        opt_g.step()
        if step % args.log_every == 0:
            log(args.out_dir, {"step": step, "loss_d": loss_d.item(), "loss_g": loss_g.item()})
        if step % args.val_every == 0:
            v = validation.park_val_l1(g, val_dl, dev)
            log(args.out_dir, {"step": step, "val": v})
            if v < best:
                best = v
                save(Path(args.out_dir, "model_best.pt"), g.state_dict(), args)
        if step >= args.steps:
            break
    save(Path(args.out_dir, "model.pt"), g.state_dict(), args)


def main():
    args = options.parse()
    set_deterministic(args.deterministic)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    (train_park2020 if args.model == "park2020" else train_unet)(args, dev)


if __name__ == "__main__":
    main()
