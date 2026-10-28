"""Validation: the losses logged during training, and the scores of a checkpoint on the
validation stacks (November 2013).

    python validation.py --ckpt runs/n2n/model.pt --data-root <data>

The validation scores use the same procedure as test.py; the validation month selected the
settings that the paper then fixed for the test data.
"""
import torch
import torch.nn.functional as F


def unet_loss(model, batch, kind):
    """Training loss of the U-Net models: MSE, on the blind spots only for N2V."""
    if kind == "n2v":
        x, y, m = batch
        return F.mse_loss(model(x)[m], y[m])
    x, y = batch
    return F.mse_loss(model(x), y)


@torch.no_grad()
def val_loss(model, loader, kind, dev, max_batches=20):
    """Mean training loss over the first `max_batches` validation batches (full precision)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        total += unet_loss(model, [t.to(dev) for t in batch], kind).item()
        n += 1
        if i + 1 >= max_batches:
            break
    model.train()
    return total / max(1, n)


@torch.no_grad()
def park_val_l1(generator, loader, dev):
    """Mean L1 (normalized units) of the Park 2020 generator against the stacks."""
    generator.eval()
    total, n = 0.0, 0
    for x, y in loader:
        total += F.l1_loss(generator(x.to(dev)), y.to(dev)).item()
        n += 1
    generator.train()
    return total / max(1, n)


if __name__ == "__main__":
    import test
    test.main(default_split="val")
