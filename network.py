"""Networks: the U-Net of the paper (and its depth/width variants) and the Park et al. (2020)
pix2pix generator and discriminator.

The modules are created in the same order as in the code that trained the published
checkpoints, so a seeded initialization gives the same weights and the checkpoints load as-is.
"""
import torch
import torch.nn as nn


class _Block(nn.Module):
    """Two 3x3 conv + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Two-level U-Net: 1 -> base -> 2*base -> bottleneck 4*base, with skip connections."""

    def __init__(self, ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _Block(ch, base)
        self.enc2 = _Block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bott = _Block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _Block(base * 2, base)
        self.out = nn.Conv2d(base, ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class UNetVar(nn.Module):
    """U-Net with `levels` down/up stages (the depth variants); widths double per level."""

    def __init__(self, ch: int = 1, base: int = 32, levels: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        in_ch = ch
        for i in range(levels):
            self.enc.append(_Block(in_ch, base * 2 ** i))
            in_ch = base * 2 ** i
        self.bott = _Block(in_ch, base * 2 ** levels)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        below = base * 2 ** levels
        for i in reversed(range(levels)):
            skip = base * 2 ** i
            self.up.append(nn.ConvTranspose2d(below, skip, 2, stride=2))
            self.dec.append(_Block(skip * 2, skip))
            below = skip
        self.out = nn.Conv2d(base, ch, 1)

    def forward(self, x):
        skips, h = [], x
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bott(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.out(h)


def unet(base: int = 32, levels: int = 0) -> nn.Module:
    """The U-Net of the paper (levels=0) or a depth variant (levels>0)."""
    return UNetVar(base=base, levels=levels) if levels > 0 else UNet(base=base)


# --- Park et al. (2020): pix2pix ---------------------------------------------------------------

def _init_weights(m: nn.Module) -> None:
    """pix2pix initialization: conv ~ N(0, 0.02), BN weight ~ N(1, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class _Down(nn.Module):
    """Conv 4x4 stride 2 -> [BN] -> LeakyReLU(0.2)."""

    def __init__(self, c_in: int, c_out: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    """ConvTranspose 4x4 stride 2 -> BN -> [Dropout 0.5] -> ReLU."""

    def __init__(self, c_in: int, c_out: int, dropout: bool = False):
        super().__init__()
        layers = [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """pix2pix U-Net generator: 8 down / 8 up with skip connections (256 px -> 1 px)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, bf: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, bf, 4, 2, 1, bias=False),
                                  nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = _Down(bf, bf * 2)
        self.enc3 = _Down(bf * 2, bf * 4)
        self.enc4 = _Down(bf * 4, bf * 8)
        self.enc5 = _Down(bf * 8, bf * 8)
        self.enc6 = _Down(bf * 8, bf * 8)
        self.enc7 = _Down(bf * 8, bf * 8)
        self.bottleneck = nn.Sequential(nn.Conv2d(bf * 8, bf * 8, 4, 2, 1, bias=False),
                                        nn.ReLU(inplace=True))
        self.dec1 = _Up(bf * 8, bf * 8, dropout=True)
        self.dec2 = _Up(bf * 16, bf * 8, dropout=True)
        self.dec3 = _Up(bf * 16, bf * 8, dropout=True)
        self.dec4 = _Up(bf * 16, bf * 8)
        self.dec5 = _Up(bf * 16, bf * 4)
        self.dec6 = _Up(bf * 8, bf * 2)
        self.dec7 = _Up(bf * 4, bf)
        self.final = nn.Sequential(nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1), nn.Tanh())
        self.apply(_init_weights)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        d = torch.cat([self.dec1(self.bottleneck(e7)), e7], dim=1)
        d = torch.cat([self.dec2(d), e6], dim=1)
        d = torch.cat([self.dec3(d), e5], dim=1)
        d = torch.cat([self.dec4(d), e4], dim=1)
        d = torch.cat([self.dec5(d), e3], dim=1)
        d = torch.cat([self.dec6(d), e2], dim=1)
        d = torch.cat([self.dec7(d), e1], dim=1)
        return self.final(d)


class Discriminator(nn.Module):
    """70x70 PatchGAN on the (input, output) pair concatenated on channels."""

    def __init__(self, in_channels: int = 2, bf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, bf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bf, bf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(bf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bf * 2, bf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(bf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bf * 4, bf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(bf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bf * 8, 1, 4, 1, 1), nn.Sigmoid())
        self.apply(_init_weights)

    def forward(self, x):
        return self.model(x)
