"""CNN super-resolution models (PyTorch).

SRResidualNet : EDSR-style deep residual CNN. Works on the LR grid (cheap), upsamples with
                sub-pixel convolution (PixelShuffle) and learns only the *residual* on top of a
                bicubic upsampled copy of the input, so the network concentrates on high-frequency
                detail. No batch-norm (it hurts SR quality and is the reason ESRGAN dropped it).
SRCNN         : the classic 3-layer baseline from the paper (9-1-5) applied after bicubic upsampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    def __init__(self, ch: int, scale: int):
        layers = []
        if scale & (scale - 1) == 0:                       # 2, 4, 8 ...
            for _ in range(int(math.log2(scale))):
                layers += [nn.Conv2d(ch, 4 * ch, 3, padding=1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(ch, 9 * ch, 3, padding=1), nn.PixelShuffle(3)]
        else:
            raise ValueError("scale must be 2, 3, 4, 8, ...")
        super().__init__(*layers)


class SRResidualNet(nn.Module):
    def __init__(self, in_ch=3, feats=64, n_blocks=16, scale=2, res_scale=0.1):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_ch, feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(feats, res_scale) for _ in range(n_blocks)],
                                  nn.Conv2d(feats, feats, 3, padding=1))
        self.up = Upsampler(feats, scale)
        self.tail = nn.Conv2d(feats, in_ch, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        f = self.head(x)
        f = f + self.body(f)
        return self.tail(self.up(f)) + base


class SRCNN(nn.Module):
    def __init__(self, in_ch=3, scale=2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(nn.Conv2d(in_ch, 64, 9, padding=4), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(32, in_ch, 5, padding=2))

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return self.net(up) + up


def build_model(name: str = "resnet", scale: int = 2, **kw) -> nn.Module:
    if name == "resnet":
        return SRResidualNet(scale=scale, **kw)
    if name == "srcnn":
        return SRCNN(scale=scale)
    raise ValueError(name)


if __name__ == "__main__":
    m = build_model("resnet")
    x = torch.rand(2, 3, 96, 96)
    print(m(x).shape, f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")
