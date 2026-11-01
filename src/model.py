"""Compact U-Net with a global context bottleneck; random initialization only."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, groups=cout, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class SmallFCN(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        self.width_mult = float(width_mult)
        self.input_size = 160
        a, b, c, d, e = [int(v * width_mult) for v in (16, 32, 64, 96, 128)]
        p, q, r, s = [int(v * width_mult) for v in (64, 48, 32, 16)]
        self.e1 = Block(3, a, 2)
        self.e2 = Block(a, b, 2)
        self.e3 = Block(b, c, 2)
        self.e4 = Block(c, d, 2)
        self.e5 = Block(d, e, 2)
        self.context = nn.Conv2d(e, e, 1)
        self.d4 = Block(e + d, p)
        self.d3 = Block(p + c, q)
        self.d2 = Block(q + b, r)
        self.d1 = Block(r + a, s)
        self.head = nn.Conv2d(s, 2, 1)

    def forward(self, x):
        size = x.shape[-2:]
        # Internal downscale counts in the timed prediction pipeline.
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.e4(c)
        e = self.e5(d)
        e = e + self.context(F.adaptive_avg_pool2d(e, 1))
        for block, skip in ((self.d4, d), (self.d3, c), (self.d2, b), (self.d1, a)):
            e = block(
                torch.cat(
                    (
                        F.interpolate(
                            e,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    1,
                )
            )
        return F.interpolate(
            self.head(e), size=size, mode="bilinear", align_corners=False
        )
