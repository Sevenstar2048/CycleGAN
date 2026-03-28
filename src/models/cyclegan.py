from typing import Dict

import torch
import torch.nn as nn


class SmallGenerator(nn.Module):
    """轻量生成器占位实现，用于框架连通与后续替换。"""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallDiscriminator(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CycleGANCore(nn.Module):
    """CycleGAN 核心网络定义。

    这里只提供最小可运行骨架，损失组合与训练细节在 train 文件实现。
    """

    def __init__(self):
        super().__init__()
        self.g_s2t = SmallGenerator()
        self.g_t2s = SmallGenerator()
        self.d_s = SmallDiscriminator()
        self.d_t = SmallDiscriminator()

    def translate(self, x_s: torch.Tensor, x_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        fake_t = self.g_s2t(x_s)
        fake_s = self.g_t2s(x_t)
        rec_s = self.g_t2s(fake_t)
        rec_t = self.g_s2t(fake_s)
        return {
            "fake_t": fake_t,
            "fake_s": fake_s,
            "rec_s": rec_s,
            "rec_t": rec_t,
        }
