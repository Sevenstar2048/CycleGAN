from typing import Tuple

import torch


def _low_freq_mask(shape: Tuple[int, int], beta: float, device: torch.device) -> torch.Tensor:
    """构造二维低频区域掩码，beta 控制窗口比例。"""
    h, w = shape
    b = int(min(h, w) * beta)
    mask = torch.zeros((h, w), device=device)
    cy, cx = h // 2, w // 2
    y1, y2 = max(cy - b, 0), min(cy + b, h)
    x1, x2 = max(cx - b, 0), min(cx + b, w)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def fda_source_to_target(src: torch.Tensor, tgt: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """FDA: 用目标图像低频幅度替换源图像低频幅度。

    参数:
    - src: [N, C, H, W]
    - tgt: [N, C, H, W]
    """
    assert src.shape == tgt.shape, "src and tgt must have same shape"

    src_fft = torch.fft.fft2(src, dim=(-2, -1))
    tgt_fft = torch.fft.fft2(tgt, dim=(-2, -1))

    src_amp, src_phase = torch.abs(src_fft), torch.angle(src_fft)
    tgt_amp = torch.abs(tgt_fft)

    src_amp_shift = torch.fft.fftshift(src_amp, dim=(-2, -1))
    tgt_amp_shift = torch.fft.fftshift(tgt_amp, dim=(-2, -1))

    mask = _low_freq_mask((src.shape[-2], src.shape[-1]), beta, src.device)
    mask = mask.unsqueeze(0).unsqueeze(0)

    mixed_amp_shift = src_amp_shift * (1 - mask) + tgt_amp_shift * mask
    mixed_amp = torch.fft.ifftshift(mixed_amp_shift, dim=(-2, -1))

    real = mixed_amp * torch.cos(src_phase)
    imag = mixed_amp * torch.sin(src_phase)
    mixed_fft = torch.complex(real, imag)

    out = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
    return out
