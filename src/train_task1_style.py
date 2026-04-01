import argparse
from itertools import cycle
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from data.officehome import build_domain_loaders
from models.cyclegan import CycleGANCore
from utils.config import load_config
from utils.fourier import fda_source_to_target
from utils.seed import set_seed


def _align_batch_size(x_s, x_t):
    """对齐两个张量的批次大小，取较小者截断。"""
    batch_s, batch_t = x_s.shape[0], x_t.shape[0]
    batch_size = min(batch_s, batch_t)
    return x_s[:batch_size], x_t[:batch_size]


def _train_one_epoch(
    model,
    src_loader,
    tgt_loader,
    optimizer_g,
    optimizer_d,
    device,
    lambda_cycle,
    lambda_identity,
    spectral_beta=None,
):
    model.train()
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    total_g = 0.0
    total_d = 0.0

    for (x_s, _), (x_t, _) in tqdm(zip(cycle(src_loader), tgt_loader), total=len(tgt_loader)):
        x_s, x_t = _align_batch_size(x_s, x_t)
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        if spectral_beta is not None:
            # 频域版先做低频替换，再执行对抗+循环一致训练
            x_s_in = fda_source_to_target(x_s, x_t, beta=spectral_beta)
        else:
            x_s_in = x_s

        valid_t = torch.ones_like(model.d_t(x_t))
        fake_t_label = torch.zeros_like(valid_t)
        valid_s = torch.ones_like(model.d_s(x_s))
        fake_s_label = torch.zeros_like(valid_s)

        out = model.translate(x_s_in, x_t)
        fake_t = out["fake_t"]
        fake_s = out["fake_s"]

        # 生成器损失: 对抗 + cycle + identity
        loss_g_adv = mse(model.d_t(fake_t), valid_t) + mse(model.d_s(fake_s), valid_s)
        loss_cycle = l1(out["rec_s"], x_s_in) + l1(out["rec_t"], x_t)
        loss_id = l1(model.g_t2s(x_s_in), x_s_in) + l1(model.g_s2t(x_t), x_t)
        loss_g = loss_g_adv + lambda_cycle * loss_cycle + lambda_identity * loss_id

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # 判别器损失: 区分真/假样本
        loss_d_t_real = mse(model.d_t(x_t), valid_t)
        loss_d_t_fake = mse(model.d_t(fake_t.detach()), fake_t_label)
        loss_d_t = 0.5 * (loss_d_t_real + loss_d_t_fake)

        loss_d_s_real = mse(model.d_s(x_s), valid_s)
        loss_d_s_fake = mse(model.d_s(fake_s.detach()), fake_s_label)
        loss_d_s = 0.5 * (loss_d_s_real + loss_d_s_fake)

        loss_d = loss_d_t + loss_d_s

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        total_g += loss_g.item()
        total_d += loss_d.item()

    return total_g / len(tgt_loader), total_d / len(tgt_loader)


def train_one_epoch_spatial(
    model,
    src_loader,
    tgt_loader,
    optimizer_g,
    optimizer_d,
    device,
    lambda_cycle,
    lambda_identity,
):
    return _train_one_epoch(
        model=model,
        src_loader=src_loader,
        tgt_loader=tgt_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        spectral_beta=None,
    )


def train_one_epoch_spectral(
    model,
    src_loader,
    tgt_loader,
    optimizer_g,
    optimizer_d,
    device,
    lambda_cycle,
    lambda_identity,
    beta,
):
    return _train_one_epoch(
        model=model,
        src_loader=src_loader,
        tgt_loader=tgt_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        spectral_beta=beta,
    )


def main():
    parser = argparse.ArgumentParser(description="Task I: Spatial vs Spectral CycleGAN (Office-Home Art->Real)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["spatial", "spectral"], required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    src_loader, tgt_loader = build_domain_loaders(
        root=cfg["paths"]["office_home_root"],
        source_domain=cfg["paths"]["source_domain"],
        target_domain=cfg["paths"]["target_domain"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = CycleGANCore().to(device)
    optimizer_g = optim.Adam(
        list(model.g_s2t.parameters()) + list(model.g_t2s.parameters()),
        lr=cfg["train"]["lr"],
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        list(model.d_s.parameters()) + list(model.d_t.parameters()),
        lr=cfg["train"]["lr"],
        betas=(0.5, 0.999),
    )

    if args.mode == "spatial":
        lambda_cycle = cfg["task1"]["cyclegan"]["lambda_cycle"]
        lambda_identity = cfg["task1"]["cyclegan"]["lambda_identity"]
    else:
        lambda_cycle = cfg["task1"]["cyclegan"]["lambda_cycle"]
        lambda_identity = cfg["task1"]["cyclegan"]["lambda_identity"]

    if args.mode == "spatial":
        epochs = cfg["task1"]["cyclegan"]["epochs"]
    else:
        epochs = cfg["task1"]["spectral_cyclegan"]["epochs"]

    for epoch in range(1, epochs + 1):
        if args.mode == "spatial":
            loss_g, loss_d = train_one_epoch_spatial(
                model,
                src_loader,
                tgt_loader,
                optimizer_g,
                optimizer_d,
                device,
                lambda_cycle,
                lambda_identity,
            )
        else:
            beta = cfg["task1"]["spectral_cyclegan"]["beta"]
            loss_g, loss_d = train_one_epoch_spectral(
                model,
                src_loader,
                tgt_loader,
                optimizer_g,
                optimizer_d,
                device,
                lambda_cycle,
                lambda_identity,
                beta,
            )

        print(f"[Task1][{args.mode}] Epoch {epoch}/{epochs} - loss_g: {loss_g:.4f} - loss_d: {loss_d:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "task1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"{args.mode}_cyclegan_art2real.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
