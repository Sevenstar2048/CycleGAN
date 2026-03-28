import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from data.officehome import build_domain_loaders
from models.cyclegan import CycleGANCore
from utils.config import load_config
from utils.fourier import fda_source_to_target
from utils.seed import set_seed


def train_one_epoch_spatial(model, src_loader, tgt_loader, optimizer_g, device):
    model.train()
    l1 = nn.L1Loss()
    total = 0.0

    for (x_s, _), (x_t, _) in tqdm(zip(src_loader, tgt_loader), total=min(len(src_loader), len(tgt_loader))):
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        out = model.translate(x_s, x_t)

        loss_cycle = l1(out["rec_s"], x_s) + l1(out["rec_t"], x_t)
        loss_id = l1(model.g_t2s(x_s), x_s) + l1(model.g_s2t(x_t), x_t)
        loss = 10.0 * loss_cycle + 0.5 * loss_id

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        total += loss.item()

    return total / min(len(src_loader), len(tgt_loader))


def train_one_epoch_spectral(model, src_loader, tgt_loader, optimizer_g, device, beta):
    model.train()
    l1 = nn.L1Loss()
    total = 0.0

    for (x_s, _), (x_t, _) in tqdm(zip(src_loader, tgt_loader), total=min(len(src_loader), len(tgt_loader))):
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        # 先做频域低频替换，再送入 CycleGAN 学习映射
        x_s_fda = fda_source_to_target(x_s, x_t, beta=beta)
        out = model.translate(x_s_fda, x_t)

        loss_cycle = l1(out["rec_s"], x_s_fda) + l1(out["rec_t"], x_t)
        loss_id = l1(model.g_t2s(x_s_fda), x_s_fda) + l1(model.g_s2t(x_t), x_t)
        loss = 10.0 * loss_cycle + 0.5 * loss_id

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        total += loss.item()

    return total / min(len(src_loader), len(tgt_loader))


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

    if args.mode == "spatial":
        epochs = cfg["task1"]["cyclegan"]["epochs"]
    else:
        epochs = cfg["task1"]["spectral_cyclegan"]["epochs"]

    for epoch in range(1, epochs + 1):
        if args.mode == "spatial":
            loss = train_one_epoch_spatial(model, src_loader, tgt_loader, optimizer_g, device)
        else:
            beta = cfg["task1"]["spectral_cyclegan"]["beta"]
            loss = train_one_epoch_spectral(model, src_loader, tgt_loader, optimizer_g, device, beta)

        print(f"[Task1][{args.mode}] Epoch {epoch}/{epochs} - loss: {loss:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "task1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"{args.mode}_cyclegan_art2real.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
