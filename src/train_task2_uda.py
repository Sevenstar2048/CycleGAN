import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.officehome import build_domain_loaders
from models.classifier import ResNet18Classifier
from models.cyclegan import CycleGANCore
from models.domain_discriminator import FeatureDomainDiscriminator
from utils.config import load_config
from utils.fourier import fda_source_to_target
from utils.seed import set_seed


def _align_batch_size(x_s, x_t):
    """对齐两个张量的批次大小，取较小者截断。"""
    batch_s, batch_t = x_s.shape[0], x_t.shape[0]
    batch_size = min(batch_s, batch_t)
    return x_s[:batch_size], x_t[:batch_size]


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())
    return accuracy_score(y_true, y_pred)


def train_one_step_cycada(
    clf,
    cyclegan,
    feat_disc,
    x_s,
    y_s,
    x_t,
    optimizer_task,
    optimizer_cyclegan_g,
    optimizer_cyclegan_d,
    optimizer_feat_d,
    cfg,
):
    """完整版 CyCADA: 输入级对抗 + 循环一致 + 语义一致 + 特征级域对抗。"""
    assert cyclegan is not None
    assert feat_disc is not None

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    lam_cycle = cfg["task2"]["cycada"]["lambda_cycle"]
    lam_identity = cfg["task2"]["cycada"]["lambda_identity"]
    lam_sem = cfg["task2"]["cycada"]["lambda_semantic"]
    lam_feat_adv = cfg["task2"]["cycada"]["lambda_feature_adv"]
    lam_img_adv = cfg["task2"]["cycada"]["lambda_image_adv"]

    # 1) 更新生成器 + 分类器 (任务网络)
    fake_t = cyclegan.g_s2t(x_s)
    fake_s = cyclegan.g_t2s(x_t)
    rec_s = cyclegan.g_t2s(fake_t)
    rec_t = cyclegan.g_s2t(fake_s)

    valid_t = torch.ones_like(cyclegan.d_t(x_t))
    valid_s = torch.ones_like(cyclegan.d_s(x_s))

    loss_img_adv = mse(cyclegan.d_t(fake_t), valid_t) + mse(cyclegan.d_s(fake_s), valid_s)
    loss_cycle = l1(rec_s, x_s) + l1(rec_t, x_t)
    loss_identity = l1(cyclegan.g_t2s(x_s), x_s) + l1(cyclegan.g_s2t(x_t), x_t)

    logits_src2t = clf(fake_t)
    loss_task = ce(logits_src2t, y_s)

    # 语义一致: 约束翻译前后类别分布一致
    with torch.no_grad():
        src_prob = F.softmax(clf(x_s), dim=1)
    loss_sem = F.kl_div(F.log_softmax(logits_src2t, dim=1), src_prob, reduction="batchmean")

    # 特征级域对抗: 让源翻译特征更接近目标域
    feat_src2t = clf.extract_features(fake_t)
    target_like = torch.ones((feat_src2t.shape[0], 1), device=feat_src2t.device)
    loss_feat_adv_g = bce(feat_disc(feat_src2t), target_like)

    loss_g_total = (
        loss_task
        + lam_sem * loss_sem
        + lam_feat_adv * loss_feat_adv_g
        + lam_img_adv * loss_img_adv
        + lam_cycle * loss_cycle
        + lam_identity * loss_identity
    )

    optimizer_task.zero_grad()
    optimizer_cyclegan_g.zero_grad()
    loss_g_total.backward()
    optimizer_task.step()
    optimizer_cyclegan_g.step()

    # 2) 更新图像判别器 (CycleGAN 的 D_s / D_t)
    fake_t_det = fake_t.detach()
    fake_s_det = fake_s.detach()

    valid_t = torch.ones_like(cyclegan.d_t(x_t))
    fake_t_label = torch.zeros_like(valid_t)
    valid_s = torch.ones_like(cyclegan.d_s(x_s))
    fake_s_label = torch.zeros_like(valid_s)

    loss_d_t = 0.5 * (mse(cyclegan.d_t(x_t), valid_t) + mse(cyclegan.d_t(fake_t_det), fake_t_label))
    loss_d_s = 0.5 * (mse(cyclegan.d_s(x_s), valid_s) + mse(cyclegan.d_s(fake_s_det), fake_s_label))
    loss_img_d = loss_d_t + loss_d_s

    optimizer_cyclegan_d.zero_grad()
    loss_img_d.backward()
    optimizer_cyclegan_d.step()

    # 3) 更新特征域判别器
    with torch.no_grad():
        feat_src2t_det = clf.extract_features(fake_t_det)
        feat_t_det = clf.extract_features(x_t)

    src_like = torch.zeros((feat_src2t_det.shape[0], 1), device=feat_src2t_det.device)
    tgt_like = torch.ones((feat_t_det.shape[0], 1), device=feat_t_det.device)
    loss_feat_d = 0.5 * (
        bce(feat_disc(feat_src2t_det), src_like) + bce(feat_disc(feat_t_det), tgt_like)
    )

    optimizer_feat_d.zero_grad()
    loss_feat_d.backward()
    optimizer_feat_d.step()

    return {
        "total": loss_g_total.item(),
        "task": loss_task.item(),
        "img_g": loss_img_adv.item(),
        "img_d": loss_img_d.item(),
        "sem": loss_sem.item(),
        "feat_d": loss_feat_d.item(),
    }


def train_classifier(strategy, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    src_loader, tgt_loader = build_domain_loaders(
        root=cfg["paths"]["office_home_root"],
        source_domain=cfg["paths"]["source_domain"],
        target_domain=cfg["paths"]["target_domain"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    clf = ResNet18Classifier(num_classes=cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    cyclegan = None
    feat_disc = None
    optimizer_cyclegan_g = None
    optimizer_cyclegan_d = None
    optimizer_feat_d = None

    if strategy in {"cyclegan", "spectral_cyclegan", "cycada"}:
        cyclegan = CycleGANCore().to(device)
        cyclegan.train()

    if strategy == "cycada":
        assert cyclegan is not None
        feat_disc = FeatureDomainDiscriminator(in_dim=clf.feature_dim).to(device)
        optimizer_cyclegan_g = optim.Adam(
            list(cyclegan.g_s2t.parameters()) + list(cyclegan.g_t2s.parameters()),
            lr=cfg["task2"]["cycada"]["lr_g"],
            betas=(0.5, 0.999),
        )
        optimizer_cyclegan_d = optim.Adam(
            list(cyclegan.d_s.parameters()) + list(cyclegan.d_t.parameters()),
            lr=cfg["task2"]["cycada"]["lr_d"],
            betas=(0.5, 0.999),
        )
        optimizer_feat_d = optim.Adam(
            feat_disc.parameters(),
            lr=cfg["task2"]["cycada"]["lr_feat_d"],
            betas=(0.5, 0.999),
        )

    epochs = cfg["task2"][strategy if strategy != "cyclegan" else "cyclegan_aug"]["epochs"]

    for epoch in range(1, epochs + 1):
        clf.train()
        if strategy == "cycada":
            assert cyclegan is not None and feat_disc is not None
            cyclegan.train()
            feat_disc.train()
        total_loss = 0.0
        total_task = 0.0
        total_img_g = 0.0
        total_img_d = 0.0
        total_sem = 0.0
        total_feat_d = 0.0

        for (x_s, y_s), (x_t, _) in tqdm(zip(cycle(src_loader), tgt_loader), total=len(tgt_loader)):
            x_s, x_t = _align_batch_size(x_s, x_t)
            y_s = y_s[:x_s.shape[0]]
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            if strategy == "source_only":
                x_in = x_s
            elif strategy == "cyclegan":
                assert cyclegan is not None
                with torch.no_grad():
                    x_in = cyclegan.g_s2t(x_s)
            elif strategy == "spectral_cyclegan":
                assert cyclegan is not None
                with torch.no_grad():
                    x_s_fda = fda_source_to_target(x_s, x_t, beta=cfg["task1"]["spectral_cyclegan"]["beta"])
                    x_in = cyclegan.g_s2t(x_s_fda)
            elif strategy == "fda":
                x_in = fda_source_to_target(x_s, x_t, beta=cfg["task2"]["fda"]["beta"])
            elif strategy == "cycada":
                assert cyclegan is not None
                assert feat_disc is not None
                assert optimizer_cyclegan_g is not None
                assert optimizer_cyclegan_d is not None
                assert optimizer_feat_d is not None
                losses = train_one_step_cycada(
                    clf=clf,
                    cyclegan=cyclegan,
                    feat_disc=feat_disc,
                    x_s=x_s,
                    y_s=y_s,
                    x_t=x_t,
                    optimizer_task=optimizer,
                    optimizer_cyclegan_g=optimizer_cyclegan_g,
                    optimizer_cyclegan_d=optimizer_cyclegan_d,
                    optimizer_feat_d=optimizer_feat_d,
                    cfg=cfg,
                )
                total_loss += losses["total"]
                total_task += losses["task"]
                total_img_g += losses["img_g"]
                total_img_d += losses["img_d"]
                total_sem += losses["sem"]
                total_feat_d += losses["feat_d"]
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            logits = clf(x_in)
            loss = criterion(logits, y_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(tgt_loader)
        acc = evaluate(clf, tgt_loader, device)
        if strategy == "cycada":
            print(
                f"[Task2][{strategy}] Epoch {epoch}/{epochs} - "
                f"loss_total: {avg_loss:.4f} - task: {total_task / len(tgt_loader):.4f} - "
                f"img_g: {total_img_g / len(tgt_loader):.4f} - img_d: {total_img_d / len(tgt_loader):.4f} - "
                f"sem: {total_sem / len(tgt_loader):.4f} - feat_d: {total_feat_d / len(tgt_loader):.4f} - "
                f"target acc: {acc:.4f}"
            )
        else:
            print(f"[Task2][{strategy}] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - target acc: {acc:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "task2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"{strategy}_classifier_art2real.pt"
    torch.save(clf.state_dict(), out)
    print(f"Saved checkpoint: {out}")

    if strategy == "cycada":
        assert cyclegan is not None and feat_disc is not None
        torch.save(cyclegan.state_dict(), ckpt_dir / "cycada_cyclegan_art2real.pt")
        torch.save(feat_disc.state_dict(), ckpt_dir / "cycada_feature_disc_art2real.pt")


def main():
    parser = argparse.ArgumentParser(description="Task II: UDA methods on Office-Home Art->Real")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["source_only", "cyclegan", "spectral_cyclegan", "cycada", "fda"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train_classifier(args.strategy, cfg)


if __name__ == "__main__":
    main()
