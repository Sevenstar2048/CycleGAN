import argparse
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.officehome import build_domain_loaders
from models.classifier import ResNet18Classifier
from models.cyclegan import CycleGANCore
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
    if strategy in {"cyclegan", "spectral_cyclegan", "cycada"}:
        cyclegan = CycleGANCore().to(device)
        cyclegan.eval()

    epochs = cfg["task2"][strategy if strategy != "cyclegan" else "cyclegan_aug"]["epochs"]

    for epoch in range(1, epochs + 1):
        clf.train()
        total_loss = 0.0

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
                # 简化版 CyCADA：输入级别先翻译，再加入一致性约束可在此扩展
                assert cyclegan is not None
                with torch.no_grad():
                    x_in = cyclegan.g_s2t(x_s)
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
        print(f"[Task2][{strategy}] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - target acc: {acc:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "task2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"{strategy}_classifier_art2real.pt"
    torch.save(clf.state_dict(), out)
    print(f"Saved checkpoint: {out}")


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
