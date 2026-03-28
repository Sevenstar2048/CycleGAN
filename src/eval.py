import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report

from data.officehome import build_domain_loaders
from models.classifier import ResNet18Classifier
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on Office-Home target domain")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    _, tgt_loader = build_domain_loaders(
        root=cfg["paths"]["office_home_root"],
        source_domain=cfg["paths"]["source_domain"],
        target_domain=cfg["paths"]["target_domain"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = ResNet18Classifier(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tgt_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"Target accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
