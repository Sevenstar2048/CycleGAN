from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OfficeHomeDataset(Dataset):
    """Office-Home 单域数据集读取。

    目录约定:
    OfficeHome/
      Art/
        Alarm_Clock/*.jpg
      Real World/
        Alarm_Clock/*.jpg
    """

    def __init__(self, root: str, domain: str, image_size: int = 224):
        self.root = Path(root) / domain
        self.samples = []

        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

        for cls in classes:
            for img in (self.root / cls).glob("*.*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img, self.class_to_idx[cls]))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_domain_loaders(
    root: str,
    source_domain: str,
    target_domain: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    src_ds = OfficeHomeDataset(root=root, domain=source_domain, image_size=image_size)
    tgt_ds = OfficeHomeDataset(root=root, domain=target_domain, image_size=image_size)

    src_loader = DataLoader(
        src_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return src_loader, tgt_loader
