"""Train FullVentralStream on CIFAR-10 (grayscale, upsampled).

CIFAR-10 native resolution is 32x32 RGB. To run the V1>V2>V4>PIT>CIT>AIT
model -- whose V4 DC kernel is 97x97 -- the images are converted to
grayscale (luminance) and bilinearly upsampled to 96x96. The model has
~12k trainable parameters and operates on a single-channel input, so it
won't compete with ResNet on CIFAR-10; the goal is to verify the
architecture trains end-to-end on a real image classification task with
genuinely diverse natural-image content.

Run:
    python examples/cifar10_train.py [--data /tmp/cifar10_imgs/CIFAR-10-images-master]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4, standard_v1_spec,
)
from zebrastack.common.it_stages import FullVentralStream


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CIFARFolderDataset(Dataset):
    """Reads CIFAR-10 jpg-folder layout: <root>/<split>/<class>/<file>.jpg"""

    def __init__(self, root: str, split: str, size: int, n_per_class: int | None = None):
        self.size = size
        self.samples: list[tuple[str, int]] = []
        split_dir = Path(root) / split
        for label_idx, cls in enumerate(CIFAR_CLASSES):
            cls_dir = split_dir / cls
            files = sorted(cls_dir.glob("*.jpg"))
            if n_per_class is not None:
                files = files[:n_per_class]
            self.samples.extend((str(f), label_idx) for f in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # Grayscale via ITU-R 601-2 luma transform (built into PIL)
        img = img.convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = torch.from_numpy(__import__("numpy").asarray(img, dtype="float32"))
        arr = arr / 255.0
        arr = (arr - 0.5) * 2.0
        return arr.unsqueeze(0), label


def confusion(model, loader, device, n_classes):
    model.eval()
    matrix = [[0] * n_classes for _ in range(n_classes)]
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)["logits"]
            preds = logits.argmax(dim=-1)
            for p, t in zip(preds.tolist(), y.tolist()):
                matrix[t][p] += 1
                if p == t:
                    correct += 1
                total += 1
    return correct / max(total, 1), matrix


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)["logits"]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += x.shape[0]
    return total_loss / total, total_correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/cifar10_imgs/CIFAR-10-images-master")
    parser.add_argument("--out", default="examples/cifar10_train.png")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--n-train-per-class", type=int, default=500,
                        help="subset size per class (default 500 = 5000 total)")
    parser.add_argument("--n-test-per-class", type=int, default=100,
                        help="test subset per class (default 100 = 1000 total)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    matplotlib.use("Agg")

    device = args.device
    print(f"Device: {device}")

    train_ds = CIFARFolderDataset(args.data, "train", args.size, args.n_train_per_class)
    test_ds = CIFARFolderDataset(args.data, "test", args.size, args.n_test_per_class)
    print(f"Train: {len(train_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device != "cpu")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device != "cpu")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.22, 0.14, 0.09, 0.06),
        kernel_size=21,
    )
    v4 = RecursiveFiltersV4(
        v1_spec=spec,
        recursive_octaves=2,
        recursive_kernel_size=33,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=True,
        v4_dc_sigma_long=12.0,
        v4_dc_surround_ratio=1.5,
        v4_dc_kernel_size=49,
        use_v2_gabor=True,
        v2_gabor_frequencies=(0.07, 0.035),
        v2_gabor_kernel_size=25,
    )
    model = FullVentralStream(
        v4_backbone=v4,
        n_classes=len(CIFAR_CLASSES),
        pit_n_reduce=24,
        cit_n_reduce=24,
        ait_n_reduce=24,
        pit_frequencies=(0.05, 0.025),
        cit_frequencies=(0.05, 0.025),
        ait_frequencies=(0.05, 0.025),
        kernel_size=17,
        downsample=2,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        history.append((loss, train_acc))
        print(f"  epoch {epoch+1:2d}: loss={loss:.4f} train_acc={train_acc:.3f}")

    print("\n  recalibrating BN running stats from trained model...")
    cal_x = []
    cal_max_batches = 16
    for i, (x, _) in enumerate(train_loader):
        if i >= cal_max_batches: break
        cal_x.append(x.to(device))
    cal_x = torch.cat(cal_x, dim=0)
    model.recalibrate_bn(cal_x, batch_size=args.batch_size)

    test_acc, cm = confusion(model, test_loader, device, len(CIFAR_CLASSES))
    print(f"\n  test accuracy: {test_acc:.3f}")
    print("Per-class recall:")
    for i, name in enumerate(CIFAR_CLASSES):
        n = sum(cm[i])
        r = cm[i][i] / n if n else 0.0
        print(f"  {name:12s}: {r:.2f}  ({cm[i][i]}/{n})")

    fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))
    cm_t = torch.tensor(cm, dtype=torch.float32)
    cm_norm = cm_t / cm_t.sum(dim=-1, keepdim=True).clamp_min(1)
    im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CIFAR_CLASSES)):
        for j in range(len(CIFAR_CLASSES)):
            v = cm[i][j]
            if v > 0:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CIFAR_CLASSES)))
    ax.set_yticks(range(len(CIFAR_CLASSES)))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"FullVentralStream on CIFAR-10 (grayscale, {args.size}px)\n"
                 f"test acc = {test_acc:.3f}, params = {n_trainable}")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
