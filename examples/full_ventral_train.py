"""Train the V1 -> V2 -> V4 -> PIT -> CIT -> AIT ventral-stream model.

Compares end-to-end-trained FullVentralStream against the V4+LieGroupCells
readout model (~96% test accuracy at 969 trainable params). The IT stack
is roughly 10x more parameters (~9.6k) but adds genuine depth so it can
build compound features on top of V4.

Run:
    python examples/full_ventral_train.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.lie_group_stimuli import (
    random_concentric, random_radial, random_spiral,
)
from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4, standard_v1_spec,
)
from zebrastack.common.it_stages import FullVentralStream


CLASS_NAMES = ["radial", "concentric", "spiral"]


def make_dataset(n_per_class: int, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    images, labels = [], []
    generators = [random_radial, random_concentric, random_spiral]
    for label_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            images.append(gen(size, rng))
            labels.append(label_idx)
    images = torch.stack(images).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(labels), generator=rng)
    return images[perm], labels[perm]


def confusion(model, images, labels, batch_size=16):
    model.eval()
    matrix = [[0]*3 for _ in range(3)]
    correct = 0
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i+batch_size]
            y = labels[i:i+batch_size]
            logits = model(x)["logits"]
            preds = logits.argmax(dim=-1)
            for p, t in zip(preds.tolist(), y.tolist()):
                matrix[t][p] += 1
                if p == t:
                    correct += 1
    return correct / len(labels), matrix


def print_matrix(matrix):
    header = "true / pred"
    print(f"{header:14s}" + "".join(f"{c:>12s}" for c in CLASS_NAMES))
    for i, row in enumerate(matrix):
        print(f"{CLASS_NAMES[i]:14s}" + "".join(f"{v:12d}" for v in row))


def train(model, train_x, train_y, epochs, lr, batch_size, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    n = train_x.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x, y = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            logits = model(x)["logits"]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        train_acc = total_correct / n
        print(f"  epoch {epoch+1:2d}: loss={total_loss/n:.4f} train_acc={train_acc:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/full_ventral_train.png")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    matplotlib.use("Agg")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.22, 0.14, 0.09, 0.06),
        kernel_size=25,
    )
    v4 = RecursiveFiltersV4(
        v1_spec=spec,
        recursive_octaves=2,
        recursive_kernel_size=41,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=True,
        v4_dc_sigma_long=16.0,
        v4_dc_surround_ratio=1.5,
        use_v2_gabor=True,
        v2_gabor_frequencies=(0.07, 0.035),
        v2_gabor_kernel_size=33,
    )

    print("Generating dataset...")
    train_x, train_y = make_dataset(args.n_train, args.size, seed=0)
    test_x, test_y = make_dataset(args.n_test, args.size, seed=1)
    print(f"  train: {len(train_y)} images, test: {len(test_y)} images")

    print("\n--- FullVentralStream (V1>V2>V4>PIT>CIT>AIT) ---")
    model = FullVentralStream(
        v4_backbone=v4,
        n_classes=len(CLASS_NAMES),
        pit_n_reduce=16,
        cit_n_reduce=16,
        ait_n_reduce=16,
        pit_frequencies=(0.05, 0.025),
        cit_frequencies=(0.05, 0.025),
        ait_frequencies=(0.05, 0.025),
        kernel_size=21,
        downsample=2,
    )

    train(model, train_x, train_y, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, weight_decay=args.weight_decay)
    acc, cm = confusion(model, test_x, test_y)
    print(f"\n  test accuracy: {acc:.3f}")
    print_matrix(cm)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    cm_t = torch.tensor(cm, dtype=torch.float32)
    cm_norm = cm_t / cm_t.sum(dim=-1, keepdim=True).clamp_min(1)
    im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"FullVentralStream (V1>V2>V4>PIT>CIT>AIT)\ntest acc = {acc:.2f}")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
