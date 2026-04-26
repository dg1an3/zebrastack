"""Train the LieGroupCells 1x1 conv on randomized stimuli.

Compares analytic-baseline weights (the orientation-matching pattern that
``LieGroupCells`` initializes to) against gradient-trained weights.
Optionally also trains the V2 1x1 mixing layer for cross-channel learning.

Run:
    python examples/v4_recursive_filters_train.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.lie_group_stimuli import (
    random_concentric, random_radial, random_spiral,
)
from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4WithReadout, standard_v1_spec,
)


CLASS_NAMES = ["radial", "concentric", "spiral"]


def make_dataset(n_per_class: int, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    images, labels = [], []
    generators = [random_radial, random_concentric, random_spiral]
    for label_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            img = gen(size, rng)
            images.append(img)
            labels.append(label_idx)
    images = torch.stack(images).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(labels), generator=rng)
    return images[perm], labels[perm]


def confusion(model, head, images, labels, batch_size: int = 16) -> tuple[float, list[list[int]]]:
    model.eval()
    if head is not None:
        head.eval()
    n_classes = len(CLASS_NAMES)
    matrix = [[0] * n_classes for _ in range(n_classes)]
    correct = 0
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i+batch_size]
            y = labels[i:i+batch_size]
            out = model(x)
            cells = out["lie_cells"]
            if head is not None:
                logits = head(cells)
            else:
                logits = cells.mean(dim=(-1, -2))
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


class ReadoutHead(nn.Module):
    """Spatial pool + calibrated per-class scale + learnable bias.

    Per-class scale is initialized to 1 / (mean cell activation observed
    on a calibration set) so the three logits start on a comparable
    magnitude and softmax doesn't saturate. Both scale and bias are
    learnable. No running statistics, robust at eval time, and unlike
    LayerNorm the network can grow logit spread to lower the loss.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    @torch.no_grad()
    def calibrate(self, backbone, calibration_x: torch.Tensor, batch_size: int = 16):
        """Set scale = 1 / |mean cell activation| from a forward pass."""
        backbone.eval()
        accum = torch.zeros(self.scale.shape[0])
        n = 0
        for i in range(0, calibration_x.shape[0], batch_size):
            x = calibration_x[i:i+batch_size]
            cells = backbone(x)["lie_cells"]
            pooled = cells.mean(dim=(-1, -2))
            accum = accum + pooled.abs().sum(dim=0)
            n += pooled.shape[0]
        magnitudes = (accum / max(n, 1)).clamp_min(1e-12)
        self.scale.data.copy_(1.0 / magnitudes)
        self.bias.data.zero_()

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        pooled = cells.mean(dim=(-1, -2))
        return self.scale * pooled + self.bias


def train(backbone_model, head, train_x, train_y, epochs, lr, batch_size, weight_decay):
    params = [p for p in backbone_model.parameters() if p.requires_grad]
    params += list(head.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in params)}")
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n = train_x.shape[0]
    for epoch in range(epochs):
        backbone_model.train()
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x, y = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            out = backbone_model(x)
            logits = head(out["lie_cells"])
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        train_acc = total_correct / n
        print(f"  epoch {epoch+1:2d}: loss={total_loss/n:.4f} train_acc={train_acc:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/v4_recursive_filters_train.png")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-v2", action="store_true",
                        help="Also train the V2 1x1 mixing layer.")
    args = parser.parse_args()
    matplotlib.use("Agg")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.22, 0.14, 0.09, 0.06),
        kernel_size=25,
    )

    print("Generating dataset...")
    train_x, train_y = make_dataset(args.n_train, args.size, seed=0)
    test_x, test_y = make_dataset(args.n_test, args.size, seed=1)
    print(f"  train: {len(train_y)} images, test: {len(test_y)} images")

    print("\n--- Analytic baseline (no training) ---")
    baseline = RecursiveFiltersV4WithReadout(
        v1_spec=spec,
        targets=tuple(CLASS_NAMES),
        recursive_octaves=2,
        recursive_kernel_size=41,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=False,
        radial_subtracts_orthogonal=True,
        trainable_lie_cells=False,
    )
    base_acc, base_cm = confusion(baseline, None, test_x, test_y)
    print(f"  test accuracy (raw argmax): {base_acc:.3f}")
    print_matrix(base_cm)

    print("\n--- Training the LieGroupCells 1x1 conv ---")
    trained = RecursiveFiltersV4WithReadout(
        v1_spec=spec,
        targets=tuple(CLASS_NAMES),
        recursive_octaves=2,
        recursive_kernel_size=41,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=True,
        v4_dc_sigma_long=16.0,
        v4_dc_surround_ratio=1.5,
        radial_subtracts_orthogonal=False,
        use_v2_mixing=args.train_v2,
        trainable_v2_mix=args.train_v2,
        trainable_lie_cells=True,
    )
    head = ReadoutHead(len(CLASS_NAMES))
    head.calibrate(trained, train_x)
    print(f"  calibrated per-class scale: {head.scale.detach().tolist()}")
    train(
        trained, head, train_x, train_y,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, weight_decay=args.weight_decay,
    )
    trained_acc, trained_cm = confusion(trained, head, test_x, test_y)
    print(f"\n  test accuracy: {trained_acc:.3f}")
    print_matrix(trained_cm)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, cm, acc) in zip(
        axes,
        [("baseline (analytic 1x1)", base_cm, base_acc),
         ("trained 1x1", trained_cm, trained_acc)],
    ):
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
        ax.set_title(f"{title}\nacc={acc:.2f}")
        plt.colorbar(im, ax=ax)

    weights = trained.lie_cells.conv.weight.detach()[:, :, 0, 0]
    ax = axes[2]
    im = ax.imshow(weights.numpy(), cmap="RdBu_r",
                   vmin=-weights.abs().max(), vmax=weights.abs().max(),
                   aspect="auto")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("V4 channel index")
    ax.set_title("Trained LieGroupCells weights")
    plt.colorbar(im, ax=ax)

    fig.suptitle("V4 LieGroupCells: analytic baseline vs trained 1x1 conv")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")

    print(f"\nSummary: baseline={base_acc:.2f} -> trained={trained_acc:.2f} "
          f"(delta {trained_acc - base_acc:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
