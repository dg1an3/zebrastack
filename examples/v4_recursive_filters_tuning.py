"""Spatial tuning test for the Lie-group V4 cells.

Following the Gallant et al. (1993, 1996) macaque V4 finding that cells
have broad spatial tuning -- shifting the stimulus across the visual
field causes a gradual response drop rather than a sharp cutoff -- this
script measures each Lie-group cell's response to its preferred stimulus
shifted across a 2D grid of positions, and produces tuning maps.

Two metrics:
  * "max-pool readout"  : the maximum cell response anywhere in the
                          spatial output map. Reflects translation
                          tolerance: the cell tracks the stimulus
                          regardless of where it is.
  * "centre readout"    : the cell response at the image centre pixel,
                          while the stimulus is shifted away from the
                          centre. Reflects the cell's local receptive-
                          field falloff.

Run:
    python examples/v4_recursive_filters_tuning.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.lie_group_stimuli import (
    annular_radial_stimulus,
    concentric_stimulus,
    spiral_stimulus,
)
from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4WithReadout, standard_v1_spec,
)


CLASSES = ["radial", "concentric", "spiral"]


def shift_image(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate an image tensor (H, W) by (dx, dy) pixels using zero pad."""
    h, w = img.shape[-2:]
    out = torch.zeros_like(img)
    src_y0 = max(0, -dy);  src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx);  src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy);   dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx);   dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = img[..., src_y0:src_y1, src_x0:src_x1]
    return out


def make_stimulus(name: str, size: int) -> torch.Tensor:
    if name == "radial":
        return annular_radial_stimulus(size, target_frequency=0.18)
    if name == "concentric":
        return concentric_stimulus(size, frequency=0.18)
    if name == "spiral":
        return spiral_stimulus(size, frequency=0.12, n_spokes=10, sign=+1)
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/v4_recursive_filters_tuning.png")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--max-shift", type=int, default=48)
    parser.add_argument("--shift-step", type=int, default=4)
    args = parser.parse_args()
    matplotlib.use("Agg")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.22, 0.14, 0.09, 0.06),
        kernel_size=25,
    )
    model = RecursiveFiltersV4WithReadout(
        v1_spec=spec,
        targets=tuple(CLASSES),
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
        radial_subtracts_orthogonal=True,
        trainable_lie_cells=False,
    )
    model.eval()

    shifts = list(range(-args.max_shift, args.max_shift + 1, args.shift_step))
    n = len(shifts)
    centre = args.size // 2

    cell_idx = {name: i for i, name in enumerate(CLASSES)}

    print(f"Sweeping {n}x{n} shifts in [{-args.max_shift}, +{args.max_shift}] px (step {args.shift_step})...")

    max_pool = {c: torch.zeros(n, n) for c in CLASSES}
    centre_resp = {c: torch.zeros(n, n) for c in CLASSES}
    base = {c: make_stimulus(c, args.size) for c in CLASSES}

    with torch.no_grad():
        for c in CLASSES:
            ci = cell_idx[c]
            for iy, dy in enumerate(shifts):
                for ix, dx in enumerate(shifts):
                    img = shift_image(base[c], dx, dy)
                    out = model(img.unsqueeze(0).unsqueeze(0))
                    cell = out["lie_cells"][0, ci]
                    max_pool[c][iy, ix] = cell.max()
                    centre_resp[c][iy, ix] = cell[centre, centre]

    print("\nResults (response normalized to centred-stimulus value):")
    for c in CLASSES:
        mp = max_pool[c]
        cr = centre_resp[c]
        zero_idx = shifts.index(0)
        max_at_centre = float(mp[zero_idx, zero_idx])
        cr_at_centre = float(cr[zero_idx, zero_idx])
        print(f"  {c}_cell:")
        print(f"    centred max-response  = {max_at_centre:.5f}")
        print(f"    centred-pixel response = {cr_at_centre:.5f}")
        if abs(max_at_centre) > 1e-12:
            half = float((mp / max_at_centre).abs().mean())
            print(f"    mean(max_pool / centred max) over grid = {half:.3f}")

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(4.6 * len(CLASSES), 8.5))
    extent = [shifts[0], shifts[-1], shifts[-1], shifts[0]]
    for col, c in enumerate(CLASSES):
        mp = max_pool[c].numpy()
        cr = centre_resp[c].numpy()

        im0 = axes[0, col].imshow(mp, cmap="viridis", extent=extent, aspect="auto")
        axes[0, col].set_title(f"{c}_cell : max(spatial response)\nstimulus shifted by (dx, dy)")
        axes[0, col].set_xlabel("dx (pixels)")
        axes[0, col].set_ylabel("dy (pixels)")
        plt.colorbar(im0, ax=axes[0, col])

        im1 = axes[1, col].imshow(cr, cmap="magma", extent=extent, aspect="auto")
        axes[1, col].set_title(f"{c}_cell : response at centre pixel\nas stimulus is shifted away")
        axes[1, col].set_xlabel("dx (pixels)")
        axes[1, col].set_ylabel("dy (pixels)")
        plt.colorbar(im1, ax=axes[1, col])

    fig.suptitle(
        "Spatial tuning of Lie-group V4 cells (Gallant-style translation test)",
        y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved tuning figure to {args.out}")

    fig2, axes2 = plt.subplots(1, len(CLASSES), figsize=(4.5 * len(CLASSES), 4))
    for col, c in enumerate(CLASSES):
        mp = max_pool[c]
        cr = centre_resp[c]
        zero_idx = shifts.index(0)
        max_centre = float(mp[zero_idx, zero_idx]) or 1.0
        cr_centre = float(cr[zero_idx, zero_idx]) or 1.0
        axes2[col].plot(shifts, (mp[zero_idx, :] / max_centre).numpy(),
                        "o-", label="max-pool readout (translation tolerance)")
        axes2[col].plot(shifts, (cr[zero_idx, :] / cr_centre).numpy(),
                        "s-", label="centre-pixel readout (local RF)")
        axes2[col].axhline(0.5, color="grey", linestyle=":", alpha=0.5, label="50%")
        axes2[col].axvline(0, color="grey", linestyle=":", alpha=0.5)
        axes2[col].set_title(f"{c}_cell horizontal tuning")
        axes2[col].set_xlabel("horizontal shift dx (pixels)")
        axes2[col].set_ylabel("response (normalized)")
        axes2[col].legend(fontsize=7)
        axes2[col].grid(alpha=0.3)

    fig2.suptitle("1D tuning curves through dy=0", y=1.01)
    fig2.tight_layout()
    out2 = args.out.replace(".png", "_1d.png")
    fig2.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"Saved 1D tuning curves to {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
