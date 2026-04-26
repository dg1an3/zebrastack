"""Demo: PyTorch recursive-filters V4 model on Lie-group eigenfunction stimuli.

Reproduces the central result from Lane (1995): concentric and radial
patterns have identical V1 (orientation x spatial-frequency) energy
distributions but distinct distributions in the V4 recursive bank.
A simple linear readout over the (initial_orientation,
recursive_orientation) plane discriminates them.

Run:
    python examples/v4_recursive_filters_demo.py [--out FIGURE_PATH]
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
    RecursiveFiltersV4,
    lie_group_readout,
    radial_dc_readout,
    standard_v1_spec,
    total_v4_energy,
)


def _v1_energy(v1_grouped: torch.Tensor) -> torch.Tensor:
    """Spatially summed V1 energy, shape (n_freq, n_orient)."""
    return v1_grouped[0].sum(dim=(-1, -2))


def _v4_energy(v4_per_freq, init_freq_index: int = 0) -> torch.Tensor:
    """Spatially summed V4 energy at one initial frequency, shape (n_orient, n_rfreq, n_rorient)."""
    return v4_per_freq[init_freq_index][0].sum(dim=(-1, -2))


def _normalize(t: torch.Tensor) -> torch.Tensor:
    s = t.sum()
    return t / s if s > 0 else t


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/v4_recursive_filters_demo.png")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    matplotlib.use("Agg")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.18, 0.09, 0.045),
        kernel_size=21,
    )
    model = RecursiveFiltersV4(spec, recursive_octaves=2, recursive_kernel_size=41)
    model.eval()

    stimuli = {
        "concentric": concentric_stimulus(args.size, frequency=0.18),
        "radial": annular_radial_stimulus(args.size, target_frequency=0.18),
        "spiral_cw": spiral_stimulus(args.size, frequency=0.12, n_spokes=12, sign=+1),
    }

    with torch.no_grad():
        outputs = {name: model(img.unsqueeze(0).unsqueeze(0)) for name, img in stimuli.items()}

    init_orients = list(spec.orientations)
    recursive_orients = list(model.recursive_orientations)

    print("V1 energy (n_freq x n_orient), spatially summed:")
    for name, out in outputs.items():
        e = _v1_energy(out["v1_power"])
        print(f"  {name}:")
        print(f"    {e.numpy()}")
        print(f"    normalized rows (per freq):")
        for f_idx in range(e.shape[0]):
            row = _normalize(e[f_idx])
            print(f"      f={spec.frequencies[f_idx]:.3f}: {row.numpy()}")

    print(
        "\nLie-group readouts (relative to total V4 energy; radial uses V2 DC envelope):"
    )
    targets = ["concentric", "radial", "spiral"]
    print(f"{'stimulus':12s} | " + " | ".join(f"{t:>12s}" for t in targets))
    print("-" * (15 + len(targets) * 15))
    readouts: dict[str, dict[str, float]] = {}
    radial_dc_raw: dict[str, float] = {}
    for name, out in outputs.items():
        norm = total_v4_energy(out["v4_power"]).clamp_min(1e-12)
        row = {}
        for target in ["concentric", "spiral"]:
            v = lie_group_readout(
                out["v4_power"],
                initial_orientations=init_orients,
                recursive_orientations=recursive_orients,
                target=target,
            )
            row[target] = float(v.mean().item() / float(norm.mean().item()))
        radial_raw = float(radial_dc_readout(out["v2_dc"], init_orients).item())
        radial_dc_raw[name] = radial_raw
        row["radial"] = radial_raw
        readouts[name] = row

    radial_max = max(radial_dc_raw.values()) or 1.0
    other_max = max(
        max(r["concentric"] for r in readouts.values()),
        max(r["spiral"] for r in readouts.values()),
    ) or 1.0
    for name in readouts:
        readouts[name]["radial"] *= other_max / radial_max
    for name, row in readouts.items():
        print(
            f"{name:12s} | " + " | ".join(f"{row[t]:12.4f}" for t in targets)
        )

    print("\nPer-cell tuning (which stimulus drives each cell hardest):")
    for cell in targets:
        responses = {name: row[cell] for name, row in readouts.items()}
        best = max(responses, key=responses.get)
        print(f"  {cell + '_cell':17s} -> {best}  ({responses})")

    n_stim = len(stimuli)
    fig, axes = plt.subplots(n_stim, 4, figsize=(13, 3.0 * n_stim))
    if n_stim == 1:
        axes = axes[None, :]

    for r, (name, img) in enumerate(stimuli.items()):
        out = outputs[name]
        axes[r, 0].imshow(img.numpy(), cmap="gray")
        axes[r, 0].set_title(f"{name}\nstimulus")
        axes[r, 0].axis("off")

        v1_e = _v1_energy(out["v1_power"]).numpy()
        im = axes[r, 1].imshow(v1_e, aspect="auto", cmap="viridis")
        axes[r, 1].set_title("V1 energy\n(freq x orient)")
        axes[r, 1].set_xticks(range(len(init_orients)))
        axes[r, 1].set_xticklabels(
            [f"{math.degrees(o):.0f}" for o in init_orients]
        )
        axes[r, 1].set_yticks(range(len(spec.frequencies)))
        axes[r, 1].set_yticklabels([f"{f:.3f}" for f in spec.frequencies])
        plt.colorbar(im, ax=axes[r, 1])

        v4_e = _v4_energy(out["v4_power"], init_freq_index=0)[:, 0]
        im = axes[r, 2].imshow(v4_e.numpy(), aspect="auto", cmap="viridis")
        axes[r, 2].set_title(
            f"V4 energy at f0={spec.frequencies[0]:.3f}\n(initial vs recursive orient)"
        )
        axes[r, 2].set_xlabel("recursive orient (deg)")
        axes[r, 2].set_ylabel("initial orient (deg)")
        axes[r, 2].set_xticks(range(len(recursive_orients)))
        axes[r, 2].set_xticklabels(
            [f"{math.degrees(o):.0f}" for o in recursive_orients]
        )
        axes[r, 2].set_yticks(range(len(init_orients)))
        axes[r, 2].set_yticklabels(
            [f"{math.degrees(o):.0f}" for o in init_orients]
        )
        plt.colorbar(im, ax=axes[r, 2])

        targets_list = list(readouts[name].keys())
        values = [readouts[name][t] for t in targets_list]
        axes[r, 3].bar(targets_list, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[r, 3].set_title("Lie-group readouts")
        axes[r, 3].set_ylabel("mean activation")

    fig.suptitle("Recursive Filters V4 model (Lane 1995) — PyTorch", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")

    concentric_score = readouts["concentric"]["concentric"] - readouts["concentric"]["radial"]
    radial_score = readouts["radial"]["radial"] - readouts["radial"]["concentric"]
    print(
        f"\nDiscrimination check: "
        f"concentric_cell prefers concentric by {concentric_score:.4f}, "
        f"radial_cell prefers radial by {radial_score:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
