"""Robustness sweep for the recursive-filters V4 model.

Generates a battery of Lie-group stimuli with:
  * a range of lower-frequency carriers (matching Lane 1995, p. 5,
    "stimuli created by amplitude modulating Gabor functions with lower
     spatial frequency Gabor functions")
  * the older simple polar gratings at multiple frequencies
  * angular phase rotations
  * spiral pitch variations
and checks for each stimulus whether the corresponding Lie-group cell
(concentric / radial / spiral) is the most active.

Reports per-class accuracy (fraction of stimuli where the matching cell
wins) and a confusion matrix.

Run:
    python examples/v4_recursive_filters_sweep.py [--out FIGURE_PATH]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from zebrastack.common.lie_group_stimuli import (
    annular_radial_stimulus,
    concentric_stimulus,
    modulated_gabor_stimulus,
    spiral_stimulus,
)
from zebrastack.common.v4_recursive_filters import (
    RecursiveFiltersV4WithReadout,
    radial_dc_readout,
    standard_v1_spec,
    total_v4_energy,
)


@dataclass
class Trial:
    label: str
    family: str
    image: torch.Tensor
    notes: str = ""


def build_trials(size: int) -> list[Trial]:
    trials: list[Trial] = []

    # Polar gratings at multiple stripe frequencies.
    for f in (0.06, 0.09, 0.12, 0.18):
        trials.append(Trial(
            label=f"concentric f={f:.2f}",
            family="concentric",
            image=concentric_stimulus(size, frequency=f),
            notes="polar grating",
        ))
        trials.append(Trial(
            label=f"radial f={f:.2f}",
            family="radial",
            image=annular_radial_stimulus(size, target_frequency=f),
            notes="annular spokes",
        ))
        for sign in (+1, -1):
            trials.append(Trial(
                label=f"spiral f={f:.2f} sign={sign:+d}",
                family="spiral",
                image=spiral_stimulus(size, frequency=f, n_spokes=10, sign=sign),
                notes="polar grating",
            ))

    # Modulated-Gabor stimuli: high-frequency carrier modulated by a low-frequency
    # polar envelope. Matches Lane (1995, p. 5).
    for carrier_f in (0.18, 0.22):
        for env_f in (0.03, 0.05, 0.07):
            trials.append(Trial(
                label=f"mod-concentric carrier={carrier_f} env={env_f}",
                family="concentric",
                image=modulated_gabor_stimulus(
                    size,
                    carrier_frequency=carrier_f,
                    envelope="concentric",
                    envelope_frequency=env_f,
                ),
                notes="modulated Gabor",
            ))
            trials.append(Trial(
                label=f"mod-radial carrier={carrier_f} env={env_f}",
                family="radial",
                image=modulated_gabor_stimulus(
                    size,
                    carrier_frequency=carrier_f,
                    envelope="radial",
                    envelope_frequency=env_f,
                ),
                notes="modulated Gabor",
            ))
            for sign in (+1, -1):
                trials.append(Trial(
                    label=f"mod-spiral carrier={carrier_f} env={env_f} sign={sign:+d}",
                    family="spiral",
                    image=modulated_gabor_stimulus(
                        size,
                        carrier_frequency=carrier_f,
                        envelope="spiral",
                        envelope_frequency=env_f,
                        n_spokes=8,
                        sign=sign,
                    ),
                    notes="modulated Gabor",
                ))

    # Phase / rotation robustness for a few baseline stimuli.
    for phase in (0.25, 0.5):
        trials.append(Trial(
            label=f"mod-concentric env=0.05 phase={phase:.2f}pi",
            family="concentric",
            image=modulated_gabor_stimulus(
                size,
                carrier_frequency=0.18,
                envelope="concentric",
                envelope_frequency=0.05,
                phase=phase * math.pi,
            ),
            notes="phase-shifted",
        ))
        trials.append(Trial(
            label=f"mod-radial env=0.05 phase={phase:.2f}pi",
            family="radial",
            image=modulated_gabor_stimulus(
                size,
                carrier_frequency=0.18,
                envelope="radial",
                envelope_frequency=0.05,
                phase=phase * math.pi,
            ),
            notes="phase-shifted",
        ))

    return trials


def raw_responses(model, trial: Trial, init_orients) -> dict[str, float]:
    out = model(trial.image.unsqueeze(0).unsqueeze(0))
    cells = out["lie_cells"]  # (1, n_targets, H, W)
    target_names = list(model.lie_cells.targets)
    return {
        name: float(cells[0, idx].mean().item())
        for idx, name in enumerate(target_names)
    }


def robust_stats(per_cell_values: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Per-cell median and MAD (median absolute deviation), robust to outliers."""
    stats = {}
    for k, vs in per_cell_values.items():
        sorted_vs = sorted(vs)
        n = len(sorted_vs)
        median = sorted_vs[n // 2]
        deviations = sorted([abs(v - median) for v in vs])
        mad = deviations[n // 2] * 1.4826
        stats[k] = (median, max(mad, 1e-12))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/v4_recursive_filters_sweep.png")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    matplotlib.use("Agg")

    spec = standard_v1_spec(
        n_orientations=4,
        frequencies=(0.22, 0.14, 0.09, 0.06),
        kernel_size=25,
    )
    model = RecursiveFiltersV4WithReadout(
        v1_spec=spec,
        targets=("radial", "concentric", "spiral"),
        recursive_octaves=2,
        recursive_kernel_size=41,
        use_v2_pooling=True,
        v2_sigma_to_period=2.0,
        use_v4_dc_channel=False,
        radial_subtracts_orthogonal=True,
    )
    model.eval()

    trials = build_trials(args.size)
    init_orients = list(spec.orientations)

    families = ["concentric", "radial", "spiral"]

    # First pass: collect raw responses for every trial.
    raw: list[tuple[Trial, dict[str, float]]] = []
    with torch.no_grad():
        for trial in trials:
            raw.append((trial, raw_responses(model, trial, init_orients)))

    # Per-cell statistics across the whole sweep, used for z-scoring so that
    # each cell type is judged relative to its own dynamic range rather than
    # absolute magnitude (the concentric cell has the largest absolute output
    # because the V4 bandpass picks up the most energy from rings).
    response_keys = list(raw[0][1].keys())
    pooled = {k: [r[1][k] for r in raw] for k in response_keys}
    stats = robust_stats(pooled)

    def classify(scores: dict[str, float]) -> tuple[str, dict[str, float]]:
        scored = {}
        for cell in ("concentric", "radial", "spiral"):
            mean, std = stats[cell]
            scored[cell] = (scores[cell] - mean) / max(std, 1e-12)
        return max(scored, key=scored.get), scored

    confusion = {true: {pred: 0 for pred in families} for true in families}
    per_family_total = {f: 0 for f in families}

    print(f"{'#':3s} {'family':10s} {'winner':10s}  | z-conc    z-rad     z-spi    | label")
    print("-" * 110)
    rows: list[tuple[Trial, dict[str, float], str]] = []
    for idx, (trial, scores) in enumerate(raw):
        winner, z_scored = classify(scores)
        rows.append((trial, z_scored, winner))
        confusion[trial.family][winner] += 1
        per_family_total[trial.family] += 1
        mark = "OK" if winner == trial.family else "  "
        print(
            f"{idx:3d} {trial.family:10s} {winner:10s}{mark}| "
            f"{z_scored['concentric']:+8.2f}  {z_scored['radial']:+8.2f}  {z_scored['spiral']:+8.2f} | {trial.label}"
        )

    print("\nPer-family accuracy:")
    for family in families:
        total = per_family_total[family]
        correct = confusion[family][family]
        pct = 100.0 * correct / total if total else 0.0
        print(f"  {family:10s} {correct}/{total}  ({pct:.1f}%)")

    print("\nConfusion matrix (rows = true family, cols = predicted cell):")
    print(f"{'':12s}" + "".join(f"{f:>12s}" for f in families))
    for true in families:
        row = "".join(f"{confusion[true][p]:12d}" for p in families)
        print(f"{true:12s}{row}")

    n_show = min(12, len(rows))
    fig, axes = plt.subplots(3, n_show, figsize=(1.6 * n_show, 5.0))
    by_family = {f: [r for r in rows if r[0].family == f] for f in families}
    for r_idx, family in enumerate(families):
        sample = by_family[family][:n_show]
        for c_idx in range(n_show):
            ax = axes[r_idx, c_idx]
            if c_idx < len(sample):
                trial, scores, winner = sample[c_idx]
                ax.imshow(trial.image.numpy(), cmap="gray")
                color = "tab:green" if winner == family else "tab:red"
                ax.set_title(f"{winner}", color=color, fontsize=8)
                if c_idx == 0:
                    ax.set_ylabel(family)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Recursive Filters V4 — robustness sweep (green = correct, red = miscategorised)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved sweep figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
