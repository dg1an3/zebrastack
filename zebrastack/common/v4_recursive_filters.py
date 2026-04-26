"""PyTorch implementation of the recursive-filters V4 model.

Based on:
    D. G. Lane (1995). A Model of Form and Texture Processing in V4 Based
    on Recursive Filtering. (document/19951207 V4 Model Paper Draft 1995.pdf)

Architecture:
    Stage 1 (V1): bank of Gabor filters at multiple spatial frequencies and
        orientations. Quadrature pairs are squared and summed to produce
        power maps (the analog of V1 complex-cell responses).
    Stage 2 (V4): for every V1 power map, a second Gabor bank is applied at
        the same orientations but only at spatial frequencies up to two
        octaves below the originating V1 frequency. Quadrature pairs are
        again squared and summed.
    Readout: linear combinations of the V4 maps along the
        (initial_orientation, recursive_orientation) plane yield cells
        selective for Lie-group eigenfunction stimuli (concentric, radial,
        spiral). The mapping follows the paper:
            concentric: recursive orientation orthogonal to initial
            radial:     recursive orientation parallel to initial
            spiral:     recursive orientation oblique to initial
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gabor_kernel(
    size: int,
    orientation: float,
    frequency: float,
    sigma: float,
    phase: float,
) -> torch.Tensor:
    half = (size - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - half,
        torch.arange(size, dtype=torch.float32) - half,
        indexing="ij",
    )
    x_rot = xs * math.cos(orientation) + ys * math.sin(orientation)
    y_rot = -xs * math.sin(orientation) + ys * math.cos(orientation)
    envelope = torch.exp(-(x_rot ** 2 + y_rot ** 2) / (2.0 * sigma ** 2))
    carrier = torch.cos(2.0 * math.pi * frequency * x_rot + phase)
    kernel = envelope * carrier
    kernel = kernel - kernel.mean()
    kernel = kernel / (kernel.abs().sum() + 1e-12)
    return kernel


def build_gabor_bank(
    orientations: Sequence[float],
    frequencies: Sequence[float],
    size: int,
    sigma_to_period: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return quadrature-pair Gabor banks of shape (n_freq, n_orient, size, size).

    The Gaussian envelope width scales inversely with frequency so each
    filter spans a roughly constant number of cycles.
    """
    even_filters: list[list[torch.Tensor]] = []
    odd_filters: list[list[torch.Tensor]] = []
    for f in frequencies:
        sigma = sigma_to_period / max(f, 1e-6)
        even_row, odd_row = [], []
        for theta in orientations:
            even_row.append(_gabor_kernel(size, theta, f, sigma, phase=0.0))
            odd_row.append(_gabor_kernel(size, theta, f, sigma, phase=math.pi / 2))
        even_filters.append(torch.stack(even_row))
        odd_filters.append(torch.stack(odd_row))
    return torch.stack(even_filters), torch.stack(odd_filters)


@dataclass(frozen=True)
class FilterBankSpec:
    orientations: tuple[float, ...]
    frequencies: tuple[float, ...]
    kernel_size: int
    sigma_to_period: float = 0.5

    @property
    def n_orientations(self) -> int:
        return len(self.orientations)

    @property
    def n_frequencies(self) -> int:
        return len(self.frequencies)


class GaborPowerBank(nn.Module):
    """Apply a Gabor filter bank to one input channel and return power maps.

    Output shape: (B, n_freq * n_orient, H, W).
    """

    def __init__(self, spec: FilterBankSpec):
        super().__init__()
        self.spec = spec
        even, odd = build_gabor_bank(
            spec.orientations,
            spec.frequencies,
            spec.kernel_size,
            spec.sigma_to_period,
        )
        flat_even = even.reshape(-1, 1, spec.kernel_size, spec.kernel_size)
        flat_odd = odd.reshape(-1, 1, spec.kernel_size, spec.kernel_size)
        self.register_buffer("even_kernels", flat_even)
        self.register_buffer("odd_kernels", flat_odd)

    @property
    def n_outputs(self) -> int:
        return self.spec.n_frequencies * self.spec.n_orientations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] != 1:
            raise ValueError(
                f"GaborPowerBank expects single-channel input, got {x.shape[1]} channels"
            )
        pad = self.spec.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        even_resp = F.conv2d(x_padded, self.even_kernels)
        odd_resp = F.conv2d(x_padded, self.odd_kernels)
        return even_resp ** 2 + odd_resp ** 2


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    half = (size - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - half,
        torch.arange(size, dtype=torch.float32) - half,
        indexing="ij",
    )
    k = torch.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


class V2Mix(nn.Module):
    """1x1 Conv2d cross-channel mixing layer for V2.

    Operates on the flattened V2 output (channels = n_init_freq * n_orient)
    and produces the same number of channels by default, so V4 can consume
    its output unchanged. Identity-initialized; set ``trainable=True`` to
    let gradient descent discover useful linear combinations of V1 power
    maps (e.g. orientation-invariant energy channels, orientation-contrast
    channels, cross-frequency channels).
    """

    def __init__(self, n_channels: int, trainable: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            for c in range(n_channels):
                self.conv.weight[c, c, 0, 0] = 1.0
        if not trainable:
            self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class V2Pool(nn.Module):
    """Per-channel Gaussian spatial pooling.

    Mirrors the LPF stage shown in Figure 2 of Lane (1995): each V1 oriented
    power map is smoothed before it enters the recursive Gabor bank. The
    pooling kernel is matched to the V1 spatial frequency so the smoothing
    radius is on the order of one V1 cycle.
    """

    def __init__(
        self,
        frequency: float,
        sigma_to_period: float = 1.0,
        size: int | None = None,
        max_size: int = 41,
    ):
        super().__init__()
        sigma = sigma_to_period / max(frequency, 1e-6)
        ks = size or (int(round(6 * sigma)) | 1)
        ks = min(ks, max_size if max_size % 2 == 1 else max_size - 1)
        self.size = ks
        self.sigma = sigma
        kernel = _gaussian_kernel(ks, sigma).view(1, 1, ks, ks)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.size // 2
        pad = min(pad, x.shape[-1] - 1, x.shape[-2] - 1)
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_padded, self.kernel, padding=self.size // 2 - pad)


class RecursiveFiltersV4(nn.Module):
    """Recursive Gabor filtering model with optional V2 pooling stage.

    Stage 1 (V1) bank uses ``v1_spec``. If ``use_v2_pooling`` is true, each
    V1 power map is smoothed by a Gaussian (one per V1 frequency band) before
    being passed to Stage 2. Stage 2 applies a recursive Gabor bank to each
    pooled V1 channel; banks are restricted to spatial frequencies below the
    originating V1 frequency by at most ``recursive_octaves``. When
    ``include_dc_channel`` is true, a sum of the smoothed V1 maps is also
    exposed as a DC envelope, which the radial readout can use to detect the
    "single segment" structure that bandpass Gabors miss.
    """

    def __init__(
        self,
        v1_spec: FilterBankSpec,
        recursive_orientations: Sequence[float] | None = None,
        recursive_octaves: int = 2,
        recursive_kernel_size: int | None = None,
        use_v2_pooling: bool = True,
        v2_sigma_to_period: float = 1.0,
        include_dc_channel: bool = True,
        use_v2_mixing: bool = False,
        trainable_v2_mix: bool = True,
    ):
        super().__init__()
        self.v1_spec = v1_spec
        self.v1 = GaborPowerBank(v1_spec)

        if recursive_orientations is None:
            recursive_orientations = v1_spec.orientations
        self.recursive_orientations = tuple(recursive_orientations)
        self.recursive_octaves = recursive_octaves
        self.use_v2_pooling = use_v2_pooling
        self.include_dc_channel = include_dc_channel
        self.use_v2_mixing = use_v2_mixing

        if use_v2_pooling:
            self.v2_pool = nn.ModuleList(
                [V2Pool(f, sigma_to_period=v2_sigma_to_period) for f in v1_spec.frequencies]
            )
        else:
            self.v2_pool = None

        if use_v2_mixing:
            n_v2_channels = v1_spec.n_frequencies * v1_spec.n_orientations
            self.v2_mix = V2Mix(n_v2_channels, trainable=trainable_v2_mix)
        else:
            self.v2_mix = None

        ks = recursive_kernel_size or (v1_spec.kernel_size * (1 << recursive_octaves))
        self.recursive_kernel_size = ks

        v4_banks: list[GaborPowerBank | None] = []
        v4_freqs: list[tuple[float, ...]] = []
        for f in v1_spec.frequencies:
            recursive_f = [
                f / (2 ** k) for k in range(1, recursive_octaves + 1)
            ]
            v4_freqs.append(tuple(recursive_f))
            if not recursive_f:
                v4_banks.append(None)
                continue
            spec_r = FilterBankSpec(
                orientations=tuple(self.recursive_orientations),
                frequencies=tuple(recursive_f),
                kernel_size=ks,
                sigma_to_period=v1_spec.sigma_to_period,
            )
            v4_banks.append(GaborPowerBank(spec_r))
        self.v4_banks = nn.ModuleList([b for b in v4_banks if b is not None])
        self.v4_freqs = v4_freqs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v1_power = self.v1(x)
        b, _, h, w = v1_power.shape
        n_f = self.v1_spec.n_frequencies
        n_o = self.v1_spec.n_orientations
        v1_grouped = v1_power.view(b, n_f, n_o, h, w)

        if self.v2_pool is not None:
            pooled = []
            for f_idx, pool in enumerate(self.v2_pool):
                channel = v1_grouped[:, f_idx].reshape(b * n_o, 1, h, w)
                channel = pool(channel).view(b, n_o, h, w)
                pooled.append(channel)
            v2_grouped = torch.stack(pooled, dim=1)
        else:
            v2_grouped = v1_grouped

        if self.v2_mix is not None:
            v2_flat = v2_grouped.reshape(b, n_f * n_o, h, w)
            v2_flat = self.v2_mix(v2_flat)
            v2_grouped = v2_flat.view(b, n_f, n_o, h, w)

        v4_per_freq: list[torch.Tensor] = []
        for f_idx, bank in enumerate(self.v4_banks):
            channels = v2_grouped[:, f_idx]
            channels_flat = channels.reshape(b * n_o, 1, h, w)
            recursive_power = bank(channels_flat)
            n_rf = len(self.v4_freqs[f_idx])
            n_ro = len(self.recursive_orientations)
            recursive_power = recursive_power.view(b, n_o, n_rf, n_ro, h, w)
            v4_per_freq.append(recursive_power)

        out = {
            "v1_power": v1_grouped,
            "v2_pooled": v2_grouped,
            "v4_power": v4_per_freq,
        }
        if self.include_dc_channel:
            out["v2_dc"] = v2_grouped
        return out


_LIE_GROUP_DELTAS: dict[str, tuple[float, ...]] = {
    "radial": (0.0,),
    "concentric": (math.pi / 2,),
    "spiral_cw": (math.pi / 4,),
    "spiral_ccw": (-math.pi / 4,),
    "spiral": (math.pi / 4, -math.pi / 4),
}


def _closest_orient_index(
    target_theta: float, orientations: Sequence[float]
) -> int:
    def circular_distance(k: int) -> float:
        diff = (orientations[k] - target_theta) % math.pi
        return min(diff, math.pi - diff)

    return min(range(len(orientations)), key=circular_distance)


def lie_group_readout(
    v4_per_freq: Sequence[torch.Tensor],
    initial_orientations: Sequence[float],
    recursive_orientations: Sequence[float],
    target: str,
    initial_freq_indices: Sequence[int] | None = None,
    recursive_freq_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Combine V4 maps for a Lie-group selective cell.

    Sums over the requested initial and recursive spatial frequencies, then
    over the (initial_orient, recursive_orient) pairs that satisfy the
    target orientation relationship:

        radial      - recursive orientation parallel to initial
        concentric  - recursive orientation orthogonal to initial
        spiral_cw   - recursive orientation +45 deg from initial
        spiral_ccw  - recursive orientation -45 deg from initial
        spiral      - both oblique signs (polarity invariant)
    """
    if target not in _LIE_GROUP_DELTAS:
        raise ValueError(f"Unknown target: {target}")
    deltas = _LIE_GROUP_DELTAS[target]

    if initial_freq_indices is None:
        initial_freq_indices = range(len(v4_per_freq))

    accum = None
    n_pairs = 0
    for f_idx in initial_freq_indices:
        block = v4_per_freq[f_idx]
        n_rf = block.shape[2]
        rfreq_idxs = (
            range(n_rf) if recursive_freq_indices is None else recursive_freq_indices
        )
        for rf_idx in rfreq_idxs:
            if rf_idx >= n_rf:
                continue
            slab = block[:, :, rf_idx]
            for i, theta in enumerate(initial_orientations):
                for delta in deltas:
                    target_theta = (theta + delta) % math.pi
                    j = _closest_orient_index(target_theta, recursive_orientations)
                    contrib = slab[:, i, j]
                    accum = contrib if accum is None else accum + contrib
                    n_pairs += 1

    if accum is None or n_pairs == 0:
        raise ValueError("No matching V4 frequencies found for readout.")
    return accum / n_pairs


def radial_dc_readout(
    v2_dc: torch.Tensor,
    initial_orientations: Sequence[float],
    initial_freq_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Detect the radial "single segment" pattern from the V2 envelope.

    The radial stimulus produces a single elongated blob in each oriented
    V1 power map (Lane 1995, p. 4); concentric stimuli produce many small
    parallel blobs distributed across the field. Bandpass recursive Gabors
    reject this DC structure, so the readout instead measures the spatial
    sparsity of the V2 envelope per orientation channel via the
    peak-to-mean ratio. Radial maps have a single high peak (high ratio);
    concentric maps spread their power over many similar peaks (lower
    ratio).
    """
    if initial_freq_indices is None:
        initial_freq_indices = range(v2_dc.shape[1])
    contrasts = []
    for f_idx in initial_freq_indices:
        channel = v2_dc[:, f_idx]
        per_orient_max = channel.amax(dim=(-1, -2))
        per_orient_mean = channel.mean(dim=(-1, -2)).clamp_min(1e-8)
        ratio = per_orient_max / per_orient_mean
        contrasts.append(ratio.mean(dim=-1))
    return torch.stack(contrasts, dim=-1).mean(dim=-1)


def total_v4_energy(v4_per_freq: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean V4 power across the entire bank. Useful for normalization."""
    total = None
    n = 0
    for block in v4_per_freq:
        contrib = block.sum(dim=(1, 2, 3))
        total = contrib if total is None else total + contrib
        n += block.shape[1] * block.shape[2] * block.shape[3]
    if total is None or n == 0:
        raise ValueError("v4_per_freq is empty.")
    return total / n


class LieGroupCells(nn.Module):
    """Lie-group selective V4 cells implemented as a 1x1 Conv2d.

    The recursive-filters V4 stage produces maps indexed by
    (initial_freq, initial_orient, recursive_freq, recursive_orient). A
    Lie-group selective cell is a linear combination of those maps at each
    pixel: pick (init_orient, recursive_orient) pairs whose orientation
    difference Δθ matches the cell's preferred symmetry, sum them, and
    optionally square the result.

        Δθ = 0           radial cells       (parallel)
        Δθ = π/2         concentric cells   (orthogonal)
        Δθ = ±π/4        spiral cells       (oblique)

    Since the readout is a per-pixel linear projection over channels, it is
    exactly a 1x1 Conv2d. The weights are initialized from the
    orientation-matching pattern so the layer reproduces the analytic
    readout out of the box; setting ``trainable=True`` lets gradient
    descent refine them. ``square=True`` adds a power transform consistent
    with the V1 → V4 squaring pipeline.

    The expected input is the ``v4_power`` list returned by
    ``RecursiveFiltersV4``; the layer flattens it into a single
    (B, C, H, W) tensor and applies the conv. ``forward_flat`` accepts the
    pre-flattened tensor for use in pure-tensor pipelines.
    """

    def __init__(
        self,
        targets: Sequence[str],
        initial_orientations: Sequence[float],
        recursive_orientations: Sequence[float],
        n_initial_frequencies: int,
        n_recursive_frequencies: int,
        square: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.targets = tuple(targets)
        self.initial_orientations = tuple(initial_orientations)
        self.recursive_orientations = tuple(recursive_orientations)
        self.n_initial_frequencies = n_initial_frequencies
        self.n_recursive_frequencies = n_recursive_frequencies
        self.square = square

        n_init_o = len(initial_orientations)
        n_rec_o = len(recursive_orientations)
        in_channels = (
            n_initial_frequencies * n_init_o * n_recursive_frequencies * n_rec_o
        )
        n_targets = len(self.targets)

        weight = torch.zeros(n_targets, in_channels, 1, 1)
        for t_idx, target in enumerate(self.targets):
            if target not in _LIE_GROUP_DELTAS:
                raise ValueError(f"Unknown target: {target}")
            deltas = _LIE_GROUP_DELTAS[target]
            n_contrib = 0
            for delta in deltas:
                for i, theta_i in enumerate(initial_orientations):
                    target_theta = (theta_i + delta) % math.pi
                    j = _closest_orient_index(target_theta, recursive_orientations)
                    for fi in range(n_initial_frequencies):
                        for rfi in range(n_recursive_frequencies):
                            ch = (
                                fi * (n_init_o * n_recursive_frequencies * n_rec_o)
                                + i * (n_recursive_frequencies * n_rec_o)
                                + rfi * n_rec_o
                                + j
                            )
                            weight[t_idx, ch, 0, 0] = 1.0
                            n_contrib += 1
            if n_contrib > 0:
                weight[t_idx] /= n_contrib

        self.conv = nn.Conv2d(in_channels, n_targets, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        if not trainable:
            self.conv.weight.requires_grad_(False)

    def flatten_v4(self, v4_per_freq: Sequence[torch.Tensor]) -> torch.Tensor:
        """Flatten the per-frequency V4 list into a single (B, C, H, W) tensor.

        Channel order matches the conv weights:
            (init_freq, init_orient, recursive_freq, recursive_orient).
        """
        if len(v4_per_freq) != self.n_initial_frequencies:
            raise ValueError(
                f"Expected {self.n_initial_frequencies} init-freq blocks, "
                f"got {len(v4_per_freq)}"
            )
        flat_blocks = []
        for block in v4_per_freq:
            b, n_init, n_rf, n_ro, h, w = block.shape
            if n_rf != self.n_recursive_frequencies:
                raise ValueError(
                    f"Block recursive_freq={n_rf} does not match "
                    f"n_recursive_frequencies={self.n_recursive_frequencies}"
                )
            flat_blocks.append(block.reshape(b, n_init * n_rf * n_ro, h, w))
        return torch.cat(flat_blocks, dim=1)

    def forward(self, v4_per_freq: Sequence[torch.Tensor]) -> torch.Tensor:
        """Returns (B, n_targets, H, W)."""
        flat = self.flatten_v4(v4_per_freq)
        return self.forward_flat(flat)

    def forward_flat(self, v4_flat: torch.Tensor) -> torch.Tensor:
        out = self.conv(v4_flat)
        if self.square:
            out = out ** 2
        return out

    def cell_index(self, target: str) -> int:
        return self.targets.index(target)


class RecursiveFiltersV4WithReadout(nn.Module):
    """RecursiveFiltersV4 with a built-in LieGroupCells readout layer.

    Wraps RecursiveFiltersV4 and a LieGroupCells module into a single
    end-to-end forward pass. ``forward`` returns a dict containing the
    intermediate activations and the per-cell maps under ``"lie_cells"``.
    """

    def __init__(
        self,
        v1_spec: FilterBankSpec,
        targets: Sequence[str] = ("radial", "concentric", "spiral"),
        recursive_orientations: Sequence[float] | None = None,
        recursive_octaves: int = 2,
        recursive_kernel_size: int | None = None,
        use_v2_pooling: bool = True,
        v2_sigma_to_period: float = 1.0,
        use_v2_mixing: bool = False,
        trainable_v2_mix: bool = True,
        square_lie_cells: bool = False,
        trainable_lie_cells: bool = False,
    ):
        super().__init__()
        self.backbone = RecursiveFiltersV4(
            v1_spec=v1_spec,
            recursive_orientations=recursive_orientations,
            recursive_octaves=recursive_octaves,
            recursive_kernel_size=recursive_kernel_size,
            use_v2_pooling=use_v2_pooling,
            v2_sigma_to_period=v2_sigma_to_period,
            use_v2_mixing=use_v2_mixing,
            trainable_v2_mix=trainable_v2_mix,
        )
        self.lie_cells = LieGroupCells(
            targets=targets,
            initial_orientations=v1_spec.orientations,
            recursive_orientations=self.backbone.recursive_orientations,
            n_initial_frequencies=v1_spec.n_frequencies,
            n_recursive_frequencies=recursive_octaves,
            square=square_lie_cells,
            trainable=trainable_lie_cells,
        )

    def forward(self, x: torch.Tensor) -> dict:
        out = self.backbone(x)
        out["lie_cells"] = self.lie_cells(out["v4_power"])
        return out


def standard_orientations(n: int) -> tuple[float, ...]:
    return tuple(k * math.pi / n for k in range(n))


def standard_v1_spec(
    n_orientations: int = 4,
    frequencies: Sequence[float] = (0.25, 0.125, 0.0625),
    kernel_size: int = 21,
    sigma_to_period: float = 0.5,
) -> FilterBankSpec:
    return FilterBankSpec(
        orientations=standard_orientations(n_orientations),
        frequencies=tuple(frequencies),
        kernel_size=kernel_size,
        sigma_to_period=sigma_to_period,
    )
