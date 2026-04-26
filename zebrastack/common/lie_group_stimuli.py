"""Lie-group eigenfunction stimulus generators.

These stimuli are eigenfunctions of one-parameter Lie groups acting on the
plane (rotations, dilations, and their combinations). The recursive
filters V4 model is designed to discriminate among them despite their
identical orientation/spatial-frequency power spectra.
"""

from __future__ import annotations

import math

import torch


def _grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    half = (size - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - half,
        torch.arange(size, dtype=torch.float32) - half,
        indexing="ij",
    )
    return xs, ys


def _aperture(size: int, fraction: float = 0.45) -> torch.Tensor:
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    radius = fraction * size
    edge = 0.05 * size
    return torch.clamp((radius - r) / max(edge, 1e-6), 0.0, 1.0)


def concentric_stimulus(size: int = 128, frequency: float = 0.1) -> torch.Tensor:
    """Concentric (radially-symmetric) ring pattern: cos(2 pi f r)."""
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    image = torch.cos(2.0 * math.pi * frequency * r)
    return image * _aperture(size)


def radial_stimulus(size: int = 128, n_spokes: int = 8) -> torch.Tensor:
    """Radial (angular) pattern: cos(n_spokes * theta)."""
    xs, ys = _grid(size)
    theta = torch.atan2(ys, xs)
    image = torch.cos(n_spokes * theta)
    return image * _aperture(size)


def annular_radial_stimulus(
    size: int = 128,
    target_frequency: float = 0.18,
    inner_fraction: float = 0.18,
    outer_fraction: float = 0.40,
) -> torch.Tensor:
    """Radial spokes confined to an annulus, with the spoke count chosen so
    the local stripe frequency at mid-radius matches ``target_frequency``."""
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    theta = torch.atan2(ys, xs)
    mid_radius = 0.5 * (inner_fraction + outer_fraction) * size
    n_spokes = max(2, int(round(target_frequency * 2 * math.pi * mid_radius)))
    image = torch.cos(n_spokes * theta)
    inner = inner_fraction * size
    outer = outer_fraction * size
    edge = 0.04 * size
    annulus = torch.clamp((r - inner) / max(edge, 1e-6), 0.0, 1.0)
    annulus = annulus * torch.clamp((outer - r) / max(edge, 1e-6), 0.0, 1.0)
    return image * annulus


def spiral_stimulus(
    size: int = 128,
    frequency: float = 0.1,
    n_spokes: int = 8,
    sign: int = 1,
) -> torch.Tensor:
    """Spiral pattern combining radial and angular phase: cos(2 pi f r + sign * n theta)."""
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    theta = torch.atan2(ys, xs)
    image = torch.cos(2.0 * math.pi * frequency * r + sign * n_spokes * theta)
    return image * _aperture(size)


def herringbone_stimulus(
    size: int = 128,
    frequency: float = 0.15,
    chevron_period: int = 16,
    flip: bool = False,
) -> torch.Tensor:
    """Herringbone-style pattern with two oblique components alternating in stripes."""
    xs, ys = _grid(size)
    stripe = ((ys + size / 2) // chevron_period).long() % 2
    if flip:
        stripe = 1 - stripe
    angle_a = math.pi / 4
    angle_b = -math.pi / 4
    grating_a = torch.cos(2.0 * math.pi * frequency * (xs * math.cos(angle_a) + ys * math.sin(angle_a)))
    grating_b = torch.cos(2.0 * math.pi * frequency * (xs * math.cos(angle_b) + ys * math.sin(angle_b)))
    image = torch.where(stripe.bool(), grating_a, grating_b)
    return image * _aperture(size)


def plaid_stimulus(size: int = 128, frequency: float = 0.1) -> torch.Tensor:
    """Crossed orthogonal gratings (control: same power-spectrum-style as concentric/radial)."""
    xs, ys = _grid(size)
    image = torch.cos(2.0 * math.pi * frequency * xs) + torch.cos(2.0 * math.pi * frequency * ys)
    return image * _aperture(size)


def random_concentric(size: int, rng: torch.Generator) -> torch.Tensor:
    """Random concentric stimulus: polar grating or modulated Gabor."""
    use_modulated = bool(torch.rand(1, generator=rng).item() < 0.5)
    if use_modulated:
        carrier = float(torch.empty(1).uniform_(0.14, 0.24, generator=rng).item())
        env = float(torch.empty(1).uniform_(0.03, 0.08, generator=rng).item())
        phase = float(torch.empty(1).uniform_(0.0, 2 * math.pi, generator=rng).item())
        return modulated_gabor_stimulus(
            size, carrier_frequency=carrier, envelope="concentric",
            envelope_frequency=env, phase=phase,
        )
    f = float(torch.empty(1).uniform_(0.06, 0.20, generator=rng).item())
    return concentric_stimulus(size, frequency=f)


def random_radial(size: int, rng: torch.Generator) -> torch.Tensor:
    """Random radial stimulus: annular spokes or modulated Gabor."""
    use_modulated = bool(torch.rand(1, generator=rng).item() < 0.5)
    if use_modulated:
        carrier = float(torch.empty(1).uniform_(0.14, 0.24, generator=rng).item())
        env = float(torch.empty(1).uniform_(0.03, 0.08, generator=rng).item())
        phase = float(torch.empty(1).uniform_(0.0, 2 * math.pi, generator=rng).item())
        return modulated_gabor_stimulus(
            size, carrier_frequency=carrier, envelope="radial",
            envelope_frequency=env, phase=phase,
        )
    f = float(torch.empty(1).uniform_(0.06, 0.20, generator=rng).item())
    return annular_radial_stimulus(size, target_frequency=f)


def random_spiral(size: int, rng: torch.Generator) -> torch.Tensor:
    """Random spiral stimulus: polar grating or modulated Gabor, random sign."""
    sign = 1 if bool(torch.rand(1, generator=rng).item() < 0.5) else -1
    use_modulated = bool(torch.rand(1, generator=rng).item() < 0.5)
    if use_modulated:
        carrier = float(torch.empty(1).uniform_(0.14, 0.24, generator=rng).item())
        env = float(torch.empty(1).uniform_(0.03, 0.08, generator=rng).item())
        n = int(torch.randint(6, 14, (1,), generator=rng).item())
        phase = float(torch.empty(1).uniform_(0.0, 2 * math.pi, generator=rng).item())
        return modulated_gabor_stimulus(
            size, carrier_frequency=carrier, envelope="spiral",
            envelope_frequency=env, n_spokes=n, sign=sign, phase=phase,
        )
    f = float(torch.empty(1).uniform_(0.06, 0.20, generator=rng).item())
    n = int(torch.randint(6, 14, (1,), generator=rng).item())
    return spiral_stimulus(size, frequency=f, n_spokes=n, sign=sign)


def modulated_gabor_stimulus(
    size: int = 128,
    carrier_frequency: float = 0.18,
    envelope: str = "concentric",
    envelope_frequency: float = 0.04,
    n_spokes: int | None = None,
    sign: int = 1,
    phase: float = 0.0,
) -> torch.Tensor:
    """High-frequency oriented carrier modulated by a low-frequency polar envelope.

    Implements the construction described in Lane (1995, p. 5):
        "stimuli created by amplitude modulating Gabor functions with lower
         spatial frequency Gabor functions."

    The carrier is a sinusoidal grating with orientation perpendicular to
    the local stripe of the envelope (so the carrier rides along the
    contour). The envelope determines the macro-scale Lie-group structure
    (concentric, radial, spiral). Setting envelope_frequency to 0 produces
    a uniform-envelope stimulus; setting carrier_frequency = envelope
    frequency reduces to the simple polar grating.
    """
    xs, ys = _grid(size)
    r = torch.sqrt(xs ** 2 + ys ** 2)
    theta = torch.atan2(ys, xs)

    if envelope == "concentric":
        env = torch.cos(2.0 * math.pi * envelope_frequency * r + phase)
        carrier_phase = 2.0 * math.pi * carrier_frequency * r
    elif envelope == "radial":
        n = n_spokes if n_spokes is not None else max(2, int(round(envelope_frequency * 2 * math.pi * 30)))
        env = torch.cos(n * theta + phase)
        carrier_arg = -theta
        carrier_phase = 2.0 * math.pi * carrier_frequency * (
            xs * torch.cos(carrier_arg) + ys * torch.sin(carrier_arg)
        )
    elif envelope == "spiral":
        n = n_spokes if n_spokes is not None else 8
        env = torch.cos(2.0 * math.pi * envelope_frequency * r + sign * n * theta + phase)
        carrier_phase = 2.0 * math.pi * carrier_frequency * r
    else:
        raise ValueError(f"Unknown envelope: {envelope}")

    carrier = torch.cos(carrier_phase)
    image = env * carrier
    return image * _aperture(size)
