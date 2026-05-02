"""V1 -> V2 -> V4 -> PIT -> CIT -> AIT ventral-stream model.

Extends the recursive-filters V4 backbone with progressively coarser-
scale Inception-style stages modelling the inferotemporal hierarchy
(PIT, CIT, AIT). Each IT stage is a small 1x1 Conv2d "reduction"
(analogous to Inception's channel-reducing 1x1) followed by a
per-channel Gabor power bank at that stage's spatial scale, followed
by spatial average-pool downsampling. Final classification reads
from the spatially-pooled AIT representation.

The fixed Gabor + power primitive carries through every level, so the
whole stack is hierarchical bandpass filtering with learnable channel
mixing at each level -- the structural pattern Lane (1995) proposed
for V1 -> V4 carried through to IT, with 1x1 convs supplying the
cross-channel mixing that lets the model build compound features.

Inception parallel:
    Inception 1x1 reduce + 3x3/5x5 conv  ~   ITStage 1x1 reduce + Gabor power
    Inception max/avg pool branch        ~   ITStage spatial avgpool downsample
    Inception concat across branches     ~   per-channel Gabor expands feature dim
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v4_recursive_filters import (
    FilterBankSpec,
    GaborPowerBank,
    LieGroupCells,
    RecursiveFiltersV4,
    standard_orientations,
)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Computes a learned per-channel gating from a global summary of the
    feature map: globally average-pool, run through a small MLP
    (squeeze + excite), apply sigmoid, multiply by the original feature
    map. Biologically analogous to attentional modulation in V4/IT --
    cells whose preferred features are present in the global context
    are amplified, others suppressed.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(n_channels // reduction, 4)
        self.fc1 = nn.Conv2d(n_channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(-1, -2), keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018).

    SE-style channel gate plus a 7x7 spatial gate. The channel branch
    uses both avg- and max-pool descriptors through a shared MLP; the
    spatial branch concatenates avg- and max-along-channels into a 2-ch
    map and convolves to a single saliency map. Spatial saliency is the
    "attentional spotlight" SE alone is missing.
    """

    def __init__(self, n_channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(n_channels // reduction, 4)
        self.fc1 = nn.Conv2d(n_channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, n_channels, kernel_size=1, bias=False)
        pad = spatial_kernel // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(-1, -2), keepdim=True)
        mx = x.amax(dim=(-1, -2), keepdim=True)
        c_avg = self.fc2(F.relu(self.fc1(avg)))
        c_mx = self.fc2(F.relu(self.fc1(mx)))
        x = x * torch.sigmoid(c_avg + c_mx)
        s_avg = x.mean(dim=1, keepdim=True)
        s_mx = x.amax(dim=1, keepdim=True)
        s = torch.sigmoid(self.spatial(torch.cat([s_avg, s_mx], dim=1)))
        return x * s


class CoordAttentionBlock(nn.Module):
    """Coordinate Attention (Hou et al. 2021).

    Pools features along H and W axes separately, runs a shared 1x1
    conv across the concatenated descriptors with a small bottleneck,
    and produces independent sigmoid gates for the two axes. Unlike
    SE's global-average squash, the per-axis pools preserve positional
    information, which maps naturally onto retinotopic gain control.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(n_channels // reduction, 4)
        self.fc1 = nn.Conv2d(n_channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.fc_h = nn.Conv2d(hidden, n_channels, kernel_size=1, bias=False)
        self.fc_w = nn.Conv2d(hidden, n_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(dim=-1, keepdim=True)            # (B, C, H, 1)
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)               # (B, C, H+W, 1)
        y = F.relu(self.bn(self.fc1(y)))
        y_h, y_w = y.split([h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)                  # (B, hidden, 1, W)
        a_h = torch.sigmoid(self.fc_h(y_h))            # (B, C, H, 1)
        a_w = torch.sigmoid(self.fc_w(y_w))            # (B, C, 1, W)
        return x * a_h * a_w


def _make_attention_block(kind: str, n_channels: int, reduction: int) -> nn.Module:
    if kind == "se":
        return SEBlock(n_channels, reduction=reduction)
    if kind == "cbam":
        return CBAMBlock(n_channels, reduction=reduction)
    if kind == "coord":
        return CoordAttentionBlock(n_channels, reduction=reduction)
    raise ValueError(f"Unknown attention kind: {kind!r}")


class NonLocalBlock(nn.Module):
    """Low-rank non-local self-attention (Wang et al. 2018).

    Pixel-pixel attention: each output position is a learned content-
    weighted mixture of all input positions. Q/K/V are projected through
    a small bottleneck `attn_dim` so the parameter count stays modest.
    Includes a residual connection so the block can act as a refinement
    on top of AIT's local features.

    Cost is O(N^2 * attn_dim) where N = H*W. Practical only at coarse
    stages -- intended for AIT here, where N = 16*16 = 256.
    """

    def __init__(self, n_channels: int, attn_dim: int = 16):
        super().__init__()
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5
        self.q_proj = nn.Conv2d(n_channels, attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(n_channels, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(n_channels, attn_dim, kernel_size=1, bias=False)
        self.out = nn.Conv2d(attn_dim, n_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.q_proj(x).flatten(2)                               # (B, d, N)
        k = self.k_proj(x).flatten(2)                               # (B, d, N)
        v = self.v_proj(x).flatten(2)                               # (B, d, N)
        attn = (q.transpose(-1, -2) @ k * self.scale).softmax(-1)   # (B, N, N)
        out = (v @ attn.transpose(-1, -2)).reshape(b, self.attn_dim, h, w)
        return self.out(out) + x


class DLPFC(nn.Module):
    """Dorsolateral PFC: cross-attention from learned slots to AIT.

    A small fixed-size set of "working-memory" query vectors cross-attend
    to AIT's spatial feature map and produce K slot vectors. The slots
    are projected back to AIT's channel space and pooled, replacing
    global-average pooling as the classifier's input. Models dlPFC's
    role as a working-memory / categorical-decision stage that reads
    from IT.

    Cost is O(K * N * d) with K << N, much cheaper than full self-
    attention at AIT, and parameter count scales with attn_dim, not C.
    """

    def __init__(self, n_channels: int, n_slots: int = 4, attn_dim: int = 32):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.queries = nn.Parameter(torch.randn(n_slots, attn_dim) * 0.02)
        self.k_proj = nn.Conv2d(n_channels, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(n_channels, attn_dim, kernel_size=1, bias=False)
        self.out = nn.Linear(attn_dim, n_channels)

    def forward(self, ait: torch.Tensor) -> torch.Tensor:
        b = ait.shape[0]
        k = self.k_proj(ait).flatten(2)                    # (B, d, N)
        v = self.v_proj(ait).flatten(2)                    # (B, d, N)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)    # (B, K, d)
        attn = (q @ k * self.scale).softmax(dim=-1)        # (B, K, N)
        slots = attn @ v.transpose(-1, -2)                 # (B, K, d)
        return self.out(slots).flatten(1)                  # (B, K * C)


class ITStage(nn.Module):
    """One IT-level stage: 1x1 channel reduce + per-channel Gabor power + pool.

    Parameters mirror the V4 recursive-filter pattern: a small set of
    orientations and a couple of spatial frequencies appropriate to this
    level of the hierarchy. Channels-in get reduced via a 1x1 Conv2d so
    the per-channel Gabor expansion stays manageable.

    Output shape: (B, n_reduce * n_orient * n_freq, H/down, W/down).
    """

    def __init__(
        self,
        in_channels: int,
        n_reduce: int,
        gabor_orientations: Sequence[float],
        gabor_frequencies: Sequence[float],
        kernel_size: int,
        sigma_to_period: float = 0.5,
        downsample: int = 2,
        use_relu_on_reduce: bool = True,
        use_bn: bool = True,
        use_attention: bool = False,
        attention_reduction: int = 4,
        attention_kind: str = "se",
    ):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, n_reduce, kernel_size=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(n_reduce) if use_bn else nn.Identity()
        self.use_relu_on_reduce = use_relu_on_reduce
        self.n_reduce = n_reduce
        self.attn = (
            _make_attention_block(attention_kind, n_reduce, attention_reduction)
            if use_attention else None
        )

        spec = FilterBankSpec(
            orientations=tuple(gabor_orientations),
            frequencies=tuple(gabor_frequencies),
            kernel_size=kernel_size,
            sigma_to_period=sigma_to_period,
        )
        self.spec = spec
        self.bank = GaborPowerBank(spec)
        self.bn_post = nn.BatchNorm2d(n_reduce * self.bank.n_outputs) if use_bn else nn.Identity()
        self.downsample = downsample

    @property
    def n_outputs(self) -> int:
        return self.n_reduce * self.bank.n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.bn(x)
        if self.use_relu_on_reduce:
            x = F.relu(x)
        if self.attn is not None:
            x = self.attn(x)
        b, c, h, w = x.shape
        x_flat = x.reshape(b * c, 1, h, w)
        out = self.bank(x_flat).view(b, c * self.bank.n_outputs, h, w)
        out = self.bn_post(out)
        if self.downsample > 1:
            out = F.avg_pool2d(out, kernel_size=self.downsample, stride=self.downsample)
        return out


class V4Backbone(nn.Module):
    """Wraps RecursiveFiltersV4 + V2 Gabor and returns a flat feature map.

    Concatenates the V4 power maps (bandpass + DC slot) and V2 Gabor
    features into a single (B, C, H, W) tensor that downstream IT
    stages can consume. This is the V1 -> V2 -> V4 stack already
    described, repackaged as a feature extractor for the ventral stack.
    """

    def __init__(self, backbone: RecursiveFiltersV4):
        super().__init__()
        self.backbone = backbone

    @property
    def n_outputs(self) -> int:
        v4_channels = (
            self.backbone.v1_spec.n_frequencies
            * self.backbone.v1_spec.n_orientations
            * (self.backbone.recursive_octaves
               + (1 if self.backbone.use_v4_dc_channel else 0))
            * len(self.backbone.recursive_orientations)
        )
        v2_extra = (
            self.backbone.v2_gabor.n_outputs
            if self.backbone.v2_gabor is not None else 0
        )
        return v4_channels + v2_extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        flat_blocks = []
        for block in out["v4_power"]:
            b, n_init, n_rf, n_ro, h, w = block.shape
            flat_blocks.append(block.reshape(b, n_init * n_rf * n_ro, h, w))
        v4_flat = torch.cat(flat_blocks, dim=1)
        if "v2_gabor" in out:
            return torch.cat([v4_flat, out["v2_gabor"]], dim=1)
        return v4_flat


class FullVentralStream(nn.Module):
    """V1 -> V2 -> V4 -> PIT -> CIT -> AIT classifier.

    The V1>V2>V4 backbone is the existing recursive-filters model. PIT,
    CIT, and AIT each follow the same template: 1x1 reduce + per-channel
    Gabor power bank + spatial avgpool. The Gabor frequencies decrease
    by an octave per stage, so each stage operates at coarser scale than
    the last, mirroring the increasing receptive-field size along the
    ventral stream. AIT does global average pooling; a final 1x1 Conv2d
    + softmax classifier reads from the pooled AIT representation.
    """

    def __init__(
        self,
        v4_backbone: RecursiveFiltersV4,
        n_classes: int,
        pit_n_reduce: int = 16,
        cit_n_reduce: int = 16,
        ait_n_reduce: int = 16,
        pit_frequencies: Sequence[float] = (0.05, 0.025),
        cit_frequencies: Sequence[float] = (0.05, 0.025),
        ait_frequencies: Sequence[float] = (0.05, 0.025),
        n_orientations: int = 4,
        kernel_size: int = 21,
        downsample: int = 2,
        use_skip: bool = False,
        use_attention: bool = False,
        attention_kind: str = "se",
        use_topdown: bool = False,
        topdown_hidden: int = 16,
        use_dlpfc: bool = False,
        dlpfc_slots: int = 4,
        dlpfc_dim: int = 32,
        use_nonlocal: bool = False,
        nonlocal_dim: int = 16,
    ):
        super().__init__()
        if use_skip and use_dlpfc:
            raise ValueError("use_skip and use_dlpfc are mutually exclusive readouts")
        self.backbone = V4Backbone(v4_backbone)
        self.use_skip = use_skip
        self.use_topdown = use_topdown
        self.use_dlpfc = use_dlpfc
        self.use_nonlocal = use_nonlocal

        orientations = standard_orientations(n_orientations)
        self.pit = ITStage(
            in_channels=self.backbone.n_outputs,
            n_reduce=pit_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=pit_frequencies,
            kernel_size=kernel_size,
            downsample=downsample,
            use_attention=use_attention,
            attention_kind=attention_kind,
        )
        self.cit = ITStage(
            in_channels=self.pit.n_outputs,
            n_reduce=cit_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=cit_frequencies,
            kernel_size=kernel_size,
            downsample=downsample,
            use_attention=use_attention,
            attention_kind=attention_kind,
        )
        self.ait = ITStage(
            in_channels=self.cit.n_outputs,
            n_reduce=ait_n_reduce,
            gabor_orientations=orientations,
            gabor_frequencies=ait_frequencies,
            kernel_size=kernel_size,
            downsample=1,
            use_attention=use_attention,
            attention_kind=attention_kind,
        )

        if use_topdown:
            self.td_ctx_proj = nn.Linear(self.ait.n_outputs, topdown_hidden)
            self.td_pit_gate = nn.Linear(topdown_hidden, self.pit.n_outputs)
            self.td_cit_gate = nn.Linear(topdown_hidden, self.cit.n_outputs)
            # The 2-pass forward sees two different feature distributions
            # (un-gated in pass-1, gated in pass-2). One set of BN running
            # stats can't represent both. Disable running-stat tracking on
            # the IT-stage BN modules so they always use batch stats — same
            # behavior in train and eval, no train/eval mismatch.
            for m in (*self.pit.modules(), *self.cit.modules(), *self.ait.modules()):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    m.num_batches_tracked = None

        self.input_norm = nn.BatchNorm2d(self.backbone.n_outputs)

        if use_skip:
            # Multi-scale skip: globally average-pool every stage,
            # BN-normalize per stage to balance magnitudes, concatenate,
            # feed to a single linear classifier.
            self.v4_pool_bn = nn.BatchNorm1d(self.backbone.n_outputs)
            self.pit_pool_bn = nn.BatchNorm1d(self.pit.n_outputs)
            self.cit_pool_bn = nn.BatchNorm1d(self.cit.n_outputs)
            self.ait_pool_bn = nn.BatchNorm1d(self.ait.n_outputs)
            total_dim = (
                self.backbone.n_outputs
                + self.pit.n_outputs
                + self.cit.n_outputs
                + self.ait.n_outputs
            )
            self.classifier = nn.Linear(total_dim, n_classes)
        elif use_dlpfc:
            self.dlpfc = DLPFC(self.ait.n_outputs, n_slots=dlpfc_slots, attn_dim=dlpfc_dim)
            self.classifier = nn.Linear(dlpfc_slots * self.ait.n_outputs, n_classes)
        else:
            self.classifier = nn.Conv2d(self.ait.n_outputs, n_classes, kernel_size=1)

        if use_nonlocal:
            self.nonlocal_block = NonLocalBlock(self.ait.n_outputs, attn_dim=nonlocal_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v4 = self.backbone(x)
        v4_norm = self.input_norm(v4)

        if self.use_topdown:
            # Pass 1: derive top-down context from AIT. The IT-stage BN
            # modules have track_running_stats=False (set in __init__), so
            # they always use batch stats — both passes are self-consistent
            # in train and eval. no_grad detaches context: top-down
            # projections train from pass-2's gradient, treating ctx as a
            # constant input.
            with torch.no_grad():
                pit0 = self.pit(v4_norm)
                cit0 = self.cit(pit0)
                ait0 = self.ait(cit0)
            ctx = ait0.mean(dim=(-1, -2))
            ctx_h = F.relu(self.td_ctx_proj(ctx))
            pit_gate = torch.sigmoid(self.td_pit_gate(ctx_h)).unsqueeze(-1).unsqueeze(-1)
            cit_gate = torch.sigmoid(self.td_cit_gate(ctx_h)).unsqueeze(-1).unsqueeze(-1)
            # Pass 2: re-run PIT/CIT/AIT with multiplicative gates from AIT.
            pit = self.pit(v4_norm) * pit_gate
            cit = self.cit(pit) * cit_gate
            ait = self.ait(cit)
        else:
            pit = self.pit(v4_norm)
            cit = self.cit(pit)
            ait = self.ait(cit)

        if self.use_nonlocal:
            ait = self.nonlocal_block(ait)

        if self.use_skip:
            v4_pool = self.v4_pool_bn(v4.mean(dim=(-1, -2)))
            pit_pool = self.pit_pool_bn(pit.mean(dim=(-1, -2)))
            cit_pool = self.cit_pool_bn(cit.mean(dim=(-1, -2)))
            ait_pool = self.ait_pool_bn(ait.mean(dim=(-1, -2)))
            concat = torch.cat([v4_pool, pit_pool, cit_pool, ait_pool], dim=1)
            logits = self.classifier(concat)
            return {
                "v4": v4, "pit": pit, "cit": cit, "ait": ait,
                "v4_pool": v4_pool, "pit_pool": pit_pool,
                "cit_pool": cit_pool, "ait_pool": ait_pool,
                "logits": logits,
            }

        if self.use_dlpfc:
            dlpfc_pool = self.dlpfc(ait)             # (B, C)
            logits = self.classifier(dlpfc_pool)
            return {
                "v4": v4,
                "pit": pit,
                "cit": cit,
                "ait": ait,
                "dlpfc_pool": dlpfc_pool,
                "logits": logits,
            }

        ait_pooled = ait.mean(dim=(-1, -2), keepdim=True)
        logits = self.classifier(ait_pooled).squeeze(-1).squeeze(-1)
        return {
            "v4": v4,
            "pit": pit,
            "cit": cit,
            "ait": ait,
            "ait_pooled": ait_pooled,
            "logits": logits,
        }

    @torch.no_grad()
    def recalibrate_bn(self, calibration_x: torch.Tensor, batch_size: int = 16):
        """Set BN running stats to cumulative averages over the training set.

        Switches every BatchNorm2d to cumulative-average mode
        (``momentum=None``) and runs one pass over the training data, so
        the running mean/var after the pass are the exact training-set
        statistics rather than a momentum-decayed approximation. Then
        restores the saved momenta and switches the model to eval mode.
        """
        saved = []
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                saved.append((m, m.momentum, m.training))
                m.momentum = None
                m.reset_running_stats()
                m.train()
        try:
            for i in range(0, calibration_x.shape[0], batch_size):
                _ = self(calibration_x[i:i+batch_size])
        finally:
            for m, momentum, was_training in saved:
                m.momentum = momentum
                if not was_training:
                    m.eval()
        self.eval()
