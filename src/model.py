"""1-D CNN architectures.

BaselineCNN     -- single-scale: one beat window in, class out. This is the
                   shape of a typical published MIT-BIH CNN and serves as the
                   reference point for the ablation.
MultiScaleCNN   -- the proposed model: a beat branch and a rhythm-context
                   branch with independent weights, fused with RR-interval
                   features before the classifier.

The multi-scale idea targets a specific, diagnosable failure. Supraventricular
ectopic beats (class S) are defined by *when* they arrive, not by how the single
beat looks -- a premature atrial beat can be morphologically near-identical to a
normal one. A model that only ever sees a 1-second window centred on the R peak
is structurally blind to that, which is why S recall is the weakest column in
almost every published inter-patient table.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, pool: int = 2, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Encoder1D(nn.Module):
    """Shared convolutional trunk: (B, C, L) -> (B, 128) embedding."""

    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel=7, dropout=dropout),
            ConvBlock(32, 64, kernel=5, dropout=dropout),
            ConvBlock(64, 128, kernel=3, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 128

    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1)


class BaselineCNN(nn.Module):
    """Single-scale baseline: beat window only (optionally + RR features)."""

    def __init__(self, in_channels: int = 2, n_classes: int = 5,
                 dropout: float = 0.3, use_rr: bool = False, n_rr: int = 4):
        super().__init__()
        self.use_rr = use_rr
        self.encoder = Encoder1D(in_channels)
        feat_dim = self.encoder.out_dim + (n_rr if use_rr else 0)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, beat, context=None, rr=None):
        z = self.encoder(beat)
        if self.use_rr:
            z = torch.cat([z, rr], dim=1)
        return self.head(z)


class MultiScaleCNN(nn.Module):
    """Proposed model: beat branch + rhythm-context branch + RR features."""

    def __init__(self, in_channels: int = 2, n_classes: int = 5,
                 dropout: float = 0.3, use_rr: bool = True, n_rr: int = 4):
        super().__init__()
        self.use_rr = use_rr
        self.beat_encoder = Encoder1D(in_channels)
        self.ctx_encoder = Encoder1D(in_channels)
        feat_dim = self.beat_encoder.out_dim * 2 + (n_rr if use_rr else 0)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, beat, context=None, rr=None):
        if context is None:
            raise ValueError("MultiScaleCNN requires the rhythm-context window.")
        parts = [self.beat_encoder(beat), self.ctx_encoder(context)]
        if self.use_rr:
            parts.append(rr)
        return self.head(torch.cat(parts, dim=1))


def build_model(arch: str, in_channels: int, n_classes: int = 5,
                dropout: float = 0.3, use_rr: bool = True) -> nn.Module:
    if arch == "baseline":
        return BaselineCNN(in_channels, n_classes, dropout, use_rr=use_rr)
    if arch == "multiscale":
        return MultiScaleCNN(in_channels, n_classes, dropout, use_rr=use_rr)
    raise ValueError(f"Unknown arch {arch!r}; expected 'baseline' or 'multiscale'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
