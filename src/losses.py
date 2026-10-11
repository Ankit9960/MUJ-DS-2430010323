"""Loss functions for the imbalance ablation.

MIT-BIH is severely imbalanced: class N is roughly 90% of beats while F and Q
are well under 1%. Plain cross-entropy therefore has an easy way to score ~90%
accuracy while never usefully detecting the classes that matter clinically.
The three losses below are the ablation's second axis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_frequency_weights(labels: np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """w_c proportional to 1 / n_c, normalised to mean 1."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def class_balanced_weights(labels: np.ndarray, beta: float = 0.9999,
                           n_classes: int = 5, max_ratio: float = 50.0) -> torch.Tensor:
    """Effective-number weighting (Cui et al., 2019), corrected for MIT-BIH.

    beta must make the effective-number cap 1/(1-beta) exceed the majority
    class, or every large class saturates to the same weight and no rebalancing
    happens. At beta=0.999 the cap is 1,000 while N holds ~38,000 beats, so S
    was weighted only 1.85x N. The ratio is then clipped because class Q holds
    ~7 beats once paced records are excluded, and would otherwise take ~78x the
    weight of S -- letting a handful of unclassifiable beats dominate.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = np.clip(w, w.min(), w.min() * max_ratio)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Class-balanced focal loss (Lin et al. 2017 focal term + CB weights).

    The focal term (1 - p_t)^gamma down-weights beats the model already gets
    right, so gradient mass moves to the hard minority beats instead of the
    tens of thousands of easy normals.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        return loss.mean()


def build_loss(name: str, train_labels: np.ndarray, n_classes: int = 5,
               gamma: float = 2.0, beta: float = 0.999) -> nn.Module:
    """Factory used by the training script; `name` is an ablation setting."""
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(train_labels, n_classes))
    if name == "focal":
        return FocalLoss(
            weight=class_balanced_weights(train_labels, beta, n_classes), gamma=gamma)
    raise ValueError(f"Unknown loss {name!r}; expected 'ce', 'weighted_ce' or 'focal'.")
