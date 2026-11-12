"""CNN architectures for Skiza.

* :class:`SingleChannelCNN` — the didactic CNN (lab equation 1): one filter per
  layer, tanh after each layer. Used as a baseline.
* :class:`CNN` — the MIMO CNN (lab equation 11) that Skiza ships with:
  layer l maps F_{l-1} features to F_l features with a K_l-tap MIMO filter,
  followed by tanh.

Both are causal and shift-equivariant: delaying the input delays the output by
the same amount (lab Proposition 1).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .filters import MIMOFilter


class CNN(nn.Module):
    """MIMO convolutional neural network.

    Args:
        taps: [K_1, ..., K_L] filter length per layer.
        features: [F_0, F_1, ..., F_L] number of features at each layer
            boundary. F_0 must match the input, F_L the output.
        fast: use the conv1d implementation (True) or the shift-operator
            reference implementation (False). Both compute the same function.

    Skiza's default is ``taps=[40, 40]`` and ``features=[1, 5, 1]``.
    """

    def __init__(self, taps: Sequence[int], features: Sequence[int], fast: bool = True):
        super().__init__()
        if len(features) != len(taps) + 1:
            raise ValueError(
                f"features must have L+1 = {len(taps) + 1} entries, got {len(features)}"
            )
        self.L = len(taps)
        self.taps = list(taps)
        self.features = list(features)
        self.layers = nn.ModuleList(
            MIMOFilter(taps[l], features[l], features[l + 1], fast=fast) for l in range(self.L)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Accept (B, N) for convenience and treat it as a single feature.
        squeeze = X.dim() == 2
        if squeeze:
            X = X.unsqueeze(1)
        for layer in self.layers:
            X = torch.tanh(layer(X))
        return X.squeeze(1) if squeeze else X


class SingleChannelCNN(CNN):
    """Didactic CNN: one feature per layer (F_l = 1 for every l)."""

    def __init__(self, taps: Sequence[int], fast: bool = True):
        super().__init__(taps, [1] * (len(taps) + 1), fast=fast)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
