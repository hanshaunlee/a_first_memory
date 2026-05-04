"""Feed-forward scorer for unit-selection logits (full neural policy head)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelectionMLP(nn.Module):
    """Maps concatenated unit-context vectors to a scalar logit per candidate."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_hidden_layers: int = 2):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, in_dim] -> [batch] logits."""
        return self.net(x).squeeze(-1)
