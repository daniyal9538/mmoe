"""The expert bank: m small two-layer MLPs per head, each reading only the units its gates allow.

All m experts of a head are computed together (vectorised) instead of looping:

* Layer 1 is one dense weight ``W1 [R, m*u]`` over the R input rows (R = 2F with observed flags:
  values then flags; R = F without). Before the matmul it is multiplied elementwise by a *block mask*
  built from the effective gate matrix ``G [U, m]``:
  - one unit per feature (U = F) with flags: rows f and F + f of expert j's u columns are scaled by
    ``G[f, j]``, so a closed gate removes both the value and whether it was measured;
  - column units (U = 2F) or no flags (U = R): row i is scaled by ``G[i, j]``.
* Layer 2 is a grouped projection u -> d_e per expert (an einsum over the expert axis), so experts never
  mix before the readout.

Why masks instead of gathering features: the mask keeps shapes static, lets a unit belong to any number
of experts, and makes the gate values differentiable multipliers of the weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class ExpertBank(nn.Module):
    """m experts, each R -(masked)-> u -> d_e.

    Args:
        n_rows: R, number of input columns (2F with observed flags, F without).
        m: number of experts in this head.
        u: hidden units per expert.
        d_e: output dim per expert.
        dropout: dropout on the expert hidden layer.
    Shapes: x [B, R], G [U, m] (U = R or U = R / 2) -> out [B, m, d_e]
    """

    def __init__(self, n_rows: int, m: int, u: int = 16, d_e: int = 2, dropout: float = 0.1):
        super().__init__()
        self.R, self.m, self.u, self.d_e = n_rows, m, u, d_e
        self.W1 = nn.Parameter(torch.empty(n_rows, m * u))
        self.b1 = nn.Parameter(torch.empty(m * u))
        self.W2 = nn.Parameter(torch.empty(m, u, d_e))
        self.b2 = nn.Parameter(torch.empty(m, d_e))
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def _init_cols(self, cols: slice | torch.Tensor, experts) -> None:
        """nn.Linear-style uniform init (bound 1/sqrt(fan_in)) for the given layer-1 columns / experts."""
        b1 = 1 / math.sqrt(self.R)                                       # layer-1 fan-in = R (dense view)
        b2 = 1 / math.sqrt(self.u)
        with torch.no_grad():
            self.W1[:, cols].uniform_(-b1, b1)
            self.b1[cols].uniform_(-b1, b1)
            self.W2[experts].uniform_(-b2, b2)
            self.b2[experts].uniform_(-b2, b2)

    def reset_parameters(self) -> None:
        self._init_cols(slice(None), slice(None))

    def expert_cols(self, j: int) -> slice:
        """Layer-1 columns belonging to expert j."""
        return slice(j * self.u, (j + 1) * self.u)

    def block_mask(self, G: torch.Tensor) -> torch.Tensor:
        """[U, m] unit gates -> [R, m*u] weight mask (a feature's value and flag rows share its gate when
        U = R / 2)."""
        rows = G if G.shape[0] == self.R else G.repeat(2, 1)
        return rows.repeat_interleave(self.u, dim=1)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        W = self.W1 * self.block_mask(G).to(self.W1.dtype)               # zero out closed (unit, expert) pairs
        h = self.drop(Fn.gelu(x @ W.to(x.dtype) + self.b1.to(x.dtype)))  # [B, m*u]
        h = h.view(h.shape[0], self.m, self.u)                           # [B, m, u]
        return torch.einsum("bmu,mud->bmd", h, self.W2.to(h.dtype)) + self.b2.to(h.dtype)  # [B, m, d_e]
