"""Regularizers on gate probabilities.

They shape the learned grouping without touching the reconstruction objective:

* ``membership_cap``  L_mem = mean_u ReLU(sum_j p[u, j] - r)^2
    A soft cap: a unit may join up to r experts for free and pays quadratically beyond that. Under expert
    choice p = sigmoid(routing logits), so it only shrinks logits; the top-k ranking decides membership.
* ``null_sparsity``   L_null = mean_u P(keep[u] > 0)
    An L0-style price on using a unit at all, so units that do not help reconstruction drift into the null.

Total: L = L_recon + lambda_mem L_mem + lambda_null L_null (+ lambda_col L_col), lambdas ramped by
``schedule.py``. There is no load-balance term: expert choice gives every expert exactly k units.
"""
from __future__ import annotations

import torch


def membership_cap(p_g: torch.Tensor, r: float) -> torch.Tensor:
    """p_g [U, m] expected memberships -> scalar. Zero when every unit has <= r expected memberships."""
    return torch.relu(p_g.sum(1) - r).pow(2).mean()


def null_sparsity(p_keep: torch.Tensor) -> torch.Tensor:
    """p_keep [U] -> mean probability of keeping a unit."""
    return p_keep.mean()
