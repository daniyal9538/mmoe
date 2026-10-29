"""Global input gate: a feature budget in front of all heads, with optional separate value / flag gates.

The idea in plain language
--------------------------
Before any head sees the data, this gate decides which features the model may use at all. It works like
automated preprocessing (feature selection) inside the model.

* **Budget.** Exactly K features are kept; a dropped feature is closed in every head and can't be picked
  by any expert. Two selectors (``budget_type``):
  - ``topk``: a learned score per feature, top-K straight-through. Caution: a feature outside the top K
    is ineligible for every expert, so its score gets no gradient and the initial selection stays frozen.
  - ``concrete``: K concrete selector units (Abid et al., 2019), each a Gumbel-softmax over all features.
    Every feature gets a gradient at every step; hardening assigns units to K distinct features.
* **Value and flag separately** (``split_value_flag``). Each feature has two input columns: its value and
  whether it was measured. Knowing a measurement was taken can be informative even when its value is not,
  so each kept feature gets two hard-concrete gates, one per column, with an L0 price (``l_col``). At
  hardening, a kept feature with both columns closed has its stronger column reopened, so exactly K
  features reach the model.

Shapes: feature keep [F] (0/1 once hardened); column mask [2F] = [value gates | flag gates].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gates import ConcreteSelectorGate, HardConcreteGate, TopKGate


class InputGate(nn.Module):
    """Args:
        n_features: F.
        k: budget K (number of features kept).
        split_value_flag: add separate value / flag column gates.
        gen: generator for the logit jitter (N(0, 0.1) around +3, i.e. every gate starts open).
        trainable: False freezes the gates (control runs).
        budget_type: ``concrete`` or ``topk``.
    """

    def __init__(self, n_features: int, k: int, split_value_flag: bool, gen: torch.Generator, trainable: bool = True,
                 budget_type: str = "concrete"):
        super().__init__()
        self.F = n_features
        self.budget_type = budget_type
        if budget_type == "concrete":
            self.budget = ConcreteSelectorGate(n_features, k, gen, trainable)
        elif budget_type == "topk":
            self.budget = TopKGate(3.0 + 0.1 * torch.randn(n_features, generator=gen), k, trainable)
        else:
            raise ValueError(f"budget_type must be 'topk' or 'concrete', got {budget_type!r}")
        self.split = bool(split_value_flag)
        if self.split:
            self.cols = HardConcreteGate(3.0 + 0.1 * torch.randn(2 * n_features, generator=gen), trainable)

    def gates(self) -> list:
        return [self.budget] + ([self.cols] if self.split else [])

    def feature_keep(self) -> torch.Tensor:
        """[F]: 1 for features inside the budget (ones during the dense warm-up)."""
        return self.budget()

    def column_mask(self) -> torch.Tensor | None:
        """[2F] multiplier for concat(x_val, x_obs), or None without the value / flag split."""
        return self.cols() if self.split else None

    def kept_hard(self) -> torch.Tensor:
        """[F] hard selection (the selection the model would use if hardened now)."""
        if self.budget.hardened:
            return self.budget.lab_mask if self.budget_type == "concrete" else self.budget.hard_mask
        return self.budget.hard_value()

    def cols_hard(self) -> torch.Tensor:
        if not self.split:
            return torch.ones(2 * self.F, device=self.budget.logits.device)
        return self.cols.hard_mask if self.cols.hardened else self.cols.hard_value()

    @torch.no_grad()
    def harden(self, temperature: float) -> None:
        """Freeze the budget and the column gates; reopen the stronger column of any kept feature that lost
        both, so that exactly K features feed the model."""
        for g in self.gates():
            g.harden(temperature)
        if self.split:
            kept = self.kept_hard() > 0
            v, o = self.cols.hard_mask[: self.F], self.cols.hard_mask[self.F:]
            orphan = kept & (v == 0) & (o == 0)
            use_value = self.cols.logits[: self.F] >= self.cols.logits[self.F:]
            v[orphan & use_value] = 1.0                                      # views: writes go into hard_mask
            o[orphan & ~use_value] = 1.0

    def col_sparsity(self) -> torch.Tensor:
        """L0 price on open columns of kept features: mean P(column open) over the 2K columns in the budget."""
        p = self.cols.prob_open()
        kept2 = self.kept_hard().detach().repeat(2)
        return (p * kept2).sum() / kept2.sum().clamp_min(1)

    @torch.no_grad()
    def summary(self) -> dict:
        """Counts under the hard gates: features kept, and how many use value + flag, value only, flag only."""
        kept = self.kept_hard() > 0
        out = {"in_kept": int(kept.sum())}
        if self.split:
            c = self.cols_hard()
            v, o = (c[: self.F] > 0) & kept, (c[self.F:] > 0) & kept
            out.update(in_both=int((v & o).sum()), in_value_only=int((v & ~o).sum()),
                       in_flag_only=int((~v & o).sum()))
        return out
