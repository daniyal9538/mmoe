"""Multi-head mixture-of-experts MLP encoder with expert-choice routing (M-MoE, EC).

Idea in plain language
----------------------
A plain MLP lets every input feature talk to every hidden unit. M-MoE instead splits the input into
*learned groups*: each head has m small experts, and each expert *chooses* the k input units it reads
(expert-choice routing, Zhou et al. 2022 — here experts choose input features, not tokens or samples, and
the choice is static: the same for every sample). A separate *null (keep) gate* per unit lets a head
ignore a unit entirely. Heads work at different granularities (e.g. 4, 16, 64 experts: coarse -> fine),
so the same feature can sit in a broad group in one head and a narrow group in another. The only thing a
head passes on is its experts' tiny outputs (d_e numbers each), which are concatenated and mapped
linearly to the embedding z.

Data flow
---------
::

    x_val [B,F] ─┐
                 ├─ concat ─► x [B, R]   (R = 2F with observed flags, F without) ───────────────┐
    x_obs [B,F] ─┘   (optional: × input-gate column mask; features outside the budget closed)   │
                                                                                                 ▼
      head h (m_h experts)                                                      (same x for every head)
      ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
      │ keep gate keep_h [U]  (per_head | global | off; × input budget)                              │
      │ routing logits [U, m_h] ─► each expert keeps its top-k eligible units ─► g_h [U, m_h] 0/1    │
      │                G_h = g_h * keep_h[:, None]                   [U, m_h]  effective mask        │
      │  x @ (W1_h ⊙ blockmask(G_h)) + b1 ─► GELU ─► dropout         [B, m_h * u]                    │
      │  grouped einsum (per expert u -> d_e)                        [B, m_h, d_e] -> [B, m_h * d_e] │
      └──────────────────────────────────────────────────────────────────────────────────────────────┘
             head 1: 4 x d_e     head 2: 16 x d_e     head 3: 64 x d_e
                   └──────────────── concat ─────────────────┘
                                       │ Linear
                                       ▼
                                  z [B, d_z]  ──► decoder (model.py)

Units: U = F (a feature's value and flag share every gate) or U = 2F with ``column_units`` (value and
flag routed separately). The gates' behaviour over training (dense warm-up -> stochastic gate learning ->
hard 0/1 fine-tuning) is driven from outside by ``schedule.PhaseController``; this module only exposes
the switches.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from . import regularizers as R
from .config import ModelConfig
from .experts import ExpertBank
from .gates import (ExpertChoiceGate, FixedNullGate, HardConcreteGate, logits_from_affinity, logits_from_assignment,
                    random_assignment, spectral_groups)
from .input_gate import InputGate


class MMoEHead(nn.Module):
    """One head: an expert-choice routing gate plus its expert bank.

    Shapes: x [B, R], keep [U] -> [B, m * d_e]
    """

    def __init__(self, n_rows: int, m: int, gate: ExpertChoiceGate, u: int, d_e: int, dropout: float):
        super().__init__()
        self.m = m
        self.gate = gate
        self.experts = ExpertBank(n_rows, m, u, d_e, dropout)

    def effective_mask(self, keep: torch.Tensor) -> torch.Tensor:
        """G [U, m] = g [U, m] * keep [U, 1]. Null units are made ineligible before the experts choose."""
        if self.gate.mode != "hard":
            self.gate.eligible.copy_((keep.detach() > 0).float())
        return self.gate() * keep.unsqueeze(1)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return self.experts(x, self.effective_mask(keep)).flatten(1)


class MMoEEncoder(nn.Module):
    """Expert-choice M-MoE encoder. See module docstring for the data flow.

    Args:
        n_features: F.
        cfg: ``ModelConfig``.
        abs_corr: [F, F] |correlation| prior (needed for ``gate_init='prior_corr'``; see ``prior.py``).
        seed: seeds clustering, logit noise and random assignments.
    Shapes: x_val [B, F], x_obs [B, F] (optional) -> z [B, d_z]
    """

    def __init__(self, n_features: int, cfg: ModelConfig, abs_corr: np.ndarray | None = None, seed: int = 0):
        super().__init__()
        cfg.validate()
        F = self.F = n_features
        self.cfg = cfg
        gen = torch.Generator().manual_seed(seed)
        ms, r = list(cfg.experts_per_head), cfg.r
        self.use_obs = cfg.use_obs_flags
        self.column_units = cfg.column_units
        U = self.U = 2 * F if self.column_units else F                    # routing units
        rows = self.rows = 2 * F if self.use_obs else F                   # input columns
        self.null_mode, self.null_type = cfg.null_mode, cfg.null_type
        k_units = int(cfg.input_budget) if cfg.input_budget is not None else U
        trainable = cfg.train_gates

        # ---- null (keep) gates: logits start at keep_init (open) ----
        n_keep = 0 if self.null_mode == "off" else (1 if self.null_mode == "global" else len(ms))
        keep_logits = [cfg.keep_init + 0.1 * torch.randn(U, generator=gen) for _ in range(n_keep)]  # jitter breaks ties
        self.keep_gates = nn.ModuleList(
            [FixedNullGate(L, cfg.null_q, trainable) if self.null_type == "fixed" else HardConcreteGate(L, trainable)
             for L in keep_logits])

        # ---- routing gates + experts per head ----
        heads = []
        for h, m in enumerate(ms):
            if cfg.gate_init == "prior_corr":
                if abs_corr is None:
                    raise ValueError("gate_init='prior_corr' needs abs_corr (see mmoe.prior.abs_corr_prior)")
                assign = spectral_groups(abs_corr, m, seed)
                aff = abs_corr
                if self.column_units:                                     # a feature's flag starts in its value's expert
                    assign = np.concatenate([assign, assign])
                    aff = np.block([[abs_corr, abs_corr], [abs_corr, abs_corr]])
            else:
                assign = random_assignment(U, m, r, gen)
            if cfg.gate_init == "prior_corr" and cfg.init_fill == "graded":
                L = logits_from_affinity(aff, assign, m, self.capacity(h, U), gen, cfg.init_scale, cfg.init_noise)
            else:
                L = logits_from_assignment(assign, m, gen, cfg.init_scale, cfg.init_noise)
            gate = ExpertChoiceGate(L, k=self.capacity(h, k_units), trainable=trainable,
                                    surrogate=cfg.ec_surrogate, noise=cfg.ec_noise)
            heads.append(MMoEHead(rows, m, gate, cfg.expert_hidden, cfg.expert_out, cfg.dropout))
        self.heads = nn.ModuleList(heads)
        self.readout = nn.Linear(sum(m * cfg.expert_out for m in ms), cfg.latent_dim)
        self.last_head_outputs: list[torch.Tensor] = []
        # Created after the heads so that, for a given seed, head initialisation matches the variants without it.
        self.input_gate = (InputGate(F, k_units, cfg.split_value_flag, gen, trainable, cfg.budget_type)
                           if cfg.input_budget is not None else None)
        if (self.input_gate is not None and self.input_gate.budget_type == "concrete"
                and cfg.capacity is None):
            # While the selection is still soft, every feature must be reachable by some expert so that its
            # selector score gets a gradient: experts use full capacity ceil(r*F/m) until hardening, then
            # ceil(r*K/m).
            for h, head in enumerate(self.heads):
                head.gate.k_final, head.gate.k = head.gate.k, self.capacity(h, U)
        for h, head in enumerate(self.heads):                             # for "how far did routing move" diagnostics
            self.register_buffer(f"init_mask{h}", head.gate.hard_value().clone())

        if not trainable:                                                 # controls: assignments frozen at init
            self.harden(cfg.phases.temp_end)
            self.set_mode("dense")                                        # warm-up still applies

    def capacity(self, h: int, n_units: int) -> int:
        """Seats per expert in head h: the fixed ``capacity`` if set, else ceil(r * n_units / m_h)."""
        c = self.cfg.capacity
        if c is not None:
            return int(c[h] if isinstance(c, list) else c)
        return math.ceil(self.cfg.r * n_units / self.heads_m(h))

    def heads_m(self, h: int) -> int:
        return self.cfg.experts_per_head[h]

    # ------------------------------------------------------------------------------------------------
    def gates(self):
        """All gate modules (routing + keep + input gate)."""
        extra = self.input_gate.gates() if self.input_gate is not None else []
        return [h.gate for h in self.heads] + list(self.keep_gates) + extra

    def _budget_keep(self) -> torch.Tensor | None:
        return self.input_gate.feature_keep() if self.input_gate is not None else None

    def _keep_values(self) -> list[torch.Tensor]:
        """Keep vector [U] for each head (one shared sample in ``global`` mode, ones when ``off``)."""
        if len(self.keep_gates) == 0:
            keeps = [torch.ones(self.U, device=self.readout.weight.device)] * len(self.heads)
        elif len(self.keep_gates) == 1:
            keeps = [self.keep_gates[0]()] * len(self.heads)
        else:
            keeps = [g() for g in self.keep_gates]
        lab = self._budget_keep()
        if lab is not None:                                               # features outside the budget: closed everywhere
            keeps = [k * lab for k in keeps]
        return keeps

    def _input(self, x_val: torch.Tensor, x_obs: torch.Tensor | None) -> torch.Tensor:
        if not self.use_obs:
            return x_val
        if x_obs is None:
            x_obs = torch.ones_like(x_val)
        x = torch.cat([x_val, x_obs.to(x_val.dtype)], dim=-1)             # [B, 2F]
        if self.input_gate is not None and self.input_gate.split:
            x = x * self.input_gate.column_mask().to(x.dtype)             # value and flag gated separately
        return x

    def forward(self, x_val: torch.Tensor, x_obs: torch.Tensor | None = None) -> torch.Tensor:
        x = self._input(x_val, x_obs)
        outs = [head(x, k) for head, k in zip(self.heads, self._keep_values())]
        self.last_head_outputs = outs                                     # for per-head probes
        return self.readout(torch.cat(outs, dim=-1))

    # ---- phase control (called by schedule.PhaseController) ---------------------------------------
    def set_mode(self, mode: str) -> None:
        for g in self.gates():
            g.set_mode(mode)

    def set_temperature(self, t: float) -> None:
        for g in self.gates():
            if not g.hardened:
                g.temperature = t

    def harden(self, temperature: float) -> None:
        """Deterministic 0/1 gates and frozen logits. Order matters: the input budget first (it decides
        eligibility and the final capacity), then the keep gates, then routing (experts pick their top-k
        among the units that survived)."""
        if self.input_gate is not None:
            self.input_gate.harden(temperature)
            for head in self.heads:                                       # capacity shrinks to the final budget
                if hasattr(head.gate, "k_final"):
                    head.gate.k = head.gate.k_final
        for g in self.keep_gates:
            g.harden(temperature)
        for head, k in zip(self.heads, self._hard_keeps()):
            head.gate.eligible.copy_((k > 0).float())
            head.gate.harden(temperature)

    # ---- hard views used by regularizers, diagnostics and analysis ------------------------------------
    def _hard_keeps(self) -> list[torch.Tensor]:
        """Per head: keep [U] 0/1 as it would be if hardened now (including the input budget)."""
        out = []
        for h in range(len(self.heads)):
            if len(self.keep_gates) == 0:
                keep = torch.ones(self.U, device=self.readout.weight.device)
            else:
                kg = self.keep_gates[0 if len(self.keep_gates) == 1 else h]
                keep = kg.hard_mask if kg.hardened else kg.hard_value()
            if self.input_gate is not None:
                kept = self.input_gate.kept_hard()
                keep = keep * (kept.repeat(2) if self.U == 2 * self.F else kept)
            out.append(keep)
        return out

    @torch.no_grad()
    def hard_assignments(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per head: (G_hard [U, m] 0/1, keep_hard [U] 0/1) — what the model would use if hardened now."""
        out = []
        for head, keep in zip(self.heads, self._hard_keeps()):
            if head.gate.hardened:
                g = head.gate.hard_mask
            else:
                saved = head.gate.eligible.clone()
                head.gate.eligible.copy_(keep)
                g = head.gate.hard_value()
                head.gate.eligible.copy_(saved)
            out.append((g * keep.unsqueeze(1), keep))
        return out

    def keep_probs(self, h: int) -> torch.Tensor:
        """[U] expected keep value for head h (hard once hardened; ones when the null is off)."""
        if len(self.keep_gates) == 0:
            return torch.ones(self.U, device=self.readout.weight.device)
        kg = self.keep_gates[0 if len(self.keep_gates) == 1 else h]
        return kg.hard_mask if kg.hardened else kg.prob_open()

    def regularizers(self) -> dict[str, torch.Tensor]:
        """{'l_mem', 'l_null'[, 'l_col']} averaged over heads (fp32 scalars).

        There is no load-balance term: expert choice is balanced exactly. ``l_mem`` acts on
        sigmoid(routing logits) because there is no well-defined expected membership under expert choice;
        it is a weak shrinkage of the logits, the ranking decides membership.
        """
        dev = self.readout.weight.device
        zero = torch.zeros((), device=dev)
        out = {"l_mem": torch.stack([R.membership_cap(h.gate.prob_open(), self.cfg.r) for h in self.heads]).mean(),
               "l_null": zero}
        if self.null_type == "learned" and len(self.keep_gates):
            out["l_null"] = torch.stack([R.null_sparsity(g.prob_open()) for g in self.keep_gates]).mean()
        if self.input_gate is not None and self.input_gate.split:
            out["l_col"] = self.input_gate.col_sparsity()                 # L0 price on open value / flag columns
        return out

    # ---- parameter accounting ---------------------------------------------------------------------------
    @torch.no_grad()
    def effective_params(self) -> int:
        """Parameters actually in use under the current hard assignments (masked W1 entries excluded,
        gate logits excluded)."""
        n = sum(p.numel() for p in self.readout.parameters())
        cols = self.input_gate.cols_hard() if (self.input_gate is not None and self.input_gate.split) else None
        for head, (G, _) in zip(self.heads, self.hard_assignments()):
            e = head.experts
            rows_per_unit = e.R // G.shape[0]                             # 2: value + flag share a unit; else 1
            if cols is None:
                w1 = rows_per_unit * e.u * G.sum().item()
            else:                                                         # only the open column rows are used
                w1 = e.u * (G * (cols[: self.F] + cols[self.F:]).unsqueeze(1)).sum().item()
            n += int(w1) + e.b1.numel() + e.W2.numel() + e.b2.numel()
        return n
