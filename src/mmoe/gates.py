"""Gates: the switches that decide which input unit feeds which expert, and which units a head ignores.

Every gate owns a tensor of learnable *logits* and is in one of three modes, set by the phase controller
(``schedule.py``):

* ``dense``      — warm-up: every gate returns 1, so the model is a plain dense MLP over all units.
* ``stochastic`` — gate learning: gates are random but differentiable (hard-concrete / Gumbel samples,
                    or straight-through top-k), so gradient descent can learn *which* switches to open.
* ``hard``       — after hardening: each gate is a fixed 0/1 value computed once from its logits; the
                    logits are frozen.

Gate families:

* ``ExpertChoiceGate``     — routing: every expert picks its top-k units (Zhou et al., NeurIPS 2022),
                              straight-through.
* ``HardConcreteGate``     — independent L0 gates (Louizos et al., ICLR 2018). Learned null (keep) gate and
                              the value / flag column gates of the input budget.
* ``FixedNullGate``        — drops exactly a fraction q of units (lowest keep logits), straight-through.
* ``TopKGate``             — input budget, top-K straight-through.
* ``ConcreteSelectorGate`` — input budget, K concrete selector units (Abid et al., ICML 2019).

The module also holds the routing-logit *initialisers* (correlation-prior spectral clustering, random).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

GAMMA, ZETA = -0.1, 1.1          # hard-concrete stretch limits (Louizos et al.)
MODES = ("dense", "stochastic", "hard")


class Gate(nn.Module):
    """Base class: logits + mode + temperature + hardening.

    Args:
        logits: initial logits (any shape; e.g. [U, m] for routing, [U] for keep).
        trainable: if False the logits never receive gradients (frozen controls).
    """

    def __init__(self, logits: torch.Tensor, trainable: bool = True):
        super().__init__()
        self.logits = nn.Parameter(logits.clone().float())
        self.logits._gate = True                          # own LR multiplier, no weight decay
        self.register_buffer("hard_mask", torch.ones_like(self.logits))
        self.mode = "dense"
        self.temperature = 1.0
        self.hardened = False
        self.trainable = trainable

    # ---- subclass API -------------------------------------------------------------------------
    def prob_open(self) -> torch.Tensor:
        """Expected gate value used by regularizers/diagnostics, same shape as logits."""
        raise NotImplementedError

    def hard_value(self) -> torch.Tensor:
        """Deterministic 0/1 value from the current logits (used at hardening)."""
        raise NotImplementedError

    def soft_value(self) -> torch.Tensor:
        """Differentiable value in stochastic mode (training)."""
        raise NotImplementedError

    def eval_value(self) -> torch.Tensor:
        """Value in stochastic mode at evaluation time (deterministic)."""
        return self.hard_value()

    # ---- common ---------------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        assert mode in MODES
        # A hardened gate stays hard except during the dense warm-up, which overrides everything.
        self.mode = "hard" if (self.hardened and mode != "dense") else mode

    @torch.no_grad()
    def harden(self, temperature: float | None = None) -> None:
        """Freeze the gate at its deterministic 0/1 value and stop logit gradients."""
        if temperature is not None:
            self.temperature = temperature
        self.hard_mask.copy_(self.hard_value())
        self.logits.requires_grad_(False)
        self.hardened = True
        self.mode = "hard"

    def forward(self) -> torch.Tensor:
        if self.mode == "dense":
            return torch.ones_like(self.logits)
        if self.mode == "hard":
            return self.hard_mask
        return self.soft_value() if self.training else self.eval_value()


class HardConcreteGate(Gate):
    """Independent hard-concrete (stretched, clamped binary-concrete) gates.

    Sample:  s = sigmoid((log u - log(1-u) + logit) / beta),  g = clamp(s * (zeta - gamma) + gamma, 0, 1)
    P(g > 0) = sigmoid(logit - beta * log(-gamma / zeta))      (differentiable L0 surrogate)
    The stretch lets samples be *exactly* 0 or 1 while remaining differentiable in between.
    """

    def prob_open(self):
        return torch.sigmoid(self.logits - self.temperature * math.log(-GAMMA / ZETA))

    def hard_value(self):
        return (self.prob_open() > 0.5).float()

    def soft_value(self):
        u = torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.logits) / self.temperature)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def eval_value(self):
        # Louizos et al.'s test-time estimator: the clamped, stretched mean of the concrete variable.
        return (torch.sigmoid(self.logits) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


class ExpertChoiceGate(Gate):
    """Expert-choice routing: every expert (column) keeps its top-k units (rows), straight-through.

    Logits [U, m]. The forward value is the hard top-k mask; gradients flow through a surrogate:

    * ``sigmoid`` — d mask / d logit = sigmoid'(logit): every (unit, expert) logit learns independently
                    whether that unit would help that expert.
    * ``softmax`` — surrogate k * softmax over units (per expert): units compete for an expert's k seats,
                    so raising one unit's score lowers the others'.

    ``noise`` > 0 adds Gumbel noise (scaled by the temperature) to the scores before the top-k during
    training, so units near the selection boundary are sometimes swapped in (exploration that anneals
    with the temperature). Load balance is exact by construction (every expert has k units). Units in
    the null (``eligible == 0``) can't be picked, so each expert has exactly min(k, #eligible) units.
    """

    def __init__(self, logits: torch.Tensor, k: int, trainable: bool = True, surrogate: str = "sigmoid",
                 noise: float = 0.0):
        super().__init__(logits, trainable)
        self.k = int(k)
        self.surrogate = surrogate
        self.noise = float(noise)
        self.register_buffer("eligible", torch.ones(logits.shape[0]))

    def prob_open(self):
        return torch.sigmoid(self.logits)

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.masked_fill(self.eligible.unsqueeze(1) <= 0, float("-inf"))
        k = int(min(self.k, int((self.eligible > 0).sum())))
        idx = scores.topk(k, dim=0).indices                              # [k, m]
        return torch.zeros_like(self.logits).scatter_(0, idx, 1.0)

    def hard_value(self):
        return self._topk_mask(self.logits.detach())

    def _surrogate(self) -> torch.Tensor:
        if self.surrogate == "sigmoid":
            return torch.sigmoid(self.logits)
        if not (self.eligible > 0).any():
            return self.logits * 0
        s = self.logits.masked_fill(self.eligible.unsqueeze(1) <= 0, float("-inf"))
        return self.k * torch.softmax(s, dim=0)

    def soft_value(self):
        scores = self.logits.detach()
        if self.noise > 0:
            u = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
            scores = scores + self.noise * self.temperature * -torch.log(-torch.log(u))
        p = self._surrogate()
        return self._topk_mask(scores) + p - p.detach()                  # straight-through estimator

    def eval_value(self):
        return self.hard_value()


class FixedNullGate(Gate):
    """Fixed-size null: drop exactly round(q * U) units with the lowest keep logits (straight-through).

    Logits [U]. No sparsity regularizer is needed because the null size is fixed.
    """

    def __init__(self, logits: torch.Tensor, q: float, trainable: bool = True):
        super().__init__(logits, trainable)
        self.n_drop = int(round(q * logits.shape[0]))

    def prob_open(self):
        return torch.sigmoid(self.logits)

    def hard_value(self):
        keep = torch.ones_like(self.logits)
        if self.n_drop:
            keep[self.logits.detach().topk(self.n_drop, largest=False).indices] = 0.0
        return keep

    def soft_value(self):
        p = self.prob_open()
        return self.hard_value() + p - p.detach()


class TopKGate(Gate):
    """Keep exactly k entries: the k largest logits, straight-through (a budget applied to the input).

    Logits [F]. Caution: an entry outside the top k is ineligible for every expert downstream, so it gets
    no gradient and the initial selection tends to stay frozen (see README, "Known caveats").
    """

    def __init__(self, logits: torch.Tensor, k: int, trainable: bool = True):
        super().__init__(logits, trainable)
        self.k = int(min(max(k, 1), logits.numel()))

    def prob_open(self):
        return torch.sigmoid(self.logits)

    def hard_value(self):
        return torch.zeros_like(self.logits).scatter_(0, self.logits.detach().topk(self.k).indices, 1.0)

    def soft_value(self):
        p = self.prob_open()
        return self.hard_value() + p - p.detach()                        # straight-through estimator

    def eval_value(self):
        return self.hard_value()


class ConcreteSelectorGate(Gate):
    """Keep exactly k of F entries with k concrete selector units (Abid et al., ICML 2019).

    Logits [k, F]: each unit is a softmax over all F entries. Gate learning uses Gumbel-softmax samples at
    the current temperature. The soft mask is s[f] = min(1, sum_units m_unit[f]), so *every* entry gets a
    gradient at every step (unlike a straight-through top-k, where unselected entries can get none).
    Hardening assigns units greedily to distinct entries (most confident unit first), which gives exactly k
    distinct entries even when two units prefer the same one.
    Forward returns the [F] mask; ``lab_mask`` holds the hard selection.
    """

    def __init__(self, n_features: int, k: int, gen: torch.Generator, trainable: bool = True):
        self.k = int(min(max(k, 1), n_features))
        super().__init__(0.01 * torch.randn(self.k, n_features, generator=gen), trainable)
        self.register_buffer("lab_mask", torch.ones(n_features))

    def _probs(self) -> torch.Tensor:
        return torch.softmax(self.logits / self.temperature, dim=1)                # [k, F]

    def prob_open(self):
        return self._probs().sum(0).clamp(max=1.0)                                  # [F]

    def hard_value(self):
        P = self._probs().detach().cpu().numpy()                                    # greedy loop: cheap on CPU
        taken = np.zeros(P.shape[1], dtype=bool)
        for u in np.argsort(-P.max(1), kind="stable"):                              # most confident unit first
            f = int(np.where(taken, -1.0, P[u]).argmax())                            # best entry not yet taken
            taken[f] = True
        return torch.from_numpy(taken).float().to(self.logits.device)

    def soft_value(self):
        m = Fn.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=1)  # [k, F]
        return m.sum(0).clamp(max=1.0)

    def eval_value(self):
        return self.prob_open()

    @torch.no_grad()
    def harden(self, temperature: float | None = None) -> None:
        if temperature is not None:
            self.temperature = temperature
        self.lab_mask.copy_(self.hard_value())
        self.logits.requires_grad_(False)
        self.hardened = True
        self.mode = "hard"

    def forward(self) -> torch.Tensor:
        if self.mode == "dense":
            return torch.ones_like(self.lab_mask)
        if self.mode == "hard":
            return self.lab_mask
        return self.soft_value() if self.training else self.eval_value()


# ------------------------------------------------------------------------------------------------
# Initialisers (return assignments / logits; no data access here — the correlation matrix is passed in)
# ------------------------------------------------------------------------------------------------
_SPECTRAL_CACHE: dict = {}


def spectral_groups(abs_corr: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Spectral embedding of the |corr| affinity followed by k-means (k = m_h). Returns labels [F] in [0, k).

    Why: features that co-vary are likely to belong together, which gives the gates a sensible start.
    Results are cached per (matrix, k, seed) because sweeps rebuild the same heads many times.
    """
    from sklearn.cluster import SpectralClustering

    key = (hashlib.sha1(np.ascontiguousarray(abs_corr)).hexdigest(), k, seed)
    if key in _SPECTRAL_CACHE:
        return _SPECTRAL_CACHE[key].copy()
    F = abs_corr.shape[0]
    if k == 1:
        labels = np.zeros(F, dtype=int)
    else:
        A = abs_corr + 1e-6                                              # keep the graph connected
        np.fill_diagonal(A, 1.0)
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed, assign_labels="kmeans",
                                n_init=10)
        labels = sc.fit_predict(A)
    _SPECTRAL_CACHE[key] = labels
    return labels.copy()


def logits_from_assignment(assign: list[list[int]] | np.ndarray, m: int, gen: torch.Generator, scale: float = 2.0,
                           noise: float = 0.1) -> torch.Tensor:
    """+scale for assigned experts, -scale elsewhere, plus N(0, noise) jitter. Returns [U, m].

    ``assign`` is either a label vector [U] (one expert per unit) or a list of expert lists.
    """
    U = len(assign)
    L = torch.full((U, m), -scale)
    for f, a in enumerate(assign):
        L[f, list(np.atleast_1d(a))] = scale
    return L + noise * torch.randn(U, m, generator=gen)


def random_assignment(U: int, m: int, r: float, gen: torch.Generator) -> list[list[int]]:
    """Each unit joins min(round(r), m) distinct random experts (at least one)."""
    n = max(1, min(int(round(r)), m))
    return [torch.randperm(m, generator=gen)[:n].tolist() for _ in range(U)]
