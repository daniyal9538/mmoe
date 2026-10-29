"""Diagnostics: what the gates are doing, per head. Logged every epoch from gate learning onward.

All counts use the *hard* assignments (what the model would use if hardened now):

* n_kept / n_null     units each head keeps / ignores
* expert sizes        units per expert (= k under expert choice, unless fewer units are eligible)
* mean_memberships    mean number of experts reading a kept unit (overlap between experts)
* n_orphan            kept units that no expert picked: in the input but unused (an expert-choice failure
                      mode — the keep gate says "use it", but no expert ranks it in its top k)
* changed_vs_init     fraction of (unit, expert) memberships that differ from the initial assignment
* entropy_keep        mean binary entropy of P(keep > 0): how undecided the null still is (-> 0)
* routing_margin      mean over experts of (k-th best - (k+1)-th best logit): how settled the top-k is
"""
from __future__ import annotations

import json

import numpy as np
import torch


def _entropy(p: torch.Tensor) -> float:
    p = p.clamp(1e-6, 1 - 1e-6)
    return float((-(p * p.log() + (1 - p) * (1 - p).log())).mean())


@torch.no_grad()
def head_diagnostics(encoder) -> list[dict]:
    """One dict per head with the quantities listed in the module docstring."""
    out = []
    for h, (head, (G, keep)) in enumerate(zip(encoder.heads, encoder.hard_assignments())):
        kept = keep > 0
        mem = G.sum(1)
        init = getattr(encoder, f"init_mask{h}")
        L = head.gate.logits.masked_fill(~kept.unsqueeze(1), float("-inf"))
        k = min(head.gate.k, int(kept.sum()))
        if 0 < k < int(kept.sum()):
            top = L.topk(k + 1, dim=0).values
            margin = float((top[k - 1] - top[k]).mean())
        else:
            margin = float("nan")
        out.append({
            "head": h,
            "m": head.m,
            "k": head.gate.k,
            "sizes": [int(s) for s in G.sum(0)],
            "n_kept": int(kept.sum()),
            "n_null": int((~kept).sum()),
            "mean_memberships": float(mem[kept].mean()) if kept.any() else 0.0,
            "n_orphan": int((kept & (mem == 0)).sum()),
            "changed_vs_init": float((G != init * keep.unsqueeze(1)).float().mean()),
            "entropy_keep": _entropy(encoder.keep_probs(h)),
            "routing_margin": margin,
        })
    return out


def diagnostics_row(encoder) -> dict:
    """Flatten ``head_diagnostics`` into columns ``h{i}_<name>`` (sizes as a JSON list)."""
    row = {}
    for d in head_diagnostics(encoder):
        h = d["head"]
        for k, v in d.items():
            if k in ("head", "m"):
                continue
            row[f"h{h}_{k}"] = json.dumps(v) if isinstance(v, list) else v
    if encoder.input_gate is not None:
        row.update(encoder.input_gate.summary())
    return row


@torch.no_grad()
def assignments(encoder) -> list[dict]:
    """Final hard assignments as numpy arrays: per head {'G': [U, m] 0/1, 'keep': [U] 0/1}."""
    return [{"G": G.cpu().numpy().astype(np.int8), "keep": keep.cpu().numpy().astype(np.int8)}
            for G, keep in encoder.hard_assignments()]
