"""Unit tests. Run: python -m pytest tests  (or: python tests/test_mmoe.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmoe import MMoEEncoder, PhaseController, fit, model_config, train_config  # noqa: E402
from mmoe.gates import ConcreteSelectorGate, ExpertChoiceGate  # noqa: E402
from mmoe.prior import abs_corr_prior  # noqa: E402

F = 24


def enc(**kw) -> MMoEEncoder:
    kw.setdefault("gate_init", "random")
    kw.setdefault("experts_per_head", [2, 4])
    return MMoEEncoder(F, model_config(**kw), seed=0)


def batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, F, generator=g), (torch.rand(n, F, generator=g) > 0.3).float()


def test_shapes_and_variants():
    x, o = batch()
    for kw in [{}, {"use_obs_flags": False}, {"column_units": True}, {"null_mode": "global"},
               {"null_mode": "off"}, {"null_type": "fixed"}, {"input_budget": 10},
               {"input_budget": 10, "budget_type": "topk", "split_value_flag": True},
               {"ec_surrogate": "softmax", "ec_noise": 1.0}, {"capacity": [5, 3]}]:
        e = enc(**kw)
        for mode in ("dense", "stochastic"):
            e.set_mode(mode)
            assert e(x, o).shape == (8, 64), kw
        e.harden(0.1)
        assert e(x, o).shape == (8, 64), kw


def test_expert_choice_exact_capacity():
    e = enc(r=2)
    for h, (G, keep) in enumerate(e.hard_assignments()):
        m = e.heads[h].m
        k = math.ceil(2 * F / m)
        assert e.heads[h].gate.k == k
        assert (G.sum(0) == min(k, int(keep.sum()))).all()


def test_capacity_override():
    e = enc(capacity=3)
    for G, _ in e.hard_assignments():
        assert (G.sum(0) == 3).all()


def test_null_units_are_ineligible():
    e = enc(null_mode="global")
    with torch.no_grad():
        e.keep_gates[0].logits[:5] = -10.0                      # close units 0..4
    e.harden(0.1)
    for head, (G, keep) in zip(e.heads, e.hard_assignments()):
        assert keep[:5].sum() == 0 and G[:5].sum() == 0
        assert (G.sum(0) == min(head.gate.k, F - 5)).all()        # seats go to eligible units only


def test_closed_unit_has_no_influence():
    e = enc(null_mode="global").eval()
    with torch.no_grad():
        e.keep_gates[0].logits[3] = -10.0
    e.harden(0.1)
    x, o = batch()
    x2, o2 = x.clone(), o.clone()
    x2[:, 3] += 100.0
    o2[:, 3] = 1 - o2[:, 3]
    assert torch.allclose(e(x, o), e(x2, o2), atol=1e-5)


def test_unit_outside_routing_has_no_influence():
    e = enc(null_mode="off", capacity=2).eval()
    e.harden(0.1)
    unused = [u for u in range(F) if all(G[u].sum() == 0 for G, _ in e.hard_assignments())]
    assert unused, "with capacity 2 most units are unrouted"
    x, o = batch()
    x2 = x.clone()
    x2[:, unused[0]] += 100.0
    assert torch.allclose(e(x, o), e(x2, o), atol=1e-5)


def test_straight_through_gradients_reach_logits():
    for sur in ("sigmoid", "softmax"):
        e = enc(ec_surrogate=sur).train()
        e.set_mode("stochastic")
        x, o = batch()
        e(x, o).pow(2).sum().backward()
        assert e.heads[0].gate.logits.grad is not None
        assert e.heads[0].gate.logits.grad.abs().sum() > 0


def test_hardening_freezes():
    e = enc()
    e.harden(0.1)
    assert all(g.hardened and not g.logits.requires_grad for g in e.gates())
    before = [G.clone() for G, _ in e.hard_assignments()]
    with torch.no_grad():
        e.heads[0].gate.logits.add_(torch.randn_like(e.heads[0].gate.logits) * 10)
    after = [G for G, _ in e.hard_assignments()]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_concrete_budget_exact_k():
    g = ConcreteSelectorGate(F, 7, torch.Generator().manual_seed(0))
    g.harden(0.1)
    assert int(g.lab_mask.sum()) == 7
    e = enc(input_budget=7, null_mode="off")
    e.harden(0.1)
    for h, (G, keep) in enumerate(e.hard_assignments()):
        assert int(keep.sum()) == 7
        assert e.heads[h].gate.k == math.ceil(2 * 7 / e.heads[h].m)


def test_split_value_flag_no_orphans():
    e = enc(input_budget=8, split_value_flag=True, null_mode="off")
    with torch.no_grad():
        e.input_gate.cols.logits.fill_(-10.0)                    # every column wants to close
    e.harden(0.1)
    s = e.input_gate.summary()
    assert s["in_kept"] == 8 and s["in_both"] + s["in_value_only"] + s["in_flag_only"] == 8


def test_frozen_control():
    e = enc(train_gates=False)
    assert all(g.hardened for g in e.gates())


def test_effective_params_drop_after_capacity_cut():
    dense = enc(capacity=F)
    sparse = enc(capacity=2)
    dense.harden(0.1); sparse.harden(0.1)
    assert sparse.effective_params() < dense.effective_params()


def test_config_validation():
    with pytest.raises(ValueError):
        model_config(null_mode=False)                           # YAML bare off
    with pytest.raises(ValueError):
        model_config(column_units=True, use_obs_flags=False)
    with pytest.raises(KeyError):
        model_config(not_a_param=1)


def test_phase_controller():
    p = model_config(phases={"warmup_end": 2, "harden_start": 6, "lambda_ramp_end": 4}).phases
    c = PhaseController(p, {"l_null": 1.0})
    assert [c.phase(e) for e in range(8)] == [0, 0, 1, 1, 1, 1, 2, 2]
    assert c.lambda_scale(1) == 0 and c.lambda_scale(3) == 0.5 and c.lambda_scale(5) == 1 and c.lambda_scale(6) == 0
    assert c.temperature(2) == pytest.approx(1.0) and c.temperature(6) == pytest.approx(0.1)
    assert c.lr_mults(3)["gates"] == 10.0 and c.lr_mults(6)["gates"] == 0.0


def test_prior():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = np.stack([a, a * 2 + 0.01 * rng.normal(size=500), rng.normal(size=500)], 1)
    C = abs_corr_prior(X, device="cpu")
    assert C[0, 1] > 0.99 and C[0, 2] < 0.2 and np.allclose(np.diag(C), 1)


def test_fit_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, F)).astype(np.float32)
    O = (rng.random((256, F)) > 0.2).astype(np.float32)
    m = model_config(experts_per_head=[2, 4], phases={"warmup_end": 1, "lambda_ramp_end": 2, "harden_start": 3})
    t = train_config(epochs=4, batch_size=64, warmup_epochs=0.5, bf16=False)
    res = fit(m, t, {"train": (X[:200], O[:200]), "val": (X[200:], O[200:])}, device="cpu", verbose=False)
    assert res.best_epoch >= 3
    assert all(g.hardened for g in res.model.encoder.gates())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
