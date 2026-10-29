"""Phase controller: the M-MoE training recipe as a function of (fractional) epoch.

Three phases (boundaries from ``ModelConfig.phases``; defaults are for a 100-epoch run):

====================  ================================================================================
Phase 0: [0, 10)      Dense warm-up. All routing and keep gates forced to 1, no regularizers.
                       Lets the experts learn useful weights before any unit is taken away.
Phase 1: [10, 60)     Gate learning. Stochastic gates, temperature annealed 1.0 -> 0.1 (hard-concrete
                       samples get closer to exact 0/1; expert-choice exploration noise shrinks). Lambdas
                       ramp 0 -> target over [10, 35). Gate logits train at 10x the scheduled LR (no
                       weight decay).
Phase 2: [60, end)    Harden + fine-tune. Gates become deterministic 0/1 and logits are frozen; weights
                       train at 0.3x the scheduled LR. Only these epochs are eligible checkpoints, so the
                       selected model is always a truly sparse, hard-gated model.
====================  ================================================================================

The shared cosine LR schedule is applied by the trainer; this controller only returns *multipliers*.
"""
from __future__ import annotations

from .config import PhaseConfig


class PhaseController:
    """Drive an ``MMoEEncoder`` through the phases.

    Args:
        phases: ``PhaseConfig``.
        lambdas: target {'l_mem': .., 'l_null': .., ['l_col': ..]}.
        train_gates: False for frozen-assignment controls (no gate learning).
    """

    def __init__(self, phases: PhaseConfig, lambdas: dict, train_gates: bool = True):
        self.p = phases
        self.lambdas = lambdas
        self.train_gates = train_gates
        self.hardened = False

    def phase(self, epoch: int) -> int:
        """0 = dense warm-up, 1 = gate learning, 2 = hardened fine-tuning."""
        if epoch < self.p.warmup_end:
            return 0
        return 1 if epoch < self.p.harden_start else 2

    def temperature(self, ef: float) -> float:
        """Linear anneal temp_start -> temp_end over [warmup_end, harden_start]."""
        a, b = self.p.warmup_end, self.p.harden_start
        t = min(1.0, max(0.0, (ef - a) / max(1e-8, b - a)))
        return self.p.temp_start + t * (self.p.temp_end - self.p.temp_start)

    def lambda_scale(self, ef: float) -> float:
        """0 during warm-up, linear ramp to 1 by lambda_ramp_end, 0 after hardening (gates are constants)."""
        if ef < self.p.warmup_end or ef >= self.p.harden_start or not self.train_gates:
            return 0.0
        a, b = self.p.warmup_end, self.p.lambda_ramp_end
        return min(1.0, (ef - a) / max(1e-8, b - a))

    def weights(self, ef: float) -> dict:
        """Current regularizer weights."""
        s = self.lambda_scale(ef)
        return {k: s * v for k, v in self.lambdas.items()}

    def lr_mults(self, epoch: int) -> dict:
        """LR multipliers per optimizer group name, on top of the shared warm-up/cosine schedule."""
        ph = self.phase(epoch)
        w = self.p.finetune_lr_mult if ph == 2 else 1.0
        g = self.p.gate_lr_mult if (ph == 1 and self.train_gates) else 0.0
        return {"decay": w, "no_decay": w, "gates": g}

    def on_epoch_start(self, epoch: int, encoder) -> dict:
        """Set gate modes and harden at the phase-2 boundary. Returns events for logging."""
        ph = self.phase(epoch)
        if ph == 0:
            encoder.set_mode("dense")
        elif ph == 1:
            encoder.set_mode("stochastic")                                 # hardened (frozen) gates stay hard
        else:
            if not self.hardened:
                encoder.harden(self.p.temp_end)
                self.hardened = True
            encoder.set_mode("hard")
        return {"phase": ph}

    def on_step(self, ef: float, encoder) -> None:
        """Per-step temperature update."""
        encoder.set_temperature(self.temperature(ef))
