"""All hyperparameters in one place, as two dataclasses: ``ModelConfig`` (the encoder) and ``TrainConfig``.

Every field is documented in the README (section "Hyperparameters") with what it controls and what it
affects. ``load_config(path)`` reads a YAML file with ``model:`` and ``train:`` sections; missing keys keep
their defaults, unknown keys raise (a typo should never silently fall back to a default).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PhaseConfig:
    """Training phases of the gates (epochs are absolute; see ``schedule.py``)."""
    warmup_end: int = 10            # [0, warmup_end): dense warm-up, every gate forced to 1
    harden_start: int = 60          # [warmup_end, harden_start): gate learning; from harden_start: hard 0/1 gates
    lambda_ramp_end: int = 35       # regularizer weights ramp 0 -> target over [warmup_end, lambda_ramp_end)
    temp_start: float = 1.0         # gate temperature at warmup_end
    temp_end: float = 0.1           # gate temperature at harden_start (also used for hardening)
    gate_lr_mult: float = 10.0      # gate-logit LR = gate_lr_mult x scheduled LR during gate learning
    finetune_lr_mult: float = 0.3   # weight LR multiplier after hardening


@dataclass
class ModelConfig:
    """Expert-choice M-MoE encoder."""
    # ---- architecture ----
    experts_per_head: list[int] = field(default_factory=lambda: [4, 16, 64])
    expert_hidden: int = 16         # u: hidden units per expert
    expert_out: int = 2             # d_e: output dimension per expert
    latent_dim: int = 64            # d_z: size of the embedding z
    dropout: float = 0.1            # on each expert's hidden layer
    use_obs_flags: bool = True      # input is [value, was-observed flag] per feature (False: value only)
    column_units: bool = False      # value and flag of a feature are separate routing units (needs use_obs_flags)
    # ---- expert-choice routing ----
    r: float = 2.0                  # capacity factor: each expert reads k = ceil(r * n_units / m) units
    capacity: int | list[int] | None = None   # fixed k per expert (int, or one per head); overrides r
    ec_surrogate: str = "sigmoid"   # gradient path of the straight-through top-k: sigmoid | softmax
    ec_noise: float = 0.0           # Gumbel noise (x temperature) on routing scores during gate learning
    # ---- null (keep) gates ----
    null_mode: str = "per_head"     # per_head | global | off
    null_type: str = "learned"      # learned (hard-concrete + L0 price) | fixed (drop exactly null_q)
    null_q: float = 0.25            # fraction of units dropped by the fixed null
    keep_init: float = 3.0          # initial keep logit (+3: P(keep) ~ 0.95, every unit starts in)
    # ---- initialisation ----
    gate_init: str = "prior_corr"   # prior_corr (spectral clustering of |corr|) | random
    init_scale: float = 2.0         # routing logit +scale for initially assigned experts, -scale elsewhere
    init_noise: float = 0.1         # N(0, init_noise) jitter on initial routing logits
    prior_min_coobs: int = 50       # pairs co-observed fewer times get |corr| = 0 in the prior
    train_gates: bool = True        # False: assignments frozen at init (control)
    # ---- global input budget (feature selection in front of all heads) ----
    input_budget: int | None = None  # K: keep exactly K features in total (None: no budget)
    budget_type: str = "concrete"   # concrete (Abid et al. 2019) | topk (straight-through; see README caveat)
    split_value_flag: bool = False  # with a budget: separate keep gates for each kept feature's value / flag
    # ---- regularizer weights (ramped by the schedule) ----
    lambda_mem: float = 1e-2        # membership cap on sigmoid(routing logits) (weak shrinkage under EC)
    lambda_null: float = 1e-3       # L0 price on keeping a unit (learned null only)
    lambda_col: float = 1e-3        # L0 price on open value / flag columns (split_value_flag only)
    phases: PhaseConfig = field(default_factory=PhaseConfig)

    def validate(self) -> None:
        def one_of(name, allowed):
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {v!r}"
                                 + (" (quote 'off' in YAML: bare off is boolean False)" if v is False else ""))
        one_of("null_mode", ("per_head", "global", "off"))
        one_of("null_type", ("learned", "fixed"))
        one_of("gate_init", ("prior_corr", "random"))
        one_of("budget_type", ("concrete", "topk"))
        one_of("ec_surrogate", ("sigmoid", "softmax"))
        if not self.experts_per_head or min(self.experts_per_head) < 1:
            raise ValueError("experts_per_head needs at least one head with >= 1 expert")
        if self.column_units and not self.use_obs_flags:
            raise ValueError("column_units needs use_obs_flags=True")
        if self.split_value_flag and (not self.use_obs_flags or self.input_budget is None):
            raise ValueError("split_value_flag needs use_obs_flags=True and an input_budget")
        if self.column_units and self.input_budget is not None:
            raise ValueError("column_units and input_budget cannot be combined")
        if isinstance(self.capacity, list) and len(self.capacity) != len(self.experts_per_head):
            raise ValueError("capacity list needs one entry per head")
        if self.r <= 0:
            raise ValueError("r must be > 0")
        p = self.phases
        if not (0 <= p.warmup_end <= p.lambda_ramp_end <= p.harden_start):
            raise ValueError("phases need warmup_end <= lambda_ramp_end <= harden_start")


@dataclass
class TrainConfig:
    """Shared autoencoder recipe (decoder, loss, optimiser, schedule)."""
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4      # not on biases, norms or gate logits
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    warmup_epochs: float = 5        # linear LR warm-up, then cosine to 0 at `epochs`
    grad_clip: float = 1.0
    decoder_hidden: list[int] = field(default_factory=lambda: [256, 512])
    decoder_dropout: float = 0.1
    huber_delta: float = 1.0        # value loss: Huber on observed entries
    obs_bce_weight: float = 0.2     # weight of the BCE on the observed flags (use_obs_flags only)
    feature_loss_floor: float | None = None  # per-feature loss weight 1 / max(coverage, floor); None: unweighted
    patience: int = 10              # early stopping, counted only over hardened epochs
    bf16: bool = True               # autocast on CUDA
    seed: int = 0


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    d = dict(d)
    if cls is ModelConfig and "phases" in d:
        d["phases"] = _build(PhaseConfig, d["phases"]) if isinstance(d["phases"], dict) else d["phases"]
    return cls(**d)


def model_config(d: dict | None = None, **overrides) -> ModelConfig:
    """ModelConfig from a dict plus keyword overrides (``phases`` may be a partial dict)."""
    base = asdict(ModelConfig())
    merged = merge(base, d or {})
    merged = merge(merged, overrides)
    cfg = _build(ModelConfig, merged)
    cfg.validate()
    return cfg


def train_config(d: dict | None = None, **overrides) -> TrainConfig:
    return _build(TrainConfig, merge(merge(asdict(TrainConfig()), d or {}), overrides))


def merge(base: dict, upd: dict) -> dict:
    """Recursive dict merge (``upd`` wins); returns a new dict."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    """Read a YAML file with optional ``model:`` and ``train:`` sections."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise KeyError(f"unknown top-level config sections: {sorted(unknown)}")
    return model_config(raw.get("model")), train_config(raw.get("train"))


def to_dict(cfg) -> dict:
    return asdict(cfg)
