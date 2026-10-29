"""Everything around the encoder: decoder, reconstruction loss, autoencoder wrapper, optimiser groups.

Parameter tags (read by ``param_groups``):
  ``p._gate = True``   -> gate logits: own LR multiplier, no weight decay
Biases and normalisation weights (all 1-D tensors) never get weight decay.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


def mlp_block(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    """Linear -> LayerNorm -> GELU -> Dropout."""
    return nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout))


class Decoder(nn.Module):
    """z -> hidden... -> (F value predictions, F observation logits or None).

    Shapes: z [B, d_z] -> (val_pred [B, F], obs_logit [B, F] | None)
    """

    def __init__(self, latent_dim: int, n_features: int, hidden=(256, 512), dropout: float = 0.1,
                 predict_obs: bool = True):
        super().__init__()
        dims = [latent_dim, *hidden]
        self.body = nn.Sequential(*[mlp_block(a, b, dropout) for a, b in zip(dims[:-1], dims[1:])])
        self.val_head = nn.Linear(dims[-1], n_features)
        self.obs_head = nn.Linear(dims[-1], n_features) if predict_obs else None

    def forward(self, z: torch.Tensor):
        h = self.body(z)
        return self.val_head(h), (self.obs_head(h) if self.obs_head is not None else None)


def recon_loss(val_pred, obs_logit, x_val, x_obs, delta: float = 1.0, bce_w: float = 0.2, reduce: bool = True,
               feature_weight: torch.Tensor | None = None):
    """Huber on observed entries (mean over observed) + bce_w * BCE(obs logits, obs) (mean over all).

    ``x_obs`` None means everything is observed (no BCE term). ``feature_weight`` [F] (optional) weights
    each observed value by its feature; with weights proportional to 1 / max(coverage, floor), every
    feature above the floor counts equally however often it is measured.

    With ``reduce=False`` returns the sums and counts so dataset-level means can be accumulated over
    batches: (huber_sum, n_observed (weighted), bce_sum, n_entries).
    """
    val_pred = val_pred.float()
    o = torch.ones_like(x_val, dtype=torch.float32) if x_obs is None else x_obs.float()
    w = o if feature_weight is None else o * feature_weight.float()
    hub = Fn.huber_loss(val_pred, x_val.float(), reduction="none", delta=delta) * w
    if obs_logit is not None and x_obs is not None:
        bce = Fn.binary_cross_entropy_with_logits(obs_logit.float(), o, reduction="none")
        bce_sum, n_entries = bce.sum(), float(o.numel())
    else:
        bce_sum, n_entries = hub.new_zeros(()), 1.0
    if not reduce:
        return hub.sum(), w.sum(), bce_sum, torch.tensor(n_entries, device=o.device)
    huber = hub.sum() / w.sum().clamp_min(1e-6)
    bce_m = bce_sum / n_entries
    return huber + bce_w * bce_m, huber, bce_m


class AutoEncoder(nn.Module):
    """Encoder + decoder. ``encoder(x_val, x_obs) -> z [B, d_z]``."""

    def __init__(self, encoder: nn.Module, decoder: Decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, x_val, x_obs=None):
        z = self.encoder(x_val, x_obs)
        val_pred, obs_logit = self.decoder(z)
        return val_pred, obs_logit, z


def count_params(module: nn.Module) -> int:
    """Number of parameter entries (dense count, including masked-out entries and gate logits)."""
    return sum(p.numel() for p in module.parameters())


def param_groups(model: nn.Module, weight_decay: float):
    """Split parameters into {decay, no_decay, gates}. Group ``name`` keys are used by the LR controller."""
    decay, nodecay, gates = [], [], []
    for p in model.parameters():
        if getattr(p, "_gate", False):
            gates.append(p)
        elif p.ndim < 2 or getattr(p, "_no_wd", False):
            nodecay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "name": "decay"},
        {"params": nodecay, "weight_decay": 0.0, "name": "no_decay"},
    ]
    if gates:
        groups.append({"params": gates, "weight_decay": 0.0, "name": "gates"})
    return groups


def lr_lambda(epoch_float: float, warmup_epochs: float, total_epochs: float) -> float:
    """Linear warm-up to 1 over ``warmup_epochs`` then cosine decay to 0 at ``total_epochs``."""
    if warmup_epochs > 0 and epoch_float < warmup_epochs:
        return epoch_float / warmup_epochs
    t = min(1.0, (epoch_float - warmup_epochs) / max(1e-8, total_epochs - warmup_epochs))
    return 0.5 * (1 + math.cos(math.pi * t))
