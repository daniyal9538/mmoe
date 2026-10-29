"""Train an expert-choice M-MoE autoencoder on in-memory tensors.

    from mmoe import fit, model_config, train_config
    res = fit(model_config(), train_config(epochs=100), {"train": (X, O), "val": (Xv, Ov)})
    res.model, res.history, res.best_epoch

The recipe: AdamW, linear LR warm-up then cosine, gradient clipping, optional bf16 autocast. The phase
controller adds the gate phases (dense warm-up -> gate learning -> hardening). Regularizers are added to
the *training* loss only; checkpoint selection uses the val reconstruction loss without them, and only
hardened epochs are eligible, so the returned model is always a real 0/1-gated sparse model.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, TrainConfig
from .diagnostics import diagnostics_row
from .encoder import MMoEEncoder
from .model import AutoEncoder, Decoder, count_params, lr_lambda, param_groups, recon_loss
from .prior import abs_corr_prior
from .schedule import PhaseController


@dataclass
class FitResult:
    model: AutoEncoder
    history: pd.DataFrame
    best_epoch: int
    best_val: float
    params: dict


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _tensors(split, device):
    x, o = split if isinstance(split, (tuple, list)) else (split, None)
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32).to(device)
    if o is not None:
        o = torch.as_tensor(np.asarray(o) if not torch.is_tensor(o) else o, dtype=torch.float32).to(device)
    return x, o


@torch.no_grad()
def feature_loss_weights(x_obs: torch.Tensor | None, F: int, floor: float, device) -> torch.Tensor:
    """[F] = 1 / max(train coverage, floor), scaled so the mean weight per observed train value is 1."""
    cov = torch.ones(F, device=device) if x_obs is None else x_obs.float().mean(0)
    w = 1.0 / cov.clamp_min(floor)
    return w * cov.sum() / (cov * w).sum()


@torch.no_grad()
def evaluate_recon(model, x, o, tcfg: TrainConfig, feature_weight=None, batch: int = 2048) -> dict:
    """Dataset-level reconstruction loss (Huber mean over observed + w * BCE mean over all), no regularizers."""
    model.eval()
    hs = no = bs = ne = 0.0
    amp = tcfg.bf16 and x.is_cuda
    for i in range(0, len(x), batch):
        xv, xo = x[i:i + batch], (o[i:i + batch] if o is not None else None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            vp, ol, _ = model(xv, xo)
        h, n, bc, m = recon_loss(vp, ol, xv, xo, tcfg.huber_delta, reduce=False, feature_weight=feature_weight)
        hs += h.item(); no += n.item(); bs += bc.item(); ne += m.item()
    huber, bce = hs / max(no, 1e-9), bs / ne
    return {"recon": huber + tcfg.obs_bce_weight * bce, "huber": huber, "bce": bce}


@torch.no_grad()
def encode(model: AutoEncoder, x, o=None, batch: int = 4096, heads: bool = False):
    """Embeddings z [N, d_z] (numpy); with ``heads=True`` also the per-head expert outputs."""
    model.eval()
    dev = next(model.parameters()).device
    x, o = _tensors((x, o), dev)
    zs, hs = [], []
    for i in range(0, len(x), batch):
        z = model.encoder(x[i:i + batch], o[i:i + batch] if o is not None else None)
        zs.append(z.float().cpu())
        if heads:
            hs.append([t.float().cpu() for t in model.encoder.last_head_outputs])
    z = torch.cat(zs).numpy()
    if not heads:
        return z
    return z, [torch.cat([b[h] for b in hs]).numpy() for h in range(len(hs[0]))]


def build_model(F: int, mcfg: ModelConfig, tcfg: TrainConfig, abs_corr=None) -> AutoEncoder:
    encoder = MMoEEncoder(F, mcfg, abs_corr=abs_corr, seed=tcfg.seed)
    decoder = Decoder(mcfg.latent_dim, F, tcfg.decoder_hidden, tcfg.decoder_dropout, predict_obs=mcfg.use_obs_flags)
    return AutoEncoder(encoder, decoder)


def fit(mcfg: ModelConfig, tcfg: TrainConfig, data: dict, abs_corr: np.ndarray | None = None,
        device: str | None = None, verbose: bool = True, log_every: int = 1) -> FitResult:
    """Train and return the best hardened checkpoint.

    Args:
        data: {'train': (x_val, x_obs | None), 'val': (x_val, x_obs | None)}, arrays [N, F].
        abs_corr: [F, F] prior; computed from the train split when needed and not given.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(tcfg.seed)
    xt, ot = _tensors(data["train"], device)
    xv, ov = _tensors(data["val"], device)
    if not mcfg.use_obs_flags:
        ot = ov = None
    F = xt.shape[1]
    if mcfg.gate_init == "prior_corr" and abs_corr is None:
        abs_corr = abs_corr_prior(xt, ot, mcfg.prior_min_coobs, device)

    model = build_model(F, mcfg, tcfg, abs_corr).to(device)
    encoder = model.encoder
    params = {"encoder_params": count_params(encoder), "total_params": count_params(model),
              "gate_logits": sum(g.logits.numel() for g in encoder.gates()),
              "encoder_effective_params_init": encoder.effective_params()}

    opt = torch.optim.AdamW(param_groups(model, tcfg.weight_decay), lr=tcfg.lr, betas=tuple(tcfg.betas))
    lambdas = {"l_mem": mcfg.lambda_mem, "l_null": mcfg.lambda_null}
    if mcfg.split_value_flag:
        lambdas["l_col"] = mcfg.lambda_col
    ctrl = PhaseController(mcfg.phases, lambdas, mcfg.train_gates)
    fw = feature_loss_weights(ot, F, tcfg.feature_loss_floor, device) if tcfg.feature_loss_floor else None

    E, bsz = tcfg.epochs, tcfg.batch_size
    steps = (len(xt) + bsz - 1) // bsz
    gen = torch.Generator(device=device).manual_seed(tcfg.seed)
    first_eligible = mcfg.phases.harden_start
    amp = tcfg.bf16 and device.startswith("cuda")
    best, best_epoch, best_state, log = float("inf"), -1, None, []
    t0 = time.time()

    for epoch in range(E):
        events = ctrl.on_epoch_start(epoch, encoder)
        mults = ctrl.lr_mults(epoch)
        model.train()
        perm = torch.randperm(len(xt), generator=gen, device=device)
        acc = {"loss": 0.0, "recon": 0.0, **{k: 0.0 for k in lambdas}}
        te = time.time()
        for s in range(steps):
            ef = epoch + s / steps
            base = tcfg.lr * lr_lambda(epoch + (s + 1) / steps, tcfg.warmup_epochs, E)
            for g in opt.param_groups:
                g["lr"] = base * mults[g["name"]]
            ctrl.on_step(ef, encoder)
            b = perm[s * bsz:(s + 1) * bsz]
            xb, ob = xt[b], (ot[b] if ot is not None else None)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                vp, ol, _ = model(xb, ob)
            recon, _, _ = recon_loss(vp, ol, xb, ob, tcfg.huber_delta, tcfg.obs_bce_weight, feature_weight=fw)
            loss = recon
            w = ctrl.weights(ef)
            if any(v > 0 for v in w.values()):
                regs = encoder.regularizers()
                loss = loss + sum(w[k] * regs[k] for k in w)
                for k in w:
                    acc[k] += regs[k].item() / steps
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            acc["loss"] += loss.item() / steps
            acc["recon"] += recon.item() / steps
        train_time = time.time() - te

        v = evaluate_recon(model, xv, ov, tcfg, fw)
        row = {"epoch": epoch, "phase": events["phase"], "lr": base, "temperature": encoder.gates()[0].temperature,
               "train_loss": acc["loss"], "train_recon": acc["recon"], "val_recon": v["recon"],
               "val_huber": v["huber"], "val_bce": v["bce"], "epoch_time_s": train_time,
               **{f"train_{k}": acc[k] for k in lambdas}, **diagnostics_row(encoder)}
        if epoch >= first_eligible and v["recon"] < best:
            best, best_epoch = v["recon"], epoch
            best_state = ({k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                          [(g.mode, g.hardened, g.temperature) for g in encoder.gates()])
        row["best_val_recon"] = best
        log.append(row)
        if verbose and (epoch % log_every == 0 or epoch == E - 1):
            print(f"ep {epoch:3d} ph {events['phase']} train {acc['recon']:.4f} val {v['recon']:.4f} "
                  f"best {best:.4f}@{best_epoch} ({train_time:.1f}s)", flush=True)
        if epoch >= first_eligible and epoch - max(best_epoch, first_eligible - 1) >= tcfg.patience:
            break

    if best_state is None:
        raise RuntimeError(f"no hardened epoch was reached: epochs={E} <= harden_start={first_eligible}")
    model.load_state_dict(best_state[0])
    for g, (mode, hardened, temp) in zip(encoder.gates(), best_state[1]):   # python-side gate state
        g.mode, g.hardened, g.temperature = mode, hardened, temp
        g.logits.requires_grad_(not hardened)
    model.eval()
    params.update(encoder_effective_params_final=encoder.effective_params(), best_epoch=best_epoch,
                  best_val_recon=best, epochs_run=len(log), train_seconds=time.time() - t0)
    return FitResult(model, pd.DataFrame(log), best_epoch, best, params)
