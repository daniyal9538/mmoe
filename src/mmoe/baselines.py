"""Reference encoders for comparisons with M-MoE: dense MLP, CNN and ViT, trained with the same recipe.

Every encoder here has the M-MoE signature ``encoder(x_val, x_obs=None) -> z [B, d_z]`` and plugs into the
same ``AutoEncoder`` (same decoder, Huber loss, AdamW, warm-up + cosine, bf16). Only the encoder differs.

    from mmoe.baselines import DenseEncoder, fit_encoder
    res = fit_encoder(lambda: DenseEncoder(784, (512, 256)), train_config(epochs=20), {"train": (X, None), "val": (Xv, None)})

The CNN and ViT read x_val [B, side*side] as one-channel images; they are only meaningful for image data.
"""
from __future__ import annotations

import copy
import time
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn

from .config import TrainConfig
from .model import AutoEncoder, Decoder, count_params, lr_lambda, mlp_block, param_groups, recon_loss
from .train import FitResult, _tensors, evaluate_recon, seed_everything


class DenseEncoder(nn.Module):
    """Plain MLP: x -> [Linear -> LayerNorm -> GELU -> Dropout]* -> Linear -> z.

    Shapes: x_val [B, F] -> z [B, d_z]
    """

    def __init__(self, n_features: int, hidden=(512, 256), latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        dims = [n_features, *hidden]
        self.body = nn.Sequential(*[mlp_block(a, b, dropout) for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], latent_dim)

    def forward(self, x_val: torch.Tensor, x_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.body(x_val))


class ConvEncoder(nn.Module):
    """Conv blocks (Conv -> BatchNorm -> GELU), one per entry of ``channels`` -> flatten -> Linear -> z.

    Default: three 3x3 blocks with strides 1, 2, 2. ``channels=(7,), strides=(2,), kernel=5`` is a single
    conv layer, i.e. one hidden layer like M-MoE.
    Shapes: x_val [B, side*side] -> z [B, d_z]
    """

    def __init__(self, latent_dim: int = 64, channels=(32, 64, 64), strides=(1, 2, 2), kernel: int = 3,
                 side: int = 28, dropout: float = 0.1):
        super().__init__()
        assert len(channels) == len(strides)
        self.side = side
        layers, c_in, out = [], 1, side
        for c, s in zip(channels, strides):
            layers += [nn.Conv2d(c_in, c, kernel, s, kernel // 2), nn.BatchNorm2d(c), nn.GELU()]
            c_in, out = c, (out + 2 * (kernel // 2) - kernel) // s + 1
        self.body = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(c_in * out * out, latent_dim)

    def forward(self, x_val: torch.Tensor, x_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.body(x_val.view(-1, 1, self.side, self.side)))


class ViTEncoder(nn.Module):
    """Vision transformer: patch embedding + CLS token + learned positions -> pre-norm blocks -> CLS -> z.

    Shapes: x_val [B, side*side] -> z [B, d_z]
    """

    def __init__(self, latent_dim: int = 64, side: int = 28, patch: int = 4, dim: int = 64, depth: int = 4,
                 heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.side = side
        n_tokens = (side // patch) ** 2
        self.embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tokens + 1, dim))
        self.cls._no_wd = self.pos._no_wd = True
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, latent_dim)

    def forward(self, x_val: torch.Tensor, x_obs: torch.Tensor | None = None) -> torch.Tensor:
        t = self.embed(x_val.view(-1, 1, self.side, self.side)).flatten(2).transpose(1, 2)   # [B, N, dim]
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        return self.head(self.norm(self.blocks(t))[:, 0])


def fit_encoder(make_encoder: Callable[[], nn.Module], tcfg: TrainConfig, data: dict, latent_dim: int = 64,
                device: str | None = None, verbose: bool = False) -> FitResult:
    """Train ``make_encoder()`` + the standard decoder with the M-MoE recipe minus the gate phases.

    Same optimiser, LR schedule, clipping, bf16 and decoder as ``train.fit``. Every epoch is eligible for
    checkpoint selection (there is nothing to harden). The encoder is built after seeding, so weight init
    depends on ``tcfg.seed``.

    Args:
        data: {'train': (x_val, None), 'val': (x_val, None)}, arrays [N, F] (no observed flags).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(tcfg.seed)
    xt, _ = _tensors(data["train"], device)
    xv, _ = _tensors(data["val"], device)
    F = xt.shape[1]
    model = AutoEncoder(make_encoder(), Decoder(latent_dim, F, tcfg.decoder_hidden, tcfg.decoder_dropout,
                                                predict_obs=False)).to(device)
    n_enc = count_params(model.encoder)
    params = {"encoder_params": n_enc, "total_params": count_params(model), "gate_logits": 0,
              "encoder_effective_params_init": n_enc}

    opt = torch.optim.AdamW(param_groups(model, tcfg.weight_decay), lr=tcfg.lr, betas=tuple(tcfg.betas))
    E, bsz = tcfg.epochs, tcfg.batch_size
    steps = (len(xt) + bsz - 1) // bsz
    gen = torch.Generator(device=device).manual_seed(tcfg.seed)
    amp = tcfg.bf16 and device.startswith("cuda")
    best, best_epoch, best_state, log = float("inf"), -1, None, []
    t0 = time.time()

    for epoch in range(E):
        model.train()
        perm = torch.randperm(len(xt), generator=gen, device=device)
        tr = 0.0
        te = time.time()
        for s in range(steps):
            lr = tcfg.lr * lr_lambda(epoch + (s + 1) / steps, tcfg.warmup_epochs, E)
            for g in opt.param_groups:
                g["lr"] = lr
            xb = xt[perm[s * bsz:(s + 1) * bsz]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                vp, ol, _ = model(xb)
            loss, _, _ = recon_loss(vp, ol, xb, None, tcfg.huber_delta, tcfg.obs_bce_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            tr += loss.item() / steps
        train_time = time.time() - te

        v = evaluate_recon(model, xv, None, tcfg)
        if v["recon"] < best:
            best, best_epoch = v["recon"], epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        log.append({"epoch": epoch, "lr": lr, "train_recon": tr, "val_recon": v["recon"],
                    "epoch_time_s": train_time, "best_val_recon": best, "wall_s": time.time() - t0})
        if verbose:
            print(f"ep {epoch:3d} train {tr:.4f} val {v['recon']:.4f} best {best:.4f}@{best_epoch} "
                  f"({train_time:.1f}s)", flush=True)
        if epoch - best_epoch >= tcfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    params.update(encoder_effective_params_final=n_enc, best_epoch=best_epoch, best_val_recon=best,
                  epochs_run=len(log), train_seconds=time.time() - t0)
    return FitResult(model, pd.DataFrame(log), best_epoch, best, params)


@torch.no_grad()
def encoder_flops(encoder: nn.Module, x: torch.Tensor) -> float:
    """FLOPs per sample of one encoder forward pass as executed (matmuls, convs, attention).

    Counted on a copy in train mode, so PyTorch's fused inference path for transformers (which the
    counter cannot see) is not taken. M-MoE counts its dense masked matmul, i.e. what actually runs.
    """
    from torch.utils.flop_counter import FlopCounterMode
    enc = copy.deepcopy(encoder).train()
    with FlopCounterMode(display=False) as fc:
        enc(x)
    return fc.get_total_flops() / len(x)


@torch.no_grad()
def encoder_throughput(encoder: nn.Module, x: torch.Tensor, reps: int = 20, bf16: bool = True) -> float:
    """Encoder inference speed in samples per second (eval mode, one batch ``x``, after a warm-up)."""
    encoder.eval()
    amp = bf16 and x.is_cuda
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            encoder(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            encoder(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return reps * len(x) / (time.perf_counter() - t)
