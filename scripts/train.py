"""Train one model from a config and save everything needed to analyse it.

Usage:
    python scripts/train.py --config configs/mnist.yaml --dataset mnist --out runs/mnist_default
    python scripts/train.py --config configs/default.yaml --data my_data.npz --out runs/x \
        --set model.r=4 model.null_mode=global train.seed=1

``--data`` is an .npz with x_train, x_val and optionally obs_train, obs_val ([N, F] each).
``--set`` overrides any config key (dotted path, YAML value).

Outputs in --out: config.yaml, history.csv (one row per epoch, incl. gate diagnostics), params.json,
model.pt (state dict), assignments.npz (per head: G{h} [U, m] and keep{h} [U]).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmoe import assignments, fit, model_config, train_config  # noqa: E402
from mmoe.config import merge, to_dict  # noqa: E402


def parse_sets(items: list[str]) -> dict:
    out: dict = {}
    for it in items:
        key, val = it.split("=", 1)
        node = out
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["mnist"])
    src.add_argument("--data", help=".npz with x_train, x_val[, obs_train, obs_val]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.r=4 train.seed=1")
    args = ap.parse_args()

    raw = merge(yaml.safe_load(Path(args.config).read_text()) or {}, parse_sets(args.set))
    mcfg, tcfg = model_config(raw.get("model")), train_config(raw.get("train"))

    if args.dataset == "mnist":
        from mmoe.data import load_mnist
        d = load_mnist(ROOT / "data")
        data = {"train": (d["train"][0], None), "val": (d["val"][0], None)}
    else:
        z = np.load(args.data)
        data = {s: (z[f"x_{s}"], z[f"obs_{s}"] if f"obs_{s}" in z else None) for s in ("train", "val")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"model": to_dict(mcfg), "train": to_dict(tcfg)}, sort_keys=False))
    res = fit(mcfg, tcfg, data)
    res.history.to_csv(out / "history.csv", index=False)
    (out / "params.json").write_text(json.dumps(res.params, indent=1))
    torch.save(res.model.state_dict(), out / "model.pt")
    np.savez_compressed(out / "assignments.npz",
                        **{f"{k}{h}": a[k] for h, a in enumerate(assignments(res.model.encoder)) for k in ("G", "keep")})
    print(f"best val recon {res.best_val:.5f} at epoch {res.best_epoch}; saved to {out}")


if __name__ == "__main__":
    main()
