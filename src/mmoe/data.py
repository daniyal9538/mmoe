"""MNIST loader (no torchvision needed): downloads the IDX files once and caches them as one .npz."""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"x_train": "train-images-idx3-ubyte.gz", "y_train": "train-labels-idx1-ubyte.gz",
         "x_test": "t10k-images-idx3-ubyte.gz", "y_test": "t10k-labels-idx1-ubyte.gz"}


def _read_idx(raw: bytes) -> np.ndarray:
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(root: str | Path = "data", n_val: int = 10_000, seed: int = 0) -> dict:
    """Pixels scaled to [0, 1], flattened to 784 features.

    Returns {'train': (X, y), 'val': (X, y), 'test': (X, y)}; val is a random split of the 60k train set.
    """
    root = Path(root)
    cache = root / "mnist.npz"
    if not cache.exists():
        root.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, name in FILES.items():
            with urllib.request.urlopen(MIRROR + name) as r:
                arrays[key] = _read_idx(gzip.decompress(r.read()))
        np.savez_compressed(cache, **arrays)
    d = np.load(cache)
    x = d["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y = d["y_train"].astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(x))
    va, tr = perm[:n_val], perm[n_val:]
    xt = d["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    return {"train": (x[tr], y[tr]), "val": (x[va], y[va]), "test": (xt, d["y_test"].astype(np.int64))}


def occlude(x: np.ndarray, size: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Structured missingness for the observed-flag demos: one random size x size square per image is
    unobserved. Returns (x with the square zeroed, obs mask)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    obs = np.ones((n, 28, 28), dtype=np.float32)
    r0, c0 = rng.integers(0, 28 - size + 1, n), rng.integers(0, 28 - size + 1, n)
    for i in range(n):
        obs[i, r0[i]:r0[i] + size, c0[i]:c0[i] + size] = 0
    obs = obs.reshape(n, 784)
    return x * obs, obs
