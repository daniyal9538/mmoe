"""Correlation prior for ``gate_init='prior_corr'``: |Pearson correlation| over co-observed pairs."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def abs_corr_prior(x_val, x_obs=None, min_coobs: int = 50, device: str | None = None) -> np.ndarray:
    """|corr| [F, F] of the values over rows where both features are observed (train split only!).

    Pairs co-observed in fewer than ``min_coobs`` rows get 0, constant features get 0, the diagonal is 1.
    Computed exactly from sufficient statistics (sums over pairs of observed entries), in float64.

    Args:
        x_val: [N, F] array or tensor of values (anything at unobserved entries is ignored).
        x_obs: [N, F] 0/1 observed mask, or None (everything observed).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(x_val) if not torch.is_tensor(x_val) else x_val).to(dev, torch.float64)
    O = torch.ones_like(X) if x_obs is None else torch.as_tensor(
        np.asarray(x_obs) if not torch.is_tensor(x_obs) else x_obs).to(dev, torch.float64)
    X = X * O
    n = O.T @ O                                     # [F, F] co-observation counts
    sx = X.T @ O                                    # sx[a, b] = sum of x_a over rows where a and b observed
    sxx = (X * X).T @ O
    sxy = X.T @ X
    cov = n * sxy - sx * sx.T
    var_a = (n * sxx - sx * sx).clamp_min(0)
    corr = cov / torch.sqrt(var_a * var_a.T).clamp_min(1e-12)
    corr = torch.where(n >= min_coobs, corr, torch.zeros_like(corr)).clamp(-1, 1)
    corr.fill_diagonal_(1.0)
    return corr.abs().cpu().numpy()
