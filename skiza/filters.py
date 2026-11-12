"""Convolutional filters written in terms of the time-shift operator S.

Everything here follows the shift-operator view of convolution:

    z = sum_{k=0}^{K-1} h_k S^k x

where S^k delays the signal by k samples (zeros enter from the left, so the
filter is causal). Causality matters for a hearing device: the output at time
n may only depend on samples that have already reached the microphone.

Shapes follow the PyTorch convention (batch, features, time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def shift(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Apply the shift operator S^k: delay ``x`` by ``k`` samples along time.

    The first ``k`` samples become zero (the signal has no past before t=0).
    """
    if k == 0:
        return x
    n = x.shape[-1]
    if k >= n:
        return torch.zeros_like(x)
    return F.pad(x[..., : n - k], (k, 0))


def convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Single-input single-output causal convolution  z = sum_k h[k] S^k x.

    ``x`` is (..., N) and ``h`` is a 1-D tensor with K taps.
    """
    z = torch.zeros_like(x)
    xk = x
    for k in range(h.shape[-1]):
        z = z + h[k] * xk
        xk = shift(xk, 1)
    return z


def mimo_convolve(X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """MIMO causal convolution  Y = sum_k S^k X H_k  (lab equation 9).

    Args:
        X: input features, shape (B, F, N).
        H: filter coefficients, shape (K, F, G). ``H[k]`` is the F x G
           coefficient matrix that multiplies the k-times-shifted input.

    Returns:
        Y: output features, shape (B, G, N). Output feature g is
           y^g = sum_f sum_k h_k^{fg} S^k x^f  (lab equation 10).
    """
    K = H.shape[0]
    Y = None
    Xk = X
    for k in range(K):
        # (B, F, N) x (F, G) -> (B, G, N)
        term = torch.einsum("bfn,fg->bgn", Xk, H[k])
        Y = term if Y is None else Y + term
        Xk = shift(Xk, 1)
    return Y


def mimo_convolve_fast(X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Same result as :func:`mimo_convolve`, computed with ``conv1d``.

    ``conv1d`` computes cross-correlation, so we left-pad by K-1 (causality)
    and flip the taps. This path is what training uses; the shift-based
    version is kept as the readable reference and is checked against this one
    in the test-suite.
    """
    K = H.shape[0]
    weight = H.permute(2, 1, 0).flip(-1)  # (G, F, K), taps reversed
    return F.conv1d(F.pad(X, (K - 1, 0)), weight)


class MIMOFilter(nn.Module):
    """A learnable MIMO convolutional filter with K taps, F inputs, G outputs."""

    def __init__(self, taps: int, in_features: int, out_features: int, fast: bool = True):
        super().__init__()
        self.taps = taps
        self.in_features = in_features
        self.out_features = out_features
        self.fast = fast
        # Uniform init scaled by fan-in (same rule PyTorch uses for Conv1d).
        bound = 1.0 / math.sqrt(in_features * taps)
        self.H = nn.Parameter(torch.empty(taps, in_features, out_features).uniform_(-bound, bound))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return mimo_convolve_fast(X, self.H) if self.fast else mimo_convolve(X, self.H)

    def extra_repr(self) -> str:
        return f"taps={self.taps}, F={self.in_features}, G={self.out_features}, fast={self.fast}"
