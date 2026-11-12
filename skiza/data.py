"""Synthetic noisy/clean audio pairs for training and evaluating Skiza.

Each pair (x_q, y_q) is one second of audio at 16 kHz:

* y_q (clean): a voice-like signal — a glottal-style harmonic series with a
  slowly drifting pitch (vibrato/intonation), shaped by syllable-rate
  amplitude envelopes and soft formant resonances.
* x_q (noisy): y_q plus a random mix of everyday noise found in Nairobi-style
  environments: low-frequency engine hum (matatus, boda-bodas, generators),
  broadband traffic/wind noise, and crowd babble (other voices), mixed at a
  random signal-to-noise ratio.

The generator is fully deterministic given a seed, so the dataset never needs
to be committed to git — run ``python -m skiza.data`` to recreate it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter

SAMPLE_RATE = 16_000
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
PEAK = 0.4  # keep signals well inside tanh's near-linear range


# --------------------------------------------------------------------------- #
# Clean, voice-like signals
# --------------------------------------------------------------------------- #
def _resonator(signal: np.ndarray, freq: float, bw: float, fs: int) -> np.ndarray:
    """Two-pole resonator, a cheap stand-in for a vocal-tract formant."""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * freq / fs
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return lfilter(b, a, signal)


def voice_like(rng: np.random.Generator, n: int = N_SAMPLES, fs: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / fs
    f0 = rng.uniform(95, 260)  # adult male .. child range
    drift = 1 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 2 * np.pi))
    vibrato = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(4, 6) * t)
    phase = 2 * np.pi * np.cumsum(f0 * drift * vibrato) / fs

    n_harm = int(min(30, (fs / 2) / (f0 * 1.1)))
    src = sum((1.0 / h) * np.sin(h * phase + rng.uniform(0, 2 * np.pi)) for h in range(1, n_harm + 1))

    # Formants (roughly an /a/-/e/-/o/ vowel space).
    f1, f2, f3 = rng.uniform(300, 850), rng.uniform(900, 2300), rng.uniform(2400, 3200)
    sig = (
        _resonator(src, f1, 90, fs)
        + 0.6 * _resonator(src, f2, 120, fs)
        + 0.3 * _resonator(src, f3, 160, fs)
    )

    # Syllables: 3-6 Hz on/off envelope with smooth edges.
    syl_rate = rng.uniform(3, 6)
    env = np.clip(np.sin(2 * np.pi * syl_rate * t + rng.uniform(0, 2 * np.pi)), 0, None) ** 0.7
    sig = sig * env
    return sig


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def engine_hum(rng: np.random.Generator, n: int = N_SAMPLES, fs: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / fs
    f = rng.uniform(35, 110)
    rpm_wobble = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t)
    phase = 2 * np.pi * np.cumsum(f * rpm_wobble) / fs
    return sum((0.8 ** h) * np.sin(h * phase + rng.uniform(0, 2 * np.pi)) for h in range(1, 8))


def broadband(rng: np.random.Generator, n: int = N_SAMPLES) -> np.ndarray:
    """Pink-ish noise (traffic, wind) via a leaky integrator on white noise."""
    white = rng.standard_normal(n)
    alpha = rng.uniform(0.0, 0.95)
    return lfilter([1.0], [1.0, -alpha], white)


def babble(rng: np.random.Generator, n: int = N_SAMPLES, fs: int = SAMPLE_RATE) -> np.ndarray:
    talkers = rng.integers(2, 5)
    return sum(voice_like(rng, n, fs) for _ in range(talkers))


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.sqrt(np.mean(x**2)) + 1e-12)


def make_pair(rng: np.random.Generator, snr_db_range=(0.0, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    clean = _normalize(voice_like(rng))

    weights = rng.dirichlet(np.ones(3))
    noise = (
        weights[0] * _normalize(engine_hum(rng))
        + weights[1] * _normalize(broadband(rng))
        + weights[2] * _normalize(babble(rng))
    )
    noise = _normalize(noise)
    snr_db = rng.uniform(*snr_db_range)
    noisy = clean + noise * 10 ** (-snr_db / 20)

    scale = PEAK / np.max(np.abs(noisy))
    return (noisy * scale).astype(np.float32), (clean * scale).astype(np.float32)


def generate_dataset(q: int = 300, seed: int = 0) -> torch.Tensor:
    """Return a tensor of shape (Q, 2, 1, N): [:, 0] noisy, [:, 1] clean."""
    rng = np.random.default_rng(seed)
    pairs = [make_pair(rng) for _ in range(q)]
    noisy = np.stack([p[0] for p in pairs])[:, None, :]
    clean = np.stack([p[1] for p in pairs])[:, None, :]
    return torch.from_numpy(np.stack([noisy, clean], axis=1))


def load_or_create(path: str | Path = "data/skiza_pairs.pt", q: int = 300, seed: int = 0) -> torch.Tensor:
    path = Path(path)
    if path.exists():
        data = torch.load(path)
        if data.shape[0] == q:
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(q, seed)
    torch.save(data, path)
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Skiza noisy/clean dataset.")
    p.add_argument("--out", default="data/skiza_pairs.pt")
    p.add_argument("-q", type=int, default=300, help="number of pairs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    data = generate_dataset(args.q, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    print(f"Saved {tuple(data.shape)} to {args.out}")


if __name__ == "__main__":
    main()
