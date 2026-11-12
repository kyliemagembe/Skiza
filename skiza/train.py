"""Train and evaluate Skiza's denoising CNN.

Default configuration (the one the README reports):

    2-layer MIMO CNN, features [1, 5, 1], taps [40, 40], tanh
    SGD, lr 0.5, batch 32, 20 epochs, L1 loss
    300 pairs -> 250 train / 50 test

Usage:
    python -m skiza.train                 # train + evaluate, write results/
    python -m skiza.train --epochs 50     # override any hyper-parameter
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SAMPLE_RATE, load_or_create
from .model import CNN, SingleChannelCNN, count_parameters


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module | None, loader: DataLoader, device: torch.device) -> float:
    """Mean L1 error over a loader. ``model=None`` scores the untouched noisy input."""
    loss_fn = nn.L1Loss(reduction="sum")
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = x if model is None else model(x)
        total += loss_fn(y_hat, y).item()
        count += y.numel()
    return total / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    name: str = "model",
) -> dict:
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    history = {"train": [], "test": []}
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        history["train"].append(evaluate(model, train_loader, device))
        history["test"].append(evaluate(model, test_loader, device))
        print(
            f"[{name}] epoch {ep:3d}/{epochs}  train L1 {history['train'][-1]:.5f}"
            f"  test L1 {history['test'][-1]:.5f}"
        )
    return history


def save_plots(histories: dict, baseline_test: float, model, test_set, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        ep = np.arange(1, len(h["train"]) + 1)
        ax.plot(ep, h["train"], "--", label=f"{name} train")
        ax.plot(ep, h["test"], "-", label=f"{name} test")
    ax.axhline(baseline_test, color="grey", ls=":", label="noisy input (no processing)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1 reconstruction error")
    ax.set_title("Skiza training curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)

    # One test example: waveforms + spectrograms
    x, y = test_set[0]
    with torch.no_grad():
        y_hat = model(x.unsqueeze(0).to(next(model.parameters()).device)).cpu()[0]
    sigs = {"noisy input": x[0], "Skiza output": y_hat[0], "clean target": y[0]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for i, (title, s) in enumerate(sigs.items()):
        s = s.numpy()
        axes[0, i].plot(np.arange(len(s)) / SAMPLE_RATE, s, lw=0.4)
        axes[0, i].set_title(title)
        axes[0, i].set_ylim(-0.45, 0.45)
        axes[0, i].set_xlabel("time (s)")
        axes[1, i].specgram(s, NFFT=512, Fs=SAMPLE_RATE, noverlap=384, cmap="magma")
        axes[1, i].set_ylim(0, 4000)
        axes[1, i].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out / "example_test_sample.png", dpi=150)
    plt.close(fig)

    from scipy.io import wavfile

    audio = out / "audio"
    audio.mkdir(exist_ok=True)
    for title, s in sigs.items():
        pcm = (np.clip(s.numpy() / 0.45, -1, 1) * 32767).astype(np.int16)
        wavfile.write(audio / f"{title.split()[0].lower()}.wav", SAMPLE_RATE, pcm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/skiza_pairs.pt")
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--taps", type=int, nargs="+", default=[40, 40])
    p.add_argument("--features", type=int, nargs="+", default=[1, 5, 1])
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results")
    p.add_argument("--no-baseline-cnn", action="store_true", help="skip the single-channel CNN baseline")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_or_create(args.data, q=args.pairs)
    x_all, y_all = data[:, 0], data[:, 1]  # (Q, 1, N) each
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed))
    test_idx, train_idx = perm[: args.test_size], perm[args.test_size :]
    train_set = TensorDataset(x_all[train_idx], y_all[train_idx])
    test_set = TensorDataset(x_all[test_idx], y_all[test_idx])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    print(f"Device {device} | {len(train_set)} train / {len(test_set)} test pairs")

    baseline_train = evaluate(None, eval_train_loader, device)
    baseline_test = evaluate(None, test_loader, device)
    print(f"Noisy input (no processing): train L1 {baseline_train:.5f}  test L1 {baseline_test:.5f}")

    histories, results = {}, {}
    t0 = time.time()

    if not args.no_baseline_cnn:
        set_seed(args.seed)
        single = SingleChannelCNN(args.taps)
        h = train_model(single, train_loader, test_loader, args.epochs, args.lr, device, "single-channel CNN")
        histories["single-channel CNN"] = h
        results["single_channel_cnn"] = {
            "params": count_parameters(single),
            "train_l1": h["train"][-1],
            "test_l1": h["test"][-1],
        }

    set_seed(args.seed)
    model = CNN(args.taps, args.features)
    h = train_model(model, train_loader, test_loader, args.epochs, args.lr, device, "Skiza MIMO CNN")
    histories["Skiza MIMO CNN"] = h
    results["skiza_mimo_cnn"] = {
        "params": count_parameters(model),
        "train_l1": h["train"][-1],
        "test_l1": h["test"][-1],
    }

    def reduction(new: float, old: float) -> float:
        return 100.0 * (old - new) / old

    summary = {
        "config": vars(args),
        "train_pairs": len(train_set),
        "test_pairs": len(test_set),
        "noisy_input": {"train_l1": baseline_train, "test_l1": baseline_test},
        **results,
        "test_l1_reduction_vs_noisy_input_pct": reduction(results["skiza_mimo_cnn"]["test_l1"], baseline_test),
        "train_seconds": round(time.time() - t0, 1),
    }
    if "single_channel_cnn" in results:
        summary["test_l1_reduction_vs_single_channel_cnn_pct"] = reduction(
            results["skiza_mimo_cnn"]["test_l1"], results["single_channel_cnn"]["test_l1"]
        )

    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt / "skiza_cnn.pt")
    save_plots(histories, baseline_test, model, test_set, out)

    print("\n=== Summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
