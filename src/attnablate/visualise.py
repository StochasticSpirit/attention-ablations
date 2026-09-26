

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display

import matplotlib.pyplot as plt
import numpy as np
import torch

from .attention import TransformerClassifier
from .data import PAD_IDX
from .training import History


def plot_training_curves(history: History, path: Path, title: str = "") -> None:
    
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history.train_loss, marker="o", ms=3, label="train")
    axes[0].plot(epochs, history.val_loss, marker="s", ms=3, label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")

    axes[1].plot(epochs, history.train_acc, marker="o", ms=3, label="train")
    axes[1].plot(epochs, history.val_acc, marker="s", ms=3, label="validation")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")

    if history.val_acc:
        best = int(np.argmax(history.val_acc))
        axes[1].axvline(
            best + 1, color="grey", ls="--", lw=1, label=f"selected (epoch {best + 1})"
        )

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_attention_heads(
    model: TransformerClassifier,
    tokens: torch.Tensor,
    idx2word: dict[int, str],
    path: Path,
    layer: int = -1,
    max_tokens: int = 28,
) -> None:
    """
    Args:
        tokens: ``(1, seq)`` token ids for a single example.
        layer: which encoder block to plot; ``-1`` is the last.
        max_tokens: truncate the display, since a 256x256 grid is unreadable.
    """
    model.eval()
    device = next(model.parameters()).device
    _, attentions = model(tokens.to(device), return_attention=True)

    weights = attentions[layer][0].cpu()  # (heads, seq, seq)
    real = (tokens[0] != PAD_IDX).sum().item()
    keep = min(real, max_tokens)

    words = [idx2word.get(int(t), "?") for t in tokens[0][:keep]]
    weights = weights[:, :keep, :keep]

    n_heads = weights.shape[0]
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)

    for head in range(n_heads):
        ax = axes[head // cols][head % cols]
        im = ax.imshow(weights[head], cmap="viridis", aspect="auto")
        ax.set_xticks(range(keep))
        ax.set_yticks(range(keep))
        ax.set_xticklabels(words, rotation=90, fontsize=6)
        ax.set_yticklabels(words, fontsize=6)
        ax.set_title(f"head {head}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for spare in range(n_heads, rows * cols):
        axes[spare // cols][spare % cols].axis("off")

    fig.suptitle(
        f"Attention weights, encoder block {layer if layer >= 0 else len(attentions) - 1}"
        "\nrow = query token, column = key token attended to"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def token_attention_ranking(
    model: TransformerClassifier,
    tokens: torch.Tensor,
    idx2word: dict[int, str],
    layer: int = -1,
    top_k: int = 12,
) -> list[tuple[str, float]]:
    
    model.eval()
    device = next(model.parameters()).device
    _, attentions = model(tokens.to(device), return_attention=True)

    weights = attentions[layer][0].cpu()  # (heads, seq, seq)
    real = (tokens[0] != PAD_IDX).sum().item()

    # Mean over heads and over query positions gives incoming attention per key.
    incoming = weights[:, :real, :real].mean(dim=0).mean(dim=0)

    words = [idx2word.get(int(t), "?") for t in tokens[0][:real]]
    ranked = sorted(zip(words, incoming.tolist()), key=lambda p: p[1], reverse=True)

    # De-duplicate repeated words, keeping the highest score for each.
    seen: dict[str, float] = {}
    for word, score in ranked:
        seen.setdefault(word, score)

    return list(seen.items())[:top_k]


def plot_ablation_comparison(results: list, path: Path) -> None:
    """Horizontal bar chart of test accuracy per configuration, with error bars."""
    names = [r.name for r in results]
    means = [r.test_mean_std[0] for r in results]
    stds = [r.test_mean_std[1] for r in results]

    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    colours = ["#c44e52" if n == "full model" else "#4c72b0" for n in names]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 2))
    ax.barh(names, means, xerr=stds, color=colours, capsize=4, height=0.6)
    ax.set_xlabel("test accuracy (mean ± std across seeds)")
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    ax.set_title("Component ablations")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_curve(points: list, path: Path, crossover: float | None = None) -> None:
    
    ordered = sorted(points, key=lambda p: p.n_train)
    sizes = [p.n_train for p in ordered]
    means = np.array([p.transformer_mean_std[0] for p in ordered])
    stds = np.array([p.transformer_mean_std[1] for p in ordered])
    baseline = [p.best_baseline[1] for p in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(sizes, means, marker="o", color="#c44e52", lw=2, label="Transformer (from scratch)")
    ax.fill_between(sizes, means - stds, means + stds, color="#c44e52", alpha=0.18)
    ax.plot(
        sizes,
        baseline,
        marker="s",
        color="#4c72b0",
        lw=2,
        ls="--",
        label="best linear baseline (tf-idf)",
    )

    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")

    if crossover is not None:
        ax.axvline(crossover, color="black", ls="-.", lw=1.2)
        ax.annotate(
            f"crossover ≈ {crossover:,.0f}",
            xy=(crossover, ax.get_ylim()[0] + 0.03),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            rotation=90,
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Suppress the log-scale minor ticks, which otherwise print labels like
    # "3 x 10^2" between the sizes actually swept.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Does the Transformer overtake a linear model, and when?")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_gap(points: list, path: Path) -> None:
    """Train minus test accuracy against training set size.

    Overfitting is the mechanism behind the crossover, so it is worth showing
    directly rather than leaving the reader to infer it.
    """
    ordered = sorted(points, key=lambda p: p.n_train)
    sizes = [p.n_train for p in ordered]
    gaps = [
        (np.mean(p.transformer_train) if p.transformer_train else 0.0)
        - p.transformer_mean_std[0]
        for p in ordered
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sizes, gaps, marker="o", color="#dd8452", lw=2)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Suppress the log-scale minor ticks, which otherwise print labels like
    # "3 x 10^2" between the sizes actually swept.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("train accuracy − test accuracy")
    ax.set_title("Generalisation gap shrinks as training data grows")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
