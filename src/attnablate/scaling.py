

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .attention import TransformerClassifier
from .baselines import BaselineResult, run_baselines
from .data import Corpus, build_vocab, encode
from .training import pick_device, set_seed, train_model

#: Training set sizes swept by default. Roughly logarithmic, starting below the
#: original coursework size of 1,000 and ending at the practical limit of the
#: IMDB training split once 5,000 are held out for validation.
DEFAULT_SIZES: list[int] = [500, 1000, 2000, 5000, 10000, 20000]


def subset_corpus(corpus: Corpus, n_train: int, max_vocab: int, seed: int = 0) -> Corpus:
    
    if n_train > len(corpus.train_tokens):
        raise ValueError(
            f"asked for {n_train} training examples but only "
            f"{len(corpus.train_tokens)} are available"
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(corpus.train_tokens))[:n_train]

    train_tokens = [corpus.train_tokens[i] for i in idx]
    train_y = corpus.train_y[idx.copy()]

    vocab = build_vocab(train_tokens, max_vocab)
    max_len = corpus.train_x.shape[1]

    return Corpus(
        train_x=encode(train_tokens, vocab, max_len),
        train_y=train_y,
        val_x=encode(corpus.val_tokens, vocab, max_len),
        val_y=corpus.val_y,
        test_x=encode(corpus.test_tokens, vocab, max_len),
        test_y=corpus.test_y,
        vocab=vocab,
        train_tokens=train_tokens,
        val_tokens=corpus.val_tokens,
        test_tokens=corpus.test_tokens,
    )


@dataclass
class ScalingPoint:
    """Transformer and baseline performance at one training set size."""

    n_train: int
    vocab_size: int
    transformer_test: list[float] = field(default_factory=list)
    transformer_train: list[float] = field(default_factory=list)
    baselines: dict[str, float] = field(default_factory=dict)
    n_params: int = 0

    def _mean_std(self, values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return (
            statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )

    @property
    def transformer_mean_std(self) -> tuple[float, float]:
        return self._mean_std(self.transformer_test)

    @property
    def best_baseline(self) -> tuple[str, float]:
        """The strongest baseline at this size, which is what must be beaten."""
        competitive = {k: v for k, v in self.baselines.items() if "majority" not in k}
        if not competitive:
            return ("none", 0.0)
        name = max(competitive, key=lambda k: competitive[k])
        return (name, competitive[name])

    @property
    def margin(self) -> float:
        """Transformer minus best baseline. Positive means the Transformer wins."""
        return self.transformer_mean_std[0] - self.best_baseline[1]


def find_crossover(points: list[ScalingPoint]) -> float | None:
    
    ordered = sorted(points, key=lambda p: p.n_train)

    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.margin < 0 <= later.margin:
            span = later.margin - earlier.margin
            if span == 0:
                return float(later.n_train)
            fraction = -earlier.margin / span
            log_n = math.log(earlier.n_train) + fraction * (
                math.log(later.n_train) - math.log(earlier.n_train)
            )
            return math.exp(log_n)

    return None


def run_scaling_sweep(
    corpus: Corpus,
    base_config: dict,
    sizes: list[int],
    seeds: list[int],
    train_kwargs: dict,
    max_vocab: int = 20_000,
    verbose: bool = False,
) -> list[ScalingPoint]:
    """Train the reference model and the baselines at each training set size."""
    device = pick_device(train_kwargs.pop("device", "auto"))
    batch_size = train_kwargs.pop("batch_size", 32)
    points: list[ScalingPoint] = []

    for n_train in sizes:
        subset = subset_corpus(corpus, n_train, max_vocab)
        point = ScalingPoint(n_train=n_train, vocab_size=len(subset.vocab))

        for baseline in run_baselines(subset):
            point.baselines[baseline.name] = baseline.test_acc

        for seed in seeds:
            generator = set_seed(seed)
            train_loader, val_loader, test_loader = subset.loaders(
                batch_size, generator=generator
            )
            model = TransformerClassifier(
                vocab_size=len(subset.vocab), **base_config
            )
            run = train_model(
                model,
                train_loader,
                val_loader,
                test_loader,
                device=device,
                verbose=verbose,
                **train_kwargs,
            )
            point.transformer_test.append(run.test_acc)
            point.transformer_train.append(run.train_acc)
            point.n_params = run.n_params

        mean, std = point.transformer_mean_std
        best_name, best_acc = point.best_baseline
        print(
            f"  n={n_train:>6,}  transformer {mean:.4f} ± {std:.4f}   "
            f"best baseline {best_acc:.4f} ({best_name})   "
            f"margin {point.margin:+.4f}"
        )
        points.append(point)

    train_kwargs["device"] = str(device)
    train_kwargs["batch_size"] = batch_size
    return points


def format_scaling_table(points: list[ScalingPoint]) -> str:
    """Markdown table, ready to paste into the README."""
    lines = [
        "| Train size | Transformer test acc | Best baseline | Baseline used | Margin |",
        "|---|---|---|---|---|",
    ]
    for point in sorted(points, key=lambda p: p.n_train):
        mean, std = point.transformer_mean_std
        name, acc = point.best_baseline
        winner = "**+**" if point.margin >= 0 else "−"
        lines.append(
            f"| {point.n_train:,} | {mean:.4f} ± {std:.4f} | {acc:.4f} | "
            f"{name} | {point.margin:+.4f} {winner} |"
        )

    crossover = find_crossover(points)
    lines.append("")
    if crossover is None:
        ordered = sorted(points, key=lambda p: p.n_train)
        if all(p.margin >= 0 for p in ordered):
            lines.append(
                "**No crossover found.** The Transformer beat the best linear "
                "baseline at every size tested, including the smallest."
            )
        else:
            lines.append(
                "**No crossover found.** The Transformer did not overtake the "
                f"best linear baseline at any size up to "
                f"{ordered[-1].n_train:,}. On this task, with this "
                "architecture and no pretraining, the linear model is the better "
                "choice across the whole range swept."
            )
    else:
        lines.append(
            f"**Crossover at roughly {crossover:,.0f} training examples.** Below "
            "that, tf-idf with a linear classifier is the better model. Above it, "
            "the Transformer is."
        )

    return "\n".join(lines)


def save_scaling_results(
    points: list[ScalingPoint], config: dict, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "crossover_n_train": find_crossover(points),
        "points": [asdict(p) for p in points],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
