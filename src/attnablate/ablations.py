
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .attention import TransformerClassifier
from .data import Corpus
from .training import RunResult, pick_device, set_seed, train_model


@dataclass
class Ablation:

    name: str
    overrides: dict
    question: str


ABLATIONS: list[Ablation] = [
    Ablation(
        name="full model",
        overrides={},
        question="Reference configuration that everything else is measured against.",
    ),
    Ablation(
        name="no positional encoding",
        overrides={"use_positional_encoding": False},
        question=(
            "Does word order carry signal here, or is sentiment mostly a bag of "
            "words? If accuracy barely moves, the model is not using order."
        ),
    ),
    Ablation(
        name="single head",
        overrides={"num_heads": 1},
        question=(
            "Do multiple attention heads help at this scale, or is one wide head "
            "of equal capacity enough?"
        ),
    ),
    Ablation(
        name="no feed-forward network",
        overrides={"use_ffn": False},
        question="How much of the model's capacity sits in the FFN rather than attention?",
    ),
    Ablation(
        name="no residual connections",
        overrides={"use_residual": False},
        question="Do residuals matter at 2 layers, or only once the stack is deep?",
    ),
    Ablation(
        name="no layer norm",
        overrides={"use_layernorm": False},
        question="Does training stay stable without normalisation at this depth?",
    ),
    Ablation(
        name="unmasked mean pooling",
        overrides={"pooling": "mean"},
        question=(
            " How much accuracy does averaging over padded positions actually cost?"
        ),
    ),
    Ablation(
        name="pre-layer-norm",
        overrides={"norm_first": True},
        question="Is pre-LN better behaved than the paper's post-LN at this scale?",
    ),
    Ablation(
        name="1 layer",
        overrides={"num_layers": 1},
        question="Does a second encoder block pay for itself?",
    ),
    Ablation(
        name="4 layers",
        overrides={"num_layers": 4},
        question="Does more depth help, or does it overfit faster on limited data?",
    ),
]


@dataclass
class AblationResult:
    """Aggregated outcome of one configuration across seeds."""

    name: str
    question: str
    overrides: dict
    seeds: list[int]
    test_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    best_epochs: list[int] = field(default_factory=list)
    n_params: int = 0

    def _mean_std(self, values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    @property
    def test_mean_std(self) -> tuple[float, float]:
        return self._mean_std(self.test_accs)

    @property
    def train_mean_std(self) -> tuple[float, float]:
        return self._mean_std(self.train_accs)

    @property
    def gap(self) -> float:
        return self.train_mean_std[0] - self.test_mean_std[0]

    def add(self, result: RunResult) -> None:
        self.test_accs.append(result.test_acc)
        self.val_accs.append(result.val_acc)
        self.train_accs.append(result.train_acc)
        self.best_epochs.append(result.best_epoch)
        self.n_params = result.n_params


def run_ablation(
    ablation: Ablation,
    corpus: Corpus,
    base_config: dict,
    seeds: list[int],
    train_kwargs: dict,
    verbose: bool = False,
) -> AblationResult:
    """Train one configuration once per seed and aggregate."""
    device = pick_device(train_kwargs.pop("device", "auto"))
    result = AblationResult(
        name=ablation.name,
        question=ablation.question,
        overrides=ablation.overrides,
        seeds=seeds,
    )

    for seed in seeds:
        generator = set_seed(seed)
        train_loader, val_loader, test_loader = corpus.loaders(
            train_kwargs.get("batch_size", 32), generator=generator
        )

        model = TransformerClassifier(
            vocab_size=len(corpus.vocab),
            **{**base_config, **ablation.overrides},
        )

        run = train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            verbose=verbose,
            **{k: v for k, v in train_kwargs.items() if k != "batch_size"},
        )
        result.add(run)
        print(
            f"  {ablation.name:<28} seed {seed:<4} "
            f"test {run.test_acc:.4f}  train {run.train_acc:.4f}  "
            f"best epoch {run.best_epoch}"
        )

    train_kwargs["device"] = str(device)
    return result


def format_results_table(
    results: list[AblationResult], baselines: list | None = None
) -> str:
    """Markdown table, ready to paste into the README."""
    lines = [
        "| Configuration | Test acc (mean ± std) | Train acc | Gap | Params |",
        "|---|---|---|---|---|",
    ]

    if baselines:
        for baseline in baselines:
            lines.append(
                f"| _{baseline.name}_ | {baseline.test_acc:.4f} | "
                f"{baseline.train_acc:.4f} | {baseline.generalisation_gap:+.4f} | "
                f"{baseline.n_params:,} |"
            )
        lines.append("| | | | | |")

    for result in results:
        test_mean, test_std = result.test_mean_std
        train_mean, _ = result.train_mean_std
        lines.append(
            f"| {result.name} | {test_mean:.4f} ± {test_std:.4f} | "
            f"{train_mean:.4f} | {result.gap:+.4f} | {result.n_params:,} |"
        )

    return "\n".join(lines)


def save_results(
    results: list[AblationResult],
    baselines: list,
    config: dict,
    path: Path,
) -> None:
    """Write the full run to JSON so the README table can be regenerated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "baselines": [asdict(b) for b in baselines],
        "ablations": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
