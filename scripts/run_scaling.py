#!/usr/bin/env python3
"""Data-scaling sweep to determine the cross over point where the Transformer overtakes a linear baseline.

Trains the reference architecture at a range of training set sizes, fits tf-idf
baselines on the same subsets, and reports the crossover point.

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attnablate.data import load_imdb, synthetic_corpus  # noqa: E402
from attnablate.scaling import (  # noqa: E402
    DEFAULT_SIZES,
    find_crossover,
    format_scaling_table,
    run_scaling_sweep,
    save_scaling_results,
)
from attnablate.training import pick_device  # noqa: E402
from attnablate.visualise import plot_scaling_curve, plot_scaling_gap  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("--synthetic", action="store_true", help="offline toy corpus")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="training set sizes to sweep",
    )
    p.add_argument("--n-val", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=25000)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--max-len", type=int, default=256)

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "scaling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sizes = sorted(args.sizes)

    print(f"device: {pick_device(args.device)}")
    print(f"sizes:  {sizes}\n")

    # Load once at the largest size needed. Validation and test are then held
    # identical across every point on the curve.
    print("loading data")
    if args.synthetic:
        corpus = synthetic_corpus(
            n_train=max(sizes), n_val=200, n_test=200, max_len=32
        )
    else:
        corpus = load_imdb(
            n_train=max(sizes),
            n_val=args.n_val,
            n_test=args.n_test,
            max_vocab=args.max_vocab,
            max_len=args.max_len,
        )
    print(corpus.describe(), "\n")

    base_config = dict(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        max_len=corpus.train_x.shape[1],
    )

    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )

    seeds = list(range(42, 42 + args.seeds))
    print(f"sweeping {len(sizes)} sizes x {len(seeds)} seeds")

    points = run_scaling_sweep(
        corpus,
        base_config,
        sizes,
        seeds,
        train_kwargs,
        max_vocab=args.max_vocab,
    )

    table = format_scaling_table(points)
    print("\n" + table)
    (args.out / "scaling_table.md").write_text(table)

    crossover = find_crossover(points)
    save_scaling_results(
        points,
        config={**base_config, **train_kwargs, "sizes": sizes, "seeds": seeds},
        path=args.out / "scaling.json",
    )

    plot_scaling_curve(points, args.out / "scaling_curve.png", crossover)
    plot_scaling_gap(points, args.out / "scaling_gap.png")

    print(f"\ndone in {time.time() - started:.0f}s. results written to {args.out}")


if __name__ == "__main__":
    main()
