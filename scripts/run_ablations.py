#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from attnablate.ablations import (  # noqa: E402
    ABLATIONS,
    format_results_table,
    run_ablation,
    save_results,
)
from attnablate.attention import TransformerClassifier  # noqa: E402
from attnablate.baselines import run_baselines  # noqa: E402
from attnablate.data import load_imdb, synthetic_corpus  # noqa: E402
from attnablate.training import pick_device, set_seed, train_model  # noqa: E402
from attnablate.visualise import (  # noqa: E402
    plot_ablation_comparison,
    plot_attention_heads,
    plot_training_curves,
    token_attention_ranking,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = p.add_argument_group("data")
    data.add_argument("--synthetic", action="store_true", help="use the offline toy corpus")
    data.add_argument("--n-train", type=int, default=5000)
    data.add_argument("--n-val", type=int, default=2000)
    data.add_argument("--n-test", type=int, default=5000)
    data.add_argument("--max-vocab", type=int, default=20000)
    data.add_argument("--max-len", type=int, default=256)

    model = p.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument("--d-ff", type=int, default=256)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1)

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--seeds", type=int, default=3, help="number of seeds per config")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--only", nargs="*", help="run only these ablation names")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"device: {pick_device(args.device)}\n")

    print("loading data")
    if args.synthetic:
        corpus = synthetic_corpus(
            n_train=args.n_train if args.n_train < 2000 else 400,
            n_val=100,
            n_test=100,
            max_len=32,
        )
    else:
        corpus = load_imdb(
            n_train=args.n_train,
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

    # Architecture summary, which the assignment brief asked for and the
    # original notebook never produced.
    reference = TransformerClassifier(vocab_size=len(corpus.vocab), **base_config)
    summary = reference.architecture_summary()
    print(summary, "\n")
    (args.out / "architecture.txt").write_text(summary)

    baselines = []
    if not args.skip_baselines:
        print("fitting baselines")
        baselines = run_baselines(corpus)
        for b in baselines:
            print(f"  {b.name:<38} test {b.test_acc:.4f}  train {b.train_acc:.4f}")
        print()

    seeds = list(range(42, 42 + args.seeds))
    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )

    selected = ABLATIONS
    if args.only:
        wanted = set(args.only)
        selected = [a for a in ABLATIONS if a.name in wanted]
        if not selected:
            raise SystemExit(f"no ablation matched {args.only}")

    print(f"running {len(selected)} configurations x {len(seeds)} seeds")
    results = []
    for ablation in selected:
        results.append(
            run_ablation(ablation, corpus, base_config, seeds, dict(train_kwargs))
        )

    table = format_results_table(results, baselines)
    print("\n" + table)
    (args.out / "results_table.md").write_text(table)

    save_results(
        results,
        baselines,
        config={**base_config, **train_kwargs, "seeds": seeds, "synthetic": args.synthetic},
        path=args.out / "results.json",
    )
    plot_ablation_comparison(results, args.out / "ablations.png")

    # Retrain the reference configuration once more so we have a live model to
    # produce training curves and attention maps from.
    print("\nretraining the full model for figures")
    generator = set_seed(seeds[0])
    train_loader, val_loader, test_loader = corpus.loaders(args.batch_size, generator)
    model = TransformerClassifier(vocab_size=len(corpus.vocab), **base_config)
    run = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=pick_device(args.device),
        verbose=True,
        **{k: v for k, v in train_kwargs.items() if k not in {"batch_size", "device"}},
    )
    plot_training_curves(
        run.history,
        args.out / "training_curves.png",
        title=f"Full model, test accuracy {run.test_acc:.4f}",
    )

    example = corpus.test_x[:1]
    plot_attention_heads(model, example, corpus.idx2word, args.out / "attention_heads.png")

    ranking = token_attention_ranking(model, example, corpus.idx2word)
    lines = ["Tokens receiving the most attention (mean over heads and queries):", ""]
    lines += [f"  {word:<20}{score:.4f}" for word, score in ranking]
    text = "\n".join(lines)
    print("\n" + text)
    (args.out / "attention_ranking.txt").write_text(text)

    torch.save(model.state_dict(), args.out / "full_model.pt")
    print(f"\ndone in {time.time() - started:.0f}s. results written to {args.out}")


if __name__ == "__main__":
    main()
