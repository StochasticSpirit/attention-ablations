

from __future__ import annotations

import math

import pytest
import torch

from attnablate.data import synthetic_corpus
from attnablate.scaling import ScalingPoint, find_crossover, subset_corpus


@pytest.fixture
def corpus():
    return synthetic_corpus(n_train=200, n_val=50, n_test=50, max_len=16)


# --------------------------------------------------------------------------
# Subsetting
# --------------------------------------------------------------------------


def test_subset_reduces_training_set_only(corpus) -> None:
    """Validation and test must be identical in size at every point on the curve."""
    subset = subset_corpus(corpus, n_train=50, max_vocab=100)

    assert len(subset.train_y) == 50
    assert subset.train_x.shape[0] == 50
    assert len(subset.val_y) == len(corpus.val_y)
    assert len(subset.test_y) == len(corpus.test_y)


def test_subset_keeps_evaluation_text_unchanged(corpus) -> None:
    """The val and test *documents* must not change, only their encoding."""
    subset = subset_corpus(corpus, n_train=50, max_vocab=100)

    assert subset.val_tokens == corpus.val_tokens
    assert subset.test_tokens == corpus.test_tokens
    assert torch.equal(subset.val_y, corpus.val_y)
    assert torch.equal(subset.test_y, corpus.test_y)


def test_subset_rebuilds_vocabulary_from_the_subset_only(corpus) -> None:
    
    subset = subset_corpus(corpus, n_train=30, max_vocab=100)

    training_words = {tok for tokens in subset.train_tokens for tok in tokens}
    for word in subset.vocab:
        if word not in {"<PAD>", "<UNK>"}:
            assert word in training_words, f"'{word}' leaked into the vocabulary"


def test_subset_vocabulary_grows_with_training_size(corpus) -> None:
    small = subset_corpus(corpus, n_train=20, max_vocab=1000)
    large = subset_corpus(corpus, n_train=200, max_vocab=1000)

    assert len(small.vocab) <= len(large.vocab)


def test_subset_labels_stay_aligned_with_documents(corpus) -> None:
    """Shuffling the subset must not decouple a review from its label."""
    subset = subset_corpus(corpus, n_train=40, max_vocab=100, seed=7)

    originals = {
        tuple(tokens): int(label)
        for tokens, label in zip(corpus.train_tokens, corpus.train_y)
    }
    for tokens, label in zip(subset.train_tokens, subset.train_y):
        assert originals[tuple(tokens)] == int(label)


def test_subset_is_deterministic_for_a_given_seed(corpus) -> None:
    a = subset_corpus(corpus, n_train=40, max_vocab=100, seed=3)
    b = subset_corpus(corpus, n_train=40, max_vocab=100, seed=3)

    assert a.train_tokens == b.train_tokens
    assert torch.equal(a.train_y, b.train_y)


def test_subset_rejects_oversized_request(corpus) -> None:
    with pytest.raises(ValueError, match="only"):
        subset_corpus(corpus, n_train=10_000, max_vocab=100)


# --------------------------------------------------------------------------
# Crossover interpolation
# --------------------------------------------------------------------------


def _point(n: int, transformer: float, baseline: float) -> ScalingPoint:
    point = ScalingPoint(n_train=n, vocab_size=100)
    point.transformer_test = [transformer]
    point.baselines = {"tf-idf + logistic regression": baseline}
    return point


def test_crossover_is_found_between_bracketing_sizes() -> None:
    points = [
        _point(500, 0.60, 0.80),
        _point(1000, 0.70, 0.80),
        _point(5000, 0.85, 0.82),  # Transformer overtakes somewhere in here
        _point(20000, 0.90, 0.83),
    ]
    crossover = find_crossover(points)

    assert crossover is not None
    assert 1000 < crossover < 5000


def test_crossover_interpolates_in_log_space() -> None:
    """Midway in log space between 1,000 and 10,000 is roughly 3,162, not 5,500."""
    points = [_point(1000, 0.70, 0.80), _point(10000, 0.90, 0.80)]
    crossover = find_crossover(points)

    assert crossover == pytest.approx(math.sqrt(1000 * 10000), rel=0.02)


def test_no_crossover_when_transformer_always_loses() -> None:
    points = [
        _point(500, 0.60, 0.80),
        _point(5000, 0.70, 0.84),
        _point(20000, 0.79, 0.86),
    ]
    assert find_crossover(points) is None


def test_no_crossover_when_transformer_always_wins() -> None:
    points = [_point(500, 0.85, 0.80), _point(5000, 0.90, 0.82)]
    assert find_crossover(points) is None


def test_crossover_ignores_the_majority_class_baseline() -> None:
    """The bar to clear is the best real baseline, not the trivial one."""
    point = ScalingPoint(n_train=1000, vocab_size=100)
    point.transformer_test = [0.70]
    point.baselines = {
        "majority class": 0.50,
        "tf-idf + logistic regression": 0.85,
    }

    name, acc = point.best_baseline
    assert name == "tf-idf + logistic regression"
    assert acc == 0.85
    assert point.margin == pytest.approx(0.70 - 0.85)


def test_points_may_be_supplied_out_of_order() -> None:
    """``find_crossover`` sorts, so the caller does not have to."""
    points = [
        _point(20000, 0.90, 0.83),
        _point(500, 0.60, 0.80),
        _point(5000, 0.85, 0.82),
        _point(1000, 0.70, 0.80),
    ]
    crossover = find_crossover(points)

    assert crossover is not None
    assert 1000 < crossover < 5000


def test_scaling_point_reports_mean_and_std_across_seeds() -> None:
    point = ScalingPoint(n_train=1000, vocab_size=100)
    point.transformer_test = [0.70, 0.74, 0.72]

    mean, std = point.transformer_mean_std
    assert mean == pytest.approx(0.72)
    assert std > 0
