

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

_HTML_BREAK = re.compile(r"<br\s*/?>")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip HTML breaks and non-alphabetic characters, split on space.

    Deliberately simple. A better tokenizer would keep negations and punctuation,
    which matter for sentiment; see the limitations section of the README.
    """
    text = text.lower()
    text = _HTML_BREAK.sub(" ", text)
    text = _NON_ALPHA.sub("", text)
    return text.split()


def build_vocab(token_lists: list[list[str]], max_size: int) -> dict[str, int]:
    """Build a word to id map from training tokens only.

    Building the vocabulary on the full dataset before splitting leaks
    information from validation and test into training. It is a small leak for a
    frequency-capped vocabulary but it is still a leak, so this takes training
    tokens only.
    """
    counts = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in counts.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab


def encode(
    token_lists: list[list[str]], vocab: dict[str, int], max_len: int
) -> torch.Tensor:
    """Map tokens to ids, truncate to ``max_len``, right-pad with ``PAD_IDX``."""
    out = torch.full((len(token_lists), max_len), PAD_IDX, dtype=torch.long)
    for row, tokens in enumerate(token_lists):
        ids = [vocab.get(tok, UNK_IDX) for tok in tokens[:max_len]]
        out[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


@dataclass
class Corpus:
    """Encoded splits plus the raw tokens, which the baselines and plots need."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    vocab: dict[str, int]
    train_tokens: list[list[str]]
    val_tokens: list[list[str]]
    test_tokens: list[list[str]]

    @property
    def idx2word(self) -> dict[int, str]:
        return {idx: word for word, idx in self.vocab.items()}

    def loaders(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train = DataLoader(
            TensorDataset(self.train_x, self.train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        val = DataLoader(
            TensorDataset(self.val_x, self.val_y), batch_size=batch_size, shuffle=False
        )
        test = DataLoader(
            TensorDataset(self.test_x, self.test_y), batch_size=batch_size, shuffle=False
        )
        return train, val, test

    def describe(self) -> str:
        def balance(y: torch.Tensor) -> str:
            pos = int(y.sum())
            return f"{pos}/{len(y) - pos} pos/neg"

        return (
            f"train {len(self.train_y):>6,} ({balance(self.train_y)})\n"
            f"val   {len(self.val_y):>6,} ({balance(self.val_y)})\n"
            f"test  {len(self.test_y):>6,} ({balance(self.test_y)})\n"
            f"vocab {len(self.vocab):>6,}"
        )


def load_imdb(
    n_train: int = 20_000,
    n_val: int = 5_000,
    n_test: int = 5_000,
    max_vocab: int = 20_000,
    max_len: int = 256,
    seed: int = 42,
) -> Corpus:
    
    from datasets import load_dataset  # imported lazily; heavy and optional

    dataset = load_dataset("imdb")
    train_pool = dataset["train"].shuffle(seed=seed)
    test_pool = dataset["test"].shuffle(seed=seed)

    if n_train < 0:
        n_train = len(train_pool) - max(n_val, 0)
    if n_val < 0:
        n_val = len(train_pool) - n_train
    if n_test < 0:
        n_test = len(test_pool)

    if n_train + n_val > len(train_pool):
        raise ValueError(
            f"n_train + n_val = {n_train + n_val} exceeds the {len(train_pool)} "
            "available training examples"
        )

    train_split = train_pool.select(range(n_train))
    val_split = train_pool.select(range(n_train, n_train + n_val))
    test_split = test_pool.select(range(min(n_test, len(test_pool))))

    train_tokens = [tokenize(r["text"]) for r in train_split]
    val_tokens = [tokenize(r["text"]) for r in val_split]
    test_tokens = [tokenize(r["text"]) for r in test_split]

    vocab = build_vocab(train_tokens, max_vocab)

    return Corpus(
        train_x=encode(train_tokens, vocab, max_len),
        train_y=torch.tensor([r["label"] for r in train_split], dtype=torch.long),
        val_x=encode(val_tokens, vocab, max_len),
        val_y=torch.tensor([r["label"] for r in val_split], dtype=torch.long),
        test_x=encode(test_tokens, vocab, max_len),
        test_y=torch.tensor([r["label"] for r in test_split], dtype=torch.long),
        vocab=vocab,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )


def synthetic_corpus(
    n_train: int = 400,
    n_val: int = 100,
    n_test: int = 100,
    max_len: int = 32,
    seed: int = 0,
) -> Corpus:
    
    rng = np.random.default_rng(seed)
    positive = ["great", "superb", "loved", "brilliant", "excellent"]
    negative = ["awful", "terrible", "hated", "dull", "poor"]
    filler = ["the", "a", "film", "movie", "was", "and", "it", "this"]

    def make(n: int) -> tuple[list[list[str]], list[int]]:
        tokens, labels = [], []
        for _ in range(n):
            label = int(rng.integers(0, 2))
            pool = positive if label == 1 else negative
            length = int(rng.integers(8, max_len))
            words = list(rng.choice(filler, size=length))
            for pos in rng.choice(length, size=2, replace=False):
                words[pos] = str(rng.choice(pool))
            tokens.append(words)
            labels.append(label)
        return tokens, labels

    train_tokens, train_labels = make(n_train)
    val_tokens, val_labels = make(n_val)
    test_tokens, test_labels = make(n_test)
    vocab = build_vocab(train_tokens, 100)

    return Corpus(
        train_x=encode(train_tokens, vocab, max_len),
        train_y=torch.tensor(train_labels, dtype=torch.long),
        val_x=encode(val_tokens, vocab, max_len),
        val_y=torch.tensor(val_labels, dtype=torch.long),
        test_x=encode(test_tokens, vocab, max_len),
        test_y=torch.tensor(test_labels, dtype=torch.long),
        vocab=vocab,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )
