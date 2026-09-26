

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import Corpus


@dataclass
class BaselineResult:
    name: str
    test_acc: float
    val_acc: float
    train_acc: float
    n_params: int

    @property
    def generalisation_gap(self) -> float:
        return self.train_acc - self.test_acc


def _as_documents(token_lists: list[list[str]]) -> list[str]:
     against a baseline on raw text measures the preprocessing, not the model.
    
    return [" ".join(tokens) for tokens in token_lists]


def run_baselines(corpus: Corpus, seed: int = 42) -> list[BaselineResult]:
    train_docs = _as_documents(corpus.train_tokens)
    val_docs = _as_documents(corpus.val_tokens)
    test_docs = _as_documents(corpus.test_tokens)

    y_train = corpus.train_y.numpy()
    y_val = corpus.val_y.numpy()
    y_test = corpus.test_y.numpy()

    max_features = max(len(corpus.vocab) - 2, 1)  # exclude PAD and UNK

    models: dict[str, Pipeline] = {
        "majority class": Pipeline(
            [
                ("vec", TfidfVectorizer(max_features=max_features)),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "tf-idf + multinomial NB": Pipeline(
            [
                ("vec", TfidfVectorizer(max_features=max_features)),
                ("clf", MultinomialNB()),
            ]
        ),
        "tf-idf + logistic regression": Pipeline(
            [
                ("vec", TfidfVectorizer(max_features=max_features)),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "tf-idf bigrams + logistic regression": Pipeline(
            [
                (
                    "vec",
                    TfidfVectorizer(
                        max_features=max_features * 2, ngram_range=(1, 2)
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
    }

    results: list[BaselineResult] = []
    for name, pipeline in models.items():
        pipeline.fit(train_docs, y_train)

        # Parameter count for a linear model is the number of learned
        # coefficients, which makes the comparison against the Transformer's
        # parameter count meaningful rather than decorative.
        classifier = pipeline.named_steps["clf"]
        if hasattr(classifier, "coef_"):
            n_params = int(np.asarray(classifier.coef_).size) + int(
                np.asarray(getattr(classifier, "intercept_", [0])).size
            )
        else:
            n_params = 1

        results.append(
            BaselineResult(
                name=name,
                train_acc=float(pipeline.score(train_docs, y_train)),
                val_acc=float(pipeline.score(val_docs, y_val)),
                test_acc=float(pipeline.score(test_docs, y_test)),
                n_params=n_params,
            )
        )

    return results
