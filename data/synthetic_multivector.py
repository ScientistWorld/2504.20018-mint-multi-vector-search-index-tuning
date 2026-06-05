"""Deterministic multi-vector search benchmark data.

The original paper combines public single-vector embedding datasets into
multi-column rows. This generator creates a compact substitute with the same
problem structure: each row has multiple vector columns, queries address column
subsets, and relevance is defined by the sum of cosine similarities over the
queried columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Query:
    name: str
    columns: tuple[int, ...]
    vectors: tuple[np.ndarray, ...]
    probability: float


@dataclass(frozen=True)
class Workload:
    name: str
    columns: tuple[np.ndarray, ...]
    queries: tuple[Query, ...]
    top_k: int
    recall_threshold: float
    storage_budget: int


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _normalize_one(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-12)


def _make_columns(rng: np.random.Generator, n_rows: int, dims: Iterable[int]) -> tuple[np.ndarray, ...]:
    latent_dim = 12
    latent = rng.normal(size=(n_rows, latent_dim)).astype("float32")
    columns: list[np.ndarray] = []
    for dim in dims:
        proj = rng.normal(size=(latent_dim, dim)).astype("float32")
        shared = latent @ proj
        private = rng.normal(scale=0.85, size=(n_rows, dim)).astype("float32")
        columns.append(_normalize((shared + private).astype("float32")))
    return tuple(columns)


def _query_for_row(
    rng: np.random.Generator,
    columns: tuple[np.ndarray, ...],
    query_cols: tuple[int, ...],
    anchor: int,
    name: str,
    probability: float,
) -> Query:
    qvecs = []
    for c in query_cols:
        if len(query_cols) == 1:
            noisy = columns[c][anchor] + rng.normal(scale=0.22, size=columns[c].shape[1]).astype("float32")
        else:
            # Multi-column queries should not be trivially solved by each
            # single column alone. Random directions create aggregate nearest
            # neighbors whose individual-column ranks vary widely, matching the
            # phenomenon MINT is designed to exploit.
            noisy = rng.normal(size=columns[c].shape[1]).astype("float32")
        qvecs.append(_normalize_one(noisy.astype("float32")))
    return Query(name=name, columns=query_cols, vectors=tuple(qvecs), probability=probability)


def make_workload(name: str = "naive_small", n_rows: int = 8000, seed: int = 7) -> Workload:
    """Create a paper-shaped workload.

    ``naive_small`` mirrors the paper's Naive case study with three columns and
    four queries. ``bicomplex_small`` mirrors the larger binomial workload with
    more columns and queries, still compact enough for CPU test jobs.
    """
    rng = np.random.default_rng(seed)
    if name == "naive_small":
        dims = (32, 40, 48)
        query_sets = ((0,), (0, 1), (1, 2), (0, 1, 2))
        top_k = 100
        recall = 0.90
        budget = 3
    elif name == "bicomplex_small":
        dims = (16, 20, 24, 28, 32)
        all_sets = []
        while len(all_sets) < 8:
            cols = tuple(i for i in range(len(dims)) if rng.random() < 0.5)
            if len(cols) >= 2 and cols not in all_sets:
                all_sets.append(cols)
        query_sets = tuple(all_sets)
        top_k = 100
        recall = 0.90
        budget = len(dims)
    else:
        raise ValueError(f"unknown workload {name!r}")

    columns = _make_columns(rng, n_rows=n_rows, dims=dims)
    anchors = rng.choice(n_rows, size=len(query_sets), replace=False)
    probs = rng.uniform(size=len(query_sets)).astype("float64")
    probs = probs / probs.sum()
    queries = tuple(
        _query_for_row(rng, columns, tuple(qcols), int(anchor), f"q{i+1}", float(prob))
        for i, (qcols, anchor, prob) in enumerate(zip(query_sets, anchors, probs))
    )
    return Workload(name=name, columns=columns, queries=queries, top_k=top_k, recall_threshold=recall, storage_budget=budget)


def exact_topk(columns: tuple[np.ndarray, ...], query: Query, top_k: int) -> np.ndarray:
    scores = np.zeros(columns[0].shape[0], dtype="float32")
    for col_id, qvec in zip(query.columns, query.vectors):
        scores += columns[col_id] @ qvec
    if top_k >= scores.shape[0]:
        return np.argsort(-scores).astype("int64")
    part = np.argpartition(-scores, top_k - 1)[:top_k]
    return part[np.argsort(-scores[part])].astype("int64")


def all_candidate_indexes(query_sets: Iterable[tuple[int, ...]], subset_diff: int = 2) -> set[tuple[int, ...]]:
    candidates: set[tuple[int, ...]] = set()
    for qcols in query_sets:
        min_size = max(1, len(qcols) - subset_diff)
        for size in range(min_size, len(qcols) + 1):
            candidates.update(tuple(c) for c in combinations(qcols, size))
    return candidates
