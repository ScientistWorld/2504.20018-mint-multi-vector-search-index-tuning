"""Deterministic multi-vector search benchmark data.

The original paper combines public single-vector embedding datasets into
multi-column rows. This generator creates a compact substitute with the same
problem structure: each row has multiple vector columns, a query names a subset
of columns (a query *template*) and supplies one vector per named column, and
relevance is the sum of cosine similarities over the named columns.

A workload consists of

* a fixed database (``columns``),
* query templates with workload probabilities, a top-k, a recall@k floor and a
  storage budget (number of indexes), and
* a visible sample of training query instances per template.

Held-out query instances of the same templates are drawn from the same
distribution by the hidden evaluator; they are not produced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Query:
    name: str  # template name, e.g. "q2"
    columns: tuple[int, ...]
    vectors: tuple[np.ndarray, ...]
    probability: float


@dataclass(frozen=True)
class Template:
    name: str
    columns: tuple[int, ...]
    probability: float


@dataclass(frozen=True)
class Workload:
    name: str
    columns: tuple[np.ndarray, ...]
    templates: tuple[Template, ...]
    train_queries: dict  # template name -> tuple[Query, ...]
    top_k: int
    recall_threshold: float
    storage_budget: int

    @property
    def n_rows(self) -> int:
        return int(self.columns[0].shape[0])


# Public workload definitions. ``naive_small`` mirrors the paper's Naive case
# study (three columns, queries on {1}, {1,2}, {2,3}, {1,2,3}, storage budget =
# number of columns). ``bicomplex_small`` is a scaled BiComplex-style workload:
# a fixed draw of eight distinct column subsets with at least two columns over
# five columns (each column included with probability 0.5), storage budget =
# number of columns.
WORKLOAD_SPECS = {
    "naive_small": {
        "dims": (32, 40, 48),
        "templates": ((0,), (0, 1), (1, 2), (0, 1, 2)),
        "n_rows": 3000,
        "top_k": 100,
        "recall_threshold": 0.90,
        "storage_budget": 3,
        "data_seed": 7,
    },
    "bicomplex_small": {
        "dims": (16, 20, 24, 28, 32),
        "templates": (
            (0, 1, 3),
            (1, 2, 4),
            (0, 2, 3, 4),
            (0, 1, 2, 3, 4),
            (1, 3),
            (2, 3, 4),
            (0, 1, 4),
            (0, 2),
        ),
        "n_rows": 3000,
        "top_k": 100,
        "recall_threshold": 0.90,
        "storage_budget": 5,
        "data_seed": 11,
    },
}

TRAIN_QUERY_SEED = 20240501
N_TRAIN_QUERIES = 1000
SINGLE_COLUMN_QUERY_NOISE = 0.22


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
        columns.append(_normalize((shared + private).astype("float32")).astype("float32"))
    return tuple(columns)


def make_database(name: str) -> tuple[np.ndarray, ...]:
    spec = WORKLOAD_SPECS[name]
    rng = np.random.default_rng(spec["data_seed"])
    return _make_columns(rng, n_rows=spec["n_rows"], dims=spec["dims"])


def template_probabilities(name: str) -> np.ndarray:
    """Workload probabilities: uniform draws normalized to sum to one."""
    spec = WORKLOAD_SPECS[name]
    rng = np.random.default_rng([spec["data_seed"], 1])
    probs = rng.uniform(size=len(spec["templates"])).astype("float64")
    return probs / probs.sum()


def make_templates(name: str) -> tuple[Template, ...]:
    spec = WORKLOAD_SPECS[name]
    probs = template_probabilities(name)
    return tuple(
        Template(name=f"q{i + 1}", columns=tuple(cols), probability=float(p))
        for i, (cols, p) in enumerate(zip(spec["templates"], probs))
    )


def sample_queries(
    columns: tuple[np.ndarray, ...],
    template: Template,
    n: int,
    rng: np.random.Generator,
) -> tuple[Query, ...]:
    """Draw ``n`` query instances of ``template``.

    Single-column templates perturb a random database row. Multi-column
    templates use an independent random direction per column, so the aggregate
    nearest neighbors are not trivially found from any single column.
    """
    out = []
    for _ in range(n):
        vecs = []
        if len(template.columns) == 1:
            c = template.columns[0]
            anchor = int(rng.integers(columns[c].shape[0]))
            noise = rng.normal(scale=SINGLE_COLUMN_QUERY_NOISE, size=columns[c].shape[1]).astype("float32")
            vecs.append(_normalize_one((columns[c][anchor] + noise).astype("float32")).astype("float32"))
        else:
            for c in template.columns:
                vecs.append(_normalize_one(rng.normal(size=columns[c].shape[1]).astype("float32")).astype("float32"))
        out.append(Query(name=template.name, columns=template.columns, vectors=tuple(vecs), probability=template.probability))
    return tuple(out)


def make_workload(name: str = "naive_small", n_train: int = N_TRAIN_QUERIES, train_seed: int = TRAIN_QUERY_SEED) -> Workload:
    """Build the public workload: database, templates and training queries."""
    if name not in WORKLOAD_SPECS:
        raise ValueError(f"unknown workload {name!r}")
    spec = WORKLOAD_SPECS[name]
    columns = make_database(name)
    templates = make_templates(name)
    train = {}
    for i, template in enumerate(templates):
        rng = np.random.default_rng([int(train_seed), int(spec["data_seed"]), i])
        train[template.name] = sample_queries(columns, template, int(n_train), rng)
    return Workload(
        name=name,
        columns=columns,
        templates=templates,
        train_queries=train,
        top_k=int(spec["top_k"]),
        recall_threshold=float(spec["recall_threshold"]),
        storage_budget=int(spec["storage_budget"]),
    )


def exact_topk_batch(columns: tuple[np.ndarray, ...], queries: tuple[Query, ...], top_k: int) -> np.ndarray:
    """Exact top-k row ids (N, k) for query instances sharing one template."""
    n_rows = columns[0].shape[0]
    scores = np.zeros((len(queries), n_rows), dtype="float32")
    for j, c in enumerate(queries[0].columns):
        qmat = np.stack([q.vectors[j] for q in queries]).astype("float32")
        scores += qmat @ columns[c].T
    k = min(int(top_k), n_rows)
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1).astype("int64")


def exact_topk(columns: tuple[np.ndarray, ...], query: Query, top_k: int) -> np.ndarray:
    return exact_topk_batch(columns, (query,), top_k)[0]


def index_name(key: tuple[int, ...]) -> str:
    return "x" + ",".join(str(c + 1) for c in key)


def parse_index_name(name: str) -> tuple[int, ...]:
    if not name.startswith("x"):
        raise ValueError(f"invalid index name {name!r}")
    return tuple(sorted(int(part) - 1 for part in name[1:].split(",") if part))


def all_candidate_indexes(query_sets: Iterable[tuple[int, ...]], subset_diff: int = 2) -> set[tuple[int, ...]]:
    candidates: set[tuple[int, ...]] = set()
    for qcols in query_sets:
        min_size = max(1, len(qcols) - subset_diff)
        for size in range(min_size, len(qcols) + 1):
            candidates.update(tuple(c) for c in combinations(qcols, size))
    return candidates
