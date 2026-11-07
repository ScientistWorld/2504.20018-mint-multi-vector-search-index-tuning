"""Hidden test workload: fixed database copy plus held-out query instances.

This module deliberately does not import ``data/``: the database, templates,
protocol and the held-out query instances used for test scoring are fixed
here, so editing the public generator cannot change what the test measures.
The database and template definitions are a verbatim copy of the public
generator; the held-out instances come from the same per-template query
distribution with a seed that only exists in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPECS = {
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

HELD_OUT_SEED = 0x6A3F19C4B27D58E1
N_HELD_OUT = 1000
SINGLE_COLUMN_QUERY_NOISE = 0.22


@dataclass(frozen=True)
class TestQuery:
    columns: tuple[int, ...]
    vectors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TestTemplate:
    name: str
    columns: tuple[int, ...]
    probability: float


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _normalize_one(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-12)


def database(name: str) -> tuple[np.ndarray, ...]:
    spec = SPECS[name]
    rng = np.random.default_rng(spec["data_seed"])
    latent = rng.normal(size=(spec["n_rows"], 12)).astype("float32")
    columns = []
    for dim in spec["dims"]:
        proj = rng.normal(size=(12, dim)).astype("float32")
        shared = latent @ proj
        private = rng.normal(scale=0.85, size=(spec["n_rows"], dim)).astype("float32")
        columns.append(_normalize((shared + private).astype("float32")).astype("float32"))
    return tuple(columns)


def templates(name: str) -> tuple[TestTemplate, ...]:
    spec = SPECS[name]
    rng = np.random.default_rng([spec["data_seed"], 1])
    probs = rng.uniform(size=len(spec["templates"])).astype("float64")
    probs = probs / probs.sum()
    return tuple(
        TestTemplate(name=f"q{i + 1}", columns=tuple(cols), probability=float(p))
        for i, (cols, p) in enumerate(zip(spec["templates"], probs))
    )


def held_out_queries(name: str, columns: tuple[np.ndarray, ...]) -> dict[str, tuple[TestQuery, ...]]:
    spec = SPECS[name]
    out = {}
    for i, template in enumerate(templates(name)):
        rng = np.random.default_rng([HELD_OUT_SEED, int(spec["data_seed"]), i])
        queries = []
        for _ in range(N_HELD_OUT):
            vecs = []
            if len(template.columns) == 1:
                c = template.columns[0]
                anchor = int(rng.integers(columns[c].shape[0]))
                noise = rng.normal(scale=SINGLE_COLUMN_QUERY_NOISE, size=columns[c].shape[1]).astype("float32")
                vecs.append(_normalize_one((columns[c][anchor] + noise).astype("float32")).astype("float32"))
            else:
                for c in template.columns:
                    vecs.append(_normalize_one(rng.normal(size=columns[c].shape[1]).astype("float32")).astype("float32"))
            queries.append(TestQuery(columns=template.columns, vectors=tuple(vecs)))
        out[template.name] = tuple(queries)
    return out


def exact_topk(columns: tuple[np.ndarray, ...], queries: tuple[TestQuery, ...], top_k: int) -> np.ndarray:
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
