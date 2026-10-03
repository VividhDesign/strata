"""Strata: an HNSW vector database written from scratch in C++.

>>> import numpy as np, strata
>>> index = strata.Index(dim=128, metric="cosine")
>>> index.add(np.random.rand(1000, 128).astype("float32"))
>>> ids, distances = index.search(np.random.rand(5, 128).astype("float32"), k=10)
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._strata import FlatIndex, Index, StrataError, __version__, simd_backend
from ._strata import _Collection

__all__ = ["Index", "FlatIndex", "Collection", "Hit", "StrataError", "simd_backend", "__version__"]


class Hit(dict):
    """One query result: {"id", "distance", "metadata"} with attribute access."""

    __getattr__ = dict.__getitem__


class Collection:
    """Durable collection: HNSW index + JSON metadata + write-ahead log.

    Writes are logged before they are applied, so a crash loses nothing that was
    acknowledged. Use ``Collection.create`` for a new directory, ``Collection.open`` to
    reopen one (recovering from the latest snapshot plus the WAL).
    """

    def __init__(self, native: _Collection):
        self._c = native

    @classmethod
    def create(cls, path: str, dim: int, metric: str = "cosine", M: int = 16,
               ef_construction: int = 200, sync_wal: bool = True) -> "Collection":
        return cls(_Collection.create(path, dim, metric, M, ef_construction, sync_wal))

    @classmethod
    def open(cls, path: str, sync_wal: bool = True) -> "Collection":
        return cls(_Collection.open(path, sync_wal))

    def upsert(self, ids: Iterable[int], vectors: np.ndarray,
               metadata: Sequence[Mapping[str, Any]] | None = None) -> None:
        ids = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
        md = [] if metadata is None else [json.dumps(m) for m in metadata]
        self._c.upsert(ids, np.asarray(vectors, dtype=np.float32), md)

    def delete(self, ids: Iterable[int]) -> int:
        return self._c.remove(np.asarray(list(ids), dtype=np.int64))

    def query(self, vectors: np.ndarray, k: int = 10, ef: int = 0,
              where: Mapping[str, Any] | None = None) -> list[list[Hit]]:
        """Nearest neighbours per query vector, optionally restricted by a metadata filter
        such as ``{"source": "fiqa", "year": {"$in": [2020, 2021]}}``."""
        raw = self._c.query(np.asarray(vectors, dtype=np.float32), k, ef, json.dumps(where) if where else "")
        return [[Hit(id=i, distance=d, metadata=json.loads(m)) for i, d, m in row] for row in raw]

    def get(self, id: int) -> tuple[np.ndarray, dict] | None:
        res = self._c.get(id)
        return None if res is None else (res[0], json.loads(res[1]))

    def checkpoint(self) -> None:
        self._c.checkpoint()

    def stats(self) -> dict:
        return json.loads(self._c.stats_json())

    @property
    def dim(self) -> int:
        return self._c.dim

    @property
    def ef_search(self) -> int:
        return self.stats()["ef_search"]

    @ef_search.setter
    def ef_search(self, ef: int) -> None:
        self._c.set_ef_search(ef)

    def __len__(self) -> int:
        return len(self._c)

    def __repr__(self) -> str:
        s = self.stats()
        return f"<strata.Collection dim={s['dim']} metric={s['metric']} size={s['size']}>"
