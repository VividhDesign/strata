"""Minimal HTTP client for strata-server (standard library only).

>>> from strata.client import Client
>>> db = Client("http://127.0.0.1:8080")
>>> db.create_collection("docs", dim=384)
>>> db.upsert("docs", ids=[1, 2], vectors=[[...], [...]], metadata=[{"src": "a"}, {"src": "b"}])
>>> db.query("docs", vector=[...], k=5, filter={"src": "a"})
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Sequence


class ServerError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class Client:
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise ServerError(e.code, msg) from None

    def health(self) -> dict:
        return self._call("GET", "/health")

    def create_collection(self, name: str, dim: int, metric: str = "cosine", M: int = 16,
                          ef_construction: int = 200, quantization: str = "none",
                          rerank: bool = True) -> dict:
        return self._call("POST", "/collections", {"name": name, "dim": dim, "metric": metric, "M": M,
                                                   "ef_construction": ef_construction,
                                                   "quantization": quantization, "rerank": rerank})

    def list_collections(self) -> list[dict]:
        return self._call("GET", "/collections")["collections"]

    def stats(self, name: str) -> dict:
        return self._call("GET", f"/collections/{name}")

    def drop_collection(self, name: str) -> dict:
        return self._call("DELETE", f"/collections/{name}")

    def upsert(self, name: str, ids: Sequence[int], vectors: Sequence[Sequence[float]],
               metadata: Sequence[dict] | None = None) -> dict:
        body = {"ids": [int(i) for i in ids], "vectors": [[float(x) for x in v] for v in vectors]}
        if metadata is not None:
            body["metadata"] = list(metadata)
        return self._call("POST", f"/collections/{name}/upsert", body)

    def query(self, name: str, vector: Sequence[float], k: int = 10, ef: int = 0,
              filter: dict | None = None, include_metadata: bool = True) -> list[dict]:
        body = {"vector": [float(x) for x in vector], "k": k, "ef": ef, "include_metadata": include_metadata}
        if filter:
            body["filter"] = filter
        return self._call("POST", f"/collections/{name}/query", body)["results"]

    def delete(self, name: str, ids: Sequence[int]) -> int:
        return self._call("POST", f"/collections/{name}/delete", {"ids": [int(i) for i in ids]})["deleted"]

    def get(self, name: str, id: int) -> dict:
        return self._call("GET", f"/collections/{name}/points/{id}")

    def checkpoint(self, name: str) -> dict:
        return self._call("POST", f"/collections/{name}/checkpoint")
