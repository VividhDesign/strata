import numpy as np
import pytest

import strata


@pytest.fixture
def docs():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((300, 16)).astype(np.float32)
    metadata = [{"source": ["wiki", "news", "blog"][i % 3], "year": 2020 + i % 4, "id_str": f"doc-{i}"}
                for i in range(300)]
    return vectors, metadata


def test_query_with_metadata_filter(tmp_path, docs):
    vectors, metadata = docs
    col = strata.Collection.create(str(tmp_path / "c"), dim=16, metric="cosine")
    col.upsert(range(300), vectors, metadata)
    assert len(col) == 300

    hits = col.query(vectors[:1], k=5)[0]
    assert hits[0].id == 0 and hits[0].metadata["id_str"] == "doc-0"

    news = col.query(vectors[:4], k=10, where={"source": "news"})
    assert all(h.metadata["source"] == "news" for row in news for h in row)
    recent = col.query(vectors[:1], k=300, where={"year": {"$in": [2022, 2023]}, "source": {"$ne": "blog"}})[0]
    assert len(recent) == 100
    assert all(h.metadata["year"] >= 2022 and h.metadata["source"] != "blog" for h in recent)


def test_durability_across_reopen(tmp_path, docs):
    vectors, metadata = docs
    path = str(tmp_path / "c")
    col = strata.Collection.create(path, dim=16)
    col.upsert(range(200), vectors[:200], metadata[:200])
    col.checkpoint()
    col.upsert(range(200, 300), vectors[200:], metadata[200:])
    assert col.delete([0, 1, 999]) == 2
    del col  # no checkpoint for the last two writes: they must come back from the WAL

    col = strata.Collection.open(path)
    assert len(col) == 298
    assert col.get(0) is None
    vec, md = col.get(250)
    assert md["id_str"] == "doc-250"
    np.testing.assert_allclose(vec, vectors[250] / np.linalg.norm(vectors[250]), rtol=1e-5)
    stats = col.stats()
    assert stats["snapshot_seq"] == 1 and stats["last_seq"] == 3


def test_bad_filters_and_metadata(tmp_path, docs):
    vectors, _ = docs
    col = strata.Collection.create(str(tmp_path / "c"), dim=16)
    col.upsert([1], vectors[:1], [{"a": 1}])
    with pytest.raises(strata.StrataError):
        col.query(vectors[:1], where={"a": {"$regex": "x"}})
    with pytest.raises(strata.StrataError):
        strata.Collection.create(str(tmp_path / "c"), dim=16)
