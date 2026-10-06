import threading

import numpy as np
import pytest

import strata


def brute_force(data, queries, k, metric):
    if metric == "l2":
        d = ((queries[:, None, :] - data[None, :, :]) ** 2).sum(-1)
    else:
        if metric == "cosine":
            data = data / np.linalg.norm(data, axis=1, keepdims=True)
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        d = 1 - queries @ data.T
    return np.argsort(d, axis=1)[:, :k]


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((4000, 48)).astype(np.float32), rng.standard_normal((100, 48)).astype(np.float32)


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_recall_matches_brute_force(data, metric):
    x, q = data
    index = strata.Index(dim=48, metric=metric)
    index.add(x)
    ids, dists = index.search(q, k=10, ef=128)
    assert ids.shape == (100, 10) and dists.shape == (100, 10)
    assert np.all(np.diff(dists, axis=1) >= 0)
    assert recall(ids, brute_force(x, q, 10, metric)) >= 0.95


def test_flat_index_is_exact(data):
    x, q = data
    flat = strata.FlatIndex(48, "l2")
    flat.add(x)
    ids, _ = flat.search(q, k=10)
    assert recall(ids, brute_force(x, q, 10, "l2")) == 1.0


def test_custom_ids_filters_and_missing_results(data):
    x, q = data
    index = strata.Index(48)
    ids = np.arange(len(x)) * 10 + 7
    index.add(x, ids)
    found, _ = index.search(q[:5], k=5)
    assert set(found.ravel()) <= set(ids)

    allowed = ids[:20]
    f_ids, _ = index.search(q, k=5, filter_ids=allowed)
    assert set(f_ids.ravel()) <= set(allowed)
    e_ids, _ = index.search(q, k=5, exclude_ids=allowed)
    assert not set(e_ids.ravel()) & set(allowed)

    few, few_d = index.search(q[:1], k=5, filter_ids=ids[:2])
    assert list(few[0, 2:]) == [-1, -1, -1]
    assert np.isinf(few_d[0, 2:]).all()


def test_remove_compact_and_contains(data):
    x, q = data
    index = strata.Index(48)
    index.add(x)
    assert index.remove(np.arange(0, 4000, 2)) == 2000
    assert len(index) == 2000 and 1 in index and 2 not in index
    found, _ = index.search(q, k=10)
    assert np.all(found % 2 == 1)
    index.compact()
    assert index.stats()["deleted"] == 0
    found2, _ = index.search(q, k=10)
    assert np.all(found2 % 2 == 1)


def test_save_load_roundtrip(tmp_path, data):
    x, q = data
    index = strata.Index(48, "cosine", M=12)
    index.add(x)
    index.ef_search = 99
    path = str(tmp_path / "idx.bin")
    index.save(path)
    loaded = strata.Index.load(path)
    assert loaded.ef_search == 99 and loaded.M == 12 and loaded.metric == "cosine"
    a, da = index.search(q, k=10)
    b, db = loaded.search(q, k=10)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(da, db)
    np.testing.assert_allclose(loaded.get_vector(5), x[5] / np.linalg.norm(x[5]), rtol=1e-5)


def test_corrupted_file_raises(tmp_path, data):
    x, _ = data
    index = strata.Index(48)
    index.add(x[:100])
    path = tmp_path / "idx.bin"
    index.save(str(path))
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(strata.StrataError, match="checksum"):
        strata.Index.load(str(path))


def test_input_validation():
    index = strata.Index(4)
    with pytest.raises(ValueError):
        index.add(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 4), dtype=np.float32), ids=[1])
    with pytest.raises(ValueError):
        index.add(np.zeros((1, 4), dtype=np.float32), ids=[-3])
    with pytest.raises(strata.StrataError):
        strata.Index(4, metric="hamming")
    # float64 input is converted
    index.add(np.ones((2, 4)), ids=[1, 2])
    assert len(index) == 2


def test_gil_is_released_during_search(data):
    x, q = data
    index = strata.Index(48)
    index.add(x)
    results = []

    def worker():
        results.append(index.search(q, k=10, num_threads=1)[0])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results[1:]:
        np.testing.assert_array_equal(r, results[0])


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_sq8_index(tmp_path, metric):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3000, 32)).astype(np.float32)
    q = rng.standard_normal((50, 32)).astype(np.float32)
    exact = strata.Index(32, metric)
    exact.add(x)
    for rerank in (True, False):
        index = strata.Index(32, metric, quantization="sq8", rerank=rerank)
        assert index.quantization == "sq8" and index.rerank == rerank
        index.add(x)
        ids, _ = index.search(q, k=10, ef=200)
        ref, _ = exact.search(q, k=10, ef=200)
        recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, ref)])
        assert recall >= (0.95 if rerank else 0.85)
        index.save(str(tmp_path / "q.bin"))
        loaded = strata.Index.load(str(tmp_path / "q.bin"))
        assert loaded.quantization == "sq8"
        np.testing.assert_array_equal(loaded.search(q, k=10, ef=200)[0], ids)


def test_sq8_collection(tmp_path):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((500, 16)).astype(np.float32)
    path = str(tmp_path / "c")
    col = strata.Collection.create(path, dim=16, quantization="sq8")
    col.upsert(range(500), x)
    del col
    col = strata.Collection.open(path)
    assert col.query(x[:1], k=1)[0][0].id == 0
