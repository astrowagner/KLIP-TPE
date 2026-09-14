"""Two runs started at the same time must not corrupt the shared download cache.

The paper rerun has two halves -- the science runs and the benchmarks -- that write to
different directories and are meant to be runnable side by side.  The one thing they share
is the cached cubes, so a cold cache is the one place they could tread on each other: both
would have written the same ``<file>.part`` and one would have renamed the interleaved
mixture into place.  The result is a cube of exactly the right size and the wrong contents,
which nothing downstream checks.
"""
import os
import threading
import time
import urllib.request

import pytest

from klip_tpe import datasets

PAYLOAD = b"klip-tpe test cube " * 12_000          # comfortably over the 2880-byte floor


@pytest.fixture
def cold_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path))
    return str(tmp_path)


def _slow_retrieve(seen):
    """A download that takes long enough for concurrent callers to overlap."""
    def retrieve(url, filename):
        seen.append(filename)
        with open(filename, "wb") as f:
            for i in range(0, len(PAYLOAD), 40_000):
                f.write(PAYLOAD[i:i + 40_000])
                f.flush()
                time.sleep(0.02)
    return retrieve


def test_concurrent_fetches_do_not_corrupt_the_cache(cold_cache, monkeypatch):
    seen, out, errs = [], {}, []
    monkeypatch.setattr(urllib.request, "urlretrieve", _slow_retrieve(seen))
    name = sorted(datasets.DATASETS)[0]

    def go(k):
        try:
            out[k] = datasets.fetch(name, quiet=True)
        except Exception as exc:                    # noqa: BLE001 - reported below
            errs.append(exc)

    threads = [threading.Thread(target=go, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert not errs, errs
    assert len(set(seen)) == len(seen), f"two fetches shared one scratch file: {seen}"
    for role, path in next(iter(out.values())).items():
        assert os.path.getsize(path) == len(PAYLOAD), f"{role}: wrong size -- bytes interleaved"
        with open(path, "rb") as f:
            assert f.read() == PAYLOAD, f"{role}: contents corrupted"


def test_no_scratch_files_are_left_behind(cold_cache, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", _slow_retrieve([]))
    datasets.fetch(sorted(datasets.DATASETS)[0], quiet=True)
    assert not [f for f in os.listdir(cold_cache) if ".part" in f], os.listdir(cold_cache)


def test_a_failed_download_leaves_no_half_file(cold_cache, monkeypatch):
    """A dropped connection must not leave something the next run mistakes for the cube."""
    def boom(url, filename):
        with open(filename, "wb") as f:
            f.write(b"half a file" * 500)
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    with pytest.raises(OSError):
        datasets.fetch(sorted(datasets.DATASETS)[0], quiet=True)
    assert not [f for f in os.listdir(cold_cache) if ".part" in f], os.listdir(cold_cache)


def test_a_warm_cache_does_not_download_again(cold_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(urllib.request, "urlretrieve", _slow_retrieve(calls))
    name = sorted(datasets.DATASETS)[0]
    datasets.fetch(name, quiet=True)
    n = len(calls)
    datasets.fetch(name, quiet=True)
    assert len(calls) == n, "a warm cache re-downloaded"
