import numpy as np

from klip_tpe.klip import KLIPParams, klip_annular
from klip_tpe.parallel import cpu_count, describe, resolve_workers


def test_resolve_workers():
    n = cpu_count()
    assert resolve_workers("auto") == n == resolve_workers(None) == resolve_workers(0) == resolve_workers("max")
    assert resolve_workers(3) == 3 and resolve_workers("4") == 4
    assert resolve_workers(-1) == max(n - 1, 1)
    assert "workers on" in describe(8, 6)


def test_threaded_targets_identical():
    """The per-target KLIP path gives bit-identical residuals with 1 and 4 threads
    (targets write disjoint frames; BLAS pinned to one thread inside)."""
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(40, 48, 48)).astype(np.float32)
    ang = np.linspace(-25, 25, 40)
    kw = dict(k_klip=6, inrad=4, outrad=20, n_ang=2, angsep=0.5, anglemax=30)
    a, ia = klip_annular(cube, ang, KLIPParams(threads=1, **kw), 4.0)
    b, ib = klip_annular(cube, ang, KLIPParams(threads=4, **kw), 4.0)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=0)
    assert ia["n_dropped"] == ib["n_dropped"] and np.array_equal(ia["nref_used"], ib["nref_used"])
    # k-scan path too
    kw["k_scan"] = True
    a, _ = klip_annular(cube, ang, KLIPParams(threads=1, **kw), 4.0)
    b, _ = klip_annular(cube, ang, KLIPParams(threads=3, **kw), 4.0)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=0)


def test_partitioned_reducer_thread_budget():
    """max_workers='auto' resolves to the core count, the per-partition thread share is
    passed down to the KLIP engine, and the image equals the serial reduction."""
    from klip_tpe import GaussianPSF, KLIPReducer, PartitionedReducer
    from klip_tpe.space import Config
    from klip_tpe.synthetic import synthetic_klip_dataset
    px = 0.05
    lod = 3.89 * px / 206265
    reds = {f"n{i}": KLIPReducer(synthetic_klip_dataset(nframes=30, size=48, pa_span=50, seed=i), px, lod * 8.2, 8.2,
                                 injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=22) for i in (1, 2)}
    par = PartitionedReducer(reds, max_workers="auto")
    assert par.max_workers == cpu_count()
    ser = PartitionedReducer(reds, max_workers=1)
    params = {"k_klip": 4, "inrad": 6, "outrad": 18, "bin": 2, "filter": 0, "angsep": 0.5, "anglemax": 35}
    c = Config(params=params, per_partition={k: dict(params) for k in reds}, selected=list(reds), x=np.zeros(0))
    a = par.reduce_config(c, tag="p")
    b = ser.reduce_config(c, tag="s")
    assert np.allclose(np.nan_to_num(a.image), np.nan_to_num(b.image), atol=1e-5)


def test_process_pool_matches_serial():
    """Forked worker processes give the same combined image as the serial reduction and
    survive concurrent map() calls (the runner's injected + clean pair)."""
    import sys
    import concurrent.futures as cf
    from klip_tpe import GaussianPSF, KLIPReducer, PartitionedReducer
    from klip_tpe.space import Config
    from klip_tpe.synthetic import synthetic_klip_dataset
    if sys.platform.startswith("win"):
        return
    px = 0.05
    lod = 3.89 * px / 206265
    reds = {f"n{i}": KLIPReducer(synthetic_klip_dataset(nframes=30, size=48, pa_span=50, seed=i), px, lod * 8.2, 8.2,
                                 injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=22) for i in (1, 2)}
    par = PartitionedReducer(reds, max_workers=4, pool="processes")
    assert par.start_workers() == "processes"
    ser = PartitionedReducer(reds, max_workers=1)
    params = {"k_klip": 4, "inrad": 6, "outrad": 18, "bin": 2, "filter": 0, "angsep": 0.5, "anglemax": 35}
    c = Config(params=params, per_partition={k: dict(params) for k in reds}, selected=list(reds), x=np.zeros(0))
    with cf.ThreadPoolExecutor(2) as ex:
        fa = ex.submit(par.reduce_config, c, tag="a")
        fb = ex.submit(par.reduce_config, c, tag="b")
        a, b = fa.result(), fb.result()
    s = ser.reduce_config(c, tag="s")
    assert np.allclose(np.nan_to_num(a.image), np.nan_to_num(s.image), atol=1e-5)
    assert np.allclose(np.nan_to_num(b.image), np.nan_to_num(s.image), atol=1e-5)
    par.close()
