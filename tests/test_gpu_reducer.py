"""GPUKLIPReducer against KLIPReducer.reduce on two synthetic NEAR-like nights.

The chain of ``klip_gpu`` runs on NumPy without CuPy (the same arithmetic on the host), which
tests the plumbing and the KLIP formulation on any machine; the card tests run where CuPy finds a
device and are skipped elsewhere.  Each case compares the image with the CPU reducer's in units of
the annulus noise, the NaN footprint pixel for pixel, the meta keys the runner reads, and the same
job twice; then the CPU fallback for a KLIP-FM request, pickling, ``wrap_partitions`` and ``warm()``.

    python -m pytest -q tests/test_gpu_reducer.py          # NumPy; the card too where there is one
    python test_gpu_reducer.py [--cupy]                    # as a script, with the timings

Written in the NEAR2 GPU session, 2026-10-31 (bench/gpu_reducer/ on patrick).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(HERE, "gpu_reducer.py")):       # the copy beside this file (bench/gpu_reducer/)
    sys.path.insert(0, HERE)
    from gpu_reducer import GPUKLIPReducer, wrap_partitions  # noqa: E402
else:                                                           # the package (tests/ in the repo)
    from klip_tpe.gpu_reducer import GPUKLIPReducer, wrap_partitions  # noqa: E402
from klip_tpe.metrics import Source  # noqa: E402
from klip_tpe.reducer import KLIPReducer, ReductionRequest  # noqa: E402

A_IN, A_OUT = 8.77, 26.32                                       # the annulus (px), padded on both sides
SOURCES = [Source(0.72, 33.0, 6e-5), Source(0.80, 150.0, 6e-5), Source(0.88, 260.0, 6e-5)]
TOL = 1e-3                                                      # max |difference| / annulus noise

# (name, night, params, sources, compact, batch_sectors)
CASES: List[Tuple[str, str, Dict[str, Any], Optional[list], int, bool]] = [
    ("bin20 4 sectors", "n1", dict(bin=20, n_ang=4, filter=7, angsep=0.2, anglemax=101.0, corr_thresh=0.3,
                                   noise_max=2.8, coronoise_max=3.0, k_klip=10), SOURCES, 0, True),
    ("bin5 2 sectors clean", "n1", dict(bin=5, n_ang=2, filter=12, angsep=1.9, anglemax=26.0, corr_thresh=0.9,
                                        noise_max=1.45, coronoise_max=1.47, k_klip=6), None, 0, True),
    ("auto-fast k>=n", "n1", dict(bin=26, n_ang=4, filter=6, angsep=0.0, anglemax=150.0, corr_thresh=0.0,
                                  noise_max=3.0, coronoise_max=3.0, k_klip=16), SOURCES, 0, True),
    ("median comb, no post hp", "n1", dict(bin=12, n_ang=3, filter=9, angsep=0.3, anglemax=80.0, corr_thresh=0.0,
                                           noise_max=3.0, coronoise_max=3.0, k_klip=8, comb_type="median",
                                           post_hp=False), SOURCES, 0, True),
    ("zone ext, hp binned", "n1", dict(bin=15, n_ang=2, filter=9, angsep=0.1, anglemax=120.0, corr_thresh=0.0,
                                       noise_max=3.0, coronoise_max=3.0, k_klip=5, zone_in=3.0, zone_out=4.0,
                                       hp_order="binned"), SOURCES, 0, True),
    ("no filter, no bin", "n2", dict(bin=1, n_ang=1, filter=0, angsep=0.5, anglemax=60.0, corr_thresh=0.0,
                                     noise_max=3.0, coronoise_max=3.0, k_klip=4), None, 0, True),
    ("starved edges dropped", "n1", dict(bin=5, n_ang=2, filter=9, angsep=0.1, anglemax=20.0, corr_thresh=0.0,
                                         noise_max=3.0, coronoise_max=3.0, k_klip=6), SOURCES, 0, True),
    ("all starved, few refs", "n2", dict(bin=4, n_ang=2, filter=9, angsep=0.1, anglemax=5.0, corr_thresh=0.0,
                                         noise_max=3.0, coronoise_max=3.0, k_klip=3), SOURCES, 0, True),
    ("compact 16", "n1", dict(bin=20, n_ang=4, filter=7, angsep=0.2, anglemax=101.0, corr_thresh=0.3,
                              noise_max=2.8, coronoise_max=3.0, k_klip=10), SOURCES, 16, True),
    ("sectors one by one", "n1", dict(bin=20, n_ang=4, filter=7, angsep=0.2, anglemax=101.0, corr_thresh=0.3,
                                      noise_max=2.8, coronoise_max=3.0, k_klip=10), SOURCES, 0, False),
]
META_KEYS = ("n_kept", "n_binned", "k_effective", "n_dropped", "fast", "inrad", "outrad", "n_starved", "drop_enabled")


# ----------------------------------------------------------------------------- the synthetic nights
def synthetic_night(name: str, n: int, ny: int = 150, seed: int = 1) -> KLIPReducer:
    """A small NEAR-like night: a smooth halo, one quasi-static speckle field scaled from frame to
    frame, white noise, PA from -60 to 70 deg, frame tags around 0.95 / 1.0 / 1.0, and a Gaussian
    PSF model at NEAR's FWHM.  Deterministic in ``seed``."""
    from klip_tpe.injection import GaussianPSF
    from klip_tpe.instruments.near import NEAR
    from klip_tpe.reducer import Dataset
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:ny]
    rr = np.hypot(xx - (ny - 1) / 2.0, yy - (ny - 1) / 2.0)
    halo = (3000.0 * np.exp(-rr / 6.0) + 40.0 * np.exp(-rr / 25.0)).astype(np.float32)
    speck = rng.normal(size=(ny, ny)).astype(np.float32) * 6.0
    cube = (halo[None] + speck[None] * np.linspace(0.8, 1.2, n)[:, None, None]
            + rng.normal(size=(n, ny, ny)).astype(np.float32) * 2.0)
    angles = np.linspace(-60.0, 70.0, n) + rng.normal(scale=0.05, size=n)
    tags = {"corrs": np.clip(rng.normal(0.95, 0.03, n), 0, 1), "noises": rng.normal(1.0, 0.1, n),
            "coronoise": rng.normal(1.0, 0.1, n)}
    ds = Dataset(cube, angles, tags, texp=float(n), name=name)
    model = GaussianPSF(NEAR.fwhm_px, star_flux=2.0e5)
    red = KLIPReducer(ds, pxscale=NEAR.pxscale, lam_m=NEAR.lam_klip, diam_m=NEAR.diam, injection_model=model,
                      fallback_model=model, truenorth=NEAR.truenorth, fwhm_px=NEAR.fwhm_px, zone_pad=2.0,
                      outrad_cap=70.0, defaults={"fast": False, "n_min_ref": 10, "comb_type": "nwadi"})
    red.name = f"near_{name}"
    return red


_NIGHTS: Dict[str, KLIPReducer] = {}


def nights() -> Dict[str, KLIPReducer]:
    """n1 (300 frames) and n2 (120 frames), made once per process."""
    if not _NIGHTS:
        _NIGHTS["n1"] = synthetic_night("n1", 300, seed=1)
        _NIGHTS["n2"] = synthetic_night("n2", 120, seed=2)
    return _NIGHTS


_GPUS: Dict[Tuple[str, str], GPUKLIPReducer] = {}


def gpu(night: str, backend: str) -> GPUKLIPReducer:
    """The night's GPUKLIPReducer on ``backend`` ("numpy" or "cupy"), made once per process (so the
    context and the cube on the card are made once)."""
    key = (night, backend)
    if key not in _GPUS:
        g = GPUKLIPReducer.from_cpu(nights()[night])
        g.backend_name = backend
        _GPUS[key] = g
    return _GPUS[key]


def has_card() -> bool:
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- one case
def annulus_noise(img: np.ndarray, inrad: float, outrad: float) -> Tuple[float, np.ndarray]:
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - nx / 2.0, yy - ny / 2.0)
    m = (r >= inrad) & (r <= outrad) & np.isfinite(img)
    return float(np.std(img[m])), m


def run_case(name: str, night: str, params: Dict[str, Any], sources, compact: int, batch_sectors: bool,
             backend: str) -> Dict[str, Any]:
    """The case on the CPU reducer and on the GPU reducer (``backend``), twice on the latter."""
    cpu, g = nights()[night], gpu(night, backend)
    g.compact, g.batch_sectors = int(compact), bool(batch_sectors)
    p = dict(params, inrad=A_IN, outrad=A_OUT, pad_in=True, pad_out=True)
    req = ReductionRequest(p, sources, False, f"t_{night}", threads=2)
    assert g.supports(req), name
    cpu._inj_cache = {}
    t0 = time.perf_counter(); rc = cpu.reduce(req); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); rg = g.reduce(req); tg = time.perf_counter() - t0
    t0 = time.perf_counter(); rg2 = g.reduce(req); tg2 = time.perf_counter() - t0
    sig, m = annulus_noise(rc.image, A_IN, A_OUT)
    both = m & np.isfinite(rg.image)
    d = np.abs(rg.image - rc.image)[both]
    return {"name": name, "dmax": float(d.max() / sig) if d.size else float("nan"),
            "drms": float(np.sqrt(np.mean(d ** 2)) / sig) if d.size else float("nan"),
            "same_nan": bool(np.array_equal(np.isfinite(rg.image), np.isfinite(rc.image))),
            "twice": bool(np.array_equal(rg.image, rg2.image, equal_nan=True)),
            "meta_differs": sorted(k for k in META_KEYS if rc.meta.get(k) != rg.meta.get(k)),
            "shape_same": rg.image.shape == rc.image.shape and rg.image.dtype == np.float32,
            "device": rg.meta.get("device"), "n_dropped": rg.meta.get("n_dropped"),
            "nref_used": rg.meta.get("nref_used"), "tc": tc, "tg": tg, "tg2": tg2}


def check(r: Dict[str, Any]) -> None:
    assert r["shape_same"], r["name"]
    assert r["device"] == "gpu", r["name"]
    assert r["same_nan"], f"{r['name']}: the NaN footprint differs from the CPU reducer's"
    assert r["twice"], f"{r['name']}: the same job twice gave different bits"
    assert not r["meta_differs"], f"{r['name']}: meta differs in {r['meta_differs']}"
    assert r["dmax"] < TOL, f"{r['name']}: max |d| / sigma {r['dmax']:.1e}"


def fm_request():
    p = dict(CASES[0][2], inrad=A_IN, outrad=A_OUT)
    return ReductionRequest(p, SOURCES, False, "fm", fm_sources=SOURCES, threads=2)


# ----------------------------------------------------------------------------- pytest
try:
    import pytest
except ImportError:                                             # the script needs none
    pytest = None

if pytest is not None:
    _ids = [c[0] for c in CASES]
    card = pytest.mark.skipif(not has_card(), reason="CuPy or a CUDA device is missing")

    @pytest.mark.parametrize("case", CASES, ids=_ids)
    def test_numpy_backend_matches_cpu(case):
        check(run_case(*case, backend="numpy"))

    def test_starved_and_few_reference_paths_taken():
        """The two cases named for them do take the paths (otherwise they test nothing)."""
        r = run_case(*CASES[6], backend="numpy")
        assert r["n_dropped"] and r["n_dropped"] > 0, "no starved target dropped"
        r = run_case(*CASES[7], backend="numpy")
        assert r["n_dropped"] == 0
        assert r["nref_used"] is not None and min(r["nref_used"]) >= 4, "the few-reference fallback did not widen"

    def test_fm_request_falls_back_to_cpu():
        g = gpu("n1", "numpy")
        req = fm_request()
        assert not g.supports(req)
        r = g.reduce(req)
        assert r.meta.get("device") == "cpu" and r.fm_image is not None
        g.cpu_fallback = False
        try:
            with pytest.raises(ValueError):
                g.reduce(req)
        finally:
            g.cpu_fallback = True

    def test_pickles_without_device_state():
        g = gpu("n1", "numpy")
        g2 = pickle.loads(pickle.dumps(g))
        assert isinstance(g2, GPUKLIPReducer) and g2._gpu is None and g2._zones == {}
        assert g2.supports(ReductionRequest(dict(CASES[0][2], inrad=A_IN, outrad=A_OUT), SOURCES, False, "p"))

    def test_wrap_partitions():
        from klip_tpe.reducer import PartitionedReducer
        pr = PartitionedReducer(dict(nights()), weighting="equal", max_workers=1, pool="threads")
        w = wrap_partitions(pr, compact=16)
        assert set(w) == set(pr.reducers)
        for pid, g in w.items():
            assert isinstance(g, GPUKLIPReducer) and g.compact == 16
            assert g.partition_weight() == pr.reducers[pid].partition_weight()
            assert g.data is pr.reducers[pid].data

    def test_warm_numpy_backend_keys():
        d = gpu("n2", "numpy").warm()
        for k in ("name", "driver", "reducer", "eigh", "compact", "batch_sectors", "cube_gb", "pid"):
            assert k in d, k

    @card
    def test_warm_on_the_card():
        d = gpu("n1", "cupy").warm()
        assert d["name"] and isinstance(d["name"], str)
        assert d["driver"] is None or d["driver"][0].isdigit()
        for k in ("cuda_driver_api", "cuda_runtime", "cupy", "memory_used_gb", "memory_total_gb"):
            assert k in d, k

    @card
    @pytest.mark.parametrize("case", CASES, ids=_ids)
    def test_card_matches_cpu(case):
        check(run_case(*case, backend="cupy"))


# ----------------------------------------------------------------------------- script
def main() -> int:
    backend = "cupy" if "--cupy" in sys.argv else "numpy"
    print(f"GPUKLIPReducer from {sys.modules[GPUKLIPReducer.__module__].__file__}, backend {backend}")
    if backend == "cupy":
        print("warm():", gpu("n1", "cupy").warm())
    worst, ok = 0.0, True
    for case in CASES:
        r = run_case(*case, backend=backend)
        try:
            check(r)
            verdict = ""
        except AssertionError as exc:
            ok, verdict = False, f"  FAIL: {exc}"
        worst = max(worst, r["dmax"])
        print(f"{r['name']:26s} max|d|/sigma {r['dmax']:.1e}  rms/sigma {r['drms']:.1e}  nan pattern same {r['same_nan']}  "
              f"twice-identical {r['twice']}  meta differs {r['meta_differs'] or '-'}  cpu {r['tc'] * 1000:.0f} ms  "
              f"dev {r['tg'] * 1000:.0f}/{r['tg2'] * 1000:.0f} ms  device {r['device']}  dropped {r['n_dropped']}{verdict}")
    g = gpu("n1", backend)
    req = fm_request()
    assert not g.supports(req)
    r = g.reduce(req)
    print("fm request -> device", r.meta.get("device"), "fm image", None if r.fm_image is None else r.fm_image.shape)
    print("stats", g.stats)
    print("WORST max|d|/sigma", f"{worst:.1e}", "PASS" if ok and worst < TOL else "FAIL")
    return 0 if ok and worst < TOL else 1


if __name__ == "__main__":
    sys.exit(main())
