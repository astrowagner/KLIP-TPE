"""Synthetic-image tests for the verification stack (verify / candidates / param_verify)."""
import os

import numpy as np
import pytest

from klip_tpe.metrics import Source, source_xy, star_center
from klip_tpe.optimizers import History
from klip_tpe.space import Param, SearchSpace
from klip_tpe import verify as V
from klip_tpe import candidates as C
from klip_tpe import param_verify as PV

FWHM = 4.0
PX = 0.05
N = 96


def _gauss(ny, nx, x0, y0, fwhm, amp):
    yy, xx = np.mgrid[0:ny, 0:nx]
    sg = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sg ** 2)


def _noise_stack(nn, seed, sigma=1.0, rmask=1.5 * FWHM, correlated=True):
    """Per-night noise images; ``correlated=True`` smooths white noise on the FWHM
    scale (speckle-like residuals, the regime of KLIP output) and renormalises to
    unit per-pixel std.  The star region is NaN as in a real reduction."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    st = rng.normal(0, 1.0, (nn, N, N))
    if correlated:
        st = np.stack([gaussian_filter(f, FWHM / 2.3548, mode="wrap") for f in st])
        st /= st.std(axis=(1, 2), keepdims=True)
    st *= sigma
    cx, cy = star_center((N, N))
    yy, xx = np.mgrid[0:N, 0:N]
    st[:, np.hypot(xx - cx, yy - cy) < rmask] = np.nan
    return st


def _planted_stack(nn, rho, theta, peak_snr, seed=1, correlated=True):
    """Noise nights plus a Gaussian source whose *matched-filter* S/N in the
    combined image is ``peak_snr`` (calibrated numerically on the noise itself)."""
    st = _noise_stack(nn, seed, correlated=correlated)
    cx, cy = star_center((N, N))
    xs, ys = source_xy(rho, theta, PX, cx, cy)
    from klip_tpe.metrics import gaussian_kernel
    k = gaussian_kernel(FWHM)
    comb = np.nanmean(st, axis=0)
    A = V.mf_convolve(comb, FWHM)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - cx, yy - cy)
    band = (np.abs(rr - rho / PX) <= FWHM) & np.isfinite(comb)
    sig_mf = float(np.std(A[band], ddof=1))
    g = _gauss(k.shape[0], k.shape[1], (k.shape[1] - 1) / 2, (k.shape[0] - 1) / 2, FWHM, 1.0)
    resp = (k * g).sum()
    amp = peak_snr * sig_mf / resp
    src = _gauss(N, N, float(xs[0]), float(ys[0]), FWHM, amp)
    return st + src[None]


# ----------------------------------------------------------------------------
def test_t_statistics_roundtrip():
    assert V.t_fap(0.0, 20) == 0.5
    tau = V.t_thresh(1e-3, 20)
    assert abs(V.t_fap(tau, 20) - 1e-3) < 1e-9
    # large-n limit -> Gaussian 5 sigma
    assert abs(V.t_fap(5.0, 10 ** 6) - 2.87e-7) / 2.87e-7 < 0.02
    assert V.t_thresh(0.0, 10) == 1e30
    # contrast limit scales linearly with contrast / inversely with S/N
    a = V.contrast_limit(1e-3, 0.5, [1e-4], [10.0], [20])
    b = V.contrast_limit(1e-3, 0.5, [2e-4], [20.0], [20])
    assert abs(a - b) < 1e-12


def test_verify_planted_source_detected():
    nn = 6
    rho, theta = 0.8, 40.0
    stack = _planted_stack(nn, rho, theta, 8.0)
    rows, maps = V.verify_candidates(stack, list(range(nn)), [(rho, theta)], FWHM, PX,
                                     n_boot=100, rng=np.random.default_rng(3), return_maps=True)
    r = rows[0]
    assert r["kind"] == "candidate"
    assert r["snr"] > 6.0
    assert r["fap"] < 2.87e-7
    assert r["detect"] is True
    # per-PIXEL night-STIM of an ~1 sigma/px/night source is modest; its annulus rank is what matters
    assert r["stim"] > 0.5 and r["stim_fap"] < 0.1
    assert r["boot_sig"] > 3.0
    assert r["persistence"] > 2.0
    assert set(r["subsets"]) == set(V.SUBSET_NAMES)
    assert maps["stim"].shape == (N, N)


def test_verify_empty_field_and_report(tmp_path):
    nn = 5
    stack = _noise_stack(nn, 11)
    inj = _planted_stack(nn, 1.0, 200.0, 8.0, seed=11)
    rows = V.verify_candidates(stack, None, [(0.7, 120.0)], FWHM, PX, n_boot=0,
                               injected=(inj, [(1.0, 200.0, 1e-4)]))
    cand = [r for r in rows if r["kind"] == "candidate"][0]
    injr = [r for r in rows if r["kind"] == "injection"][0]
    assert not cand["detect"]
    assert np.isnan(cand["boot_sig"])
    assert injr["snr"] > 5.0 and injr["contrast"] == 1e-4
    lim = V.limit_tables([injr], FWHM, PX)
    assert lim is not None and len(lim["bins"]) == 1
    b = lim["bins"][0]
    assert b["n_inj"] == 1 and np.isfinite(b["c_tap"][1]) and b["c_tap"][2] > b["c_tap"][0]
    p = tmp_path / "verify_report.txt"
    txt = V.write_verify_table(str(p), rows)
    assert p.exists() and "TABLE 1" in txt and "decision" in txt
    V.write_curve_table(str(tmp_path / "verify_curve.txt"), lim, "test")
    assert (tmp_path / "verify_curve.txt").exists()


# ----------------------------------------------------------------------------
def test_candidates_tier_A_when_present_all_nights(tmp_path):
    nn = 6
    rho, theta = 0.9, 300.0
    stack = _planted_stack(nn, rho, theta, 8.0, seed=1)
    rows = C.find_candidates(None, stack, FWHM, PX, night_ids=list(range(nn)), verify_top=1,
                             verify_kwargs={"n_boot": 20})
    assert rows, "no candidates found"
    top = rows[0]
    assert top["tier"] == "A"
    assert abs(top["rho"] - rho) < 1.5 * PX
    assert abs(((top["theta"] - theta) + 180) % 360 - 180) < 6.0
    assert top["n_nights"] >= 3 and top["psf_corr"] > 0.5 and top["snr"] >= 5
    assert top["fap"] < 1e-3
    assert "verify" in top and top["verify"]["detect"]
    # only one tier-A peak
    assert sum(r["tier"] == "A" for r in rows) == 1
    txt = C.write_candidates(str(tmp_path / "candidates.txt"), rows)
    assert "rank" in txt and (tmp_path / "candidates.txt").exists()
    # persistence map term: a zero map suppresses the score, a high map keeps it
    hi = C.score_candidates([dict(r) for r in rows], nn, np.full((N, N), 10.0))
    lo = C.score_candidates([dict(r) for r in rows], nn, np.zeros((N, N)))
    assert lo[0]["score"] < hi[0]["score"]


def test_candidates_empty_field_no_tier_A():
    nn = 6
    stack = _noise_stack(nn, 21)
    rows = C.find_candidates(None, stack, FWHM, PX)
    assert all(r["tier"] != "A" for r in rows)
    assert all(r["flag"] == "" for r in rows)


def test_detect_peaks_and_psfcorr():
    m = np.zeros((N, N))
    cx, cy = star_center((N, N))
    m[int(cy) + 20, int(cx) + 5] = 6.0
    m[int(cy) + 20, int(cx) + 6] = 5.0     # inside the suppression disc
    m[int(cy) - 25, int(cx) - 10] = 3.0
    px, py, ps = C.detect_peaks(m, 2.0, max(0.9 * FWHM, 2), 1.2 * FWHM, N / 2 - FWHM)
    assert list(ps) == [6.0, 3.0]
    img = _gauss(N, N, 60, 30, FWHM, 5.0)
    assert C.psf_correlation(img, 60, 30, FWHM) > 0.85   # <1: stamp median- vs template mean-subtracted (IDL)


# ----------------------------------------------------------------------------
def test_param_verify_fake_reduce(tmp_path):
    space = SearchSpace([Param("k_klip", 1, 30, kind="int"), Param("bin", 1, 20, kind="float")])
    rng = np.random.default_rng(7)
    hist = History(ndim=2)
    for i in range(30):
        x = space.random(rng)
        hist.append(x, float(rng.random() * 5), {"phase": "warmup" if i < 10 else "tpe"})
    # near-duplicates of the best must be dropped by the diversity floor
    best = hist.best()[0]
    hist.append(hist.X[best] + np.array([0.0, 0.05]), hist.y[best] - 0.01, {"phase": "tpe"})
    sel = PV.select_configs(hist, space, n_pv=8, pv_divmin=0.05, n_init=10)
    assert 2 <= len(sel) <= 8 and all(e >= 10 for e in sel)
    for a in sel:
        for b in sel:
            if a != b:
                d = np.mean(np.abs(hist.X[a] - hist.X[b]) / (space.hi - space.lo))
                assert d >= 0.05

    calls = []

    def reduce_fn(x, sources):
        calls.append(x.copy())
        seed = int(abs(hash(tuple(np.round(x, 3)))) % 2 ** 31)
        r = np.random.default_rng(seed)
        clean = r.normal(0, 1, (N, N))
        cx, cy = star_center((N, N))
        yy, xx = np.mgrid[0:N, 0:N]
        clean[np.hypot(xx - cx, yy - cy) < 1.5 * FWHM] = np.nan
        inj = clean.copy()
        for s in sources:
            xs, ys = source_xy(s.rho, s.theta, PX, cx, cy)
            inj += _gauss(N, N, float(xs[0]), float(ys[0]), FWHM, 40.0 * s.contrast / 1e-4)
        return clean, inj

    out_dir = tmp_path / "param_verify"
    res = PV.param_verify(hist, space, reduce_fn, FWHM, PX, rlo=8.0, rhi=30.0, out_dir=str(out_dir),
                          ccal=1e-4, n_pv=6, n_init=10, nsrc=4, rng=np.random.default_rng(1), log=lambda s: None)
    assert res is not None and len(calls) == len(res["eval_indices"]) >= 2
    for kind in PV.PV_KINDS:
        assert (out_dir / f"paramverify_{kind}_01.fits").exists()
        assert res[kind].shape == (N, N)
    assert (out_dir / "paramverify_calib_01.txt").exists()
    txt = (out_dir / "paramverify_calib_01.txt").read_text()
    assert "n_configs" in txt and txt.count("\nCC ") == 12
    # injected sources are recovered consistently, the clean field is not
    assert np.nanmax(res["recovery"]) > 0.5
    assert res["calib"]["true_median"] > res["calib"]["null_p95"]
    assert np.nanmean(res["detfrac"]) < 0.1
    assert np.isfinite(res["Kc"]) and res["Kc"] > 0
    assert np.all(np.isfinite(res["curve"]["c5"]))
    # stitched maps
    st = PV.stitch_param_maps([res, res], [(8.0, 20.0), (20.0, 30.0)], out_dir=str(tmp_path))
    assert (tmp_path / "paramverify_stim_stitched.fits").exists()
    cx, cy = star_center((N, N))
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - cx, yy - cy)
    assert np.all(np.isnan(st["stim"][rr > 31.5]))
    inner = st["stim"][(rr > 10) & (rr < 18)]
    assert np.isfinite(inner).mean() > 0.9

    # one-argument reduce_fn is accepted with explicit fixed sources
    fixed = PV.fixed_injection_set(3, 8 * PX, 30 * PX, FWHM, PX, np.random.default_rng(2), 1e-4)
    res2 = PV.param_verify(hist, space, lambda x: reduce_fn(x, fixed), FWHM, PX, rlo=8.0, rhi=30.0,
                           out_dir=str(tmp_path / "pv2"), ccal=1e-4, n_pv=3, n_init=10, fixed_sources=fixed,
                           log=lambda s: None)
    assert res2 is not None and len(res2["sources"]) == 3


# ----------------------------------------------------------------------------
# live comparison against an IDL run (tooling; data-free)
# ----------------------------------------------------------------------------
def test_idl_compare_tooling(tmp_path):
    import json
    import numpy as np
    from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig
    from klip_tpe.idl_compare import LiveComparison, agreement_stats, matched_run_config, read_step_setup
    from klip_tpe.instruments import near
    from conftest import build_synthetic_run

    # a fake IDL run directory: run_setup.txt + a 4-row log + one exact-position step file
    idl = tmp_path / "run_20260101_000000"
    (idl / "annulus01").mkdir(parents=True)
    (idl / "run_setup.txt").write_text(
        "# KLIP-TPE run setup\nann_edges (px)     : 8.00, 30.00\nn_iter / n_init    : 6 / 3\n"
        "n_valid / n_top    : 2 / 1\ngamma/ncand/explore: 0.250 / 48 / 0.15\nseed_best_frac     : 0.00\n"
        "tpe_mv (density)   : 0\np_local / n_elite  : 0.15 / 5\ncmax_cal (ceiling) :  6.00E-05\n"
        "use_contrast       :  2.00E-02\nnights (seq_values): 1,2\nseed / bench_tag   : 4242 / none\n"
        "\n# searched parameters\n  bin_a         : [5.000, 30.000]  int=1\n")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=4, n_init=2)
    names = [p.name for p in space.params]
    hdr = "# annulus iter " + " ".join(names) + "   k_klip   medSNR\n"
    rows = []
    rng = np.random.default_rng(0)
    for it in range(1, 5):
        x = space.random(rng)
        rows.append(f"  1  {it:4d} " + " ".join(f"{v:9.3f}" for v in x) + f"   {int(x[names.index('k_klip_n1')])}   {2.0 + 0.3 * it:8.3f}\n")
    (idl / "optimize_tpe_results.txt").write_text(hdr + "".join(rows))
    (idl / "annulus01" / "eval0002_setup.txt").write_text(
        "# KLIP-TPE step setup\nphase         : eval\nannulus       : 1\nstep          : 2\nmedian_SNR    : 2.600\n"
        "# injected sources:  rho_arcsec   PA_deg     contrast\n       0.900       10.00     2.000E-02\n"
        "       0.900      190.00     2.000E-02\n")
    st = read_step_setup(str(idl / "annulus01" / "eval0002_setup.txt"))
    assert st["step"] == 2 and len(st["sources"]) == 2 and st["sources"][1].theta == 190.0

    mcfg, info = matched_run_config(str(idl / "run_setup.txt"), seed=7)
    assert mcfg.ann_edges == [8.0, 30.0] and mcfg.n_iter == 6 and mcfg.n_init == 3
    assert mcfg.validation.n_valid == 2 and mcfg.calibration.forced == [0.02] and mcfg.blocks == "univariate"
    assert info["nights"] == [1, 2] and info["idl_seed"] == 4242

    cfg = RunConfig(ann_edges=[8, 30], n_iter=1, n_init=1, seed=1, save_fits=False, fm_curve=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[0.02]),
                    write_setup_files=False)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "rn"), log=lambda s: None)
    lc = LiveComparison(str(idl), r, str(tmp_path / "cmp"), log=lambda s: None)
    assert np.isclose(lc.contrast, 0.02)
    n = lc.sync(max_new=3)
    assert n == 3 and len(lc.rows) == 3 and lc.rows[1]["exact_positions"] and not lc.rows[0]["exact_positions"]
    assert lc.rows[1]["sources"][0][1] == 10.0
    n2 = lc.sync()                                   # incremental: only the 4th row is new
    assert n2 == 1 and len(lc.rows) == 4
    lc2 = LiveComparison(str(idl), r, str(tmp_path / "cmp"), log=lambda s: None)   # reload from jsonl
    assert len(lc2.rows) == 4 and lc2.sync() == 0
    s = agreement_stats([r_["idl_score"] for r_ in lc2.rows], [r_["py_score"] for r_ in lc2.rows])
    assert s["n"] == 4 and "pearson" in s
    assert (tmp_path / "cmp" / "compare.txt").exists() and (tmp_path / "cmp" / "compare_stats.json").exists()
