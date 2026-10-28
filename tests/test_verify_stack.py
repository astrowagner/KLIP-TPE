"""Synthetic-image tests for the verification stack (verify / candidates / param_verify)."""
import os
import warnings

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
    assert "leaves out the other sources and the known ones (1.5 FWHM keep-out)" in txt


def _odd_noise(nn, seed, n):
    """``_noise_stack`` on an odd ``n x n`` frame, so the star sits on a pixel centre and a
    source at an integer offset sits on a pixel."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    st = np.stack([gaussian_filter(f, FWHM / 2.3548, mode="wrap") for f in rng.normal(0, 1.0, (nn, n, n))])
    st /= st.std(axis=(1, 2), keepdims=True)
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    st[:, np.hypot(xx - c, yy - c) < 1.5 * FWHM] = np.nan
    return st


@pytest.mark.parametrize("dx, dy, parity", [(8, 15, 0), (12, 16, 1)])    # 17 px: nap 26; 20 px: nap 31
def test_verify_injections_pair_at_one_radius(dx, dy, parity):
    """Two injections 180 deg apart at one radius each get the S/N they get injected alone, to
    the noise.  With ``nap`` even a ring sample lands on the other source, with ``nap`` odd two
    sit beside it; neither counts now (NEAR2 note, 2026-10-20)."""
    from klip_tpe.metrics import gaussian_kernel
    n, nn, snr = 81, 4, 15.0
    c = (n - 1) / 2
    r = float(np.hypot(dx, dy))
    pa = float(np.degrees(np.arctan2(dy, dx))) - 90.0
    A, B = (r * PX, pa % 360.0, 1e-3), (r * PX, (pa + 180.0) % 360.0, 1e-3)
    k = gaussian_kernel(FWHM)
    resp = (k * _gauss(k.shape[0], k.shape[1], (k.shape[1] - 1) / 2, (k.shape[0] - 1) / 2, FWHM, 1.0)).sum()
    yy, xx = np.mgrid[0:n, 0:n]
    band = np.abs(np.hypot(xx - c, yy - c) - r) <= FWHM
    alone_r, ring_r, before_r, stim, stim_before = [], [], [], [], []
    for seed in range(8):
        st = _odd_noise(nn, seed, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # the masked star: all-NaN columns
            comb = np.nanmean(st, axis=0)
        amp = snr * float(np.std(V.mf_convolve(comb, FWHM)[band & np.isfinite(comb)], ddof=1)) / resp
        ga, gb = _gauss(n, n, c + dx, c + dy, FWHM, amp), _gauss(n, n, c - dx, c - dy, FWHM, amp)
        pair, pm = V.verify_injections(st + (ga + gb)[None], [A, B], FWHM, PX)
        for p, g, s, o in zip(pair, (ga, gb), (A, B), (B, A)):
            assert int(np.floor(2 * np.pi * np.hypot(p["x"] - c, p["y"] - c) / FWHM)) % 2 == parity
            alone = V.verify_injections(st + g[None], [s], FWHM, PX)[0][0]
            # the same ring: the other source's position left out of the lone source's ring too
            ring = V.verify_injections(st + g[None], [s], FWHM, PX, known=[o[:2]])[0][0]
            alone_r.append(p["snr"] / alone["snr"])
            ring_r += [p["snr"] / ring["snr"]] + [p["subsets"][q] / ring["subsets"][q] for q in V.SUBSET_NAMES[1:]]
            stim.append(p["stim_fap"] - alone["stim_fap"])
            # what the ring held before 2026-10-20, and holds in the IDL: the other source in it
            before = V.snr_fap(pm["subsets"]["combined"], s[0], s[1], PX, FWHM, 1.5 * FWHM, 1.5,
                               mfmap=pm["mf"]["combined"])["snr"]
            before_r.append(before / alone["snr"])
            stim_before.append(V.stim_fap(pm["stim"], p["x"], p["y"], r, FWHM)[1] - p["stim_fap"])
    assert abs(np.mean(alone_r) - 1.0) < 0.1           # the pair's rings lost a few noise samples
    assert np.max(np.abs(np.asarray(ring_r) - 1.0)) < 0.1    # left: the other source's wings past 1.5 FWHM
    assert np.max(np.abs(stim)) < 2e-3
    assert np.mean(before_r) < 0.6 and np.mean(stim_before) > 0


def test_solved_contrasts_scale_the_source_not_the_noise():
    """Solved on clean + a (injected - clean), a source's contrasts do not depend on the
    brightness it was injected at, where the response is linear (here there is no KLIP).
    The limit scaled from the S/N at the injected contrast does, since it scales the noise at
    the source too (NEAR2 note of 2026-10-23, revised)."""
    from klip_tpe.metrics import gaussian_kernel
    n = 81
    c = (n - 1) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clean = _odd_noise(1, 5, n)[0]
    k = gaussian_kernel(FWHM)
    resp = (k * _gauss(k.shape[0], k.shape[1], (k.shape[1] - 1) / 2, (k.shape[0] - 1) / 2, FWHM, 1.0)).sum()
    yy, xx = np.mgrid[0:n, 0:n]
    band = (np.abs(np.hypot(xx - c, yy - c) - 15.0) <= FWHM) & np.isfinite(clean)
    sig = float(np.std(V.mf_convolve(clean, FWHM)[band], ddof=1))
    rho, pa = 15.0 * PX, 40.0
    xs, ys = source_xy(rho, pa, PX, c, c)
    unit = _gauss(n, n, float(xs[0]), float(ys[0]), FWHM, sig / resp)     # matched-filter S/N ~1 per unit
    clean = clean + 3.0 * unit            # a noise excursion of +3 sigma at the source, in the clean image
    sc, lin = {}, {}
    for f in (1.0, 10.0):
        inj = clean + 7.0 * f * unit
        con = 1e-5 * f
        sc[f] = V.solved_contrasts(clean, inj, rho, pa, con, PX, FWHM)
        s1 = V.snr_fap(inj, rho, pa, PX, FWHM)
        assert sc[f]["snr1"] == pytest.approx(s1["snr"], rel=1e-9)        # a = 1: the injected image's S/N
        lin[f] = V.contrast_limit(2.87e-7, 0.90, [con], [s1["snr"]], [s1["nap"]])
    a, b = sc[1.0], sc[10.0]
    assert a["c_det"] == pytest.approx(b["c_det"], rel=1e-6)
    assert all(x == pytest.approx(y, rel=1e-6) for x, y in zip(a["c_tap"], b["c_tap"]))
    assert a["c_tap"][0] == pytest.approx(a["c_det"], rel=1e-6)          # TAP50's threshold is tau itself
    assert a["c_det"] < a["c_tap"][1] < a["c_tap"][2] < 1e-4
    # the scaled limit carries the excursion along with the source: 10x brighter, another limit
    assert lin[10.0] / lin[1.0] > 1.08
    # at the solved contrast the source is just detected
    ad = a["c_det"] / 1e-5
    up = V.snr_fap(clean + 7.0 * ad * (1 + 1e-6) * unit, rho, pa, PX, FWHM)
    dn = V.snr_fap(clean + 7.0 * ad * (1 - 1e-4) * unit, rho, pa, PX, FWHM)
    assert up["fap"] < 2.87e-7 <= dn["fap"]
    # out of reach and no contrast
    assert np.isinf(V.solved_contrasts(clean, clean, rho, pa, 1e-5, PX, FWHM)["c_det"])
    assert np.isnan(V.solved_contrasts(clean, clean + unit, rho, pa, 0.0, PX, FWHM)["c_det"])


def test_tau_cap_named_and_exact_in_table3():
    """t_thresh caps tau at 50, as the IDL does, while t_fap decides exactly.  TABLE 1 says
    where the cap binds, and TABLE 3's solved limits take the exact tau (NEAR2 note of
    2026-10-23, night)."""
    from scipy.stats import t as student
    exact6 = float(student.isf(2.87e-7, 4))                             # n = 6 samples
    assert V.t_thresh(2.87e-7, 6) == 50.0 and exact6 > 56.0
    assert V._t_exact(2.87e-7, 6) == pytest.approx(exact6)
    assert V.t_fap(55.0, 6) > 2.87e-7                                    # S/N 55 is no detection at n = 6
    assert not V._reaches(55.0, 6, 2.87e-7, 0.5) and V._reaches(exact6 + 1e-6, 6, 2.87e-7, 0.5)
    rows = [{"kind": "injection", "rho": 0.31, "theta": 0.0, "contrast": 1.5e-3, "snr": 9.0, "nap": 4},
            {"kind": "injection", "rho": 0.77, "theta": 0.0, "contrast": 5e-5, "snr": 7.0, "nap": 12}]
    lim = V.limit_tables(rows, 6.24, 0.0456)
    txt = "\n".join(V._table_lines(lim))
    assert "tau is capped at 50" in txt and f"{student.isf(2.87e-7, 2):.0f} at 0.31" in txt
    assert lim["bins"][0]["tau"] == 50.0 and lim["bins"][1]["tau_exact"] < 50.0
    assert "capped" not in "\n".join(V._table_lines(V.limit_tables(rows[1:], 6.24, 0.0456)))


def test_verify_path_a_on_the_combined_image_given(tmp_path):
    """Given the image the run made (and its injected twin), Path A measures it, the
    subsets' combined entry is it, and the other subsets stay on the nights.  Without it the
    weighted mean of the nights, as before (NEAR2 note of 2026-10-20)."""
    nn = 4
    stack = _noise_stack(nn, 21)
    inj = _planted_stack(nn, 1.0, 200.0, 8.0, seed=21)        # the same noise, plus the source
    src = [(1.0, 200.0, 1e-4)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_c, mean_i = np.nanmean(stack, axis=0), np.nanmean(inj, axis=0)
    extra = 0.7 * _noise_stack(1, 77)[0]                       # the product is not the nights' mean
    prod_c, prod_i = mean_c + extra, mean_i + extra
    kw = dict(n_boot=0, return_maps=True)
    rows, maps = V.verify_candidates(stack, None, [(0.7, 120.0)], FWHM, PX, injected=(inj, src),
                                     combined=prod_c, combined_inj=prod_i, **kw)
    rows0, _ = V.verify_candidates(stack, None, [(0.7, 120.0)], FWHM, PX, injected=(inj, src), **kw)
    cand, injr = rows[0], rows[1]
    cand0, injr0 = rows0[0], rows0[1]
    exr = 1.5 * FWHM
    assert cand["snr"] == pytest.approx(V.snr_fap(prod_c, 0.7, 120.0, PX, FWHM, exr, 1.5)["snr"], rel=1e-12)
    assert injr["snr"] == pytest.approx(V.snr_fap(prod_i, 1.0, 200.0, PX, FWHM, exr, 1.5)["snr"], rel=1e-12)
    assert injr0["snr"] == pytest.approx(V.snr_fap(mean_i, 1.0, 200.0, PX, FWHM, exr, 1.5)["snr"], rel=1e-12)
    assert abs(injr["snr"] - injr0["snr"]) > 0.05
    assert injr["subsets"]["combined"] == injr["snr"] and cand["subsets"]["combined"] == cand["snr"]
    for q in V.SUBSET_NAMES[1:]:
        assert injr["subsets"][q] == injr0["subsets"][q] and cand["subsets"][q] == cand0["subsets"][q]
    assert injr["stim"] == injr0["stim"] and cand["stim"] == cand0["stim"]
    # the solved contrasts pair the product with its injected twin
    sc = V.solved_contrasts(prod_c, prod_i, 1.0, 200.0, 1e-4, PX, FWHM, exr, 1.5)
    assert injr["c_det"] == pytest.approx(sc["c_det"], rel=1e-12)
    assert injr["c_tap_solved"] == pytest.approx(sc["c_tap"], rel=1e-12)
    assert maps["meta"]["path_a"] == "the combined image given"
    txt = V.write_verify_table(str(tmp_path / "r.txt"), rows, maps["limits"])
    assert "# PATH A = the combined image given" in txt
    assert "TABLE 3" in txt and "each source, rho" in txt
    txt0 = V.write_verify_table(str(tmp_path / "r0.txt"), rows0)
    assert "# PATH A = the equal-weight mean of the nights" in txt0
    with pytest.raises(ValueError):
        V.verify_candidates(stack, None, [(0.7, 120.0)], FWHM, PX, injected=(inj, src), combined=prod_c, n_boot=0)
    with pytest.raises(ValueError):
        V.verify_candidates(stack, None, [], FWHM, PX, combined=prod_c[:-1], n_boot=0)


def test_table3_counts_and_quantile():
    """TABLE 3's bins: the mean of the finite solved limits, and the faintest contrast at which
    90 % of the bin's sources are detected and the count detected at TABLE 1's c@TAP90, both
    with every source evaluated at that contrast.  Every source with solved contrasts counts,
    whatever its S/N at the injected contrast."""
    inf = float("inf")
    rows, items = [], []
    for k in range(10):
        r = {"kind": "injection", "rho": 1.0 + 0.001 * k, "theta": 10.0 * k, "contrast": 1e-5,
             "snr": 6.0 + 0.1 * k, "nap": 30, "c_det": (k + 1) * 1e-6,
             "c_tap_solved": [(k + 1) * 1e-6, (k + 2) * 1e-6, (k + 3) * 1e-6]}
        rows.append(r)
        items.append((r, (lambda t: (lambda c: c >= t))((k + 1) * 1e-6)))
    # one source detected only in a window, from 1e-6 to 1.5e-6, and one from 1e-5 on
    items[0] = (rows[0], lambda c: 1e-6 <= c < 1.5e-6)
    rows[9].update(c_det=1e-5, c_tap_solved=[inf, inf, inf])
    items[9] = (rows[9], lambda c: c >= 1e-5)
    # S/N <= 0 at the injected contrast: not in TABLE 1, in TABLE 3, in a bin of its own
    r = {"kind": "injection", "rho": 2.0, "theta": 0.0, "contrast": 1e-5, "snr": -0.5, "nap": 60,
         "c_det": 3e-5, "c_tap_solved": [3e-5, 4e-5, 5e-5]}
    rows.append(r)
    items.append((r, lambda c: c >= 3e-5))
    lim = V.limit_tables(rows, FWHM, PX)
    assert lim["solved"] is None and len(lim["bins"]) == 1 and lim["bins"][0]["n_inj"] == 10
    s0, s1 = V._solved_bins(lim, items)
    assert s0["n"] == 10 and s1["n"] == 1
    assert s0["c_tap"][1] == pytest.approx(np.mean(np.arange(2, 11)) * 1e-6)      # the inf left out
    # 9 of 10: the windowed source never counts with the others, so 1e-5, where the last one joins
    assert s0["c_det_beta"] == pytest.approx(1e-5, rel=1e-8)
    t1 = lim["bins"][0]["c_tap"][1]
    assert s0["n_det_t1"] == sum(d(t1) for _, d in items[:10])
    assert s1["n_det_t1"] is None and s1["c_det_beta"] == pytest.approx(3e-5, rel=1e-8) and s1["sep"] == 2.0
    items[9] = (rows[9], lambda c: False)                                         # two of ten never with the rest
    assert np.isinf(V._solved_bins(lim, items)[0]["c_det_beta"])
    lim["solved"] = [s0, s1]
    txt = "\n".join(V._table3_lines(lim))
    assert "TABLE 3" in txt and "c(90%det)" in txt and "inf" in txt and f"{s0['n_det_t1']}/10" in txt
    assert V._solved_bins(lim, []) is None


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


def test_candidates_measure_the_image_given(tmp_path):
    """Given the run's stitch, a candidate's combined S/N, FAP and PSF-likeness are measured
    on it, its per-night S/N, STIM and even / odd S/N on the nights.  Without it the nights'
    mean, as in the IDL."""
    nn = 4
    rho, theta = 0.9, 300.0
    stack = _planted_stack(nn, rho, theta, 8.0, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # the masked star: all-NaN pixels
        img = np.nanmean(stack, axis=0) + 0.6 * _noise_stack(1, 99)[0]
    cx, cy = star_center((N, N))
    xs, ys = source_xy(rho, theta, PX, cx, cy)
    px, py = np.array([int(round(xs[0]))]), np.array([int(round(ys[0]))])
    r0 = C.candidate_metrics(px, py, np.array([5.0]), stack, FWHM, PX, 6.0, 40.0)[0]
    r1 = C.candidate_metrics(px, py, np.array([5.0]), stack, FWHM, PX, 6.0, 40.0, image=img)[0]
    fin = np.isfinite(stack)
    mean = np.where(fin, stack, 0.0).sum(axis=0) / np.maximum(fin.sum(axis=0), 1)
    args = (r1["rho"], r1["theta"], PX, FWHM, 1.5 * FWHM, 1.5)
    assert r1["snr"] == pytest.approx(V.snr_fap(img, *args)["snr"], rel=1e-12)
    assert r0["snr"] == pytest.approx(V.snr_fap(mean, *args)["snr"], rel=1e-12)
    assert abs(r1["snr"] - r0["snr"]) > 0.05
    assert r1["psf_corr"] == pytest.approx(C.psf_correlation(img, int(px[0]), int(py[0]), FWHM), rel=1e-12)
    for key in ("per_night", "n_nights", "stim", "sub1", "sub2"):
        assert r1[key] == r0[key]
    rows = C.find_candidates(img, stack, FWHM, PX, verify_top=1, verify_kwargs={"n_boot": 0})
    top = rows[0]
    assert top["meta"]["comb"] == "image"
    assert top["verify"]["snr"] == pytest.approx(
        V.snr_fap(img, top["rho"], top["theta"], PX, FWHM, 1.5 * FWHM, 1.5)["snr"], rel=1e-12)
    assert "S/N, FAP and PSF on the source image" in C.write_candidates(str(tmp_path / "c.txt"), rows)
    assert C.find_candidates(None, stack, FWHM, PX)[0]["meta"]["comb"] == "nights"


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
    assert mcfg.validation.dedup == "vector" and mcfg.validation.n_rescore == 0, "the IDL's validation rules"
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
