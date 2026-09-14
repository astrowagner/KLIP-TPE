"""Multi-night source verification -- port of ``near2_verify.pro``.

Two independent "paths" are evaluated for every candidate ``(rho, theta)``:

* **Path A** -- matched-filter Mawet et al. (2014) S/N on the combined image with
  the small-sample false-alarm probability (Student-t, ``df = n_ap - 2``;
  ``near2v_snrfap`` / ``near2v_tfap``).  The decision is ``fap < fap_op``.
* **Path B** -- night-to-night consistency: the night-STIM map
  (``near2v_stim``) with an empirical FAP from the same-radius annulus, a
  bootstrap over nights of the matched-filter peak (``near2v_boot``), and the
  S/N in five semi-independent subsets (combined / first half / second half /
  even / odd nights) whose minimum over the four partitions is the
  *persistence*.

When an injected twin of the night stack is given, the same machinery is run
at the injected positions and the Path-A detection limits
``c_lim = c_inj (tau_alpha + tau_{1-beta}) / SNR_inj`` (``near2v_climit``) are
binned by separation (TABLE 1 / TABLE 2 of the IDL report).

Conventions: images are ``(ny, nx)`` numpy arrays indexed ``[y, x]``; the star is
at ``((nx-1)/2, (ny-1)/2)``; ``theta`` is a position angle East of North with
``x = cx + r cos(theta+90)``, ``y = cy + r sin(theta+90)`` (verify 128-129).

Deliberate deviations from the IDL (each also flagged at the call site):

* ``near2v_tthresh`` bisects ``tau`` on ``[0, 50]``; we use ``scipy.stats.t.isf``
  and clip to the same ``[-50, 50]`` range so extreme (``df=1``) thresholds
  agree with the IDL value instead of exploding.
* The bootstrap (verify 208) calls ``near2v_snrfap`` with ``exr = srad`` only to
  read back the matched-filter peak value (``s[3]``), which the ring exclusion
  never influences.  We evaluate the peak directly; results are identical.
* The injected-STIM empirical FAP (verify 660) omits the 1.5-FWHM keep-out
  around the source that the candidate FAP (verify 601) applies; we apply the
  keep-out in both cases (brief C.6).
* ``median`` of the per-bin S/N uses ``numpy.median`` (mean of the two middle
  values) rather than IDL's upper-middle default.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import t as _student_t

from .metrics import gaussian_kernel, radprof, source_xy, star_center, nanmedian_even

__all__ = ["t_fap", "t_thresh", "contrast_limit", "contrast_limit_image", "mf_convolve",
           "snr_fap", "combine_nights", "night_stim", "stim_fap", "bootstrap_nights",
           "night_subsets", "verify_maps", "verify_candidates", "verify_injections",
           "limit_tables", "write_verify_table", "write_curve_table"]

SUBSET_NAMES = ("combined", "first-half", "second-half", "even", "odd")
DEFAULT_FAP_LIST = (2.87e-7, 1.350e-3, 1.0e-2, 0.10)      # verify 514: 5sig-eq, 3sig-eq, 1%, 10%
DEFAULT_TAP_LIST = (0.50, 0.90, 0.95)                     # verify 515


# ----------------------------------------------------------------------------
# C.1.1 - C.1.3  statistical primitives
# ----------------------------------------------------------------------------
def t_fap(snr: float, n: int) -> float:
    """Mawet (2014) small-sample false-alarm probability per resolution element
    (``near2v_tfap``, verify 66-72): Student-t survival ``P(T > snr)`` with
    ``df = max(n-2, 1)``.  ``snr <= 0`` returns 0.5 exactly (IDL behaviour);
    ``n`` is the number of noise apertures actually used."""
    if not np.isfinite(snr):
        return float("nan")
    df = max(int(n) - 2, 1)
    if snr <= 0:
        return 0.5
    return float(_student_t.sf(float(snr), df))


def t_thresh(alpha: float, n: int) -> float:
    """S/N threshold ``tau`` with ``t_fap(tau, n) = alpha`` (``near2v_tthresh``,
    verify 80-91).  ``alpha <= 0 -> 1e30``.  DEVIATION: closed-form
    ``t.isf`` clipped to the IDL bisection range ``[-50, 50]``."""
    if alpha <= 0:
        return 1.0e30
    df = max(int(n) - 2, 1)
    return float(np.clip(_student_t.isf(float(alpha), df), -50.0, 50.0))


def contrast_limit(alpha: float, beta: float, ci, si, napi) -> float:
    """Detection-limit contrast at per-resel FAP ``alpha`` and completeness
    ``beta`` scaled from injections (``near2v_climit``, verify 99-107):
    ``lim_k = c_k (tau_alpha + tau_{1-beta}) / SNR_k`` with ``n_ap = max(round
    (nap_k), 3)``, averaged over injections with ``SNR>0`` and ``c>0``."""
    ci, si, napi = (np.atleast_1d(np.asarray(a, float)) for a in (ci, si, napi))
    lim, nl = 0.0, 0
    for k in range(ci.size):
        if si[k] > 0 and np.isfinite(ci[k]) and ci[k] > 0 and np.isfinite(si[k]):
            na = max(int(round(napi[k])), 3)
            lim += ci[k] * (t_thresh(alpha, na) + t_thresh(1.0 - beta, na)) / si[k]
            nl += 1
    return lim / nl if nl else float("nan")


def contrast_limit_image(alpha_img: float, beta: float, ci, si, napi) -> float:
    """Same as :func:`contrast_limit` for a PER-IMAGE FAP converted to per-resel
    with ``ar = min(alpha_img / n_ap, 0.49)`` (``near2v_climg``, verify 352-360)."""
    ci, si, napi = (np.atleast_1d(np.asarray(a, float)) for a in (ci, si, napi))
    lim, nl = 0.0, 0
    for k in range(ci.size):
        if si[k] > 0 and np.isfinite(ci[k]) and ci[k] > 0 and np.isfinite(si[k]):
            na = max(int(round(napi[k])), 3)
            ar = min(alpha_img / na, 0.49)
            lim += ci[k] * (t_thresh(ar, na) + t_thresh(1.0 - beta, na)) / si[k]
            nl += 1
    return lim / nl if nl else float("nan")


# ----------------------------------------------------------------------------
# C.1.4  matched-filter S/N + FAP at a source
# ----------------------------------------------------------------------------
def mf_convolve(img: np.ndarray, fwhm: float, kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """``near2v_mfconv`` (verify 117-120): NaN -> 0 then convolution with the
    unit-sum FWHM Gaussian (``near2v_mfkern``, verify 53-59) with edge
    truncation (``mode='nearest'``)."""
    img = np.asarray(img, float)
    k = gaussian_kernel(fwhm) if kernel is None else kernel
    return ndimage.convolve(np.where(np.isfinite(img), img, 0.0), k, mode="nearest")


def snr_fap(img: np.ndarray, rho: float, theta: float, pxscale: float, fwhm: float,
            exr: Optional[float] = None, srad: float = 1.5, *,
            exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
            angle_convention: str = "pa") -> Dict[str, float]:
    """Matched-filter Mawet S/N and small-sample FAP at one source
    (``near2v_snrfap``, verify 122-154).

    1. ``A`` = ``mfmap`` if given (same shape) else :func:`mf_convolve` (127).
    2. Peak search on the disc ``dx^2+dy^2 <= srad^2`` around the rounded nominal
       position; the pixel with max ``A`` where ``img`` is finite wins (131-136).
    3. ``r = |peak - centre|``, ``nap = floor(2 pi r / fwhm)``; ``r < fwhm`` or
       ``nap < 5`` -> NaN (137-138).
    4. Ring samples at ``th0 + q 2pi/nap``, ``q = 1..nap-1``, skipping pixels
       within ``exr`` of the peak (radial exclusion) or of any ``exclude_xy``
       coordinate, out of bounds or non-finite (139-148).
    5. ``nv < 3`` or ``sd <= 0`` -> NaN; ``snr = (A_pk - mean) / (sd sqrt(1+1/nv))``
       with the sample (N-1) std (150-152); ``fap = t_fap(snr, nv)`` (153).

    Returns ``{'snr', 'fap', 'nap', 'peak', 'x', 'y', 'r'}`` where ``nap`` is the
    number of ring samples actually used (``nv``) on success, the nominal count
    on the early exits (as in the IDL return vector).
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    if exr is None:
        exr = 1.5 * fwhm
    A = mfmap if (mfmap is not None and np.shape(mfmap) == img.shape) else mf_convolve(img, fwhm)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    xs, ys = float(xs[0]), float(ys[0])
    px0, py0 = int(round(xs)), int(round(ys))
    px, py, pkv = px0, py0, -1e30
    isr = int(np.ceil(srad))
    for dy in range(-isr, isr + 1):
        for dx in range(-isr, isr + 1):
            if dx * dx + dy * dy > srad * srad:
                continue
            qx, qy = px0 + dx, py0 + dy
            if 0 <= qx < nx and 0 <= qy < ny and np.isfinite(img[qy, qx]) and A[qy, qx] > pkv:
                pkv, px, py = float(A[qy, qx]), qx, qy
    peak = pkv if pkv > -1e30 else float("nan")
    r = float(np.hypot(px - cx, py - cy))
    nap = int(np.floor(2 * np.pi * r / fwhm))
    out = {"snr": np.nan, "fap": np.nan, "nap": nap, "peak": peak, "x": px, "y": py, "r": r}
    if r < fwhm or nap < 5:
        return out
    th0 = np.arctan2(py - cy, px - cx)
    vals: List[float] = []
    for q in range(1, nap):
        th = th0 + q * 2 * np.pi / nap
        xk, yk = int(round(cx + r * np.cos(th))), int(round(cy + r * np.sin(th)))
        if np.hypot(xk - px, yk - py) < exr:
            continue
        if any(np.hypot(xk - ex, yk - ey) < exr for ex, ey in exclude_xy):
            continue
        if not (0 <= xk < nx and 0 <= yk < ny) or not np.isfinite(img[yk, xk]):
            continue
        vals.append(float(A[yk, xk]))
    nv = len(vals)
    out["nap"] = nv
    if nv < 3:
        return out
    use = np.array(vals)
    sd = float(np.std(use, ddof=1))
    if sd <= 0:
        return out
    snr = (float(A[py, px]) - float(np.mean(use))) / (sd * np.sqrt(1.0 + 1.0 / nv))
    out.update(snr=float(snr), fap=t_fap(snr, nv), peak=float(A[py, px]))
    return out


# ----------------------------------------------------------------------------
# C.1.5 - C.1.7  night combine, night-STIM, bootstrap
# ----------------------------------------------------------------------------
def combine_nights(stack: np.ndarray, weights: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """Weighted NaN-aware mean of the selected slices (``near2v_combine``, verify
    160-170): ``sum w_j img_j fin_j / sum w_j fin_j``; zero total weight -> NaN.
    Duplicate indices (bootstrap draws) count with multiplicity."""
    stack = np.asarray(stack, float)
    w = np.asarray(weights, float)
    num = np.zeros(stack.shape[1:])
    den = np.zeros(stack.shape[1:])
    for j in idx:
        im = stack[j]
        fin = np.isfinite(im)
        num += np.where(fin, im, 0.0) * w[j]
        den += fin * w[j]
    return num / np.where(den > 0, den, np.nan)


def night_stim(stack: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Night-STIM map (``near2v_stim``, verify 177-196): weighted mean over the
    weighted (BIASED, no N-1) std across nights, with the std floored at
    ``0.3 * median(std > 0)`` and ``1e-12``.  All-NaN if fewer than 2 nights;
    NaN where the total weight is 0."""
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    if nn < 2:
        return np.full(stack.shape[1:], np.nan)
    w = np.ones(nn) if weights is None else np.asarray(weights, float)
    fin = np.isfinite(stack)
    z = np.where(fin, stack, 0.0)
    wt = np.tensordot(w, fin.astype(float), axes=(0, 0))
    mn = np.tensordot(w, z, axes=(0, 0)) / np.maximum(wt, 1e-12)
    d = np.where(fin, stack - mn[None], 0.0)
    vr = np.tensordot(w, d * d, axes=(0, 0))
    sdv = np.sqrt(vr / np.maximum(wt, 1e-12))
    gd = np.isfinite(sdv) & (sdv > 0)
    sfloor = float(np.median(sdv[gd])) if gd.any() else 1.0
    stim = mn / np.maximum(np.maximum(sdv, 0.3 * sfloor), 1e-12)
    stim[wt <= 0] = np.nan
    return stim


def stim_fap(stim: np.ndarray, x: float, y: float, rho_px: float, fwhm: float,
             keepout: Optional[float] = None) -> Tuple[float, float]:
    """Empirical STIM FAP (verify 594-602): the value at the rounded source pixel
    ranked against the annulus ``|r - rho_px| <= fwhm`` (finite pixels farther
    than ``keepout`` (default ``1.5 fwhm``) from the source).  Returns
    ``(stim_value, fraction >= value)``."""
    ny, nx = stim.shape
    cx, cy = star_center(stim.shape)
    sx, sy = int(round(x)), int(round(y))
    val = float(stim[sy, sx]) if (0 <= sx < nx and 0 <= sy < ny) else float("nan")
    if keepout is None:
        keepout = 1.5 * fwhm
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    dsrc = np.hypot(xx - sx, yy - sy)
    ann = (np.abs(rr - rho_px) <= fwhm) & np.isfinite(stim) & (dsrc > keepout)
    if not ann.any() or not np.isfinite(val):
        return val, float("nan")
    return val, float(np.mean(stim[ann] >= val))


def bootstrap_nights(stack: np.ndarray, weights: np.ndarray, rho: float, theta: float,
                     pxscale: float, fwhm: float, srad: float = 1.5, n_boot: int = 500,
                     rng: Optional[np.random.Generator] = None, angle_convention: str = "pa") -> float:
    """Bootstrap-over-nights significance (``near2v_boot``, verify 203-214):
    resample the ``nn`` nights with replacement ``n_boot`` times, recombine
    (weighted, no radprof), re-locate the matched-filter peak within ``srad`` and
    record its VALUE; result ``mean / std`` over finite draws (NaN if < 3 finite
    or std = 0).  DEVIATION: the peak value is read directly (the IDL ring
    exclusion ``exr = srad`` cannot affect it)."""
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)
    kern = gaussian_kernel(fwhm)
    vals = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        idx = rng.integers(0, nn, nn)
        cb = combine_nights(stack, weights, idx)
        A = mf_convolve(cb, fwhm, kern)
        vals[b] = snr_fap(cb, rho, theta, pxscale, fwhm, exr=srad, srad=srad, mfmap=A,
                          angle_convention=angle_convention)["peak"]
    g = vals[np.isfinite(vals)]
    if g.size < 3:
        return float("nan")
    s = float(np.std(g, ddof=1))
    return float(np.mean(g) / s) if s > 0 else float("nan")


# ----------------------------------------------------------------------------
# C.2  driver
# ----------------------------------------------------------------------------
def night_subsets(nn: int) -> Dict[str, np.ndarray]:
    """Night index subsets (verify 542-544): all, first half ``0..nn//2-1``,
    second half ``nn//2..``, even ``j%2==0``, odd ``j%2==1``."""
    a = np.arange(nn)
    return {"combined": a, "first-half": a[: nn // 2], "second-half": a[nn // 2:],
            "even": a[a % 2 == 0], "odd": a[a % 2 == 1]}


def verify_maps(stack: np.ndarray, weights: Optional[np.ndarray] = None, fwhm: Optional[float] = None
                ) -> Dict[str, Any]:
    """Subset images (radprof-flattened weighted combines, verify 545-552), the
    night-STIM map (555) and, when ``fwhm`` is given, the matched-filter map of
    every subset.  Empty subsets (nn = 1) give all-NaN images."""
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    w = np.ones(nn) if weights is None else np.asarray(weights, float)
    subs = night_subsets(nn)
    imgs: Dict[str, np.ndarray] = {}
    for name, idx in subs.items():
        imgs[name] = radprof(combine_nights(stack, w, idx)) if idx.size else np.full(stack.shape[1:], np.nan)
    out: Dict[str, Any] = {"subsets": imgs, "stim": night_stim(stack, w), "weights": w, "subset_index": subs}
    if fwhm is not None:
        out["mf"] = {k: mf_convolve(v, fwhm) for k, v in imgs.items()}
    return out


def _known_xy(known, pxscale, cx, cy, angle_convention):
    if not len(known):
        return []
    kx, ky = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention)
    return list(zip(kx, ky))


def verify_candidates(stack: np.ndarray, angles_or_night_ids: Optional[Sequence[Any]],
                      candidates: Sequence[Tuple[float, float]], fwhm: float, pxscale: float, *,
                      known: Sequence[Tuple[float, float]] = (), injected=None,
                      fap_op: float = 2.87e-7, drift_px: float = 1.5, n_boot: int = 500,
                      rng: Optional[np.random.Generator] = None, weights: Optional[Sequence[float]] = None,
                      excl_fwhm: float = 1.5, fap_list: Sequence[float] = DEFAULT_FAP_LIST,
                      tap_list: Sequence[float] = DEFAULT_TAP_LIST, rbin_as: Optional[float] = None,
                      angle_convention: str = "pa", return_maps: bool = False) -> List[Dict[str, Any]]:
    """Verify candidates on a per-night image stack (``near2_verify``, verify 493-746).

    Parameters
    ----------
    stack
        ``(n_night, ny, nx)`` per-night (per-partition) images, NOT flattened.
    angles_or_night_ids
        one label per slice (night id / epoch / partition id) used for the
        report; ``None`` -> ``0..n-1``.  The subset split (halves, even/odd)
        follows the slice ORDER, so pass the slices in chronological order.
    candidates
        ``[(rho_arcsec, theta_deg), ...]``.
    known
        real sources excluded from the noise rings (verify ``exclude``).
    injected
        optional ``(inj_stack, sources)`` with ``sources = [(rho, theta,
        contrast), ...]``: the injected twin of ``stack`` used for the
        calibration block and the detection-limit tables.
    weights
        per-night weights (IDL: ``sqrt(texp)``); default equal.
    rng
        ``numpy.random.Generator`` for the bootstrap (IDL ``seed=42``);
        ``n_boot=0`` skips it (``/lite``).

    Returns a list of dicts, one per candidate (``kind='candidate'``) followed
    by one per injected source (``kind='injection'``) when ``injected`` is given.
    The candidate dict holds ``rho, theta, x, y, snr, fap, nap, detect,
    subsets{name: snr}, persistence, stim, stim_fap, boot_sig``.  With
    ``return_maps=True`` the return is ``(rows, maps)``; ``maps`` also holds the
    ``limits`` tables (see :func:`limit_tables`).
    """
    stack = np.asarray(stack, float)
    if stack.ndim == 2:
        stack = stack[None]
    nn, ny, nx = stack.shape
    cx, cy = star_center((ny, nx))
    ids = list(range(nn)) if angles_or_night_ids is None else [str(v) for v in angles_or_night_ids]
    w = np.ones(nn) if weights is None else np.asarray(weights, float)
    exr = excl_fwhm * fwhm
    if rng is None:
        rng = np.random.default_rng(42)
    kxy = _known_xy(known, pxscale, cx, cy, angle_convention)

    maps = verify_maps(stack, w, fwhm)
    stim = maps["stim"]
    rows: List[Dict[str, Any]] = []
    for c, (rho, theta) in enumerate(candidates):
        rho, theta = float(rho), float(theta)
        # Path A on the combined image (verify 588)
        sa = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, drift_px,
                     exclude_xy=kxy, mfmap=maps["mf"]["combined"], angle_convention=angle_convention)
        ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, drift_px,
                              exclude_xy=kxy, mfmap=maps["mf"][name], angle_convention=angle_convention)["snr"]
                for name in SUBSET_NAMES}
        # Path B: STIM at the nominal (rounded) position with the annulus null (594-602)
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(stim, float(xs[0]), float(ys[0]), rho / pxscale, fwhm)
        bsig = bootstrap_nights(stack, w, rho, theta, pxscale, fwhm, drift_px, n_boot, rng,
                                angle_convention) if (n_boot and nn >= 2) else float("nan")
        part = [ssub[k] for k in SUBSET_NAMES[1:]]
        persist = float(np.nanmin(part)) if np.any(np.isfinite(part)) else float("nan")
        rows.append({"kind": "candidate", "index": c + 1, "rho": rho, "theta": theta,
                     "x": sa["x"], "y": sa["y"], "snr": sa["snr"], "fap": sa["fap"], "nap": sa["nap"],
                     "peak": sa["peak"], "detect": bool(np.isfinite(sa["fap"]) and sa["fap"] < fap_op),
                     "subsets": ssub, "persistence": persist, "stim": sval, "stim_fap": sfap,
                     "boot_sig": bsig})

    maps["limits"] = None
    if injected is not None:
        inj_stack, sources = injected
        irows, imaps = verify_injections(inj_stack, sources, fwhm, pxscale, known=known,
                                         weights=w[: np.asarray(inj_stack).shape[0]], drift_px=drift_px,
                                         excl_fwhm=excl_fwhm, angle_convention=angle_convention,
                                         fap_list=fap_list, tap_list=tap_list, rbin_as=rbin_as)
        rows.extend(irows)
        maps["injected"] = imaps
        maps["limits"] = imaps["limits"]
    maps["meta"] = {"nights": ids, "fap_op": fap_op, "drift_px": drift_px, "excl_fwhm": excl_fwhm,
                    "fwhm": fwhm, "pxscale": pxscale, "n_boot": n_boot}
    for r in rows:
        r.setdefault("meta", maps["meta"])
    if return_maps:
        return rows, maps
    return rows


def verify_injections(inj_stack: np.ndarray, sources: Sequence[Tuple[float, float, float]], fwhm: float,
                      pxscale: float, *, known: Sequence[Tuple[float, float]] = (),
                      weights: Optional[Sequence[float]] = None, drift_px: float = 1.5,
                      excl_fwhm: float = 1.5, angle_convention: str = "pa",
                      fap_list: Sequence[float] = DEFAULT_FAP_LIST, tap_list: Sequence[float] = DEFAULT_TAP_LIST,
                      rbin_as: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Injection calibration block (verify 621-735): Path-A S/N / FAP, subset S/N,
    injected-STIM value + annulus FAP per injected source ``(rho, theta,
    contrast)``, then the detection-limit tables from sources with finite
    ``snr > 0`` and ``contrast > 0``.  DEVIATION: the STIM FAP applies the
    1.5-FWHM keep-out (the IDL, line 660, does not)."""
    inj_stack = np.asarray(inj_stack, float)
    if inj_stack.ndim == 2:
        inj_stack = inj_stack[None]
    nn, ny, nx = inj_stack.shape
    cx, cy = star_center((ny, nx))
    w = np.ones(nn) if weights is None else np.asarray(weights, float)[:nn]
    exr = excl_fwhm * fwhm
    kxy = _known_xy(known, pxscale, cx, cy, angle_convention)
    maps = verify_maps(inj_stack, w, fwhm)
    rows: List[Dict[str, Any]] = []
    for m, src in enumerate(sources):
        rho, theta = float(src[0]), float(src[1])
        con = float(src[2]) if len(src) > 2 else float("nan")
        s = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, drift_px,
                    exclude_xy=kxy, mfmap=maps["mf"]["combined"], angle_convention=angle_convention)
        ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, drift_px,
                              exclude_xy=kxy, mfmap=maps["mf"][name], angle_convention=angle_convention)["snr"]
                for name in SUBSET_NAMES}
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(maps["stim"], float(xs[0]), float(ys[0]), rho / pxscale, fwhm)
        rows.append({"kind": "injection", "index": m + 1, "rho": rho, "theta": theta, "contrast": con,
                     "x": s["x"], "y": s["y"], "snr": s["snr"], "fap": s["fap"], "nap": s["nap"],
                     "peak": s["peak"], "subsets": ssub, "stim": sval, "stim_fap": sfap})
    maps["limits"] = limit_tables(rows, fwhm, pxscale, fap_list, tap_list, rbin_as)
    return rows, maps


def limit_tables(inj_rows: Sequence[Dict[str, Any]], fwhm: float, pxscale: float,
                 fap_list: Sequence[float] = DEFAULT_FAP_LIST, tap_list: Sequence[float] = DEFAULT_TAP_LIST,
                 rbin_as: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Binned Path-A detection limits (verify 674-718 / ``near2v_curvetxt`` 447-488).

    Injections with finite ``snr > 0`` and ``contrast > 0`` are binned by ``rho``
    in ``rbin_as`` (default ``fwhm * pxscale``) bins from ``min(rho)``.  Per bin:
    ``sep = mean(rho)``, ``n_inj``, ``snr_med``, ``tau = t_thresh(fap_list[0],
    max(round(mean(nap)), 3))``; TABLE 1 ``c@TAP`` = :func:`contrast_limit`
    ``(fap_list[0], tap)`` per ``tap``; TABLE 2 (TAP closest to 0.90) per
    ``fap_list`` entry plus the per-image 10 % column ``aimg = 0.10 / max(mean
    (nap), 1)``.  Returns ``None`` when no injection qualifies."""
    ri, si, ci, napi = [], [], [], []
    for r in inj_rows:
        if r.get("kind", "injection") != "injection":
            continue
        if np.isfinite(r["snr"]) and r["snr"] > 0 and np.isfinite(r.get("contrast", np.nan)) and r["contrast"] > 0:
            ri.append(r["rho"]); si.append(r["snr"]); ci.append(r["contrast"]); napi.append(r["nap"])
    if not ri:
        return None
    ri, si, ci, napi = (np.asarray(a, float) for a in (ri, si, ci, napi))
    rbw = float(rbin_as) if rbin_as else float(fwhm * pxscale)
    rmin, rmax = ri.min(), ri.max()
    nb = max(int(np.ceil((rmax - rmin) / rbw)), 1) if rmax > rmin else 1
    bidx = np.minimum(np.floor((ri - rmin) / rbw).astype(int), nb - 1) if nb > 1 else np.zeros(ri.size, int)
    fap_list, tap_list = list(fap_list), list(tap_list)
    i90 = int(np.argmin(np.abs(np.asarray(tap_list) - 0.90)))
    bins = []
    for bb in range(nb):
        ww = bidx == bb
        if not ww.any():
            continue
        nap_b = max(int(round(float(np.mean(napi[ww])))), 3)
        aimg = 0.10 / max(float(np.mean(napi[ww])), 1.0)
        bins.append({
            "sep": float(np.mean(ri[ww])), "n_inj": int(ww.sum()), "snr_med": float(np.median(si[ww])),
            "tau": t_thresh(fap_list[0], nap_b),
            "c_tap": [contrast_limit(fap_list[0], tp, ci[ww], si[ww], napi[ww]) for tp in tap_list],
            "c_fap": [contrast_limit(fa, tap_list[i90], ci[ww], si[ww], napi[ww]) for fa in fap_list],
            "c_img10": contrast_limit(aimg, tap_list[i90], ci[ww], si[ww], napi[ww]),
        })
    return {"rbin_as": rbw, "fap_list": fap_list, "tap_list": tap_list, "i90": i90, "n_inj": int(ri.size),
            "bins": bins, "samples": {"rho": ri, "snr": si, "contrast": ci, "nap": napi}}


# ----------------------------------------------------------------------------
# text products
# ----------------------------------------------------------------------------
def _f(v, fmt="{:.2f}"):
    try:
        return fmt.format(float(v)) if np.isfinite(float(v)) else "nan"
    except Exception:
        return str(v)


def _table_lines(lim: Dict[str, Any], indent: str = "#   ") -> List[str]:
    fl, tl, i90 = lim["fap_list"], lim["tap_list"], lim["i90"]
    out = [f"{indent}TABLE 1 -- contrast at the OPERATING FAP ({fl[0]:8.1E} per resel) vs separation",
           f"{indent}  rho[\"]  n_inj  S/N_med   tau " + "".join(f"   c@TAP{int(round(t*100)):02d}" for t in tl)]
    for b in lim["bins"]:
        out.append(f"     {b['sep']:5.2f}   {b['n_inj']:4d}   {b['snr_med']:6.2f}  {b['tau']:6.2f}"
                   + "".join(f"  {c:10.3E}" for c in b["c_tap"]))
    out.append(f"{indent}TABLE 2 -- TAP{int(round(100*tl[i90]))} (~90%) completeness contrast vs separation, by FAP threshold")
    out.append(f"{indent}  rho[\"] " + "".join(f"   {a:8.1E}/rs" for a in fl) + "   10%/img")
    for b in lim["bins"]:
        out.append(f"     {b['sep']:5.2f}" + "".join(f"  {c:10.3E}" for c in b["c_fap"]) + f"  {b['c_img10']:10.3E}")
    out.append(f"{indent}(FAP columns are per-resel; last col = per-image 10%, i.e. per-resel x n_resel at that rho.)")
    return out


def write_verify_table(path: str, rows: Sequence[Dict[str, Any]], limits: Optional[Dict[str, Any]] = None,
                       header: Optional[Dict[str, Any]] = None) -> str:
    """Write the ``verify_report.txt`` of ``near2_verify`` (verify 574-724) from
    the rows of :func:`verify_candidates`.  ``limits`` defaults to the tables
    derived from the injection rows present.  Returns the text."""
    meta = dict(header or {})
    for r in rows:
        if "meta" in r:
            meta = {**r["meta"], **meta}
            break
    fap_op = meta.get("fap_op", 2.87e-7)
    L = [f"# klip_tpe.verify -- {_dt.datetime.now().isoformat(timespec='seconds')}"]
    if "run" in meta:
        L.append(f"# run: {meta['run']}")
    L.append(f"# nights: {','.join(str(v) for v in meta.get('nights', []))}")
    L.append(f"# operating FAP threshold: {fap_op:10.2E}   drift tol: {meta.get('drift_px', 1.5):.1f} px")
    L += ["#", "# PATH A = final combined image (Mawet small-sample t-FAP)",
          "# PATH B = night-to-night (STIM + bootstrap-over-nights + subset S/N)", "#"]
    cands = [r for r in rows if r.get("kind", "candidate") == "candidate"]
    injs = [r for r in rows if r.get("kind") == "injection"]
    for r in cands:
        s = r["subsets"]
        L.append(f"== candidate {r['index']}  rho={r['rho']:.3f}\"  PA={r['theta']:.1f} deg ==")
        L.append(f"   [A] combined S/N = {_f(r['snr'])}   FAP = {_f(r['fap'], '{:10.2E}')}   (nap={int(r['nap'])})")
        L.append(f"   [A] decision @FAP<{fap_op:8.1E} : {'DETECT' if r['detect'] else 'no'}")
        L.append(f"   [B] night-STIM = {_f(r['stim'])}   empirical FAP = {_f(r['stim_fap'], '{:10.2E}')}")
        L.append(f"   [B] bootstrap-over-nights significance = {_f(r['boot_sig'])}")
        L.append(f"   [B] subset S/N  comb={_f(s['combined'])}  1st={_f(s['first-half'])}  2nd={_f(s['second-half'])}"
                 f"  even={_f(s['even'])}  odd={_f(s['odd'])}")
        L.append(f"   [B] persistence (min half/even/odd S/N) = {_f(r['persistence'])}")
        L.append("")
    if injs:
        L.append(f"# ---- injection calibration ({len(injs)} sources) ----")
        L.append("#   columns: rho[\"] PA[deg] contrast  | combined S/N  FAP  | injected-STIM  STIM-FAP")
        for r in injs:
            s = r["subsets"]
            cstr = f"{r['contrast']:10.3E}" if np.isfinite(r.get("contrast", np.nan)) and r["contrast"] > 0 else "  (n/a)  "
            L.append(f"   rho={r['rho']:.3f}\" PA={r['theta']:.1f}  c={cstr}  | S/N={_f(r['snr'])}  FAP={_f(r['fap'], '{:10.2E}')}"
                     f"  | STIM={_f(r['stim'])}  STIM-FAP={_f(r['stim_fap'], '{:10.2E}')}")
            L.append(f"       subset S/N  comb={_f(s['combined'])}  1st={_f(s['first-half'])}  2nd={_f(s['second-half'])}"
                     f"  even={_f(s['even'])}  odd={_f(s['odd'])}")
        if limits is None:
            limits = limit_tables(injs, meta.get("fwhm", 1.0), meta.get("pxscale", 1.0))
        if limits is not None:
            L += ["#", "# ==== Path-A detection limits vs separation (contrast = companion/star flux ratio) ====",
                  f"#   scaled from {limits['n_inj']} injections (linear flux<->contrast), binned by rho in "
                  f"{limits['rbin_as']:.2f}\" bins.",
                  "#   tau = matched-filter S/N needed for that per-resel FAP (small-sample; grows at small rho).", "#"]
            L += _table_lines(limits)
        L.append("#  -> Table 1 c@TAP90 at the operating FAP is the headline 90%-completeness limit per separation.")
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def write_curve_table(path: str, limits: Dict[str, Any], header: str = "") -> str:
    """``near2v_curvetxt`` (verify 447-488): the TABLE 1 / TABLE 2 content of an
    aggregated injection set (e.g. all per-eval injections of an annulus,
    ``near2_verify_annulus``) as a stand-alone ``verify_curve.txt``."""
    L = [f"# klip_tpe.verify detection-limit curve -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         f"# {header}",
         f"# contrast = companion/star flux ratio; binned by rho in {limits['rbin_as']:.2f}\" bins."]
    L += _table_lines(limits, indent="# ")
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text
