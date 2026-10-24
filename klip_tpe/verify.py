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
binned by separation (TABLE 1 / TABLE 2 of the IDL report).  With the clean
image beside the injected one, each injection's S/N is also solved on ``clean +
a (injected - clean)`` for the contrasts at which it reaches the thresholds
(:func:`solved_contrasts`), and TABLE 3 gives the limits and detected counts from
those, which the IDL report does not have.

The combined image of Path A can be given (``combined`` / ``combined_inj`` of
:func:`verify_candidates`).  The runner gives the image the run made, an annulus'
combined image or the final stitch, so the verify measures the product itself.
Without it Path A uses the weighted mean of the nights, as the IDL does.

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
* The driver calls ``near2v_snrfap`` without ``exclude`` (verify 588-591 and
  655-656), so a noise ring holds the known sources and, in the injection
  calibration block, the other injected sources.  We leave out of every ring the
  known sources, and out of an injection's ring (combined and subsets) the other
  injections of the block, within ``excl_fwhm`` FWHM, as the run's metric
  (:func:`~klip_tpe.metrics.mawet_peak_snr`, IDL ``near2_snrpk``) does.  The
  STIM FAP annulus drops the same positions within the same keep-out.  Two
  injections at one radius, 180 deg apart (the two-source layout of
  ``near2m_randpos``), otherwise put a ring sample on, or two beside, the other
  source.
* The driver measures Path A on ``near2v_combine`` of the per-night stitched
  images (verify 545-552, 588), which is not the run's stitched image.  At an edge
  between two annuli each tile reaches into the other annulus, a night kept by
  only one of the two holds that annulus' value there at full weight, and the
  stitch weighs the two annuli by their sensitivity, not by night.  A night
  dropped by a quiet annulus and kept by a noisy neighbor brings the neighbor's
  noise into the quiet annulus' edge (NEAR2 run_20261015_171115, annulus 2).  The
  runner gives the image it made as ``combined`` (and ``combined_inj``), and Path A
  runs there.  Path B (STIM, bootstrap, subsets) stays on the per-night images, and
  the ``combined`` subset is Path A's image.
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
           "snr_fap", "solved_contrasts", "combine_nights", "night_stim", "stim_fap", "bootstrap_nights",
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
    ``t.isf`` clipped to the IDL bisection range ``[-50, 50]``.

    The decision, :func:`t_fap`, is exact.  So where the exact threshold passes 50 (``n <= 6``
    at the operating FAP 2.87e-7, where it is 57, 157, 1320 and 1.1e6 for ``n`` = 6 to 3), a
    source at a limit made with this ``tau`` does not reach the FAP.  The report's TABLE 1 says
    where, and TABLE 3 takes the exact threshold (:func:`_t_exact`)."""
    if alpha <= 0:
        return 1.0e30
    df = max(int(n) - 2, 1)
    return float(np.clip(_student_t.isf(float(alpha), df), -50.0, 50.0))


#: :func:`t_thresh`'s cap, the IDL's bisection range
TAU_CAP = 50.0


def _t_exact(alpha: float, n: int) -> float:
    """:func:`t_thresh` without the cap: the S/N at which :func:`t_fap` is ``alpha``."""
    if alpha <= 0:
        return float("inf")
    return float(_student_t.isf(float(alpha), max(int(n) - 2, 1)))


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


class _ScaledImage:
    """The S/N of injected sources on ``clean + a (inj - clean)``, with the clean image's noise as
    it is and the sources' response scaled by ``a``.  Pixels not finite in both images are left
    out of both.  The matched filter is linear, so ``mf(clean + a D) = mf(clean) + a mf(D)``."""

    def __init__(self, clean_img: np.ndarray, inj_img: np.ndarray, fwhm: float):
        cl, ij = np.asarray(clean_img, float), np.asarray(inj_img, float)
        if cl.shape != ij.shape:
            raise ValueError(f"clean image {cl.shape} and injected image {ij.shape} differ in shape")
        both = np.isfinite(cl) & np.isfinite(ij)
        self.img0 = np.where(both, cl, np.nan)
        self.mf0 = mf_convolve(self.img0, fwhm)
        self.mf1 = mf_convolve(np.where(both, ij - cl, np.nan), fwhm)

    def snr(self, a: float, rho: float, theta: float, pxscale: float, fwhm: float, exr: float, srad: float,
            exclude_xy: Sequence[Tuple[float, float]] = (), angle_convention: str = "pa") -> Tuple[float, int]:
        s = snr_fap(self.img0, rho, theta, pxscale, fwhm, exr, srad, exclude_xy=exclude_xy,
                    mfmap=self.mf0 + a * self.mf1, angle_convention=angle_convention)
        return float(s["snr"]), int(s["nap"])


def _first_crossing(cond, lo: float, hi: float, n_grid: int) -> float:
    """The smallest ``x`` in ``[lo, hi]`` at which ``cond(x)`` holds, from ``n_grid`` values spaced
    evenly in log and a bisection of the first step that crosses.  ``inf`` when none holds, ``lo``
    when ``cond(lo)`` does."""
    grid = np.geomspace(float(lo), float(hi), int(n_grid))
    prev = None
    for x in grid:
        if cond(float(x)):
            if prev is None:
                return float(x)
            a, b = prev, float(x)
            for _ in range(64):
                if b / a - 1.0 < 1e-9:
                    break
                mid = float(np.sqrt(a * b))
                if cond(mid):
                    b = mid
                else:
                    a = mid
            return b
        prev = float(x)
    return float("inf")


def _detected(s: float, n: int, fap: float) -> bool:
    return bool(np.isfinite(s) and n >= 3 and t_fap(s, n) < fap)


def _reaches(s: float, n: int, fap: float, beta: float) -> bool:
    """``s`` reaches ``tau_fap + tau_(1-beta)``, both exact, as :func:`t_fap` decides."""
    return bool(np.isfinite(s) and n >= 3 and s >= _t_exact(fap, n) + _t_exact(1.0 - beta, n))


def solved_contrasts(clean_img: np.ndarray, inj_img: np.ndarray, rho: float, theta: float, contrast: float,
                     pxscale: float, fwhm: float, exr: Optional[float] = None, srad: float = 1.5, *,
                     exclude_xy: Sequence[Tuple[float, float]] = (), fap: float = DEFAULT_FAP_LIST[0],
                     taps: Sequence[float] = DEFAULT_TAP_LIST, angle_convention: str = "pa",
                     a_range: Tuple[float, float] = (1e-3, 1e3), n_grid: int = 121,
                     _scaled: Optional["_ScaledImage"] = None) -> Dict[str, Any]:
    """The contrasts at which an injected source reaches the verify's thresholds, solved on
    its own image.

    The source's S/N at contrast ``a * contrast`` is :func:`snr_fap` on ``clean + a (inj -
    clean)`` with the exclusions given.  The image's noise stays as it is, the noise at the
    source's position included, and only the source's response scales with ``a``, its
    residual in its own noise ring included.  At ``a = 1`` it is the injected image's own S/N
    (``snr1``).  Pixels not finite in both images are left out of both.

    Returns ``c_det``, the faintest contrast at which the source is detected at ``fap`` (its
    FAP below ``fap``), and ``c_tap``, for each ``taps`` entry beta the faintest contrast at
    which its S/N reaches ``tau_fap + tau_(1-beta)``, both exact (:func:`t_thresh` caps them
    at 50).  That is the single-source limit of :func:`contrast_limit` solved, where
    :func:`contrast_limit` scales the whole S/N at the injected contrast, noise included.  A contrast is ``inf`` where no ``a`` up to
    ``a_range[1]`` gets there, and ``a_range[0] * contrast`` where even that does.  The search
    steps through ``n_grid`` values of ``a`` spaced evenly in log and bisects the first step
    that crosses.  The S/N need not grow with ``a``.  The source's own residual in its ring
    grows with it, and the peak search can move to a pixel at another radius, whose ring
    differs.  So the faintest contrast that gets there is not always one above which the source
    stays there.  KLIP's response to a bright source is not linear either, so the contrasts
    hold best for an injection near them (NEAR2 note of 2026-10-23, revised)."""
    out: Dict[str, Any] = {"c_det": float("nan"), "c_tap": [float("nan")] * len(taps), "snr1": float("nan"),
                           "nap1": 0}
    c0 = float(contrast)
    if not (np.isfinite(c0) and c0 > 0):
        return out
    if exr is None:
        exr = 1.5 * fwhm
    sc = _scaled if _scaled is not None else _ScaledImage(clean_img, inj_img, fwhm)
    args = (rho, theta, pxscale, fwhm, exr, srad, exclude_xy, angle_convention)
    lo, hi = float(a_range[0]), float(a_range[1])
    a_det = _first_crossing(lambda a: _detected(*sc.snr(a, *args), fap), lo, hi, n_grid)
    a_tap = [_first_crossing(lambda a, b=float(b): _reaches(*sc.snr(a, *args), fap, b), lo, hi, n_grid)
             for b in taps]
    s1, n1 = sc.snr(1.0, *args)
    out.update(c_det=c0 * a_det, c_tap=[c0 * a for a in a_tap], snr1=s1, nap1=n1)
    return out


def _solved_bins(lim: Dict[str, Any], items: Sequence[Tuple[Dict[str, Any], Any]], n_grid_decade: int = 20,
                 a_range: Tuple[float, float] = (1e-3, 1e3)) -> Optional[List[Dict[str, Any]]]:
    """TABLE 3 in the bins of TABLE 1 (``lim``, from :func:`limit_tables`), and in bins of the same
    width beyond them.  ``items`` are ``(row, detected_at)`` per injection with solved contrasts,
    ``detected_at(c)`` telling whether the source is detected at contrast ``c`` (its S/N on its own
    image, scaled to ``c``).  With beta = ``tap_list[i90]``, each bin gives the following.

    * ``n``, its sources.
    * ``c_tap``, the mean of the rows' finite ``c_tap_solved``.
    * ``c_det_beta``, the faintest contrast at which at least ``ceil(beta n)`` of the bin's
      sources are detected, all of them at that contrast.
    * ``n_det_t1``, the sources detected at TABLE 1's ``c@TAP`` for beta in the same bin
      (``None`` where TABLE 1 has no such bin).
    * ``rows``, ``(rho, theta, contrast, c_det, c@TAP for beta)`` per source."""
    if not items:
        return None
    rbw, i90 = lim["rbin_as"], lim["i90"]
    smp = lim["samples"]["rho"]
    rmin, rmax = float(np.min(smp)), float(np.max(smp))
    nb = max(int(np.ceil((rmax - rmin) / rbw)), 1) if rmax > rmin else 1
    beta = float(lim["tap_list"][i90])
    t1 = {b["ib"]: b for b in lim["bins"]}

    def bin_of(rho: float) -> int:
        raw = int(np.floor((rho - rmin) / rbw))
        if rmin <= rho <= rmax:
            return min(raw, nb - 1) if nb > 1 else 0
        return raw

    groups: Dict[int, List[Tuple[Dict[str, Any], Any]]] = {}
    for row, det in items:
        groups.setdefault(bin_of(float(row["rho"])), []).append((row, det))
    out = []
    for bb in sorted(groups):
        grp = groups[bb]
        n = len(grp)
        c_tap = []
        for t in range(len(lim["tap_list"])):
            v = np.array([float(r["c_tap_solved"][t]) for r, _ in grp], float)
            v = v[np.isfinite(v)]
            c_tap.append(float(np.mean(v)) if v.size else float("inf"))
        need = max(int(np.ceil(beta * n - 1e-9)), 1)
        cs = [float(r["contrast"]) for r, _ in grp]
        lo, hi = min(cs) * float(a_range[0]), max(cs) * float(a_range[1])
        ngrid = int(np.ceil(np.log10(hi / lo) * n_grid_decade)) + 1
        c_beta = _first_crossing(lambda c: sum(bool(d(c)) for _, d in grp) >= need, lo, hi, ngrid)
        ref = t1.get(bb)
        n_det = None
        if ref is not None and np.isfinite(ref["c_tap"][i90]) and ref["c_tap"][i90] > 0:
            n_det = int(sum(bool(d(float(ref["c_tap"][i90]))) for _, d in grp))
        out.append({"sep": float(np.mean([r["rho"] for r, _ in grp])), "n": n, "c_tap": c_tap,
                    "c_det_beta": c_beta, "n_det_t1": n_det, "ib": bb,
                    "rows": [(float(r["rho"]), float(r["theta"]), float(r["contrast"]), float(r["c_det"]),
                              float(r["c_tap_solved"][i90])) for r, _ in grp]})
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
             keepout: Optional[float] = None, *,
             exclude_xy: Sequence[Tuple[float, float]] = ()) -> Tuple[float, float]:
    """Empirical STIM FAP (verify 594-602): the value at the rounded source pixel
    ranked against the annulus ``|r - rho_px| <= fwhm`` (finite pixels farther
    than ``keepout`` (default ``1.5 fwhm``) from the source and from every
    ``exclude_xy`` position: other sources, which are not noise).  Returns
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
    for ex, ey in exclude_xy:
        ann &= np.hypot(xx - ex, yy - ey) > keepout
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


def verify_maps(stack: np.ndarray, weights: Optional[np.ndarray] = None, fwhm: Optional[float] = None,
                flatten: bool = False, *, combined: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Subset images (weighted combines, verify 545-552), the night-STIM map (555)
    and, when ``fwhm`` is given, the matched-filter map of every subset.  Empty
    subsets (nn = 1) give all-NaN images.

    ``flatten`` applies :func:`radprof` to each combine, as the IDL does; off by
    default so the subsets match what the run scored.  ``combined``, an image of the
    stack's frame size, replaces the combine of all nights (the run's own combined
    image, which Path A measures)."""
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    w = np.ones(nn) if weights is None else np.asarray(weights, float)
    subs = night_subsets(nn)
    if combined is not None:
        combined = np.asarray(combined, float)
        if combined.shape != stack.shape[1:]:
            raise ValueError(f"verify_maps: combined image {combined.shape} is not the stack's frame size "
                             f"{stack.shape[1:]}")
    imgs: Dict[str, np.ndarray] = {}
    for name, idx in subs.items():
        if name == "combined" and combined is not None:
            imgs[name] = radprof(combined) if flatten else combined
            continue
        if not idx.size:
            imgs[name] = np.full(stack.shape[1:], np.nan)
            continue
        c = combine_nights(stack, w, idx)
        imgs[name] = radprof(c) if flatten else c
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
                      angle_convention: str = "pa", return_maps: bool = False,
                      combined: Optional[np.ndarray] = None, combined_inj: Optional[np.ndarray] = None,
                      path_a: Optional[str] = None) -> List[Dict[str, Any]]:
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
        real sources left out of the noise rings and the STIM annulus (verify
        ``exclude``).
    injected
        optional ``(inj_stack, sources)`` with ``sources = [(rho, theta,
        contrast), ...]``: the injected twin of ``stack`` used for the
        calibration block and the detection-limit tables.  Each source's noise
        is measured without the other sources (:func:`verify_injections`), and
        each source's contrasts are solved on its own image
        (:func:`solved_contrasts`, TABLE 3).
    weights
        per-night weights (IDL: ``sqrt(texp)``); default equal.
    rng
        ``numpy.random.Generator`` for the bootstrap (IDL ``seed=42``);
        ``n_boot=0`` skips it (``/lite``).
    combined, combined_inj
        the image Path A measures and its injected twin, for the image the run made
        from these nights (an annulus' combined image, the final stitch).  By default
        the weighted mean of the nights, as the IDL does.  With ``injected``, give
        both or neither.  ``path_a`` names the image in the report.

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
    if injected is not None and (combined is None) != (combined_inj is None):
        raise ValueError("verify_candidates: give combined and combined_inj together, so the injections are "
                         "measured on the injected twin of the image the candidates are measured on")
    if path_a is None:
        path_a = ("the combined image given" if combined is not None else
                  "the equal-weight mean of the nights" if weights is None else "the weighted mean of the nights")

    maps = verify_maps(stack, w, fwhm, combined=combined)
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
        sval, sfap = stim_fap(stim, float(xs[0]), float(ys[0]), rho / pxscale, fwhm, exr, exclude_xy=kxy)
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
        inj_stack = np.asarray(inj_stack, float)
        if inj_stack.ndim == 2:
            inj_stack = inj_stack[None]
        # the clean twin of the injections' Path-A image, for the solved contrasts: the image
        # given, or the clean nights combined as the injected ones are
        clean = maps["subsets"]["combined"] if (combined is not None or inj_stack.shape == stack.shape) else None
        irows, imaps = verify_injections(inj_stack, sources, fwhm, pxscale, known=known,
                                         weights=w[: inj_stack.shape[0]], drift_px=drift_px,
                                         excl_fwhm=excl_fwhm, angle_convention=angle_convention,
                                         fap_list=fap_list, tap_list=tap_list, rbin_as=rbin_as,
                                         combined=combined_inj, clean=clean)
        rows.extend(irows)
        maps["injected"] = imaps
        maps["limits"] = imaps["limits"]
    maps["meta"] = {"nights": ids, "fap_op": fap_op, "drift_px": drift_px, "excl_fwhm": excl_fwhm,
                    "fwhm": fwhm, "pxscale": pxscale, "n_boot": n_boot, "path_a": path_a}
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
                      rbin_as: Optional[float] = None, combined: Optional[np.ndarray] = None,
                      clean: Optional[np.ndarray] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Injection calibration block (verify 621-735): Path-A S/N / FAP, subset S/N,
    injected-STIM value + annulus FAP per injected source ``(rho, theta,
    contrast)``, then the detection-limit tables from sources with finite
    ``snr > 0`` and ``contrast > 0``.

    Each source's noise ring (combined image and subsets) leaves out the other
    sources and the ``known`` ones within ``excl_fwhm`` FWHM of their nominal
    positions, as the run's metric does, and its STIM annulus drops them, and the
    source itself, within the same keep-out.  So a source's S/N is what it would
    be injected alone, to the noise.  DEVIATIONS: the IDL (verify 655-656, 660)
    leaves every other source in the ring and the annulus, and keeps no keep-out
    around the source in the STIM annulus.

    ``combined`` is the injected image Path A measures (by default the weighted mean of
    the nights).  ``clean``, the clean twin of that image, adds each source's contrasts
    solved on its own image (:func:`solved_contrasts`), ``c_det`` and ``c_tap_solved`` (one
    per ``tap_list`` entry, at ``fap_list[0]``), and TABLE 3 as ``maps['limits']['solved']``
    (:func:`_solved_bins`).  Not in the IDL."""
    inj_stack = np.asarray(inj_stack, float)
    if inj_stack.ndim == 2:
        inj_stack = inj_stack[None]
    nn, ny, nx = inj_stack.shape
    cx, cy = star_center((ny, nx))
    w = np.ones(nn) if weights is None else np.asarray(weights, float)[:nn]
    exr = excl_fwhm * fwhm
    kxy = _known_xy(known, pxscale, cx, cy, angle_convention)
    sxy = _known_xy([(s[0], s[1]) for s in sources], pxscale, cx, cy, angle_convention)
    maps = verify_maps(inj_stack, w, fwhm, combined=combined)
    scaled = None
    if clean is not None:
        clean = np.asarray(clean, float)
        if clean.shape != (ny, nx):
            raise ValueError(f"verify_injections: clean image {clean.shape} is not the frame size {(ny, nx)}")
        scaled = _ScaledImage(clean, maps["subsets"]["combined"], fwhm)
    fap0 = float(list(fap_list)[0])
    items: List[Tuple[Dict[str, Any], Any]] = []
    rows: List[Dict[str, Any]] = []
    for m, src in enumerate(sources):
        rho, theta = float(src[0]), float(src[1])
        con = float(src[2]) if len(src) > 2 else float("nan")
        excl = kxy + sxy[:m] + sxy[m + 1:]            # the known sources and the block's others
        s = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, drift_px,
                    exclude_xy=excl, mfmap=maps["mf"]["combined"], angle_convention=angle_convention)
        ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, drift_px,
                              exclude_xy=excl, mfmap=maps["mf"][name], angle_convention=angle_convention)["snr"]
                for name in SUBSET_NAMES}
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(maps["stim"], float(xs[0]), float(ys[0]), rho / pxscale, fwhm, exr, exclude_xy=excl)
        row = {"kind": "injection", "index": m + 1, "rho": rho, "theta": theta, "contrast": con,
               "x": s["x"], "y": s["y"], "snr": s["snr"], "fap": s["fap"], "nap": s["nap"],
               "peak": s["peak"], "subsets": ssub, "stim": sval, "stim_fap": sfap}
        if scaled is not None and np.isfinite(con) and con > 0:
            sol = solved_contrasts(clean, maps["subsets"]["combined"], rho, theta, con, pxscale, fwhm, exr, drift_px,
                                   exclude_xy=excl, fap=fap0, taps=list(tap_list), angle_convention=angle_convention,
                                   _scaled=scaled)
            row.update(c_det=sol["c_det"], c_tap_solved=list(sol["c_tap"]))
            args = (rho, theta, pxscale, fwhm, exr, drift_px, excl, angle_convention)
            items.append((row, lambda c, con=con, args=args: _detected(*scaled.snr(c / con, *args), fap0)))
        rows.append(row)
    maps["limits"] = limit_tables(rows, fwhm, pxscale, fap_list, tap_list, rbin_as)
    if maps["limits"] is not None and items:
        maps["limits"]["solved"] = _solved_bins(maps["limits"], items)
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
    (nap), 1)``.  Returns ``None`` when no injection qualifies.

    ``solved`` is None here.  :func:`verify_injections` fills it, TABLE 3, when it has
    the clean twin of the injected image (:func:`_solved_bins`)."""
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
            "ib": bb,
            # the largest exact tau of the bin's injections (contrast_limit takes each one's own)
            "tau_exact": max(_t_exact(fap_list[0], max(int(round(v)), 3)) for v in napi[ww]),
        })
    return {"rbin_as": rbw, "fap_list": fap_list, "tap_list": tap_list, "i90": i90, "n_inj": int(ri.size),
            "bins": bins, "samples": {"rho": ri, "snr": si, "contrast": ci, "nap": napi}, "solved": None}


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
    capped = [b for b in lim["bins"] if b.get("tau_exact", 0.0) > TAU_CAP]
    if capped:
        def _tau(v):
            return f"{v:.0f}" if v < 1e4 else f"{v:.1E}"
        where = [f"{_tau(b['tau_exact'])} at {b['sep']:.2f}\"" for b in capped]
        where = where[0] if len(where) == 1 else ", ".join(where[:-1]) + " and " + where[-1]
        out.append(f"{indent}(tau is capped at {TAU_CAP:.0f}, as in the IDL.  The exact tau is {where}, so a source at")
        out.append(f"{indent} TABLE 1's c@TAP there does not reach the operating FAP.  TABLE 3 takes the exact tau.)")
    out.append(f"{indent}TABLE 2 -- TAP{int(round(100*tl[i90]))} (~90%) completeness contrast vs separation, by FAP threshold")
    out.append(f"{indent}  rho[\"] " + "".join(f"   {a:8.1E}/rs" for a in fl) + "   10%/img")
    for b in lim["bins"]:
        out.append(f"     {b['sep']:5.2f}" + "".join(f"  {c:10.3E}" for c in b["c_fap"]) + f"  {b['c_img10']:10.3E}")
    out.append(f"{indent}(FAP columns are per-resel; last col = per-image 10%, i.e. per-resel x n_resel at that rho.)")
    return out


def _e(v: float) -> str:
    v = float(v)
    return f"{v:10.3E}" if np.isfinite(v) else f"{'inf' if np.isinf(v) else 'nan':>10}"


def _table3_lines(lim: Dict[str, Any], indent: str = "#   ") -> List[str]:
    """TABLE 3 of the report, from ``lim['solved']`` (:func:`limit_tables`)."""
    fl, tl, i90 = lim["fap_list"], lim["tap_list"], lim["i90"]
    pc = int(round(100 * tl[i90]))
    out = [f"{indent}TABLE 3 -- solved on clean + a (injected - clean), at the OPERATING FAP ({fl[0]:8.1E} per resel)",
           f"{indent}  rho[\"]  n_inj" + "".join(f"   c@TAP{int(round(t*100)):02d}" for t in tl)
           + f"   c({pc}%det)  det@T1"]
    for b in lim["solved"]:
        nd = "-" if b["n_det_t1"] is None else f"{b['n_det_t1']}/{b['n']}"
        out.append(f"     {b['sep']:5.2f}   {b['n']:4d}" + "".join(f"  {_e(c)}" for c in b["c_tap"])
                   + f"  {_e(b['c_det_beta'])}  {nd:>6}")
    out += [f"{indent}(Each source's S/N is TABLE 1's S/N, measured on clean + a (injected - clean), a being the",
            f"{indent} contrast over the injected one.  The image's noise stays as it is, the noise at the source",
            f"{indent} included, and only the source's response scales, its residual in its own ring included.",
            f"{indent} c@TAP is the bin mean of the contrasts at which each source reaches tau + tau_(1-TAP), TABLE 1",
            f"{indent} solved instead of scaled, with the exact tau where TABLE 1 caps it at {TAU_CAP:.0f} as the IDL does.",
            f"{indent} c({pc}%det) is the faintest contrast at which {pc} % of the bin's",
            f"{indent} sources are detected at the operating FAP, and det@T1 counts those detected at TABLE 1's",
            f"{indent} c@TAP{pc}.  inf means no contrast up to 1000 times the injected one gets there.  The scaling",
            f"{indent} takes KLIP's response to a source as linear, so these hold best for injections near the limit.)",
            f"{indent}each source, rho[\"]  PA[deg]  contrast  detected from  c@TAP{pc}"]
    for b in lim["solved"]:
        for rho, th, c, cd, ct in b["rows"]:
            out.append(f"{indent}  {rho:7.3f}  {th:7.1f}  {_e(c)}  {_e(cd)}  {_e(ct)}")
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
    L += ["#", f"# PATH A = {meta.get('path_a', 'the combined image')} (Mawet small-sample t-FAP)",
          "# PATH B = night-to-night (STIM + bootstrap-over-nights + subset S/N, whose comb is the PATH A image)", "#"]
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
        L.append(f"#   each source's noise ring leaves out the other sources and the known ones "
                 f"({float(meta.get('excl_fwhm', 1.5)):.1f} FWHM keep-out)")
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
            if limits.get("solved"):
                L.append("#")
                L += _table3_lines(limits)
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
