"""Multi-night source verification -- port of ``near2_verify.pro``.

Two independent "paths" are evaluated for every candidate ``(rho, theta)``:

* **Path A** -- matched-filter Mawet et al. (2014) S/N on the combined image, the
  search's (:func:`snr_fap`), with the small-sample false-alarm probability
  (Student-t, ``df = n_ap - 2``; ``near2v_snrfap`` / ``near2v_tfap``).  The decision
  is ``fap < fap_op``.
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
* Since 2026-11-01 (Kevin Wagner) every S/N of the verify is the search's
  (:func:`~klip_tpe.metrics.mawet_peak_snr`, through :func:`snr_fap`) under the run's rules
  (``rules=``, :func:`~klip_tpe.metrics.ring_rules`): its matched filter (a measured kernel
  too), its despike (6 sigma unless the run turns it off), the keep-outs around the sources
  as given, its radial band (the noise of a ring shorter than ``min_ring``, and a floor under
  the ring's scatter), its small-sample correction and its pixel mask.  ``near2v_snrfap``
  convolved a Gaussian kernel, kept ``exr`` around the located peak and had no band.  The
  bootstrap reads the peak of the same matched filter, the STIM annulus leaves the pixel mask
  out, and the FAP counts the ring samples used.  Without ``rules`` the search metric's
  defaults apply.  A run saved before 2026-11-01 keeps the verify it had
  (``RunConfig.verify_snr = "idl"``, :data:`~klip_tpe.metrics.VERIFY_IDL`): the IDL's
  ``near2v_snrfap`` with its Gaussian maps, and no pixel mask.
* Under the run's sidelobe rule (``sidelobes=``, :class:`~klip_tpe.metrics.Sidelobes`)
  every ring of Path A and of the subsets also leaves out the ADI arcs of the source, of
  the known companions and of the injection block's other sources, the STIM annulus
  leaves out the same, and the solved contrasts keep them out (:func:`ring_snr`).  Where
  fewer than 4 samples remain, the source's own arc narrows to let in the farthest, then
  the other sources' arcs and the known companions' are let in.  The IDL has no such rule,
  and ``sidelobes=None`` keeps its rings.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import t as _student_t

from . import metrics as _metrics
from .metrics import (_arc_regions, _in_arcs, gaussian_kernel, matched_filter, mawet_peak_snr, nanmedian_even,
                      radprof, rule_kwargs, sidelobe_snr, source_xy,
                      star_center, verify_idl)

__all__ = ["t_fap", "t_thresh", "contrast_limit", "contrast_limit_image", "mf_convolve",
           "snr_fap", "solved_contrasts", "combine_nights", "night_stim", "stim_fap", "bootstrap_nights",
           "night_subsets", "verify_maps", "verify_candidates", "verify_injections", "verify_tiles",
           "tile_comparison_lines", "limit_tables", "write_verify_table", "write_curve_table"]

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
    """``near2v_mfconv`` (verify 117-120): NaN -> 0 then the matched filter, the image
    correlated with the unit-sum FWHM Gaussian (``near2v_mfkern``, verify 53-59) with edge
    truncation (``mode='nearest'``).  The Gaussian is round, so a convolution gives the same
    map, and a ``kernel`` given is correlated, as a matched filter takes its template."""
    img = np.asarray(img, float)
    k = gaussian_kernel(fwhm) if kernel is None else kernel
    return ndimage.correlate(np.where(np.isfinite(img), img, 0.0), k, mode="nearest")


def _snr_fap_idl(img: np.ndarray, rho: float, theta: float, pxscale: float, fwhm: float,
                 exr: Optional[float] = None, srad: float = 1.5, *,
                 exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
                 angle_convention: str = "pa", excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                 peak_xy: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """:func:`snr_fap` as it measured before 2026-11-01, for runs that keep that verify
    (``RunConfig.verify_snr = "idl"``, :data:`~klip_tpe.metrics.VERIFY_IDL`): the matched-filter
    Mawet S/N and small-sample FAP of the IDL verify (``near2v_snrfap``, verify 122-154).

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
    on the early exits (as in the IDL return vector).  With samples, ``n_left`` is their
    count and ``bg`` and ``sigma`` their mean and standard deviation.

    ``excl_arcs`` (``(rho_arcsec, theta_deg, half_width_deg, radial_fwhm)``, as
    :func:`~klip_tpe.metrics.mawet_peak_snr` takes them) leaves out the samples in those
    arcs: ADI sidelobes, under the sidelobe rule (:func:`ring_snr`).  ``peak_xy`` places the
    peak there instead of searching for it.
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
    if peak_xy is not None:
        px, py = int(round(float(peak_xy[0]))), int(round(float(peak_xy[1])))
        if 0 <= px < nx and 0 <= py < ny and np.isfinite(img[py, px]):
            pkv = float(A[py, px])
        isr = -1
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
    out = {"snr": np.nan, "fap": np.nan, "nap": nap, "peak": peak, "x": px, "y": py, "r": r, "nclean": 0}
    if r < fwhm or nap < 5:
        out["why"] = "fewer than 5 apertures on its ring"
        return out
    th0 = np.arctan2(py - cy, px - cx)
    regions = _arc_regions(excl_arcs, pxscale, fwhm, cx, cy, angle_convention) if len(excl_arcs) else None
    vals: List[float] = []
    ring: Optional[List[List[Any]]] = [] if _metrics._RING_RECORDER is not None else None
    for q in range(1, nap):
        th = th0 + q * 2 * np.pi / nap
        xk, yk = int(round(cx + r * np.cos(th))), int(round(cy + r * np.sin(th)))
        if np.hypot(xk - px, yk - py) < exr:
            if ring is not None:
                ring.append([xk, yk, "source"])
            continue
        if any(np.hypot(xk - ex, yk - ey) < exr for ex, ey in exclude_xy):
            if ring is not None:
                ring.append([xk, yk, "excluded"])
            continue
        if not (0 <= xk < nx and 0 <= yk < ny) or not np.isfinite(img[yk, xk]):
            if ring is not None:
                ring.append([xk, yk, "off"])
            continue
        if regions and bool(_in_arcs(xk, yk, regions, cx, cy)):
            if ring is not None:
                ring.append([xk, yk, "arc"])
            continue
        vals.append(float(A[yk, xk]))
        if ring is not None:
            ring.append([xk, yk, "used"])
    nv = len(vals)
    out["nap"] = nv
    out["n_left"] = nv
    out["nclean"] = nv
    if ring is not None:
        if nv < 3:
            ring = [[a, b, "unused" if w == "used" else w] for a, b, w in ring]
        _metrics._ring_emit(_metrics._ring_rec(
            "snr_fap", img.shape, cx, cy, dict(out, nap=nap, nclean=nv), fwhm, exr, ring, [px], [py],
            [e[0] for e in exclude_xy], [e[1] for e in exclude_xy], excl_arcs,
            noise="ring" if nv >= 3 else None, excluded_label="excluded"))
    if nv < 3:
        return out
    use = np.array(vals)
    sd = float(np.std(use, ddof=1))
    out.update(bg=float(np.mean(use)), sigma=sd)
    if sd <= 0:
        return out
    snr = (float(A[py, px]) - float(np.mean(use))) / (sd * np.sqrt(1.0 + 1.0 / nv))
    out.update(snr=float(snr), fap=t_fap(snr, nv), peak=float(A[py, px]))
    return out


def snr_fap(img: np.ndarray, rho: float, theta: float, pxscale: float, fwhm: float,
            exr: Optional[float] = None, srad: float = 1.5, *,
            exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
            angle_convention: str = "pa", excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
            peak_xy: Optional[Tuple[float, float]] = None, rules: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The verify's S/N and its small-sample FAP at one source: the search's matched-filter
    Mawet S/N (:func:`~klip_tpe.metrics.mawet_peak_snr`) under the run's ring rules (``rules``,
    :func:`~klip_tpe.metrics.ring_rules`: its matched filter, despike, radial band, shortest
    ring, small-sample correction, pixel mask and inner guard), or that function's defaults.

    The source is the matched-filter peak within ``srad`` px of the nominal position, the ring
    runs through it and skips ``exr`` (default 1.5 FWHM) around the source as given and around
    every ``exclude_xy`` position (the other sources and the known companions), the pixel mask
    and ``excl_arcs``.  ``peak_xy`` places the peak there instead of searching for it, and
    ``mfmap`` is the matched-filter map made already (:func:`~klip_tpe.metrics.matched_filter`).
    ``fap = t_fap(snr, n)`` with ``n`` the ring samples used, or the ring's count when the
    radial band gave the noise.

    Since 2026-11-01 (Kevin Wagner) the verify and the candidate search measure as the search
    does.  Before, this function ported the IDL's ``near2v_snrfap`` (verify 122-154): a Gaussian
    kernel, no despike, ``exr`` kept around the located peak, the samples used counted in the
    small-sample correction and no radial band.  A run saved before then keeps that verify
    (``RunConfig.verify_snr = "idl"``), whose ``rules``
    (:data:`~klip_tpe.metrics.VERIFY_IDL`) hand the measure to :func:`_snr_fap_idl`.

    Returns ``{'snr', 'fap', 'nap', 'peak', 'x', 'y', 'r'}`` where ``nap`` is the ``n`` of the
    FAP on success and the ring's count on the early exits, with ``nclean`` the ring samples
    kept, ``n_left`` those left after the exclusions, ``noise`` (``"ring"`` or ``"band"``) and,
    with samples, ``bg`` and ``sigma``.
    """
    if verify_idl(rules):
        return _snr_fap_idl(img, rho, theta, pxscale, fwhm, exr, srad, exclude_xy=exclude_xy, mfmap=mfmap,
                            angle_convention=angle_convention, excl_arcs=excl_arcs, peak_xy=peak_xy)
    img = np.asarray(img, float)
    # a map made already needs no kernel
    kw = rule_kwargs(rules if mfmap is None else {k: v for k, v in (rules or {}).items() if k != "kernel_at"},
                     rho, img.shape)
    kw["excl_fwhm"] = (float(exr) / float(fwhm)) if exr is not None else float(kw.get("excl_fwhm", 1.5))
    kw["search_px"] = float(srad)
    if (rules or {}).get("flatten") and mfmap is None:
        img = radprof(img)
    snr, det = mawet_peak_snr(img, [float(rho)], [float(theta)], pxscale, fwhm, exclude_xy=list(exclude_xy),
                              mfmap=mfmap, angle_convention=angle_convention, excl_arcs=excl_arcs,
                              peak_xy=None if peak_xy is None else [tuple(peak_xy)], return_details=True, **kw)
    d = det[0]
    s = float(snr[0])
    nap = int(d.get("nap", 0) or 0)
    used = bool(d.get("ring_used"))
    n = int(d.get("nclean", 0)) if used else nap
    out = {"snr": s, "fap": t_fap(s, n) if (np.isfinite(s) and n >= 3) else float("nan"),
           "nap": n if np.isfinite(s) else nap, "peak": float(d.get("peak", np.nan)), "x": d["x"], "y": d["y"],
           "r": float(d["r"]), "nclean": int(d.get("nclean", 0) or 0), "n_left": int(d.get("n_left", 0) or 0),
           "noise": ("ring" if used else "band") if np.isfinite(s) else None}
    if d.get("why"):
        out["why"] = d["why"]
    if "sigma" in d and float(d.get("sigma", -1.0)) > 0:
        out.update(bg=float(d["bg"]), sigma=float(d["sigma"]))
    return out


def _fap_measure(img, rho, theta, pxscale, fwhm, *, known=(), excl_arcs=(), angle_convention="pa",
                 return_details=True, exr=None, srad=1.5, exclude_xy=(), mfmap=None, peak_xy=None, rules=None,
                 **_):
    """:func:`snr_fap` in the form :func:`~klip_tpe.metrics.sidelobe_snr` asks of a measure:
    one source, ``(snr array, [details])`` with ``nclean`` the ring samples used and ``nap`` the
    ring's apertures.  The known companions' keep-outs come in ``exclude_xy``."""
    s = snr_fap(img, float(rho[0]), float(theta[0]), pxscale, fwhm, exr, srad, exclude_xy=exclude_xy, mfmap=mfmap,
                angle_convention=angle_convention, excl_arcs=excl_arcs,
                peak_xy=None if peak_xy is None else peak_xy[0], rules=rules)
    nominal = int(np.floor(2 * np.pi * float(s["r"]) / fwhm)) if np.isfinite(s["r"]) else 0
    d = dict(s, nap=nominal, nclean=int(s.get("nclean", 0)), n_left=int(s.get("n_left", 0)))
    if not np.isfinite(s["peak"]):
        d["why"] = "no finite pixel at the source"
    elif s.get("why"):
        d["why"] = s["why"]
    return np.array([s["snr"]], float), [d]


_fap_measure.takes_peak_xy = True


def ring_snr(img: np.ndarray, i: int, rhos: Sequence[float], thetas: Sequence[float], pxscale: float,
             fwhm: float, sidelobes, *, exr: Optional[float] = None, srad: float = 1.5,
             exclude_xy: Sequence[Tuple[float, float]] = (), known: Sequence[Tuple[float, float]] = (),
             mfmap: Optional[np.ndarray] = None, angle_convention: str = "pa",
             rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """:func:`snr_fap` of source ``i`` of ``(rhos, thetas)`` under the run's sidelobe rule
    (:class:`~klip_tpe.metrics.Sidelobes`, :func:`~klip_tpe.metrics.sidelobe_snr`): its ring
    leaves out its own ADI arc, the other sources' and the ``known`` companions'.  Where fewer
    than ``min_apertures`` samples remain, its own arc narrows to let in the farthest, then
    the other sources' arcs are let in, then the known companions'.  The last is the ring
    without the rule.  The fallback is always the farthest samples: the verify has no clean
    ring through every source.  Returns :func:`snr_fap`'s dict with the rule's ``arc_deg``,
    ``n_arcs``, ``fallback`` (or ``fallback_tried``) and the ``arcs`` it left out."""
    from dataclasses import replace as _replace
    if exr is None:
        exr = 1.5 * fwhm
    rule = sidelobes if str(getattr(sidelobes, "mode", "farthest")) == "farthest" else _replace(sidelobes,
                                                                                                mode="farthest")
    # the run's ring rules, with the rule's shortest ring: where fewer are left the rule's fallback
    # takes over, not the radial band
    rk = dict(rules or {}, min_ring=int(rule.min_apertures))
    _, det = sidelobe_snr(img, list(rhos), list(thetas), pxscale, fwhm, rule, known=list(known),
                          measure=_fap_measure, angle_convention=angle_convention, only=int(i),
                          drop=("others", "known"), exr=exr, srad=srad, exclude_xy=list(exclude_xy),
                          mfmap=mfmap, rules=rk)
    d = dict(det[int(i)])
    d["nap"] = int(d.get("nclean", 0)) if np.isfinite(d.get("snr", np.nan)) else int(d.get("nap", 0))
    return d


def _without_self(known: Sequence[Tuple[float, float]], rho: float, theta: float, pxscale: float,
                  fwhm: float, exr: float, angle_convention: str) -> List[Tuple[float, float]]:
    """The known companions other than the one at ``(rho, theta)``, within ``exr`` px of it."""
    if not len(known):
        return []
    c = 0.0
    x0, y0 = source_xy([rho], [theta], pxscale, c, c, angle_convention)
    out = []
    for k in known:
        kx, ky = source_xy([k[0]], [k[1]], pxscale, c, c, angle_convention)
        if np.hypot(kx[0] - x0[0], ky[0] - y0[0]) >= exr:
            out.append((float(k[0]), float(k[1])))
    return out


def _noise(d: Dict[str, Any]) -> Dict[str, Any]:
    """How a ring came out under the sidelobe rule, for a verify row: the source's own arc as
    measured, the samples outside every arc, the fallback (None when none was needed)."""
    return {"arc_deg": float(d.get("arc_deg", float("nan"))), "n_arcs": int(d.get("n_arcs", 0)),
            "n_used": int(d.get("nclean", 0)), "fallback": d.get("fallback") or
            (f"{d['fallback_tried']} tried, {int(d.get('n_fallback', 0))} found" if d.get("fallback_tried") else None)}


class _ScaledImage:
    """The S/N of injected sources on ``clean + a (inj - clean)``, with the clean image's noise as
    it is and the sources' response scaled by ``a``.  Pixels not finite in both images are left
    out of both.  The map is the line through the run's two maps, ``mf(clean) + a (mf(inj) -
    mf(clean))``, each made as the run's rules make it (:func:`~klip_tpe.metrics.matched_filter`,
    at each source's separation for a measured kernel).  At ``a = 0`` and ``a = 1`` it is the
    search's own map of each image.  The convolution is linear, and only the despike, where it
    replaces a pixel, bends the map off that line in between."""

    def __init__(self, clean_img: np.ndarray, inj_img: np.ndarray, fwhm: float,
                 rules: Optional[Dict[str, Any]] = None):
        cl, ij = np.asarray(clean_img, float), np.asarray(inj_img, float)
        if cl.shape != ij.shape:
            raise ValueError(f"clean image {cl.shape} and injected image {ij.shape} differ in shape")
        both = np.isfinite(cl) & np.isfinite(ij)
        self.img0 = np.where(both, cl, np.nan)
        self.img1 = np.where(both, ij, np.nan)
        self._both = both
        self.fwhm, self.rules = float(fwhm), rules
        self._maps: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        # each (a, source) measured once: the thresholds' searches step through the same values
        self._snr: Dict[Any, Tuple[float, int]] = {}

    def maps(self, rho: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``(mf(clean), mf(inj) - mf(clean))`` with the kernel at ``rho``.  Under the measures of
        before 2026-11-01 (:data:`~klip_tpe.metrics.VERIFY_IDL`) the Gaussian maps of the clean
        image and of ``inj - clean``, as they were made then."""
        if verify_idl(self.rules):
            if "idl" not in self._maps:
                self._maps["idl"] = (mf_convolve(self.img0, self.fwhm),
                                     mf_convolve(np.where(self._both, self.img1 - self.img0, np.nan), self.fwhm))
            return self._maps["idl"]
        k = rule_kwargs(self.rules, rho if rho is not None else 0.0, self.img0.shape).get("kernel")
        key = None if k is None else np.asarray(k, float).tobytes()
        if key not in self._maps:
            m0 = matched_filter(self.img0, self.fwhm, self.rules, rho)
            m1 = matched_filter(self.img1, self.fwhm, self.rules, rho) - m0
            self._maps[key] = (m0, m1)
        return self._maps[key]

    @property
    def mf0(self) -> np.ndarray:
        return self.maps()[0]

    @property
    def mf1(self) -> np.ndarray:
        return self.maps()[1]

    def snr(self, a: float, rho: float, theta: float, pxscale: float, fwhm: float, exr: float, srad: float,
            exclude_xy: Sequence[Tuple[float, float]] = (), angle_convention: str = "pa",
            excl_arcs: Sequence[Tuple[float, float, float, float]] = ()) -> Tuple[float, int]:
        key = (float(a), float(rho), float(theta), float(pxscale), float(fwhm), float(exr), float(srad),
               tuple((float(x), float(y)) for x, y in exclude_xy), str(angle_convention),
               tuple(tuple(float(v) for v in arc) for arc in excl_arcs))
        hit = self._snr.get(key)
        if hit is not None:
            return hit
        m0, m1 = self.maps(rho)
        s = snr_fap(self.img0, rho, theta, pxscale, fwhm, exr, srad, exclude_xy=exclude_xy, mfmap=m0 + a * m1,
                    angle_convention=angle_convention, excl_arcs=excl_arcs, rules=self.rules)
        self._snr[key] = out = (float(s["snr"]), int(s["nap"]))
        return out


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
                     _scaled: Optional["_ScaledImage"] = None,
                     excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                     rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    hold best for an injection near them (NEAR2 note of 2026-10-23, revised).  ``excl_arcs``: the
    sidelobe rule's arcs, as the source's own S/N left them out (:func:`ring_snr`)."""
    out: Dict[str, Any] = {"c_det": float("nan"), "c_tap": [float("nan")] * len(taps), "snr1": float("nan"),
                           "nap1": 0}
    c0 = float(contrast)
    if not (np.isfinite(c0) and c0 > 0):
        return out
    if exr is None:
        exr = 1.5 * fwhm
    sc = _scaled if _scaled is not None else _ScaledImage(clean_img, inj_img, fwhm, rules=rules)
    args = (rho, theta, pxscale, fwhm, exr, srad, exclude_xy, angle_convention, excl_arcs)
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
             exclude_xy: Sequence[Tuple[float, float]] = (),
             excl_arcs: Sequence[Tuple[float, float, float, float]] = (), pxscale: Optional[float] = None,
             angle_convention: str = "pa", pixel_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Empirical STIM FAP (verify 594-602): the value at the rounded source pixel
    ranked against the annulus ``|r - rho_px| <= fwhm`` (finite pixels farther
    than ``keepout`` (default ``1.5 fwhm``) from the source and from every
    ``exclude_xy`` position: other sources, which are not noise).  ``excl_arcs``
    (with ``pxscale``) drops the pixels in those arcs too, the ADI sidelobes the
    source's ring left out (:func:`ring_snr`).  ``pixel_mask`` (True = left out, the run's
    disk mask) drops its pixels, as every noise ring does.  Returns ``(stim_value, fraction >=
    value)``."""
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
    if len(excl_arcs) and pxscale is not None:
        ann &= ~_in_arcs(xx, yy, _arc_regions(excl_arcs, pxscale, fwhm, cx, cy, angle_convention), cx, cy)
    if pixel_mask is not None and np.shape(pixel_mask) == stim.shape:
        ann &= ~np.asarray(pixel_mask, bool)
    if _metrics._RING_RECORDER is not None:
        _metrics._ring_emit({"fn": "stim_fap", "shape": tuple(stim.shape), "center": (float(cx), float(cy)),
                             "x": float(sx), "y": float(sy), "r": float(rho_px), "fwhm": float(fwhm),
                             "exr": float(keepout), "band_px": float(fwhm), "mask": ann.copy(),
                             "known_xy": [(float(a), float(b)) for a, b in exclude_xy],
                             "arcs": [tuple(float(v) for v in a) for a in excl_arcs], "ring": [],
                             "sources_xy": [(float(sx), float(sy))], "nclean": int(ann.sum()),
                             "noise": "annulus", "why": None, "excluded_label": "excluded"})
    if not ann.any() or not np.isfinite(val):
        return val, float("nan")
    return val, float(np.mean(stim[ann] >= val))


def bootstrap_nights(stack: np.ndarray, weights: np.ndarray, rho: float, theta: float,
                     pxscale: float, fwhm: float, srad: float = 1.5, n_boot: int = 500,
                     rng: Optional[np.random.Generator] = None, angle_convention: str = "pa",
                     rules: Optional[Dict[str, Any]] = None) -> float:
    """Bootstrap-over-nights significance (``near2v_boot``, verify 203-214):
    resample the ``nn`` nights with replacement ``n_boot`` times, recombine
    (weighted, no radprof), re-locate the matched-filter peak within ``srad`` and
    record its VALUE; result ``mean / std`` over finite draws (NaN if < 3 finite
    or std = 0).  DEVIATION: the peak value is read directly (the IDL ring
    exclusion ``exr = srad`` cannot affect it).  The matched filter is the run's
    (``rules``, :func:`~klip_tpe.metrics.matched_filter`)."""
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        idx = rng.integers(0, nn, nn)
        cb = combine_nights(stack, weights, idx)
        A = mf_convolve(cb, fwhm) if verify_idl(rules) else matched_filter(cb, fwhm, rules, rho)
        vals[b] = snr_fap(cb, rho, theta, pxscale, fwhm, exr=srad, srad=srad, mfmap=A,
                          angle_convention=angle_convention, rules=rules)["peak"]
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
    and, when ``fwhm`` is given, the Gaussian matched-filter map of every subset
    (``mf``, a look at the subsets: since 2026-11-01 each S/N makes the run's own map,
    :func:`snr_fap`).  Empty subsets (nn = 1) give all-NaN images.

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
                      path_a: Optional[str] = None, sidelobes=None,
                      rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
    sidelobes
        the run's sidelobe rule (:class:`~klip_tpe.metrics.Sidelobes`), or None for the
        rings of the IDL's verify (before 2026-11-01).  Each S/N's ring, Path A's and the
        subsets', leaves out the ADI arcs of the source and of the known companions (and in
        the injection block of the block's other sources), with the farthest fallback
        (:func:`ring_snr`), and the STIM annulus and the solved contrasts leave out what the
        source's Path-A ring left out.  A candidate at a known companion is not left out of
        its own ring as a companion.  Each row adds ``noise``: ``{arc_deg, n_arcs,
        fallback}``.
    rules
        the run's ring rules (:func:`~klip_tpe.metrics.ring_rules`): every S/N is the search's
        (:func:`snr_fap`), with its matched filter and pixel mask, and the STIM annulus and the
        bootstrap take the mask and the matched filter too.  None: the search metric's defaults.

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
        noise = None
        if sidelobes is None:
            # Path A on the combined image (verify 588), the search's S/N under the run's rules
            sa = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, drift_px,
                         exclude_xy=kxy, angle_convention=angle_convention, rules=rules)
            ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, drift_px,
                                  exclude_xy=kxy, angle_convention=angle_convention, rules=rules)["snr"]
                    for name in SUBSET_NAMES}
            arcs = ()
        else:
            # the sidelobe rule: the candidate's own arc and the other known companions'
            others = _without_self(known, rho, theta, pxscale, fwhm, exr, angle_convention)
            rk = dict(exr=exr, srad=drift_px, exclude_xy=kxy, known=others, angle_convention=angle_convention,
                      rules=rules)
            sa = ring_snr(maps["subsets"]["combined"], 0, [rho], [theta], pxscale, fwhm, sidelobes, **rk)
            ssub = {name: (sa["snr"] if name == "combined" else
                           ring_snr(maps["subsets"][name], 0, [rho], [theta], pxscale, fwhm, sidelobes, **rk)["snr"])
                    for name in SUBSET_NAMES}
            arcs = sa.get("arcs") or ()
            noise = _noise(sa)
        # Path B: STIM at the nominal (rounded) position with the annulus null (594-602)
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(stim, float(xs[0]), float(ys[0]), rho / pxscale, fwhm, exr, exclude_xy=kxy,
                              excl_arcs=arcs, pxscale=pxscale, angle_convention=angle_convention,
                              pixel_mask=(rules or {}).get("pixel_mask"))
        bsig = bootstrap_nights(stack, w, rho, theta, pxscale, fwhm, drift_px, n_boot, rng,
                                angle_convention, rules=rules) if (n_boot and nn >= 2) else float("nan")
        part = [ssub[k] for k in SUBSET_NAMES[1:]]
        persist = float(np.nanmin(part)) if np.any(np.isfinite(part)) else float("nan")
        rows.append({"kind": "candidate", "index": c + 1, "rho": rho, "theta": theta,
                     "x": sa["x"], "y": sa["y"], "snr": sa["snr"], "fap": sa["fap"], "nap": sa["nap"],
                     "peak": sa["peak"], "detect": bool(np.isfinite(sa["fap"]) and sa["fap"] < fap_op),
                     "subsets": ssub, "persistence": persist, "stim": sval, "stim_fap": sfap,
                     "boot_sig": bsig})
        if noise is not None:
            rows[-1]["noise"] = noise

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
                                         combined=combined_inj, clean=clean, sidelobes=sidelobes, rules=rules)
        rows.extend(irows)
        maps["injected"] = imaps
        maps["limits"] = imaps["limits"]
    maps["meta"] = {"nights": ids, "fap_op": fap_op, "drift_px": drift_px, "excl_fwhm": excl_fwhm,
                    "fwhm": fwhm, "pxscale": pxscale, "n_boot": n_boot, "path_a": path_a,
                    "sidelobes": None if sidelobes is None else sidelobes.describe()}
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
                      clean: Optional[np.ndarray] = None, sidelobes=None,
                      rules: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
    (:func:`_solved_bins`).  Not in the IDL.

    ``sidelobes`` (:class:`~klip_tpe.metrics.Sidelobes`): each source's rings leave out the ADI
    arcs of every source of the block and of the known companions, with the farthest fallback
    (:func:`ring_snr`), and its STIM annulus and solved contrasts leave out what its Path-A
    ring left out.  ``noise`` in each row says how its Path-A ring came out.  ``rules``: the
    run's ring rules, as :func:`verify_candidates` takes them."""
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
        scaled = _ScaledImage(clean, maps["subsets"]["combined"], fwhm, rules=rules)
    fap0 = float(list(fap_list)[0])
    items: List[Tuple[Dict[str, Any], Any]] = []
    rows: List[Dict[str, Any]] = []
    for m, src in enumerate(sources):
        rho, theta = float(src[0]), float(src[1])
        con = float(src[2]) if len(src) > 2 else float("nan")
        excl = kxy + sxy[:m] + sxy[m + 1:]            # the known sources and the block's others
        if sidelobes is None:
            s = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, drift_px,
                        exclude_xy=excl, angle_convention=angle_convention, rules=rules)
            ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, drift_px,
                                  exclude_xy=excl, angle_convention=angle_convention, rules=rules)["snr"]
                    for name in SUBSET_NAMES}
            arcs = ()
        else:
            rk = dict(exr=exr, srad=drift_px, exclude_xy=excl, known=list(known), angle_convention=angle_convention,
                      rules=rules)
            srho, sth = [float(v[0]) for v in sources], [float(v[1]) for v in sources]
            s = ring_snr(maps["subsets"]["combined"], m, srho, sth, pxscale, fwhm, sidelobes, **rk)
            ssub = {name: (s["snr"] if name == "combined" else
                           ring_snr(maps["subsets"][name], m, srho, sth, pxscale, fwhm, sidelobes, **rk)["snr"])
                    for name in SUBSET_NAMES}
            arcs = tuple(s.get("arcs") or ())
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(maps["stim"], float(xs[0]), float(ys[0]), rho / pxscale, fwhm, exr, exclude_xy=excl,
                              excl_arcs=arcs, pxscale=pxscale, angle_convention=angle_convention,
                              pixel_mask=(rules or {}).get("pixel_mask"))
        row = {"kind": "injection", "index": m + 1, "rho": rho, "theta": theta, "contrast": con,
               "x": s["x"], "y": s["y"], "snr": s["snr"], "fap": s["fap"], "nap": s["nap"],
               "peak": s["peak"], "subsets": ssub, "stim": sval, "stim_fap": sfap}
        if sidelobes is not None:
            row["noise"] = _noise(s)
        if scaled is not None and np.isfinite(con) and con > 0:
            sol = solved_contrasts(clean, maps["subsets"]["combined"], rho, theta, con, pxscale, fwhm, exr, drift_px,
                                   exclude_xy=excl, fap=fap0, taps=list(tap_list), angle_convention=angle_convention,
                                   _scaled=scaled, excl_arcs=arcs, rules=rules)
            row.update(c_det=sol["c_det"], c_tap_solved=list(sol["c_tap"]))
            args = (rho, theta, pxscale, fwhm, exr, drift_px, excl, angle_convention, arcs)
            items.append((row, lambda c, con=con, args=args: _detected(*scaled.snr(c / con, *args), fap0)))
        rows.append(row)
    maps["limits"] = limit_tables(rows, fwhm, pxscale, fap_list, tap_list, rbin_as)
    if maps["limits"] is not None and items:
        maps["limits"]["solved"] = _solved_bins(maps["limits"], items)
    # for TABLE 3 over several blocks (verify_tiles): each source's row and detection test
    maps["solved_items"] = items
    return rows, maps


def _place_key(s: Sequence[float]) -> Tuple[float, float]:
    """A source's place, ``(rho, PA)`` rounded, to tell whether an image holds it."""
    return (round(float(s[0]), 6), round(float(s[1]), 4))


def _tile_stack(stack: Optional[np.ndarray], img: np.ndarray,
                parts: Optional[Sequence[Any]]) -> Tuple[np.ndarray, Optional[List[str]]]:
    """An annulus' per-partition images and their ids, or its image alone."""
    if stack is None:
        return np.asarray(img, float)[None], None
    st = np.asarray(stack, float)
    if st.ndim == 2:
        st = st[None]
    ids = [str(p) for p in parts] if parts is not None and len(parts) == st.shape[0] else None
    return st, ids


def verify_tiles(annuli: Sequence[Dict[str, Any]], candidates: Sequence[Tuple[float, float]],
                 sources: Sequence[Tuple[float, ...]], fwhm: float, pxscale: float, *,
                 known: Sequence[Tuple[float, float]] = (), fap_op: float = 2.87e-7, drift_px: float = 1.5,
                 n_boot: int = 500, rng: Optional[np.random.Generator] = None, excl_fwhm: float = 1.5,
                 fap_list: Sequence[float] = DEFAULT_FAP_LIST, tap_list: Sequence[float] = DEFAULT_TAP_LIST,
                 rbin_as: Optional[float] = None, angle_convention: str = "pa", sidelobes=None,
                 rules: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The per-tile measure (NEAR2 s7 note, 2026-11-02): each candidate and each injected
    source measured on the image of the annulus whose tile holds it, the way the search and
    the validation measure an annulus' sources, instead of on the stitch, where a source's
    matched filter and its pad can take in the neighboring annuli' images.

    ``annuli`` holds one dict per annulus in the stitch's order: ``tile`` ``(rin, rout)`` px,
    ``clean`` and ``inj`` (its Path-A images as the stitch took them), ``clean_stack`` and
    ``inj_stack`` (its per-partition images, else the image alone), ``parts`` (their ids),
    ``inj_sources`` (the sources its injected image holds) and ``annulus`` (its index).  The
    tile holding a position is :func:`~klip_tpe.stitch.tile_of` of its separation.

    Each candidate is measured as :func:`verify_candidates` measures it on the stitch, its ring
    without the known companions.  Each source is measured as :func:`verify_injections` does,
    its ring without the other sources its image holds, those of the other tiles too.  A source
    whose tile's annulus did not inject it (a run whose annuli each injected only their own,
    ``RunConfig.stitch_injections = "own"``) gets a row marked ``absent`` and no S/N.

    Returns ``(rows, maps)``: the candidate rows, then the source rows, in the order given, each
    with ``annulus`` (1-based) and ``tile_px``, and ``maps["limits"]``, the detection-limit
    tables of the sources measured, TABLE 3 included.  A position outside every tile gets no
    row."""
    from .stitch import tile_of
    if rng is None:
        rng = np.random.default_rng(42)
    tiles = [(float(a["tile"][0]), float(a["tile"][1])) for a in annuli]
    nan = float("nan")
    crow: Dict[int, Dict[str, Any]] = {}
    srow: Dict[int, Dict[str, Any]] = {}
    items: List[Tuple[Dict[str, Any], Any]] = []
    nights: List[str] = []
    for j, a in enumerate(annuli):
        cj = [k for k, c in enumerate(candidates) if tile_of(float(c[0]) / pxscale, tiles) == j]
        sj = [m for m, s in enumerate(sources) if tile_of(float(s[0]) / pxscale, tiles) == j]
        if not cj and not sj:
            continue
        tag = {"annulus": int(a.get("annulus", j)) + 1, "tile_px": tiles[j]}
        clean = np.asarray(a["clean"], float)
        stack, ids = _tile_stack(a.get("clean_stack"), clean, a.get("parts"))
        nights += [p for p in (ids or []) if p not in nights]
        if cj:
            rows = verify_candidates(stack, ids, [candidates[k] for k in cj], fwhm, pxscale, known=known,
                                     fap_op=fap_op, drift_px=drift_px, n_boot=n_boot, rng=rng, excl_fwhm=excl_fwhm,
                                     fap_list=fap_list, tap_list=tap_list, rbin_as=rbin_as,
                                     angle_convention=angle_convention, combined=clean, sidelobes=sidelobes,
                                     rules=rules)
            for k, r in zip(cj, rows):
                r.update(index=k + 1, **tag)
                crow[k] = r
        if not sj:
            continue
        have = {_place_key(s) for s in (a.get("inj_sources") or [])}
        meas = [m for m in sj if _place_key(sources[m]) in have] if a.get("inj") is not None else []
        for m in sj:
            if m in meas:
                continue
            s = sources[m]
            srow[m] = {"kind": "injection", "index": m + 1, "rho": float(s[0]), "theta": float(s[1]),
                       "contrast": float(s[2]) if len(s) > 2 else nan, "x": nan, "y": nan, "snr": nan,
                       "fap": nan, "nap": 0, "peak": nan, "subsets": {n: nan for n in SUBSET_NAMES},
                       "stim": nan, "stim_fap": nan, "absent": True, **tag}
        if not meas:
            continue
        inj = np.asarray(a["inj"], float)
        istack, _ = _tile_stack(a.get("inj_stack"), inj, a.get("parts"))
        mine = {_place_key(sources[m]) for m in meas}
        others = [(float(s[0]), float(s[1])) for s in (a.get("inj_sources") or []) if _place_key(s) not in mine]
        irows, imaps = verify_injections(istack, [sources[m] for m in meas], fwhm, pxscale,
                                         known=list(known) + others, drift_px=drift_px, excl_fwhm=excl_fwhm,
                                         angle_convention=angle_convention, fap_list=fap_list, tap_list=tap_list,
                                         rbin_as=rbin_as, combined=inj, clean=clean, sidelobes=sidelobes,
                                         rules=rules)
        for m, r in zip(meas, irows):
            r.update(index=m + 1, **tag)
            srow[m] = r
        items += list(imaps.get("solved_items") or [])
    inj_rows = [srow[m] for m in sorted(srow)]
    measured = [r for r in inj_rows if not r.get("absent")]
    limits = limit_tables(measured, fwhm, pxscale, fap_list, tap_list, rbin_as) if measured else None
    if limits is not None and items:
        limits["solved"] = _solved_bins(limits, items)
    rows = [crow[k] for k in sorted(crow)] + inj_rows
    meta = {"nights": nights, "fap_op": fap_op, "drift_px": drift_px, "excl_fwhm": excl_fwhm, "fwhm": fwhm,
            "pxscale": pxscale, "n_boot": n_boot,
            "path_a": "the image of the annulus whose tile holds the position, as the stitch took it",
            "sidelobes": None if sidelobes is None else sidelobes.describe()}
    for r in rows:
        r["meta"] = meta
    return rows, {"limits": limits, "meta": meta}


def tile_comparison_lines(rows: Sequence[Dict[str, Any]], tile_rows: Sequence[Dict[str, Any]],
                          report: str = "tile_verify_report.txt") -> List[str]:
    """The closing section of the stitched report: each candidate's and source's S/N on the
    stitch beside its S/N on the image of the annulus whose tile holds it (:func:`verify_tiles`,
    in full in ``report``).  Every line is a comment."""
    def key(r):
        return (r.get("kind", "candidate"), int(r["index"]))

    tile = {key(r): r for r in tile_rows}
    rho_h = 'rho["]'
    L = ["#", f"# ==== the stitch beside the tiles ({report}) ====",
         "#   each S/N on the stitched image, and on the image of the annulus whose tile holds the position,",
         "#   the image the search and the validation measure that annulus' sources on.  In full in that report.",
         f"#   {'kind':<9} {'#':>3} {rho_h:>7} {'PA[deg]':>7} {'contrast':>10}  {'tile':>4}  {'S/N stitch':>10}  {'S/N tile':>8}"]
    for r in rows:
        t = tile.get(key(r))
        con = r.get("contrast", float("nan"))
        cstr = f"{con:10.3E}" if np.isfinite(con) and con > 0 else f"{'-':>10}"
        if t is None:
            where, st = "-", "outside every tile"
        elif t.get("absent"):
            where, st = f"a{t['annulus']}", "not injected there"
        else:
            where, st = f"a{t['annulus']}", _f(t["snr"])
        L.append(f"#   {r.get('kind', 'candidate'):<9} {int(r['index']):>3} {r['rho']:7.3f} {r['theta']:7.1f} {cstr}  "
                 f"{where:>4}  {_f(r['snr']):>10}  {st:>8}")
    return L


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


def _noise_lines(injs: Sequence[Dict[str, Any]]) -> List[str]:
    """The injection block's rings under the sidelobe rule, as comment lines: how many kept
    their samples outside every arc, and which took a fallback."""
    noted = [(r, r.get("noise") or {}) for r in injs]
    plain = sum(1 for _, n in noted if not n.get("fallback"))
    L = [f"#   sidelobe rule: {plain} of {len(noted)} rings on their samples outside every arc"
         + ("." if plain == len(noted) else ", the others:")]
    for r, n in noted:
        if n.get("fallback"):
            L.append(f"#     rho={r['rho']:.3f}\" PA={r['theta']:.1f}: {n['fallback']}, {int(n.get('n_used', 0))} samples, "
                     f"own arc +/- {float(n.get('arc_deg', float('nan'))):.0f} deg")
    return L


def write_verify_table(path: str, rows: Sequence[Dict[str, Any]], limits: Optional[Dict[str, Any]] = None,
                       header: Optional[Dict[str, Any]] = None, extra_lines: Sequence[str] = ()) -> str:
    """Write the ``verify_report.txt`` of ``near2_verify`` (verify 574-724) from
    the rows of :func:`verify_candidates`.  ``limits`` defaults to the tables
    derived from the injection rows present.  Rows of :func:`verify_tiles` also say whose
    image measured them.  ``extra_lines`` close the report.  Returns the text."""
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
    rule = meta.get("sidelobes")
    if rule:
        rot = rule.get("rotation_deg")
        span = "the floor" if rot is None else f"{float(rot):.0f} deg"
        L += ["# noise rings: the sidelobe rule.  Each ring leaves out the ADI arcs of the source, of the known "
              "companions and",
              f"#   of the injection block's other sources, +/- {span} (at least {float(rule.get('floor_fwhm', 2.0)):g} "
              f"FWHM of arc) within {float(rule.get('radial_fwhm', 1.5)):g} FWHM of their separations.",
              f"#   Where fewer than {int(rule.get('min_apertures', 4))} samples remain, the source's own arc narrows "
              "to let in the farthest, then the other sources'",
              "#   arcs are let in, then the known companions'.  The STIM annulus leaves out the same.", "#"]
    cands = [r for r in rows if r.get("kind", "candidate") == "candidate"]
    injs = [r for r in rows if r.get("kind") == "injection"]
    def _on(r):
        t = r.get("tile_px")
        return "" if "annulus" not in r else (f"annulus {r['annulus']}'s image" + (
            f", tile {float(t[0]):.2f}-{float(t[1]):.2f} px" if t is not None else ""))

    for r in cands:
        s = r["subsets"]
        L.append(f"== candidate {r['index']}  rho={r['rho']:.3f}\"  PA={r['theta']:.1f} deg ==")
        if "annulus" in r:
            L.append(f"   measured on {_on(r)}")
        fb = (r.get("noise") or {}).get("fallback")
        L.append(f"   [A] combined S/N = {_f(r['snr'])}   FAP = {_f(r['fap'], '{:10.2E}')}   (nap={int(r['nap'])}"
                 + (f", {fb}" if fb else "") + ")")
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
                 f"({float(meta.get('excl_fwhm', 1.5)):.1f} FWHM keep-out)"
                 + (", and the ADI arcs of the sidelobe rule (above)" if rule else ""))
        tiled = any("annulus" in r for r in injs)
        L.append("#   columns: rho[\"] PA[deg] contrast  | combined S/N  FAP  | injected-STIM  STIM-FAP"
                 + ("  | the annulus whose image measured it" if tiled else ""))
        for r in injs:
            s = r["subsets"]
            cstr = f"{r['contrast']:10.3E}" if np.isfinite(r.get("contrast", np.nan)) and r["contrast"] > 0 else "  (n/a)  "
            if r.get("absent"):
                t = r.get("tile_px")
                tt = "" if t is None else f" (tile {float(t[0]):.2f}-{float(t[1]):.2f} px)"
                L.append(f"   rho={r['rho']:.3f}\" PA={r['theta']:.1f}  c={cstr}  | not in annulus {r['annulus']}'s "
                         f"injected image{tt}: not measured")
                continue
            L.append(f"   rho={r['rho']:.3f}\" PA={r['theta']:.1f}  c={cstr}  | S/N={_f(r['snr'])}  FAP={_f(r['fap'], '{:10.2E}')}"
                     f"  | STIM={_f(r['stim'])}  STIM-FAP={_f(r['stim_fap'], '{:10.2E}')}"
                     + (f"  | a{r['annulus']}" if "annulus" in r else ""))
            L.append(f"       subset S/N  comb={_f(s['combined'])}  1st={_f(s['first-half'])}  2nd={_f(s['second-half'])}"
                     f"  even={_f(s['even'])}  odd={_f(s['odd'])}")
        if rule:
            L += _noise_lines([r for r in injs if not r.get("absent")])
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
    L += [str(v) for v in extra_lines]
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
