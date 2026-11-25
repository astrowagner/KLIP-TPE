"""Multi-night source verification, two independent paths for every candidate ``(rho, theta)``.

* Path A is the search's matched-filter S/N on the combined image (Mawet et al. 2014,
  :func:`snr_fap`), with the small-sample false-alarm probability (Student t, ``df = n_ap - 2``).
  The decision is ``fap < fap_op``.  A source whose position is given, a known companion or an
  injection, is read at that position.  A candidate a blind search found is read at the matched
  filter's peak within FWHM/2 of where it was found, and its FAP also comes corrected for the search
  over the field (:func:`field_fap`).
* Path B is the night-to-night consistency: the night-STIM map with an empirical FAP from the
  same-radius annulus, a bootstrap over nights of the matched-filter peak, and the S/N in five
  semi-independent subsets (combined, first half, second half, even and odd nights), whose
  minimum over the four partitions is the persistence.

With an injected twin of the night stack, the same machinery runs at the injected positions, and
the Path-A detection limits ``c_lim = c_inj (tau_alpha + tau_{1-beta}) / SNR_inj`` are binned by
separation (TABLE 1 and TABLE 2).  With the clean image beside the injected one, each injection's
S/N is also solved on ``clean + a (injected - clean)`` for the contrasts at which it reaches the
thresholds (:func:`solved_contrasts`), which TABLE 3 reports.  The runner gives Path A the image
the run made (``combined``, ``combined_inj``), an annulus' combined image or the final stitch.
Without it Path A uses the weighted mean of the nights.

Images are ``(ny, nx)`` arrays indexed ``[y, x]``, with the star at ``((nx-1)/2, (ny-1)/2)``.
``theta`` is a position angle East of North, ``x = cx + r cos(theta+90)``, ``y = cy + r sin(theta+90)``.

Notes
-----
The module succeeds ``near2_verify.pro`` of the IDL predecessor, from which it departs on purpose
where listed here, each also flagged at its call site:

* ``near2v_tthresh`` bisects ``tau`` on [0, 50], and :func:`t_thresh` uses ``scipy.stats.t.isf``
  clipped to the same range, so extreme thresholds (``df = 1``) agree with the predecessor's.
* The predecessor's injected-STIM FAP (verify 660) omits the 1.5 FWHM keep-out around the source
  that its candidate FAP (verify 601) applies.  The keep-out applies to both here (brief C.6).
* The predecessor's driver calls ``near2v_snrfap`` without ``exclude`` (verify 588-591, 655-656), so
  a noise ring holds the known sources and the other injections of a calibration block.  Here
  every ring leaves out the known sources, and an injection's ring the block's other injections,
  within ``excl_fwhm`` FWHM, as the run's metric does.  Two injections at one radius 180 degrees
  apart otherwise put a ring sample on, or two beside, the other source.
* The predecessor measures Path A on ``near2v_combine`` of the per-night stitched images (verify
  545-552, 588), which is not the run's stitched image.  At an edge between two annuli a night
  dropped by a quiet annulus and kept by a noisy neighbor brings the neighbor's noise into the
  quiet annulus' edge (NEAR2 run_20261015_171115, annulus 2), so Path A runs on the run's own
  image.  Path B stays on the per-night images, and its ``combined`` subset is Path A's image.
* The median of the per-bin S/N is ``numpy.median``, the mean of the two middle values, rather than
  the upper middle that the predecessor's IDL ``median`` takes.
* Since 2026-11-01 (Kevin Wagner) every S/N of the verify is the search's
  (:func:`~klip_tpe.metrics.mawet_peak_snr`, through :func:`snr_fap`) under the run's rules
  (``rules``, :func:`~klip_tpe.metrics.ring_rules`), with its matched filter, despike, keep-outs,
  radial band, small-sample correction and pixel mask.  ``near2v_snrfap`` convolved a Gaussian
  kernel, kept ``exr`` around the located peak and had no band.  A run saved before keeps the verify
  it had (``RunConfig.verify_snr = "idl"``, :data:`~klip_tpe.metrics.VERIFY_IDL`).
* Under the run's sidelobe rule (``sidelobes``, :class:`~klip_tpe.metrics.Sidelobes`) every ring of
  Path A and of the subsets also leaves out the ADI arcs of the source, of the known companions and
  of the block's other sources, as do the STIM annulus and the solved contrasts (:func:`ring_snr`).
  The predecessor has no such rule, and ``sidelobes=None`` keeps its rings.
* Since 2026-11-25 (the audit of that date) a source whose position is given is read at it
  (:func:`search_px`).  The predecessor took every source's matched-filter peak within 1.5 px and
  read the FAP as if at a fixed pixel.  On source-free Gaussian noise correlated at the PSF that
  peak's S/N averages +0.29 to +0.75 sigma (FWHM 6.4 to 2.3 px), and its FAP is reached 1.6 to 4.6
  times as often as it says, between 0.1 and 1e-4.  At the given position the rate equals the FAP.
  A run saved before keeps the 1.5 px search (``RunConfig.verify_position = "peak"``).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

from . import metrics as _metrics
from .metrics import (_arc_regions, _in_arcs, gaussian_kernel, matched_filter, mawet_peak_snr, nanmedian_even,
                      radprof, rule_kwargs, sidelobe_snr, source_xy,
                      star_center, verify_idl)

__all__ = ["t_fap", "t_thresh", "legacy_position", "search_px", "field_fap", "field_smoothness", "field_geometry",
           "contrast_limit", "contrast_limit_image", "mf_convolve",
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
    """Return Mawet et al. (2014)'s small-sample false-alarm probability per resolution element.

    It is the Student-t survival ``P(T > snr)`` with ``df = max(n - 2, 1)``.

    Parameters
    ----------
    snr : float
        The S/N.
    n : int
        Number of noise apertures used.

    Returns
    -------
    fap : float
        The probability, exactly 0.5 for ``snr <= 0`` as in the IDL predecessor, NaN for a non-finite
        S/N.

    Notes
    -----
    It is ``near2v_tfap`` of the IDL predecessor (verify 66-72).
    """
    if not np.isfinite(snr):
        return float("nan")
    df = max(int(n) - 2, 1)
    if snr <= 0:
        return 0.5
    return float(_student_t.sf(float(snr), df))


def t_thresh(alpha: float, n: int) -> float:
    """Return the S/N threshold ``tau`` at which :func:`t_fap` is ``alpha``, capped at 50.

    It is ``t.isf`` clipped to [-50, 50], the IDL predecessor's bisection range.  The decision,
    :func:`t_fap`, is exact, so where the exact threshold passes 50 (``n <= 6`` at the operating FAP
    2.87e-7, where it is 57, 157, 1320 and 1.1e6 for ``n`` 6 to 3) a source at a limit made with this
    ``tau`` does not reach the FAP.  TABLE 1 says where, and TABLE 3 takes the exact threshold
    (:func:`_t_exact`).

    Parameters
    ----------
    alpha : float
        The false-alarm probability.
    n : int
        Number of noise apertures.

    Returns
    -------
    tau : float
        The threshold, 1e30 for ``alpha <= 0``.

    Notes
    -----
    It is ``near2v_tthresh`` of the IDL predecessor (verify 80-91), which bisects instead.
    """
    if alpha <= 0:
        return 1.0e30
    df = max(int(n) - 2, 1)
    return float(np.clip(_student_t.isf(float(alpha), df), -50.0, 50.0))


#: :func:`t_thresh`'s cap, the IDL predecessor's bisection range
TAU_CAP = 50.0


def _t_exact(alpha: float, n: int) -> float:
    """Return the S/N at which :func:`t_fap` is ``alpha``, without :func:`t_thresh`'s cap.

    Parameters
    ----------
    alpha : float
        The false-alarm probability.
    n : int
        Number of noise apertures.

    Returns
    -------
    tau : float
        The threshold, infinite for ``alpha <= 0``.
    """
    if alpha <= 0:
        return float("inf")
    return float(_student_t.isf(float(alpha), max(int(n) - 2, 1)))


#: The verify's peak search for every source of a run saved before 2026-11-25
#: (``RunConfig.verify_position = "peak"``), in pixels.
LEGACY_SEARCH_PX = 1.5


def legacy_position(rules: Optional[Dict[str, Any]]) -> bool:
    """Tell whether a run's rules keep the verify's measures of before 2026-11-25.

    Parameters
    ----------
    rules : dict or None
        The run's ring rules (:meth:`klip_tpe.runner.Runner._ring_rules`).

    Returns
    -------
    legacy : bool
        True for rules whose ``position`` is ``"peak"`` (``RunConfig.verify_position``) and for those of
        a verify of before 2026-11-01 (:data:`~klip_tpe.metrics.VERIFY_IDL`): every source at the matched
        filter's peak within 1.5 px, and a blind candidate's field-wide FAP by ``1 - (1 - p)^N_resel``.
    """
    return verify_idl(rules) or str((rules or {}).get("position", "given")).lower() == "peak"


def search_px(fwhm: float, found: bool = False, rules: Optional[Dict[str, Any]] = None,
              drift_px: Optional[float] = None) -> float:
    """Return how far from a source's position the verify takes the matched filter's peak, in pixels.

    A source whose position is given, a known companion or an injection, is read at its position, so
    its S/N is the statistic the small-sample t-test calibrates.  A candidate a blind search found is
    read at the peak within half an FWHM, the aperture of the S/N, of where the search found it.  Its
    local FAP then needs the correction for the search over the field (:func:`field_fap`), which
    covers the half FWHM too.

    Parameters
    ----------
    fwhm : float
        FWHM of the PSF, in pixels.
    found : bool, default False
        Whether a blind search found the source, rather than its position being given.
    rules : dict or None, default None
        The run's ring rules (:meth:`klip_tpe.runner.Runner._ring_rules`).  Rules whose ``position`` is
        ``"peak"``, or the rules of a verify of before 2026-11-01
        (:data:`~klip_tpe.metrics.VERIFY_IDL`), give every source :data:`LEGACY_SEARCH_PX`.
    drift_px : float or None, default None
        A radius the caller sets, which wins.

    Returns
    -------
    radius : float
        The search radius, in pixels, 0 to read the source at its position.

    Notes
    -----
    The IDL predecessor and runs saved before 2026-11-25 took every source's peak within 1.5 px.  On
    source-free Gaussian noise correlated at the PSF, the peak's FAP was reached 1.6 to 4.6 times as
    often as it said (FWHM 6.4 to 2.3 px, FAPs 0.1 to 1e-4), and about 10 times at 2.87e-7 by
    extrapolation, while the S/N at the given position follows its FAP (the audit of 2026-11-25).
    """
    if drift_px is not None:
        return float(drift_px)
    if legacy_position(rules):
        return LEGACY_SEARCH_PX
    return 0.5 * float(fwhm) if found else 0.0


def field_fap(fap_local: float, area_px: float, perimeter_px: float, fwhm_eff_px: float,
              euler: float = 0.0) -> float:
    """Return the false-alarm probability of the strongest peak a search over a field finds.

    The local FAP ``p`` of the peak, its small-sample t-test at the peak, becomes the Gaussian
    threshold ``u = Phi^-1(1 - p)``.  The expected Euler characteristic of the excursion set above
    ``u`` of a smooth Gaussian field over the search region (Worsley et al. 1996) is

    ``E = R0 Q(u) + R1 sqrt(4 ln 2) / (2 pi) exp(-u^2/2) + R2 4 ln 2 / (2 pi)^(3/2) u exp(-u^2/2)``,

    with ``R0`` the region's Euler characteristic (0 for an annulus), ``R1`` half its perimeter and
    ``R2`` its area, in units of the field's FWHM and its square.  The probability that the field's
    maximum passes ``u`` is ``1 - exp(-E)``.

    Parameters
    ----------
    fap_local : float
        The peak's FAP as if at a fixed pixel (:func:`t_fap`).
    area_px : float
        Area searched, in square pixels.
    perimeter_px : float
        Length of the region's boundary, in pixels.
    fwhm_eff_px : float
        FWHM of the smoothing that would give white noise the field's roughness, in pixels
        (:func:`field_smoothness`).  For a matched filter of noise correlated at the PSF it is about
        sqrt(2) times the PSF's FWHM.
    euler : float, default 0.0
        Euler characteristic of the region, 0 for an annulus, 1 for a disk.

    Returns
    -------
    fap : float
        The field-wide FAP, at least ``fap_local``, at most 1, and NaN for a non-finite input.

    Notes
    -----
    The same theory gives the look-elsewhere correction of particle physics (Gross & Vitells 2010,
    Eur. Phys. J. C 70, 525) and the family-wise error of statistical parametric maps in brain imaging.
    It replaces the candidate search's ``1 - (1 - p)^N_resel`` with ``N_resel = area / FWHM^2``, the
    IDL predecessor's.  On source-free fields of Gaussian noise correlated at the PSF (FWHM 4 px, a
    0.24 to 2.8" annulus at 0.05"/px), that FAP was reached 2.7, 3.0 and 3.4 times as often as it
    said at 0.1, 0.05 and 0.01 (18,000 fields), and this one 0.83, 0.85 and 0.67 times (12,000
    fields, the audit of 2026-11-25).  From 0.5 to 0.01 this one is reached 0.68 to 0.82 times as
    often at a FWHM of 2.3 px and 0.89 to 1.05 times at 6.4 px.  It errs on the safe side, as the
    theory does on a field sampled in pixels, and on a Gaussian field of known scatter at 4 px it is
    reached 0.82 to 0.93 times as often as it says.
    """
    p = float(fap_local)
    if not np.isfinite(p) or not (np.isfinite(area_px) and np.isfinite(fwhm_eff_px)) or fwhm_eff_px <= 0:
        return float("nan")
    if p <= 0.0:
        return 0.0
    if p >= 0.5:
        return 1.0
    u = float(_norm.isf(p))
    e = float(np.exp(-0.5 * u * u))
    r1 = 0.5 * max(float(perimeter_px), 0.0) / float(fwhm_eff_px)
    r2 = max(float(area_px), 0.0) / float(fwhm_eff_px) ** 2
    c = 4.0 * np.log(2.0)
    ec = float(euler) * p + r1 * np.sqrt(c) / (2.0 * np.pi) * e + r2 * c / (2.0 * np.pi) ** 1.5 * u * e
    return float(min(1.0, max(p, -np.expm1(-max(ec, 0.0)))))


def field_smoothness(snrmap: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return the FWHM of the smoothing that gives white noise an S/N map's roughness, in pixels.

    White noise smoothed by a Gaussian of FWHM ``s`` has a Gaussian autocorrelation whose value one
    pixel away is ``exp(-2 ln 2 / s^2)``.  The map is put to unit robust scatter, the scatter of the
    differences between neighboring pixels along each axis gives that value, ``1 - var/2``, and the two
    axes' mean gives ``s``.  Robust scatters keep a few bright sources from setting it.

    Parameters
    ----------
    snrmap : ndarray, shape (ny, nx)
        The S/N map.  NaN pixels are left out.
    mask : ndarray of bool or None, default None
        The pixels to use, True kept, such as the search annulus.

    Returns
    -------
    fwhm : float
        The smoothing's FWHM, in pixels, NaN when fewer than 50 pixel pairs are usable.

    Notes
    -----
    For the matched filter of white noise ``s`` is the kernel's FWHM, and for the matched filter of
    noise correlated at the PSF about sqrt(2) times the PSF's.  The estimator is that of Forman et
    al. (1995, Magn. Reson. Med. 33, 636), the classic method of AFNI's ``3dFWHMx``.  Brain imaging
    also estimates the smoothness from the residuals' gradients (Kiebel et al. 1999, NeuroImage 10,
    756).
    """
    z = np.asarray(snrmap, float)
    ok = np.isfinite(z) if mask is None else (np.isfinite(z) & np.asarray(mask, bool))
    if ok.sum() < 50:
        return float("nan")
    v = z[ok]
    med = float(np.median(v))
    sc = 1.4826 * float(np.median(np.abs(v - med)))
    if not sc > 0:
        return float("nan")
    zz = (z - med) / sc
    rhos = []
    for axis in (0, 1):
        a, b = (zz[:-1, :], zz[1:, :]) if axis == 0 else (zz[:, :-1], zz[:, 1:])
        m = (ok[:-1, :] & ok[1:, :]) if axis == 0 else (ok[:, :-1] & ok[:, 1:])
        if m.sum() < 50:
            continue
        d = (a - b)[m]
        vd = (1.4826 * float(np.median(np.abs(d - np.median(d))))) ** 2
        rhos.append(1.0 - 0.5 * vd)
    if not rhos:
        return float("nan")
    rho = float(np.clip(np.mean(rhos), 1e-6, 1.0 - 1e-9))
    return float(np.sqrt(-2.0 * np.log(2.0) / np.log(rho)))


def field_geometry(snrmap: np.ndarray, iwa_px: float, owa_px: float, fwhm: float) -> Dict[str, float]:
    """Return what :func:`field_fap` needs of a blind search's field: its area, boundary and smoothness.

    Parameters
    ----------
    snrmap : ndarray, shape (ny, nx)
        The S/N map the search finds its peaks on, with the star at ``((nx-1)/2, (ny-1)/2)``.
    iwa_px : float
        Inner radius of the search, in pixels.
    owa_px : float
        Outer radius of the search, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels, for the fallback smoothness.

    Returns
    -------
    field : dict
        ``area_px`` (the map's finite pixels between the radii), ``perimeter_px`` (the two circles'
        lengths), ``fwhm_eff_px`` (:func:`field_smoothness` of the map there, or sqrt(2) times ``fwhm``
        when it cannot be measured), ``fwhm_eff_from`` ("map" or "psf"), ``euler`` (0, an annulus) and
        ``n_resel`` (the area over ``fwhm^2``, the IDL predecessor's count, for comparison).
    """
    z = np.asarray(snrmap, float)
    ny, nx = z.shape
    cx, cy = star_center(z.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    ann = (rr >= float(iwa_px)) & (rr <= float(owa_px))
    fe = field_smoothness(z, ann)
    src = "map"
    if not (np.isfinite(fe) and fe > 0):
        fe, src = float(np.sqrt(2.0) * fwhm), "psf"
    area = float(np.count_nonzero(ann & np.isfinite(z)))
    return {"area_px": area, "perimeter_px": float(2.0 * np.pi * (float(iwa_px) + float(owa_px))),
            "fwhm_eff_px": float(fe), "fwhm_eff_from": src, "euler": 0.0,
            "n_resel": max(np.pi * (float(owa_px) ** 2 - float(iwa_px) ** 2) / float(fwhm) ** 2, 1.0)}


def contrast_limit(alpha: float, beta: float, ci, si, napi) -> float:
    """Return the detection-limit contrast at a per-resolution-element FAP and completeness, scaled from injections.

    Each injection gives ``lim_k = c_k (tau_alpha + tau_{1-beta}) / SNR_k`` with
    ``n_ap = max(round(nap_k), 3)``, and the limit is their mean over the injections with a positive
    S/N and contrast.

    Parameters
    ----------
    alpha : float
        False-alarm probability per resolution element.
    beta : float
        Completeness, the fraction of sources detected.
    ci : array_like
        Contrast of each injection.
    si : array_like
        S/N of each injection.
    napi : array_like
        Number of noise apertures of each injection.

    Returns
    -------
    limit : float
        The contrast limit, NaN without a usable injection.

    Notes
    -----
    It is ``near2v_climit`` of the IDL predecessor (verify 99-107).
    """
    ci, si, napi = (np.atleast_1d(np.asarray(a, float)) for a in (ci, si, napi))
    lim, nl = 0.0, 0
    for k in range(ci.size):
        if si[k] > 0 and np.isfinite(ci[k]) and ci[k] > 0 and np.isfinite(si[k]):
            na = max(int(round(napi[k])), 3)
            lim += ci[k] * (t_thresh(alpha, na) + t_thresh(1.0 - beta, na)) / si[k]
            nl += 1
    return lim / nl if nl else float("nan")


def contrast_limit_image(alpha_img: float, beta: float, ci, si, napi) -> float:
    """Return :func:`contrast_limit` for a per-image FAP, converted per resolution element.

    The per-element FAP is ``min(alpha_img / n_ap, 0.49)``.

    Parameters
    ----------
    alpha_img : float
        False-alarm probability per image.
    beta : float
        Completeness.
    ci : array_like
        Contrast of each injection.
    si : array_like
        S/N of each injection.
    napi : array_like
        Number of noise apertures of each injection.

    Returns
    -------
    limit : float
        The contrast limit, NaN without a usable injection.

    Notes
    -----
    It is ``near2v_climg`` of the IDL predecessor (verify 352-360).
    """
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
    """Apply the IDL predecessor's matched filter, the image correlated with a unit-sum FWHM Gaussian.

    NaN pixels become 0 and the edges are truncated (``mode="nearest"``).  The Gaussian is round, so a
    convolution gives the same map, and a ``kernel`` given is correlated, as a matched filter takes its
    template.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    fwhm : float
        FWHM of the Gaussian, in pixels.
    kernel : ndarray or None, default None
        Kernel to use instead of the Gaussian.

    Returns
    -------
    mfmap : ndarray, shape (ny, nx)
        The filtered image.

    Notes
    -----
    It is ``near2v_mfconv`` (verify 117-120) with ``near2v_mfkern`` (verify 53-59).
    """
    img = np.asarray(img, float)
    k = gaussian_kernel(fwhm) if kernel is None else kernel
    return ndimage.correlate(np.where(np.isfinite(img), img, 0.0), k, mode="nearest")


def _snr_fap_idl(img: np.ndarray, rho: float, theta: float, pxscale: float, fwhm: float,
                 exr: Optional[float] = None, srad: float = 1.5, *,
                 exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
                 angle_convention: str = "pa", excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                 peak_xy: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Measure :func:`snr_fap` as it measured before 2026-11-01, for runs that keep that verify.

    It is the matched-filter Mawet S/N and small-sample FAP of the IDL predecessor's ``near2v_snrfap``
    (verify 122-154), kept for ``RunConfig.verify_snr = "idl"`` (:data:`~klip_tpe.metrics.VERIFY_IDL`).
    The peak is the maximum of the Gaussian-filtered map within ``srad`` of the position, and the ring
    of ``nap = floor(2 pi r / fwhm)`` positions skips ``exr`` around the peak and around each
    ``exclude_xy``.  The S/N is ``(A_pk - mean) / (sd sqrt(1 + 1/nv))`` with the ``nv`` samples used.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    exr : float or None, default None
        Radius of the keep-outs, in pixels.  None takes 1.5 FWHM.
    srad : float, default 1.5
        Radius of the peak search, in pixels.
    exclude_xy : sequence of tuple of 2 floats, default ()
        More ``(x, y)`` pixels whose keep-out the ring skips, such as the other sources and the known
        companions.
    mfmap : ndarray or None, default None
        Matched-filter map made already, of the image's shape.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the ring, ADI sidelobes under the sidelobe rule, as
        :func:`~klip_tpe.metrics.mawet_peak_snr` takes them.
    peak_xy : tuple of 2 floats or None, default None
        The ``(x, y)`` pixel where the peak is read instead of searched for.

    Returns
    -------
    result : dict
        ``snr``, ``fap``, ``nap`` (the samples used, or the nominal count on an early exit), ``peak``,
        ``x``, ``y``, ``r`` and ``nclean``, with ``n_left``, ``bg`` and ``sigma`` when there are samples.
        The S/N is NaN for ``r < fwhm``, ``nap < 5``, fewer than 3 samples or no scatter.
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
            exr: Optional[float] = None, srad: Optional[float] = None, *,
            exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
            angle_convention: str = "pa", excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
            peak_xy: Optional[Tuple[float, float]] = None, rules: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Measure the verify's S/N and small-sample FAP at one source, the search's measure under the run's rules.

    The S/N is the search's matched-filter Mawet S/N (:func:`~klip_tpe.metrics.mawet_peak_snr`) under
    the run's ring rules (:func:`~klip_tpe.metrics.ring_rules`), or that function's defaults.  The
    source is read at its position, or with ``srad`` at the matched-filter peak within ``srad`` of it
    (:func:`search_px`), and the ring through it skips ``exr`` around the source as given and around
    each ``exclude_xy``, the pixel mask and ``excl_arcs``.
    ``fap = t_fap(snr, n)`` with ``n`` the ring samples used, or the ring's count when the radial band
    gave the noise.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    exr : float or None, default None
        Radius of the keep-outs, in pixels.  None takes 1.5 FWHM.
    srad : float or None, default None
        Radius of the peak search, in pixels.  0 reads the source at its position, where its FAP
        holds, since a search raises the S/N and the FAP then says too little.  None reads where
        ``rules`` say (:func:`search_px`), at the position or, for a run made before 2026-11-25, at
        the peak within 1.5 px.
    exclude_xy : sequence of tuple of 2 floats, default ()
        More ``(x, y)`` pixels whose keep-out the ring skips, such as the other sources and the known
        companions.
    mfmap : ndarray or None, default None
        Matched-filter map made already, of the image's shape.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the ring, ADI sidelobes under the sidelobe rule, as
        :func:`~klip_tpe.metrics.mawet_peak_snr` takes them.
    peak_xy : tuple of 2 floats or None, default None
        The ``(x, y)`` pixel where the peak is read instead of searched for.
    rules : dict or None, default None
        The run's ring rules.  Rules of :data:`~klip_tpe.metrics.VERIFY_IDL` hand the measure to
        :func:`_snr_fap_idl`.

    Returns
    -------
    result : dict
        ``snr``, ``fap``, ``nap`` (the FAP's ``n``, or the ring's count on an early exit), ``peak``,
        ``x``, ``y``, ``r``, ``nclean``, ``n_left`` and ``noise`` ("ring" or "band").  ``bg`` and ``sigma``
        come with samples, and ``why`` with a NaN.

    Notes
    -----
    Since 2026-11-01 (Kevin Wagner) the verify and the candidate search measure as the search does.
    Before, this function ported the IDL predecessor's ``near2v_snrfap``, with a Gaussian kernel, no
    despike, ``exr`` kept around the located peak and no radial band.
    """
    srad = search_px(fwhm, False, rules) if srad is None else float(srad)
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
                 return_details=True, exr=None, srad=None, exclude_xy=(), mfmap=None, peak_xy=None, rules=None,
                 **_):
    """Measure one source with :func:`snr_fap` in the form :func:`~klip_tpe.metrics.sidelobe_snr` asks of a measure.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    rho : sequence of float
        Separation of the one source, in arcsec.
    theta : sequence of float
        Angle of the one source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    known : sequence of tuple of 2 floats, default ()
        Known companions, unused here, since their keep-outs come in ``exclude_xy``.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the ring.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.
    return_details : bool, default True
        Unused, the details always come back.
    exr : float or None, default None
        Radius of the keep-outs, in pixels.
    srad : float or None, default None
        Radius of the peak search, in pixels, None where ``rules`` say (:func:`search_px`).
    exclude_xy : sequence of tuple of 2 floats, default ()
        More pixels whose keep-out the ring skips.
    mfmap : ndarray or None, default None
        Matched-filter map made already.
    peak_xy : sequence of tuple of 2 floats or None, default None
        One ``(x, y)`` pixel where the peak is read.
    rules : dict or None, default None
        The run's ring rules.
    _ : dict
        Other keywords of the measure, ignored.

    Returns
    -------
    snr : ndarray, shape (1,)
        The S/N.
    details : list of dict
        The source's details, with ``nap`` the ring's apertures and ``nclean`` the samples used.
    """
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
             fwhm: float, sidelobes, *, exr: Optional[float] = None, srad: Optional[float] = None,
             exclude_xy: Sequence[Tuple[float, float]] = (), known: Sequence[Tuple[float, float]] = (),
             mfmap: Optional[np.ndarray] = None, angle_convention: str = "pa",
             rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Measure :func:`snr_fap` of one source under the run's sidelobe rule.

    The source's ring leaves out its own ADI arc, the other sources' and the known companions'
    (:func:`~klip_tpe.metrics.sidelobe_snr`).  Where fewer than ``min_apertures`` samples remain, its
    own arc narrows to let in the farthest, then the other sources' arcs are let in, then the known
    companions', which leaves the ring without the rule.  The fallback is always the farthest samples,
    since the verify has no clean ring through every source.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    i : int
        Index of the source measured.
    rhos : sequence of float
        Separation of every source, in arcsec.
    thetas : sequence of float
        Angle of every source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    sidelobes : Sidelobes
        The sidelobe rule, taken in its "farthest" mode.
    exr : float or None, default None
        Radius of the keep-outs, in pixels.  None takes 1.5 FWHM.
    srad : float or None, default None
        Radius of the peak search, in pixels, None where ``rules`` say (:func:`search_px`).
    exclude_xy : sequence of tuple of 2 floats, default ()
        More pixels whose keep-out the ring skips.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    mfmap : ndarray or None, default None
        Matched-filter map made already.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    rules : dict or None, default None
        The run's ring rules, taken with the rule's ``min_apertures`` as the shortest ring, so the
        rule's fallback takes over from the radial band.

    Returns
    -------
    result : dict
        :func:`snr_fap`'s dict, with ``nap`` the samples used (the ring's count for a NaN), and the
        rule's ``arc_deg``, ``n_arcs``, ``fallback`` (or ``fallback_tried``) and the ``arcs`` it left out.
    """
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
    """Return the known companions at least ``exr`` from a source.

    Parameters
    ----------
    known : sequence of tuple of 2 floats
        Known companions, as ``(rho_arcsec, theta_deg)``.
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    exr : float
        Radius within which a companion is the source itself, in pixels.
    angle_convention : {"pa", "math"}
        How the angles are read.

    Returns
    -------
    others : list of tuple of 2 floats
        The other known companions.
    """
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
    """Summarize how a ring came out under the sidelobe rule, for a verify row.

    Parameters
    ----------
    d : dict
        The source's details, from :func:`ring_snr`.

    Returns
    -------
    noise : dict
        ``arc_deg`` (the source's own arc as measured), ``n_arcs`` (the samples outside every arc),
        ``n_used`` (the samples used) and ``fallback`` (the fallback taken, or tried and how many it
        found, None when none was needed).
    """
    return {"arc_deg": float(d.get("arc_deg", float("nan"))), "n_arcs": int(d.get("n_arcs", 0)),
            "n_used": int(d.get("nclean", 0)), "fallback": d.get("fallback") or
            (f"{d['fallback_tried']} tried, {int(d.get('n_fallback', 0))} found" if d.get("fallback_tried") else None)}


class _ScaledImage:
    """S/N of injected sources with their response scaled by ``a`` and the clean image's noise as it is.

    The image is ``clean + a (inj - clean)``, and its map is the line through the run's two maps,
    ``mf(clean) + a (mf(inj) - mf(clean))``, each made as the run's rules make it
    (:func:`~klip_tpe.metrics.matched_filter`, at each source's separation for a measured kernel).  At
    ``a = 0`` and ``a = 1`` it is the search's own map of each image.  The convolution is linear, so
    only the despike, where it replaces a pixel, bends the map off that line in between.  Pixels not
    finite in both images are left out of both.

    Parameters
    ----------
    clean_img : ndarray, shape (ny, nx)
        The clean reduction.
    inj_img : ndarray, shape (ny, nx)
        The same reduction with the sources injected.
    fwhm : float
        FWHM of the PSF, in pixels.
    rules : dict or None, default None
        The run's ring rules.

    Attributes
    ----------
    img0 : ndarray, shape (ny, nx)
        The clean image, NaN where either image is not finite.
    img1 : ndarray, shape (ny, nx)
        The injected image, NaN where either image is not finite.

    Raises
    ------
    ValueError
        If the two images differ in shape.
    """

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
        """Return the clean image's matched-filter map and the injected image's difference from it.

        Each kernel's maps are made once.  Under the measures of before 2026-11-01
        (:data:`~klip_tpe.metrics.VERIFY_IDL`) they are the Gaussian maps of the clean image and of
        ``inj - clean``, as made then.

        Parameters
        ----------
        rho : float or None, default None
            Separation where the kernel is taken, in arcsec.  None takes 0.

        Returns
        -------
        mf0 : ndarray, shape (ny, nx)
            ``mf(clean)``.
        mf1 : ndarray, shape (ny, nx)
            ``mf(inj) - mf(clean)``.
        """
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
        """The clean image's matched-filter map, with the kernel at separation 0.
        """
        return self.maps()[0]

    @property
    def mf1(self) -> np.ndarray:
        """The injected image's matched-filter map minus the clean one's, with the kernel at separation 0.
        """
        return self.maps()[1]

    def snr(self, a: float, rho: float, theta: float, pxscale: float, fwhm: float, exr: float, srad: float,
            exclude_xy: Sequence[Tuple[float, float]] = (), angle_convention: str = "pa",
            excl_arcs: Sequence[Tuple[float, float, float, float]] = ()) -> Tuple[float, int]:
        """Return a source's S/N at the scale ``a``, measured once per scale and source.

        Parameters
        ----------
        a : float
            Scale of the sources' response.
        rho : float
            Separation of the source, in arcsec.
        theta : float
            Angle of the source, in degrees.
        pxscale : float
            Plate scale, in arcsec per pixel.
        fwhm : float
            FWHM of the PSF, in pixels.
        exr : float
            Radius of the keep-outs, in pixels.
        srad : float
            Radius of the peak search, in pixels.
        exclude_xy : sequence of tuple of 2 floats, default ()
            More pixels whose keep-out the ring skips.
        angle_convention : {"pa", "math"}, default "pa"
            How ``theta`` is read.
        excl_arcs : sequence of tuple of 4 floats, default ()
            Arcs left out of the ring.

        Returns
        -------
        snr : float
            The S/N.
        nap : int
            The FAP's ``n`` (:func:`snr_fap`).
        """
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
    """Find the smallest value in a range at which a condition holds.

    The range is stepped through ``n_grid`` values spaced evenly in log, and the first step that
    crosses is bisected.

    Parameters
    ----------
    cond : callable
        Function ``x -> bool``.
    lo : float
        Lower end of the range, positive.
    hi : float
        Upper end of the range.
    n_grid : int
        Number of grid values.

    Returns
    -------
    x : float
        The smallest such value, ``lo`` when the condition holds there, infinite when it holds nowhere.
    """
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
    """Tell whether an S/N on ``n`` samples is detected at a false-alarm probability.

    Parameters
    ----------
    s : float
        The S/N.
    n : int
        Number of noise samples, at least 3 for a detection.
    fap : float
        The false-alarm probability.

    Returns
    -------
    detected : bool
        True when :func:`t_fap` is below ``fap``.
    """
    return bool(np.isfinite(s) and n >= 3 and t_fap(s, n) < fap)


def _reaches(s: float, n: int, fap: float, beta: float) -> bool:
    """Tell whether an S/N reaches ``tau_fap + tau_(1-beta)``, both thresholds exact.

    Parameters
    ----------
    s : float
        The S/N.
    n : int
        Number of noise samples, at least 3.
    fap : float
        The false-alarm probability.
    beta : float
        The completeness.

    Returns
    -------
    reaches : bool
        True when it reaches the sum (:func:`_t_exact`).
    """
    return bool(np.isfinite(s) and n >= 3 and s >= _t_exact(fap, n) + _t_exact(1.0 - beta, n))


def solved_contrasts(clean_img: np.ndarray, inj_img: np.ndarray, rho: float, theta: float, contrast: float,
                     pxscale: float, fwhm: float, exr: Optional[float] = None, srad: Optional[float] = None, *,
                     exclude_xy: Sequence[Tuple[float, float]] = (), fap: float = DEFAULT_FAP_LIST[0],
                     taps: Sequence[float] = DEFAULT_TAP_LIST, angle_convention: str = "pa",
                     a_range: Tuple[float, float] = (1e-3, 1e3), n_grid: int = 121,
                     _scaled: Optional["_ScaledImage"] = None,
                     excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                     rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Solve the contrasts at which an injected source reaches the verify's thresholds, on its own image.

    The source's S/N at contrast ``a * contrast`` is :func:`snr_fap` on ``clean + a (inj - clean)``
    (:class:`_ScaledImage`), with pixels not finite in both images left out of both.  The image's noise
    stays as it is, the noise at the source's position included, and only the source's response scales,
    its residual in its own ring included.  Both thresholds are exact (:func:`_t_exact`), where
    :func:`t_thresh` caps them at 50.  This is the single-source limit of :func:`contrast_limit` solved,
    where that function scales the whole S/N at the injected contrast, noise included.

    Parameters
    ----------
    clean_img : ndarray, shape (ny, nx)
        The clean reduction.
    inj_img : ndarray, shape (ny, nx)
        The same reduction with the source injected.
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    contrast : float
        Contrast the source was injected at.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    exr : float or None, default None
        Radius of the keep-outs, in pixels.  None takes 1.5 FWHM.
    srad : float or None, default None
        Radius of the peak search, in pixels, None where ``rules`` say (:func:`search_px`).
    exclude_xy : sequence of tuple of 2 floats, default ()
        More pixels whose keep-out the ring skips.
    fap : float, default DEFAULT_FAP_LIST[0]
        The false-alarm probability of the detection.
    taps : sequence of float, default DEFAULT_TAP_LIST
        The completeness levels beta.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.
    a_range : tuple of 2 floats, default (1e-3, 1e3)
        Range of the scale ``a`` searched.
    n_grid : int, default 121
        Number of values of ``a``, spaced evenly in log, before the bisection.
    _scaled : _ScaledImage or None, default None
        The scaled image made already, shared between the sources of one image.
    excl_arcs : sequence of tuple of 4 floats, default ()
        The sidelobe rule's arcs, as the source's own S/N left them out (:func:`ring_snr`).
    rules : dict or None, default None
        The run's ring rules.

    Returns
    -------
    result : dict
        ``c_det``, the faintest contrast detected at ``fap``, ``c_tap``, for each beta the faintest at
        which the S/N reaches ``tau_fap + tau_(1-beta)``, ``snr1`` and ``nap1``, the injected image's own
        S/N and ``n``.  A contrast is infinite where no ``a`` gets there and ``a_range[0] * contrast``
        where even that does, and all are NaN, with ``nap1`` 0, for a contrast not finite and positive.

    Notes
    -----
    The S/N need not grow with ``a``.  The source's residual in its ring grows with it, and the peak
    search can move to a pixel at another radius, whose ring differs.  So the faintest contrast that
    gets there is not always one above which the source stays there.  KLIP's response to a bright
    source is not linear either, so the contrasts hold best for an injection near them (NEAR2 note of
    2026-10-23, revised).
    """
    out: Dict[str, Any] = {"c_det": float("nan"), "c_tap": [float("nan")] * len(taps), "snr1": float("nan"),
                           "nap1": 0}
    c0 = float(contrast)
    if not (np.isfinite(c0) and c0 > 0):
        return out
    if exr is None:
        exr = 1.5 * fwhm
    srad = search_px(fwhm, False, rules) if srad is None else float(srad)
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
    """Bin the solved contrasts into TABLE 3, in the bins of TABLE 1 and in bins of the same width beyond them.

    With beta the completeness nearest 0.9 (``tap_list[i90]``), each bin gives the faintest contrast at
    which at least ``ceil(beta n)`` of its ``n`` sources are detected, all of them at that contrast, and
    counts the sources detected at TABLE 1's ``c@TAP`` for beta.

    Parameters
    ----------
    lim : dict
        TABLE 1 and TABLE 2, from :func:`limit_tables`.
    items : sequence of tuple
        ``(row, detected_at)`` per injection with solved contrasts, ``detected_at(c)`` telling whether
        the source is detected at contrast ``c`` (its S/N on its own image, scaled to ``c``).
    n_grid_decade : int, default 20
        Number of contrasts per decade searched before the bisection.
    a_range : tuple of 2 floats, default (1e-3, 1e3)
        Range searched, as factors on the bin's faintest and brightest injected contrasts.

    Returns
    -------
    bins : list of dict or None
        One dict per bin that holds sources, None without ``items``.  Each holds ``sep`` (the mean
        separation, in arcsec), ``n``, ``c_tap`` (for each beta the mean of the rows' finite
        ``c_tap_solved``, infinite where none is), ``c_det_beta`` (the faintest contrast above),
        ``n_det_t1`` (None where TABLE 1 has no such bin), ``ib`` (the bin's index) and ``rows``,
        ``(rho, theta, contrast, c_det, c@TAP for beta)`` per source.
    """
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
    """Combine the selected nights into their weighted mean, NaN-aware.

    Each pixel is ``sum w_j img_j fin_j / sum w_j fin_j``, with ``fin_j`` whether night ``j`` is finite
    there, and NaN where the total weight is 0.  A night drawn more than once, as in a bootstrap, counts
    each time.

    Parameters
    ----------
    stack : ndarray, shape (n, ny, nx)
        The images of the ``n`` nights.
    weights : ndarray, shape (n,)
        Weight of each night.
    idx : sequence of int
        Indices of the nights combined.

    Returns
    -------
    image : ndarray, shape (ny, nx)
        The weighted mean.

    Notes
    -----
    It is ``near2v_combine`` of the IDL predecessor (verify 160-170).
    """
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
    """Make the night-STIM map, the weighted mean over the nights divided by their weighted scatter.

    The scatter is the biased weighted standard deviation (no ``N - 1``), floored at
    ``0.3 median(std > 0)`` and at 1e-12.

    Parameters
    ----------
    stack : ndarray, shape (n, ny, nx)
        The images of the ``n`` nights.
    weights : ndarray or None, default None
        Weight of each night.  None weighs them equally.

    Returns
    -------
    stim : ndarray, shape (ny, nx)
        The STIM map, NaN where the total weight is 0 and everywhere for fewer than 2 nights.

    Notes
    -----
    It is ``near2v_stim`` of the IDL predecessor (verify 177-196).
    """
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
    """Return the night-STIM value at a source and its empirical FAP against the annulus at its separation.

    The value is read at the rounded source pixel and ranked against the finite pixels of
    ``|r - rho_px| <= fwhm`` farther than ``keepout`` from the source and from each ``exclude_xy``,
    with ``excl_arcs`` and ``pixel_mask`` left out as every noise ring leaves them out.  A
    :class:`~klip_tpe.metrics.RingRecorder` that is on records the annulus.

    Parameters
    ----------
    stim : ndarray, shape (ny, nx)
        The night-STIM map.
    x : float
        Column of the source, in pixels.
    y : float
        Row of the source, in pixels.
    rho_px : float
        Separation of the source, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels, also the annulus' half width.
    keepout : float or None, default None
        Radius of the keep-outs, in pixels.  None takes 1.5 FWHM.
    exclude_xy : sequence of tuple of 2 floats, default ()
        The ``(x, y)`` pixels of the other sources, which are not noise.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the annulus, the ADI sidelobes the source's ring left out (:func:`ring_snr`).
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel, without which ``excl_arcs`` is ignored.
    angle_convention : {"pa", "math"}, default "pa"
        How the arcs' angles are read.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the annulus, True to exclude, such as the run's disk mask.  A mask not of the
        map's shape is ignored.

    Returns
    -------
    value : float
        The STIM value at the source, NaN off the map.
    fap : float
        The fraction of the annulus at or above the value, NaN for an empty annulus or a NaN value.

    Notes
    -----
    It is the IDL predecessor's STIM FAP (verify 594-602), with the keep-out around the source for the
    injections too (the module's Notes).
    """
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
                     pxscale: float, fwhm: float, srad: Optional[float] = None, n_boot: int = 500,
                     rng: Optional[np.random.Generator] = None, angle_convention: str = "pa",
                     rules: Optional[Dict[str, Any]] = None) -> float:
    """Return the bootstrap-over-nights significance of a source, the mean over the scatter of its peak.

    The ``n`` nights are drawn with replacement ``n_boot`` times.  Each draw is combined
    (:func:`combine_nights`, weighted, not flattened), and its matched-filter value at the source, or
    its peak within ``srad``, is read from :func:`snr_fap`, whose ring never moves the peak.  The
    matched filter is the run's (:func:`~klip_tpe.metrics.matched_filter`), or the Gaussian of
    :func:`mf_convolve` under :data:`~klip_tpe.metrics.VERIFY_IDL`.

    Parameters
    ----------
    stack : ndarray, shape (n, ny, nx)
        The images of the ``n`` nights.
    weights : ndarray, shape (n,)
        Weight of each night.
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    srad : float or None, default None
        Radius of the peak search, in pixels, None where ``rules`` say (:func:`search_px`).
    n_boot : int, default 500
        Number of draws.
    rng : numpy.random.Generator or None, default None
        Generator of the draws.  None seeds one with 42, the IDL predecessor's seed.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.
    rules : dict or None, default None
        The run's ring rules.

    Returns
    -------
    significance : float
        Mean over standard deviation (``ddof=1``) of the finite peaks, NaN for fewer than 3 or no scatter.

    Notes
    -----
    It is ``near2v_boot`` of the IDL predecessor (verify 203-214), which also calls its S/N function
    with ``exr = srad`` only to read back the peak.
    """
    stack = np.asarray(stack, float)
    nn = stack.shape[0]
    srad = search_px(fwhm, False, rules) if srad is None else float(srad)
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.full(int(n_boot), np.nan)
    for b in range(int(n_boot)):
        idx = rng.integers(0, nn, nn)
        cb = combine_nights(stack, weights, idx)
        A = mf_convolve(cb, fwhm) if verify_idl(rules) else matched_filter(cb, fwhm, rules, rho)
        vals[b] = snr_fap(cb, rho, theta, pxscale, fwhm, exr=max(srad, LEGACY_SEARCH_PX), srad=srad, mfmap=A,
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
    """Return the subsets of the nights, by the order of the slices.

    Parameters
    ----------
    nn : int
        Number of nights.

    Returns
    -------
    subsets : dict of str to ndarray
        The night indices of ``combined`` (all), ``first-half`` (``0`` to ``nn//2 - 1``),
        ``second-half`` (``nn//2`` on), ``even`` and ``odd``.

    Notes
    -----
    They are the IDL predecessor's subsets (verify 542-544).
    """
    a = np.arange(nn)
    return {"combined": a, "first-half": a[: nn // 2], "second-half": a[nn // 2:],
            "even": a[a % 2 == 0], "odd": a[a % 2 == 1]}


def verify_maps(stack: np.ndarray, weights: Optional[np.ndarray] = None, fwhm: Optional[float] = None,
                flatten: bool = False, *, combined: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Make the subset images, the night-STIM map and, with ``fwhm``, a Gaussian matched-filter map of each subset.

    Each subset of :func:`night_subsets` is the weighted mean of its nights (:func:`combine_nights`),
    and a subset without nights (one night) is all NaN.  The ``mf`` maps are a look at the subsets
    only, since every S/N makes the run's own map (:func:`snr_fap`).

    Parameters
    ----------
    stack : ndarray, shape (n, ny, nx)
        The images of the ``n`` nights.
    weights : ndarray or None, default None
        Weight of each night.  None weighs them equally.
    fwhm : float or None, default None
        FWHM of the PSF, in pixels.  None makes no ``mf`` maps.
    flatten : bool, default False
        Whether to subtract each subset's azimuthal mean profile (:func:`~klip_tpe.metrics.radprof`), as
        the IDL predecessor does.  Off by default, so the subsets match what the run scored.
    combined : ndarray or None, default None
        Image of the frame size that replaces the combine of all nights, the run's own image that Path A
        measures.

    Returns
    -------
    maps : dict
        ``subsets`` (name to image), ``stim``, ``weights``, ``subset_index`` (name to night indices) and,
        with ``fwhm``, ``mf`` (name to map).

    Raises
    ------
    ValueError
        If ``combined`` is not of the stack's frame size.

    Notes
    -----
    It follows the IDL predecessor's verify 545-555.
    """
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
    """Return the pixel of each source given as ``(rho_arcsec, theta_deg)``, such as the known companions.

    Parameters
    ----------
    known : sequence of tuple of 2 floats
        The sources, as ``(rho_arcsec, theta_deg)``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    angle_convention : {"pa", "math"}
        How the angles are read.

    Returns
    -------
    xy : list of tuple of 2 floats
        The ``(x, y)`` pixel of each source.
    """
    if not len(known):
        return []
    kx, ky = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention)
    return list(zip(kx, ky))


def verify_candidates(stack: np.ndarray, angles_or_night_ids: Optional[Sequence[Any]],
                      candidates: Sequence[Tuple[float, float]], fwhm: float, pxscale: float, *,
                      known: Sequence[Tuple[float, float]] = (), injected=None,
                      fap_op: float = 2.87e-7, drift_px: Optional[float] = None, n_boot: int = 500,
                      rng: Optional[np.random.Generator] = None, weights: Optional[Sequence[float]] = None,
                      excl_fwhm: float = 1.5, fap_list: Sequence[float] = DEFAULT_FAP_LIST,
                      tap_list: Sequence[float] = DEFAULT_TAP_LIST, rbin_as: Optional[float] = None,
                      angle_convention: str = "pa", return_maps: bool = False,
                      combined: Optional[np.ndarray] = None, combined_inj: Optional[np.ndarray] = None,
                      path_a: Optional[str] = None, sidelobes=None,
                      rules: Optional[Dict[str, Any]] = None, found: bool = False,
                      field: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Verify candidates on a per-night image stack, by Path A and Path B.

    Path A is each candidate's S/N and FAP on the combined image (:func:`snr_fap`, or :func:`ring_snr`
    under the sidelobe rule).  A candidate whose position is given, such as a known companion, is read
    at it, and one a blind search found at the matched filter's peak within FWHM/2 (:func:`search_px`),
    with its FAP over the field beside the local one (:func:`field_fap`).  Path B is its S/N in the night subsets, its night-STIM value with the
    annulus FAP (:func:`stim_fap`) and the bootstrap over nights (:func:`bootstrap_nights`).  With
    ``injected``, the injections are measured too (:func:`verify_injections`), for the calibration
    block and the detection-limit tables.

    Parameters
    ----------
    stack : ndarray, shape (n, ny, nx) or (ny, nx)
        The per-night (or per-partition) images of the ``n`` nights, not flattened.  One image is one
        night.
    angles_or_night_ids : sequence or None
        One label per slice, such as a night, epoch or partition id, for the report.  None labels them
        ``0`` to ``n - 1``.
    candidates : sequence of tuple of 2 floats
        The candidates, as ``(rho_arcsec, theta_deg)``.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    known : sequence of tuple of 2 floats, default ()
        Known companions, left out of the noise rings and the STIM annulus.
    injected : tuple or None, default None
        ``(inj_stack, sources)``, the injected twin of ``stack`` and its sources as
        ``(rho_arcsec, theta_deg, contrast)``.
    fap_op : float, default 2.87e-7
        Operating FAP of Path A's decision, per resolution element.
    drift_px : float or None, default None
        Radius of the peak search, in pixels.  None takes :func:`search_px`: 0 for given positions,
        FWHM/2 for ``found`` ones, 1.5 px for a run saved before 2026-11-25.
    n_boot : int, default 500
        Number of bootstrap draws.  0 skips the bootstrap, as the IDL predecessor's ``/lite`` does.
    rng : numpy.random.Generator or None, default None
        Generator of the bootstrap.  None seeds one with 42, the IDL predecessor's seed.
    weights : sequence of float or None, default None
        Weight of each night, ``sqrt(texp)`` in the IDL predecessor.  None weighs them equally.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    fap_list : sequence of float, default DEFAULT_FAP_LIST
        FAPs per resolution element of the tables, the first the operating FAP of TABLE 1.
    tap_list : sequence of float, default DEFAULT_TAP_LIST
        Completeness levels of the tables.
    rbin_as : float or None, default None
        Width of the separation bins, in arcsec.  None takes one FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    return_maps : bool, default False
        Whether to return the maps too.
    combined : ndarray or None, default None
        The image Path A measures, the image the run made from these nights, such as an annulus'
        combined image or the final stitch.  None takes the weighted mean of the nights.
    combined_inj : ndarray or None, default None
        The injected twin of ``combined``, given with it when ``injected`` is.
    path_a : str or None, default None
        Name of Path A's image in the report.  None names it from ``combined`` and ``weights``.
    sidelobes : Sidelobes or None, default None
        The run's sidelobe rule (:class:`~klip_tpe.metrics.Sidelobes`).  None keeps the rings of the
        verify before 2026-11-01.
    rules : dict or None, default None
        The run's ring rules (:func:`~klip_tpe.metrics.ring_rules`), whose matched filter and pixel mask
        every S/N, the STIM annulus and the bootstrap take.  None takes the search metric's defaults.
    found : bool, default False
        Whether a blind search found the candidates, rather than their positions being given.
    field : dict or None, default None
        The blind search's field (:func:`field_geometry`), which gives ``found`` candidates their FAP
        over the field.

    Returns
    -------
    rows : list of dict
        One row per candidate (``kind="candidate"``), then with ``injected`` one per source
        (``kind="injection"``, :func:`verify_injections`).  A candidate's row holds ``index``, ``rho``,
        ``theta``, ``x``, ``y``, ``snr``, ``fap``, ``nap``, ``peak``, ``detect``, ``subsets`` (name to
        S/N), ``persistence``, ``stim``, ``stim_fap``, ``boot_sig``, ``meta``, ``search_px`` (the peak
        search used), ``position`` ("given" or "found"), under the sidelobe rule ``noise`` (:func:`_noise`),
        for a given candidate read at its position ``snr_peak`` (the S/N of the matched filter's peak
        within FWHM/2, for comparison only), and for a found one with ``field`` ``fap_field``.
    maps : dict [return_maps=True]
        The maps of :func:`verify_maps`, with ``limits`` (:func:`limit_tables`, None without
        ``injected``), ``injected`` (the injections' maps) and ``meta``.

    Raises
    ------
    ValueError
        If ``injected`` comes with only one of ``combined`` and ``combined_inj``.

    Notes
    -----
    The subsets split by the order of the slices, so the slices go in chronological order.  The
    persistence is the smallest S/N of the four partitions (the halves, even and odd).

    Under the sidelobe rule each ring, Path A's and the subsets', leaves out the ADI arcs of the
    candidate and of the known companions, with the farthest fallback (:func:`ring_snr`), and the STIM
    annulus leaves out what Path A's ring left out.  A candidate at a known companion is not left out
    of its own ring as a companion.

    It is ``near2_verify`` of the IDL predecessor (verify 493-746), with the departures in the module's
    Notes.
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
    srad = search_px(fwhm, found, rules, drift_px)
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
        peak_snr = None
        if sidelobes is None:
            # Path A on the combined image (verify 588), the search's S/N under the run's rules
            sa = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, srad,
                         exclude_xy=kxy, angle_convention=angle_convention, rules=rules)
            ssub = {name: snr_fap(maps["subsets"][name], rho, theta, pxscale, fwhm, exr, srad,
                                  exclude_xy=kxy, angle_convention=angle_convention, rules=rules)["snr"]
                    for name in SUBSET_NAMES}
            if srad == 0.0:
                peak_snr = snr_fap(maps["subsets"]["combined"], rho, theta, pxscale, fwhm, exr, 0.5 * fwhm,
                                   exclude_xy=kxy, angle_convention=angle_convention, rules=rules)["snr"]
            arcs = ()
        else:
            # the sidelobe rule: the candidate's own arc and the other known companions'
            others = _without_self(known, rho, theta, pxscale, fwhm, exr, angle_convention)
            rk = dict(exr=exr, srad=srad, exclude_xy=kxy, known=others, angle_convention=angle_convention,
                      rules=rules)
            sa = ring_snr(maps["subsets"]["combined"], 0, [rho], [theta], pxscale, fwhm, sidelobes, **rk)
            ssub = {name: (sa["snr"] if name == "combined" else
                           ring_snr(maps["subsets"][name], 0, [rho], [theta], pxscale, fwhm, sidelobes, **rk)["snr"])
                    for name in SUBSET_NAMES}
            if srad == 0.0:
                peak_snr = ring_snr(maps["subsets"]["combined"], 0, [rho], [theta], pxscale, fwhm, sidelobes,
                                    **dict(rk, srad=0.5 * fwhm))["snr"]
            arcs = sa.get("arcs") or ()
            noise = _noise(sa)
        # Path B: STIM at the nominal (rounded) position with the annulus null (594-602)
        xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
        sval, sfap = stim_fap(stim, float(xs[0]), float(ys[0]), rho / pxscale, fwhm, exr, exclude_xy=kxy,
                              excl_arcs=arcs, pxscale=pxscale, angle_convention=angle_convention,
                              pixel_mask=(rules or {}).get("pixel_mask"))
        bsig = bootstrap_nights(stack, w, rho, theta, pxscale, fwhm, srad, n_boot, rng,
                                angle_convention, rules=rules) if (n_boot and nn >= 2) else float("nan")
        part = [ssub[k] for k in SUBSET_NAMES[1:]]
        persist = float(np.nanmin(part)) if np.any(np.isfinite(part)) else float("nan")
        rows.append({"kind": "candidate", "index": c + 1, "rho": rho, "theta": theta,
                     "x": sa["x"], "y": sa["y"], "snr": sa["snr"], "fap": sa["fap"], "nap": sa["nap"],
                     "peak": sa["peak"], "detect": bool(np.isfinite(sa["fap"]) and sa["fap"] < fap_op),
                     "subsets": ssub, "persistence": persist, "stim": sval, "stim_fap": sfap,
                     "boot_sig": bsig, "search_px": srad, "position": "found" if found else "given"})
        if peak_snr is not None:
            rows[-1]["snr_peak"] = float(peak_snr)
        if found and field:
            rows[-1]["fap_field"] = field_fap(sa["fap"], field["area_px"], field["perimeter_px"],
                                              field["fwhm_eff_px"], field.get("euler", 0.0))
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
                                         weights=w[: inj_stack.shape[0]], drift_px=None if found else drift_px,
                                         excl_fwhm=excl_fwhm, angle_convention=angle_convention,
                                         fap_list=fap_list, tap_list=tap_list, rbin_as=rbin_as,
                                         combined=combined_inj, clean=clean, sidelobes=sidelobes, rules=rules)
        rows.extend(irows)
        maps["injected"] = imaps
        maps["limits"] = imaps["limits"]
    maps["meta"] = {"nights": ids, "fap_op": fap_op, "drift_px": srad, "excl_fwhm": excl_fwhm,
                    "fwhm": fwhm, "pxscale": pxscale, "n_boot": n_boot, "path_a": path_a,
                    "sidelobes": None if sidelobes is None else sidelobes.describe(),
                    "position": "found" if found else "given", "field": dict(field) if field else None}
    for r in rows:
        r.setdefault("meta", maps["meta"])
    if return_maps:
        return rows, maps
    return rows


def verify_injections(inj_stack: np.ndarray, sources: Sequence[Tuple[float, float, float]], fwhm: float,
                      pxscale: float, *, known: Sequence[Tuple[float, float]] = (),
                      weights: Optional[Sequence[float]] = None, drift_px: Optional[float] = None,
                      excl_fwhm: float = 1.5, angle_convention: str = "pa",
                      fap_list: Sequence[float] = DEFAULT_FAP_LIST, tap_list: Sequence[float] = DEFAULT_TAP_LIST,
                      rbin_as: Optional[float] = None, combined: Optional[np.ndarray] = None,
                      clean: Optional[np.ndarray] = None, sidelobes=None,
                      rules: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Measure the injection calibration block, then the detection-limit tables from its sources.

    Each source ``(rho, theta, contrast)`` gets its Path-A S/N and FAP, its subset S/N and its injected
    night-STIM value with the annulus FAP.  Its noise ring, on the combined image and the subsets,
    leaves out the other sources and the known ones within ``excl_fwhm`` FWHM of their nominal
    positions, as the run's metric does, and its STIM annulus leaves them and the source itself out
    within the same keep-out.  So a source's S/N is what it would be injected alone, to the noise.  The
    tables take the sources with a finite positive S/N and a positive contrast.

    Parameters
    ----------
    inj_stack : ndarray, shape (n, ny, nx) or (ny, nx)
        The injected per-night images of the ``n`` nights.  One image is one night.
    sources : sequence of tuple of floats
        The injected sources, as ``(rho_arcsec, theta_deg, contrast)``, the contrast NaN when left out.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    weights : sequence of float or None, default None
        Weight of each night, of which the first ``n`` are taken.  None weighs them equally.
    drift_px : float or None, default None
        Radius of the peak search, in pixels.  None reads each injection at its position
        (:func:`search_px`), or within 1.5 px for a run saved before 2026-11-25.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    fap_list : sequence of float, default DEFAULT_FAP_LIST
        FAPs per resolution element of the tables, the first the operating FAP.
    tap_list : sequence of float, default DEFAULT_TAP_LIST
        Completeness levels of the tables.
    rbin_as : float or None, default None
        Width of the separation bins, in arcsec.  None takes one FWHM.
    combined : ndarray or None, default None
        The injected image Path A measures.  None takes the weighted mean of the nights.
    clean : ndarray or None, default None
        The clean twin of Path A's image, which adds each source's solved contrasts and TABLE 3.
    sidelobes : Sidelobes or None, default None
        The run's sidelobe rule.  Each source's rings then leave out the ADI arcs of every source of the
        block and of the known companions (:func:`ring_snr`), and its STIM annulus and solved contrasts
        leave out what its Path-A ring left out.
    rules : dict or None, default None
        The run's ring rules, as :func:`verify_candidates` takes them.

    Returns
    -------
    rows : list of dict
        One row per source, holding ``kind`` ("injection"), ``index``, ``rho``, ``theta``, ``contrast``,
        ``x``, ``y``, ``snr``, ``fap``, ``nap``, ``peak``, ``subsets``, ``stim`` and ``stim_fap``, with
        ``noise`` under the sidelobe rule and, with ``clean``, ``c_det`` and ``c_tap_solved``
        (:func:`solved_contrasts` at ``fap_list[0]``).
    maps : dict
        The maps of :func:`verify_maps`, with ``limits`` (:func:`limit_tables`, TABLE 3 as
        ``limits["solved"]`` with ``clean``) and ``solved_items``, each solved source's row and
        detection test, for TABLE 3 over several blocks (:func:`verify_tiles`).

    Raises
    ------
    ValueError
        If ``clean`` is not of the frame size.

    Notes
    -----
    It is the calibration block of the IDL predecessor's ``near2_verify`` (verify 621-735).  The
    predecessor (verify 655-656, 660) leaves every other source in the ring and the annulus, and keeps
    no keep-out around the source in the STIM annulus.  The solved contrasts are not in the
    predecessor.
    """
    inj_stack = np.asarray(inj_stack, float)
    if inj_stack.ndim == 2:
        inj_stack = inj_stack[None]
    nn, ny, nx = inj_stack.shape
    cx, cy = star_center((ny, nx))
    w = np.ones(nn) if weights is None else np.asarray(weights, float)[:nn]
    exr = excl_fwhm * fwhm
    drift_px = search_px(fwhm, False, rules, drift_px)
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
    """Return a source's place, ``(rho, PA)`` rounded, to tell whether an image holds it.

    Parameters
    ----------
    s : sequence of float
        The source, as ``(rho_arcsec, theta_deg, ...)``.

    Returns
    -------
    key : tuple of 2 floats
        The separation rounded to 1e-6 arcsec and the angle to 1e-4 degrees.
    """
    return (round(float(s[0]), 6), round(float(s[1]), 4))


def _tile_stack(stack: Optional[np.ndarray], img: np.ndarray,
                parts: Optional[Sequence[Any]]) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return an annulus' per-partition images and their ids, or its image alone.

    Parameters
    ----------
    stack : ndarray or None
        The annulus' per-partition images, ``(n, ny, nx)`` or one ``(ny, nx)``.  None takes the image
        alone.
    img : ndarray, shape (ny, nx)
        The annulus' image.
    parts : sequence or None
        Id of each partition.

    Returns
    -------
    stack : ndarray, shape (n, ny, nx)
        The images, one for the image alone.
    ids : list of str or None
        The partitions' ids, None for the image alone or when ``parts`` does not match the stack.
    """
    if stack is None:
        return np.asarray(img, float)[None], None
    st = np.asarray(stack, float)
    if st.ndim == 2:
        st = st[None]
    ids = [str(p) for p in parts] if parts is not None and len(parts) == st.shape[0] else None
    return st, ids


def verify_tiles(annuli: Sequence[Dict[str, Any]], candidates: Sequence[Tuple[float, float]],
                 sources: Sequence[Tuple[float, ...]], fwhm: float, pxscale: float, *,
                 known: Sequence[Tuple[float, float]] = (), fap_op: float = 2.87e-7,
                 drift_px: Optional[float] = None, n_boot: int = 500, rng: Optional[np.random.Generator] = None, excl_fwhm: float = 1.5,
                 fap_list: Sequence[float] = DEFAULT_FAP_LIST, tap_list: Sequence[float] = DEFAULT_TAP_LIST,
                 rbin_as: Optional[float] = None, angle_convention: str = "pa", sidelobes=None,
                 rules: Optional[Dict[str, Any]] = None, stitch: Optional[Dict[str, Any]] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Measure each candidate and injected source with the pixels of the annulus whose tile holds it.

    Each position is measured as that annulus reduced it, beside the stitch, which blends or patches a
    tile's pixels in places (NEAR2 s7 note, 2026-11-02).  The tile holding a position is
    :func:`~klip_tpe.stitch.tile_of` of its separation, and a position outside every tile gets no row.
    Each candidate is measured as :func:`verify_candidates` measures it on the stitch, its ring without
    the known companions.  Each source is measured as :func:`verify_injections` measures it, its ring
    without the other sources its image holds, those of the other tiles and of the stitch too.

    Parameters
    ----------
    annuli : sequence of dict
        One dict per annulus, in the stitch's order, holding ``tile`` (``(rin, rout)`` in pixels),
        ``clean`` and ``inj`` (its Path-A images as the stitch took them), ``clean_stack`` and
        ``inj_stack`` (its per-partition images, else the image alone), ``parts`` (their ids),
        ``inj_sources`` (the sources its injected image holds) and ``annulus`` (its index).
    candidates : sequence of tuple of 2 floats
        The candidates, as ``(rho_arcsec, theta_deg)``.
    sources : sequence of tuple of floats
        The injected sources, as ``(rho_arcsec, theta_deg, contrast)``.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    fap_op : float, default 2.87e-7
        Operating FAP of Path A's decision.
    drift_px : float or None, default None
        Radius of the peak search, in pixels.  None reads the given positions at themselves
        (:func:`search_px`), or within 1.5 px for a run saved before 2026-11-25.
    n_boot : int, default 500
        Number of bootstrap draws, 0 to skip the bootstrap.
    rng : numpy.random.Generator or None, default None
        Generator of the bootstrap, shared by the annuli.  None seeds one with 42.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    fap_list : sequence of float, default DEFAULT_FAP_LIST
        FAPs per resolution element of the tables, the first the operating FAP.
    tap_list : sequence of float, default DEFAULT_TAP_LIST
        Completeness levels of the tables.
    rbin_as : float or None, default None
        Width of the separation bins, in arcsec.  None takes one FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    sidelobes : Sidelobes or None, default None
        The run's sidelobe rule.
    rules : dict or None, default None
        The run's ring rules.
    stitch : dict or None, default None
        The stitched images, which the runner passes, holding ``clean``, ``inj``, the per-partition cubes
        ``nights`` and ``nights_inj`` with their ids ``night_ids``, and ``inj_sources``, the sources the
        injected stitch holds.  Each annulus is then measured on the stitch with its tile put in
        (:func:`~klip_tpe.stitch.put_tile_in`, each partition by its id).

    Returns
    -------
    rows : list of dict
        The candidate rows, then the source rows, in the order given, each with ``annulus`` (1-based),
        ``tile_px``, ``meta`` and, when measured on the stitch, ``in_stitch``.  A source whose tile's
        annulus did not inject it gets a row marked ``absent`` and no S/N.
    maps : dict
        ``limits``, the detection-limit tables of the sources measured, TABLE 3 included, and ``meta``.

    Notes
    -----
    Without ``stitch``, each annulus is measured on its image alone, as the search and the validation
    measure its sources.  That image is NaN outside the annulus' zone, and the matched filter and the
    noise ring take zeros there.  Where the next annulus in has residuals several times the zone's,
    that leaves noise out of the ring and the radial band, and inside about 1.3" of NEAR2's runs the
    image alone read 1.3 to 6 times the stitch's S/N (NEAR2 note, 2026-11-04).  Where a tile takes no
    blend, the stitch with the tile put in is the stitch itself, so the two S/N agree there.

    A source is absent where each annulus injected only its own sources
    (``RunConfig.stitch_injections = "own"``).
    """
    from .stitch import put_tile_in, put_tile_in_stack, tile_index, tile_of
    if rng is None:
        rng = np.random.default_rng(42)
    tiles = [(float(a["tile"][0]), float(a["tile"][1])) for a in annuli]
    nan = float("nan")
    crow: Dict[int, Dict[str, Any]] = {}
    srow: Dict[int, Dict[str, Any]] = {}
    items: List[Tuple[Dict[str, Any], Any]] = []
    nights: List[str] = []
    st = stitch if stitch and stitch.get("clean") is not None else None
    idx = None if st is None else tile_index(np.shape(st["clean"]), tiles)
    for j, a in enumerate(annuli):
        cj = [k for k, c in enumerate(candidates) if tile_of(float(c[0]) / pxscale, tiles) == j]
        sj = [m for m, s in enumerate(sources) if tile_of(float(s[0]) / pxscale, tiles) == j]
        if not cj and not sj:
            continue
        tag = {"annulus": int(a.get("annulus", j)) + 1, "tile_px": tiles[j]}
        clean = np.asarray(a["clean"], float)
        in_st = st is not None and np.shape(st["clean"]) == clean.shape
        if in_st:
            tag["in_stitch"] = True
            mine_tile = idx == j
            clean = put_tile_in(st["clean"], clean, mine_tile)
        stack, ids = _tile_stack(a.get("clean_stack"), clean, a.get("parts"))
        if in_st and a.get("clean_stack") is not None:
            stack = put_tile_in_stack(st.get("nights"), st.get("night_ids"), stack, ids, mine_tile)
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
        if in_st:
            inj = put_tile_in(st.get("inj"), inj, mine_tile)
        istack, iids = _tile_stack(a.get("inj_stack"), inj, a.get("parts"))
        if in_st and a.get("inj_stack") is not None:
            istack = put_tile_in_stack(st.get("nights_inj"), st.get("night_ids"), istack, iids, mine_tile)
        mine = {_place_key(sources[m]) for m in meas}
        held = list(a.get("inj_sources") or []) + (list(st.get("inj_sources") or []) if in_st else [])
        others, seen = [], set(mine)
        for s in held:                    # the image's other sources: the annulus' and, around its tile, the stitch's
            if _place_key(s) not in seen:
                seen.add(_place_key(s))
                others.append((float(s[0]), float(s[1])))
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
    meta = {"nights": nights, "fap_op": fap_op, "drift_px": search_px(fwhm, False, rules, drift_px),
            "excl_fwhm": excl_fwhm, "fwhm": fwhm,
            "pxscale": pxscale, "n_boot": n_boot,
            "path_a": ("the stitched image with the tile of the annulus that holds the position put in: that "
                       "annulus' own pixels on its tile, the stitch's around it" if st is not None else
                       "the image of the annulus whose tile holds the position, as the stitch took it"),
            "sidelobes": None if sidelobes is None else sidelobes.describe()}
    for r in rows:
        r["meta"] = meta
    return rows, {"limits": limits, "meta": meta}


def tile_comparison_lines(rows: Sequence[Dict[str, Any]], tile_rows: Sequence[Dict[str, Any]],
                          report: str = "tile_verify_report.txt") -> List[str]:
    """Return the closing section of the stitched report, each S/N on the stitch beside its S/N on its tile.

    The S/N on the tile is that of :func:`verify_tiles`, given in full in ``report``.

    Parameters
    ----------
    rows : sequence of dict
        The rows of the stitched verify.
    tile_rows : sequence of dict
        The rows of :func:`verify_tiles`.
    report : str, default "tile_verify_report.txt"
        Name of the per-tile report, for the section's heading.

    Returns
    -------
    lines : list of str
        The section, every line a comment.
    """
    def key(r):
        return (r.get("kind", "candidate"), int(r["index"]))

    tile = {key(r): r for r in tile_rows}
    rho_h = 'rho["]'
    if any(r.get("in_stitch") for r in tile_rows):
        what = ["#   each S/N on the stitched image, and on the stitch with the tile of the annulus that holds the",
                "#   position put in.  The two differ only where the stitch blends or patches that tile.  In full in",
                "#   that report."]
    else:
        what = ["#   each S/N on the stitched image, and on the image of the annulus whose tile holds the position,",
                "#   the image the search and the validation measure that annulus' sources on.  In full in that report."]
    L = ["#", f"# ==== the stitch beside the tiles ({report}) ====", *what,
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
    """Bin the Path-A detection limits of the injections by separation, TABLE 1 and TABLE 2.

    The injections with a finite positive S/N and a positive contrast are binned by separation, from
    the smallest.  TABLE 1 gives each bin's :func:`contrast_limit` at the operating FAP for each
    completeness, and TABLE 2 at the completeness nearest 0.9 for each FAP, with a per-image 10 %
    column at ``alpha = 0.1 / max(mean(nap), 1)``.

    Parameters
    ----------
    inj_rows : sequence of dict
        Rows of :func:`verify_injections`, rows of another kind skipped.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fap_list : sequence of float, default DEFAULT_FAP_LIST
        FAPs per resolution element, the first the operating FAP.
    tap_list : sequence of float, default DEFAULT_TAP_LIST
        Completeness levels.
    rbin_as : float or None, default None
        Width of the bins, in arcsec.  None takes ``fwhm * pxscale``.

    Returns
    -------
    limits : dict or None
        ``rbin_as``, ``fap_list``, ``tap_list``, ``i90`` (the index of the completeness nearest 0.9),
        ``n_inj``, ``bins``, ``samples`` (the injections' ``rho``, ``snr``, ``contrast`` and ``nap``) and
        ``solved``, None until :func:`verify_injections` fills it with TABLE 3.  None when no injection
        qualifies.

    Notes
    -----
    Each bin holds ``sep`` (its mean separation), ``n_inj``, ``snr_med``, ``tau``
    (``t_thresh(fap_list[0], max(round(mean(nap)), 3))``), ``c_tap`` (TABLE 1), ``c_fap`` and
    ``c_img10`` (TABLE 2), ``ib`` (its index) and ``tau_exact``, the largest exact threshold of its
    injections.  It is the IDL predecessor's verify 674-718 and ``near2v_curvetxt`` (447-488).
    """
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
    """Format a number, "nan" where it is not finite.

    Parameters
    ----------
    v : object
        The value.
    fmt : str, default "{:.2f}"
        The format.

    Returns
    -------
    text : str
        The value formatted, or ``str(v)`` for a value that is not a number.
    """
    try:
        return fmt.format(float(v)) if np.isfinite(float(v)) else "nan"
    except Exception:
        return str(v)


def _table_lines(lim: Dict[str, Any], indent: str = "#   ") -> List[str]:
    """Return TABLE 1 and TABLE 2 of the report as lines.

    Where a bin's exact threshold passes the cap (:data:`TAU_CAP`), a note says where.

    Parameters
    ----------
    lim : dict
        The tables, from :func:`limit_tables`.
    indent : str, default "#   "
        Prefix of the headings and notes.

    Returns
    -------
    lines : list of str
        The tables.
    """
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
    """Format a contrast as ``{:10.3E}``, "inf" or "nan" right-aligned where it is not finite.

    Parameters
    ----------
    v : float
        The contrast.

    Returns
    -------
    text : str
        The contrast, 10 characters wide.
    """
    v = float(v)
    return f"{v:10.3E}" if np.isfinite(v) else f"{'inf' if np.isinf(v) else 'nan':>10}"


def _table3_lines(lim: Dict[str, Any], indent: str = "#   ") -> List[str]:
    """Return TABLE 3 of the report as lines, from ``lim["solved"]``.

    Parameters
    ----------
    lim : dict
        The tables, from :func:`limit_tables`, with TABLE 3 filled (:func:`_solved_bins`).
    indent : str, default "#   "
        Prefix of the headings and notes.

    Returns
    -------
    lines : list of str
        The table, its notes and a line per source.
    """
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
    """Return the injection block's rings under the sidelobe rule as comment lines.

    Parameters
    ----------
    injs : sequence of dict
        The injection rows, each with its ``noise`` (:func:`_noise`).

    Returns
    -------
    lines : list of str
        How many rings kept their samples outside every arc, and a line for each that took a fallback.
    """
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
    """Write the verify report, ``verify_report.txt``, from the rows of :func:`verify_candidates`.

    The report gives each candidate's Path A and Path B, the injection calibration block and the
    detection-limit tables, TABLE 3 included where solved.  Rows of :func:`verify_tiles` also say whose
    image measured them.

    Parameters
    ----------
    path : str
        Path of the report.
    rows : sequence of dict
        The rows, candidates and injections.
    limits : dict or None, default None
        The detection-limit tables.  None makes them from the injection rows present.
    header : dict or None, default None
        Settings for the header, such as ``run``, which take precedence over the rows' ``meta``.
    extra_lines : sequence of str, default ()
        Lines that close the report.

    Returns
    -------
    text : str
        The report, also written to ``path``.

    Notes
    -----
    It is the report of the IDL predecessor's ``near2_verify`` (verify 574-724).
    """
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
    srad = float(meta.get("drift_px", LEGACY_SEARCH_PX) if meta.get("drift_px") is not None else LEGACY_SEARCH_PX)
    where = (f"the matched filter's peak within {srad:.1f} px of each position" if srad > 0 else
             "each source at its given position")
    L.append(f"# operating FAP threshold (per resolution element): {fap_op:10.2E}   read at: {where}")
    fld = meta.get("field")
    if meta.get("position") == "found" and fld:
        L.append(f"#   found by a blind search: FAP_field corrects for the search over {float(fld['area_px']):.0f} px "
                 f"(smoothness FWHM {float(fld['fwhm_eff_px']):.2f} px, random-field theory)")
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
        tt = f", tile {float(t[0]):.2f}-{float(t[1]):.2f} px" if t is not None else ""
        if "annulus" not in r:
            return ""
        if r.get("in_stitch"):
            return f"the stitch with annulus {r['annulus']}'s tile put in{tt}"
        return f"annulus {r['annulus']}'s image{tt}"

    for r in cands:
        s = r["subsets"]
        L.append(f"== candidate {r['index']}  rho={r['rho']:.3f}\"  PA={r['theta']:.1f} deg ==")
        if "annulus" in r:
            L.append(f"   measured on {_on(r)}")
        fb = (r.get("noise") or {}).get("fallback")
        L.append(f"   [A] combined S/N = {_f(r['snr'])}   FAP = {_f(r['fap'], '{:10.2E}')}   (nap={int(r['nap'])}"
                 + (f", {fb}" if fb else "") + ")")
        if r.get("snr_peak") is not None:
            L.append(f"   [A] peak S/N within FWHM/2 = {_f(r['snr_peak'])}   (for comparison, not the FAP's)")
        if r.get("fap_field") is not None:
            L.append(f"   [A] FAP over the searched field = {_f(r['fap_field'], '{:10.2E}')}")
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
        in_st = any(r.get("in_stitch") for r in injs)
        L.append("#   columns: rho[\"] PA[deg] contrast  | combined S/N  FAP  | injected-STIM  STIM-FAP"
                 + ("  | the annulus whose tile, put into the stitch, measured it" if in_st else
                    "  | the annulus whose image measured it" if tiled else ""))
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
    """Write TABLE 1 and TABLE 2 of an aggregated injection set as a stand-alone ``verify_curve.txt``.

    Such a set is, for example, every injection of an annulus' evaluations, as the IDL predecessor's
    ``near2_verify_annulus`` gathers them.

    Parameters
    ----------
    path : str
        Path of the file.
    limits : dict
        The tables, from :func:`limit_tables`.
    header : str, default ""
        Line under the title.

    Returns
    -------
    text : str
        The tables, also written to ``path``.

    Notes
    -----
    It is ``near2v_curvetxt`` of the IDL predecessor (verify 447-488).
    """
    L = [f"# klip_tpe.verify detection-limit curve -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         f"# {header}",
         f"# contrast = companion/star flux ratio; binned by rho in {limits['rbin_as']:.2f}\" bins."]
    L += _table_lines(limits, indent="# ")
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text
