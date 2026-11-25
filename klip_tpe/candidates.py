"""Blind candidate search, per-peak metrics, scoring and tiering -- port of
``near2_candidates.pro`` (candidates 150-435).

Pipeline: iterative peak picking on a Mawet S/N map (``near2c_detect``), then for
every peak the per-night matched-filter S/N, the aperture STIM across nights, the
even/odd subset S/N, a Gaussian PSF correlation (``near2c_psfcorr``), the
look-elsewhere-corrected small-sample FAP, cross-matches with known / injected
sources, a multiplicative score and the A/B/C tier.  Optionally the top-N ranked
peaks are passed to :func:`klip_tpe.verify.verify_candidates`.

Conventions as in :mod:`klip_tpe.verify` (``[y, x]`` images, star at
``((nx-1)/2, (ny-1)/2)``, PA East of North).  Inverse mapping of a peak pixel:
``PA = (atan2(dy, dx) deg - 90) mod 360`` (candidates 259).

Deliberate deviations / omissions:

* The cross-annulus metric cache (``cache`` / ``r_freeze_px``, candidates
  249-272, 322-330) is a run-time optimization of the IDL "running" pass; the
  port always recomputes (metrics are deterministic, so results are identical).
* The S/N map is built with :func:`klip_tpe.products.snr_map` on the
  ``radprof``-flattened image when not supplied (the IDL reads the pre-computed
  ``klip_stitched_snr.fits`` from ``near2m_snrmap``).
* The detection-map PDF and ``README.txt`` are not produced.
* The combined S/N, its FAP and the PSF-likeness are measured on the image given,
  the run's stitch, when one is given (candidates 230 takes the nights' mean).  At an
  edge between two annuli that mean is not the stitch, where the annuli keep
  different nights (see :mod:`klip_tpe.verify`).  The per-night S/N, the STIM and
  the even / odd subsets stay on the nights.
* Under the run's sidelobe rule (``sidelobes=``) each peak's S/N leaves its own ADI arc
  and the known companions' out of its rings, and the S/N map made here leaves out the
  known companions' (:func:`klip_tpe.verify.ring_snr`).
* Since 2026-11-25 (the audit of that date) each peak is read at the matched filter's
  maximum within FWHM/2 of where the map peaked, in place of 1.5 px, and its FAP over the
  field comes from random-field theory (:func:`klip_tpe.verify.field_fap`) with the S/N
  map's measured smoothness, in place of ``1 - (1 - p)^N_resel``.  On source-free fields of
  Gaussian noise correlated at the PSF that count's FAP was reached 2.6 to 3 times as often
  as it said, at 0.1 to 0.01.  A run saved before keeps both (``RunConfig.verify_position =
  "peak"``).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import matched_filter, nanmedian_even, radprof, rule_kwargs, source_xy, star_center, verify_idl
from .products import snr_map
from .verify import (_noise, _without_self, field_fap, field_geometry, legacy_position, mf_convolve, ring_snr,
                     search_px, snr_fap, verify_candidates)

__all__ = ["detect_peaks", "psf_correlation", "candidate_metrics", "score_candidates",
           "find_candidates", "write_candidates", "coverage_rim"]


def coverage_rim(image: np.ndarray, stack: Optional[np.ndarray], r_lo: float, r_hi: float,
                 drop: float = 0.8, stack_from: Optional[float] = None) -> Optional[float]:
    """Return the rim where the frames stop covering the field, the radius the search should stop at.

    It is the innermost integer radius in ``[r_lo, r_hi]`` where the image, or from ``stack_from`` on a
    layer of ``stack`` that reaches that ring at all, is finite on less than ``drop`` of the largest
    fraction it covered on the rings inside.

    Parameters
    ----------
    image : ndarray, shape (ny, nx)
        The combined image.
    stack : ndarray or None
        The per-night images, ``(nn, ny, nx)``.  A layer with nothing at a radius is a night left out
        of that annulus and does not count.
    r_lo, r_hi : float
        Radii checked, in pixels.
    drop : float, default 0.8
        Fraction of the best coverage inside below which a ring is the rim.
    stack_from : float or None, default None
        Radius from which the nights are checked, in pixels, which keeps their check off the seams
        inside the field.  None takes ``r_lo``.

    Returns
    -------
    rim : float or None
        The rim's radius, in pixels, None when no ring drops.

    Notes
    -----
    A share of every ring that is always missing, as along the MIRI 4QPM boundaries, keeps the
    fraction level and is not a rim.
    """
    img = np.asarray(image, float)
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rint = np.rint(np.hypot(xx - cx, yy - cy)).astype(int)
    layers = [img]
    if stack is not None and np.ndim(stack) == 3:
        layers += [np.asarray(lay, float) for lay in stack]
    best = [0.0] * len(layers)
    s_from = float(r_lo) if stack_from is None else float(stack_from)
    for r in range(int(np.ceil(r_lo)), int(np.floor(r_hi)) + 1):
        ring = rint == r
        if not ring.any():
            continue
        for j, lay in enumerate(layers):
            if j > 0 and r < s_from:
                continue
            f = float(np.isfinite(lay[ring]).mean())
            if f == 0.0:
                continue
            if best[j] > 0 and f < drop * best[j]:
                return float(r)
            best[j] = max(best[j], f)
    return None


# ----------------------------------------------------------------------------
# step 1 -- peak detection  (near2c_detect, candidates 32-51)
# ----------------------------------------------------------------------------
def detect_peaks(snrmap: np.ndarray, snrmin: float, sep: float, iwa: float, owa: float,
                 nmax: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the peaks of an S/N map, the highest first, each at least ``sep`` from those found before.

    Non-finite pixels and those outside ``[iwa, owa]`` are left out.  The map's maximum is taken and a
    disk of radius ``sep`` around it suppressed, again and again, until the maximum falls below
    ``snrmin`` or ``nmax`` peaks are found (``near2c_detect``, candidates 32-51).

    Parameters
    ----------
    snrmap : ndarray, shape (ny, nx)
        The S/N map, with the star at ``((nx-1)/2, (ny-1)/2)``.
    snrmin : float
        Smallest S/N taken.
    sep : float
        Radius of the suppression, in pixels.
    iwa, owa : float
        Radii of the search, in pixels.
    nmax : int, default 400
        Most peaks taken.

    Returns
    -------
    px, py : ndarray of int
        The peaks' columns and rows.
    snr : ndarray
        The peaks' values.
    """
    m = np.asarray(snrmap, float)
    ny, nx = m.shape
    cx, cy = star_center(m.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    work = np.where(np.isfinite(m) & (rr >= iwa) & (rr <= owa), m, -1e30)
    px, py, ps = [], [], []
    for _ in range(int(nmax)):
        im = int(np.argmax(work))
        mx = float(work.flat[im])
        if not np.isfinite(mx) or mx < snrmin:
            break
        iy, ix = divmod(im, nx)
        px.append(ix); py.append(iy); ps.append(mx)
        work[(xx - ix) ** 2 + (yy - iy) ** 2 <= sep * sep] = -1e30
    return np.array(px, int), np.array(py, int), np.array(ps, float)


# ----------------------------------------------------------------------------
# PSF-likeness  (near2c_psfcorr, candidates 57-72)
# ----------------------------------------------------------------------------
def psf_correlation(img: np.ndarray, px: int, py: int, fwhm: float) -> float:
    """Return how much a peak looks like a point source, its stamp's correlation with a Gaussian.

    The stamp, of half-width ``max(round(1.5 fwhm), 3)`` and its median subtracted, is correlated with a
    zero-mean Gaussian of the FWHM centered on the peak, over its finite pixels (``near2c_psfcorr``).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    px, py : int
        The peak's column and row.
    fwhm : float
        FWHM of the PSF, in pixels.

    Returns
    -------
    corr : float
        The normalized correlation, in ``[-1, 1]``, NaN for fewer than 5 finite pixels or a flat stamp.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    hb = max(int(round(1.5 * fwhm)), 3)
    x0, x1 = max(px - hb, 0), min(px + hb, nx - 1)
    y0, y1 = max(py - hb, 0), min(py + hb, ny - 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    st = img[y0:y1 + 1, x0:x1 + 1]
    g = np.isfinite(st)
    if g.sum() < 5:
        return float("nan")
    st = st - np.median(st[g])
    gy, gx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
    gx = gx - (px - x0)
    gy = gy - (py - y0)
    sig = fwhm / 2.3548
    psf = np.exp(-0.5 * (gx ** 2 + gy ** 2) / sig ** 2)
    psf = psf - psf.mean()
    num = float(np.sum(st[g] * psf[g]))
    den = float(np.sqrt(np.sum(st[g] ** 2) * np.sum(psf[g] ** 2)))
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------------------
# step 2 -- per-candidate metrics  (candidates 222-307)
# ----------------------------------------------------------------------------
def candidate_metrics(px: np.ndarray, py: np.ndarray, snr0: np.ndarray, stack: np.ndarray, fwhm: float,
                      pxscale: float, iwa: float, owa: float, *, known: Sequence[Tuple[float, float]] = (),
                      injected: Sequence[Tuple[float, ...]] = (), excl_fwhm: float = 1.5,
                      srad: Optional[float] = None, angle_convention: str = "pa", flatten: bool = False,
                      image: Optional[np.ndarray] = None, sidelobes=None,
                      rules: Optional[Dict[str, Any]] = None,
                      field: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Measure each peak of a blind search on the per-night stack and the combined image.

    Each peak is read at the matched filter's maximum within ``srad`` of the map's peak pixel: its S/N
    in each night (``pn``), in the even and odd nights' means (``sub1``, ``sub2``) and on the combined
    image, its aperture STIM over the nights, its Gaussian PSF correlation, and its FAPs.  ``fap_resel``
    is the small-sample t FAP at the peak (:func:`klip_tpe.verify.t_fap`), and ``fap`` the FAP over the
    searched field (:func:`klip_tpe.verify.field_fap`).

    Parameters
    ----------
    px, py : ndarray of int
        The peaks' columns and rows (:func:`detect_peaks`).
    snr0 : ndarray
        The peaks' values on the S/N map, which the combined S/N replaces where it is finite.
    stack : ndarray, shape (nn, ny, nx) or (ny, nx)
        The per-night images, not flattened.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    iwa, owa : float
        Inner and outer radii of the search, in pixels, for the field when ``field`` is None.
    known : sequence of tuple, default ()
        Known companions as ``(rho_arcsec, theta_deg)``, flagged ``KNOWN`` within 0.8 FWHM and left out
        of every ring.
    injected : sequence of tuple, default ()
        Injected sources, flagged ``INJ`` within 0.8 FWHM.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    srad : float or None, default None
        Radius of the peak search, in pixels.  None takes :func:`klip_tpe.verify.search_px` of a found
        source: FWHM/2, or 1.5 px for a run saved before 2026-11-25.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    flatten : bool, default False
        Whether to subtract each image's radial profile first.
    image : ndarray or None, default None
        The combined image the run made, its stitch, on which the combined S/N, the FAPs and the PSF
        correlation are measured.  None takes the nights' mean, as the IDL predecessor does.
    sidelobes : Sidelobes or None, default None
        The run's sidelobe rule, under which every ring leaves out the peak's own ADI arc and the known
        companions' (:func:`klip_tpe.verify.ring_snr`), and each row adds ``noise``.
    rules : dict or None, default None
        The run's ring rules (:func:`~klip_tpe.metrics.ring_rules`), so every S/N is the search's.
    field : dict or None, default None
        The search's field (:func:`klip_tpe.verify.field_geometry`).  None takes the annulus from
        ``iwa`` to ``owa`` with a smoothness of sqrt(2) FWHM.

    Returns
    -------
    rows : list of dict
        One per peak: ``x``, ``y``, ``rho``, ``theta``, ``map_snr``, ``snr``, ``fap_resel``, ``fap``,
        ``nap``, ``per_night``, ``n_nights`` (nights at S/N 2 or more), ``pn_min``, ``pn_med``,
        ``pn_max``, ``stim``, ``sub1``, ``sub2``, ``psf_corr``, ``flag`` and, under the sidelobe rule,
        ``noise``.

    Notes
    -----
    It is the per-peak part of ``near2_candidates`` of the IDL predecessor (candidates 222-307), whose
    1.5 px search and ``1 - (1 - p)^N_resel``, ``N_resel = pi (owa^2 - iwa^2) / fwhm^2``, a run saved
    before 2026-11-25 keeps (``RunConfig.verify_position = "peak"``).
    """
    stack = np.asarray(stack, float)
    if stack.ndim == 2:
        stack = stack[None]
    nn, ny, nx = stack.shape
    cx, cy = star_center((ny, nx))
    exr = excl_fwhm * fwhm
    srad = search_px(fwhm, True, rules, srad)
    nresel = max(np.pi * (owa ** 2 - iwa ** 2) / fwhm ** 2, 1.0)
    legacy = legacy_position(rules)
    if field is None:
        field = {"area_px": np.pi * (owa ** 2 - iwa ** 2), "perimeter_px": 2.0 * np.pi * (owa + iwa),
                 "fwhm_eff_px": float(np.sqrt(2.0) * fwhm), "euler": 0.0}
    ap = max(0.5 * fwhm, 1.5)
    _flat = radprof if flatten else (lambda a: np.asarray(a, float))
    slices = np.stack([_flat(stack[j]) for j in range(nn)])
    fin = np.isfinite(stack)

    def _nanmean(sel):
        z = np.where(fin[sel], stack[sel], 0.0)
        return z.sum(axis=0) / np.maximum(fin[sel].sum(axis=0), 1)

    if image is not None:
        image = np.asarray(image, float)
        if image.shape != (ny, nx):
            raise ValueError(f"candidate_metrics: image {image.shape} is not the stack's frame size {(ny, nx)}")
        comb = _flat(image)
    else:
        comb = _flat(_nanmean(slice(None))) if nn > 1 else slices[0]
    if nn >= 2:
        i1, i2 = np.arange(nn) % 2 == 0, np.arange(nn) % 2 == 1
        s1 = _flat(_nanmean(i1)) if i1.any() else comb
        s2 = _flat(_nanmean(i2)) if i2.any() else comb
    else:
        s1 = s2 = comb
    # each image's matched-filter map under the run's rules, made once per kernel (a measured
    # kernel changes with separation), or its Gaussian map under the measures of before
    # 2026-11-01 (RunConfig.verify_snr = "idl").  The nights are held as one list, so each keeps
    # its own object, and the cache keeps the image it was made from: a view made anew for each
    # peak can take the address of one freed before it, and was handed that one's map
    nights = [slices[j] for j in range(nn)]
    maps: Dict[Tuple[int, Any], Tuple[np.ndarray, np.ndarray]] = {}
    idl = verify_idl(rules)

    def _mf(img_: np.ndarray, rho_: float) -> np.ndarray:
        k = None if idl else rule_kwargs(rules, rho_, img_.shape).get("kernel")
        key = (id(img_), None if k is None else np.asarray(k, float).tobytes())
        hit = maps.get(key)
        if hit is None or hit[0] is not img_:
            hit = maps[key] = (img_, mf_convolve(img_, fwhm) if idl else matched_filter(img_, fwhm, rules, rho_))
        return hit[1]
    yy, xx = np.mgrid[0:ny, 0:nx]
    kx, ky = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention) \
        if len(known) else (np.zeros(0), np.zeros(0))
    ix_, iy_ = source_xy([k[0] for k in injected], [k[1] for k in injected], pxscale, cx, cy, angle_convention) \
        if len(injected) else (np.zeros(0), np.zeros(0))
    kxy = list(zip(kx, ky))

    rows: List[Dict[str, Any]] = []
    for c in range(len(px)):
        dx, dy = px[c] - cx, py[c] - cy
        rho = float(np.hypot(dx, dy) * pxscale)
        if angle_convention == "pa":
            pa = (np.degrees(np.arctan2(dy, dx)) - 90.0) % 360.0
        else:
            pa = np.degrees(np.arctan2(dy, dx)) % 360.0
        pa = float(pa)
        args = (rho, pa, pxscale, fwhm, exr, srad)
        kw = dict(exclude_xy=kxy, angle_convention=angle_convention, rules=rules)
        if sidelobes is not None:
            # the sidelobe rule: the peak's own arc and the other known companions'
            rk = dict(exr=exr, srad=srad, exclude_xy=kxy, angle_convention=angle_convention, rules=rules,
                      known=_without_self(known, rho, pa, pxscale, fwhm, exr, angle_convention))

            def _fap(img_):
                return ring_snr(img_, 0, [rho], [pa], pxscale, fwhm, sidelobes, mfmap=_mf(img_, rho), **rk)
        else:
            def _fap(img_):
                return snr_fap(img_, *args, mfmap=_mf(img_, rho), **kw)
        pn = np.array([_fap(nights[j])["snr"] for j in range(nn)])
        if not np.any(np.isfinite(pn)):
            pn[:] = 0.0
        n_nights = int(np.sum(pn >= 2.0))
        m = (xx - px[c]) ** 2 + (yy - py[c]) ** 2 <= ap * ap
        av = np.array([np.nanmean(slices[j][m]) if np.any(np.isfinite(slices[j][m])) else np.nan
                       for j in range(nn)])
        gav = av[np.isfinite(av)]
        stim = float(np.mean(gav) / np.std(gav, ddof=1)) if (gav.size >= 2 and np.std(gav, ddof=1) > 0) else float("nan")
        sub1 = _fap(s1)["snr"]
        sub2 = _fap(s2)["snr"] if nn >= 2 else float("nan")
        psfc = psf_correlation(comb, int(px[c]), int(py[c]), fwhm)
        cf = _fap(comb)
        csnr = float(cf["snr"]) if np.isfinite(cf["snr"]) else float(snr0[c])
        if not np.isfinite(cf["fap"]):
            fap = float("nan")
        elif legacy:
            fap = float(min(1.0 - (1.0 - cf["fap"]) ** nresel, 1.0))
        else:
            fap = field_fap(cf["fap"], field["area_px"], field["perimeter_px"], field["fwhm_eff_px"],
                            field.get("euler", 0.0))
        flag = ""
        if kx.size and np.any(np.hypot(px[c] - kx, py[c] - ky) <= 0.8 * fwhm):
            flag = "KNOWN"
        if ix_.size and np.any(np.hypot(px[c] - ix_, py[c] - iy_) <= 0.8 * fwhm):
            flag = "INJ" if flag == "" else flag + "/INJ"
        rows.append({"x": int(px[c]), "y": int(py[c]), "rho": rho, "theta": pa, "map_snr": float(snr0[c]),
                     "snr": csnr, "fap_resel": float(cf["fap"]), "fap": fap, "nap": int(cf["nap"]),
                     "per_night": pn.tolist(), "n_nights": n_nights,
                     "pn_min": float(np.nanmin(pn)), "pn_med": nanmedian_even(pn), "pn_max": float(np.nanmax(pn)),
                     "stim": stim, "sub1": float(sub1), "sub2": float(sub2), "psf_corr": float(psfc), "flag": flag})
        if sidelobes is not None:
            rows[-1]["noise"] = _noise(cf)
    return rows


# ----------------------------------------------------------------------------
# step 3 -- score & tier  (candidates 336-366)
# ----------------------------------------------------------------------------
def score_candidates(rows: List[Dict[str, Any]], nn: int, persistence_map: Optional[np.ndarray] = None,
                     stim_half: float = 1.5) -> List[Dict[str, Any]]:
    """Score, tier and rank the candidates, in place, and return them sorted by score.

    ::

        f_nt  = n_nights / nn
        f_psf = clip(psf_corr, 0, 1)             (NaN -> 0)
        f_stm = stim / (|stim| + 1.5)            (NaN or stim <= 0 -> 0)
        f_pv  = pv / (|pv| + 1.5)                (pv = max of the 3x3 box of the
                                                  param-persistence map; 1 if absent)
        score = snr (0.25+0.75 f_nt)(0.25+0.75 f_psf)(0.40+0.60 f_stm)(0.40+0.60 f_pv)
        tier A: snr >= 5 and n_nights >= ceil(nn/2) and psf_corr >= 0.5
        tier B: snr >= 3.5 and (n_nights >= 2 or psf_corr >= 0.4)
        tier C: otherwise

    Parameters
    ----------
    rows : list of dict
        The candidates (:func:`candidate_metrics`), which take ``score``, ``tier`` and ``rank``.
    nn : int
        Number of nights.
    persistence_map : ndarray or None, default None
        The parameter-persistence map (``paramverify_stim_stitched``), for ``f_pv``.  None sets it to 1.
    stim_half : float, default 1.5
        The STIM and persistence values at which their terms reach half, 1.5 above.

    Returns
    -------
    rows : list of dict
        The candidates, the highest score first.
    """
    nhalf = int(np.ceil(0.5 * nn))
    pv = None
    if persistence_map is not None:
        pv = np.asarray(persistence_map, float)
    for r in rows:
        f_nt = r["n_nights"] / max(nn, 1)
        f_psf = float(np.clip(r["psf_corr"], 0, 1)) if np.isfinite(r["psf_corr"]) else 0.0
        st = r["stim"]
        f_stm = st / (abs(st) + stim_half) if (np.isfinite(st) and st > 0) else 0.0
        f_pv = 1.0
        r["pv"] = float("nan")
        if pv is not None and pv.ndim == 2:
            ny, nx = pv.shape
            ix = int(np.clip(r["x"], 1, nx - 2))
            iy = int(np.clip(r["y"], 1, ny - 2))
            box = pv[iy - 1:iy + 2, ix - 1:ix + 2]
            v = float(np.nanmax(box)) if np.any(np.isfinite(box)) else float("nan")
            r["pv"] = v
            f_pv = v / (abs(v) + stim_half) if (np.isfinite(v) and v > 0) else 0.0
        r["score"] = float(r["snr"] * (0.25 + 0.75 * f_nt) * (0.25 + 0.75 * f_psf)
                           * (0.40 + 0.60 * f_stm) * (0.40 + 0.60 * f_pv))
        psfc = r["psf_corr"] if np.isfinite(r["psf_corr"]) else -np.inf
        if r["snr"] >= 5.0 and r["n_nights"] >= nhalf and psfc >= 0.5:
            r["tier"] = "A"
        elif r["snr"] >= 3.5 and (r["n_nights"] >= 2 or psfc >= 0.4):
            r["tier"] = "B"
        else:
            r["tier"] = "C"
    rows.sort(key=lambda r: -(r["score"] if np.isfinite(r["score"]) else -np.inf))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def find_candidates(image: Optional[np.ndarray], stack: np.ndarray, fwhm: float, pxscale: float, *,
                    snrmap: Optional[np.ndarray] = None, iwa_px: Optional[float] = None,
                    owa_px: Optional[float] = None, snrmin: float = 2.0, nmax: int = 400,
                    known: Sequence[Tuple[float, float]] = (), injected: Sequence[Tuple[float, ...]] = (),
                    persistence_map: Optional[np.ndarray] = None, night_ids: Optional[Sequence[Any]] = None,
                    weights: Optional[Sequence[float]] = None, verify_top: int = 0,
                    verify_kwargs: Optional[Dict[str, Any]] = None, angle_convention: str = "pa",
                    flatten: bool = False, sidelobes=None,
                    rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search an image for point-source candidates without knowing where they are (``near2_candidates``).

    The peaks of the S/N map above ``snrmin``, at least ``max(0.9 fwhm, 2)`` px apart between
    ``iwa_px`` and ``owa_px``, are measured (:func:`candidate_metrics`), scored and ranked
    (:func:`score_candidates`), and the best ``verify_top`` verified
    (:func:`klip_tpe.verify.verify_candidates` with ``found=True``).  Each candidate's local FAP is its
    small-sample t-test at its peak, and its FAP over the field corrects for the search over the
    annulus by random-field theory (:func:`klip_tpe.verify.field_fap`), with the smoothness measured on
    the S/N map (:func:`klip_tpe.verify.field_geometry`).

    Parameters
    ----------
    image : ndarray or None
        The combined (stitched) image: the S/N map's source when ``snrmap`` is not given, and the image
        the candidates' combined S/N, FAP and PSF-likeness and the verify's Path A are measured on.
        None takes the nights' mean for all of these, as the IDL predecessor does.
    stack : ndarray, shape (nn, ny, nx) or (ny, nx)
        The per-night images, not flattened, in chronological order.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    snrmap : ndarray or None, default None
        The S/N map to search.  None makes one, ``snr_map(radprof(image), fwhm)`` with the known
        companions and the pixel mask left out of its noise.
    iwa_px, owa_px : float or None, default None
        Radii of the search, in pixels.  None takes ``1.2 fwhm`` and ``min(nx, ny)/2 - fwhm``.
    snrmin : float, default 2.0
        Detection threshold on the S/N map.
    nmax : int, default 400
        Most peaks taken.
    known, injected : sequence of tuple, default ()
        ``(rho, theta[, contrast])`` lists for the ``KNOWN`` and ``INJ`` flags.  The known companions
        are left out of the noise rings and of the noise of the S/N map made here, 1.5 FWHM around each.
    persistence_map : ndarray or None, default None
        The ``paramverify_stim_stitched`` map, for the score's ``f_pv`` term.
    night_ids : sequence or None, default None
        The nights' names, for the report.
    weights : sequence of float or None, default None
        The nights' weights, for the verify.
    verify_top : int, default 0
        Number of the best candidates to verify.  Each result goes into ``row["verify"]``.
    verify_kwargs : dict or None, default None
        More keywords of the verify, such as ``n_boot``, ``rng`` or ``injected``.
    angle_convention : {"pa", "math"}, default "pa"
        How the angles are read.
    flatten : bool, default False
        Whether the measures subtract each image's radial profile first.
    sidelobes : Sidelobes or None, default None
        The run's sidelobe rule.  The S/N map made here leaves the known companions' ADI arcs out of its
        noise, and each peak's rings leave out its own arc and the known companions'.
    rules : dict or None, default None
        The run's ring rules (:func:`~klip_tpe.metrics.ring_rules`): the S/N map made here leaves the
        pixel mask out of its noise, and every S/N and the verify are the search's.

    Returns
    -------
    rows : list of dict
        The ranked candidates, with the fields :func:`write_candidates` writes and ``meta`` (the search's
        settings and its field), empty when no peak reaches ``snrmin``.
    """
    stack = np.asarray(stack, float)
    if stack.ndim == 2:
        stack = stack[None]
    nn, ny, nx = stack.shape
    given = None if image is None else np.asarray(image, float)
    if snrmap is None:
        if image is None:
            fin = np.isfinite(stack)
            image = np.where(fin, stack, 0.0).sum(axis=0) / np.where(fin.sum(axis=0) > 0, fin.sum(axis=0), np.nan)
        im = np.asarray(image, float)
        arcs = [] if sidelobes is None else [(float(k[0]), float(k[1]), sidelobes.half_width(float(k[0]) / pxscale, fwhm),
                                              float(sidelobes.radial_fwhm)) for k in known]
        # the known companions are no noise, whatever the sidelobe rule (a user's note of 2026-11-22)
        kxy: List[Tuple[float, float]] = []
        if len(known):
            cx0, cy0 = star_center(im.shape)
            kx0, ky0 = source_xy([float(k[0]) for k in known], [float(k[1]) for k in known], pxscale, cx0, cy0,
                                 angle_convention)
            kxy = list(zip(np.atleast_1d(kx0).tolist(), np.atleast_1d(ky0).tolist()))
        snrmap = snr_map(radprof(im) if flatten else im, fwhm, excl_arcs=arcs, pxscale=pxscale,
                         angle_convention=angle_convention, pixel_mask=(rules or {}).get("pixel_mask"),
                         exclude_xy=kxy)
    snrmap = np.asarray(snrmap, float)
    iwa = 1.2 * fwhm if iwa_px is None else float(iwa_px)
    owa = (min(nx, ny) / 2.0 - fwhm) if owa_px is None else float(owa_px)
    sep = max(0.9 * fwhm, 2.0)
    px, py, ps = detect_peaks(snrmap, snrmin, sep, iwa, owa, nmax)
    if px.size == 0:
        return []
    field = field_geometry(snrmap, iwa, owa, fwhm)
    rows = candidate_metrics(px, py, ps, stack, fwhm, pxscale, iwa, owa, known=known, injected=injected,
                             angle_convention=angle_convention, flatten=flatten, image=given, sidelobes=sidelobes,
                             rules=rules, field=field)
    rows = score_candidates(rows, nn, persistence_map)
    meta = {"nights": list(range(nn)) if night_ids is None else [str(v) for v in night_ids],
            "snrmin": snrmin, "sep": sep, "iwa": iwa, "owa": owa,
            "nresel": field["n_resel"], "fwhm": fwhm, "pxscale": pxscale,
            "comb": "image" if given is not None else "nights", "field": field,
            "look_elsewhere": "n_resel" if legacy_position(rules) else "random field",
            "search_px": search_px(fwhm, True, rules)}
    for r in rows:
        r["meta"] = meta
    if verify_top and verify_top > 0:
        top = rows[: min(len(rows), int(verify_top))]
        kw = dict(known=known, weights=weights, angle_convention=angle_convention, sidelobes=sidelobes, rules=rules)
        vk = dict(verify_kwargs or {})
        if given is not None and (vk.get("injected") is None or vk.get("combined_inj") is not None):
            kw.update(combined=given, path_a="the source image")
        kw.update(vk)
        kw.setdefault("found", True)
        kw.setdefault("field", field)
        vrows = verify_candidates(stack, meta["nights"], [(r["rho"], r["theta"]) for r in top], fwhm, pxscale, **kw)
        vc = [v for v in vrows if v.get("kind") == "candidate"]
        for r, v in zip(top, vc):
            r["verify"] = v
    return rows


def write_candidates(path: str, rows: Sequence[Dict[str, Any]], source: str = "") -> str:
    """Write ``candidates.txt``, a ``#`` header and one fixed-width line per candidate.

    The columns are ``rank x y rho" PA S/N Nnt pn_md STIM sub1 sub2 PSF FAP score tier flag``, in the
    formats ``I4 I5 I5 F7.3 F7.1 F7.2 I4 F7.2 F7.2 F6.2 F6.2 F6.2 E10.2 F8.2 A A`` of the IDL
    predecessor (candidates 369-393).  FAP is over the searched field, and the header says how it was
    corrected.

    Parameters
    ----------
    path : str
        File to write.
    rows : sequence of dict
        The ranked candidates (:func:`find_candidates`).
    source : str, default ""
        Name of the searched image, for the header.

    Returns
    -------
    text : str
        The text written.
    """
    meta = rows[0].get("meta", {}) if rows else {}
    L = [f"# KLIP-TPE automatic candidate search -- {_dt.datetime.now().isoformat(timespec='seconds')}"]
    if source:
        L.append(f"# source image : {source}")
    if meta:
        L.append(f"# detection    : S/N >= {meta['snrmin']:.1f}, >= {meta['sep']:.1f} px apart, "
                 f"IWA {meta['iwa']:.1f} - OWA {meta['owa']:.1f} px")
        if meta.get("look_elsewhere", "n_resel") == "random field" and meta.get("field"):
            fld = meta["field"]
            L.append(f"# nights       : {','.join(str(v) for v in meta['nights'])}   field {float(fld['area_px']):.0f} px, "
                     f"perimeter {float(fld.get('perimeter_px', float('nan'))):.0f} px, smoothness FWHM {float(fld['fwhm_eff_px']):.2f} px ({fld.get('fwhm_eff_from', 'map')}), "
                     f"N_resel = {meta['nresel']:.1f}")
        else:
            L.append(f"# nights       : {','.join(str(v) for v in meta['nights'])}   look-elsewhere N_resel = {meta['nresel']:.1f}")
        if meta.get("comb") == "image":
            L.append("# measured on  : S/N, FAP and PSF on the source image, Nnt, pn_md, STIM, sub1 and sub2 on the nights")
    L += ["# score = S/N * persistence(nights) * PSF-likeness * consistency(STIM) * persistence(params)",
          "# tier  : A=strong (S/N>=5, >=half nights, PSF-like)  B=marginal  C=likely speckle",
          "# S/N   : matched-filter Mawet (2014) small-sample S/N (Student-t, df=n_ap-2)",
          ("# FAP   : over the searched field, random-field theory (Worsley et al. 1996) on p, the small-sample-t "
           "FAP at the peak (verify.field_fap)" if meta.get("look_elsewhere") == "random field" else
           "# FAP   : 1 - (1 - p)^N_resel, p the per-resel small-sample-t false-alarm prob (look-elsewhere corrected)"),
          ""]
    rho_hdr = 'rho"'
    L.append("  " + f"{'rank':>4}{'x':>5}{'y':>5}{rho_hdr:>7}{'PA':>7}{'S/N':>7}{'Nnt':>4}{'pn_md':>7}{'STIM':>7}"
             f"{'sub1':>6}{'sub2':>6}{'PSF':>6}{'FAP':>10}{'score':>8}  tier  flag")

    def f(v, w, d):
        return f"{v:{w}.{d}f}" if np.isfinite(v) else f"{'nan':>{w}}"

    for r in rows:
        fap = f"{r['fap']:10.2E}" if np.isfinite(r["fap"]) else f"{'nan':>10}"
        L.append("  " + f"{r['rank']:4d}{r['x']:5d}{r['y']:5d}{f(r['rho'],7,3)}{f(r['theta'],7,1)}{f(r['snr'],7,2)}"
                 f"{r['n_nights']:4d}{f(r['pn_med'],7,2)}{f(r['stim'],7,2)}{f(r['sub1'],6,2)}{f(r['sub2'],6,2)}"
                 f"{f(r['psf_corr'],6,2)}{fap}{f(r['score'],8,2)}   {r['tier']}   {r['flag']}")
    text = "\n".join(L) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text
