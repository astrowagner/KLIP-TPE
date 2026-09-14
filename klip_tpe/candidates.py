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
  249-272, 322-330) is a run-time optimisation of the IDL "running" pass; the
  port always recomputes (metrics are deterministic, so results are identical).
* The S/N map is built with :func:`klip_tpe.products.snr_map` on the
  ``radprof``-flattened image when not supplied (the IDL reads the pre-computed
  ``klip_stitched_snr.fits`` from ``near2m_snrmap``).
* The detection-map PDF and ``README.txt`` are not produced.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import radprof, source_xy, star_center, nanmedian_even
from .products import snr_map
from .verify import mf_convolve, snr_fap, verify_candidates

__all__ = ["detect_peaks", "psf_correlation", "candidate_metrics", "score_candidates",
           "find_candidates", "write_candidates"]


# ----------------------------------------------------------------------------
# step 1 -- peak detection  (near2c_detect, candidates 32-51)
# ----------------------------------------------------------------------------
def detect_peaks(snrmap: np.ndarray, snrmin: float, sep: float, iwa: float, owa: float,
                 nmax: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative local-maximum extraction: mask non-finite pixels and ``r < iwa``,
    ``r > owa``; repeatedly take the global maximum (stop below ``snrmin`` or
    after ``nmax`` peaks) and suppress the disc of radius ``sep`` around it.
    Returns integer ``(px, py, snr)`` arrays."""
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
    """Normalised cross-correlation in ``[-1, 1]`` of the median-subtracted stamp
    (half-box ``max(round(1.5 fwhm), 3)``) with a zero-mean Gaussian of
    ``sigma = fwhm / 2.3548`` centred on the peak, over finite pixels (>= 5)."""
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
                      injected: Sequence[Tuple[float, ...]] = (), excl_fwhm: float = 1.5, srad: float = 1.5,
                      angle_convention: str = "pa") -> List[Dict[str, Any]]:
    """Per-peak metrics from the per-night stack ``(nn, ny, nx)`` (unflattened).

    Precomputed once (227-247): ``slices[j] = radprof(stack[j])``, ``comb =
    radprof(nanmean)`` (unweighted), even / odd nanmeans ``s1, s2``, and their
    matched-filter maps.  Constants: ``exr = 1.5 fwhm``, ``srad = 1.5`` px,
    aperture ``ap = max(0.5 fwhm, 1.5)``, ``n_resel = pi (owa^2 - iwa^2) / fwhm^2``.
    Per peak: per-night S/N ``pn`` (275), ``n_nights = count(pn >= 2)``,
    ``pn_min/med/max``; aperture STIM ``mean(av)/std(av)`` over >= 2 finite nights
    (279-284); ``sub1/sub2`` (286-287); ``psf_corr`` (289); combined S/N replaces
    the map value when finite (293); look-elsewhere FAP ``min(1 - (1 - p)^n_resel,
    1)`` (294); flags ``KNOWN`` / ``INJ`` within ``0.8 fwhm`` (296-306)."""
    stack = np.asarray(stack, float)
    if stack.ndim == 2:
        stack = stack[None]
    nn, ny, nx = stack.shape
    cx, cy = star_center((ny, nx))
    exr = excl_fwhm * fwhm
    nresel = max(np.pi * (owa ** 2 - iwa ** 2) / fwhm ** 2, 1.0)
    ap = max(0.5 * fwhm, 1.5)
    slices = np.stack([radprof(stack[j]) for j in range(nn)])
    fin = np.isfinite(stack)

    def _nanmean(sel):
        z = np.where(fin[sel], stack[sel], 0.0)
        return z.sum(axis=0) / np.maximum(fin[sel].sum(axis=0), 1)

    comb = radprof(_nanmean(slice(None))) if nn > 1 else slices[0]
    if nn >= 2:
        i1, i2 = np.arange(nn) % 2 == 0, np.arange(nn) % 2 == 1
        s1 = radprof(_nanmean(i1)) if i1.any() else comb
        s2 = radprof(_nanmean(i2)) if i2.any() else comb
    else:
        s1 = s2 = comb
    mf_comb, mf_s1, mf_s2 = (mf_convolve(a, fwhm) for a in (comb, s1, s2))
    mf_slices = [mf_convolve(slices[j], fwhm) for j in range(nn)]
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
        kw = dict(exclude_xy=kxy, angle_convention=angle_convention)
        pn = np.array([snr_fap(slices[j], *args, mfmap=mf_slices[j], **kw)["snr"] for j in range(nn)])
        if not np.any(np.isfinite(pn)):
            pn[:] = 0.0
        n_nights = int(np.sum(pn >= 2.0))
        m = (xx - px[c]) ** 2 + (yy - py[c]) ** 2 <= ap * ap
        av = np.array([np.nanmean(slices[j][m]) if np.any(np.isfinite(slices[j][m])) else np.nan
                       for j in range(nn)])
        gav = av[np.isfinite(av)]
        stim = float(np.mean(gav) / np.std(gav, ddof=1)) if (gav.size >= 2 and np.std(gav, ddof=1) > 0) else float("nan")
        sub1 = snr_fap(s1, *args, mfmap=mf_s1, **kw)["snr"]
        sub2 = snr_fap(s2, *args, mfmap=mf_s2, **kw)["snr"] if nn >= 2 else float("nan")
        psfc = psf_correlation(comb, int(px[c]), int(py[c]), fwhm)
        cf = snr_fap(comb, *args, mfmap=mf_comb, **kw)
        csnr = float(cf["snr"]) if np.isfinite(cf["snr"]) else float(snr0[c])
        fap = float(min(1.0 - (1.0 - cf["fap"]) ** nresel, 1.0)) if np.isfinite(cf["fap"]) else float("nan")
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
    return rows


# ----------------------------------------------------------------------------
# step 3 -- score & tier  (candidates 336-366)
# ----------------------------------------------------------------------------
def score_candidates(rows: List[Dict[str, Any]], nn: int, persistence_map: Optional[np.ndarray] = None,
                     stim_half: float = 1.5) -> List[Dict[str, Any]]:
    """Add ``score``, ``tier`` and ``rank`` in place and return the rows sorted by
    descending score::

        f_nt  = n_nights / nn
        f_psf = clip(psf_corr, 0, 1)             (NaN -> 0)
        f_stm = stim / (|stim| + 1.5)            (NaN or stim <= 0 -> 0)
        f_pv  = pv / (|pv| + 1.5)                (pv = max of the 3x3 box of the
                                                  param-persistence map; 1 if absent)
        score = snr (0.25+0.75 f_nt)(0.25+0.75 f_psf)(0.40+0.60 f_stm)(0.40+0.60 f_pv)
        tier A: snr >= 5 and n_nights >= ceil(nn/2) and psf_corr >= 0.5
        tier B: snr >= 3.5 and (n_nights >= 2 or psf_corr >= 0.4)
        tier C: otherwise
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
                    verify_kwargs: Optional[Dict[str, Any]] = None, angle_convention: str = "pa"
                    ) -> List[Dict[str, Any]]:
    """Blind candidate search (``near2_candidates``).

    Parameters
    ----------
    image
        combined (stitched) image used to build the S/N map when ``snrmap`` is
        not given (``snr_map(radprof(image), fwhm)``); may be ``None`` then.
    stack
        ``(nn, ny, nx)`` per-night images (unflattened), chronological order.
    iwa_px, owa_px
        search radii; defaults ``1.2 fwhm`` and ``min(nx, ny)/2 - fwhm`` (214-215).
    snrmin, nmax
        detection threshold (2.0) and peak cap (400); suppression radius
        ``max(0.9 fwhm, 2)`` px (216).
    known, injected
        ``(rho, theta[, contrast])`` lists for the ``KNOWN`` / ``INJ`` flags (and
        ``known`` is excluded from the noise rings).
    persistence_map
        ``paramverify_stim_stitched`` map for the ``f_pv`` score term.
    verify_top
        run :func:`klip_tpe.verify.verify_candidates` on the top-N ranked peaks
        (IDL ``npage=25`` unless ``/quick``); ``verify_kwargs`` are forwarded
        (e.g. ``n_boot``, ``rng``, ``injected=(inj_stack, sources)``).  The
        result is attached as ``row['verify']``.

    Returns the ranked rows (dicts) with the fields written by
    :func:`write_candidates`; an empty list when no peak reaches ``snrmin``.
    """
    stack = np.asarray(stack, float)
    if stack.ndim == 2:
        stack = stack[None]
    nn, ny, nx = stack.shape
    if snrmap is None:
        if image is None:
            fin = np.isfinite(stack)
            image = np.where(fin, stack, 0.0).sum(axis=0) / np.where(fin.sum(axis=0) > 0, fin.sum(axis=0), np.nan)
        snrmap = snr_map(radprof(np.asarray(image, float)), fwhm)
    snrmap = np.asarray(snrmap, float)
    iwa = 1.2 * fwhm if iwa_px is None else float(iwa_px)
    owa = (min(nx, ny) / 2.0 - fwhm) if owa_px is None else float(owa_px)
    sep = max(0.9 * fwhm, 2.0)
    px, py, ps = detect_peaks(snrmap, snrmin, sep, iwa, owa, nmax)
    if px.size == 0:
        return []
    rows = candidate_metrics(px, py, ps, stack, fwhm, pxscale, iwa, owa, known=known, injected=injected,
                             angle_convention=angle_convention)
    rows = score_candidates(rows, nn, persistence_map)
    meta = {"nights": list(range(nn)) if night_ids is None else [str(v) for v in night_ids],
            "snrmin": snrmin, "sep": sep, "iwa": iwa, "owa": owa,
            "nresel": max(np.pi * (owa ** 2 - iwa ** 2) / fwhm ** 2, 1.0), "fwhm": fwhm, "pxscale": pxscale}
    for r in rows:
        r["meta"] = meta
    if verify_top and verify_top > 0:
        top = rows[: min(len(rows), int(verify_top))]
        kw = dict(known=known, weights=weights, angle_convention=angle_convention)
        kw.update(verify_kwargs or {})
        vrows = verify_candidates(stack, meta["nights"], [(r["rho"], r["theta"]) for r in top], fwhm, pxscale, **kw)
        vc = [v for v in vrows if v.get("kind") == "candidate"]
        for r, v in zip(top, vc):
            r["verify"] = v
    return rows


def write_candidates(path: str, rows: Sequence[Dict[str, Any]], source: str = "") -> str:
    """Write ``candidates.txt`` (candidates 369-393): ``#`` header then the
    fixed-width columns ``rank x y rho" PA S/N Nnt pn_md STIM sub1 sub2 PSF FAP
    score tier flag`` (formats I4 I5 I5 F7.3 F7.1 F7.2 I4 F7.2 F7.2 F6.2 F6.2 F6.2
    E10.2 F8.2 A A).  Returns the text."""
    meta = rows[0].get("meta", {}) if rows else {}
    L = [f"# KLIP-TPE automatic candidate search -- {_dt.datetime.now().isoformat(timespec='seconds')}"]
    if source:
        L.append(f"# source image : {source}")
    if meta:
        L.append(f"# detection    : S/N >= {meta['snrmin']:.1f}, >= {meta['sep']:.1f} px apart, "
                 f"IWA {meta['iwa']:.1f} - OWA {meta['owa']:.1f} px")
        L.append(f"# nights       : {','.join(str(v) for v in meta['nights'])}   look-elsewhere N_resel = {meta['nresel']:.1f}")
    L += ["# score = S/N * persistence(nights) * PSF-likeness * consistency(STIM) * persistence(params)",
          "# tier  : A=strong (S/N>=5, >=half nights, PSF-like)  B=marginal  C=likely speckle",
          "# S/N   : matched-filter Mawet (2014) small-sample S/N (Student-t, df=n_ap-2)",
          "# FAP   : per-resel small-sample-t false-alarm prob, look-elsewhere corrected (x N_resel)", ""]
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
