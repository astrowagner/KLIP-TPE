"""Parameter-ensemble persistence and reliability, the IDL predecessor's ``near2_param_verify.pro``.

It also ports the configuration selection of the IDL predecessor's main loop (opt 7795-7850).  The
search's top-N most *diverse* configurations of one annulus are each reduced again, clean and with ONE
shared fixed injection set.  Their matched-filter S/N maps give the param-STIM (the mean across the
configurations over their unbiased standard deviation), the detection fraction (clean S/N > 3), the
recovery fraction (injected S/N > 3), a throughput-normalized meta-combine with its S/N map, a null and
true STIM calibration table, and a throughput-calibrated 5-sigma contrast curve of the meta-combine.
The products are written as FITS and text into an output directory, tagged ``_NN``, and
:func:`stitch_param_maps` merges the annuli (the IDL predecessor's ``near2m_pvstitch``).

The reduction comes in as ``reduce_fn``, so the module does not depend on the pipeline.  The
conventions are those of :mod:`klip_tpe.verify`.

It deviates from the IDL predecessor on purpose:

* ``near2m_paramverify`` (147) and ``near2m_pvstitch`` (381) use ``cx = nx/2``, and the port uses
  ``(n-1)/2`` everywhere (brief 8.4, spec C.6).
* ``near2m_snrmap`` is :func:`klip_tpe.products.snr_map`, band statistics per radius with a MAD floor.
  Its bookkeeping of the ring indices differs slightly, and the quantity is the same.
* The PDF and PNG products and ``README.txt`` are not produced.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import (Source, mawet_peak_snr, nanmedian_even, radprof, rule_kwargs, source_xy, star_center,
                      verify_idl)
from .optimizers import History
from .positions import PositionSampler
from .products import snr_map
from .space import SearchSpace

__all__ = ["select_configs", "fixed_injection_set", "pv_meanstd", "radial_rms", "param_stim",
           "param_verify_cubes", "param_verify", "stitch_param_maps", "PV_KINDS"]

#: the maps the stage writes and stitches
PV_KINDS = ("stim", "detfrac", "recovery", "combine", "combsnr")
#: the phases :func:`select_configs` leaves out of its pool without ``n_init``
_NON_TPE_PHASES = ("seed", "warmup", "random", "grid", "calib", "calibration")


# ----------------------------------------------------------------------------
# C.4.1  config selection  (opt 7795-7817)
# ----------------------------------------------------------------------------
def select_configs(history: History, space: SearchSpace, n_pv: int = 20, pv_divmin: float = 0.05,
                   n_init: Optional[int] = None, min_score: float = -9000.0) -> List[int]:
    """Return the ``n_pv`` best configurations that keep a diversity floor from each other.

    The pool is the TPE-phase evaluations with a finite score above ``min_score``, or every such evaluation
    when there is none (opt 7796-7797).  Best first, a configuration is kept when its mean normalized L1
    distance to *every* configuration kept, over the base dimensions, is at least ``pv_divmin`` (opt
    7800-7810).  The distance is ``d = (1/ndim_base) sum_d |x_d - x'_d| / (hi_d - lo_d)``, and the base
    dimensions are the reduction dimensions of no partition.

    Parameters
    ----------
    history : History
        The annulus' search history.
    space : SearchSpace
        Its space.
    n_pv : int, default 20
        Most configurations kept.
    pv_divmin : float, default 0.05
        The diversity floor.
    n_init : int or None, default None
        Index of the first TPE-phase evaluation.  None takes the evaluations whose ``phase`` flag is none of
        the warm-up, random, grid, seed and calibration phases.
    min_score : float, default -9000.0
        The score an evaluation must exceed.

    Returns
    -------
    indices : list of int
        The evaluations kept, best first, possibly fewer than 2, which callers then skip.
    """
    y = history.y
    n = len(history)
    idx = np.arange(n)
    ok = np.isfinite(y) & (y > min_score)
    if n_init is not None:
        pool = idx[ok & (idx >= int(n_init))]
    else:
        tpe = np.array([str(f.get("phase", "tpe")) not in _NON_TPE_PHASES for f in history.flags], bool) \
            if history.flags else np.ones(n, bool)
        pool = idx[ok & tpe]
    if pool.size < 1:
        pool = idx[ok]
    if pool.size < 1:
        return []
    base = [i for i in space.reduction_dims if space.params[i].partition is None]
    if not base:
        base = list(space.reduction_dims) or list(range(space.ndim))
    base = np.asarray(base, int)
    rng_d = np.maximum(space.hi[base] - space.lo[base], 1e-6)
    order = pool[np.argsort(-y[pool], kind="stable")]
    kept: List[int] = []
    for e in order:
        if len(kept) >= n_pv:
            break
        xe = history.X[e, base]
        if all(np.mean(np.abs(xe - history.X[k, base]) / rng_d) >= pv_divmin for k in kept):
            kept.append(int(e))
    return kept


def fixed_injection_set(nsrc: int, rlo_as: float, rhi_as: float, fwhm: float, pxscale: float,
                        rng: Optional[np.random.Generator] = None, contrast: float = 0.0,
                        known: Sequence[Tuple[float, float]] = (), sampler=None) -> List[Source]:
    """Return the shared fixed injection set, as the IDL predecessor's ``near2m_randpos`` draws it.

    It is a radial ladder ``rlo + (i + 0.5)(rhi - rlo)/n`` with one random azimuth anchor and ``360/n``
    spacing, away from the ``known`` sources (:class:`klip_tpe.positions.PositionSampler`, strategy
    ``"spread"``, opt 4141+).  The run's own ``sampler`` makes every stage inject with the same geometry,
    as the :class:`~klip_tpe.runner.Runner` passes it.

    Parameters
    ----------
    nsrc : int
        Number of sources.
    rlo_as : float
        Inner radius, in arcseconds.
    rhi_as : float
        Outer radius, in arcseconds.
    fwhm : float
        FWHM, in pixels.
    pxscale : float
        Pixel scale, in arcseconds per pixel.
    rng : numpy.random.Generator or None, default None
        The generator, None for one seeded with 0.
    contrast : float, default 0.0
        The sources' contrast.
    known : sequence of tuple, default ()
        Real companions to keep away from, ``(rho_arcsec, pa_deg)``, when no sampler is given.
    sampler : PositionSampler or None, default None
        The sampler, None for a ``"spread"`` one of this FWHM.

    Returns
    -------
    sources : list of Source
        The sources.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ps = sampler if sampler is not None else PositionSampler(fwhm_as=fwhm * pxscale, strategy="spread", known=list(known))
    return ps.sample(int(nsrc), float(rlo_as), float(rhi_as), rng, contrast)


# ----------------------------------------------------------------------------
# C.4.2  primitives
# ----------------------------------------------------------------------------
def pv_meanstd(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean, the unbiased standard deviation and the count over a cube's configuration axis.

    Each pixel's statistics skip NaN, and the standard deviation divides by N - 1 in two passes, as the
    IDL predecessor's ``near2m_pv_meanstd`` does (param_verify 44-56).

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The configurations' maps.

    Returns
    -------
    mean : ndarray, shape (ny, nx)
        The mean, NaN where fewer than 2 configurations are finite.
    sd : ndarray, shape (ny, nx)
        The standard deviation, NaN there too.
    count : ndarray, shape (ny, nx)
        The number of finite configurations.
    """
    cube = np.asarray(cube, float)
    fin = np.isfinite(cube)
    z = np.where(fin, cube, 0.0)
    cnt = fin.sum(axis=0).astype(float)
    mn = z.sum(axis=0) / np.maximum(cnt, 1)
    ss = (((z - mn[None]) * fin) ** 2).sum(axis=0)
    sd = np.sqrt(np.maximum(ss / np.maximum(cnt - 1.0, 1.0), 0.0))
    bad = cnt < 2
    mn[bad] = np.nan
    sd[bad] = np.nan
    return mn, sd, cnt


def param_stim(sn_cube: np.ndarray) -> np.ndarray:
    """Return the param-STIM of a cube of S/N maps, ``mn / max(sd, 0.3 median(sd > 0), 1e-6)``.

    It follows the IDL predecessor's param_verify (157-163 and 176-178).

    Parameters
    ----------
    sn_cube : ndarray, shape (n, ny, nx)
        The configurations' S/N maps.

    Returns
    -------
    stim : ndarray, shape (ny, nx)
        The map, NaN where fewer than 2 configurations are finite.
    """
    mn, sd, cnt = pv_meanstd(sn_cube)
    g = np.isfinite(sd) & (sd > 0)
    sfloor = float(np.median(sd[g])) if g.any() else 1.0
    stim = mn / np.maximum(np.maximum(sd, 0.3 * sfloor), 1e-6)
    stim[cnt < 2] = np.nan
    return stim


def radial_rms(img: np.ndarray, reval: Sequence[float]) -> np.ndarray:
    """Return the sample standard deviation of an image in the annulus ``[r - 1, r + 1)`` at each radius.

    It is the IDL predecessor's ``near2m_pv_radrms`` (param_verify 62-73).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, centered on the star.
    reval : sequence of float
        The radii, in pixels.

    Returns
    -------
    rms : ndarray, shape (len(reval),)
        The standard deviation at each radius, NaN with fewer than 3 finite pixels.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    out = np.full(len(reval), np.nan)
    for i, r in enumerate(reval):
        w = (rr >= r - 1.0) & (rr < r + 1.0) & np.isfinite(img)
        if w.sum() >= 3:
            out[i] = float(np.std(img[w], ddof=1))
    return out


def _box_max(m: np.ndarray, x: float, y: float) -> float:
    """Return the largest finite value of the 3 x 3 box around a position, the box kept off the edge.

    Parameters
    ----------
    m : ndarray, shape (ny, nx)
        The map.
    x : float
        The position's column, in pixels.
    y : float
        Its row, in pixels.

    Returns
    -------
    peak : float
        The maximum, NaN when the box holds no finite value.
    """
    ny, nx = m.shape
    ix = int(np.clip(round(x), 1, nx - 2))
    iy = int(np.clip(round(y), 1, ny - 2))
    box = m[iy - 1:iy + 2, ix - 1:ix + 2]
    return float(np.nanmax(box)) if np.any(np.isfinite(box)) else float("nan")


def _percentile_idl(v: np.ndarray, p: float) -> float:
    """Return a percentile as the IDL predecessor takes it, ``v[sort(v)[floor(p (n - 1))]]``.

    Parameters
    ----------
    v : ndarray
        The values.
    p : float
        The fraction, from 0 to 1.

    Returns
    -------
    value : float
        The value at that rank (param_verify 242-244).
    """
    s = np.sort(v)
    return float(s[int(np.floor(p * (s.size - 1)))])


# ----------------------------------------------------------------------------
# C.4.2  near2m_paramverify on ready-made cubes
# ----------------------------------------------------------------------------
def param_verify_cubes(cclean: np.ndarray, cinj: np.ndarray, wts: Sequence[float],
                       sources: Sequence[Source], ccal: float, pxscale: float, fwhm: float,
                       rlo: float, rhi: float, out_dir: Optional[str] = None, tag: str = "01",
                       snr_thr: float = 3.0, angle_convention: str = "pa",
                       flatten: bool = False, write_fits: bool = True,
                       inner_fwhm: float = 1.0, rules: Optional[Dict[str, Any]] = None,
                       known: Sequence[Tuple[float, float]] = ()) -> Dict[str, Any]:
    """Run the IDL predecessor's ``near2m_paramverify`` on an ensemble's clean and injected cubes.

    The steps, numbered by the lines of the IDL predecessor's param_verify (135-249), are:

    1. the S/N map of each configuration's clean image, ``sncl``, and of its injected image with the fixed
       sources left out of the noise, ``snij`` (153-154),
    2. the param-STIM of ``sncl`` (157-163), ``detfrac``, the fraction of configurations whose clean S/N
       exceeds ``snr_thr`` (166-167), and ``recovery``, the same on ``snij`` (172),
    3. ``true``, the 3 x 3 maximum of the injected ensemble's STIM at each source (176-183), and ``null``,
       the clean STIM in ``rlo <= r <= rhi`` farther than ``1.5 fwhm`` from every source, or five zeros
       with fewer than 5 pixels (185-191),
    4. the meta-combine (194-217), with ``K_i`` the median over the sources of the 3 x 3 peak of
       ``snij_i``, ``Kmed = median(K)``, ``we_i = w_i max(K_i, 0)`` and
       ``comb = sum we_i (Kmed/K_i) cclean_i / sum we_i``, ``combj`` the same of the injected cube and
       ``combsnr`` the S/N map of ``comb``,
    5. the contrast curve (221-229), with ``ps`` the sources' S/N in ``combj``
       (:func:`~klip_tpe.metrics.mawet_peak_snr`), ``reval = rlo + (rhi - rlo)(i + 0.5)/12``,
       ``K_s = ps_s sigma(r_s) / ccal``, ``Kc = median(K > 0)`` and ``c5 = 5 sigma(r) / Kc``.

    ``rules`` (:func:`~klip_tpe.metrics.ring_rules`) make ``ps`` the search's S/N, with its matched filter,
    despike, band and pixel mask, and the maps leave the mask out of their noise.  The ``known``
    companions stay out of ``ps``' rings, as out of the search's.

    Parameters
    ----------
    cclean : ndarray, shape (n, ny, nx)
        The configurations' clean images.
    cinj : ndarray, shape (n, ny, nx)
        Their images with the fixed sources injected.
    wts : sequence of float
        Each configuration's weight, ones when there are not ``n``.
    sources : sequence of Source
        The fixed sources.
    ccal : float
        Their contrast, the annulus' calibrated one.
    pxscale : float
        Pixel scale, in arcseconds per pixel.
    fwhm : float
        FWHM, in pixels.
    rlo : float
        The annulus' inner radius, in pixels.
    rhi : float
        Its outer radius, in pixels.
    out_dir : str or None, default None
        Directory of the products, None to write none.
    tag : str, default "01"
        The products' tag, the annulus' number.
    snr_thr : float, default 3.0
        The S/N of a detection, for ``detfrac`` and ``recovery``.
    angle_convention : str, default "pa"
        How the sources' angles are measured (:func:`~klip_tpe.metrics.source_xy`).
    flatten : bool, default False
        Whether to subtract each image's radial profile (:func:`~klip_tpe.metrics.radprof`) before its maps.
    write_fits : bool, default True
        Whether to write the maps' FITS files into ``out_dir``.
    inner_fwhm : float, default 1.0
        The metric's inner guard for ``ps``, in FWHM, the run's own.
    rules : dict or None, default None
        The run's noise-ring rules, None for the metric's defaults.
    known : sequence of tuple, default ()
        Real companions, ``(rho_arcsec, pa_deg)``.

    Returns
    -------
    result : dict
        The maps (``stim``, ``detfrac``, ``recovery``, ``combine``, ``combsnr``, ``stim_inj``,
        ``combine_inj``, ``snr_clean``, ``snr_inj``), ``K``, ``Kmed``, ``Kc``, ``Kcal``, ``curve``,
        ``calib``, ``sources``, ``weights``, ``tag`` and ``files``, the products written.

    Raises
    ------
    ValueError
        If there are fewer than 2 configurations.

    Notes
    -----
    With ``out_dir`` it writes ``paramverify_<kind>_NN.fits`` for each of :data:`PV_KINDS` and
    ``paramverify_calib_NN.txt``.  Under the measures of before 2026-11-01
    (:data:`~klip_tpe.metrics.VERIFY_IDL`) ``ps`` takes the metric's defaults and the companions stay in its
    rings, as they did then.
    """
    cclean = np.asarray(cclean, float)
    cinj = np.asarray(cinj, float)
    nn, ny, nx = cclean.shape
    if nn < 2:
        raise ValueError("param_verify needs at least 2 configurations")
    cx, cy = star_center((ny, nx))
    rho = np.array([s.rho for s in sources], float)
    th = np.array([s.theta for s in sources], float)
    sxp, syp = source_xy(rho, th, pxscale, cx, cy, angle_convention)
    ex_xy = list(zip(sxp, syp))
    nf = rho.size

    _flat = radprof if flatten else (lambda a: np.asarray(a, float))
    # the run's ring rules (Kevin Wagner, 2026-11-01): the maps' noise leaves the pixel mask
    # out, and the curve's S/N is the search's
    mask = (rules or {}).get("pixel_mask")
    sncl = np.stack([snr_map(_flat(cclean[i]), fwhm, pixel_mask=mask) for i in range(nn)])
    snij = np.stack([snr_map(_flat(cinj[i]), fwhm, exclude_xy=ex_xy, pixel_mask=mask) for i in range(nn)])

    stim = param_stim(sncl)
    fin = np.isfinite(sncl)
    detfrac = ((sncl > snr_thr) & fin).sum(axis=0) / np.maximum(fin.sum(axis=0), 1)
    finj = np.isfinite(snij)
    recovery = ((snij > snr_thr) & finj).sum(axis=0) / np.maximum(finj.sum(axis=0), 1)

    stimj = param_stim(snij)
    truev = np.array([_box_max(stimj, sxp[s], syp[s]) for s in range(nf)])
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    inring = (rr >= rlo) & (rr <= rhi)
    for s in range(nf):
        inring &= np.hypot(xx - sxp[s], yy - syp[s]) > 1.5 * fwhm
    wnul = inring & np.isfinite(stim)
    nullv = stim[wnul] if wnul.sum() >= 5 else np.zeros(5)

    w = np.maximum(np.asarray(wts, float), 0.0) if len(wts) == nn else np.ones(nn)
    if w.sum() <= 0:
        w = np.ones(nn)
    K = np.array([nanmedian_even([_box_max(snij[i], sxp[s], syp[s]) for s in range(nf)]) for i in range(nn)])
    Kmed = nanmedian_even(K)
    if not np.isfinite(Kmed) or Kmed <= 0:
        Kmed = 1.0
    we = w * np.where(np.isfinite(K), np.maximum(K, 0.0), 0.0)
    if we.sum() <= 0:
        we = np.ones(nn)
    scale = np.where(np.isfinite(K) & (K > 0), Kmed / np.where(K > 0, K, 1.0), 1.0)
    # IDL sums raw values (NaN propagates); keep NaN-aware to be robust to masked pixels
    def _wsum(cube):
        num = np.zeros((ny, nx)); den = np.zeros((ny, nx))
        for i in range(nn):
            f = np.isfinite(cube[i])
            num += np.where(f, cube[i], 0.0) * we[i] * scale[i]
            den += f * we[i]
        return num / np.where(den > 0, den, np.nan)
    comb = _wsum(cclean)
    combj = _wsum(cinj)
    combsnr = snr_map(_flat(comb), fwhm, pixel_mask=mask)

    rkw = rule_kwargs(rules, float(np.median(rho)) if rho.size else 0.0, (ny, nx))
    rkw["inner_fwhm"] = inner_fwhm
    # a run that keeps the measures of before 2026-11-01 left the companions in these rings
    ps = mawet_peak_snr(_flat(combj), rho, th, pxscale, fwhm, angle_convention=angle_convention,
                        known=[] if verify_idl(rules) else list(known), **rkw)
    reval = rlo + (rhi - rlo) * (np.arange(12) + 0.5) / 12.0
    rp_comb = _flat(comb)
    sig_r = radial_rms(rp_comb, reval)
    sig_s = radial_rms(rp_comb, rho / pxscale)
    Kcal = np.where(np.isfinite(ps) & (sig_s > 0), ps * sig_s / ccal, np.nan)
    gK = np.isfinite(Kcal) & (Kcal > 0)
    Kc = nanmedian_even(Kcal[gK]) if gK.any() else 1.0
    c5 = 5.0 * sig_r / max(Kc, 1e-30)

    calib = {"n_configs": nn, "detfrac_thr": snr_thr, "null_median": float(np.median(nullv)),
             "null_p90": _percentile_idl(nullv, 0.90), "null_p95": _percentile_idl(nullv, 0.95),
             "null_p99": _percentile_idl(nullv, 0.99), "true_median": nanmedian_even(truev),
             "true_min": float(np.nanmin(truev)) if np.any(np.isfinite(truev)) else float("nan"),
             "true": truev, "n_null": int(wnul.sum())}
    out = {"tag": tag, "stim": stim, "detfrac": detfrac, "recovery": recovery, "combine": comb, "combsnr": combsnr,
           "stim_inj": stimj, "combine_inj": combj, "snr_clean": sncl, "snr_inj": snij, "K": K, "Kmed": Kmed,
           "Kc": Kc, "Kcal": Kcal, "curve": {"r_px": reval, "r_as": reval * pxscale, "c5": c5, "sigma": sig_r},
           "calib": calib, "sources": list(sources), "weights": w, "files": {}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if write_fits:
            for kind in PV_KINDS:
                p = os.path.join(out_dir, f"paramverify_{kind}_{tag}.fits")
                _write_fits(p, out[kind], {"ANNULUS": tag, "NCONFIG": nn, "RLO": rlo, "RHI": rhi, "KIND": kind})
                out["files"][kind] = p
        p = os.path.join(out_dir, f"paramverify_calib_{tag}.txt")
        with open(p, "w") as f:
            f.write(f"# param_verify reliability calibration (annulus {tag})\n")
            f.write(f"n_configs      {nn}\n")
            f.write(f"detfrac_thr    {snr_thr:5.2f}\n")
            for k in ("null_median", "null_p90", "null_p95", "null_p99", "true_median", "true_min"):
                f.write(f"{k:<14} {calib[k]:9.3f}\n")
            f.write("# contrast curve (r_px  r_as  c5)\n")
            for i in range(reval.size):
                f.write(f"CC {reval[i]:8.2f}{reval[i]*pxscale:9.4f}{c5[i]:13.4E}\n")
        out["files"]["calib"] = p
    return out


def _write_fits(path: str, data: np.ndarray, header: Optional[Dict[str, Any]] = None) -> None:
    """Write a map as float32 FITS, its header keywords cut to 8 characters.

    Parameters
    ----------
    path : str
        The file, written over.
    data : ndarray
        The map.
    header : dict or None, default None
        Keywords, any value FITS refuses written as its text.
    """
    from astropy.io import fits
    h = fits.Header()
    for k, v in (header or {}).items():
        try:
            h[k[:8].upper()] = v
        except Exception:
            h[k[:8].upper()] = str(v)
    fits.PrimaryHDU(np.asarray(data, np.float32), header=h).writeto(path, overwrite=True)


# ----------------------------------------------------------------------------
# driver: History -> selection -> re-reduction -> products
# ----------------------------------------------------------------------------
def param_verify(history: History, space: SearchSpace, reduce_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
                 fwhm: float, pxscale: float, rlo: float, rhi: float, out_dir: str, *,
                 ccal: float, tag: str = "01", n_pv: int = 20, pv_divmin: float = 0.05,
                 n_init: Optional[int] = None, nsrc: int = 4, fixed_sources: Optional[Sequence[Source]] = None,
                 rng: Optional[np.random.Generator] = None, known: Sequence[Tuple[float, float]] = (),
                 snr_thr: float = 3.0, angle_convention: str = "pa", flatten: bool = False,
                 log: Callable[[str], None] = print, run_calls: Optional[Callable] = None,
                 inner_fwhm: float = 1.0, rules: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Run the param_verify stage for one annulus (opt 7781-7935 and ``near2m_paramverify``).

    The best diverse configurations (:func:`select_configs`) are reduced again, clean and with the fixed
    sources, and :func:`param_verify_cubes` takes their images, weighted by ``max(score, 0)`` (opt 7850).
    A configuration whose reduction fails is dropped, as the IDL predecessor's ``continue`` drops it.

    Parameters
    ----------
    history : History
        The annulus' search history.
    space : SearchSpace
        Its space, for the diversity distance.
    reduce_fn : callable
        Function ``reduce_fn(x, sources)`` returning the clean and the injected image of the configuration
        vector ``x``.  A one-argument ``reduce_fn(x)`` must inject the ``fixed_sources`` given here itself.
    fwhm : float
        FWHM, in pixels.
    pxscale : float
        Pixel scale, in arcseconds per pixel.
    rlo : float
        The annulus' inner radius in pixels, an edge of the null zone and of the fixed set's ladder.
    rhi : float
        Its outer radius, in pixels.
    out_dir : str
        Directory of the products.
    ccal : float
        The annulus' calibrated injection contrast (S/N 5), for the fixed set and the K calibration.
    tag : str, default "01"
        The products' tag.
    n_pv : int, default 20
        Most configurations (:func:`select_configs`).
    pv_divmin : float, default 0.05
        Their diversity floor.
    n_init : int or None, default None
        Index of the first TPE-phase evaluation, None to let the ``phase`` flags decide.
    nsrc : int, default 4
        Number of fixed sources when they are drawn here, the IDL predecessor's ``nsrc_a``.
    fixed_sources : sequence of Source or None, default None
        The fixed sources, None to draw them (:func:`fixed_injection_set`).
    rng : numpy.random.Generator or None, default None
        The generator of the fixed set's anchor.
    known : sequence of tuple, default ()
        Real companions, ``(rho_arcsec, pa_deg)``, kept out of the fixed set and of the curve's rings.
    snr_thr : float, default 3.0
        The S/N of a detection.
    angle_convention : str, default "pa"
        How the sources' angles are measured.
    flatten : bool, default False
        Whether to subtract each image's radial profile before its maps.
    log : callable, default print
        Function taking each progress line.
    run_calls : callable or None, default None
        Function ``run_calls(calls)`` yielding ``(i, result)`` for the re-reductions in order, an exception
        standing for a failed one, as :meth:`Runner._run_ordered` does with several in flight.  None runs
        them one after another.
    inner_fwhm : float, default 1.0
        The metric's inner guard for the curve's S/N, which the Runner sets from the injection floor.
    rules : dict or None, default None
        The run's noise-ring rules.

    Returns
    -------
    result : dict or None
        The dict of :func:`param_verify_cubes` with ``eval_indices`` and ``X`` added, None when fewer than 2
        configurations could be selected or reduced.
    """
    sel = select_configs(history, space, n_pv, pv_divmin, n_init)
    if len(sel) < 2:
        log(f"param_verify: only {len(sel)} distinct configs -- skipped")
        return None
    if fixed_sources is None:
        fixed_sources = fixed_injection_set(nsrc, rlo * pxscale, rhi * pxscale, fwhm, pxscale, rng, ccal, known)
    fixed_sources = [s if isinstance(s, Source) else Source(*s) for s in fixed_sources]
    try:
        pars = list(inspect.signature(reduce_fn).parameters.values())
        npos = len([p for p in pars if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        two = npos >= 2 or any(p.kind == p.VAR_POSITIONAL for p in pars)
    except (TypeError, ValueError):
        two = True
    cclean, cinj, wts, kept = [], [], [], []
    calls = [(lambda x=history.X[e]: reduce_fn(x, fixed_sources) if two else reduce_fn(x)) for e in sel]
    if run_calls is None:
        def run_calls(cs):
            for j, fn in enumerate(cs):
                try:
                    yield j, fn()
                except Exception as exc:          # noqa: BLE001 - handed to the loop below
                    yield j, exc
    for (i, e), (_, res) in zip(enumerate(sel), run_calls(calls)):
        log(f"param_verify config {i+1}/{len(sel)} (eval {e+1}, score {history.y[e]:.2f})")
        if isinstance(res, BaseException):    # a failed re-reduction just drops the config (IDL: continue)
            log(f"  reduction failed: {res!r}")
            continue
        clean, inj = res
        if clean is None or inj is None:
            continue
        cclean.append(np.asarray(clean, float))
        cinj.append(np.asarray(inj, float))
        wts.append(max(float(history.y[e]), 0.0))
        kept.append(int(e))
    if len(cclean) < 2:
        log("param_verify: fewer than 2 configs reduced -- skipped")
        return None
    out = param_verify_cubes(np.stack(cclean), np.stack(cinj), wts, fixed_sources, ccal, pxscale, fwhm,
                             rlo, rhi, out_dir, tag, snr_thr, angle_convention, flatten=flatten,
                             inner_fwhm=inner_fwhm, rules=rules, known=known)
    out["eval_indices"] = kept
    out["X"] = history.X[kept].copy()
    log(f"param_verify annulus {tag}: {len(kept)} configs; combine K~{out['Kc']:.3E}")
    return out


# ----------------------------------------------------------------------------
# C.4.3  stitch  (near2m_pvstitch, param_verify 367-394)
# ----------------------------------------------------------------------------
def stitch_param_maps(results: Sequence[Dict[str, Any]], zones: Sequence[Tuple[float, float]],
                      out_dir: Optional[str] = None, kinds: Sequence[str] = PV_KINDS,
                      pads: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
    """Merge the annuli's maps of each kind, as the IDL predecessor's ``near2m_pvstitch`` does.

    Each annulus gives its finite pixels in ``rlo - 1 <= r <= rhi + 1``, the overlaps (2 px seams) are
    averaged, and the rest is NaN.  The star is at ``(n - 1)/2``, where the IDL predecessor put it at
    ``n/2``.

    Parameters
    ----------
    results : sequence of dict
        Each annulus' :func:`param_verify_cubes` dict, or ``{kind: map}``.
    zones : sequence of tuple of 2 float
        Each annulus' ``(rlo, rhi)``, in pixels.
    out_dir : str or None, default None
        Directory of ``paramverify_<kind>_stitched.fits``, None to write none.
    kinds : sequence of str, default PV_KINDS
        The kinds of map.
    pads : sequence of tuple of 2 float or None, default None
        Each zone's ``(pad_in, pad_out)`` in pixels in place of 1 px, since the runner pads only where
        annuli meet.

    Returns
    -------
    maps : dict
        Each kind's stitched map, a kind no annulus has left out.
    """
    out: Dict[str, np.ndarray] = {}
    for kind in kinds:
        acc = wgt = None
        for j, (res, (rlo, rhi)) in enumerate(zip(results, zones)):
            pin, pout = (1.0, 1.0) if pads is None else (float(pads[j][0]), float(pads[j][1]))
            m = res.get(kind) if isinstance(res, dict) else None
            if m is None:
                continue
            m = np.asarray(m, float)
            if acc is None:
                acc = np.zeros(m.shape); wgt = np.zeros(m.shape)
                ny, nx = m.shape
                cx, cy = star_center(m.shape)
                yy, xx = np.mgrid[0:ny, 0:nx]
                rr = np.hypot(xx - cx, yy - cy)
            if m.shape != acc.shape:
                continue
            z = (rr >= rlo - pin) & (rr <= rhi + pout) & np.isfinite(m)
            acc[z] += m[z]
            wgt[z] += 1.0
        if acc is None:
            continue
        out[kind] = acc / np.where(wgt > 0, wgt, np.nan)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            _write_fits(os.path.join(out_dir, f"paramverify_{kind}_stitched.fits"), out[kind], {"KIND": kind})
    return out
