"""Parameter-ensemble persistence / reliability -- port of ``near2_param_verify.pro``
and the config-selection block of the optimizer main loop (opt 7795-7850).

The optimizer's top-N most *diverse* configurations of one annulus are each
re-reduced clean and with ONE shared fixed injection set; from the per-config
matched-filter S/N maps we form the param-STIM (mean / unbiased std across
configs), the detection fraction (clean S/N > 3), the recovery fraction
(injected S/N > 3), a throughput-normalised meta-combine with its S/N map and
a null / true STIM calibration table plus a throughput-calibrated 5-sigma
contrast curve of the meta-combine.  Products are written as FITS + txt into an
output directory tagged ``_NN``; :func:`stitch_param_maps` merges annuli
(``near2m_pvstitch``).

The reduction is injected through ``reduce_fn`` so this module is
pipeline-agnostic.  Conventions as in :mod:`klip_tpe.verify`.

Deliberate deviations from the IDL:

* ``near2m_paramverify`` (147) and ``near2m_pvstitch`` (381) use ``cx = nx/2``;
  the port uses ``(n-1)/2`` everywhere (brief 8.4, spec C.6).
* ``near2m_snrmap`` is :func:`klip_tpe.products.snr_map` (per-radius MAD-floored
  band statistics); its exact ring-index bookkeeping differs slightly but the
  quantity is the same.
* The PDF / PNG products and ``README.txt`` are not produced.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source, mawet_peak_snr, nanmedian_even, radprof, source_xy, star_center
from .optimizers import History
from .positions import PositionSampler
from .products import snr_map
from .space import SearchSpace

__all__ = ["select_configs", "fixed_injection_set", "pv_meanstd", "radial_rms", "param_stim",
           "param_verify_cubes", "param_verify", "stitch_param_maps", "PV_KINDS"]

PV_KINDS = ("stim", "detfrac", "recovery", "combine", "combsnr")
_NON_TPE_PHASES = ("seed", "warmup", "random", "grid", "calib", "calibration")


# ----------------------------------------------------------------------------
# C.4.1  config selection  (opt 7795-7817)
# ----------------------------------------------------------------------------
def select_configs(history: History, space: SearchSpace, n_pv: int = 20, pv_divmin: float = 0.05,
                   n_init: Optional[int] = None, min_score: float = -9000.0) -> List[int]:
    """Top-``n_pv`` distinct configs with a diversity floor.

    Pool = TPE-phase evaluations (index ``>= n_init`` when given, else trials
    whose ``flags['phase']`` is not a warm-up / random / seed phase) with finite
    score ``> min_score``; falls back to all scored trials if empty (7796-7797).
    Sorted by score descending, a config is kept when its mean normalised L1
    distance ``d = (1/ndim_base) sum_d |x_d - x'_d| / (hi_d - lo_d)`` over the
    base (non-partition reduction) dims to EVERY kept config is ``>= pv_divmin``
    (7800-7810).  Returns evaluation indices (may be fewer than 2 -- callers skip
    then)."""
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
    """The shared fixed injection set (``near2m_randpos``, opt 4141+): a
    deterministic radial ladder ``rlo + (i+0.5)(rhi-rlo)/n`` with a random common
    azimuth anchor and ``360/n`` spacing, avoiding ``known`` sources.  Delegates to
    :class:`klip_tpe.positions.PositionSampler` (strategy ``'spread'``); pass the run's
    own ``sampler`` so every stage injects with identical geometry (the Runner does)."""
    if rng is None:
        rng = np.random.default_rng(0)
    ps = sampler if sampler is not None else PositionSampler(fwhm_as=fwhm * pxscale, strategy="spread", known=list(known))
    return ps.sample(int(nsrc), float(rlo_as), float(rhi_as), rng, contrast)


# ----------------------------------------------------------------------------
# C.4.2  primitives
# ----------------------------------------------------------------------------
def pv_meanstd(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel NaN-aware mean, UNBIASED (N-1) two-pass std and count over the
    config axis (``near2m_pv_meanstd``, param_verify 44-56).  Mean / std are NaN
    where fewer than 2 configs are finite."""
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
    """param-STIM ``mn / max(sd, 0.3 median(sd > 0), 1e-6)`` (param_verify 157-163
    and 176-178), NaN where count < 2."""
    mn, sd, cnt = pv_meanstd(sn_cube)
    g = np.isfinite(sd) & (sd > 0)
    sfloor = float(np.median(sd[g])) if g.any() else 1.0
    stim = mn / np.maximum(np.maximum(sd, 0.3 * sfloor), 1e-6)
    stim[cnt < 2] = np.nan
    return stim


def radial_rms(img: np.ndarray, reval: Sequence[float]) -> np.ndarray:
    """Azimuthal sample std of ``img`` in ``[r-1, r+1)`` annuli at each ``reval``
    (``near2m_pv_radrms``, param_verify 62-73); NaN with < 3 pixels."""
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
    ny, nx = m.shape
    ix = int(np.clip(round(x), 1, nx - 2))
    iy = int(np.clip(round(y), 1, ny - 2))
    box = m[iy - 1:iy + 2, ix - 1:ix + 2]
    return float(np.nanmax(box)) if np.any(np.isfinite(box)) else float("nan")


def _percentile_idl(v: np.ndarray, p: float) -> float:
    """``v[sort(v)[floor(p (n-1))]]`` (param_verify 242-244)."""
    s = np.sort(v)
    return float(s[int(np.floor(p * (s.size - 1)))])


# ----------------------------------------------------------------------------
# C.4.2  near2m_paramverify on ready-made cubes
# ----------------------------------------------------------------------------
def param_verify_cubes(cclean: np.ndarray, cinj: np.ndarray, wts: Sequence[float],
                       sources: Sequence[Source], ccal: float, pxscale: float, fwhm: float,
                       rlo: float, rhi: float, out_dir: Optional[str] = None, tag: str = "01",
                       snr_thr: float = 3.0, angle_convention: str = "pa",
                       flatten: bool = False, write_fits: bool = True) -> Dict[str, Any]:
    """``near2m_paramverify`` (param_verify 135-249) on ``(N, ny, nx)`` clean and
    injected cubes of the ensemble.

    1. ``sncl[i] = snr_map(radprof(cclean_i))``; ``snij[i]`` likewise with the
       fixed sources excluded from the noise (153-154).
    2. param-STIM of ``sncl`` (157-163); ``detfrac`` = fraction of configs with
       clean S/N > ``snr_thr`` (166-167); ``recovery`` = same on ``snij`` (172).
    3. ``true[s]`` = 3x3 max of the injected-ensemble STIM at each source
       (176-183); ``null`` = clean STIM in ``rlo <= r <= rhi`` farther than
       ``1.5 fwhm`` from every source (185-191; ``[0]*5`` if < 5 pixels).
    4. Meta-combine (194-217): ``K_i`` = median over sources of the 3x3 peak of
       ``snij_i``; ``Kmed = median(K)``; ``we_i = w_i max(K_i, 0)``;
       ``comb = sum we_i (Kmed/K_i) cclean_i / sum we_i`` (same for ``combj``);
       ``combsnr = snr_map(radprof(comb))``.
    5. Contrast curve (221-229): ``ps = mawet_peak_snr(radprof(combj), sources)``;
       ``reval = rlo + (rhi-rlo)(i+0.5)/12``; ``K_s = ps_s sigma(r_s) / ccal``;
       ``Kc = median(K > 0)``; ``c5 = 5 sigma(r) / Kc``.

    Writes ``paramverify_{stim,detfrac,recovery,combine,combsnr}_NN.fits`` and
    ``paramverify_calib_NN.txt`` into ``out_dir`` when given.  Returns a dict
    with the maps, ``K``, ``Kc``, ``curve`` and ``calib``.
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
    sncl = np.stack([snr_map(_flat(cclean[i]), fwhm) for i in range(nn)])
    snij = np.stack([snr_map(_flat(cinj[i]), fwhm, exclude_xy=ex_xy) for i in range(nn)])

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
    combsnr = snr_map(_flat(comb), fwhm)

    ps = mawet_peak_snr(_flat(combj), rho, th, pxscale, fwhm, angle_convention=angle_convention)
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
                 log: Callable[[str], None] = print) -> Optional[Dict[str, Any]]:
    """Run the param_verify stage for one annulus (opt 7781-7935 + ``near2m_paramverify``).

    Parameters
    ----------
    history, space
        the annulus search history and its space (for the diversity metric).
    reduce_fn
        ``reduce_fn(x, sources) -> (clean_image, injected_image)`` re-reduces the
        configuration vector ``x`` clean and with ``sources`` injected.  A
        one-argument callable ``reduce_fn(x)`` is also accepted; it must then
        inject the ``fixed_sources`` you pass here.
    rlo, rhi
        annulus radii (px) -- the null zone and the radial ladder of the fixed set.
    ccal
        the annulus's calibrated (S/N = 5) injection contrast; used for the
        fixed set when ``fixed_sources`` is not given and for the K calibration.
    nsrc
        number of fixed sources when building the set (IDL ``nsrc_a``).
    n_init
        index of the first TPE-phase trial (else ``flags['phase']`` decides).

    Weights are ``max(score, 0)`` of each kept config (opt 7850).  Returns the
    dict of :func:`param_verify_cubes` extended with ``eval_indices`` and ``X``,
    or ``None`` when fewer than 2 configs could be selected / reduced.
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
    for i, e in enumerate(sel):
        x = history.X[e]
        log(f"param_verify config {i+1}/{len(sel)} (eval {e+1}, score {history.y[e]:.2f})")
        try:
            clean, inj = reduce_fn(x, fixed_sources) if two else reduce_fn(x)
        except Exception as exc:  # a failed re-reduction just drops the config (IDL: continue)
            log(f"  reduction failed: {exc!r}")
            continue
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
                             rlo, rhi, out_dir, tag, snr_thr, angle_convention, flatten=flatten)
    out["eval_indices"] = kept
    out["X"] = history.X[kept].copy()
    log(f"param_verify annulus {tag}: {len(kept)} configs; combine K~{out['Kc']:.3E}")
    return out


# ----------------------------------------------------------------------------
# C.4.3  stitch  (near2m_pvstitch, param_verify 367-394)
# ----------------------------------------------------------------------------
def stitch_param_maps(results: Sequence[Dict[str, Any]], zones: Sequence[Tuple[float, float]],
                      out_dir: Optional[str] = None, kinds: Sequence[str] = PV_KINDS) -> Dict[str, np.ndarray]:
    """Merge per-annulus maps: each annulus contributes its finite pixels in
    ``rlo-1 <= r <= rhi+1``; overlaps (2-px seams) are averaged, elsewhere NaN.
    ``results`` are :func:`param_verify_cubes` dicts (or ``{kind: map}``);
    ``zones`` the matching ``(rlo, rhi)`` in px.  Writes
    ``paramverify_<kind>_stitched.fits`` when ``out_dir`` is given.  DEVIATION:
    star centre ``(n-1)/2`` (IDL ``n/2``)."""
    out: Dict[str, np.ndarray] = {}
    for kind in kinds:
        acc = wgt = None
        for res, (rlo, rhi) in zip(results, zones):
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
            z = (rr >= rlo - 1.0) & (rr <= rhi + 1.0) & np.isfinite(m)
            acc[z] += m[z]
            wgt[z] += 1.0
        if acc is None:
            continue
        out[kind] = acc / np.where(wgt > 0, wgt, np.nan)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            _write_fits(os.path.join(out_dir, f"paramverify_{kind}_stitched.fits"), out[kind], {"KIND": kind})
    return out
