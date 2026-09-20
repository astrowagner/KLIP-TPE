"""Generic adapter: any registered ADI cube (+ angles, + optional PSF template and
reference cube) -> :class:`~klip_tpe.reducer.Dataset`, reducer, search space, guard and
objective.  This is the entry point for data that come out of VIP, pyKLIP, spaceKLIP or
your own pipeline as plain arrays / FITS files.

Minimal use::

    from klip_tpe.instruments import generic
    ds  = generic.load_cube("cube.fits", "angles.fits", psf="psf.fits", name="betapic")
    red = generic.make_reducer({ds.name: ds}, pxscale=0.02719, lam_m=3.8e-6, diam_m=8.2)
    space = generic.make_space(red)
    space.project = generic.make_guard(red)
    objective, sampler = generic.default_config(red)
    # -> Runner(red, space, objective, sampler, RunConfig(...), run_dir).run()

Conventions
-----------
* ``angles``: klip-tpe derotates every frame counter-clockwise by ``angle + truenorth`` to
  put North up, i.e. the same sign as VIP's ``angle_list`` and pyKLIP's ``PAs``.  Pass
  ``angle_sign=-1`` for the opposite convention.
* Frames must be registered on the star; the star is assumed at the array centre
  ``((nx-1)/2, (ny-1)/2)`` unless ``center=`` says otherwise (``crop_half`` recentres by
  cropping around it).
* ``psf``: an unsaturated off-axis / non-coronagraphic PSF image at the same pixel scale
  (VIP's ``psf`` array, spaceKLIP's offset PSF, pyNOMIC's Airy fit is handled by the
  NOMIC adapter).  Its total flux (inside ``ee_radius_px`` when given) is the star flux
  unit, so injected contrasts are relative to the template; when the template was taken
  with a different exposure / neutral density, pass ``star_flux`` = the star's flux in the
  science frames' units explicitly.  Without a PSF a Gaussian of ``1.028 lambda/D`` and
  flux unit 1 is used (fine for parameter *ranking*, not for calibrated contrasts).
* Frame-quality tags for the searched frame-selection dims are computed from the data
  when the cube does not bring any (``quality_tags``: correlation of each frame with the
  median frame and its noise level, both inside an annulus) so ``corr_thresh`` /
  ``noise_max`` mean the same thing as for NEAR / NOMIC data.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..feasibility import ReferenceLibraryGuard, FrameSelectionGuard, MinBinGuard, ReferenceCountGuard, compose
from ..injection import GaussianPSF, InjectionModel, TemplatePSF
from ..metrics import MawetPeakSNR, Objective
from ..positions import PositionSampler
from ..reducer import Dataset, PartitionedReducer, reducer_class
from ..space import SearchSpace
from . import near as _near

__all__ = ["load_cube", "quality_tags", "make_reducer", "make_space", "make_guard",
           "default_config", "injection_model_for", "aperture_sum",
           "star_flux_from_aperture_photometry"]

ArrayLike = Union[str, np.ndarray, Sequence[float]]


def _read(x: ArrayLike, ext: int = 0) -> np.ndarray:
    """FITS path or array -> float array."""
    if isinstance(x, str):
        from astropy.io import fits
        with fits.open(x, memmap=False) as h:
            hdu = h[ext] if h[ext].data is not None else next(u for u in h if u.data is not None)
            return np.asarray(hdu.data, float)
    return np.asarray(x, float)


def quality_tags(cube: np.ndarray, r_in: float = 6.0, r_out: Optional[float] = None,
                 center: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """Per-frame quality tags from the data themselves, in the units the optimizer's
    frame-selection dims expect: ``corrs`` = Pearson correlation of each frame with the
    median frame inside the annulus ``[r_in, r_out]`` px (``corr_thresh`` keeps frames with
    ``corrs >= corr_thresh``), ``noises`` = the frame's robust scatter in the annulus
    divided by the median over frames (``noise_max`` multiplier), ``coronoise`` = the same
    for the inner annulus ``[r_in, 2 r_in]`` (``coronoise_max``)."""
    n, ny, nx = cube.shape
    cx, cy = center if center is not None else ((nx - 1) / 2.0, (ny - 1) / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - cx, yy - cy)
    r_out = float(r_out) if r_out else min(nx, ny) / 2.0 - 2.0
    ann = (r >= r_in) & (r <= r_out)
    inner = (r >= r_in) & (r <= max(2.0 * r_in, r_in + 4))
    med = np.nanmedian(cube, axis=0)
    ref = med[ann]
    ref = ref - np.nanmean(ref)
    corrs = np.empty(n)
    noises = np.empty(n)
    coron = np.empty(n)
    for i in range(n):
        v = cube[i][ann]
        good = np.isfinite(v) & np.isfinite(ref)
        a, b = v[good] - np.nanmean(v[good]), ref[good]
        den = np.sqrt((a * a).sum() * (b * b).sum())
        corrs[i] = float((a * b).sum() / den) if den > 0 else 1.0
        d = v[good] - ref[good] - np.nanmedian(v[good] - ref[good])
        noises[i] = 1.4826 * np.nanmedian(np.abs(d)) if d.size else np.nan
        w = cube[i][inner]
        coron[i] = 1.4826 * np.nanmedian(np.abs(w - np.nanmedian(w))) if np.isfinite(w).any() else np.nan
    for arr in (noises, coron):
        m = np.nanmedian(arr)
        arr /= m if (np.isfinite(m) and m > 0) else 1.0
        arr[~np.isfinite(arr)] = 1.0
    corrs[~np.isfinite(corrs)] = 1.0
    return {"corrs": corrs, "noises": noises, "coronoise": coron}


def load_cube(cube: ArrayLike, angles: ArrayLike, psf: Optional[ArrayLike] = None, name: str = "data",
              crop_half: Optional[int] = None, center: Optional[Tuple[float, float]] = None,
              ref_cube: Optional[ArrayLike] = None, angle_sign: float = 1.0, angle_offset: float = 0.0,
              texp: Optional[float] = None, tags: Union[None, str, Dict[str, np.ndarray]] = "auto",
              tag_r_in: float = 6.0, nan_policy: str = "keep", meta: Optional[Dict[str, Any]] = None,
              frames: Optional[Sequence[int]] = None, wv_index: Optional[int] = None) -> Dataset:
    """Build a :class:`Dataset` from a cube (``(n, ny, nx)`` array or FITS) and its angles.

    ``psf`` (array / FITS) is kept in ``meta['psf']`` for :func:`injection_model_for`;
    ``ref_cube`` attaches a PSF-reference library (RDI / ARDI); ``crop_half`` crops to
    ``2*crop_half+1`` around ``center`` (default: the array centre); ``frames`` selects
    frame indices; ``wv_index`` picks a channel of a 4-d ``(nwv, n, ny, nx)`` IFS cube;
    ``tags='auto'`` computes :func:`quality_tags`, ``None`` leaves frame selection out,
    a dict supplies your own; ``nan_policy='zero'`` replaces NaN by 0 (``'keep'`` leaves
    them; the KLIP engine handles NaN pixels as missing)."""
    c = _read(cube)
    if c.ndim == 4:
        c = c[int(wv_index or 0)]
    if c.ndim != 3:
        raise ValueError(f"cube must be (n, ny, nx), got {c.shape}")
    a = np.asarray(_read(angles), float).ravel()
    if a.size == 1 and c.shape[0] > 1:
        a = np.repeat(a, c.shape[0])
    if a.size != c.shape[0]:
        raise ValueError(f"{c.shape[0]} frames but {a.size} angles")
    a = float(angle_sign) * a + float(angle_offset)
    if frames is not None:
        idx = np.asarray(frames, int)
        c, a = c[idx], a[idx]
    ny, nx = c.shape[1:]
    cx, cy = center if center is not None else ((nx - 1) / 2.0, (ny - 1) / 2.0)
    rc = None if ref_cube is None else _read(ref_cube)
    if crop_half:
        h = int(crop_half)
        x0, y0 = int(round(cx)) - h, int(round(cy)) - h
        if x0 < 0 or y0 < 0 or x0 + 2 * h + 1 > nx or y0 + 2 * h + 1 > ny:
            raise ValueError(f"crop_half={h} around ({cx:.1f}, {cy:.1f}) leaves the {nx}x{ny} frame")
        c = c[:, y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1]
        if rc is not None:
            rc = rc[:, y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1]
        cx, cy = cx - x0, cy - y0
    c = np.asarray(c, np.float32)
    if nan_policy == "zero":
        c = np.where(np.isfinite(c), c, 0.0).astype(np.float32)
    if isinstance(tags, str) and tags == "auto":
        tags = quality_tags(c, r_in=tag_r_in, center=(cx, cy))
    m = dict(meta or {})
    m.update({"center": (float(cx), float(cy)), "angle_sign": float(angle_sign), "angle_offset": float(angle_offset)})
    if psf is not None:
        m["psf"] = np.asarray(_read(psf), float)
    return Dataset(c, np.asarray(a, float), tags if isinstance(tags, dict) else None,
                   texp=float(texp) if texp else float(c.shape[0]), name=name, meta=m,
                   ref_cube=None if rc is None else np.asarray(rc, np.float32))


def _quadrant_area(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Area of ``{0<=u<=x, 0<=v<=y, u^2+v^2<=r^2}`` for ``x, y >= 0`` (exact)."""
    x = np.clip(np.asarray(x, float), 0.0, r)                 # beyond r the circle bounds it
    y = np.clip(np.asarray(y, float), 0.0, r)
    g = lambda t: 0.5 * (t * np.sqrt(np.clip(r * r - t * t, 0.0, None))
                         + r * r * np.arcsin(np.clip(t / r, -1.0, 1.0)))   # int_0^t sqrt(r^2-u^2)
    a = np.minimum(x, np.sqrt(np.clip(r * r - y * y, 0.0, None)))          # where the circle is above y
    return a * y + (g(x) - g(a))


def aperture_sum(img: np.ndarray, cx: float, cy: float, radius_px: float) -> float:
    """Flux in a circular aperture with *exact* partial-pixel areas.

    Matches photutils' ``method='exact'`` to floating point, without the dependency.  A
    whole-pixel mask (``hypot(x - cx, y - cy) <= r``) is NOT good enough here: at the radii
    published photometry is quoted in -- 2 px, one FWHM -- it differs from the exact
    aperture by several percent, and sub-pixel sampling converges only as 1/nsub, which is
    still 1e-3 at 40 samples per pixel.  Either error lands straight on the contrast axis.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    r = float(radius_px)
    if r <= 0:
        return 0.0
    x0, x1 = max(int(np.floor(cx - r)), 0), min(int(np.ceil(cx + r)) + 1, nx)
    y0, y1 = max(int(np.floor(cy - r)), 0), min(int(np.ceil(cy + r)) + 1, ny)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    # pixel (i, j) covers [j-0.5, j+0.5] x [i-0.5, i+0.5], shifted to the circle's frame
    xa = (np.arange(x0, x1) - 0.5 - cx)[None, :]
    xb = (np.arange(x0, x1) + 0.5 - cx)[None, :]
    ya = (np.arange(y0, y1) - 0.5 - cy)[:, None]
    yb = (np.arange(y0, y1) + 0.5 - cy)[:, None]
    s = lambda x, y: np.sign(x) * np.sign(y) * _quadrant_area(np.abs(x), np.abs(y), r)
    frac = s(xb, yb) - s(xa, yb) - s(xb, ya) + s(xa, ya)       # covered area per pixel
    return float((np.nan_to_num(img[y0:y1, x0:x1]) * frac).sum())


def star_flux_from_aperture_photometry(template: np.ndarray, starphot: float,
                                       aperture_px: float, center: Optional[Tuple[float, float]] = None,
                                       ee_radius_px: Optional[float] = None) -> float:
    """``star_flux`` for :class:`TemplatePSF` from a *published* stellar aperture flux.

    ``starphot`` is the star's flux inside ``aperture_px`` **in the science frames' own
    units** (an unsaturated/off-axis measurement rescaled to the science exposure time and
    neutral density).  The template is usually distributed normalised -- its own counts mean
    nothing -- so the star flux this returns is ``starphot`` rescaled from ``aperture_px``
    into whatever normalisation ``TemplatePSF`` will use (``ee_radius_px``, or the whole
    stamp when None), which is the only number that makes an injected ``contrast`` a real
    contrast.

    This is the *replacement* for the halo fit that used to live here: it uses photometry
    the observer actually made instead of fitting an off-axis template to a coronagraphic
    halo, which compares two different functions and was 4-8x wrong on exactly this data.
    """
    t = np.asarray(template, float)
    t = np.where(np.isfinite(t), t, 0.0)
    if center is None:
        center = ((t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0)
    a = aperture_sum(t, center[0], center[1], aperture_px)
    if not np.isfinite(a) or a <= 0:
        raise ValueError(f"the template has no flux inside {aperture_px} px of "
                         f"{center} -- wrong centre, or the core is masked out")
    f = float(t.sum()) if ee_radius_px is None else aperture_sum(t, center[0], center[1], ee_radius_px)
    return float(starphot) * f / a


def injection_model_for(ds: Dataset, fwhm_px: float, psf: Optional[np.ndarray] = None,
                        ee_radius_px: Optional[float] = None, star_flux: Optional[float] = None,
                        psf_center: Optional[Tuple[float, float]] = None) -> InjectionModel:
    """:class:`TemplatePSF` from ``psf`` (or ``ds.meta['psf']``).

    A template is **required**.  The previous fallback -- a Gaussian with flux unit 1 --
    produced a reducer that injected happily and reported a "contrast" that was really raw
    detector units, with nothing downstream able to tell the difference; paper run D shipped
    a contrast axis 2.3e5 off that way.  If you have no off-axis PSF, say so explicitly by
    constructing :class:`GaussianPSF` yourself, so the choice is in your code and not in a
    silent default.
    """
    t = ds.meta.get("psf") if psf is None else psf
    if t is None:
        raise ValueError(
            f"dataset {ds.name!r} has no PSF template: pass psf= to load_cube (or to "
            f"injection_model_for).  A template is required -- injecting a Gaussian of "
            f"unit flux gives an injection 'contrast' in raw detector units, which no "
            f"downstream product can distinguish from a real contrast.  To do that on "
            f"purpose, build GaussianPSF(fwhm_px, star_flux=...) yourself and pass it.")
    t = np.asarray(t, float)
    if t.ndim == 3:
        t = np.nanmedian(t, axis=0)
    if psf_center is None:
        iy, ix = np.unravel_index(np.nanargmax(np.where(np.isfinite(t), t, -np.inf)), t.shape)
        # sub-pixel centre from a 5x5 flux-weighted centroid around the peak
        y0, y1, x0, x1 = max(iy - 2, 0), min(iy + 3, t.shape[0]), max(ix - 2, 0), min(ix + 3, t.shape[1])
        sub = np.clip(np.nan_to_num(t[y0:y1, x0:x1]), 0, None)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        psf_center = (float((sub * xx).sum() / sub.sum()), float((sub * yy).sum() / sub.sum())) if sub.sum() > 0 else (ix, iy)
    return TemplatePSF(t, center=psf_center, ee_radius_px=ee_radius_px, star_flux=star_flux)


def make_reducer(datasets: Dict[str, Dataset], pxscale: float, lam_m: float, diam_m: float,
                 fwhm_px: Optional[float] = None, truenorth: float = 0.0, psf: Optional[np.ndarray] = None,
                 ee_radius_px: Optional[float] = None, star_flux: Optional[Union[float, Dict[str, float]]] = None,
                 weighting: str = "equal", max_workers="auto", pool: str = "auto", fast: bool = False,
                 n_min_ref: int = 10, outrad_cap: Optional[float] = None, use_rdi: Optional[bool] = None,
                 rdi_mode: str = "ardi", backend: str = "klip", defaults: Optional[Dict[str, Any]] = None,
                 partition_label: str = "dataset", log: Callable[[str], None] = print) -> PartitionedReducer:
    """One reducer per :class:`Dataset` (partition) on the chosen ``backend`` (``'klip'``
    built-in | ``'pyklip'`` | ``'vip'``), combined by a :class:`PartitionedReducer`.
    ``pxscale`` arcsec/px, ``lam_m`` / ``diam_m`` set lambda/D (``lam_m`` and ``fwhm_px`` may
    be ``{partition: value}`` dicts for multi-wavelength partitions such as IRDIS K1/K2);
    ``fwhm_px`` defaults to ``1.028 lambda/D``.  ``use_rdi`` defaults to True when a dataset carries a
    ``ref_cube``.  ``outrad_cap`` (px) defaults to the largest annulus that fits."""
    reducers = {}
    for pid, ds in datasets.items():
        lam = lam_m.get(pid) if isinstance(lam_m, dict) else lam_m           # per-partition wavelength (IRDIS K1/K2, IFS)
        lod = (lam / diam_m) * 206265.0 / pxscale
        fwp = fwhm_px.get(pid) if isinstance(fwhm_px, dict) else fwhm_px
        fw = float(fwp) if fwp else 1.028 * lod
        sf = star_flux.get(pid) if isinstance(star_flux, dict) else star_flux
        model = injection_model_for(ds, fw, psf=psf, ee_radius_px=ee_radius_px, star_flux=sf)
        cap = float(outrad_cap) if outrad_cap else float(min(ds.cube.shape[1:]) // 2 - 4)
        rdi = bool(ds.ref_cube is not None) if use_rdi is None else bool(use_rdi)
        d = {"fast": fast, "n_min_ref": n_min_ref, "comb_type": "nwadi", "use_rdi": rdi and ds.ref_cube is not None,
             "rdi_mode": rdi_mode}
        d.update(defaults or {})
        red = reducer_class(backend)(ds, pxscale=pxscale, lam_m=lam, diam_m=diam_m, injection_model=model,
                                     fallback_model=None, truenorth=truenorth, fwhm_px=fw, zone_pad=2.0,
                                     outrad_cap=cap, defaults=d)
        red.name = f"generic_{pid}"
        reducers[pid] = red
        log(f"  {pid}: {ds.cube.shape[0]} frames {ds.cube.shape[1]}x{ds.cube.shape[2]} px, PA span "
            f"{float(np.ptp(ds.angles)):.1f} deg, injection {model.name} (flux unit {model.flux_unit:.3g}), "
            f"fwhm {fw:.2f} px, lambda/D {lod:.2f} px" + (f", RDI library {ds.ref_cube.shape[0]} frames" if d["use_rdi"] else ""))
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool, log=log)
    pr.name = "generic"
    pr.partition_label = str(partition_label or "dataset")   # display wording only
    return pr


def make_space(reducer: PartitionedReducer, min_bins: int = 8, **kw) -> SearchSpace:
    """The NEAR/NOMIC parameterisation (bin, n_ang, filter, angsep, anglemax, frame
    selection, k_klip per partition + drop dims), with the ranges scaled to *these* data:

    * frame-selection dims only when the data carry quality tags;
    * ``k_klip`` up to ``nframes / 5`` (capped at 100), as in
      :func:`klip_tpe.instruments.near.make_space`;
    * the temporal-binning range is ``1 .. nframes / min_bins`` (default: at least 8 bins
      survive, so KLIP always has references).  The NEAR production range (5-30 frames per
      bin) assumes sequences of thousands of frames and collapses a 61-frame cube to two
      bins; pass ``bin_range=(lo, hi)`` to set it yourself.

    Other keywords: see :func:`klip_tpe.instruments.near.make_space`."""
    has_tags = all(r.frame_tags() for r in reducer.reducers.values())
    kw.setdefault("opt_framesel", has_tags)
    if "bin_range" not in kw:
        nf = min(r.data.nframes for r in reducer.reducers.values())
        kw["bin_range"] = (1, max(int(nf // max(min_bins, 2)), 1))
    space = _near.make_space(reducer, **kw)
    # A reducer carrying a searched reference library contributes one count per pool.
    # These are global dimensions, not per-partition ones: a pool is a pool whichever
    # partition is being reduced, and replicating them would let two partitions disagree
    # about how much of the same reference star exists.
    for r in reducer.reducers.values():
        for pr in getattr(r, "reference_params", lambda: [])():
            if pr.name not in {q.name for q in space.params}:
                space.add(pr)
    return space


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: Optional[bool] = None, bin_min: int = 1):
    """Frame-selection snapping + reference-count guardrail from the reducer's own
    lambda/D and FWHM.

    ``bin_min`` floors the temporal bin (:class:`~klip_tpe.feasibility.MinBinGuard`).  Leave
    it at 1 for short sequences; on a few thousand frames per partition set it to the NEAR
    production minimum of 5, or a single ``bin=1`` draw costs hours where its neighbours
    cost seconds.  It is applied first, so the reference-count census sees the floored bin.
    """
    r0 = next(iter(reducer.reducers.values()))
    ref = ReferenceCountGuard(angles_fn=reducer.frame_angles, tags_fn=reducer.frame_tags,
                              fwhm_px=r0.fwhm, lam_over_d_px=r0.lam_over_d_px,
                              n_min_ref=n_min_ref, ref_frac=ref_frac, k_max=k_max)
    mb = MinBinGuard(bin_min) if int(bin_min) > 1 else None
    if frame_selection is None:
        frame_selection = all(r.frame_tags() for r in reducer.reducers.values())
    fs = None
    if frame_selection:
        fs = FrameSelectionGuard(tags_fn=reducer.frame_tags,
                                 nframes_fn=lambda pid: reducer.reducers[pid].data.nframes)
    lib = None
    for r in reducer.reducers.values():
        spec = getattr(r, "_reflib_spec", None)
        if spec is not None:
            lib = ReferenceLibraryGuard(n_min_ref=int(spec.get("n_min_ref", 2)), k_max=k_max)
            break
    steps = [p for p in (mb, fs, ref, lib) if p is not None]
    return steps[0] if len(steps) == 1 else compose(*steps)


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   metric: str = "mawet", forbidden_pa=(), **metric_kw):
    """Detection objective + position sampler.

    ``metric='mawet'`` (default) is the matched-filter Mawet S/N with the injection PSF as
    the kernel; ``metric='fmmf'`` is :class:`klip_tpe.fmmf.FMMFSNR`, the same statistics
    with the template forward-modelled through each configuration's own subtraction (it
    needs ``clean_subtract=True`` on a backend without analytic KLIP-FM).  ``known`` =
    ``[(rho_arcsec, pa_deg), ...]`` real companions to keep injections away from.

    ``forbidden_pa`` = ``[(centre_pa_deg, half_width_deg), ...]`` azimuthal wedges the
    injections must avoid -- resolved emission at a position angle known in advance, a
    near-edge-on debris disk being the case it exists for.  It keeps sources *off* the
    structure; to also keep the structure out of the noise estimate, pass the matching
    ``pixel_mask=`` (:func:`klip_tpe.metrics.pa_wedge_mask` builds one from the same bands),
    which reaches the metric through ``metric_kw``.  Doing only one of the two is worse than
    doing neither: injections avoiding a disk that still inflates the ring sigma are scored
    against a noise level nothing is measuring them at.
    """
    kind = str(metric).lower()
    kw = dict(pxscale=reducer.pxscale, fwhm=reducer.fwhm, kernel_fn=reducer.matched_filter_kernel,
              known=list(known), **metric_kw)
    if kind in ("fmmf", "fm", "forward"):
        from ..fmmf import FMMFSNR
        m = FMMFSNR(**kw)
    elif kind in ("mawet", "mawet_peak", "peak"):
        m = MawetPeakSNR(**kw)
    else:
        raise ValueError(f"unknown metric {metric!r} (expected 'mawet' or 'fmmf')")
    objective = Objective(m, clean_subtract=clean_subtract)
    sampler = PositionSampler(fwhm_as=reducer.fwhm * reducer.pxscale, known=list(known),
                              forbidden_pa=[tuple(b) for b in forbidden_pa])
    return objective, sampler
