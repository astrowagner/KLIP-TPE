"""Generic adapter for any registered ADI cube, with its angles and an optional PSF template and reference cube.

It builds the :class:`~klip_tpe.reducer.Dataset`, the reducer, the search space, the guard and
the objective for data that come out of VIP, pyKLIP, spaceKLIP or another pipeline as plain
arrays or FITS files.

Notes
-----
* Angles: klip-tpe derotates every frame counter-clockwise by ``angle + truenorth`` to put North
  up, the sign of VIP's ``angle_list`` and pyKLIP's ``PAs``.  Pass ``angle_sign=-1`` for the
  opposite convention.
* Frames must be registered on the star, which sits at ``((nx-1)/2, (ny-1)/2)`` for frames of
  ny by nx pixels unless ``center`` says otherwise.  ``crop_half`` recenters by cropping around
  it.
* ``psf`` is an unsaturated off-axis or non-coronagraphic PSF image at the same plate scale, such
  as VIP's ``psf`` array or spaceKLIP's offset PSF (pyNOMIC's Airy fit has its own adapter,
  :mod:`klip_tpe.instruments.nomic`).  Its flux, within ``ee_radius_px`` when given, is the flux
  unit, so injected contrasts are relative to the template.  For a template taken with another
  exposure or neutral density, pass ``star_flux``, the star's flux in the science frames' units.
* Without quality tags in the data, :func:`quality_tags` computes them, so ``corr_thresh`` and
  ``noise_max`` mean what they mean for NEAR and NOMIC data.

Examples
--------
::

    from klip_tpe.instruments import generic
    ds = generic.load_cube("cube.fits", "angles.fits", psf="psf.fits", name="betapic")
    red = generic.make_reducer({ds.name: ds}, pxscale=0.02719, lam_m=3.8e-6, diam_m=8.2)
    space = generic.make_space(red)
    space.project = generic.make_guard(red)
    objective, sampler = generic.default_config(red)
    # then Runner(red, space, objective, sampler, RunConfig(...), run_dir).run()
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..feasibility import (ReferenceLibraryGuard, PyKLIPLibraryGuard, FrameSelectionGuard, MinBinGuard,
                           ReferenceCountGuard, compose)
from ..injection import GaussianPSF, InjectionModel, TemplatePSF
from ..metrics import MawetPeakSNR, Objective
from ..positions import PositionSampler
from ..reducer import Dataset, PartitionedReducer, reducer_class
from ..space import SearchSpace
from . import search_space as _space

__all__ = ["load_cube", "quality_tags", "make_reducer", "make_space", "make_guard",
           "default_config", "injection_model_for", "aperture_sum",
           "star_flux_from_aperture_photometry"]

ArrayLike = Union[str, np.ndarray, Sequence[float]]


def _read(x: ArrayLike, ext: int = 0) -> np.ndarray:
    """Read a FITS file, or take an array, as a float array.

    Parameters
    ----------
    x : str or array_like
        A FITS path or the values.
    ext : int, default 0
        FITS extension to read.  An extension without data gives way to the first that has some.

    Returns
    -------
    data : ndarray
        The values, in float64.
    """
    if isinstance(x, str):
        from astropy.io import fits
        with fits.open(x, memmap=False) as h:
            hdu = h[ext] if h[ext].data is not None else next(u for u in h if u.data is not None)
            return np.asarray(hdu.data, float)
    return np.asarray(x, float)


def quality_tags(cube: np.ndarray, r_in: float = 6.0, r_out: Optional[float] = None,
                 center: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """Compute per-frame quality tags from the data, in the units the frame-selection dimensions expect.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each.
    r_in : float, default 6.0
        Inner radius of the annulus the tags are measured in, in pixels.
    r_out : float or None, default None
        Outer radius of the annulus, in pixels.  None takes ``min(nx, ny)/2 - 2``.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``.

    Returns
    -------
    tags : dict of str to ndarray
        ``corrs``, the Pearson correlation of each frame with the median frame in the annulus, which
        ``corr_thresh`` keeps at or above.  ``noises``, the frame's robust scatter about the median
        frame there over its median across frames (``noise_max``), and ``coronoise``, the frame's own
        robust scatter from ``r_in`` to ``max(2 r_in, r_in + 4)`` over its median (``coronoise_max``).
    """
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
    """Build a :class:`~klip_tpe.reducer.Dataset` from a cube and its angles.

    Parameters
    ----------
    cube : str or ndarray
        The frames, shape (n, ny, nx), or (nwv, n, ny, nx) for an IFS cube, or a FITS path.
    angles : str or array_like
        Parallactic angle of each frame in degrees, or a FITS path.  One value is repeated for every
        frame.
    psf : str, ndarray or None, default None
        Off-axis PSF template, kept in ``meta["psf"]`` for :func:`injection_model_for`.
    name : str, default "data"
        Name of the dataset.
    crop_half : int or None, default None
        Half width of a crop to ``2 * crop_half + 1`` pixels around ``center``.  None does not crop.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``.
    ref_cube : str, ndarray or None, default None
        PSF-reference frames for RDI and ARDI, cropped as the cube is.
    angle_sign : float, default 1.0
        Factor on the angles, -1 for the opposite sign convention.
    angle_offset : float, default 0.0
        Offset added to the angles after the sign, in degrees.
    texp : float or None, default None
        Total exposure time, in seconds.  None takes the number of frames.
    tags : {"auto"}, dict or None, default "auto"
        Frame-quality tags.  "auto" computes :func:`quality_tags`, a dict supplies them and None
        leaves frame selection out.
    tag_r_in : float, default 6.0
        Inner radius of the annulus the "auto" tags are measured in, in pixels.
    nan_policy : {"keep", "zero"}, default "keep"
        What to do with NaN pixels.  "zero" replaces them by 0, and "keep" leaves them to the KLIP
        engine, which takes them as missing.
    meta : dict or None, default None
        More entries for the dataset's ``meta``.
    frames : sequence of int or None, default None
        Indices of the frames to keep.  None keeps all.
    wv_index : int or None, default None
        Channel of a 4-d IFS cube.  None takes the first.

    Returns
    -------
    ds : Dataset
        The dataset, with ``center``, ``angle_sign`` and ``angle_offset`` in its ``meta``.

    Raises
    ------
    ValueError
        If the cube is not 3-d once a channel is picked, the angles do not match the frames, or the
        crop leaves the frame.
    """
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
    """Compute the exact area of a circle's quadrant inside a rectangle that starts at the origin.

    It is the area of ``{0 <= u <= x, 0 <= v <= y, u^2 + v^2 <= r^2}``.

    Parameters
    ----------
    x : ndarray
        Width of the rectangle, at least 0.
    y : ndarray
        Height of the rectangle, at least 0.
    r : float
        Radius of the circle.

    Returns
    -------
    area : ndarray
        The area, shaped as ``x`` and ``y`` broadcast.
    """
    x = np.clip(np.asarray(x, float), 0.0, r)                 # beyond r the circle bounds it
    y = np.clip(np.asarray(y, float), 0.0, r)
    g = lambda t: 0.5 * (t * np.sqrt(np.clip(r * r - t * t, 0.0, None))
                         + r * r * np.arcsin(np.clip(t / r, -1.0, 1.0)))   # int_0^t sqrt(r^2-u^2)
    a = np.minimum(x, np.sqrt(np.clip(r * r - y * y, 0.0, None)))          # where the circle is above y
    return a * y + (g(x) - g(a))


def aperture_sum(img: np.ndarray, cx: float, cy: float, radius_px: float) -> float:
    """Sum the flux in a circular aperture, with exact partial-pixel areas.

    It matches photutils' ``method="exact"`` to floating point, without the dependency.  A
    whole-pixel mask differs from the exact aperture by several percent at the radii published
    photometry is quoted in, such as 2 px or one FWHM, and sub-pixel sampling converges only as
    1/nsub.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.  NaN pixels count as 0.
    cx : float
        Column of the aperture's center, in pixels.
    cy : float
        Row of the aperture's center, in pixels.
    radius_px : float
        Radius of the aperture, in pixels.

    Returns
    -------
    flux : float
        The flux, 0 for a radius of 0 or less or an aperture off the image.
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
    """Compute :class:`~klip_tpe.injection.TemplatePSF`'s ``star_flux`` from a published stellar aperture flux.

    The template is usually distributed normalized, so its own counts mean nothing.  The star flux is
    ``starphot`` rescaled from ``aperture_px`` into the normalization ``TemplatePSF`` uses, the only
    number that makes an injected contrast a real contrast.

    Parameters
    ----------
    template : ndarray, shape (sy, sx)
        The PSF template.  NaN pixels count as 0.
    starphot : float
        The star's flux inside ``aperture_px`` in the science frames' own units, an unsaturated or
        off-axis measurement rescaled to the science exposure time and neutral density.
    aperture_px : float
        Radius of the published aperture, in pixels.
    center : tuple of 2 floats or None, default None
        The template's ``(x, y)`` center.  None takes ``((sx-1)/2, (sy-1)/2)``.
    ee_radius_px : float or None, default None
        Radius of ``TemplatePSF``'s normalization, in pixels.  None takes the whole stamp.

    Returns
    -------
    star_flux : float
        The star's flux in the template's normalization.

    Raises
    ------
    ValueError
        If the template has no flux inside ``aperture_px``.

    Notes
    -----
    It replaces a halo fit that fitted an off-axis template to a coronagraphic halo, which compares
    two different functions and was 4-8x wrong on this data.
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
    """Build a :class:`~klip_tpe.injection.TemplatePSF` from a dataset's PSF template.

    A template is required.  Without one, build :class:`~klip_tpe.injection.GaussianPSF` yourself, so
    the choice is in your code and not in a silent default.

    Parameters
    ----------
    ds : Dataset
        The dataset, whose ``meta["psf"]`` serves when ``psf`` is None.
    fwhm_px : float
        FWHM of the PSF, in pixels, not used by the template.
    psf : ndarray or None, default None
        The template, 2-D, or a 3-d stack whose median is taken.
    ee_radius_px : float or None, default None
        Radius of the unit-flux normalization, in pixels.  None normalizes the total.
    star_flux : float or None, default None
        The star's flux in the template's normalization.  None takes the template's own.
    psf_center : tuple of 2 floats or None, default None
        The template's ``(x, y)`` center.  None takes a 5x5 flux-weighted centroid around the peak.

    Returns
    -------
    model : TemplatePSF
        The injection model.

    Raises
    ------
    ValueError
        If there is no template.

    Notes
    -----
    The earlier fallback, a Gaussian with flux unit 1, reported a contrast in raw detector units that
    nothing downstream could tell apart, and paper run D shipped a contrast axis 2.3e5 off that way.
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
        # sub-pixel center from a 5x5 flux-weighted centroid around the peak
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
    """Build one reducer per dataset on the chosen backend, combined by a :class:`~klip_tpe.reducer.PartitionedReducer`.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The datasets, one partition each.
    pxscale : float
        Plate scale, in arcsec per pixel.
    lam_m : float or dict
        Wavelength in meters, or ``{partition: value}`` for multi-wavelength partitions such as IRDIS
        K1 and K2.
    diam_m : float
        Telescope diameter, in meters.
    fwhm_px : float, dict or None, default None
        FWHM of the PSF in pixels, one value or one per partition.  None takes ``1.028`` lambda/D.
    truenorth : float, default 0.0
        True-north offset, in degrees.
    psf : ndarray or None, default None
        PSF template for every partition.  None takes each dataset's own.
    ee_radius_px : float or None, default None
        Radius of the template's unit-flux normalization, in pixels.
    star_flux : float, dict or None, default None
        The star's flux in the template's normalization, one value or one per partition.
    weighting : {"equal", "sqrt_texp"}, default "equal"
        Weights of the combination.
    max_workers : int or str, default "auto"
        The thread budget.
    pool : {"auto", "processes", "threads"}, default "auto"
        Kind of worker pool.
    fast : bool, default False
        Default of the reducers' ``fast``.
    n_min_ref : int, default 10
        Default minimum number of references per target.
    outrad_cap : float or None, default None
        Maximum outer radius of a zone, in pixels.  None takes the largest annulus that fits.
    use_rdi : bool or None, default None
        Whether to build the basis from the reference cube.  None does so when a dataset carries
        one.
    rdi_mode : {"ardi", "rdi"}, default "ardi"
        How the reference cube enters the basis.
    backend : {"klip", "pyklip", "vip"}, default "klip"
        The reduction backend (:func:`~klip_tpe.reducer.reducer_class`).
    defaults : dict or None, default None
        More default request parameters for every reducer.
    partition_label : str, default "dataset"
        What the partitions are, for the display's wording.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    reducer : PartitionedReducer
        The combined reducer.
    """
    reducers = {}
    for pid, ds in datasets.items():
        lam = lam_m.get(pid) if isinstance(lam_m, dict) else lam_m           # per-partition wavelength (IRDIS K1/K2, IFS)
        lod = (lam / diam_m) * 206265.0 / pxscale
        fwp = fwhm_px.get(pid) if isinstance(fwhm_px, dict) else fwhm_px
        fw = float(fwp) if fwp else 1.028 * lod
        sf = star_flux.get(pid) if isinstance(star_flux, dict) else star_flux
        # a loader that knows where the template's star is (a header's STARX/STARY) says so in
        # meta['psf_center']; otherwise the centroid of the peak finds it
        model = injection_model_for(ds, fw, psf=psf, ee_radius_px=ee_radius_px, star_flux=sf,
                                    psf_center=ds.meta.get("psf_center") if psf is None else None)
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
    """Build the NEAR and NOMIC parameterization, with ranges scaled to these data.

    Frame-selection dimensions appear only when the data carry quality tags, and ``k_klip`` reaches
    ``nframes / 5``, capped at 100.  The binning range is 1 to ``nframes / min_bins``, so KLIP always
    has references.  A searched reference library adds one global count per pool, not one per
    partition, since a pool is the same whichever partition is reduced.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer.
    min_bins : int, default 8
        Minimum number of bins that must survive the largest binning.
    kw : dict
        More keywords for :func:`klip_tpe.instruments.search_space.make_space`, such as
        ``bin_range=(lo, hi)``.

    Returns
    -------
    space : SearchSpace
        The search space.

    Notes
    -----
    The NEAR production range of 5 to 30 frames per bin assumes sequences of thousands of frames and
    collapses a 61-frame cube to two bins.  pyKLIP's ``maxnumbasis`` defaults to the default
    ``k_klip``, so a run with the library searched starts from what pyKLIP does unasked.
    """
    has_tags = all(r.frame_tags() for r in reducer.reducers.values())
    kw.setdefault("opt_framesel", has_tags)
    if "bin_range" not in kw:
        nf = min(r.data.nframes for r in reducer.reducers.values())
        kw["bin_range"] = (1, max(int(nf // max(min_bins, 2)), 1))
    space = _space.make_space(reducer, **kw)
    # A reducer carrying a searched reference library contributes one count per pool.
    # These are global dimensions, not per-partition ones: a pool is a pool whichever
    # partition is being reduced, and replicating them would let two partitions disagree
    # about how much of the same reference star exists.
    for r in reducer.reducers.values():
        for pr in getattr(r, "reference_params", lambda: [])():
            if pr.name not in {q.name for q in space.params}:
                space.add(pr)
    # pyKLIP's maxnumbasis defaults to the default k_klip, so the seeded configuration keeps
    # what pyKLIP always did unasked -- the k_klip most-correlated frames -- and a run with the
    # library searched starts from the same point as one without.
    # (Unconditionally: Param fills a missing default with its own grid midpoint.)
    for p in space.params:
        if p.name == "maxnumbasis":
            kdef = next((q.default for q in space.params if q.base == "k_klip" or q.name == "k_klip"), None)
            p.default = float(np.clip(kdef if kdef is not None else p.hi, p.lo, p.hi))
    return space


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: Optional[bool] = None, bin_min: int = 1):
    """Build the search space's guard, from the reducer's own lambda/D and FWHM.

    It chains the minimum bin, the frame-selection snapping, the reference-count guardrail and the
    reference library's guard, as each applies.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer.
    n_min_ref : int, default 10
        Minimum number of references per target.
    ref_frac : float, default 0.95
        Fraction of the target frames that must keep ``n_min_ref`` references.
    k_max : int, default 100
        Maximum number of KL modes.
    frame_selection : bool or None, default None
        Whether to snap the frame-selection dimensions.  None does so when every partition has
        quality tags.
    bin_min : int, default 1
        Minimum temporal bin (:class:`~klip_tpe.feasibility.MinBinGuard`), applied first.  On a few
        thousand frames per partition set it to 5, the NEAR production minimum, or a single
        ``bin=1`` draw costs hours.

    Returns
    -------
    guard : callable
        The guard, or the composition of several (:func:`~klip_tpe.feasibility.compose`).
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
        if spec is not None and getattr(r, "supports_reference_library", True):
            # the guard reads the basis size off the searched counts, which is the basis
            # size only when every pool is searched: a pool that keeps every eligible frame
            # (``searched`` of set_reference_library) is counted by the angular census instead
            if spec.get("searched") is None:
                lib = ReferenceLibraryGuard(n_min_ref=int(spec.get("n_min_ref", 2)), k_max=k_max)
            break
        nat = getattr(r, "_native_lib", None)
        if nat is not None:
            lib = PyKLIPLibraryGuard(n_alt=nat["n_alt"], n_ref=nat["n_ref"],
                                     default_mode=nat["default_mode"], k_max=k_max)
            break
    steps = [p for p in (mb, fs, ref, lib) if p is not None]
    return steps[0] if len(steps) == 1 else compose(*steps)


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   metric: str = "mawet", forbidden_pa=(), disk_mask=None, disk_mask_clear_fwhm: float = 0.0,
                   **metric_kw):
    """Build the detection objective and the position sampler.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer, whose plate scale, FWHM and matched filter the metric takes.
    clean_subtract : bool, default True
        Whether the objective subtracts the clean image's score (:class:`~klip_tpe.metrics.Objective`).
    known : sequence of tuple of 2 floats, default ()
        Real companions, as ``(rho_arcsec, pa_deg)``, kept out of the noise and away from the
        injections.
    metric : {"mawet", "fmmf"}, default "mawet"
        "mawet" is the matched-filter S/N with the injection PSF as the kernel.  "fmmf" is
        :class:`klip_tpe.fmmf.FMMFSNR`, the same statistics with the template forward-modeled through
        each configuration's own subtraction, which needs ``clean_subtract`` on a backend without
        analytic KLIP-FM.
    forbidden_pa : sequence of tuple of 2 floats, default ()
        Wedges the injections avoid, as ``(center_pa_deg, half_width_deg)``, for resolved emission at
        a position angle known in advance, such as a near-edge-on debris disk.  Pass the matching
        ``pixel_mask`` too (:func:`klip_tpe.metrics.pa_wedge_mask`), since injections that avoid a
        disk still inflating the ring's sigma are scored against noise nothing measures them at.
    disk_mask : ndarray of bool or None, default None
        Disk pixels, True on the disk, in the reduced frame's shape
        (:func:`klip_tpe.metrics.disk_mask_from_image`).  It joins the metric's ``pixel_mask`` and
        becomes the sampler's ``forbidden_mask``, both halves for a disk of any shape.
    disk_mask_clear_fwhm : float, default 0.0
        Margin kept clear around the disk mask for the injections, in FWHM.
    metric_kw : dict
        More keywords for the metric, such as ``pixel_mask``.

    Returns
    -------
    objective : Objective
        The objective.
    sampler : PositionSampler
        The sampler of injection positions.

    Raises
    ------
    ValueError
        If ``metric`` is unknown, or ``pixel_mask`` and ``disk_mask`` differ in shape.
    """
    kind = str(metric).lower()
    dm = None
    if disk_mask is not None:
        dm = np.asarray(disk_mask, bool)
        pm = metric_kw.get("pixel_mask")
        if pm is not None and np.shape(pm) != dm.shape:
            raise ValueError(f"pixel_mask {np.shape(pm)} and disk_mask {dm.shape} differ in shape")
        metric_kw["pixel_mask"] = dm if pm is None else (np.asarray(pm, bool) | dm)
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
                              forbidden_pa=[tuple(b) for b in forbidden_pa], forbidden_mask=dm,
                              mask_pxscale=reducer.pxscale if dm is not None else None,
                              mask_clear_fwhm=float(disk_mask_clear_fwhm),
                              mask_angle_convention=str(getattr(m, "angle_convention", "pa")))
    return objective, sampler
