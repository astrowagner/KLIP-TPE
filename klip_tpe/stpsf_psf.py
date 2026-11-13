"""STPSF (formerly WebbPSF) point-spread functions for the JWST reductions.

For a JWST coronagraphic mode it gives the optimizer:

* a separation-dependent injection model, the off-axis PSF of the actual mask, which is neither a
  Gaussian nor independent of separation inside a few lambda/D of a coronagraph, as a
  :class:`~klip_tpe.injection.LibraryPSF`,
* the mask's throughput ``T(rho)``, measured from the same grid, so injected and recovered
  contrasts are on the instrument's own scale,
* the matched filter of the detection metric, taken from the same PSFs and, with
  :mod:`klip_tpe.fmmf`, forward-modeled through the KLIP subtraction.

Usage::

    from klip_tpe import stpsf_psf
    grid  = stpsf_psf.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R",
                                   pupil_mask="MASKRND", seps_as=np.arange(0.2, 2.6, 0.2))
    model = stpsf_psf.library(grid, star_flux=F_star)      # -> LibraryPSF
    red   = spaceklip.make_reducer(dsets, injection_model=model, ...)

or, straight off the data, ``model = stpsf_psf.model_for_datasets(dsets, star_flux=F_star)``.

Each slice is an off-axis PSF already centered on the source and cropped to ``stamp_px``, the
convention :class:`~klip_tpe.injection.LibraryPSF` expects, and ``transmission[i]`` is the fraction
of the unocculted aperture flux that survives the mask and Lyot stop at that separation.  Every PSF
costs seconds to compute, so a grid is cached as one FITS per configuration under
``$KLIP_TPE_DATA/stpsf_cache`` (default ``~/.klip_tpe/stpsf_cache``).

STPSF itself (``pip install stpsf``, Python >= 3.10) and its ~90 MB data files (``STPSF_PATH``) are
an optional dependency.  Everything here raises a clear error when they are missing, and no other
klip-tpe module imports this one at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .injection import LibraryPSF

__all__ = ["offaxis_grid", "library", "throughput_fn", "model_for_datasets",
           "mode_from_header", "cache_dir", "DEFAULTS", "have_stpsf",
           "unocculted_ee", "star_flux_from_flux_density"]

#: ``image_mask`` -> the instrument and pupil stop that go with it, so callers only have
#: to name the coronagraph they used.
DEFAULTS: Dict[str, Dict[str, str]] = {
    "MASK210R": {"instrument": "NIRCam", "pupil_mask": "MASKRND"},
    "MASK335R": {"instrument": "NIRCam", "pupil_mask": "MASKRND"},
    "MASK430R": {"instrument": "NIRCam", "pupil_mask": "MASKRND"},
    "MASKSWB": {"instrument": "NIRCam", "pupil_mask": "MASKSWB"},
    "MASKLWB": {"instrument": "NIRCam", "pupil_mask": "MASKLWB"},
    "FQPM1065": {"instrument": "MIRI", "pupil_mask": "MASKFQPM"},
    "FQPM1140": {"instrument": "MIRI", "pupil_mask": "MASKFQPM"},
    "FQPM1550": {"instrument": "MIRI", "pupil_mask": "MASKFQPM"},
    "LYOT2300": {"instrument": "MIRI", "pupil_mask": "MASKLYOT"},
}

_CACHE_VERSION = 3

#: ``source_offset_theta`` that puts the source on **+x** of the detector array: STPSF
#: measures the offset angle from +y, increasing towards -x, so +x is 270 deg.  Verified
#: against the computed PSFs.  It fixes where in the stamp the source sits; it does NOT make
#: the stamp's *orientation* a function of the source's azimuth -- see :func:`library`.
_THETA_PLUS_X = 270.0


def have_stpsf() -> bool:
    """Tell whether ``stpsf`` imports and its data files are reachable.

    It asks STPSF where its data is rather than reading ``STPSF_PATH`` from the environment.  The
    variable is only one of the ways STPSF finds its data, since a conda install or an ``~/.stpsf``
    config file sets it up without exporting one, so testing the variable reports no STPSF on a machine
    where it works.  The variables stay as the fallback for an older STPSF without the accessor.

    Returns
    -------
    available : bool
        True when STPSF and its data are there.
    """
    try:
        import stpsf
    except Exception:
        return False
    try:
        p = stpsf.utils.get_stpsf_data_path()
        if p and os.path.isdir(p):
            return True
    except Exception:
        pass
    p = os.environ.get("STPSF_PATH") or os.environ.get("WEBBPSF_PATH")
    return bool(p and os.path.isdir(p))


def cache_dir() -> str:
    """Return the directory new cache files are written to, made where needed.

    It is ``$KLIP_TPE_DATA/stpsf_cache`` when the variable is set, else ``~/.klip_tpe/stpsf_cache``.
    Reads go through :func:`_cache_path`, which also looks in the other layout, so a grid computed in
    one shell is found from another.

    Returns
    -------
    path : str
        The directory.
    """
    d = os.environ.get("KLIP_TPE_DATA") or os.path.join(os.path.expanduser("~"), ".klip_tpe")
    d = os.path.join(d, "stpsf_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_dirs() -> list:
    """Return every directory a cached PSF may be read from, in order of preference.

    ``cache_dir()`` depends on ``KLIP_TPE_DATA``: set to ``~/.klip_tpe/data`` it puts the cache at
    ``~/.klip_tpe/data/stpsf_cache``, and unset at ``~/.klip_tpe/stpsf_cache``.  A grid computed under
    one setting was invisible under the other, which on a machine without STPSF is fatal, so a read
    tries both layouts.

    Returns
    -------
    dirs : list of str
        The absolute directories, each once.

    Notes
    -----
    Paper run D refused to start on 2026-09-16 with "the grid is not in the cache" while the grid sat
    one directory over.
    """
    home = os.path.join(os.path.expanduser("~"), ".klip_tpe")
    cands = [cache_dir()]
    env = os.environ.get("KLIP_TPE_DATA")
    if env:
        cands.append(os.path.join(env, "stpsf_cache"))
    cands += [os.path.join(home, "data", "stpsf_cache"), os.path.join(home, "stpsf_cache")]
    out: list = []
    for c in cands:
        c = os.path.abspath(c)
        if c not in out:
            out.append(c)
    return out


def _cache_path(name: str) -> str:
    """Return the path of a cache file, where it exists, else its path in :func:`cache_dir`.

    Parameters
    ----------
    name : str
        Name of the file.

    Returns
    -------
    path : str
        The path.
    """
    for d in _cache_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(cache_dir(), name)


def _key(meta: Dict[str, Any]) -> str:
    """Return a short, stable key of a configuration, the first 16 hex digits of the SHA-1 of its JSON.

    Parameters
    ----------
    meta : dict
        The configuration.

    Returns
    -------
    key : str
        The key.
    """
    s = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _odd(n: float) -> int:
    """Return a number rounded up to the next odd integer.

    Parameters
    ----------
    n : float
        The number.

    Returns
    -------
    odd : int
        The odd integer.
    """
    n = int(np.ceil(n))
    return n + 1 if n % 2 == 0 else n


def _instrument(name: str, filter_: str, image_mask: Optional[str], pupil_mask: Optional[str],
                aperture: Optional[str], detector_position: Optional[Tuple[int, int]],
                date: Optional[str]):
    """Return an STPSF instrument set up for a coronagraphic mode.

    Parameters
    ----------
    name : str
        The instrument's class name in STPSF, such as "NIRCam".
    filter_ : str
        The filter.
    image_mask : str or None
        The coronagraph's mask.
    pupil_mask : str or None
        The pupil stop.
    aperture : str or None
        The aperture's name, which sets the field position.
    detector_position : tuple of 2 ints or None
        The position on the detector, in pixels.
    date : str or None
        Date of the measured wavefront to load, or None for the default.

    Returns
    -------
    inst : stpsf instrument
        The instrument.

    Raises
    ------
    ImportError
        If STPSF is not installed.
    """
    try:
        import stpsf
    except ImportError as exc:                                  # pragma: no cover - optional dep
        raise ImportError("STPSF is required for klip_tpe.stpsf_psf (pip install stpsf, "
                          "Python >= 3.10, plus its data files via STPSF_PATH)") from exc
    inst = getattr(stpsf, name)()
    inst.filter = filter_
    if image_mask:
        inst.image_mask = image_mask
    if pupil_mask:
        inst.pupil_mask = pupil_mask
    if aperture:
        inst.set_position_from_aperture_name(aperture)
    if detector_position:
        inst.detector_position = tuple(int(v) for v in detector_position)
    if date:                                                    # measured wavefront of the epoch
        inst.load_wss_opd_by_date(date, plot=False, verbose=False)
    return inst


def _ee_radius(ref: np.ndarray, frac: float = 0.5, scale: float = 1.5) -> float:
    """Return ``scale`` times the radius enclosing the fraction ``frac`` of a centered PSF's flux.

    Parameters
    ----------
    ref : ndarray, shape (ny, nx)
        The PSF.
    frac : float, default 0.5
        The encircled fraction.
    scale : float, default 1.5
        Factor on the radius.

    Returns
    -------
    radius : float
        The radius, in pixels, 3 for a PSF without flux.
    """
    ref = np.asarray(ref, float)
    cy, cx = (ref.shape[0] - 1) / 2.0, (ref.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:ref.shape[0], 0:ref.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    tot = float(np.nansum(ref))
    if not np.isfinite(tot) or tot <= 0:
        return 3.0
    rad = np.arange(1.0, max(3.0, min(ref.shape) / 2.0))
    cog = np.array([np.nansum(ref[rr <= r]) / tot for r in rad])
    if cog[-1] <= frac:
        return float(rad[-1]) * scale
    return float(rad[int(np.searchsorted(cog, frac))]) * scale


def _ap_sum(img: np.ndarray, r: float) -> float:
    """Return the flux of a centered aperture.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    r : float
        Radius of the aperture, in pixels.

    Returns
    -------
    flux : float
        The sum of the finite pixels within ``r`` of the center.
    """
    img = np.asarray(img, float)
    cy, cx = (img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float(np.nansum(img[np.hypot(xx - cx, yy - cy) <= r]))


def _source_center(img: np.ndarray, guess: Tuple[float, float], win: float = 5.0) -> Tuple[float, float]:
    """Return the flux-weighted center of the single source in an image, refined from a guess.

    ``calc_psf`` with a ``source_offset`` computes one off-axis point source and nothing else, so the
    brightest structure is the source, and measuring it beats trusting a convention for where STPSF
    puts the array's center after detector binning.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    guess : tuple of 2 floats
        The guess, ``(x, y)`` in pixels.
    win : float, default 5.0
        Half-width of the box searched for the peak, in pixels.

    Returns
    -------
    x : float
        Column of the center, in pixels.
    y : float
        Row of the center, in pixels.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    gx, gy = float(guess[0]), float(guess[1])
    w = int(max(2, np.ceil(win)))
    x0, x1 = max(0, int(round(gx)) - w), min(nx, int(round(gx)) + w + 1)
    y0, y1 = max(0, int(round(gy)) - w), min(ny, int(round(gy)) + w + 1)
    box = np.where(np.isfinite(img[y0:y1, x0:x1]), img[y0:y1, x0:x1], 0.0)
    if box.size == 0 or box.max() <= 0:                        # pragma: no cover - defensive
        return gx, gy
    iy, ix = np.unravel_index(int(np.argmax(box)), box.shape)
    px, py = x0 + ix, y0 + iy
    h = 2                                                       # centroid on the PSF core only
    bx0, bx1 = max(0, px - h), min(nx, px + h + 1)
    by0, by1 = max(0, py - h), min(ny, py + h + 1)
    sub = np.where(np.isfinite(img[by0:by1, bx0:bx1]), img[by0:by1, bx0:bx1], 0.0)
    sub = np.clip(sub - sub.min(), 0, None)
    tot = float(sub.sum())
    if tot <= 0:                                               # pragma: no cover - defensive
        return float(px), float(py)
    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    return float((sub * xx).sum() / tot), float((sub * yy).sum() / tot)


def _stamp_at(img: np.ndarray, src_xy: Tuple[float, float], stamp: int) -> np.ndarray:
    """Cut a square stamp around a source, shifting the sub-pixel remainder so the source lands on the central pixel.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    src_xy : tuple of 2 floats
        The source, ``(x, y)`` in pixels.
    stamp : int
        Width of the stamp, in pixels.

    Returns
    -------
    stamp : ndarray, shape (stamp, stamp)
        The stamp, 0 outside the image.
    """
    from scipy import ndimage
    ny, nx = img.shape
    sx, sy = float(src_xy[0]), float(src_xy[1])
    ix, iy = int(round(sx)), int(round(sy))
    rx, ry = sx - ix, sy - iy                    # sub-pixel remainder
    h = (stamp - 1) // 2
    x0, y0 = ix - h, iy - h
    out = np.zeros((stamp, stamp), float)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + stamp, nx), min(y0 + stamp, ny)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = np.where(
            np.isfinite(img[ys0:ys1, xs0:xs1]), img[ys0:ys1, xs0:xs1], 0.0)
    if abs(rx) > 1e-3 or abs(ry) > 1e-3:
        out = ndimage.shift(out, (-ry, -rx), order=3, mode="constant", cval=0.0)
    return out


def offaxis_grid(instrument: str = "NIRCam", filter: str = "F444W",
                 image_mask: Optional[str] = "MASK335R", pupil_mask: Optional[str] = None,
                 seps_as: Optional[Sequence[float]] = None, stamp_px: int = 31,
                 oversample: int = 2, nlambda: int = 3, fov_px: Optional[int] = None,
                 aperture: Optional[str] = None, detector_position: Optional[Tuple[int, int]] = None,
                 date: Optional[str] = None, source=None, cache: bool = True,
                 log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Compute the off-axis PSF of a coronagraphic mode on a ladder of separations.

    ``fov_px`` is sized per separation so the source is always inside the computed field, and should be
    left None.  The results are cached by configuration under :func:`cache_dir`.

    Parameters
    ----------
    instrument : str, default "NIRCam"
        The instrument.
    filter : str, default "F444W"
        The filter.
    image_mask : str or None, default "MASK335R"
        The coronagraph's mask.
    pupil_mask : str or None, default None
        The pupil stop.  None takes the Lyot stop that goes with ``image_mask`` (:data:`DEFAULTS`).
    seps_as : sequence of float or None, default None
        The separations, in arcsec, at least two.  None takes 0.2 to 2.6 in steps of 0.2.
    stamp_px : int, default 31
        Width of the stamps, in pixels, made odd.
    oversample : int, default 2
        STPSF's oversampling.
    nlambda : int, default 3
        Number of wavelengths over the band.
    fov_px : int or None, default None
        Field of each computation, in pixels.
    aperture : str or None, default None
        The aperture's name.
    detector_position : tuple of 2 ints or None, default None
        The position on the detector, in pixels.
    date : str or None, default None
        Date of the measured wavefront to load.
    source : object, default None
        STPSF's source spectrum.
    cache : bool, default True
        Whether to read and write the cache.  False recomputes, and does not write.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    grid : dict
        ``slices`` (``(nsep, stamp_px, stamp_px)`` source-centered off-axis PSFs), ``seps`` (arcsec),
        ``transmission`` (the coronagraph's throughput at each separation, aperture photometry of the
        occulted PSF over the same aperture on the unocculted one), ``unocculted`` (the same-size stamp
        of the unocculted PSF), ``pxscale``, ``fwhm_px``, ``ee_radius_px`` (the aperture of
        ``transmission``), ``center`` (the stamp's center), ``meta`` (the configuration) and ``path``.

    Raises
    ------
    ValueError
        If fewer than two separations are given.
    RuntimeError
        If the grid is not in the cache and STPSF is not installed, the error naming the file to copy.
    """
    if pupil_mask is None and image_mask in DEFAULTS:
        pupil_mask = DEFAULTS[image_mask]["pupil_mask"]
    seps = np.asarray(list(seps_as) if seps_as is not None else np.arange(0.2, 2.61, 0.2), float)
    seps = np.sort(seps[np.isfinite(seps) & (seps > 0)])
    if seps.size < 2:
        raise ValueError("offaxis_grid needs at least two separations")
    stamp_px = _odd(stamp_px)
    meta = {"instrument": instrument, "filter": filter, "image_mask": image_mask,
            "pupil_mask": pupil_mask, "stamp_px": int(stamp_px), "oversample": int(oversample),
            "nlambda": int(nlambda), "fov_px": None if fov_px is None else int(fov_px),
            "aperture": aperture, "detector_position": detector_position, "date": date,
            "seps": [round(float(s), 4) for s in seps],
            "source": str(source) if source is not None else None, "version": _CACHE_VERSION}
    path = _cache_path(f"stpsf_{instrument}_{filter}_{image_mask or 'nomask'}_{_key(meta)}.fits")
    if cache and os.path.exists(path):
        try:
            out = _read_cache(path)
            log(f"  stpsf: {len(out['seps'])} off-axis PSFs from cache {os.path.basename(path)}")
            return out
        except Exception as exc:                                # pragma: no cover - corrupt cache
            log(f"  stpsf: ignoring unreadable cache {os.path.basename(path)} ({exc})")

    if not have_stpsf():
        # The whole point of the cache is that a machine without STPSF can still run from a
        # copied one -- the Mac this is developed against has Python 3.9, which STPSF will
        # never install on.  So a cache miss there is not "install STPSF", it is "you are
        # missing one file", and the error has to say WHICH.  Without this the failure was a
        # bare ImportError six frames down, on a run that had already loaded 26 files.
        raise RuntimeError(
            f"{f'the {instrument} {filter} off-axis PSF grid'} is not in the cache and STPSF is not installed to compute it. Copy "
            f"{os.path.basename(path)} into {cache_dir()} from a machine that has STPSF "
            f"(Python >= 3.10), or install STPSF and its data files (STPSF_PATH).")
    inst = _instrument(instrument, filter, image_mask, pupil_mask, aperture, detector_position, date)
    pxscale = float(getattr(inst, "pixelscale", 0.0)) or float("nan")
    if not np.isfinite(pxscale) or pxscale <= 0:                # pragma: no cover - defensive
        raise RuntimeError(f"stpsf gave no pixel scale for {instrument}")
    kw: Dict[str, Any] = dict(oversample=int(oversample), nlambda=int(nlambda))
    if source is not None:
        kw["source"] = source
    inst.options.pop("source_offset_r", None)
    inst.options.pop("source_offset_theta", None)

    log(f"  stpsf: {instrument} {filter} {image_mask or 'no mask'} / {pupil_mask or 'no stop'}, "
        f"{pxscale*1e3:.1f} mas/px, {len(seps)} separations")
    mask_keep, inst.image_mask = inst.image_mask, None                   # unocculted reference
    f0 = _odd(stamp_px + 8)
    ref_full = np.asarray(inst.calc_psf(fov_pixels=f0, **kw)["DET_SAMP"].data, float)
    inst.image_mask = mask_keep
    ref = _stamp_at(ref_full, _source_center(ref_full, ((f0 - 1) / 2.0, (f0 - 1) / 2.0)), stamp_px)
    fwhm_px = _fwhm_of(ref)
    ee_px = _ee_radius(ref, frac=0.5, scale=1.5)
    ref_ap = _ap_sum(ref, ee_px)

    slices, trans = [], []
    for s in seps:
        dx = float(s) / pxscale
        fov = int(fov_px) if fov_px else _odd(2.0 * abs(dx) + stamp_px + 6)
        inst.options["source_offset_r"] = float(s)
        inst.options["source_offset_theta"] = _THETA_PLUS_X               # source along +x
        p = inst.calc_psf(fov_pixels=fov, **kw)
        full = np.asarray(p["DET_SAMP"].data, float)
        guess = ((fov - 1) / 2.0 + dx, (fov - 1) / 2.0)
        st = _stamp_at(full, _source_center(full, guess, win=max(4.0, 0.15 * abs(dx) + 3.0)), stamp_px)
        t = _ap_sum(st, ee_px) / ref_ap if ref_ap > 0 else np.nan
        slices.append(st)
        trans.append(float(t))
        log(f"    rho = {s:5.2f}\"  ({fov:3d} px FOV)  throughput {t:.3f}")
    inst.options.pop("source_offset_r", None)
    inst.options.pop("source_offset_theta", None)

    c = (stamp_px - 1) / 2.0
    out = {"slices": np.array(slices), "seps": seps, "transmission": np.array(trans, float),
           "unocculted": ref, "pxscale": pxscale, "fwhm_px": float(fwhm_px),
           "ee_radius_px": float(ee_px), "center": (c, c), "meta": meta, "path": path}
    if cache:
        _write_cache(path, out)
        log(f"  stpsf: cached -> {os.path.basename(path)}")
    return out


def _fwhm_of(psf: np.ndarray) -> float:
    """Return the FWHM of a centered PSF, from where its azimuthal profile first falls below half the peak.

    Parameters
    ----------
    psf : ndarray, shape (ny, nx)
        The PSF.

    Returns
    -------
    fwhm : float
        The FWHM, in pixels, linearly interpolated, 3 for a PSF without a positive peak.
    """
    psf = np.asarray(psf, float)
    cy, cx = (psf.shape[0] - 1) / 2.0, (psf.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    pk = float(np.nanmax(psf))
    if not np.isfinite(pk) or pk <= 0:
        return 3.0
    rad = np.arange(0.0, min(psf.shape) / 2.0)
    prof = np.array([np.nanmean(psf[(rr >= r - 0.5) & (rr < r + 0.5)]) if
                     ((rr >= r - 0.5) & (rr < r + 0.5)).any() else np.nan for r in rad])
    prof = np.where(np.isfinite(prof), prof, 0.0) / pk
    below = np.nonzero(prof < 0.5)[0]
    if below.size == 0:
        return float(2 * rad[-1])
    i = int(below[0])
    if i == 0:
        return 1.0
    f = (prof[i - 1] - 0.5) / max(prof[i - 1] - prof[i], 1e-12)
    return float(2.0 * (rad[i - 1] + f * (rad[i] - rad[i - 1])))


def _write_cache(path: str, g: Dict[str, Any]) -> None:
    """Write a grid to its cache file.

    Parameters
    ----------
    path : str
        Path of the file.
    g : dict
        The grid.
    """
    from astropy.io import fits
    hdr = fits.Header()
    hdr["PIXELSCL"] = (float(g["pxscale"]), "arcsec / detector pixel")
    hdr["SRCX"], hdr["SRCY"] = float(g["center"][0]), float(g["center"][1])
    hdr["FWHMPX"] = float(g["fwhm_px"])
    hdr["EERADPX"] = float(g["ee_radius_px"])
    hdr["KTVER"] = _CACHE_VERSION
    for i, line in enumerate(_chunk(json.dumps(g["meta"]), 60)):
        hdr[f"KTMETA{i:02d}"] = line
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(np.asarray(g["slices"], np.float32), name="SLICES"),
                  fits.ImageHDU(np.asarray(g["seps"], float), name="SEPS"),
                  fits.ImageHDU(np.asarray(g["transmission"], float), name="TRANS"),
                  fits.ImageHDU(np.asarray(g["unocculted"], np.float32), name="UNOCC"),
                  ]).writeto(path, overwrite=True)


def _chunk(s: str, n: int):
    """Cut a string into pieces of ``n`` characters, for FITS cards.

    Parameters
    ----------
    s : str
        The string.
    n : int
        Length of the pieces.

    Returns
    -------
    pieces : list of str
        The pieces, one empty piece for an empty string.
    """
    return [s[i:i + n] for i in range(0, len(s), n)] or [""]


def _read_cache(path: str) -> Dict[str, Any]:
    """Read a grid from its cache file.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    grid : dict
        The grid, as :func:`offaxis_grid` returns it.
    """
    from astropy.io import fits
    with fits.open(path) as h:
        hdr = h[0].header
        meta = "".join(str(hdr[k]) for k in sorted(k for k in hdr if k.startswith("KTMETA")))
        return {"slices": np.asarray(h["SLICES"].data, float),
                "seps": np.asarray(h["SEPS"].data, float),
                "transmission": np.asarray(h["TRANS"].data, float),
                "unocculted": np.asarray(h["UNOCC"].data, float),
                "pxscale": float(hdr["PIXELSCL"]), "fwhm_px": float(hdr["FWHMPX"]),
                "ee_radius_px": float(hdr["EERADPX"]),
                "center": (float(hdr["SRCX"]), float(hdr["SRCY"])),
                "meta": json.loads(meta) if meta else {}, "path": path}


def throughput_fn(g: Dict[str, Any]) -> Callable[[Any], Any]:
    """Return a grid's throughput as a function of separation, linear between the separations and held outside.

    Parameters
    ----------
    g : dict
        The grid, from :func:`offaxis_grid`.

    Returns
    -------
    fn : callable
        Function ``rho_arcsec -> throughput``.
    """
    seps, t = np.asarray(g["seps"], float), np.asarray(g["transmission"], float)

    def f(rho):
        return np.interp(np.asarray(rho, float), seps, t, left=t[0], right=t[-1])
    return f


def library(g: Dict[str, Any], star_flux: float = 1.0, ee_radius_px: Optional[float] = None,
            refpa_deg: Optional[float] = None) -> LibraryPSF:
    """Build a :class:`~klip_tpe.injection.LibraryPSF` from an :func:`offaxis_grid`.

    The stamps are source-centered already, and the mask's throughput becomes
    ``LibraryPSF.throughput``.

    Parameters
    ----------
    g : dict
        The grid, from :func:`offaxis_grid`.
    star_flux : float, default 1.0
        The star's flux in the data's own units, measured in the same aperture ``ee_radius_px``, such as
        from the target-acquisition image or the reference star's photometry.  1 injects in units of the
        template.
    ee_radius_px : float or None, default None
        The normalization aperture, in pixels.  None takes the grid's.
    refpa_deg : float or None, default None
        Reference position angle of the stamps.  None translates each stamp to its source without
        turning it, and a number turns it by the source's azimuth, as runs made before 2026-09-22 did.

    Returns
    -------
    model : LibraryPSF
        The injection model.

    Notes
    -----
    Until 2026-09-22 ``refpa_deg`` was 0.0, which made the injector spin the stamp by the source's
    detector azimuth.  That is wrong, because the structure visible in a JWST coronagraphic PSF (the
    Lyot stop's pattern, the segmented pupil's lobes) is fixed to the spacecraft, not to where the
    source sits relative to the mask.  It does not turn as a companion moves round the field, so
    neither should the template, and the injected sources came out with their side lobes pointing the
    wrong way.  The IDL predecessor's ``reduce_nircam_v13.pro`` settles it: it reads one WebbPSF
    template per filter, centers it with ``cntrd`` and ``fshift`` (a translation), and at injection
    does ``fshift(big_ref * contrast, xshift, yshift)`` per frame and nothing else.
    """
    sl = np.asarray(g["slices"], float)
    c = tuple(g.get("center") or ((sl.shape[-1] - 1) / 2.0, (sl.shape[-2] - 1) / 2.0))
    rap = float(ee_radius_px if ee_radius_px is not None else g.get("ee_radius_px") or 3.0)
    return LibraryPSF(sl, g["seps"], center=c, ee_radius_px=rap,
                      throughput_fn=throughput_fn(g),
                      refpa_deg=None if refpa_deg is None else float(refpa_deg),
                      flux_unit=float(star_flux))


# ------------------------------------------------------------------ absolute flux scale
def unocculted_ee(radius_px: float, instrument: str = "NIRCam", filter: str = "F444W",
                  pupil_mask: Optional[str] = "MASKRND", fov_arcsec: float = 30.0,
                  oversample: int = 2, nlambda: int = 3, aperture: Optional[str] = None,
                  detector_position: Optional[Tuple[int, int]] = None, date: Optional[str] = None,
                  cache: bool = True, log: Callable[[str], None] = print) -> float:
    """Return the fraction of an unocculted point source's flux inside a radius.

    The source is unocculted through the Lyot stop, ``image_mask`` removed and ``pupil_mask`` kept,
    because that is the configuration the data's flux calibration refers to.  The coronagraphic
    ``PHOTOM`` reference files are derived from standards observed in this same mode, so the MJy/sr in
    the file already carries the Lyot stop and the mask substrate, and a model supplies only the PSF's
    shape and the occulter's transmission ``T(rho)``, which :func:`offaxis_grid` measures against this
    same reference.  An ordinary imaging PSF here would count the Lyot stop's effect on the PSF's shape
    twice.  ``fov_arcsec`` has to be large: the encircled energy of a NIRCam LW PSF is still climbing
    at 5 arcsec (0.96 at 5", 0.99 at 12.5" for F444W), so a stamp-sized field overstates the fraction
    by tens of percent, and the contrast axis inherits it.

    Parameters
    ----------
    radius_px : float
        The radius, in pixels.
    instrument : str, default "NIRCam"
        The instrument.
    filter : str, default "F444W"
        The filter.
    pupil_mask : str or None, default "MASKRND"
        The pupil stop.
    fov_arcsec : float, default 30.0
        Field of the computation, in arcsec.
    oversample : int, default 2
        STPSF's oversampling.
    nlambda : int, default 3
        Number of wavelengths over the band.
    aperture : str or None, default None
        The aperture's name.
    detector_position : tuple of 2 ints or None, default None
        The position on the detector, in pixels.
    date : str or None, default None
        Date of the measured wavefront to load.
    cache : bool, default True
        Whether to read and write the cache.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    fraction : float
        The encircled fraction.

    Raises
    ------
    RuntimeError
        If the PSF is not in the cache and STPSF is not installed, or the PSF has no flux.
    """
    from .instruments.generic import aperture_sum
    meta = {"kind": "unocculted_ee", "instrument": instrument, "filter": filter,
            "pupil_mask": pupil_mask, "fov_arcsec": float(fov_arcsec), "oversample": int(oversample),
            "nlambda": int(nlambda), "aperture": aperture, "detector_position": detector_position,
            "date": date, "version": _CACHE_VERSION}
    path = _cache_path(f"eeunocc_{instrument}_{filter}_{_key(meta)}.fits")
    if cache and os.path.exists(path):
        try:
            from astropy.io import fits
            with fits.open(path) as h:
                img = np.asarray(h[0].data, float)
                tot, px = float(h[0].header["PSFTOTAL"]), float(h[0].header["PIXELSCL"])
            c = ((img.shape[1] - 1) / 2.0, (img.shape[0] - 1) / 2.0)
            return float(aperture_sum(img, c[0], c[1], float(radius_px)) / tot)
        except Exception as exc:                                # pragma: no cover - corrupt cache
            log(f"  stpsf: ignoring unreadable EE cache {os.path.basename(path)} ({exc})")

    if not have_stpsf():
        # The whole point of the cache is that a machine without STPSF can still run from a
        # copied one -- the Mac this is developed against has Python 3.9, which STPSF will
        # never install on.  So a cache miss there is not "install STPSF", it is "you are
        # missing one file", and the error has to say WHICH.  Without this the failure was a
        # bare ImportError six frames down, on a run that had already loaded 26 files.
        raise RuntimeError(
            f"{f'the {instrument} {filter} unocculted PSF (for the encircled energy)'} is not in the cache and STPSF is not installed to compute it. Copy "
            f"{os.path.basename(path)} into {cache_dir()} from a machine that has STPSF "
            f"(Python >= 3.10), or install STPSF and its data files (STPSF_PATH).")
    inst = _instrument(instrument, filter, None, pupil_mask, aperture, detector_position, date)
    inst.image_mask = None                                      # the occulter, and only it, comes out
    p = inst.calc_psf(fov_arcsec=float(fov_arcsec), oversample=int(oversample), nlambda=int(nlambda))
    img = np.asarray(p["DET_SAMP"].data, float)
    tot = float(np.nansum(img))
    if not np.isfinite(tot) or tot <= 0:                        # pragma: no cover - defensive
        raise RuntimeError("the unocculted PSF has no flux")
    if cache:
        from astropy.io import fits
        hdr = fits.Header()
        hdr["PSFTOTAL"] = (tot, "sum over the computed field")
        hdr["PIXELSCL"] = (float(inst.pixelscale), "arcsec / detector pixel")
        hdr["KTVER"] = _CACHE_VERSION
        for i, line in enumerate(_chunk(json.dumps(meta), 60)):
            hdr[f"KTMETA{i:02d}"] = line
        fits.PrimaryHDU(np.asarray(img, np.float32), header=hdr).writeto(path, overwrite=True)
        log(f"  stpsf: cached unocculted PSF -> {os.path.basename(path)}")
    c = ((img.shape[1] - 1) / 2.0, (img.shape[0] - 1) / 2.0)
    return float(aperture_sum(img, c[0], c[1], float(radius_px)) / tot)


def star_flux_from_flux_density(g: Dict[str, Any], flux_density_jy: float, pixar_sr: float,
                                ee_radius_px: Optional[float] = None, bunit: str = "MJy/sr",
                                optics_transmission: float = 1.0,
                                log: Callable[[str], None] = print, **kw) -> float:
    """Return ``star_flux`` for :func:`library` from the star's flux density in the band.

    The JWST pipeline calibrates NRC_CORON data to surface brightness, so a point source of flux
    density ``S`` deposits ``S / (1e6 * PIXAR_SR)`` summed over its pixels, and only the fraction
    inside the injection model's normalization radius, from :func:`unocculted_ee`, belongs in
    ``flux_unit``.  The occulter is not applied here, since that is ``throughput(rho)``, and applying
    it twice is the classic way to get a contrast axis wrong by 1/T.

    Parameters
    ----------
    g : dict
        The grid, from :func:`offaxis_grid`.
    flux_density_jy : float
        The star's flux density in this filter, in Jy, from synthetic photometry of a stellar model or
        published photometry.
    pixar_sr : float
        The ``PIXAR_SR`` of the science frames.
    ee_radius_px : float or None, default None
        The normalization radius, in pixels.  None takes the grid's.
    bunit : str, default "MJy/sr"
        Unit of the data, which must be MJy/sr.
    optics_transmission : float, default 1.0
        Factor for a transmissive loss that neither the data's calibration nor the model carries.  For
        JWST calints it stays 1.
    log : callable, default print
        Function that takes each line of the log.
    kw : dict
        Keywords of :func:`unocculted_ee`, such as ``fov_arcsec`` and ``date``.

    Returns
    -------
    star_flux : float
        The flux unit, ``S / (1e6 PIXAR_SR) EE optics_transmission``.

    Raises
    ------
    ValueError
        If the data are not in MJy/sr.

    Notes
    -----
    STPSF's ``normalize='first'`` propagates only diffractive losses, so the model PSF misses the COM
    and Lyot substrates, but so does every calibration standard observed through them.  ``PHOTMJSR``
    for ``PUPIL=MASKRND`` (2.486 in F444W, against about 0.4 for CLEAR imaging) is derived from stars
    observed in this optical train, so the MJy/sr in the file already puts an off-mask point source at
    its true flux, and the EE is a fraction of the Lyot-stop PSF, in which the stop's own 0.18 cancels.

    A value of 0.561 was carried here for HIP 65426 until 2026-09-16, "anchored" on the companion.  It
    was not an optics number: the calints loader's sigma-clip repair had median-filtered the companion's
    core to 39% of its peak while the fakes, injected after the repair, kept theirs, and 0.561 = 1/1.9
    hid that.  With the repair fixed and 1.0 here, HIP 65426 b gives dF444W = 8.796 +/- 0.092 against
    Carter et al. (2023)'s 8.703 +/- 0.055, 1.0 sigma with nothing tuned
    (``scripts/check_hip65426_contrast.py``).  It read 8.74 until 2026-09-22, when :func:`library`
    stopped turning the template by the source's azimuth.
    """
    if str(bunit).replace(" ", "").upper() not in ("MJY/SR",):
        raise ValueError(f"star_flux_from_flux_density expects MJy/sr data, got {bunit!r}; "
                         "convert the frames or compute the flux unit for their own units")
    m = g.get("meta") or {}
    rap = float(ee_radius_px if ee_radius_px is not None else g.get("ee_radius_px") or 3.0)
    ee = unocculted_ee(rap, instrument=m.get("instrument", "NIRCam"), filter=m.get("filter", "F444W"),
                       pupil_mask=m.get("pupil_mask"), aperture=m.get("aperture"),
                       detector_position=m.get("detector_position"), date=m.get("date"),
                       oversample=int(m.get("oversample", 2)), nlambda=int(m.get("nlambda", 3)),
                       log=log, **kw)
    total = float(flux_density_jy) / (1e6 * float(pixar_sr))
    t = float(optics_transmission)
    log(f"  stpsf: S = {flux_density_jy:.4f} Jy -> {total:.4e} MJy/sr summed over the PSF; "
        f"EE({rap:.2f} px) = {ee:.4f}; optics transmission {t:.4f} "
        f"-> star_flux = {total * ee * t:.4e}")
    if t != 1.0:
        log(f"  stpsf: optics_transmission {t:.3f} != 1 -- PHOTMJSR of a JWST coronagraphic "
            "mode already carries its optics; only use this for data whose calibration "
            "demonstrably excludes part of the train (see the docstring)")
    return total * ee * t


# ------------------------------------------------------------------ from the data itself
def mode_from_header(hdr) -> Dict[str, Any]:
    """Read a JWST coronagraphic mode from a science header.

    Parameters
    ----------
    hdr : dict or astropy.io.fits.Header
        The header, with ``INSTRUME``, ``FILTER``, ``CORONMSK``, ``PUPIL`` and ``APERNAME``.

    Returns
    -------
    mode : dict
        ``instrument``, ``filter``, ``image_mask``, ``pupil_mask`` (the mask's Lyot stop where
        ``PUPIL`` holds a filter-wheel entry) and ``aperture``.
    """
    def g(*keys):
        for k in keys:
            v = hdr.get(k) if hasattr(hdr, "get") else None
            if v not in (None, "", "NONE", "N/A"):
                return str(v).strip()
        return None
    inst = (g("INSTRUME") or "NIRCAM").upper()
    inst = {"NIRCAM": "NIRCam", "MIRI": "MIRI", "NIRISS": "NIRISS"}.get(inst, inst.title())
    msk = g("CORONMSK", "IMAGEMSK")
    if msk:
        msk = msk.upper().replace("MASKA", "MASK").replace("MASKB", "MASK")
    pup = g("PUPIL")
    if pup and pup.upper() not in ("MASKRND", "MASKSWB", "MASKLWB", "MASKFQPM", "MASKLYOT",
                                   "CIRCLYOT", "WEDGELYOT"):
        pup = None                                   # PUPIL often carries the filter-wheel entry
    if pup is None and msk in DEFAULTS:
        pup = DEFAULTS[msk]["pupil_mask"]
    return {"instrument": inst, "filter": g("FILTER", "PUPIL"), "image_mask": msk,
            "pupil_mask": pup, "aperture": g("APERNAME")}


def model_for_datasets(dsets, star_flux: float = 1.0, seps_as: Optional[Sequence[float]] = None,
                       rho_max_as: Optional[float] = None, pxscale: Optional[float] = None,
                       log: Callable[[str], None] = print, **kw) -> LibraryPSF:
    """Build the STPSF injection model of a JWST sequence already loaded.

    It reads the coronagraphic mode from the first dataset's ``meta['header']``, or its
    ``meta['INSTRUME']``-style keys, builds the off-axis grid over the separations the data cover, and
    returns the :class:`~klip_tpe.injection.LibraryPSF`.  A MIRI coronagraph, whose throughput depends
    on where a source is rather than how far out, takes the MIRI module's library with a
    two-dimensional throughput map (:func:`klip_tpe.instruments.miri.library`).

    Parameters
    ----------
    dsets : Dataset or sequence of Dataset
        The datasets.
    star_flux : float, default 1.0
        The star's flux in the data's units (:func:`library`).
    seps_as : sequence of float or None, default None
        The separations, in arcsec.  None takes 0.15 to ``rho_max_as`` in steps of 0.15.
    rho_max_as : float or None, default None
        Largest separation, in arcsec.  None takes 0.45 of the frame's width.
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel.  None takes the dataset's.
    log : callable, default print
        Function that takes each line of the log.
    kw : dict
        Keywords of :func:`offaxis_grid`, such as ``stamp_px``, ``nlambda`` and ``date``, and overrides
        of the mode.

    Returns
    -------
    model : LibraryPSF
        The injection model.

    Raises
    ------
    ValueError
        If the header gives no filter and none is passed.
    """
    d0 = dsets[0] if isinstance(dsets, (list, tuple)) else dsets
    meta = getattr(d0, "meta", {}) or {}
    hdr = meta.get("header") or meta
    mode = mode_from_header(hdr)
    mode.update({k: kw.pop(k) for k in ("instrument", "filter", "image_mask", "pupil_mask",
                                        "aperture") if k in kw})
    if not mode.get("filter"):
        raise ValueError("no FILTER in the dataset header; pass filter= explicitly")
    px = float(pxscale or meta.get("pxscale") or 0.0)
    # MIRI's coronagraphs are four-quadrant phase masks (and one Lyot with a support bar),
    # whose throughput depends on where a source is rather than how far out -- by a factor
    # of two to four around a circle of constant separation.  offaxis_grid measures it
    # radially, so on MIRI it returns an azimuthal average of two different things.  The
    # MIRI module builds the same stamp library with a two-dimensional throughput map.
    # Imported here, not at module scope: klip_tpe.instruments.miri imports from this one.
    if str(mode.get("instrument", "")).upper() == "MIRI" and mode.get("image_mask"):
        from .instruments.miri import MODES as MIRI_MODES, library as miri_library
        if str(mode["filter"]).upper() in MIRI_MODES:
            log(f"  stpsf: mode MIRI {mode['filter']} {mode.get('image_mask')} "
                f"-- 2-D throughput (four-quadrant mask)")
            return miri_library(filter=mode["filter"], star_flux=star_flux,
                                seps_as=seps_as, log=log,
                                **{k: v for k, v in kw.items()
                                   if k in ("date", "stamp_px", "nlambda", "oversample")})
    if seps_as is None:
        ny = int(np.shape(getattr(d0, "cube"))[-1])
        rmax = float(rho_max_as) if rho_max_as else (0.45 * ny * px if px > 0 else 2.6)
        seps_as = np.arange(0.15, max(rmax, 0.45) + 1e-6, 0.15)
    log(f"  stpsf: mode {mode['instrument']} {mode['filter']} {mode.get('image_mask')}")
    g = offaxis_grid(seps_as=seps_as, log=log, **{**mode, **kw})
    return library(g, star_flux=star_flux)
