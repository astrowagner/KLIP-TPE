"""STPSF (formerly WebbPSF) point-spread functions for the JWST reductions.

What this gives the optimizer, for a JWST coronagraphic mode:

* a **separation-dependent injection model** -- the off-axis PSF of the actual mask,
  which is neither a Gaussian nor separation-independent inside a few lambda/D of a
  coronagraph -- as a :class:`~klip_tpe.injection.LibraryPSF`;
* the **mask throughput** ``T(rho)`` measured from the same grid, so injected
  contrasts and recovered contrasts are on the instrument's own scale;
* the **matched filter** used by the detection metric, taken from the same PSFs and,
  with :mod:`klip_tpe.fmmf`, forward-modelled through the KLIP subtraction.

Usage::

    from klip_tpe import stpsf_psf
    grid  = stpsf_psf.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R",
                                   pupil_mask="MASKRND", seps_as=np.arange(0.2, 2.6, 0.2))
    model = stpsf_psf.library(grid, star_flux=F_star)      # -> LibraryPSF
    red   = spaceklip.make_reducer(dsets, injection_model=model, ...)

or, straight off the data,
``model = stpsf_psf.model_for_datasets(dsets, star_flux=F_star)``.

Each slice is an off-axis PSF **already centred on the source** and cropped to
``stamp_px``, which is the convention :class:`~klip_tpe.injection.LibraryPSF` expects;
``transmission[i]`` is the fraction of the unocculted aperture flux that survives the
mask and Lyot stop at that separation.  Every PSF costs seconds to compute, so a grid is
cached as one FITS per configuration under ``$KLIP_TPE_DATA/stpsf_cache`` (default
``~/.klip_tpe/stpsf_cache``).

STPSF itself (``pip install stpsf``, Python >= 3.10) and its ~90 MB data files
(``STPSF_PATH``) are an optional dependency: everything here raises a clear error when
they are missing, and no other klip-tpe module imports this one at import time.
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
#: against the computed PSFs; keeping the source on +x is what makes ``refpa_deg=0`` the
#: right anisotropy reference for :class:`~klip_tpe.injection.LibraryPSF`.
_THETA_PLUS_X = 270.0


def have_stpsf() -> bool:
    """True when ``stpsf`` imports *and* its data files are reachable.

    Ask STPSF where its data is rather than reading ``STPSF_PATH`` out of the environment.
    The environment variable is only one of the ways STPSF finds its data -- a conda
    install or an ``~/.stpsf`` config file sets it up without ever exporting one -- so
    testing the variable reports "no STPSF" on a machine where STPSF works perfectly well,
    and the caller then falls back to a documented constant or refuses to start.  The
    variables stay as the fallback for an older STPSF without the accessor.
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
    """Where NEW cache files are written: ``$KLIP_TPE_DATA/stpsf_cache`` when the variable
    is set, else ``~/.klip_tpe/stpsf_cache``.  Reads go through :func:`_cache_path`, which
    also looks in the other layout, so a grid computed in one shell is found from another."""
    d = os.environ.get("KLIP_TPE_DATA") or os.path.join(os.path.expanduser("~"), ".klip_tpe")
    d = os.path.join(d, "stpsf_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_dirs() -> list:
    """Every directory a cached PSF may be READ from, in order of preference.

    ``cache_dir()`` depends on ``KLIP_TPE_DATA``: set to ``~/.klip_tpe/data`` (a common way
    to point the package at its own default data directory explicitly) it puts the cache at
    ``~/.klip_tpe/data/stpsf_cache``; unset, at ``~/.klip_tpe/stpsf_cache``.  A grid computed
    under one setting was invisible under the other, and on a machine without STPSF that is
    fatal: paper run D refused to start on 2026-09-16 with "the grid is not in the cache"
    while the grid sat one directory over.  So a read tries both layouts.
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
    """The existing cache file ``name`` wherever it is, else its path in :func:`cache_dir`."""
    for d in _cache_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(cache_dir(), name)


def _key(meta: Dict[str, Any]) -> str:
    s = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _odd(n: float) -> int:
    n = int(np.ceil(n))
    return n + 1 if n % 2 == 0 else n


def _instrument(name: str, filter_: str, image_mask: Optional[str], pupil_mask: Optional[str],
                aperture: Optional[str], detector_position: Optional[Tuple[int, int]],
                date: Optional[str]):
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
    """``scale`` x the encircled-energy-``frac`` radius of a centred PSF, in pixels."""
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
    img = np.asarray(img, float)
    cy, cx = (img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float(np.nansum(img[np.hypot(xx - cx, yy - cy) <= r]))


def _source_center(img: np.ndarray, guess: Tuple[float, float], win: float = 5.0) -> Tuple[float, float]:
    """Flux-weighted centre of the (single) source in ``img``, refined from ``guess``.

    ``calc_psf`` with a ``source_offset`` computes one off-axis point source and nothing
    else, so the brightest structure *is* the source; measuring it beats trusting a
    convention for where STPSF puts the array centre after detector binning."""
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
    """Cut a ``stamp`` x ``stamp`` box around ``src_xy`` and shift the sub-pixel
    remainder, so the source lands exactly on the central pixel of the stamp."""
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
    """Off-axis PSF of a coronagraphic mode on a ladder of separations.

    Returns a dict with

    ``slices``      ``(nsep, stamp_px, stamp_px)`` source-centred off-axis PSFs;
    ``seps``        the separations in arcsec;
    ``transmission``the coronagraph throughput at each separation (aperture photometry
                    of the occulted PSF over the same aperture on the unocculted one);
    ``unocculted``  the same-size stamp of the unocculted PSF;
    ``pxscale``     arcsec/pixel of the detector sampling;
    ``fwhm_px``     FWHM of the unocculted PSF in pixels;
    ``ee_radius_px``the aperture ``transmission`` was measured in;
    ``center``      ``(cx, cy)`` of the source inside a stamp -- the stamp centre;
    ``meta``        the configuration, for provenance.

    ``pupil_mask`` defaults to the Lyot stop that goes with ``image_mask``
    (:data:`DEFAULTS`).  ``fov_px`` is sized per separation so that the source is always
    inside the computed field; leave it None.  Results are cached by configuration under
    :func:`cache_dir` (``cache=False`` recomputes and rewrites).
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
    """FWHM in pixels of a centred PSF, from the radius where the azimuthal profile
    first falls below half the peak (linearly interpolated)."""
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
    return [s[i:i + n] for i in range(0, len(s), n)] or [""]


def _read_cache(path: str) -> Dict[str, Any]:
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
    """``rho_arcsec -> coronagraph throughput`` (linear in separation, clamped outside)."""
    seps, t = np.asarray(g["seps"], float), np.asarray(g["transmission"], float)

    def f(rho):
        return np.interp(np.asarray(rho, float), seps, t, left=t[0], right=t[-1])
    return f


def library(g: Dict[str, Any], star_flux: float = 1.0, ee_radius_px: Optional[float] = None,
            refpa_deg: float = 0.0) -> LibraryPSF:
    """:class:`~klip_tpe.injection.LibraryPSF` from an :func:`offaxis_grid`.

    The stamps are source-centred already, the mask throughput becomes
    ``LibraryPSF.throughput``, and ``flux_unit = star_flux`` is the star's flux in the
    data's own units (e.g. from the target-acquisition image or the reference star's
    photometry, measured in the *same* aperture ``ee_radius_px``; leave 1.0 to inject in
    units of the template).  ``refpa_deg=0`` because each slice was computed with the
    source along +x of the stamp, so the injector rotates it to the source's azimuth.
    """
    sl = np.asarray(g["slices"], float)
    c = tuple(g.get("center") or ((sl.shape[-1] - 1) / 2.0, (sl.shape[-2] - 1) / 2.0))
    rap = float(ee_radius_px if ee_radius_px is not None else g.get("ee_radius_px") or 3.0)
    return LibraryPSF(sl, g["seps"], center=c, ee_radius_px=rap,
                      throughput_fn=throughput_fn(g), refpa_deg=float(refpa_deg),
                      flux_unit=float(star_flux))


# ------------------------------------------------------------------ absolute flux scale
def unocculted_ee(radius_px: float, instrument: str = "NIRCam", filter: str = "F444W",
                  pupil_mask: Optional[str] = "MASKRND", fov_arcsec: float = 30.0,
                  oversample: int = 2, nlambda: int = 3, aperture: Optional[str] = None,
                  detector_position: Optional[Tuple[int, int]] = None, date: Optional[str] = None,
                  cache: bool = True, log: Callable[[str], None] = print) -> float:
    """Fraction of an **unocculted** point source's flux that lands inside ``radius_px``.

    Unocculted *through the Lyot stop* -- ``image_mask`` removed, ``pupil_mask`` kept --
    because that is the configuration the data's own flux calibration refers to.  The
    coronagraphic ``PHOTOM`` reference files are derived from standards observed in this
    same mode, so the MJy/sr in the file already carries the Lyot stop and the mask
    substrate; the only things left for a model to supply are the PSF's *shape* and the
    occulter's spatial transmission ``T(rho)``, which :func:`offaxis_grid` measures
    separately against this same reference.  Feeding an ordinary imaging PSF in here
    instead would count the Lyot stop's effect on the PSF shape twice.

    ``fov_arcsec`` has to be big: the encircled energy of a NIRCam LW PSF is still climbing
    at 5 arcsec (0.96 at 5", 0.99 at 12.5" for F444W), so a stamp-sized field overstates
    the fraction by tens of percent and the contrast axis inherits it.
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
    """``star_flux`` for :func:`library` from the star's flux density in the band.

    ``flux_density_jy`` is the star's flux density [Jy] in this filter -- synthetic
    photometry of a stellar model, or published photometry -- and ``pixar_sr`` the
    ``PIXAR_SR`` of the science frames.  The JWST pipeline calibrates NRC_CORON data to
    surface brightness, so a point source of flux density ``S`` deposits
    ``S / (1e6 * PIXAR_SR)`` summed over its pixels; only the fraction inside the injection
    model's normalisation radius belongs in ``flux_unit``, and :func:`unocculted_ee`
    supplies it.  The occulter is *not* applied here -- that is ``throughput(rho)``, and
    applying it twice is the classic way to get a contrast axis wrong by 1/T.

    ``optics_transmission`` scales the result for any transmissive loss that neither the
    data's calibration nor the model carries.  **For JWST calints, leave it at 1.0.**
    STPSF's ``normalize='first'`` propagates only diffractive losses, so the model PSF is
    missing the COM substrate and the Lyot substrate -- but so is every calibration
    standard that was observed through them: ``PHOTMJSR`` for ``PUPIL=MASKRND`` (2.486 in
    F444W, against ~0.4 for CLEAR imaging) is derived from stars observed in this very
    optical train, so the MJy/sr in the file already put an off-mask point source at its
    true flux, and the ``EE`` here is a *fraction* of the Lyot-stop PSF, in which the
    stop's own 0.18 cancels.  The occulter's ``T(rho)`` is the one thing the photom file
    cannot hold (it varies across the field), and it comes in through the injection
    model's ``throughput``.

    A value of 0.561 was carried here for HIP 65426 until 2026-09-16, "anchored" on the
    companion.  It was not an optics number: the calints loader's sigma-clip repair had
    median-filtered the companion's core to 39% of its peak while the fakes, injected
    after the repair, kept theirs; 0.561 = 1/1.9 was what hid that.  With the repair fixed
    and 1.0 here, HIP 65426 b gives dF444W = 8.74 +/- 0.09 against Carter et al. (2023)'s
    8.703 +/- 0.055 with nothing tuned (``scripts/check_hip65426_contrast.py``).  Use a
    value other than 1.0 only for data whose flux calibration demonstrably excludes part
    of the optical train, and say where the number comes from.

    Extra keywords go to :func:`unocculted_ee` (``fov_arcsec``, ``date``, ...).
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
    """``{'instrument', 'filter', 'image_mask', 'pupil_mask', 'aperture'}`` from a JWST
    science header (``INSTRUME``/``FILTER``/``CORONMSK``/``PUPIL``/``APERNAME``)."""
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
    """The STPSF injection model for an already-loaded JWST sequence.

    Reads the coronagraphic mode out of the first dataset's ``meta['header']`` (or
    ``meta['INSTRUME']``-style keys), builds the off-axis grid over the separations the
    data actually cover, and returns the :class:`LibraryPSF`.  Extra keyword arguments go
    to :func:`offaxis_grid` (``stamp_px``, ``nlambda``, ``date``, ...).
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
