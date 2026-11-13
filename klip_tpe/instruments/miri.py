"""JWST/MIRI coronagraphy, the four-quadrant phase masks and the Lyot spot.

Three of MIRI's four coronagraphs are four-quadrant phase masks (4QPM), which suppress the stellar
PSF with a pi phase step across two perpendicular lines through the star rather than by blocking a
disk.  Everything that assumes a round occulter is wrong here, above all the throughput model, so
this module carries a two-dimensional throughput map on a (separation, detector azimuth) grid,
cached like the radial grids.

Notes
-----
Measured from STPSF for F1065C/FQPM1065 (:func:`throughput_map`), the extremes over azimuth at each
separation are::

     rho      min    max   ratio
    0.30"    0.176  0.421   x2.4
    0.58"    0.142  0.593   x4.2
    1.10"    0.159  1.007   x6.3
    1.53"    0.156  1.003   x6.5
    2.93"    0.170  0.998   x5.9
    5.63"    0.233  0.999   x4.3
   10.80"    0.406  0.991   x2.4

The throughput varies by up to a factor of 6.5 at constant separation.  The plateau between the
boundaries sits at 1.00 within half a percent, since the unocculted reference carries the same Lyot
stop.  The minimum stays near 0.15 out to 5 arcsec, so the ratio falls only because the plateau is
already at 1.  At 2 arcsec, ``|T(az) - T(az+180)| <= 0.003`` while ``|T(az) - T(az+90)|`` reaches
0.18, a two-fold symmetry, which is why the map covers the whole circle.

A throughput of separation alone, such as ``klip_tpe.stpsf_psf.offaxis_grid``'s, is wrong on a 4QPM
in a way that tracks position angle.  A companion near a quadrant boundary is reported up to three
times fainter than it is, and one between boundaries too bright.  Three more consequences of the
geometry are handled here:

* The boundaries sit at az = 356, 86, 176 and 266, four degrees off the detector axes on all four,
  the mask's mounting angle.  :func:`locate_boundaries` measures them and :func:`quadrant_mask`
  thresholds the measured map, so both follow the instrument model's rotation.
* The boundaries are not usable, since the mask distorts the stamp there as well as attenuating it.
  :func:`quadrant_mask` marks those pixels on the detector, :func:`dead_zone_pixel_mask` carries
  them through the rolls to keep them out of the noise statistics, and :func:`forbidden_pa` keeps
  the sampler from injecting into them.  They stay in the cube and the KLIP basis, since NaN ahead
  of the high-pass destroys the whole frame (:func:`apply_quadrant_mask`).
* Throughput is a per-frame quantity.  The boundaries are fixed to the detector and a companion at
  a fixed sky PA crosses them as the telescope rolls, which is why :func:`throughput_map_fn` takes
  a detector azimuth.

The stamp library stays radial, and only the scalar throughput is two-dimensional.  Near a boundary
the PSF is distorted as well as attenuated, which is survivable only because :func:`quadrant_mask`
removes that region from the analysis.  A full (separation, azimuth) stamp library would cost
``n_azimuth`` times more PSFs.

The instrument numbers come from Boccaletti et al. 2022 (A&A 667, A165): the 110 mas pixel scale,
the filters' central wavelengths and the 2.16 arcsec Lyot spot radius.  The code takes the pixel
scale from the STPSF instrument model itself, 0.109655 arcsec per pixel.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..injection import LibraryPSF
from ..stpsf_psf import (_cache_path, _ee_radius, _instrument, _key, _odd, cache_dir,
                         have_stpsf, offaxis_grid)

__all__ = ["MODES", "DIAMETER_M", "mode_for_filter", "pixelscale", "throughput_map",
           "default_azimuths", "default_separations", "locate_boundaries",
           "throughput_map_fn", "quadrant_mask", "library", "load_miri", "apply_quadrant_mask",
           "forbidden_pa", "dead_zone_pixel_mask", "star_flux_from_flux_density",
           "MIRILibraryPSF"]

DIAMETER_M = 6.5

# Central wavelengths and bandwidths from Boccaletti et al. 2022, Table 1.  ``kind`` is
# what the focal-plane mask actually is, which is what decides the throughput geometry:
# a 4QPM has two suppression lines through the star, a Lyot has a disk and a support bar.
MODES: Dict[str, Dict[str, Any]] = {
    "F1065C": dict(image_mask="FQPM1065", pupil_mask="MASKFQPM", kind="4qpm",
                   lam_m=10.575e-6, dlam_m=0.75e-6, fov_as=24.0),
    "F1140C": dict(image_mask="FQPM1140", pupil_mask="MASKFQPM", kind="4qpm",
                   lam_m=11.30e-6, dlam_m=0.80e-6, fov_as=24.0),
    "F1550C": dict(image_mask="FQPM1550", pupil_mask="MASKFQPM", kind="4qpm",
                   lam_m=15.50e-6, dlam_m=0.90e-6, fov_as=24.0),
    "F2300C": dict(image_mask="LYOT2300", pupil_mask="MASKLYOT", kind="lyot",
                   lam_m=22.75e-6, dlam_m=5.50e-6, fov_as=30.0, spot_radius_as=2.16),
}

_CACHE_VERSION = 1


def mode_for_filter(filter: str) -> Dict[str, Any]:
    """Return the coronagraphic mode a MIRI filter names, with its masks and wavelength.

    Parameters
    ----------
    filter : str
        The filter, such as "F1065C", in any case.

    Returns
    -------
    mode : dict
        ``image_mask``, ``pupil_mask``, ``kind`` ("4qpm" or "lyot"), ``lam_m``, ``dlam_m``,
        ``fov_as``, ``filter`` and, for the Lyot, ``spot_radius_as``.

    Raises
    ------
    ValueError
        If the filter is not a MIRI coronagraphic one.
    """
    f = str(filter).upper().strip()
    if f not in MODES:
        raise ValueError(f"{filter!r} is not a MIRI coronagraphic filter; expected one of "
                         f"{sorted(MODES)}")
    return dict(MODES[f], filter=f)


def pixelscale(filter: str = "F1065C") -> float:
    """Return MIRI's pixel scale from the STPSF instrument model.

    The documentation rounds it to 110 mas, the model carries 0.109655, and a 0.3% scale error is a
    third of a pixel across a 24 arcsec field.

    Parameters
    ----------
    filter : str, default "F1065C"
        The coronagraphic filter.

    Returns
    -------
    pxscale : float
        The pixel scale, in arcsec per pixel, or the documented 0.110 when STPSF is not installed
        (:func:`~klip_tpe.stpsf_psf.have_stpsf` tells which).
    """
    if not have_stpsf():
        return 0.110
    m = mode_for_filter(filter)
    inst = _instrument("MIRI", m["filter"], m["image_mask"], m["pupil_mask"], None, None, None)
    return float(inst.pixelscale)


# --------------------------------------------------------------------------- throughput
def _map_cache_name(meta: Dict[str, Any]) -> str:
    """Return the cache file name of a throughput map.

    Parameters
    ----------
    meta : dict
        The map's configuration.

    Returns
    -------
    name : str
        The file name.
    """
    return f"miri_thrumap_{meta['filter']}_{_key(meta)}.npz"


# Offsets from a quadrant boundary, in degrees.  Non-uniform on purpose: the throughput
# changes by a factor of four over the few degrees either side of a boundary and is flat
# between them, so a uniform grid either cannot resolve the boundary or wastes most of its
# PSFs on the flat part.  Sampled every 30 degrees the dead zone came out 0.2% of the
# field -- the interpolant simply stepped over it -- while the mask it is meant to define
# is the one region of the image where the photometry cannot be trusted.
_BOUNDARY_OFFSETS = (0.0, 2.5, 6.0, 12.0, 22.0, 34.0, 45.0)


def default_azimuths(boundaries: Sequence[float] = (0.0, 90.0, 180.0, 270.0)) -> np.ndarray:
    """Return detector azimuths for :func:`throughput_map`, dense on the quadrant boundaries.

    The grid mirrors :data:`_BOUNDARY_OFFSETS` about each boundary, and points shared between
    neighboring sectors collapse, so four boundaries give 48 azimuths rather than 56.

    Parameters
    ----------
    boundaries : sequence of float, default (0.0, 90.0, 180.0, 270.0)
        Azimuths of the boundaries, in degrees.  The default is the detector axes, where the
        boundaries are not, so pass the angles :func:`locate_boundaries` measured.

    Returns
    -------
    az : ndarray
        The azimuths, in degrees, sorted.
    """
    az = {(b + s * o) % 360.0 for b in np.asarray(boundaries, float).ravel()
          for o in _BOUNDARY_OFFSETS for s in (1, -1)}
    return np.array(sorted(az), float)


def default_separations(filter: str = "F1065C") -> np.ndarray:
    """Return separations for :func:`throughput_map`, out to the edge of the coronagraphic field.

    The ladder is geometric, 12 steps from 0.3 arcsec to 0.45 times the field of view.  The throughput
    climbs steeply over the first arcsecond and is flat past a few, and a grid that stops short is
    worse than a coarse one, since the interpolator clamps outside its range.

    Parameters
    ----------
    filter : str, default "F1065C"
        The coronagraphic filter, whose field sets the outer separation.

    Returns
    -------
    seps : ndarray
        The separations, in arcsec, rounded to the milliarcsecond.
    """
    m = mode_for_filter(filter)
    r_max = 0.45 * float(m["fov_as"])                     # corner-to-corner is further
    n = 12
    return np.round(np.geomspace(0.3, r_max, n), 3)


def locate_boundaries(inst, rho_as: float, fov: float, oversample: int, nlambda: int,
                      rap_px: float, step: float = 2.0, window: float = 14.0,
                      log: Callable[[str], None] = print) -> np.ndarray:
    """Measure where the four suppression lines are, in detector azimuth.

    It scans ``+/- window`` about each nominal axis at ``step``, finer and cheaper than a sweep of the
    whole circle, which at 15 degree steps reported the boundaries exactly on the detector axes.  A
    minimum at the edge of the window means the premise is wrong rather than the mask, so the window
    widens once, by 2.5, and says so.

    Parameters
    ----------
    inst : object
        The STPSF MIRI instrument with the coronagraph in place.
    rho_as : float
        Separation of the scan, in arcsec.
    fov : float
        Field of view of the PSFs, in arcsec.
    oversample : int
        Oversampling of the PSFs.
    nlambda : int
        Number of wavelengths of the PSFs.
    rap_px : float
        Radius of the photometric aperture, in pixels.
    step : float, default 2.0
        Step of the scan, in degrees.
    window : float, default 14.0
        Half width of the scan about each axis, in degrees, which bounds how far the mask may be
        rotated.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    boundaries : ndarray, shape (4,)
        The four azimuths, in degrees, in [0, 360).
    """
    nominal = np.array([0.0, 90.0, 180.0, 270.0])
    out = []
    for b in nominal:
        w = float(window)
        for attempt in (0, 1):
            a = b + np.arange(-w, w + 1e-9, float(step))
            t = np.array([_offset_sum(inst, rho_as, float(v), fov, oversample, nlambda, rap_px)
                          for v in a])
            i = int(np.argmin(t))
            if i not in (0, t.size - 1) or attempt:
                break
            log(f"    boundary near az={b:.0f} minimises at the edge of +/-{w:.0f} deg -- "
                f"widening once; a mask rotated this far is not what this scan assumes")
            w *= 2.5
        out.append(float(a[i] % 360.0))
    out = np.array(out, float)
    off = ((out - nominal + 180.0) % 360.0) - 180.0
    log(f"    quadrant boundaries at az = {np.round(out, 1).tolist()} deg "
        f"(offset from the detector axes: {np.round(off, 1).tolist()})")
    if np.ptp(off) > 4.0:
        log(f"    note: the four offsets disagree by {np.ptp(off):.1f} deg; a mounting "
            f"angle should be the same on all four")
    return out


def throughput_map(filter: str = "F1065C", seps_as: Optional[Sequence[float]] = None,
                   az_deg: Optional[Sequence[float]] = None, oversample: int = 2,
                   nlambda: int = 1, date: Optional[str] = None, cache: bool = True,
                   log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Compute the coronagraph's throughput on a (separation, detector azimuth) grid.

    Each grid node takes one STPSF PSF, summed over an aperture and divided by the unocculted PSF of
    the same configuration summed over the same aperture.  The map is never folded into one quadrant,
    since the four-fold symmetry is only nominal (it repeats under 180 degrees to three decimals but
    not under 90).  Each separation row is saved as it completes, so a restart picks up where an
    interrupted map stopped.

    Parameters
    ----------
    filter : str, default "F1065C"
        The coronagraphic filter.
    seps_as : sequence of float or None, default None
        Separations, in arcsec, at least two.  None takes :func:`default_separations`.
    az_deg : sequence of float or None, default None
        Azimuths in the detector frame, counter-clockwise from +x, in degrees, at least two.  None
        locates the boundaries first (:func:`locate_boundaries`) and samples densely about them
        (:func:`default_azimuths`).
    oversample : int, default 2
        Oversampling of the PSFs.
    nlambda : int, default 1
        Number of wavelengths, 1 since the chromatic terms of a ratio of two PSFs in one filter
        largely cancel.
    date : str or None, default None
        Date of the observation, for STPSF's optical model.
    cache : bool, default True
        Whether to read and write the cache under :func:`~klip_tpe.stpsf_psf.cache_dir`.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    map : dict
        ``seps``, ``az``, ``trans`` of shape (nsep, naz), ``boundaries``, ``pxscale``, ``fwhm_px``,
        ``ee_radius_px``, ``meta`` and ``path``.

    Raises
    ------
    ValueError
        If fewer than two separations or azimuths are given.
    RuntimeError
        If the map is not cached and STPSF is not installed.
    """
    m = mode_for_filter(filter)
    seps = np.asarray(list(seps_as) if seps_as is not None else
                      default_separations(m["filter"]), float)
    seps = np.sort(seps[np.isfinite(seps) & (seps > 0)])
    if seps.size < 2:
        raise ValueError("throughput_map needs at least two separations")
    az = None if az_deg is None else np.asarray(list(az_deg), float)
    if az is not None and az.size < 2:
        raise ValueError("throughput_map needs at least two azimuths")
    # "auto" rather than the resolved angles, because the angles are a measurement: the
    # cache key has to describe the request, or the first call could never hit the cache.
    meta = {"filter": m["filter"], "image_mask": m["image_mask"], "pupil_mask": m["pupil_mask"],
            "oversample": int(oversample), "nlambda": int(nlambda), "date": date,
            "seps": [round(float(s), 4) for s in seps],
            "az": "auto" if az is None else [round(float(a), 3) for a in az],
            "version": _CACHE_VERSION}
    path = _cache_path(_map_cache_name(meta))
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == _key(meta):
            log(f"  miri: throughput map from cache ({os.path.basename(path)})")
            return {"seps": z["seps"], "az": z["az"], "trans": z["trans"],
                    "boundaries": z["boundaries"] if "boundaries" in z.files else None,
                    "pxscale": float(z["pxscale"]), "fwhm_px": float(z["fwhm_px"]),
                    "ee_radius_px": float(z["ee_radius_px"]), "meta": meta, "path": path}
    if not have_stpsf():
        raise RuntimeError(f"the MIRI throughput map for {m['filter']} is not in the cache "
                           f"and STPSF is not installed to compute it "
                           f"(looked for {os.path.basename(path)})")

    fov = float(2.4 * (seps.max() + 0.6))

    # unocculted reference: same filter and Lyot stop, focal-plane mask removed
    ref_inst = _instrument("MIRI", m["filter"], None, m["pupil_mask"], None, None, date)
    ref = ref_inst.calc_psf(fov_arcsec=fov, oversample=oversample, nlambda=max(nlambda, 3))
    ref_img = np.asarray(ref[0].data, float)
    px = float(ref[0].header.get("PIXELSCL", ref_inst.pixelscale))
    rap = _ee_radius(ref_img)
    fwhm_px = _fwhm(ref_img)

    inst = _instrument("MIRI", m["filter"], m["image_mask"], m["pupil_mask"], None, None, date)
    # Same nlambda as the occulted PSFs: a monochromatic PSF concentrates more flux inside
    # a fixed aperture than a polychromatic one, so mixing the two inflates every ratio.
    ref_sum = _offset_sum(ref_inst, seps[0], 0.0, fov, oversample, nlambda, rap)

    # A full map is 576 PSFs, twenty-five minutes, and an all-or-nothing loop: a process
    # killed at PSF 570 leaves nothing behind.  That happened twice here on a machine that
    # does not keep long jobs alive, each time discarding most of a map.  So each
    # separation row is written as it completes and a restart picks up from the first row
    # not yet done.  Partial, because a half-finished map must never be loadable as a
    # finished one -- it is a separate file that only this function knows how to read.
    part = os.path.join(cache_dir(), _map_cache_name(meta)[:-4] + ".partial.npz")
    bounds, trans, done = None, None, 0
    if cache and os.path.exists(part):
        try:
            z = np.load(part, allow_pickle=False)
            if str(z["key"]) == _key(meta) and int(z["done"]) < seps.size:
                az, trans, done = z["az"], z["trans"], int(z["done"])
                bounds = z["boundaries"] if "boundaries" in z.files else None
                log(f"  miri: resuming a partial map at separation {done + 1}/{seps.size}")
        except Exception as exc:                       # a corrupt partial is not fatal
            log(f"  miri: ignoring an unreadable partial map ({exc!r})")
    if az is None:
        log(f"  miri: locating the quadrant boundaries for {m['filter']}/{m['image_mask']}")
        bounds = locate_boundaries(inst, float(np.median(seps)), fov, oversample, nlambda,
                                   rap, log=log)
        az = default_azimuths(bounds)
    if trans is None:
        trans = np.zeros((seps.size, np.asarray(az).size), float)
    log(f"  miri: computing {seps.size - done} x {np.asarray(az).size} = "
        f"{(seps.size - done) * np.asarray(az).size} off-axis PSFs (cached afterwards)")
    for i, r in enumerate(seps):
        if i < done:
            continue
        for j, a in enumerate(az):
            trans[i, j] = _offset_sum(inst, float(r), float(a), fov, oversample, nlambda,
                                      rap) / max(ref_sum, 1e-30)
        log(f"    rho={r:.2f}\"  throughput {trans[i].min():.3f}-{trans[i].max():.3f} "
            f"(x{trans[i].max() / max(trans[i].min(), 1e-6):.1f} across azimuth)")
        if cache:
            os.makedirs(cache_dir(), exist_ok=True)
            np.savez_compressed(part, seps=seps, az=az, trans=trans, done=i + 1,
                                key=_key(meta),
                                **({} if bounds is None else {"boundaries": bounds}))
    out = {"seps": seps, "az": az, "trans": trans, "boundaries": bounds, "pxscale": px,
           "fwhm_px": fwhm_px, "ee_radius_px": rap, "meta": meta, "path": path}
    if cache:
        os.makedirs(cache_dir(), exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir(), _map_cache_name(meta)),
                            seps=seps, az=az, trans=trans, pxscale=px, fwhm_px=fwhm_px,
                            ee_radius_px=rap, key=_key(meta), meta=json.dumps(meta),
                            **({} if bounds is None else {"boundaries": bounds}))
        # The finished map exists now, so the partial is only a way to load a half-built
        # one by mistake.
        try:
            os.remove(part)
        except OSError:
            pass
    return out


def _fwhm(img: np.ndarray) -> float:
    """Estimate the FWHM of a centered PSF from the pixels above half its peak.

    Parameters
    ----------
    img : ndarray
        The PSF.

    Returns
    -------
    fwhm : float
        ``2 sqrt(n / pi)`` for the n pixels at or above half the peak, in pixels, or NaN for a peak
        that is not positive.
    """
    img = np.asarray(img, float)
    pk = float(np.nanmax(img))
    if not np.isfinite(pk) or pk <= 0:
        return float("nan")
    return float(2.0 * np.sqrt(max(np.sum(img >= 0.5 * pk), 1) / np.pi))


def _offset_sum(inst, rho_as: float, az_deg: float, fov: float, oversample: int,
                nlambda: int, rap_px: float) -> float:
    """Sum the PSF of an offset source in an aperture centered on it.

    Parameters
    ----------
    inst : object
        The STPSF instrument, whose source offset is set.
    rho_as : float
        Separation of the source, in arcsec.
    az_deg : float
        Detector azimuth of the source, counter-clockwise from +x, in degrees.
    fov : float
        Field of view of the PSF, in arcsec.
    oversample : int
        Oversampling of the PSF.
    nlambda : int
        Number of wavelengths.
    rap_px : float
        Radius of the aperture, in pixels.

    Returns
    -------
    total : float
        The PSF's sum in the aperture.

    Notes
    -----
    STPSF's ``source_offset_theta`` runs from +Y towards -X, so ``theta = az - 90``, verified on the
    PSF: at ``theta`` 0, 45, 90, 180 and 270 the peak lands at az 90, 135, 180, 270 and 0.  The other
    sign puts the aperture where the source is not, except at 90 and 270, and the map comes back with
    two bright lobes that look like a mask feature.
    """
    inst.options["source_offset_r"] = float(rho_as)
    inst.options["source_offset_theta"] = float(az_deg - 90.0)
    h = inst.calc_psf(fov_arcsec=fov, oversample=oversample, nlambda=nlambda)
    img = np.asarray(h[0].data, float)
    ny, nx = img.shape
    px = float(h[0].header.get("PIXELSCL", inst.pixelscale))
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    phi = np.deg2rad(az_deg)
    sx, sy = cx + (rho_as / px) * np.cos(phi), cy + (rho_as / px) * np.sin(phi)
    yy, xx = np.mgrid[0:ny, 0:nx]
    return float(np.nansum(img[np.hypot(xx - sx, yy - sy) <= rap_px]))


def throughput_map_fn(g: Dict[str, Any]) -> Callable[[Any, Any], Any]:
    """Return the throughput as a function of separation and detector azimuth, bilinear in both.

    The azimuth wraps at 360 degrees, so a source between the last sampled azimuth and 0 interpolates
    across the seam.  Separations outside the map are clamped to its ends.

    Parameters
    ----------
    g : dict
        A throughput map (:func:`throughput_map`).

    Returns
    -------
    fn : callable
        Function ``(rho_arcsec, az_deg) -> throughput``, for scalars or arrays.
    """
    seps = np.asarray(g["seps"], float)
    az = np.asarray(g["az"], float)
    t = np.asarray(g["trans"], float)
    azc = np.concatenate([az, az[:1] + 360.0])             # close the circle
    tc = np.concatenate([t, t[:, :1]], axis=1)

    def f(rho, az_deg):
        r = np.clip(np.asarray(rho, float), seps[0], seps[-1])
        a = np.mod(np.asarray(az_deg, float) - azc[0], 360.0) + azc[0]
        i = np.clip(np.searchsorted(seps, r, side="right") - 1, 0, seps.size - 2)
        j = np.clip(np.searchsorted(azc, a, side="right") - 1, 0, azc.size - 2)
        fr = (r - seps[i]) / (seps[i + 1] - seps[i])
        fa = (a - azc[j]) / (azc[j + 1] - azc[j])
        return ((1 - fr) * (1 - fa) * tc[i, j] + fr * (1 - fa) * tc[i + 1, j]
                + (1 - fr) * fa * tc[i, j + 1] + fr * fa * tc[i + 1, j + 1])
    return f


# ---------------------------------------------------------------------- the dead zones
def quadrant_mask(shape: Tuple[int, int], center: Tuple[float, float], pxscale: float,
                  filter: str = "F1065C", g: Optional[Dict[str, Any]] = None,
                  min_throughput: float = 0.30,
                  bar_width_px: Optional[float] = None) -> np.ndarray:
    """Mark the pixels where the mask has taken enough of the flux that they are unusable.

    The threshold is on the measured throughput rather than on a boundary width written down once, so
    the mask follows the STPSF model for each filter.  For the Lyot mode the spot is masked out to its
    catalogued 2.16 arcsec radius.  Pixels inside the map's innermost separation are masked too.

    Parameters
    ----------
    shape : tuple of int
        Shape of the frames, whose last two entries are ``(ny, nx)``.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    pxscale : float
        Plate scale, in arcsec per pixel.
    filter : str, default "F1065C"
        The coronagraphic filter.
    g : dict or None, default None
        A throughput map to reuse.  None loads one from the cache.
    min_throughput : float, default 0.30
        Throughput below which a pixel is unusable, about half the best the 4QPMs reach, where the
        stamp starts to be distorted rather than attenuated.
    bar_width_px : float or None, default None
        Width of the Lyot support bar, in pixels.  None lets the threshold find it, since the map
        covers the bar.

    Returns
    -------
    bad : ndarray of bool, shape (ny, nx)
        True for each unusable pixel.
    """
    m = mode_for_filter(filter)
    ny, nx = int(shape[-2]), int(shape[-1])
    cx, cy = float(center[0]), float(center[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    rho = np.hypot(dx, dy) * float(pxscale)
    azd = np.degrees(np.arctan2(dy, dx))
    if g is None:
        g = throughput_map(m["filter"], log=lambda *_: None)
    bad = throughput_map_fn(g)(rho, azd) < float(min_throughput)
    if m["kind"] == "lyot":
        bad |= rho < float(m.get("spot_radius_as", 2.16))
        if bar_width_px:
            bad |= np.abs(dx) <= 0.5 * float(bar_width_px)
    # inside the innermost sampled separation the map has nothing to say: the star is there
    bad |= rho < float(np.asarray(g["seps"], float)[0])
    return bad


# ------------------------------------------------------------------- the injection model
class MIRILibraryPSF(LibraryPSF):
    """:class:`~klip_tpe.injection.LibraryPSF` whose throughput depends on where the source is, not just how far out.

    Called without an azimuth, :meth:`throughput` returns the azimuthal median at that separation and
    warns once, the least misleading single number, which is still not good enough to publish.

    Parameters
    ----------
    a : tuple
        Positional arguments of :class:`~klip_tpe.injection.LibraryPSF`.
    thru2d : callable or None, default None
        Throughput map ``(rho_arcsec, az_deg) -> throughput`` (:func:`throughput_map_fn`).  None makes
        an ordinary radial library, which does not pay for the per-frame path.
    kw : dict
        Keyword arguments of :class:`~klip_tpe.injection.LibraryPSF`.
    """

    name = "miri_library"
    azimuth_dependent = True            # inject_sources evaluates throughput per frame
    _warned = False

    def __init__(self, *a, thru2d: Optional[Callable[[Any, Any], Any]] = None, **kw):
        super().__init__(*a, **kw)
        self._thru2d = thru2d
        # A map is what makes this model position-dependent; without one it is an ordinary
        # radial library and should not pay for the per-frame path.
        self.azimuth_dependent = thru2d is not None

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        """Return the throughput at a separation and detector azimuth.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        az_deg : float or None, default None
            Detector azimuth, in degrees.  None returns the azimuthal median and warns once.

        Returns
        -------
        throughput : float
            The throughput.

        Warns
        -----
        RuntimeWarning
            Once per process, when asked without an azimuth.
        """
        if self._thru2d is None:
            return super().throughput(rho_as)
        if az_deg is None:
            if not MIRILibraryPSF._warned:
                import warnings
                warnings.warn(
                    "MIRI throughput asked for without a detector azimuth: on a four-quadrant "
                    "phase mask the throughput varies by a factor of ~3 around a circle, so "
                    "the azimuthal median returned here is not a substitute for the real "
                    "value. Pass the source's detector azimuth.", RuntimeWarning, stacklevel=2)
                MIRILibraryPSF._warned = True
            return self.typical_throughput(rho_as)
        return float(self._thru2d(float(rho_as), float(az_deg)))

    def typical_throughput(self, rho_as: float) -> float:
        """Return the azimuthal median throughput at a separation, without the warning.

        It serves callers that need a scale rather than a value, such as the float32 headroom check, which
        would otherwise warn on every injection.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        throughput : float
            The median over a uniform 1 degree grid of azimuths.
        """
        if self._thru2d is None:
            return float(super().throughput(rho_as))
        az = np.asarray(self._az_grid, float)
        return float(np.median(self._thru2d(np.full(az.shape, float(rho_as)), az)))

    @property
    def _az_grid(self) -> np.ndarray:
        """Uniform 1 degree grid of azimuths for the median, so the boundaries, where the map's own grid is dense, do not bias it low.
        """
        return np.arange(0.0, 360.0, 1.0)


def library(filter: str = "F1065C", star_flux: float = 1.0,
            seps_as: Optional[Sequence[float]] = None, date: Optional[str] = None,
            log: Callable[[str], None] = print, **kw) -> MIRILibraryPSF:
    """Build the injection model of a MIRI coronagraphic mode, radial stamps with a 2-D throughput.

    The stamps are translated to the source, never rotated, since the lobe pattern comes from the
    MASKFQPM Lyot stop and the segmented pupil, both fixed to the spacecraft.

    Parameters
    ----------
    filter : str, default "F1065C"
        The coronagraphic filter.
    star_flux : float, default 1.0
        The flux unit, one unit of contrast in the science frames' units
        (:func:`star_flux_from_flux_density`).
    seps_as : sequence of float or None, default None
        Separations of the stamp library, in arcsec.  None takes :func:`default_separations`, and the
        throughput map keeps its own ladder either way.
    date : str or None, default None
        Date of the observation, for STPSF's optical model.
    log : callable, default print
        Function called with each log line.
    kw : dict
        More keywords for :func:`~klip_tpe.stpsf_psf.offaxis_grid`.

    Returns
    -------
    model : MIRILibraryPSF
        The injection model.

    Notes
    -----
    Tying the stamps' ladder to the map's once cut a map computed over 0.3 to 10.8 arcsec down to 0.2
    to 3.0, discarded a 576-PSF cache and recomputed a shorter map that clamped across most of the
    field.  :func:`klip_tpe.stpsf_psf.library` has the measurement of the fixed lobes.
    """
    m = mode_for_filter(filter)
    seps = np.asarray(list(seps_as) if seps_as is not None else
                      default_separations(m["filter"]), float)
    grid = offaxis_grid(instrument="MIRI", filter=m["filter"], image_mask=m["image_mask"],
                        pupil_mask=m["pupil_mask"], seps_as=seps, date=date, log=log, **kw)
    tmap = throughput_map(m["filter"], date=date, log=log)
    sl = np.asarray(grid["slices"], float)
    c = tuple(grid.get("center") or ((sl.shape[-1] - 1) / 2.0, (sl.shape[-2] - 1) / 2.0))
    # refpa_deg=None: translated to the source, never rotated.  The lobe pattern of a MIRI
    # coronagraphic PSF comes from the MASKFQPM Lyot stop and the segmented pupil, both fixed
    # to the SPACECRAFT -- it does not turn as a companion moves round the field, so neither
    # does the template.  See klip_tpe.stpsf_psf.library for the measurement and for the IDL
    # predecessor's reduction that does the same thing (translation only, one template per
    # data set).
    return MIRILibraryPSF(sl, grid["seps"], center=c,
                          ee_radius_px=float(grid.get("ee_radius_px") or 3.0),
                          refpa_deg=None, flux_unit=float(star_flux),
                          thru2d=throughput_map_fn(tmap))


def star_flux_from_flux_density(filter: str, flux_density_jy: float, pixar_sr: float,
                                bunit: str = "MJy/sr", date: Optional[str] = None,
                                seps_as: Optional[Sequence[float]] = None,
                                optics_transmission: float = 1.0,
                                log: Callable[[str], None] = print) -> float:
    """Compute :func:`library`'s ``star_flux`` from the star's flux density in this filter.

    It wraps :func:`klip_tpe.stpsf_psf.star_flux_from_flux_density` on the grid :func:`library`
    builds, so the flux is normalized to the encircled-energy radius the stamps are normalized by.

    Parameters
    ----------
    filter : str
        The coronagraphic filter.
    flux_density_jy : float
        The star's flux density in the filter, in Jy.
    pixar_sr : float
        Solid angle of a pixel, in steradians.
    bunit : str, default "MJy/sr"
        Unit of the science frames.
    date : str or None, default None
        Date of the observation, for STPSF's optical model.
    seps_as : sequence of float or None, default None
        Separations of the stamp library, as for :func:`library`.
    optics_transmission : float, default 1.0
        Transmission of the optics, applied to the flux.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    star_flux : float
        One unit of contrast in the science frames' units.

    Notes
    -----
    The default flux unit of 1.0 is one count against a MIRI cube whose pixels reach several hundred
    MJy/sr, 1.1e5 times too faint for HIP 65426 in F1140C, and the calibration then walked its whole
    ladder with the S/N flat at zero.  ``scripts/run_miri`` refuses to start at 1.0.
    """
    from ..stpsf_psf import star_flux_from_flux_density as _sf
    m = mode_for_filter(filter)
    seps = np.asarray(list(seps_as) if seps_as is not None else
                      default_separations(m["filter"]), float)
    grid = offaxis_grid(instrument="MIRI", filter=m["filter"], image_mask=m["image_mask"],
                        pupil_mask=m["pupil_mask"], seps_as=seps, date=date, log=log)
    return _sf(grid, float(flux_density_jy), float(pixar_sr), bunit=bunit,
               optics_transmission=float(optics_transmission), log=log)


def forbidden_pa(angles, rho_as: float, filter: str = "F1065C", truenorth: float = 0.0,
                 g: Optional[Dict[str, Any]] = None, min_throughput: float = 0.30,
                 dead_frac: float = 0.34, step_deg: float = 1.0,
                 log: Callable[[str], None] = print):
    """Return the sky position-angle sectors an injection must not be placed in, for :class:`~klip_tpe.positions.PositionSampler`.

    The dead zones are fixed to the detector and the sampler works in sky PA, related through each
    frame's roll, ``az = theta - truenorth - 270 - parang``.  A PA is forbidden when it is suppressed in
    more than ``dead_frac`` of the frames.  Without this the search reads the mask's suppression as a
    property of the parameters it tries.

    Parameters
    ----------
    angles : array_like
        Parallactic angle of every frame, in degrees.
    rho_as : float
        Separation of the injections, in arcsec.
    filter : str, default "F1065C"
        The coronagraphic filter.
    truenorth : float, default 0.0
        True-north offset, in degrees.
    g : dict or None, default None
        A throughput map to reuse.  None loads one from the cache.
    min_throughput : float, default 0.30
        Throughput below which a position is dead.
    dead_frac : float, default 0.34
        Fraction of the frames in which a PA must be dead to be forbidden.  It is below a half since a
        JWST sequence is usually two rolls, where a ``> 0.5`` test would forbid nothing on an exact tie.
    step_deg : float, default 1.0
        Step of the PA scan, in degrees.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    sectors : list of tuple of 2 floats
        The sectors, as ``(center_deg, half_width_deg)``, joined across 0 and 360.
    """
    ang = np.asarray(angles, float).ravel()
    if not ang.size:
        return []
    if g is None:
        g = throughput_map(mode_for_filter(filter)["filter"], log=lambda *_: None)
    f = throughput_map_fn(g)
    pa = np.arange(0.0, 360.0, float(step_deg))
    az = pa[:, None] - float(truenorth) - 270.0 - ang[None, :]         # (npa, nframe)
    dead = f(np.full(az.shape, float(rho_as)), az) < float(min_throughput)
    frac = dead.mean(axis=1)
    bad = frac > float(dead_frac)
    if not bad.any():
        return []
    # contiguous runs of forbidden PA, wrapped at 360
    idx = np.flatnonzero(bad)
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, splits)
    if len(runs) > 1 and bad[0] and bad[-1]:                            # joins across 0/360
        runs[0] = np.concatenate([runs[-1] - pa.size, runs[0]])
        runs.pop()
    out = []
    for r in runs:
        lo, hi = pa[r[0] % pa.size], pa[r[0] % pa.size] + step_deg * (r.size - 1)
        out.append((float((0.5 * (lo + hi)) % 360.0), float(0.5 * step_deg * r.size)))
    log(f"  miri: {len(out)} forbidden PA sector(s) at rho={rho_as:.2f}\" covering "
        f"{100.0 * bad.mean():.0f}% of the ring: "
        + ", ".join(f"{c:.0f}+/-{w:.0f}" for c, w in out))
    return out


def dead_zone_pixel_mask(shape: Tuple[int, int], pxscale: float, angles,
                         filter: str = "F1065C", truenorth: float = 0.0,
                         g: Optional[Dict[str, Any]] = None, min_throughput: float = 0.30,
                         dead_frac: float = 0.34, center: Optional[Tuple[float, float]] = None,
                         log: Callable[[str], None] = print) -> np.ndarray:
    """Mark the derotated image's pixels that must stay out of the noise estimate, for :class:`~klip_tpe.metrics.MawetPeakSNR`'s ``pixel_mask``.

    It is the pixel-space twin of :func:`forbidden_pa`, with the same geometry, rolls,
    ``min_throughput`` and ``dead_frac``, and the two go together, as
    ``default_config(red, forbidden_pa=fpa, pixel_mask=pmask)``.  The pixels stay in the cube and the
    KLIP basis, as attenuated measurements, and out of the statistic.

    Parameters
    ----------
    shape : tuple of int
        Shape of the reduced image, whose last two entries are ``(ny, nx)``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angles : array_like
        Parallactic angle of every frame, in degrees.
    filter : str, default "F1065C"
        The coronagraphic filter.
    truenorth : float, default 0.0
        True-north offset, in degrees.
    g : dict or None, default None
        A throughput map to reuse.  None loads one from the cache.
    min_throughput : float, default 0.30
        Throughput below which a position is dead.
    dead_frac : float, default 0.34
        Fraction of the frames in which a pixel must be dead to be masked.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel in the reduced image.  None takes ``((nx-1)/2, (ny-1)/2)``, where
        derotation puts it.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True for each pixel to exclude, the map's innermost separation included.

    Notes
    -----
    NaN in the cube cannot do this job.  The reducer's high-pass spreads each NaN over the filter's
    box and the dead zones reach the frame edges, so a 12 px filter turns 5% of the frame into all of
    it and every bin is dropped.  Where the mask has taken the stellar PSF it has taken the speckles
    too, so a dead-zone pixel is quieter than its ring.  Leaving it in depresses sigma and inflates
    every S/N, so taking it out is the conservative direction, justified by the mask's design before
    anyone looks at the image.
    """
    ang = np.asarray(angles, float).ravel()
    ny, nx = int(shape[-2]), int(shape[-1])
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else (float(center[0]),
                                                                     float(center[1]))
    if g is None:
        g = throughput_map(mode_for_filter(filter)["filter"], log=lambda *_: None)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    rho = np.hypot(dx, dy) * float(pxscale)
    # the same pixel -> position-angle convention as klip_tpe.metrics.pa_wedge_mask, then
    # the same PA -> detector-azimuth map as forbidden_pa.  Composing the two is what keeps
    # this mask and those sectors describing one geometry instead of two.
    pa = np.degrees(np.arctan2(dy, dx)) - 90.0
    f = throughput_map_fn(g)
    hit = np.zeros((ny, nx), float)
    if ang.size:
        uang, cnt = np.unique(np.round(ang, 6), return_counts=True)
        for a, n in zip(uang, cnt):
            hit += float(n) * (f(rho, pa - float(truenorth) - 270.0 - float(a))
                               < float(min_throughput))
        hit /= float(ang.size)
    bad = hit > float(dead_frac)
    # inside the innermost sampled separation the map has nothing to say: the star is there
    bad |= rho < float(np.asarray(g["seps"], float)[0])
    log(f"  miri: dead zones cover {int(bad.sum())} px of the de-rotated frame "
        f"({100.0 * bad.mean():.1f}%) over {ang.size} frame(s); excluded from the noise "
        f"estimate, kept in the cube")
    return bad


# ------------------------------------------------------------------------------ loading
def apply_quadrant_mask(datasets, min_throughput: float = 0.30, filter: Optional[str] = None,
                        log: Callable[[str], None] = print):
    """Set the dead zones of every MIRI dataset to NaN, for reductions without a high-pass.

    Do not put this in front of the reducer.  Its high-pass spreads each NaN over the filter's box, so
    at the default 12 px every frame is dropped as empty, and :func:`dead_zone_pixel_mask` is the tool
    for the noise statistic.  It is kept for reductions with ``filter <= 1`` and for inspecting the
    mask's geometry.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The datasets.
    min_throughput : float, default 0.30
        Throughput below which a pixel is set to NaN (:func:`quadrant_mask`).
    filter : str or None, default None
        The coronagraphic filter.  None reads each dataset's ``meta["filter"]`` or header.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    masked : dict of str to Dataset
        New datasets with NaN in the dead zones, the others passed through, so a mixed or NIRCam set
        is safe.
    """
    out = {}
    for pid, ds in datasets.items():
        meta = dict(ds.meta or {})
        hdr = meta.get("header") or {}
        filt = filter or meta.get("filter") or (hdr.get("FILTER") if hasattr(hdr, "get") else None)
        if not filt or str(filt).upper() not in MODES:
            out[pid] = ds
            continue
        m = mode_for_filter(filt)
        px = float(meta.get("pxscale") or pixelscale(m["filter"]))
        c = meta.get("center") or ((ds.cube.shape[-1] - 1) / 2.0, (ds.cube.shape[-2] - 1) / 2.0)
        bad = quadrant_mask(ds.cube.shape, c, px, m["filter"], min_throughput=min_throughput)
        cube = np.asarray(ds.cube, np.float32).copy()
        cube[:, bad] = np.nan
        meta.update({"quadrant_mask": bad, "quadrant_masked_px": int(bad.sum()),
                     "image_mask": m["image_mask"], "lam_m": m["lam_m"], "diam_m": DIAMETER_M})
        out[pid] = ds.__class__(cube, ds.angles, ds.tags, texp=ds.texp, name=ds.name,
                                meta=meta, ref_cube=ds.ref_cube)
        log(f"  {pid}: {m['filter']} {m['image_mask']} -- masked {int(bad.sum())} dead-zone "
            f"pixels ({100.0 * bad.mean():.1f}% of the frame)")
    return out


def load_miri(cube, angles, filter: Optional[str] = None, name: str = "miri",
              mask_quadrants: bool = True, min_throughput: float = 0.30,
              header=None, **kw):
    """Build a :class:`~klip_tpe.reducer.Dataset` from MIRI coronagraphic frames.

    It does what :func:`klip_tpe.instruments.generic.load_cube` does, and takes the filter from
    ``header["FILTER"]`` when not given and the pixel scale from the STPSF model.  With
    ``mask_quadrants`` the dead zones are set to NaN, which the default high-pass (``hp_nan_aware``
    False) spreads over every frame (see :func:`apply_quadrant_mask`).  Such a reduction wants
    ``mask_quadrants=False`` and :func:`dead_zone_pixel_mask` instead.

    Parameters
    ----------
    cube : str or ndarray
        The frames, or a FITS path.
    angles : str or array_like
        Parallactic angle of each frame in degrees, or a FITS path.
    filter : str or None, default None
        The coronagraphic filter.  None reads it from ``header``.
    name : str, default "miri"
        Name of the dataset.
    mask_quadrants : bool, default True
        Whether to set the dead zones to NaN (:func:`quadrant_mask`).
    min_throughput : float, default 0.30
        Throughput below which a pixel is a dead zone.
    header : mapping or None, default None
        The frames' FITS header, kept in ``meta["header"]``.
    kw : dict
        More keywords for :func:`~klip_tpe.instruments.generic.load_cube`.

    Returns
    -------
    ds : Dataset
        The dataset, with the instrument, filter, masks, pixel scale, wavelength and diameter in its
        ``meta``.

    Raises
    ------
    ValueError
        If there is no filter, or it is not a MIRI coronagraphic one.
    """
    from .generic import load_cube
    if filter is None:
        filter = None if header is None else (header.get("FILTER") if hasattr(header, "get") else None)
    if not filter:
        raise ValueError("no MIRI filter given and none in the header; pass filter=")
    m = mode_for_filter(filter)
    ds = load_cube(cube, angles, name=name, **kw)
    px = pixelscale(m["filter"])
    meta = dict(ds.meta or {})
    meta.update({"instrument": "MIRI", "filter": m["filter"], "image_mask": m["image_mask"],
                 "pupil_mask": m["pupil_mask"], "pxscale": px, "lam_m": m["lam_m"],
                 "diam_m": DIAMETER_M})
    if header is not None:
        meta["header"] = header
    if mask_quadrants:
        bad = quadrant_mask(ds.cube.shape, meta["center"], px, m["filter"],
                            min_throughput=min_throughput)
        cube_out = np.asarray(ds.cube, np.float32).copy()
        cube_out[:, bad] = np.nan
        meta["quadrant_mask"] = bad
        meta["quadrant_masked_px"] = int(bad.sum())
        ds = ds.__class__(cube_out, ds.angles, ds.tags, texp=ds.texp, name=ds.name, meta=meta,
                          ref_cube=ds.ref_cube)
    else:
        ds.meta.update(meta)
    return ds
