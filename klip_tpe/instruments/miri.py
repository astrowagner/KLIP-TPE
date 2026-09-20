"""JWST/MIRI coronagraphy: four-quadrant phase masks and the Lyot spot.

MIRI is not NIRCam with different numbers in it.  Three of its four coronagraphs are
*four-quadrant phase masks*, which suppress starlight by putting a pi phase step across
two perpendicular lines through the star rather than by blocking a disc.  Everything
downstream that assumes a round occulter is wrong here, and the one that matters most is
the throughput model.

Measured from STPSF for F1065C/FQPM1065 (:func:`throughput_map`), as a function of
detector azimuth at each separation::

    rho\\az    0     30     60     90    120    150     range
    0.4"    0.161  0.470  0.433  0.212  0.518  0.382     x3.2
    0.8"    0.206  0.819  0.604  0.205  0.637  0.786     x4.0
    1.2"    0.260  0.935  0.693  0.263  0.918  0.912     x3.6
    1.6"    0.325  0.928  0.907  0.336  0.943  0.922     x2.9
    2.0"    0.418  0.927  0.949  0.414  0.962  0.940     x2.3
    2.4"    0.504  0.939  0.954  0.486  0.943  0.932     x2.0

The minima mark the quadrant boundaries and the map repeats under a 180 degree rotation
to three decimals, which is the symmetry the mask actually has.  A factor of two to four
at constant separation.

The boundaries are NOT on the detector axes.  Scanning 2 degree steps at 2 arcsec puts
the minimum at az = -4 on one axis and az = +86 on the other -- the mask is rotated by
four to five degrees, the same amount on both, so this is the mask's mounting angle and
not noise.  Masking the dead zone by assuming it runs along detector rows and columns
would therefore mask the wrong pixels: at 2 arcsec a 4.5 degree error is about 1.5
pixels, enough to leave the true dead zone in the data and throw away good pixels beside
it.  Nothing here assumes where the boundaries are; :func:`quadrant_mask` thresholds the
measured map, so it finds whatever rotation the instrument model carries.

``klip_tpe.stpsf_psf.offaxis_grid`` returns
``transmission`` as a function of separation alone, and ``throughput_fn`` hands back
``f(rho)``; on a 4QPM that model does not merely lose precision, it is systematically
wrong in a way that tracks position angle -- a companion sitting near a quadrant boundary
is reported up to three times fainter than it is, and one sitting between boundaries too
bright.  A contrast curve built that way is an azimuthal average of two different things.

So this module carries its own two-dimensional throughput map, sampled on an
(separation, detector azimuth) grid and cached like the radial grids are.  Two further
consequences follow from the same geometry and are handled here:

* **The boundaries are not usable.**  Where the mask has taken most of the flux the
  photometry is unreliable and the stamp is distorted, not merely attenuated.
  :func:`quadrant_mask` marks those pixels so they stay out of the KLIP basis, out of the
  noise statistics the objective is computed from, and out of the set of positions the
  sampler is allowed to inject into.  Injecting into a dead zone and then "recovering"
  nothing is not a measurement of contrast, it is a measurement of the mask.

* **Throughput is a per-frame quantity.**  The quadrant boundaries are fixed to the
  detector, so as the telescope rolls, a companion at a fixed sky position angle moves
  across them.  The attenuation of a real source therefore differs frame to frame, which
  is why :func:`throughput_map_fn` takes a detector azimuth rather than a sky PA.

KNOWN LIMITATION.  The stamp library stays radial: only the scalar throughput is
two-dimensional.  Near a boundary the PSF is distorted as well as attenuated, so a radial
stamp is the wrong shape exactly where the throughput is lowest.  That is survivable only
because :func:`quadrant_mask` removes that region from the analysis entirely -- the model
is used where it is trustworthy and the rest is masked, rather than extrapolated into.
Lifting this would mean a full (separation, azimuth) stamp library, which is a factor of
``n_azimuth`` more PSF computations.

References for the instrument numbers: Boccaletti et al. 2022 (A&A 667, A165, JWST/MIRI
coronagraphic performances as measured on-sky) for the 110 mas pixel scale, the filter
central wavelengths and the 2.16 arcsec Lyot spot radius; the pixel scale used in the
code is taken from the STPSF instrument model itself (0.109655 arcsec/px) rather than the
rounded documentation value.
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
           "throughput_map_fn", "quadrant_mask", "library", "load_miri", "MIRILibraryPSF"]

DIAMETER_M = 6.5

# Central wavelengths and bandwidths from Boccaletti et al. 2022, Table 1.  ``kind`` is
# what the focal-plane mask actually is, which is what decides the throughput geometry:
# a 4QPM has two suppression lines through the star, a Lyot has a disc and a support bar.
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
    """The coronagraphic mode a MIRI filter names, with its masks and wavelength."""
    f = str(filter).upper().strip()
    if f not in MODES:
        raise ValueError(f"{filter!r} is not a MIRI coronagraphic filter; expected one of "
                         f"{sorted(MODES)}")
    return dict(MODES[f], filter=f)


def pixelscale(filter: str = "F1065C") -> float:
    """MIRI's pixel scale, from the STPSF instrument model.

    The documentation rounds it to 110 mas; the model carries 0.109655, and a 0.3% scale
    error is a third of a pixel across a 24 arcsec field, which is enough to move a
    companion between annuli at the outer edge.  Falls back to the documented value when
    STPSF is not installed, and says so in the returned value's provenance only through
    this docstring -- callers that care should check :func:`have_stpsf`.
    """
    if not have_stpsf():
        return 0.110
    m = mode_for_filter(filter)
    inst = _instrument("MIRI", m["filter"], m["image_mask"], m["pupil_mask"], None, None, None)
    return float(inst.pixelscale)


# --------------------------------------------------------------------------- throughput
def _map_cache_name(meta: Dict[str, Any]) -> str:
    return f"miri_thrumap_{meta['filter']}_{_key(meta)}.npz"


# Offsets from a quadrant boundary, in degrees.  Non-uniform on purpose: the throughput
# changes by a factor of four over the few degrees either side of a boundary and is flat
# between them, so a uniform grid either cannot resolve the boundary or wastes most of its
# PSFs on the flat part.  Sampled every 30 degrees the dead zone came out 0.2% of the
# field -- the interpolant simply stepped over it -- while the mask it is meant to define
# is the one region of the image where the photometry cannot be trusted.
_BOUNDARY_OFFSETS = (0.0, 2.5, 6.0, 12.0, 22.0, 34.0, 45.0)


def default_azimuths(boundaries: Sequence[float] = (0.0, 90.0, 180.0, 270.0)) -> np.ndarray:
    """Detector azimuths for :func:`throughput_map`: dense on the quadrant boundaries.

    The grid mirrors :data:`_BOUNDARY_OFFSETS` about each entry of ``boundaries``; points
    shared between neighbouring sectors collapse, which is why four boundaries give 48
    azimuths rather than 56.  The default is the detector axes, which is where the
    boundaries are *not* -- pass the angles :func:`locate_boundaries` measured.
    """
    az = {(b + s * o) % 360.0 for b in np.asarray(boundaries, float).ravel()
          for o in _BOUNDARY_OFFSETS for s in (1, -1)}
    return np.array(sorted(az), float)


def locate_boundaries(inst, rho_as: float, fov: float, oversample: int, nlambda: int,
                      rap_px: float, coarse_step: float = 15.0,
                      log: Callable[[str], None] = print) -> np.ndarray:
    """Where the four suppression lines actually are, in detector azimuth.

    Sampling densely on the detector axes and calling that "dense on the boundaries" is
    only right if the mask is square to the detector, and it is not: the minima sit four
    to five degrees off, the same on both axes.  A dense grid centred 4.5 degrees away
    from the feature it is meant to resolve is dense in the wrong place, and the dead-zone
    mask that comes out of it is offset by the same amount -- about 1.5 pixels at 2 arcsec,
    which leaves the real dead zone unmasked and throws away good pixels beside it.

    So the boundaries are measured first, with one coarse scan at a single separation, and
    the dense grid is built around what that finds.  Returns four angles in [0, 360).
    """
    az = np.arange(0.0, 360.0, float(coarse_step))
    t = np.array([_offset_sum(inst, rho_as, float(a), fov, oversample, nlambda, rap_px)
                  for a in az])
    out = []
    for b in (0.0, 90.0, 180.0, 270.0):                  # one minimum per nominal axis
        d = np.abs(((az - b + 180.0) % 360.0) - 180.0)
        near = d <= 1.5 * coarse_step
        out.append(float(az[near][int(np.argmin(t[near]))]))
    out = np.array(out, float)
    off = ((out - np.array([0.0, 90.0, 180.0, 270.0]) + 180.0) % 360.0) - 180.0
    log(f"    quadrant boundaries at az = {np.round(out, 1).tolist()} deg "
        f"(offset from the detector axes: {np.round(off, 1).tolist()})")
    return out


def throughput_map(filter: str = "F1065C", seps_as: Optional[Sequence[float]] = None,
                   az_deg: Optional[Sequence[float]] = None, oversample: int = 2,
                   nlambda: int = 1, date: Optional[str] = None, cache: bool = True,
                   log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Coronagraph throughput on an (separation, detector azimuth) grid.

    One STPSF PSF per grid node, each divided by the unocculted PSF of the same
    configuration summed over the same aperture, so the result is the fraction of a
    point source's flux that survives the mask at that position.

    ``az_deg`` is the azimuth in the DETECTOR frame, measured counter-clockwise from +x,
    because that is the frame the quadrant boundaries are fixed in.  Left as ``None`` the
    grid is chosen in two passes: one coarse scan locates the boundaries
    (:func:`locate_boundaries`), then :func:`default_azimuths` concentrates the sampling
    on the angles that scan found.  The one-pass alternative -- dense on the detector axes
    -- is dense four to five degrees away from the feature it is meant to resolve.  The
    map is never folded into one quadrant: the four-fold symmetry is only nominal, and
    measuring the full circle is what shows which parts of it the as-built model has.
    This map repeats under 180 degrees to three decimals but not under 90.

    ``nlambda=1`` is the default because this is a ratio of two PSFs in the same filter,
    where the chromatic terms largely cancel; the stamps in :func:`library` are computed
    polychromatically as usual.

    Returns ``{'seps', 'az', 'trans', 'boundaries', 'pxscale', 'fwhm_px', 'ee_radius_px',
    'meta'}`` with ``trans`` of shape ``(nsep, naz)``.  Cached by configuration under
    :func:`~klip_tpe.stpsf_psf.cache_dir`.
    """
    m = mode_for_filter(filter)
    seps = np.asarray(list(seps_as) if seps_as is not None else
                      np.arange(0.2, 3.01, 0.2), float)
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

    bounds = None
    if az is None:
        log(f"  miri: locating the quadrant boundaries for {m['filter']}/{m['image_mask']}")
        bounds = locate_boundaries(inst, float(np.median(seps)), fov, oversample, nlambda,
                                   rap, log=log)
        az = default_azimuths(bounds)
    log(f"  miri: computing {seps.size} x {az.size} = {seps.size * az.size} off-axis PSFs "
        f"(cached afterwards)")
    trans = np.zeros((seps.size, az.size), float)
    for i, r in enumerate(seps):
        for j, a in enumerate(az):
            trans[i, j] = _offset_sum(inst, float(r), float(a), fov, oversample, nlambda,
                                      rap) / max(ref_sum, 1e-30)
        log(f"    rho={r:.2f}\"  throughput {trans[i].min():.3f}-{trans[i].max():.3f} "
            f"(x{trans[i].max() / max(trans[i].min(), 1e-6):.1f} across azimuth)")
    out = {"seps": seps, "az": az, "trans": trans, "boundaries": bounds, "pxscale": px,
           "fwhm_px": fwhm_px, "ee_radius_px": rap, "meta": meta, "path": path}
    if cache:
        os.makedirs(cache_dir(), exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir(), _map_cache_name(meta)),
                            seps=seps, az=az, trans=trans, pxscale=px, fwhm_px=fwhm_px,
                            ee_radius_px=rap, key=_key(meta), meta=json.dumps(meta),
                            **({} if bounds is None else {"boundaries": bounds}))
    return out


def _fwhm(img: np.ndarray) -> float:
    """FWHM in pixels of a centred PSF, by counting the pixels above half its peak."""
    img = np.asarray(img, float)
    pk = float(np.nanmax(img))
    if not np.isfinite(pk) or pk <= 0:
        return float("nan")
    return float(2.0 * np.sqrt(max(np.sum(img >= 0.5 * pk), 1) / np.pi))


def _offset_sum(inst, rho_as: float, az_deg: float, fov: float, oversample: int,
                nlambda: int, rap_px: float) -> float:
    """Aperture sum of the PSF of a source ``rho_as`` from the star at detector azimuth
    ``az_deg``, measured in the same aperture as the reference.

    STPSF's ``source_offset_theta`` runs from +Y towards -X, not towards +X, so the
    detector azimuth (counter-clockwise from +X) is ``theta + 90`` and the conversion
    below is ``theta = az - 90``.  Verified against the PSF rather than assumed: at
    ``theta`` = 0, 45, 90, 180, 270 the peak lands at az = 90, 135, 180, 270, 0.

    Getting this backwards puts the aperture somewhere the source is not, and the two
    coincide only where ``az = 180 - az``, i.e. at 90 and 270 degrees.  The map then comes
    back with two bright lobes and near-zero everywhere else -- which looks enough like a
    mask feature to believe, and is not one.
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
    """``(rho_arcsec, az_deg) -> throughput``, bilinear in separation and azimuth.

    Azimuth wraps at 360 degrees, so a source between the last sampled azimuth and 0
    interpolates across the seam instead of being clamped to one end of the table.
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
    """``True`` where the mask has taken enough of the flux that the pixel is unusable.

    The threshold is on measured throughput rather than on a hardcoded boundary width,
    so the mask follows whatever the STPSF model says for this filter instead of a number
    written down once and then wrong for the other three.  ``min_throughput=0.30`` is
    roughly half the best throughput the 4QPMs reach, which is where the stamp starts
    being distorted rather than simply attenuated -- the regime the radial stamp library
    cannot represent (see the module docstring).

    For the Lyot mode the occulting spot is masked from its catalogued 2.16 arcsec radius
    and the support bar from ``bar_width_px`` if given; the throughput map covers the bar
    too, so the default is to let the threshold find it.

    Pass ``g`` to reuse a map already computed; otherwise one is loaded from the cache.
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
    """:class:`~klip_tpe.injection.LibraryPSF` whose throughput depends on where the
    source is, not just how far out.

    ``throughput(rho, az_deg)`` reads the two-dimensional map.  Called without an azimuth
    -- which is what every call site that predates MIRI does -- it returns the azimuthal
    *median* at that separation and warns once, rather than silently handing back a number
    that is wrong by up to a factor of three in either direction.  A median is the least
    misleading single number available, and it is still not good enough to publish.
    """

    name = "miri_library"
    _warned = False

    def __init__(self, *a, thru2d: Optional[Callable[[Any, Any], Any]] = None, **kw):
        super().__init__(*a, **kw)
        self._thru2d = thru2d

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
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
            az = np.asarray(self._az_grid, float)
            return float(np.median(self._thru2d(np.full(az.shape, float(rho_as)), az)))
        return float(self._thru2d(float(rho_as), float(az_deg)))

    @property
    def _az_grid(self) -> np.ndarray:
        """Uniform, and fine, on purpose.  The median stands for "a source at an azimuth
        we were not told", so it has to be taken over azimuth uniformly; taking it over
        the map's own grid would weight the boundaries heavily, since that grid is dense
        exactly where the throughput is worst, and bias the answer low."""
        return np.arange(0.0, 360.0, 1.0)


def library(filter: str = "F1065C", star_flux: float = 1.0,
            seps_as: Optional[Sequence[float]] = None, date: Optional[str] = None,
            log: Callable[[str], None] = print, **kw) -> MIRILibraryPSF:
    """Injection model for a MIRI coronagraphic mode: radial stamps, 2-D throughput."""
    m = mode_for_filter(filter)
    seps = np.asarray(list(seps_as) if seps_as is not None else np.arange(0.2, 3.01, 0.2), float)
    grid = offaxis_grid(instrument="MIRI", filter=m["filter"], image_mask=m["image_mask"],
                        pupil_mask=m["pupil_mask"], seps_as=seps, date=date, log=log, **kw)
    tmap = throughput_map(m["filter"], seps_as=seps, date=date, log=log)
    sl = np.asarray(grid["slices"], float)
    c = tuple(grid.get("center") or ((sl.shape[-1] - 1) / 2.0, (sl.shape[-2] - 1) / 2.0))
    return MIRILibraryPSF(sl, grid["seps"], center=c,
                          ee_radius_px=float(grid.get("ee_radius_px") or 3.0),
                          refpa_deg=0.0, flux_unit=float(star_flux),
                          thru2d=throughput_map_fn(tmap))


# ------------------------------------------------------------------------------ loading
def load_miri(cube, angles, filter: Optional[str] = None, name: str = "miri",
              mask_quadrants: bool = True, min_throughput: float = 0.30,
              header=None, **kw):
    """A :class:`~klip_tpe.datasets.Dataset` from MIRI coronagraphic frames.

    Everything :func:`klip_tpe.instruments.generic.load_cube` does, plus: the filter is
    taken from ``header['FILTER']`` when not given, the pixel scale comes from the STPSF
    model, and -- unless ``mask_quadrants=False`` -- the dead zones are set to NaN so they
    stay out of the KLIP basis and out of the noise statistics.  NaN rather than zero
    because the engine treats NaN as missing, while a zero is a measurement of zero and
    drags every statistic computed over the annulus towards it.
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
