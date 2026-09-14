"""Synthetic-companion injection.

An :class:`InjectionModel` answers two questions for a separation ``rho``: what does
an off-axis PSF look like there (a unit-normalised stamp) and what fraction of the
flux survives the coronagraph (``throughput``).  :func:`inject_sources` places the
scaled stamps into every frame of a pupil-tracking cube at the detector azimuth
implied by the frame's parallactic angle, optionally rotating the stamp so an
anisotropic (vortex-distorted) PSF keeps its distortion axis radial.

Conventions (match ``reduce_near_2.pro`` ``do_inject``):
``az = theta - truenorth - 270 - parang`` degrees is the detector azimuth from +x;
after derotation (CCW by ``parang + truenorth``) the source lands at
``theta + 90`` deg from +x, i.e. at PA ``theta`` East of North.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .metrics import Source

__all__ = ["InjectionModel", "GaussianPSF", "TemplatePSF", "AiryPSF", "FramePSF", "LibraryPSF", "inject_sources",
           "shift_bilinear", "add_stamp"]


def shift_bilinear(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel shift by bilinear interpolation (IDL ``fshift`` fractional part)."""
    return ndimage.shift(np.asarray(img, float), (dy, dx), order=1, mode="constant", cval=0.0)


def add_stamp(frame: np.ndarray, stamp: np.ndarray, stamp_center: Tuple[float, float],
              x: float, y: float) -> None:
    """Add ``stamp`` to ``frame`` in place so that ``stamp_center`` (x, y in stamp
    pixels) lands at frame position ``(x, y)``; integer part by placement, fractional
    part by bilinear shift (``near2_psfstamp``).  Clipped at the frame edges."""
    sh, sw = stamp.shape
    fx = x - stamp_center[0]
    fy = y - stamp_center[1]
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    s = shift_bilinear(stamp, fx - ix, fy - iy)
    ny, nx = frame.shape
    x0, y0 = max(ix, 0), max(iy, 0)
    x1, y1 = min(ix + sw, nx), min(iy + sh, ny)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] += s[y0 - iy:y1 - iy, x0 - ix:x1 - ix]


class InjectionModel:
    """Base class.

    ``stamp(rho_as) -> (stamp, (xc, yc), ok)``: PSF template normalised so that the
    flux unit is 1 (encircled energy within ``ee_radius_px`` or total sum, model
    dependent).  ``ok=False`` means the model has no template at that separation and
    the caller should fall back (``fallback`` model) -- the throughput is still taken
    from this model.
    ``flux_unit``: flux (in data units) of the *star* in the same normalisation, so
    an injected companion of ``contrast`` carries ``contrast * flux_unit *
    throughput(rho)``.
    ``refpa_deg``: if not None, stamps are rotated so their reference axis follows
    the source's detector azimuth (anisotropic PSF libraries).
    """

    flux_unit: float = 1.0
    refpa_deg: Optional[float] = None
    per_frame: bool = False          # True -> inject_sources calls frame_stamp(j, rho) for every frame j
    name = "injection_model"

    def stamp(self, rho_as: float):  # pragma: no cover
        raise NotImplementedError

    def frame_stamp(self, j: int, rho_as: float):  # pragma: no cover
        """Per-frame models: ``(stamp, (xc, yc), flux_unit_j)`` for frame ``j`` -- the
        frame's own PSF, unit-normalised, and the star flux of THAT frame in the same
        normalisation (so contrast tracks transparency / Strehl frame by frame)."""
        raise NotImplementedError

    def throughput(self, rho_as: float) -> float:
        return 1.0

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        """Optional measured matched filter for :class:`~klip_tpe.metrics.MawetPeakSNR`;
        None -> Gaussian."""
        return None

    def describe(self) -> dict:
        return {"name": self.name, "flux_unit": float(self.flux_unit), "refpa_deg": self.refpa_deg}


class GaussianPSF(InjectionModel):
    """Analytic Gaussian PSF of the given FWHM, unit total flux.  ``star_flux`` is the
    flux unit (peak-normalised stars: pass the star's total flux)."""

    name = "gaussian"

    def __init__(self, fwhm_px: float, star_flux: float = 1.0, size: int = 41):
        self.fwhm = float(fwhm_px)
        self.flux_unit = float(star_flux)
        self.size = int(size) | 1
        c = (self.size - 1) / 2.0
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        sg = self.fwhm / 2.3548
        g = np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2) / sg ** 2)
        self._stamp = g / g.sum()
        self._center = (c, c)

    def stamp(self, rho_as: float):
        return self._stamp, self._center, True


class TemplatePSF(InjectionModel):
    """Empirical PSF template (e.g. an unsaturated median PSF).  ``center`` in stamp
    pixels; the stamp is normalised to unit flux inside ``ee_radius_px`` (or unit
    total if None) and ``flux_unit`` defaults to the template's own flux in that
    normalisation, i.e. contrast is relative to the template star."""

    name = "template"

    def __init__(self, template: np.ndarray, center: Optional[Tuple[float, float]] = None,
                 ee_radius_px: Optional[float] = None, star_flux: Optional[float] = None,
                 throughput_fn=None):
        t = np.asarray(template, float)
        t = np.where(np.isfinite(t), t, 0.0)
        if center is None:
            center = ((t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0)
        self._center = tuple(center)
        if ee_radius_px is None:
            f = float(t.sum())
        else:
            yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
            m = np.hypot(xx - center[0], yy - center[1]) <= ee_radius_px
            f = float(t[m].sum())
        self._stamp = t / f if f != 0 else t
        self.template_flux = f
        self.flux_unit = float(f if star_flux is None else star_flux)
        self._thru = throughput_fn

    def stamp(self, rho_as: float):
        return self._stamp, self._center, True

    def throughput(self, rho_as: float) -> float:
        return 1.0 if self._thru is None else float(self._thru(rho_as))


class AiryPSF(InjectionModel):
    """Obstructed, possibly elliptical Airy PSF -- the ``airy_disk`` model pyNOMIC fits to
    every LBTI/NOMIC frame (``amp, sigmax, sigmay, offset, p, x0, y0``; secondary
    obscuration ratio ``e``).  ``sigmax/sigmay`` are pyNOMIC's scale parameters (the
    first dark ring sits at ``1.22 * sigma`` px), ``p`` the rotation in radians.

    The stamp is normalised to unit flux within ``ee_radius_px`` (default 2.5 sigma) and
    ``flux_unit`` is the star's flux in that normalisation for amplitude ``amp``, so an
    injected companion of contrast ``c`` is exactly pyNOMIC's ``airy_disk(c * amp, ...)``.
    """

    name = "airy"

    def __init__(self, sigmax: float, sigmay: Optional[float] = None, p: float = 0.0, amp: float = 1.0,
                 e: float = 0.11, size: Optional[int] = None, ee_radius_px: Optional[float] = None,
                 frame_params: Optional[np.ndarray] = None):
        """``frame_params`` (n, >=5) = pyNOMIC ``reffits`` rows ``(amp, sigmax, sigmay, offset,
        p, ...)`` per frame: the model then injects frame ``j`` with frame ``j``'s own fit --
        exactly pyNOMIC's ``inject_source`` -- and ``sigmax/sigmay/p/amp`` describe the
        median (used for ``stamp()``, the matched filter and ``flux_unit``)."""
        self.sigmax = float(sigmax)
        self.sigmay = float(self.sigmax if sigmay is None else sigmay)
        self.p, self.amp, self.e = float(p), float(amp), float(e)
        sig = max(self.sigmax, self.sigmay)
        self.size = int(size) | 1 if size else (int(np.ceil(6 * sig)) | 1)
        c = (self.size - 1) / 2.0
        self._center = (c, c)
        self.ee_radius_px = float(ee_radius_px) if ee_radius_px else 2.5 * sig
        star = self._model(self.amp, self.sigmax, self.sigmay, self.p)
        f = self._core_flux(star)
        self._stamp = star / f
        self.flux_unit = f
        # FWHM of the (circularised) core, for the matched filter / metric
        self.fwhm_px = 1.028 * 0.5 * (self.sigmax + self.sigmay)
        self._frame_params = None
        if frame_params is not None:
            fp = np.asarray(frame_params, float)
            ok = np.isfinite(fp[:, :5]).all(axis=1) & (fp[:, 1] > 0) & (fp[:, 2] > 0) & (fp[:, 0] > 0)
            self._frame_params, self._frame_ok = fp, ok
            self.per_frame = True
            self.n_bad = int((~ok).sum())
            self._cache: dict = {}

    def _model(self, amp, sx, sy, p):
        from scipy.special import j1
        c = self._center[0]
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        rad = np.pi * np.sqrt((((xx - c) * np.cos(p) + (yy - c) * np.sin(p)) / sx) ** 2
                              + (((xx - c) * np.sin(p) - (yy - c) * np.cos(p)) / sy) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (2 * j1(rad) / rad - 2 * self.e * j1(self.e * rad) / rad) ** 2
        m[rad == 0] = (1 - self.e ** 2) ** 2
        return amp * m

    def _core_flux(self, star: np.ndarray) -> float:
        c = self._center[0]
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        return float(star[np.hypot(xx - c, yy - c) <= self.ee_radius_px].sum())

    def stamp(self, rho_as: float):
        return self._stamp, self._center, True

    def frame_stamp(self, j: int, rho_as: float):
        """Frame ``j``'s own Airy fit (unit core flux, and that frame's star flux)."""
        if self._frame_params is None or not self._frame_ok[j]:
            return self._stamp, self._center, float(self.flux_unit)
        amp, sx, sy, _off, p = self._frame_params[j, :5]
        key = (round(float(amp), 6), round(float(sx), 4), round(float(sy), 4), round(float(p), 4))
        hit = self._cache.get(key)
        if hit is None:
            star = self._model(float(amp), float(sx), float(sy), float(p))
            f = self._core_flux(star)
            hit = ((star / f).astype(np.float32), f)
            if len(self._cache) < 4096:
                self._cache[key] = hit
        return hit[0], self._center, hit[1]

    def describe(self) -> dict:
        d = super().describe()
        d.update(sigmax=self.sigmax, sigmay=self.sigmay, p=self.p, amp=self.amp, e=self.e,
                 ee_radius_px=self.ee_radius_px, per_frame=bool(self.per_frame))
        return d


class FramePSF(InjectionModel):
    """Frame-by-frame empirical PSF: the (unsaturated) star of **each frame itself** is
    the template for the companion injected into that frame.

    For frame ``j`` the stamp is the ``2*r_stamp+1`` cut-out around the star, with a
    cosine taper between ``r_ee`` and ``r_stamp`` so the halo / speckles beyond the core
    (which would otherwise be copied at ``contrast`` level) fade to zero, normalised to
    unit flux within ``r_ee``; ``flux_unit_j`` is that frame's star flux in the same
    normalisation.  Frames must be registered (star at ``center``; default the frame
    centre ``((nx-1)/2, (ny-1)/2)``) and background-subtracted -- exactly what pyNOMIC's
    ``frame_registration`` + ``subtract_background`` deliver.  A frame whose core flux
    is not positive falls back to the median template (``self.stamp``).

    ``flux_unit`` (the class-level value used by the FM code / setup files) is the median
    of the per-frame values; ``stamp()`` returns the median-normalised template, which is
    also what the matched filter of the metric is built from.
    """

    name = "frame"
    per_frame = True

    def __init__(self, cube: np.ndarray, r_ee: float, r_stamp: Optional[float] = None,
                 center: Optional[Tuple[float, float]] = None, throughput_fn=None):
        cube = np.asarray(cube)
        n, ny, nx = cube.shape
        cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
        self.r_ee = float(r_ee)
        self.r_stamp = float(r_stamp) if r_stamp else 1.6 * self.r_ee
        h = int(np.ceil(self.r_stamp))
        ix, iy = int(round(cx)), int(round(cy))
        x0, y0 = ix - h, iy - h
        if x0 < 0 or y0 < 0 or x0 + 2 * h + 1 > nx or y0 + 2 * h + 1 > ny:
            raise ValueError("FramePSF: stamp radius does not fit in the frame")
        self._center = (cx - x0, cy - y0)
        yy, xx = np.mgrid[0:2 * h + 1, 0:2 * h + 1]
        rr = np.hypot(xx - self._center[0], yy - self._center[1])
        taper = np.ones_like(rr)
        band = (rr > self.r_ee) & (rr < self.r_stamp)
        taper[band] = 0.5 * (1 + np.cos(np.pi * (rr[band] - self.r_ee) / (self.r_stamp - self.r_ee)))
        taper[rr >= self.r_stamp] = 0.0
        inside = rr <= self.r_ee
        cut = np.asarray(cube[:, y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1], float)
        cut = np.where(np.isfinite(cut), cut, 0.0) * taper[None]
        flux = cut[:, inside].sum(axis=1)
        self.frame_flux = flux
        good = np.isfinite(flux) & (flux > 0)
        self._stamps = np.zeros_like(cut, dtype=np.float32)
        self._stamps[good] = (cut[good] / flux[good, None, None]).astype(np.float32)
        self._good = good
        med = np.median(cut[good], axis=0) if good.any() else np.zeros(cut.shape[1:])
        f = float(med[inside].sum()) if good.any() else 1.0
        self._median_stamp = (med / f if f > 0 else med).astype(np.float32)
        self.flux_unit = float(np.median(flux[good])) if good.any() else 1.0
        self.n_bad = int((~good).sum())
        self._thru = throughput_fn

    def stamp(self, rho_as: float):
        return self._median_stamp, self._center, True

    def frame_stamp(self, j: int, rho_as: float):
        if self._good[j]:
            return self._stamps[j], self._center, float(self.frame_flux[j])
        return self._median_stamp, self._center, float(self.flux_unit)

    def throughput(self, rho_as: float) -> float:
        return 1.0 if self._thru is None else float(self._thru(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        from .metrics import kernel_from_profile
        return kernel_from_profile(self._median_stamp, fwhm)

    def describe(self) -> dict:
        d = super().describe()
        d.update(r_ee=self.r_ee, r_stamp=self.r_stamp, n_frames=int(self._good.size), n_bad=self.n_bad)
        return d


class LibraryPSF(InjectionModel):
    """Separation-dependent PSF library: ``slices[i]`` valid at ``seps[i]`` arcsec,
    linearly interpolated in separation and re-normalised to unit flux within
    ``ee_radius_px`` of ``center``.  Outside ``[seps[0], seps[-1]]`` ``ok=False``."""

    name = "library"

    def __init__(self, slices: np.ndarray, seps: Sequence[float], center: Tuple[float, float],
                 ee_radius_px: float, throughput_fn=None, refpa_deg: Optional[float] = None,
                 flux_unit: float = 1.0):
        self.slices = np.asarray(slices, float)
        self.seps = np.asarray(seps, float)
        self.center = tuple(center)
        self.rap = float(ee_radius_px)
        yy, xx = np.mgrid[0:self.slices.shape[1], 0:self.slices.shape[2]]
        self._eemask = np.hypot(xx - center[0], yy - center[1]) <= self.rap
        self._thru = throughput_fn
        self.refpa_deg = refpa_deg
        self.flux_unit = float(flux_unit)
        for i in range(self.slices.shape[0]):
            s = self.slices[i][self._eemask].sum()
            if s > 0:
                self.slices[i] = self.slices[i] / s

    def stamp(self, rho_as: float):
        r = float(rho_as)
        if not np.isfinite(r) or r < self.seps[0] or r > self.seps[-1]:
            return None, self.center, False
        j = int(np.clip(np.searchsorted(self.seps, r, side="right") - 1, 0, len(self.seps) - 2))
        f = (r - self.seps[j]) / (self.seps[j + 1] - self.seps[j])
        st = (1 - f) * self.slices[j] + f * self.slices[j + 1]
        s = st[self._eemask].sum()
        if s > 0:
            st = st / s
        return st, self.center, True

    def throughput(self, rho_as: float) -> float:
        return 1.0 if self._thru is None else float(self._thru(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        from .metrics import kernel_from_profile
        st, _, ok = self.stamp(rho_as)
        if not ok:
            return None
        return kernel_from_profile(st, fwhm)


def inject_sources(cube: np.ndarray, angles: np.ndarray, sources: Sequence[Source], model: InjectionModel,
                   pxscale: float, truenorth: float = 0.0, center: Optional[Tuple[float, float]] = None,
                   fallback: Optional[InjectionModel] = None, angle_convention: str = "pa",
                   copy: bool = True) -> np.ndarray:
    """Return ``cube`` with synthetic companions added to every frame.

    For each source: amplitude ``contrast * model.flux_unit * model.throughput(rho)``;
    per frame the stamp is placed at
    ``center + (rho/pxscale) * (cos az, sin az)`` with ``az = theta - truenorth - 270 -
    parang`` (``angle_convention='pa'``) or ``az = theta - parang`` (``'math'``).
    Anisotropic models (``refpa_deg`` set) have their stamp rotated CCW by
    ``az - refpa`` first.
    """
    out = np.array(cube, dtype=np.float32, copy=copy)
    n, ny, nx = out.shape
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    angles = np.asarray(angles, float)
    for s in sources:
        st, sc, ok = model.stamp(s.rho)
        m = model
        if not ok:
            if fallback is None:
                raise ValueError(f"no PSF template at rho={s.rho} and no fallback model")
            st, sc, ok = fallback.stamp(s.rho)
            m = fallback
        amp = s.contrast * m.flux_unit * model.throughput(s.rho)
        base = np.asarray(st, float) * amp
        r_px = s.rho / pxscale
        per_frame = bool(getattr(model, "per_frame", False)) and ok
        thru = model.throughput(s.rho)
        for j in range(n):
            if angle_convention == "pa":
                az = s.theta - truenorth - 270.0 - angles[j]
            else:
                az = s.theta - angles[j]
            azr = np.deg2rad(az)
            xs, ys = center[0] + r_px * np.cos(azr), center[1] + r_px * np.sin(azr)
            if per_frame:
                stj, sc, fu_j = model.frame_stamp(j, s.rho)
                base = np.asarray(stj, float) * (s.contrast * fu_j * thru)
            stamp = base
            if m.refpa_deg is not None:
                from .klip import rotate_ccw
                stamp = rotate_ccw(base, az - m.refpa_deg, cval=0.0)
            add_stamp(out[j], stamp, sc, xs, ys)
    return out
