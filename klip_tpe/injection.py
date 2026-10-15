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

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .metrics import Source

__all__ = ["InjectionModel", "GaussianPSF", "TemplatePSF", "AiryPSF", "FramePSF", "LibraryPSF", "inject_sources",
           "inject_frames", "shift_bilinear", "add_stamp", "takes_azimuth", "read_sources"]


def read_sources(path: str, columns: int = 3):
    """Point sources from a text file, one per line: ``rho pa contrast`` with ``rho`` in
    arcsec, ``pa`` in degrees East of North and ``contrast`` in the injection model's flux
    unit (for NEAR, alpha Cen A).  Whitespace or commas separate the values, ``#`` starts a
    comment, and blank lines are skipped.  Columns after the third are ignored here
    (:func:`klip_tpe.chainfit.read_chain` reads a node label from the fourth).  Returns a
    list of :class:`~klip_tpe.metrics.Source`."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            s = line.split("#", 1)[0].replace(",", " ").split()
            if not s:
                continue
            if len(s) < columns:
                raise ValueError(f"{path}:{i}: need rho, pa and contrast, got {line.strip()!r}")
            try:
                rho, pa, c = (float(v) for v in s[:3])
            except ValueError:
                raise ValueError(f"{path}:{i}: not three numbers: {line.strip()!r}") from None
            if not (np.isfinite(rho) and np.isfinite(pa) and np.isfinite(c)) or rho < 0:
                raise ValueError(f"{path}:{i}: rho must be finite and >= 0, pa and contrast finite: "
                                 f"{line.strip()!r}")
            out.append(Source(rho, pa, c))
    return out


def takes_azimuth(model) -> bool:
    """True when this model's throughput depends on azimuth as well as separation.

    Read from the class attribute ``azimuth_dependent`` rather than inferred from the
    signature of ``throughput``.  Every model in this package accepts the ``az_deg``
    argument, because a uniform signature is worth having, but almost none of them look
    at it -- so a signature test answers "does this model tolerate an azimuth", which is
    true everywhere and therefore useless, rather than "does this model need one".

    A model written outside this package that overrides ``throughput`` to use an azimuth
    must set ``azimuth_dependent = True`` to be called with one.  That is the safe
    default: a position-dependent model treated as radial is a visible factor-of-a-few
    error in its own contrast curve, while a radial model driven down the per-frame path
    would silently pay for it on every injection of every evaluation.
    """
    return bool(getattr(model, "azimuth_dependent", False))


def shift_bilinear(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel shift by bilinear interpolation (IDL ``fshift`` fractional part)."""
    return ndimage.shift(np.asarray(img, float), (dy, dx), order=1, mode="constant", cval=0.0)


def stamp_crop(shape: Tuple[int, int], stamp: np.ndarray, stamp_center: Tuple[float, float],
               x: float, y: float, buf: Optional[np.ndarray] = None):
    """Where :func:`add_stamp` puts ``stamp`` in a frame of ``shape`` and what it adds there:
    ``(y0, y1, x0, x1, values)``, the frame's rows ``y0:y1`` and columns ``x0:x1`` and the
    shifted stamp's float64 values over them, or None when the stamp falls outside the
    frame.  ``values`` is a view into ``buf`` when one is given (copy it to keep it past the
    next call)."""
    sh, sw = stamp.shape
    fx = x - stamp_center[0]
    fy = y - stamp_center[1]
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    ny, nx = int(shape[0]), int(shape[1])
    x0, y0 = max(ix, 0), max(iy, 0)
    x1, y1 = min(ix + sw, nx), min(iy + sh, ny)
    if x1 <= x0 or y1 <= y0:
        return None
    if buf is None:
        s = shift_bilinear(stamp, fx - ix, fy - iy)
    else:
        s = ndimage.shift(np.asarray(stamp, float), (fy - iy, fx - ix), output=buf, order=1,
                          mode="constant", cval=0.0)
    return y0, y1, x0, x1, s[y0 - iy:y1 - iy, x0 - ix:x1 - ix]


def add_crop(frame: np.ndarray, crop) -> None:
    """Add a :func:`stamp_crop` to ``frame`` in place (a float32 frame takes the float64
    values the way ``+=`` takes them: summed in double, rounded once)."""
    y0, y1, x0, x1, v = crop
    frame[y0:y1, x0:x1] += v


def add_stamp(frame: np.ndarray, stamp: np.ndarray, stamp_center: Tuple[float, float],
              x: float, y: float, buf: Optional[np.ndarray] = None) -> None:
    """Add ``stamp`` to ``frame`` in place so that ``stamp_center`` (x, y in stamp
    pixels) lands at frame position ``(x, y)``; integer part by placement, fractional
    part by bilinear shift (``near2_psfstamp``).  Clipped at the frame edges.

    ``buf``: a float64 array of the stamp's shape that receives the shifted stamp, so a
    caller adding many stamps does not allocate one each time (the numbers are the same)."""
    c = stamp_crop(frame.shape, stamp, stamp_center, x, y, buf)
    if c is not None:
        add_crop(frame, c)


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
    azimuth_dependent: bool = False  # True -> throughput() is evaluated per frame, with that frame's azimuth
    name = "injection_model"

    def stamp(self, rho_as: float):  # pragma: no cover
        raise NotImplementedError

    def frame_stamp(self, j: int, rho_as: float):  # pragma: no cover
        """Per-frame models: ``(stamp, (xc, yc), flux_unit_j)`` for frame ``j`` -- the
        frame's own PSF, unit-normalised, and the star flux of THAT frame in the same
        normalisation (so contrast tracks transparency / Strehl frame by frame)."""
        raise NotImplementedError

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        """Fraction of a point source's flux that survives the coronagraph at ``rho_as``.

        ``az_deg`` is the source's azimuth in the DETECTOR frame.  Most coronagraphs are
        round and ignore it; a four-quadrant phase mask does not, and on MIRI the
        throughput varies by a factor of two to four around a circle of constant
        separation (see :mod:`klip_tpe.instruments.miri`).  A model that uses it declares
        so by accepting the argument, which :func:`takes_azimuth` detects, so a model
        written before this argument existed is still called the old way.
        """
        return 1.0

    def typical_throughput(self, rho_as: float) -> float:
        """A representative throughput at ``rho_as``, for diagnostics that need a scale
        rather than a value -- and that must not warn about the azimuth they cannot
        supply.  Position-dependent models override this with an azimuthal average."""
        return float(self.throughput(rho_as))

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

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
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

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
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

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        return 1.0 if self._thru is None else float(self._thru(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        from .metrics import kernel_from_profile
        st, _, ok = self.stamp(rho_as)
        if not ok:
            return None
        return kernel_from_profile(st, fwhm)


def cube_peak(cube: np.ndarray) -> float:
    """The level of the brightest pixels of ``cube`` as the float32 cube holds them: the
    99.9th percentile of ``|pixel|`` in a strided subsample of about two million pixels (what
    :func:`_check_float32_headroom` reads).  A percentile rather than the maximum, so that one
    hot pixel does not set the level: on NEAR night 6 the sample's maximum, one pixel at 1.9e3
    counts 29 px out, made the check warn that a 1e-6 source at 0.5" would be lost, where the
    pixels it lands on are of order 10 counts and the subtraction of such a chain was exact to
    3e-7 of its flux.  The percentile there is 67 counts."""
    if not cube.size:
        return float("nan")
    s = max(1, int(cube.size // 2_000_000))
    v = np.abs(np.asarray(cube.reshape(-1)[::s], np.float32))
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99.9)) if v.size else float("nan")


def _check_float32_headroom(cube: np.ndarray, sources: Sequence[Source], model: InjectionModel,
                            fallback: Optional[InjectionModel] = None, peak: Optional[float] = None) -> None:
    """Warn when an injection is too faint to survive being added to a float32 cube.

    The cube is held in float32, so adding a stamp whose pixels sit below the float32
    quantum of the science pixels they land on does nothing at all -- the source is not
    faint, it is GONE, and silently: on this data an injection whose stamp peaks at 1e-4
    counts loses a quarter of its flux and the loss grows as the contrast falls, so the
    recovered S/N stops being proportional to the contrast and every curve derived from it
    bends.  It is a real failure mode rather than a theoretical one: it is what happens the
    moment ``flux_unit`` is a normalised template's own sum (order 1) instead of the star's
    flux (order 1e6), which is the state a missing ``star_flux`` leaves you in.
    """
    if not cube.size:
        return
    # A subsample: this runs on every injection, and a full pass over a production cube
    # would cost real time for a number that only has to be right to an order of magnitude.
    # The brightest pixels are the star's and are in every frame, so striding is safe.  The
    # stride still touches most of the cube's cache lines (every 22nd pixel of a NEAR night
    # is a line in 1.4), so a caller that injects often passes the ``peak`` it computed once
    # (:func:`cube_peak`, which the reducer keeps).
    if peak is None:
        peak = cube_peak(cube)
    if not np.isfinite(peak) or peak <= 0:
        return
    quantum = peak * float(np.finfo(np.float32).eps)      # ULP of the brightest pixels
    for s in sources:
        m = model
        try:
            st, _, ok = model.stamp(s.rho)
            if not ok and fallback is not None:
                st, _, ok = fallback.stamp(s.rho)
                m = fallback
            if not ok:
                continue
        except Exception:                                  # the real error comes later
            continue
        amp = abs(s.contrast) * m.flux_unit * model.typical_throughput(s.rho) * float(np.nanmax(st))
        if amp < 32 * quantum:
            warnings.warn(
                f"injection at rho={s.rho:.3f}\" contrast={s.contrast:.3e} peaks at "
                f"{amp:.3e} counts, within float32 rounding of a cube whose brightest pixels reach "
                f"{peak:.3e}: most of its flux will be lost and the recovered S/N will not "
                f"scale with the contrast.  flux_unit is {m.flux_unit:.4g} -- if that is a "
                f"normalised template's own sum rather than the star's flux in the science "
                f"frames' units, set star_flux (see "
                f"generic.star_flux_from_aperture_photometry).",
                RuntimeWarning, stacklevel=3)
            return                                         # one warning per call is enough


def inject_sources(cube: np.ndarray, angles: np.ndarray, sources: Sequence[Source], model: InjectionModel,
                   pxscale: float, truenorth: float = 0.0, center: Optional[Tuple[float, float]] = None,
                   fallback: Optional[InjectionModel] = None, angle_convention: str = "pa",
                   copy: bool = True, frames: Optional[Sequence[int]] = None, threads: int = 1) -> np.ndarray:
    """Return ``cube`` with synthetic companions added to every frame.

    For each source: amplitude ``contrast * model.flux_unit * model.throughput(rho)``;
    per frame the stamp is placed at
    ``center + (rho/pxscale) * (cos az, sin az)`` with ``az = theta - truenorth - 270 -
    parang`` (``angle_convention='pa'``) or ``az = theta - parang`` (``'math'``).
    Anisotropic models (``refpa_deg`` set) have their stamp rotated CCW by
    ``az - refpa`` first.

    When the model's throughput depends on azimuth as well as separation
    (:func:`takes_azimuth`), it is evaluated per frame at that frame's ``az``, because the
    mask is fixed to the detector while the field rotates past it: a companion at one sky
    position angle is attenuated differently in every frame of a roll sequence.  For a
    round occulter the throughput is constant and is computed once.

    ``frames``: inject only these frames (indices into the cube); the others come back as
    they went in.  ``threads``: the frames go to that many threads (:func:`inject_frames`).
    """
    out = np.array(cube, dtype=np.float32, copy=copy)
    _check_float32_headroom(out, sources, model, fallback)
    inject_frames(out, out, angles, sources, model, pxscale, frames, truenorth=truenorth, center=center,
                  fallback=fallback, angle_convention=angle_convention, threads=threads)
    return out


@dataclass
class _Prepared:
    """What one source needs in every frame, worked out once: the stamp, the model it
    came from (the fallback where the model has no template there), and the scaled stamp
    shared by every frame when neither the throughput nor the PSF changes frame to frame."""

    src: Source
    model: InjectionModel           # the model the stamp came from
    stamp: np.ndarray               # unit stamp, float64
    center: Tuple[float, float]
    r_px: float
    per_frame: bool
    az_aware: bool
    thru: Optional[float]
    base: Optional[np.ndarray]      # stamp * contrast * flux_unit * throughput, or None


def _prepare(sources: Sequence[Source], model: InjectionModel, fallback: Optional[InjectionModel],
             pxscale: float) -> list:
    out = []
    for s in sources:
        st, sc, ok = model.stamp(s.rho)
        m = model
        if not ok:
            if fallback is None:
                rng = getattr(model, "seps", None)
                span = "" if rng is None or not len(rng) else \
                    f" (the model spans {float(rng[0]):.3f}-{float(rng[-1]):.3f} arcsec)"
                raise ValueError(
                    f"no PSF template at rho={s.rho:.3f} arcsec and no fallback model{span}. "
                    "Build the library over the separations the search injects into -- the "
                    "annulus, not just the known companion -- or pass fallback=.")
            st, sc, ok = fallback.stamp(s.rho)
            m = fallback
        st = np.asarray(st, float)
        per_frame = bool(getattr(model, "per_frame", False)) and ok
        # A position-dependent coronagraph attenuates a source by a different factor in
        # every frame: the mask is fixed to the detector and the field rotates past it, so
        # a companion at one sky PA crosses the quadrant boundaries as the telescope rolls.
        # Hoisting the throughput out of the frame loop is only correct for a round occulter.
        az_aware = takes_azimuth(model)
        thru = None if az_aware else float(model.throughput(s.rho))
        base = None if (az_aware or per_frame) else st * (s.contrast * m.flux_unit * thru)
        out.append(_Prepared(s, m, st, sc, s.rho / pxscale, per_frame, az_aware, thru, base))
    return out


def inject_frames(out: np.ndarray, cube: np.ndarray, angles: np.ndarray, sources: Sequence[Source],
                  model: InjectionModel, pxscale: float, frames: Optional[Sequence[int]] = None, *,
                  truenorth: float = 0.0, center: Optional[Tuple[float, float]] = None,
                  fallback: Optional[InjectionModel] = None, angle_convention: str = "pa",
                  threads: int = 1) -> None:
    """``out[j] = cube[j]`` with the sources added, for every frame ``j`` of ``frames`` (all
    by default), the numbers :func:`inject_sources` gives.  ``out`` is a float32 cube of
    the same shape and may be ``cube`` itself.  Frames not listed are not touched.

    Each frame's stamps are added in the order of ``sources``, as before.  The frames go
    to ``threads`` threads in blocks, and each block writes only its own frames.  Within a
    block, a source whose stamp is the same in every frame is rotated to all the block's
    frames by one :func:`klip_tpe.klip.rotate_ccw_many` call, which runs without the GIL.
    The shift and the add stay frame by frame."""
    from .klip import frame_blocks, run_blocks
    prepared = _prepare(sources, model, fallback, pxscale)
    n, ny, nx = out.shape
    fr = np.arange(n) if frames is None else np.asarray(frames, dtype=np.intp).reshape(-1)
    if not prepared and out is cube:
        return
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    angles = np.asarray(angles, float)
    copy_in = out is not cube

    def block(lo, hi):
        J = fr[lo:hi].tolist()
        if copy_in:
            for j in J:
                out[j] = cube[j]
        for p in prepared:
            if p.base is not None:
                _add_shared(out, J, p, angles, truenorth, center, angle_convention)
            else:
                _add_each(out, J, p, model, angles, truenorth, center, angle_convention)
    # a block's temporaries are a few stamp-sized arrays per frame; keep them near 1 MB
    px = max((int(p.stamp.size) for p in prepared), default=0)
    run_blocks(block, frame_blocks(fr.size, threads, 4 * px), threads)


def _azimuth(s: Source, angle: float, truenorth: float, angle_convention: str):
    if angle_convention == "pa":
        return s.theta - truenorth - 270.0 - angle
    return s.theta - angle


def _place_shared(J, p: _Prepared, angles, truenorth, center, angle_convention, shape, sink) -> None:
    """Place a source whose scaled stamp is the same in every frame on the frames ``J``:
    ``sink(i, crop)`` receives the :func:`stamp_crop` of frame ``J[i]`` (a view into a
    buffer the next frame reuses)."""
    s, m = p.src, p.model
    k = len(J)
    xs, ys = np.empty(k), np.empty(k)
    rot = []
    for i, j in enumerate(J):
        az = _azimuth(s, angles[j], truenorth, angle_convention)
        azr = np.deg2rad(az)
        xs[i], ys[i] = center[0] + p.r_px * np.cos(azr), center[1] + p.r_px * np.sin(azr)
        if m.refpa_deg is not None:
            rot.append(az - m.refpa_deg)
    if m.refpa_deg is not None:
        from .klip import rotate_ccw_many
        stamps = rotate_ccw_many(p.base, rot, cval=0.0)
    else:
        stamps = None
    buf = np.empty(p.base.shape)
    for i in range(k):
        c = stamp_crop(shape, p.base if stamps is None else stamps[i], p.center, xs[i], ys[i], buf)
        if c is not None:
            sink(i, c)


def _place_each(J, p: _Prepared, model, angles, truenorth, center, angle_convention, shape, sink) -> None:
    """Place a source whose stamp or throughput changes frame to frame on the frames ``J``
    (``sink`` as for :func:`_place_shared`)."""
    s, m = p.src, p.model
    sc = p.center
    buf = None
    for i, j in enumerate(J):
        az = _azimuth(s, angles[j], truenorth, angle_convention)
        azr = np.deg2rad(az)
        xs, ys = center[0] + p.r_px * np.cos(azr), center[1] + p.r_px * np.sin(azr)
        t_j = float(model.throughput(s.rho, az)) if p.az_aware else p.thru
        if p.per_frame:
            stj, sc, fu_j = model.frame_stamp(j, s.rho)
            stamp = np.asarray(stj, float) * (s.contrast * fu_j * t_j)
        else:
            stamp = p.stamp * (s.contrast * m.flux_unit * t_j)
        if m.refpa_deg is not None:
            from .klip import rotate_ccw
            stamp = rotate_ccw(stamp, az - m.refpa_deg, cval=0.0)
        if buf is None or buf.shape != stamp.shape:
            buf = np.empty(stamp.shape)
        c = stamp_crop(shape, stamp, sc, xs, ys, buf)
        if c is not None:
            sink(i, c)


def place_source(J, p: _Prepared, model, angles, truenorth, center, angle_convention, shape, sink) -> None:
    """One prepared source (:func:`prepare_sources`) on the frames ``J`` of a cube of frame
    ``shape``: ``sink(i, crop)`` receives what frame ``J[i]`` gets (:func:`stamp_crop`), with
    the numbers :func:`inject_frames` adds.  ``model`` is the injection model (the
    throughput's), ``center`` the star's ``(x, y)``."""
    if p.base is not None:
        _place_shared(J, p, angles, truenorth, center, angle_convention, shape, sink)
    else:
        _place_each(J, p, model, angles, truenorth, center, angle_convention, shape, sink)


def prepare_sources(sources: Sequence[Source], model: InjectionModel, fallback: Optional[InjectionModel],
                    pxscale: float) -> list:
    """What each source needs in every frame, worked out once (for :func:`place_source`)."""
    return _prepare(sources, model, fallback, pxscale)


def _add_shared(out, J, p: _Prepared, angles, truenorth, center, angle_convention) -> None:
    """Add a source whose scaled stamp is the same in every frame to the frames ``J``."""
    _place_shared(J, p, angles, truenorth, center, angle_convention, out.shape[1:],
                  lambda i, c: add_crop(out[J[i]], c))


def _add_each(out, J, p: _Prepared, model, angles, truenorth, center, angle_convention) -> None:
    """Add a source whose stamp or throughput changes frame to frame to the frames ``J``."""
    _place_each(J, p, model, angles, truenorth, center, angle_convention, out.shape[1:],
                lambda i, c: add_crop(out[J[i]], c))
