"""Synthetic-companion injection into pupil-tracking cubes.

An :class:`InjectionModel` gives, for a separation, the unit-normalized off-axis PSF and the
fraction of its flux that survives the coronagraph.  :func:`inject_sources` adds the scaled
stamps to every frame at the detector azimuth each frame's parallactic angle implies, rotating
the stamp of an anisotropic PSF so its distortion axis stays radial.

A source at position angle ``theta`` (degrees East of North) sits at detector azimuth
``az = theta - truenorth - 270 - parang`` from +x.  After derotation, counter-clockwise by
``parang + truenorth``, it lands at ``theta + 90`` degrees from +x, at PA ``theta``.

Notes
-----
The azimuth convention is that of ``do_inject`` in ``reduce_near_2.pro``, the IDL predecessor.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .metrics import Source

__all__ = ["InjectionModel", "GaussianPSF", "TemplatePSF", "AiryPSF", "FramePSF", "LibraryPSF", "inject_sources",
           "inject_frames", "shift_bilinear", "add_stamp", "takes_azimuth", "read_sources", "injected_model"]


def read_sources(path: str, columns: int = 3):
    """Read point sources from a text file, one ``rho pa contrast`` per line.

    Whitespace or commas separate the values, ``#`` starts a comment and blank lines are skipped.
    Values after the third are ignored here (:func:`klip_tpe.chainfit.read_chain` reads a node
    label from the fourth).

    Parameters
    ----------
    path : str
        Path of the text file.  ``rho`` is in arcsec, ``pa`` in degrees East of North and
        ``contrast`` in the injection model's flux unit, such as the flux of alpha Cen A for NEAR.
    columns : int, default 3
        Minimum number of values on a line.

    Returns
    -------
    sources : list of Source
        One :class:`~klip_tpe.metrics.Source` per line.

    Raises
    ------
    ValueError
        If a line has too few values, values that are not numbers, a negative or non-finite
        ``rho``, or a non-finite ``pa`` or ``contrast``.
    """
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
    """Tell whether a model's throughput depends on azimuth as well as separation.

    The answer is the class attribute ``azimuth_dependent``.  Every model here accepts ``az_deg``
    for a uniform signature, so the signature cannot tell which models need it.  A model written
    elsewhere that uses the azimuth sets ``azimuth_dependent = True`` to be called with one.

    Parameters
    ----------
    model : InjectionModel
        The injection model.

    Returns
    -------
    dependent : bool
        True when :func:`inject_sources` must evaluate the throughput per frame at that frame's
        azimuth.
    """
    return bool(getattr(model, "azimuth_dependent", False))


def shift_bilinear(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift an image by a fraction of a pixel with bilinear interpolation.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, ny by nx pixels.
    dx : float
        Shift along x, in pixels.
    dy : float
        Shift along y, in pixels.

    Returns
    -------
    shifted : ndarray, shape (ny, nx)
        Shifted image, in float64.  Pixels shifted in from outside the image are 0.

    Notes
    -----
    It matches the fractional shift of ``fshift`` in the IDL predecessor.
    """
    return ndimage.shift(np.asarray(img, float), (dy, dx), order=1, mode="constant", cval=0.0)


def stamp_crop(shape: Tuple[int, int], stamp: np.ndarray, stamp_center: Tuple[float, float],
               x: float, y: float, buf: Optional[np.ndarray] = None):
    """Compute where :func:`add_stamp` puts a stamp in a frame and the values it adds there.

    Parameters
    ----------
    shape : tuple of int
        Shape of the frame, ``(ny, nx)``.
    stamp : ndarray, shape (sy, sx)
        The stamp, sy by sx pixels.
    stamp_center : tuple of 2 floats
        The ``(x, y)`` stamp pixel that lands at ``(x, y)`` in the frame.
    x : float
        Frame column of the stamp's center, in pixels.
    y : float
        Frame row of the stamp's center, in pixels.
    buf : ndarray or None, default None
        A float64 array of the stamp's shape that receives the shifted stamp.  None allocates one.

    Returns
    -------
    crop : tuple or None
        ``(y0, y1, x0, x1, values)``, the frame's rows ``y0:y1`` and columns ``x0:x1`` and the
        shifted stamp's float64 values over them, or None when the stamp falls outside the frame.
        ``values`` is a view into ``buf`` when one is given.
    """
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
    """Add a :func:`stamp_crop` to a frame in place.

    A float32 frame sums the float64 values in double and rounds once, as ``+=`` does.

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        The frame, changed in place.
    crop : tuple
        The ``(y0, y1, x0, x1, values)`` that :func:`stamp_crop` gives.
    """
    y0, y1, x0, x1, v = crop
    frame[y0:y1, x0:x1] += v


def add_stamp(frame: np.ndarray, stamp: np.ndarray, stamp_center: Tuple[float, float],
              x: float, y: float, buf: Optional[np.ndarray] = None) -> None:
    """Add a stamp to a frame in place, shifting it to a sub-pixel position.

    The integer part of the position is taken by placement and the fractional part by a
    bilinear shift.  The stamp is clipped at the frame's edges.

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        The frame, changed in place.
    stamp : ndarray, shape (sy, sx)
        The stamp, sy by sx pixels.
    stamp_center : tuple of 2 floats
        The ``(x, y)`` stamp pixel that lands at ``(x, y)`` in the frame.
    x : float
        Frame column of the stamp's center, in pixels.
    y : float
        Frame row of the stamp's center, in pixels.
    buf : ndarray or None, default None
        A float64 array of the stamp's shape that receives the shifted stamp, so a caller adding
        many stamps does not allocate one each time.  None allocates one.

    Notes
    -----
    It does what ``near2_psfstamp`` does in the IDL predecessor.
    """
    c = stamp_crop(frame.shape, stamp, stamp_center, x, y, buf)
    if c is not None:
        add_crop(frame, c)


class InjectionModel:
    """Base class of the injection models, which give an injected companion's PSF and throughput.

    A model answers two questions for a separation ``rho_as``: what an off-axis PSF looks like
    there (:meth:`stamp`) and what fraction of its flux survives the coronagraph
    (:meth:`throughput`).  A companion of contrast ``c`` carries ``c * flux_unit * throughput``.

    Attributes
    ----------
    flux_unit : float
        Flux of the star, in data units, in the normalization of the stamps.
    refpa_deg : float or None
        Reference axis of the stamps, in degrees.  When set, each stamp is rotated so this axis
        follows the source's detector azimuth, for an anisotropic PSF.
    per_frame : bool
        Whether :func:`inject_sources` takes each frame's own stamp from :meth:`frame_stamp`.
    azimuth_dependent : bool
        Whether :meth:`throughput` depends on the detector azimuth as well as the separation.
    name : str
        Name of the model, for a run's record.
    """

    flux_unit: float = 1.0
    refpa_deg: Optional[float] = None
    per_frame: bool = False          # True -> inject_sources calls frame_stamp(j, rho) for every frame j
    azimuth_dependent: bool = False  # True -> throughput() is evaluated per frame, with that frame's azimuth
    name = "injection_model"

    def stamp(self, rho_as: float):  # pragma: no cover
        """Return the unit-normalized PSF stamp at a separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray or None
            PSF template normalized so the flux unit is 1, within an encircled-energy radius or in
            total depending on the model.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the PSF's center.
        ok : bool
            False when the model has no template there, and the caller falls back to another model.
            The throughput still comes from this model.
        """
        raise NotImplementedError

    def frame_stamp(self, j: int, rho_as: float):  # pragma: no cover
        """Return one frame's own PSF stamp and star flux, for per-frame models.

        Parameters
        ----------
        j : int
            Index of the frame.
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray
            The frame's PSF, unit-normalized.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the PSF's center.
        flux_unit_j : float
            Star flux of that frame in the same normalization, so the contrast follows the
            transparency and Strehl ratio frame by frame.
        """
        raise NotImplementedError

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        """Return the fraction of a point source's flux that survives the coronagraph.

        Most coronagraphs are round and ignore the azimuth.  A four-quadrant phase mask does not, and
        on MIRI the throughput varies by a factor of two to four around a circle of constant
        separation (see :mod:`klip_tpe.instruments.miri`).

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        az_deg : float or None, default None
            Azimuth of the source in the detector frame, in degrees.  A model that uses it sets
            ``azimuth_dependent`` True (:func:`takes_azimuth`).

        Returns
        -------
        throughput : float
            Fraction of the flux transmitted, 1 for this base class.
        """
        return 1.0

    def typical_throughput(self, rho_as: float) -> float:
        """Return a representative throughput at a separation.

        It serves diagnostics that need a scale rather than a value and cannot supply an azimuth.
        Position-dependent models override it with an azimuthal average.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        throughput : float
            Fraction of the flux transmitted.
        """
        return float(self.throughput(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        """Return a measured matched filter for :class:`~klip_tpe.metrics.MawetPeakSNR`.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        fwhm : float
            FWHM of the PSF, in pixels.

        Returns
        -------
        kernel : ndarray or None
            Unit-sum kernel, or None for the metric's Gaussian, which this base class gives.
        """
        return None

    def describe(self) -> dict:
        """Return the model's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, ``flux_unit`` and ``refpa_deg``, with more keys in subclasses.
        """
        return {"name": self.name, "flux_unit": float(self.flux_unit), "refpa_deg": self.refpa_deg}


class GaussianPSF(InjectionModel):
    """Analytic Gaussian PSF of unit total flux.

    Parameters
    ----------
    fwhm_px : float
        FWHM of the Gaussian, in pixels.
    star_flux : float, default 1.0
        Total flux of the star in data units, which becomes ``flux_unit``.
    size : int, default 41
        Width of the stamp in pixels, made odd.
    """

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
        """Return the Gaussian stamp, the same at every separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray, shape (size, size)
            Gaussian of unit sum.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of its center.
        ok : bool
            Always True.
        """
        return self._stamp, self._center, True


class TemplatePSF(InjectionModel):
    """Empirical PSF template, such as an unsaturated median PSF.

    The stamp is normalized to unit flux inside ``ee_radius_px``, or to unit total.  By default
    ``flux_unit`` is the template's own flux in that normalization, so the contrast is relative to
    the template star.

    Parameters
    ----------
    template : ndarray, shape (sy, sx)
        The PSF image.  NaN pixels become 0.
    center : tuple of 2 floats or None, default None
        The ``(x, y)`` pixel of the PSF's center.  None takes the template's middle,
        ``((sx-1)/2, (sy-1)/2)``.
    ee_radius_px : float or None, default None
        Radius of the unit-flux normalization, in pixels.  None normalizes the total.
    star_flux : float or None, default None
        Flux of the science star in the same normalization.  None takes the template's flux.
    throughput_fn : callable or None, default None
        Throughput as a function of separation in arcsec.  None gives 1.

    Attributes
    ----------
    template_flux : float
        Flux of the template in the normalization, before it is divided out.
    """

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
        """Return the normalized template, the same at every separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray, shape (sy, sx)
            Normalized template.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the PSF's center.
        ok : bool
            Always True.
        """
        return self._stamp, self._center, True

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        """Return the throughput at a separation from ``throughput_fn``.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        az_deg : float or None, default None
            Detector azimuth, in degrees, ignored.

        Returns
        -------
        throughput : float
            ``throughput_fn(rho_as)``, or 1 without one.
        """
        return 1.0 if self._thru is None else float(self._thru(rho_as))


class AiryPSF(InjectionModel):
    """Obstructed, possibly elliptical Airy PSF, the ``airy_disk`` model pyNOMIC fits to NOMIC frames.

    The stamp is normalized to unit flux within ``ee_radius_px``, and ``flux_unit`` is the star's
    flux in that normalization for amplitude ``amp``.  An injected companion of contrast ``c`` is
    then exactly pyNOMIC's ``airy_disk(c * amp, ...)``.

    Parameters
    ----------
    sigmax : float
        pyNOMIC's scale parameter along the first axis, in pixels.  The first dark ring of an
        unobstructed PSF sits at ``1.22 * sigmax``.
    sigmay : float or None, default None
        Scale parameter along the second axis, in pixels.  None takes ``sigmax``.
    p : float, default 0.0
        Rotation of the axes, in radians.
    amp : float, default 1.0
        Amplitude of the star's fit, in data units.
    e : float, default 0.11
        Ratio of the secondary obscuration's diameter to the primary's.
    size : int or None, default None
        Width of the stamp in pixels, made odd.  None takes 6 times the larger scale.
    ee_radius_px : float or None, default None
        Radius of the unit-flux normalization, in pixels.  None takes 2.5 times the larger scale.
    frame_params : ndarray or None, default None
        pyNOMIC ``reffits`` rows ``(amp, sigmax, sigmay, offset, p, ...)``, shape (n, 5 or more),
        one per frame.  With them, frame ``j`` is injected with its own fit, as pyNOMIC's
        ``inject_source`` does, and the other parameters describe the median.

    Attributes
    ----------
    fwhm_px : float
        FWHM of the circularized core, ``1.028 * (sigmax + sigmay) / 2``, in pixels.
    n_bad : int
        Number of frames whose fit is unusable and that take the median stamp, set with
        ``frame_params``.
    """

    name = "airy"

    def __init__(self, sigmax: float, sigmay: Optional[float] = None, p: float = 0.0, amp: float = 1.0,
                 e: float = 0.11, size: Optional[int] = None, ee_radius_px: Optional[float] = None,
                 frame_params: Optional[np.ndarray] = None):
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
        """Evaluate the obstructed Airy pattern on the stamp's pixels.

        Parameters
        ----------
        amp : float
            Amplitude, in data units.
        sx : float
            Scale parameter along the first axis, in pixels.
        sy : float
            Scale parameter along the second axis, in pixels.
        p : float
            Rotation of the axes, in radians.

        Returns
        -------
        model : ndarray, shape (size, size)
            The PSF, ``amp`` at its center.
        """
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
        """Sum a stamp within ``ee_radius_px`` of its center.

        Parameters
        ----------
        star : ndarray, shape (size, size)
            The stamp.

        Returns
        -------
        flux : float
            Flux within the radius.
        """
        c = self._center[0]
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        return float(star[np.hypot(xx - c, yy - c) <= self.ee_radius_px].sum())

    def stamp(self, rho_as: float):
        """Return the median fit's stamp, the same at every separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray, shape (size, size)
            The PSF, of unit flux within ``ee_radius_px``.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of its center.
        ok : bool
            Always True.
        """
        return self._stamp, self._center, True

    def frame_stamp(self, j: int, rho_as: float):
        """Return frame ``j``'s own Airy fit, of unit core flux, and that frame's star flux.

        A frame without a usable fit takes the median stamp and ``flux_unit``.

        Parameters
        ----------
        j : int
            Index of the frame.
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray, shape (size, size)
            The frame's PSF, in float32.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of its center.
        flux_unit_j : float
            Core flux of the frame's star, in data units.
        """
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
        """Return the model's settings, for a run's record.

        Returns
        -------
        info : dict
            The base keys plus the fit's parameters, ``ee_radius_px`` and ``per_frame``.
        """
        d = super().describe()
        d.update(sigmax=self.sigmax, sigmay=self.sigmay, p=self.p, amp=self.amp, e=self.e,
                 ee_radius_px=self.ee_radius_px, per_frame=bool(self.per_frame))
        return d


class FramePSF(InjectionModel):
    """Frame-by-frame empirical PSF, each frame's own unsaturated star as the template for that frame.

    For frame ``j`` the stamp is the cut-out of radius ``r_stamp`` around the star, tapered by a
    cosine between ``r_ee`` and ``r_stamp`` so the halo and speckles beyond the core fade to zero,
    and normalized to unit flux within ``r_ee``.  The frames must be registered and
    background-subtracted, as pyNOMIC's ``frame_registration`` and ``subtract_background`` leave
    them.  A frame whose core flux is not positive takes the median template.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each, with the star at ``center``.
    r_ee : float
        Radius of the unit-flux normalization, in pixels.
    r_stamp : float or None, default None
        Radius of the cut-out and outer edge of the taper, in pixels.  None takes ``1.6 * r_ee``.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``.
    throughput_fn : callable or None, default None
        Throughput as a function of separation in arcsec.  None gives 1.

    Attributes
    ----------
    frame_flux : ndarray, shape (n,)
        Core flux of each frame's star, in data units.
    flux_unit : float
        Median of ``frame_flux`` over the good frames.
    n_bad : int
        Number of frames that take the median template.

    Raises
    ------
    ValueError
        If the cut-out does not fit in the frame.
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
        """Return the median template, used where no frame is named and for the matched filter.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray
            Median template, of unit flux within ``r_ee``, in float32.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the star.
        ok : bool
            Always True.
        """
        return self._median_stamp, self._center, True

    def frame_stamp(self, j: int, rho_as: float):
        """Return frame ``j``'s own stamp and star flux.

        Parameters
        ----------
        j : int
            Index of the frame.
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray
            The frame's stamp, of unit flux within ``r_ee``, or the median template for a bad frame.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the star.
        flux_unit_j : float
            Core flux of the frame's star, or ``flux_unit`` for a bad frame.
        """
        if self._good[j]:
            return self._stamps[j], self._center, float(self.frame_flux[j])
        return self._median_stamp, self._center, float(self.flux_unit)

    def throughput(self, rho_as: float, az_deg: Optional[float] = None) -> float:
        """Return the throughput at a separation from ``throughput_fn``.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        az_deg : float or None, default None
            Detector azimuth, in degrees, ignored.

        Returns
        -------
        throughput : float
            ``throughput_fn(rho_as)``, or 1 without one.
        """
        return 1.0 if self._thru is None else float(self._thru(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        """Return the matched filter built from the median template.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec, ignored.
        fwhm : float
            FWHM of the PSF, in pixels.

        Returns
        -------
        kernel : ndarray
            Azimuthally averaged unit-sum kernel (:func:`klip_tpe.metrics.kernel_from_profile`).
        """
        from .metrics import kernel_from_profile
        return kernel_from_profile(self._median_stamp, fwhm)

    def describe(self) -> dict:
        """Return the model's settings, for a run's record.

        Returns
        -------
        info : dict
            The base keys plus ``r_ee``, ``r_stamp``, ``n_frames`` and ``n_bad``.
        """
        d = super().describe()
        d.update(r_ee=self.r_ee, r_stamp=self.r_stamp, n_frames=int(self._good.size), n_bad=self.n_bad)
        return d


class LibraryPSF(InjectionModel):
    """Separation-dependent PSF library, interpolated linearly between its slices.

    Each stamp is renormalized to unit flux within ``ee_radius_px`` of ``center``.  Outside the
    library's separations :meth:`stamp` reports no template.

    Parameters
    ----------
    slices : ndarray, shape (m, sy, sx)
        The m PSF stamps, ``slices[i]`` valid at ``seps[i]``.  A float64 array is normalized in
        place.
    seps : sequence of float
        Separation of each slice, in arcsec, increasing.
    center : tuple of 2 floats
        The ``(x, y)`` stamp pixel of the PSF's center.
    ee_radius_px : float
        Radius of the unit-flux normalization, in pixels.
    throughput_fn : callable or None, default None
        Throughput as a function of separation in arcsec.  None gives 1.
    refpa_deg : float or None, default None
        Reference axis of the slices, in degrees, for an anisotropic PSF.  None leaves the stamps
        unrotated.
    flux_unit : float, default 1.0
        Flux of the star in the normalization of the slices, in data units.
    """

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
        """Return the library's stamp at a separation, interpolated between the two nearest slices.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        stamp : ndarray or None
            Interpolated stamp of unit flux within ``ee_radius_px``, or None outside ``seps``.
        center : tuple of 2 floats
            The ``(x, y)`` stamp pixel of the PSF's center.
        ok : bool
            False outside ``[seps[0], seps[-1]]`` or for a non-finite ``rho_as``.
        """
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
        """Return the throughput at a separation from ``throughput_fn``.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        az_deg : float or None, default None
            Detector azimuth, in degrees, ignored.

        Returns
        -------
        throughput : float
            ``throughput_fn(rho_as)``, or 1 without one.
        """
        return 1.0 if self._thru is None else float(self._thru(rho_as))

    def matched_filter_kernel(self, rho_as: float, fwhm: float):
        """Return the matched filter built from the stamp at a separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.
        fwhm : float
            FWHM of the PSF, in pixels.

        Returns
        -------
        kernel : ndarray or None
            Azimuthally averaged unit-sum kernel (:func:`klip_tpe.metrics.kernel_from_profile`), or
            None outside the library.
        """
        from .metrics import kernel_from_profile
        st, _, ok = self.stamp(rho_as)
        if not ok:
            return None
        return kernel_from_profile(st, fwhm)


def cube_peak(cube: np.ndarray) -> float:
    """Return the level of a cube's brightest pixels, as the float32 cube holds them.

    It is the 99.9th percentile of ``|pixel|`` in a strided subsample of about two million pixels,
    which :func:`_check_float32_headroom` reads.  A percentile rather than the maximum keeps one
    hot pixel from setting the level.

    Parameters
    ----------
    cube : ndarray
        The cube, of any shape.

    Returns
    -------
    peak : float
        The level, in data units, or NaN for an empty cube or one without finite pixels.

    Notes
    -----
    On NEAR night 6 the subsample's maximum, one pixel at 1.9e3 counts 29 px out, made the check
    warn that a 1e-6 source at 0.5" would be lost, where the pixels it lands on are of order 10
    counts and the subtraction of such a chain was exact to 3e-7 of its flux.  The percentile there
    is 67 counts.
    """
    if not cube.size:
        return float("nan")
    s = max(1, int(cube.size // 2_000_000))
    v = np.abs(np.asarray(cube.reshape(-1)[::s], np.float32))
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99.9)) if v.size else float("nan")


def _check_float32_headroom(cube: np.ndarray, sources: Sequence[Source], model: InjectionModel,
                            fallback: Optional[InjectionModel] = None, peak: Optional[float] = None) -> None:
    """Warn when an injection is too faint to survive being added to a float32 cube.

    A stamp whose pixels sit below the float32 quantum of the science pixels it lands on is lost
    when added, silently.  The recovered S/N then stops scaling with the contrast, and every curve
    derived from it bends.  That happens when ``flux_unit`` is a normalized template's own sum
    instead of the star's flux, the state a missing ``star_flux`` leaves.

    Parameters
    ----------
    cube : ndarray
        The float32 cube the sources go into.
    sources : sequence of Source
        The sources to inject.
    model : InjectionModel
        The injection model.
    fallback : InjectionModel or None, default None
        Model that supplies the stamp where ``model`` has none.
    peak : float or None, default None
        Level of the cube's brightest pixels, as :func:`cube_peak` gives it.  None computes it.

    Warns
    -----
    RuntimeWarning
        Once per call, when a source's peak is under 32 float32 quanta of ``peak``.
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
    """Return a cube with synthetic companions added to every frame.

    Each source's amplitude is ``contrast * flux_unit * throughput(rho)``, and in each frame its
    stamp goes to ``center + (rho / pxscale) * (cos az, sin az)``.  A model with ``refpa_deg`` set
    has its stamp rotated counter-clockwise by ``az - refpa_deg`` first.  A model whose throughput
    depends on azimuth (:func:`takes_azimuth`) is evaluated per frame at that frame's ``az``, since
    the mask is fixed to the detector while the field rotates past it.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each, not derotated.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    sources : sequence of Source
        Companions to inject, each with its separation in arcsec, position angle and contrast.
    model : InjectionModel
        PSF and throughput of the companions.
    pxscale : float
        Plate scale, in arcsec per pixel.
    truenorth : float or ndarray, default 0.0
        True-north offset, in degrees.  One value, or one per frame.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``.
    fallback : InjectionModel or None, default None
        Model that supplies the stamp and flux unit where ``model`` has no template.  The
        throughput still comes from ``model``, and without a fallback a missing template raises
        ValueError.
    angle_convention : {"pa", "math"}, default "pa"
        How a source's angle ``theta`` sets its detector azimuth.  "pa" gives
        ``az = theta - truenorth - 270 - parang``, so the source lands at PA ``theta`` after
        derotation, and "math" gives ``az = theta - parang``.
    copy : bool, default True
        Whether to copy ``cube``.  False injects in place when ``cube`` is already float32.
    frames : sequence of int or None, default None
        Indices of the frames to inject.  The others come back as they went in, and None injects
        all.
    threads : int, default 1
        Number of threads the frames are spread over.

    Returns
    -------
    out : ndarray, shape (n, ny, nx)
        The cube with the companions added, in float32.

    Raises
    ------
    ValueError
        If a source has no template and no fallback, or ``truenorth`` has a per-frame value count
        other than n.

    Warns
    -----
    RuntimeWarning
        When a source is too faint to survive float32 rounding (:func:`_check_float32_headroom`).
    """
    out = np.array(cube, dtype=np.float32, copy=copy)
    _check_float32_headroom(out, sources, model, fallback)
    inject_frames(out, out, angles, sources, model, pxscale, frames, truenorth=truenorth, center=center,
                  fallback=fallback, angle_convention=angle_convention, threads=threads)
    return out


@dataclass
class _Prepared:
    """What one source needs in every frame, worked out once.

    Attributes
    ----------
    src : Source
        The source.
    model : InjectionModel
        Model the stamp came from, the fallback where the main model has no template.
    stamp : ndarray
        Unit stamp, in float64.
    center : tuple of 2 floats
        The ``(x, y)`` stamp pixel of the PSF's center.
    r_px : float
        Separation, in pixels.
    per_frame : bool
        Whether each frame takes its own stamp from the model's ``frame_stamp``.
    az_aware : bool
        Whether the throughput is evaluated per frame at the frame's azimuth.
    thru : float or None
        Throughput at the source's separation, or None when it depends on azimuth.
    base : ndarray or None
        Stamp scaled by contrast, flux unit and throughput, shared by every frame.  None when the
        stamp or the throughput changes frame to frame.
    """

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
    """Work out what each source needs in every frame.

    Parameters
    ----------
    sources : sequence of Source
        The sources to inject.
    model : InjectionModel
        The injection model.
    fallback : InjectionModel or None
        Model that supplies the stamp where ``model`` has no template.
    pxscale : float
        Plate scale, in arcsec per pixel.

    Returns
    -------
    prepared : list of _Prepared
        One entry per source, in order.

    Raises
    ------
    ValueError
        If ``model`` has no template at a source's separation and there is no fallback.
    """
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
    """Set ``out[j]`` to ``cube[j]`` with the sources added, for every listed frame ``j``.

    The numbers are those :func:`inject_sources` gives, each frame's stamps added in the order of
    ``sources``.  The frames go to the threads in blocks, and each block writes only its own
    frames.  Within a block, a stamp that is the same in every frame is rotated to all of them by
    one :func:`klip_tpe.klip.rotate_ccw_many` call.

    Parameters
    ----------
    out : ndarray, shape (n, ny, nx)
        The float32 cube written, which may be ``cube`` itself.  Frames not listed are not touched.
    cube : ndarray, shape (n, ny, nx)
        The n frames to inject into, ny by nx pixels each.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    sources : sequence of Source
        The sources to inject.
    model : InjectionModel
        PSF and throughput of the sources.
    pxscale : float
        Plate scale, in arcsec per pixel.
    frames : sequence of int or None, default None
        Indices of the frames to inject.  None injects all.
    truenorth : float or ndarray, default 0.0
        True-north offset, in degrees.  One value, or one per frame of ``angles``.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``.
    fallback : InjectionModel or None, default None
        Model that supplies the stamp where ``model`` has no template.
    angle_convention : {"pa", "math"}, default "pa"
        How a source's angle sets its detector azimuth, as for :func:`inject_sources`.
    threads : int, default 1
        Number of threads.
    """
    from .klip import frame_blocks, run_blocks
    prepared = _prepare(sources, model, fallback, pxscale)
    n, ny, nx = out.shape
    fr = np.arange(n) if frames is None else np.asarray(frames, dtype=np.intp).reshape(-1)
    if not prepared and out is cube:
        return
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    angles = np.asarray(angles, float)
    truenorth = _north_values(truenorth, angles)
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
    """Return a source's detector azimuth in one frame.

    Parameters
    ----------
    s : Source
        The source.
    angle : float
        Parallactic angle of the frame, in degrees.
    truenorth : float
        True-north offset of the frame, in degrees.
    angle_convention : {"pa", "math"}
        "pa" gives ``theta - truenorth - 270 - angle``, any other value ``theta - angle``.

    Returns
    -------
    az : float
        Azimuth from +x, in degrees.
    """
    if angle_convention == "pa":
        return s.theta - truenorth - 270.0 - angle
    return s.theta - angle


def _north_values(truenorth, angles):
    """Return ``truenorth`` as the injection uses it, one angle or one per frame.

    Parameters
    ----------
    truenorth : float or array_like
        True-north offset, in degrees.  One value, or one per frame of ``angles``.
    angles : ndarray, shape (n,)
        Parallactic angles of the n frames, in degrees.

    Returns
    -------
    north : float or ndarray
        The one value as given, or a float array of length n.

    Raises
    ------
    ValueError
        If there are per-frame values but not n of them.
    """
    if np.ndim(truenorth) == 0:
        return truenorth
    tn = np.asarray(truenorth, float).ravel()
    if tn.shape[0] != np.asarray(angles).shape[0]:
        raise ValueError(f"{tn.shape[0]} true-north values for {np.asarray(angles).shape[0]} frames")
    return tn


def _north_at(truenorth, j):
    """Return frame ``j``'s true north, the one angle or the frame's own.

    Parameters
    ----------
    truenorth : float or ndarray
        True-north offset, as :func:`_north_values` gives it.
    j : int
        Index of the frame.

    Returns
    -------
    north : float
        The frame's true-north offset, in degrees.
    """
    return truenorth if np.ndim(truenorth) == 0 else truenorth[j]


def _place_shared(J, p: _Prepared, angles, truenorth, center, angle_convention, shape, sink) -> None:
    """Place a source whose scaled stamp is the same in every frame on the frames ``J``.

    Parameters
    ----------
    J : list of int
        Indices of the frames.
    p : _Prepared
        The source, with ``base`` set.
    angles : ndarray
        Parallactic angle of each frame of the cube, in degrees.
    truenorth : float or ndarray
        True-north offset, as :func:`_north_values` gives it.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    angle_convention : {"pa", "math"}
        How the source's angle sets its detector azimuth (:func:`_azimuth`).
    shape : tuple of int
        Shape of a frame, ``(ny, nx)``.
    sink : callable
        Function called as ``sink(i, crop)`` with the :func:`stamp_crop` of frame ``J[i]``.  The
        crop's values are a view into a buffer the next frame reuses.
    """
    s, m = p.src, p.model
    k = len(J)
    xs, ys = np.empty(k), np.empty(k)
    rot = []
    for i, j in enumerate(J):
        az = _azimuth(s, angles[j], _north_at(truenorth, j), angle_convention)
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
    """Place a source whose stamp or throughput changes frame to frame on the frames ``J``.

    Parameters
    ----------
    J : list of int
        Indices of the frames.
    p : _Prepared
        The source.
    model : InjectionModel
        The injection model, which gives the throughput and any per-frame stamp.
    angles : ndarray
        Parallactic angle of each frame of the cube, in degrees.
    truenorth : float or ndarray
        True-north offset, as :func:`_north_values` gives it.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    angle_convention : {"pa", "math"}
        How the source's angle sets its detector azimuth (:func:`_azimuth`).
    shape : tuple of int
        Shape of a frame, ``(ny, nx)``.
    sink : callable
        Function called as ``sink(i, crop)`` with the :func:`stamp_crop` of frame ``J[i]``.
    """
    s, m = p.src, p.model
    sc = p.center
    buf = None
    for i, j in enumerate(J):
        az = _azimuth(s, angles[j], _north_at(truenorth, j), angle_convention)
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
    """Place one prepared source on the frames ``J``, with the numbers :func:`inject_frames` adds.

    Parameters
    ----------
    J : list of int
        Indices of the frames.
    p : _Prepared
        The source, from :func:`prepare_sources`.
    model : InjectionModel
        The injection model, which gives the throughput.
    angles : ndarray
        Parallactic angle of each frame of the cube, in degrees.
    truenorth : float or ndarray
        True-north offset, in degrees.  One value, or one per frame of ``angles``.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    angle_convention : {"pa", "math"}
        How the source's angle sets its detector azimuth (:func:`_azimuth`).
    shape : tuple of int
        Shape of a frame, ``(ny, nx)``.
    sink : callable
        Function called as ``sink(i, crop)`` with what frame ``J[i]`` gets (:func:`stamp_crop`).
    """
    truenorth = _north_values(truenorth, angles)
    if p.base is not None:
        _place_shared(J, p, angles, truenorth, center, angle_convention, shape, sink)
    else:
        _place_each(J, p, model, angles, truenorth, center, angle_convention, shape, sink)


def prepare_sources(sources: Sequence[Source], model: InjectionModel, fallback: Optional[InjectionModel],
                    pxscale: float) -> list:
    """Work out what each source needs in every frame, for :func:`place_source`.

    Parameters
    ----------
    sources : sequence of Source
        The sources to inject.
    model : InjectionModel
        The injection model.
    fallback : InjectionModel or None
        Model that supplies the stamp where ``model`` has no template.
    pxscale : float
        Plate scale, in arcsec per pixel.

    Returns
    -------
    prepared : list of _Prepared
        One entry per source, in order.
    """
    return _prepare(sources, model, fallback, pxscale)


def _add_shared(out, J, p: _Prepared, angles, truenorth, center, angle_convention) -> None:
    """Add a source whose scaled stamp is the same in every frame to the frames ``J``.

    Parameters
    ----------
    out : ndarray, shape (n, ny, nx)
        The cube, changed in place.
    J : list of int
        Indices of the frames.
    p : _Prepared
        The source, with ``base`` set.
    angles : ndarray
        Parallactic angle of each frame, in degrees.
    truenorth : float or ndarray
        True-north offset, as :func:`_north_values` gives it.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    angle_convention : {"pa", "math"}
        How the source's angle sets its detector azimuth (:func:`_azimuth`).
    """
    _place_shared(J, p, angles, truenorth, center, angle_convention, out.shape[1:],
                  lambda i, c: add_crop(out[J[i]], c))


def _add_each(out, J, p: _Prepared, model, angles, truenorth, center, angle_convention) -> None:
    """Add a source whose stamp or throughput changes frame to frame to the frames ``J``.

    Parameters
    ----------
    out : ndarray, shape (n, ny, nx)
        The cube, changed in place.
    J : list of int
        Indices of the frames.
    p : _Prepared
        The source.
    model : InjectionModel
        The injection model.
    angles : ndarray
        Parallactic angle of each frame, in degrees.
    truenorth : float or ndarray
        True-north offset, as :func:`_north_values` gives it.
    center : tuple of 2 floats
        The star's ``(x, y)`` pixel.
    angle_convention : {"pa", "math"}
        How the source's angle sets its detector azimuth (:func:`_azimuth`).
    """
    _place_each(J, p, model, angles, truenorth, center, angle_convention, out.shape[1:],
                lambda i, c: add_crop(out[J[i]], c))


def injected_model(reducer, sources, shape) -> Optional[np.ndarray]:
    """Return the injected PSFs of some sources as they appear in the final north-up image.

    The model is built at the data's own roll angles, then derotated and combined as the science
    frames are.  Frames that share a roll give the same picture, so each distinct angle is rotated
    once and weighted by how many frames carry it.  A multi-roll data set therefore shows the
    superposition that is in the combined image.

    Parameters
    ----------
    reducer : object
        Reducer whose injection model, plate scale, true north and frame angles are used.  A
        partitioned reducer lends its first sub-reducer that has a model.
    sources : sequence
        The sources, each a :class:`~klip_tpe.metrics.Source` or a ``(rho, theta, contrast)``
        sequence.
    shape : tuple of int
        Shape of the final image, ``(ny, nx)``.

    Returns
    -------
    image : ndarray or None
        The model image, shape ``shape``, or None when there is no model, no source or anything
        fails.

    Notes
    -----
    It plays the role of ``disp_injmdl`` in the IDL predecessor.  It used to inject into one
    ``parang = 0`` frame and derotate by the true-north offset alone, which assumed a roll of 0.
    Once the template became spacecraft-fixed (``815fc55``) the model's lobes showed rotated away
    from the data by the roll itself, about 112 degrees on HIP 65426's rolls of 108.0 and 117.4.
    Until 2026-11-10 it was ``display.injected_model_image``, which now calls it.
    """
    try:
        from .klip import rotate_ccw
        red = reducer
        sub = red if getattr(red, "model", None) is not None else next(
            (r for r in getattr(red, "reducers", {}).values() if getattr(r, "model", None) is not None), None)
        if sub is None or not sources or shape is None or len(shape) != 2:
            return None
        src = [Source(*tuple(float(v) for v in (s.as_tuple() if hasattr(s, "as_tuple") else s)[:3])) for s in sources]
        tn = float(getattr(sub, "truenorth", 0.0))
        try:
            ang = np.asarray(sub.frame_angles(), float).ravel()
            ang = ang[np.isfinite(ang)]
        except Exception:
            ang = np.zeros(0)
        if ang.size == 0:
            ang = np.zeros(1)
        uang, cnt = np.unique(np.round(ang, 2), return_counts=True)
        blank = np.zeros((uang.size,) + tuple(int(v) for v in shape), np.float32)
        img = inject_sources(blank, uang, src, sub.model, sub.pxscale, truenorth=tn,
                             fallback=getattr(sub, "fallback", None),
                             angle_convention=getattr(sub, "angle_convention", "pa"))
        # cval=0 rather than derotate()'s NaN fill: this is a display model, and NaN
        # corners would drag the panel's robust stretch around.
        der = np.stack([rotate_ccw(np.asarray(img[i], float), float(uang[i]) + tn, cval=0.0)
                        for i in range(uang.size)])
        return np.average(der, axis=0, weights=cnt.astype(float))
    except Exception:
        return None
