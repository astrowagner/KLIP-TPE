"""Injection-position samplers, successors of the IDL predecessor's ``near2m_randpos`` and its variants.

Every evaluation, calibration trial and validation trial draws fresh positions, so the optimizer
cannot garden speckles at fixed sites.  The default strategy keeps the radii deterministic, a
centered ladder across the band so every evaluation of an annulus injects at the same separations,
and draws only the azimuth's anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source

__all__ = ["PositionSampler", "ReplaySampler"]


@dataclass
class PositionSampler:
    """Sampler of ``n`` sources in the radial band ``[r_lo, r_hi]``.

    This one object serves the search, the calibration, the validation, ``param_verify`` and the final
    products alike, so every stage injects with the same geometry.  The two-source rule (both at the
    annulus' area-weighted middle radius, addendum 2 section 2b) is applied by the caller collapsing the
    band to ``r_lo == r_hi``, where ``span == 0`` puts every source at that radius, the inner-working
    clamp included, as the IDL predecessor's ``near2m_randpos`` does.

    Attributes
    ----------
    fwhm_as : float
        FWHM, in arcsec, for the inner floor, the known sources' keep-outs and the mutual separation.
    strategy : {"spread", "fixed_pa"}, default "spread"
        ``"spread"`` for a deterministic radial ladder with a random anchor of evenly spaced azimuths
        (NEAR, LMIRCam, HPBoo), ``"fixed_pa"`` for every source at the band's middle radius at random
        PAs (the NIRCam variant).
    min_r_fwhm : float, default 1.5
        Inner floor of the band, in FWHM.
    excl_fwhm : float, default 1.5
        Radius of the known sources' keep-outs, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Real sources, as ``(rho, theta)``, no injection may land within ``excl_fwhm`` FWHM of.
    forbidden_pa : sequence of tuple of 2 floats, default ()
        Sectors of PA to avoid, as ``(center_deg, half_width_deg)``, such as the JWST N/S axis.
    min_mutual_fwhm : float, default 2.0
        Smallest distance between two sources of ``"fixed_pa"``, in FWHM.
    max_tries : int, default 30
        Number of draws tried before the last is kept, at least 400 with a mask.
    forbidden_mask : ndarray of bool or None, default None
        Pixels of the reduced frame no injection lands on or within ``mask_clear_fwhm`` FWHM of, True
        forbidden, the star at the center.  It is for resolved emission no wedge follows, such as a
        spiral or ring disk (:func:`~klip_tpe.metrics.disk_mask_from_image`), and goes with the same
        mask as the metric's ``pixel_mask`` (``default_config(disk_mask=...)`` sets both).
    mask_pxscale : float or None, default None
        Plate scale of the mask, in arcsec per pixel, which the mask needs.
    mask_clear_fwhm : float, default 0.0
        Clearance from the mask, in FWHM.
    mask_angle_convention : {"pa", "math"}, default "pa"
        The sources' angle convention, the metric's.
    sidelobes : Sidelobes or None, default None
        The noise rings' sidelobe rule (:class:`~klip_tpe.metrics.Sidelobes`), under which no source is
        drawn in another's ADI sidelobe arc, where its signal would carry the other's negative lobe.
        The :class:`~klip_tpe.runner.Runner` sets it from ``RunConfig.sidelobes``.
    _mask_dist : ndarray or None
        Cache of each pixel's distance from the mask, in pixels, made on the first check.
    """

    fwhm_as: float
    strategy: str = "spread"
    min_r_fwhm: float = 1.5
    excl_fwhm: float = 1.5
    known: Sequence[Tuple[float, float]] = ()
    forbidden_pa: Sequence[Tuple[float, float]] = ()
    min_mutual_fwhm: float = 2.0
    max_tries: int = 30
    forbidden_mask: Optional[np.ndarray] = None
    mask_pxscale: Optional[float] = None
    mask_clear_fwhm: float = 0.0
    mask_angle_convention: str = "pa"
    #: the noise rings' sidelobe rule (:class:`~klip_tpe.metrics.Sidelobes`): no source is drawn
    #: in another's ADI sidelobe arc, where its signal would carry the other's negative lobe.
    #: The :class:`~klip_tpe.runner.Runner` sets it from ``RunConfig.sidelobes``.
    sidelobes: Optional[object] = field(default=None, repr=False, compare=False)
    _mask_dist: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    def _xy(self, rho, theta):
        """Return positions as offsets from the star, PA East of North.

        Parameters
        ----------
        rho : float or array_like
            Separation of each source, in arcsec.
        theta : float or array_like
            Angle of each source, in degrees.

        Returns
        -------
        x : ndarray
            Offset along x, in arcsec.
        y : ndarray
            Offset along y, in arcsec.
        """
        phi = np.deg2rad(np.asarray(theta, float) + 90.0)
        return np.asarray(rho, float) * np.cos(phi), np.asarray(rho, float) * np.sin(phi)

    def _on_mask(self, rho, theta) -> bool:
        """Tell whether any source falls on ``forbidden_mask`` or within ``mask_clear_fwhm`` FWHM of it.

        Parameters
        ----------
        rho : float or array_like
            Separation of each source, in arcsec.
        theta : float or array_like
            Angle of each source, in degrees.

        Returns
        -------
        on_mask : bool
            True when one does, False without a mask.

        Raises
        ------
        ValueError
            If the mask comes without ``mask_pxscale``.
        """
        if self.forbidden_mask is None:
            return False
        if not self.mask_pxscale:
            raise ValueError("forbidden_mask needs mask_pxscale")
        from scipy import ndimage
        from .metrics import source_xy, star_center
        m = np.asarray(self.forbidden_mask, bool)
        if self._mask_dist is None or self._mask_dist.shape != m.shape:
            # distance (px) from every pixel to the nearest forbidden one, 0 on the mask
            self._mask_dist = ndimage.distance_transform_edt(~m) if m.any() else np.full(m.shape, np.inf)
        cx, cy = star_center(m.shape)
        xs, ys = source_xy(rho, theta, float(self.mask_pxscale), cx, cy, self.mask_angle_convention)
        clear = float(self.mask_clear_fwhm) * self.fwhm_as / float(self.mask_pxscale)
        for x, y in zip(xs, ys):
            ix, iy = int(round(float(x))), int(round(float(y)))
            if 0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and self._mask_dist[iy, ix] <= clear:
                return True
        return False

    def _bad(self, rho, theta) -> bool:
        """Tell whether a draw breaks a rule: a known source's keep-out, a forbidden sector, a sidelobe or the mask.

        Parameters
        ----------
        rho : float or array_like
            Separation of each source, in arcsec.
        theta : float or array_like
            Angle of each source, in degrees.

        Returns
        -------
        bad : bool
            True when the draw breaks one.
        """
        exr = self.excl_fwhm * self.fwhm_as
        x, y = self._xy(rho, theta)
        for kr, kt in self.known:
            kx, ky = self._xy(kr, kt)
            if np.any(np.hypot(x - kx, y - ky) < exr):
                return True
        for c, hw in self.forbidden_pa:
            d = np.abs((np.asarray(theta) - c + 180.0) % 360.0 - 180.0)
            if np.any(d < hw):
                return True
        if self._in_sidelobes(rho, theta):
            return True
        return self._on_mask(rho, theta)

    def _in_sidelobes(self, rho, theta) -> bool:
        """Tell whether any source sits in another's sidelobe arc, under ``sidelobes``.

        Parameters
        ----------
        rho : float or array_like
            Separation of each source, in arcsec.
        theta : float or array_like
            Angle of each source, in degrees.

        Returns
        -------
        inside : bool
            True when one does, False without a rule or with fewer than 2 sources.
        """
        rule = self.sidelobes
        rho, theta = np.atleast_1d(np.asarray(rho, float)), np.atleast_1d(np.asarray(theta, float))
        if rule is None or rho.size < 2:
            return False
        dr = float(rule.radial_fwhm) * self.fwhm_as
        for i in range(rho.size):
            for j in range(rho.size):
                if i == j or abs(rho[i] - rho[j]) > dr:
                    continue
                if abs((theta[i] - theta[j] + 180.0) % 360.0 - 180.0) <= rule.half_width(rho[j], self.fwhm_as):
                    return True
        return False

    def sample(self, n: int, r_lo: float, r_hi: float, rng: np.random.Generator,
               contrast: float = 0.0) -> List[Source]:
        """Draw ``n`` sources in a band, until a draw breaks no rule or the tries run out.

        Parameters
        ----------
        n : int
            Number of sources, at least 1.
        r_lo : float
            Inner radius of the band, in arcsec, raised to the inner floor.
        r_hi : float
            Outer radius of the band, in arcsec.
        rng : numpy.random.Generator
            The generator.
        contrast : float, default 0.0
            Contrast of the sources.

        Returns
        -------
        sources : list of Source
            The sources, the last draw's when none passed.
        """
        n = max(int(n), 1)
        rmin = self.min_r_fwhm * self.fwhm_as
        r_lo = max(float(r_lo), rmin)
        r_hi = max(float(r_hi), r_lo)
        span = r_hi - r_lo
        rho = np.zeros(n)
        theta = np.zeros(n)
        # a disk mask can forbid most of a ring (two opposite sources must both miss it), so it
        # gets more anchors to try; without one the draws are exactly what they were
        tries = self.max_tries if self.forbidden_mask is None else max(int(self.max_tries), 400)
        for _ in range(tries):
            if self.strategy == "fixed_pa":
                rho[:] = 0.5 * (r_lo + r_hi)
                ok = False
                for _t in range(200):
                    theta = rng.random(n) * 360.0
                    x, y = self._xy(rho, theta)
                    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
                    np.fill_diagonal(d, np.inf)
                    if d.min() >= self.min_mutual_fwhm * self.fwhm_as or n == 1:
                        ok = True
                        break
                if not ok:
                    continue
            else:
                th0 = rng.random() * 360.0
                r0 = r_lo + 0.5 * span / n
                for i in range(n):
                    theta[i] = (th0 + i * 360.0 / n) % 360.0
                    rho[i] = r_lo + (((r0 - r_lo) + i * span / n) % span) if span > 0 else r_lo
            if not self._bad(rho, theta):
                break
        return [Source(float(r), float(t), float(contrast)) for r, t in zip(rho, theta)]

    def describe(self) -> dict:
        """Return the sampler's settings, for a run's record.

        Returns
        -------
        info : dict
            ``strategy``, ``min_r_fwhm``, ``excl_fwhm``, ``known``, ``forbidden_pa`` and, with a mask,
            ``forbidden_mask_px`` and ``mask_clear_fwhm``, since positions drawn off a mask are not those
            drawn without.
        """
        d = {"strategy": self.strategy, "min_r_fwhm": self.min_r_fwhm, "excl_fwhm": self.excl_fwhm,
             "known": [list(k) for k in self.known], "forbidden_pa": [list(f) for f in self.forbidden_pa]}
        if self.forbidden_mask is not None:
            # as the metric's pixel_mask_px: positions drawn off a mask are not those drawn without
            d.update(forbidden_mask_px=int(np.count_nonzero(self.forbidden_mask)),
                     mask_clear_fwhm=float(self.mask_clear_fwhm))
        return d


class ReplaySampler:
    """Recorded injection draws handed back in order, instead of new ones.

    A sampler's draws depend on the frames: its forbidden sectors come from the rolls and the dead
    zones, so the same seed draws different positions on another processing of the same data, such as
    the archive calints and a reduction from the raw ramps.  To score a configuration on other frames
    at the same injections, the positions an earlier run recorded (``library_ablation.json``'s
    ``draws``, or an evaluation's ``meta['draw_sources']``) are replayed through this.  Running out of
    recorded draws is an error, not a quiet switch to fresh ones.

    Parameters
    ----------
    draws : sequence of sequence of tuple
        The draws, each a list of ``(rho, theta)`` or ``(rho, theta, contrast)``.  The contrast given to
        :meth:`sample` is the one injected.
    base : PositionSampler or None, default None
        The sampler stood in for, which gives the other attributes (``fwhm_as``, ``known``,
        ``forbidden_pa``, ``describe``).
    """

    def __init__(self, draws, base: Optional[PositionSampler] = None):
        self.draws = [[tuple(float(v) for v in s[:2]) for s in d] for d in draws]
        self.base = base
        self.i = 0

    def __getattr__(self, name):
        """Return an attribute of the base sampler.

        Parameters
        ----------
        name : str
            The attribute.

        Returns
        -------
        value : object
            The base's attribute.

        Raises
        ------
        AttributeError
            If there is no base.
        """
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def sample(self, n: int, r_lo: float, r_hi: float, rng=None, contrast: float = 0.0) -> List[Source]:
        """Return the next recorded draw at a contrast.

        Parameters
        ----------
        n : int
            Number of sources, which must be the draw's.
        r_lo : float
            Inner radius of the band, unused.
        r_hi : float
            Outer radius of the band, unused.
        rng : numpy.random.Generator or None, default None
            The generator, unused.
        contrast : float, default 0.0
            Contrast of the sources.

        Returns
        -------
        sources : list of Source
            The draw.

        Raises
        ------
        IndexError
            If every recorded draw has been replayed.
        ValueError
            If the draw holds another number of sources than ``n``.
        """
        if self.i >= len(self.draws):
            raise IndexError(f"all {len(self.draws)} recorded draws have been replayed")
        d = self.draws[self.i]
        if len(d) != max(int(n), 1):
            raise ValueError(f"recorded draw {self.i} holds {len(d)} sources, {n} asked for")
        self.i += 1
        return [Source(r, t, float(contrast)) for r, t in d]

    def describe(self) -> dict:
        """Return the base sampler's settings, with ``replay`` and ``n_recorded``.

        Returns
        -------
        info : dict
            The settings.
        """
        d = dict(self.base.describe()) if self.base is not None else {}
        d.update(replay=True, n_recorded=len(self.draws))
        return d


def max_sources_for_noise(inner_px: float, fwhm: float, *, excl_fwhm: float = 1.5,
                          min_ring: int = 6, n_known: int = 0,
                          blocked_fraction: float = 0.0) -> int:
    """Return how many co-radial sources a ring can hold and still leave ``min_ring`` clean noise apertures.

    The geometry is :func:`klip_tpe.metrics.mawet_peak_snr`'s: it lays ``nap = floor(2 pi r / fwhm)``
    apertures round the ring, walks ``q = 1 .. nap-1`` (the source's own aperture is never a noise
    sample), and drops any aperture whose center falls within ``excl_fwhm * fwhm`` of any injected
    source, not only the one measured.  So every extra source costs the ring about ``2 excl_fwhm + 1``
    apertures, and below ``min_ring`` survivors the estimator stops using the ring and falls back to the
    radial band.

    Parameters
    ----------
    inner_px : float
        The radius the count has to hold at, in pixels, the annulus's inner edge, where the ring starves
        first.  The middle radius would over-fill the inner half.
    fwhm : float
        FWHM of the PSF, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    min_ring : int, default 6
        Fewest clean apertures the ring must keep.
    n_known : int, default 0
        Number of real companions on the ring.
    blocked_fraction : float, default 0.0
        Share of the ring lost to dead zones, forbidden sectors or the frame's edge, such as the about
        12 % of MIRI's 4QPM quadrant boundaries.

    Returns
    -------
    n : int
        The count, at least 1, since a separation too tight to hold one source with a clean ring is a
        real case for the caller to decide on.

    Notes
    -----
    It assumes every source sits on the same ring, which is how :meth:`PositionSampler.sample` places
    them only for ``fixed_pa``, a collapsed band (``opt_width``) and the ``pair_area_midpoint`` pair.
    The default strategy steps ``rho`` across the band, so sources mostly stay out of each other's
    keep-outs, and there this bound is far too pessimistic: on MIRI's 6.7-36 px annulus it returns 2
    where the measured answer is past 14.  ``Runner._ring_survives`` therefore samples real positions
    and scores them with :func:`~klip_tpe.metrics.mawet_peak_snr`, and this serves only as a cheap bound
    where there is no sampler to ask.
    """
    r = float(inner_px)
    f = float(fwhm)
    if not np.isfinite(r) or not np.isfinite(f) or r <= 0 or f <= 0:
        return 1
    nap = int(np.floor(2.0 * np.pi * r / f))
    if nap < 5:                                  # mawet_peak_snr abandons the ring below this
        return 1
    # Apertures one source removes.  The estimator's exclusion test is a CHORD distance
    # between pixel centers, so count the q that actually fail it rather than converting
    # excl_fwhm to a number of apertures -- at small nap the chord is meaningfully shorter
    # than the arc.  Worst case is sources far enough apart that their zones do not
    # overlap; clustered sources block fewer in total.
    exr = float(excl_fwhm) * f
    q = np.arange(1, nap // 2 + 1)
    per = 1 + 2 * int(np.count_nonzero(2.0 * r * np.sin(q * np.pi / nap) < exr))
    nap = int(np.floor(nap * max(0.0, 1.0 - float(blocked_fraction))))
    # nclean = nap - per*(nsrc + n_known): the measured source's own aperture is never a
    # noise sample, which is exactly the one aperture that makes this come out even.
    return max(1, int(np.floor((nap - int(min_ring)) / per)) - max(0, int(n_known)))


def n_sources_rule(annulus_index: int, outer_as: float, override: Optional[int] = None, *,
                   inner_px: Optional[float] = None, fwhm: Optional[float] = None,
                   excl_fwhm: float = 1.5, min_ring: int = 6, n_known: int = 0,
                   blocked_fraction: float = 0.0) -> int:
    """Return the default number of sources of an annulus, the IDL predecessor's ``nsrc_a``.

    It is 4 in the innermost annulus and 6 outside, 3 where the outer edge is under 2", and at most 2
    under 1".  With ``inner_px`` and ``fwhm`` the count, an explicit one included, is also capped by
    :func:`max_sources_for_noise`, so a small field of view cannot be packed so full that the noise ring
    starves.  That cap binds on NIRCam, whose coronagraphic annuli are only a few FWHM around, where the
    predecessor's angular thresholds, written for NEAR's larger field, do not notice.

    Parameters
    ----------
    annulus_index : int
        The annulus, counted from 0.
    outer_as : float
        Outer edge of the annulus, in arcsec.
    override : int or None, default None
        An explicit count, at least 1, which the cap still binds.
    inner_px : float or None, default None
        Inner edge of the annulus, in pixels, for the cap.
    fwhm : float or None, default None
        FWHM of the PSF, in pixels, for the cap.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    min_ring : int, default 6
        Fewest clean apertures the ring must keep.
    n_known : int, default 0
        Number of real companions on the ring.
    blocked_fraction : float, default 0.0
        Share of the ring lost.

    Returns
    -------
    n : int
        The count, at least 1.
    """
    if override is not None and override >= 1:
        n = int(override)
    else:
        n = 4 if annulus_index == 0 else 6
        if outer_as < 2.0:
            n = 3
        if outer_as < 1.0:
            n = min(n, 2)
    # The packing cap applies to an explicit count too.  It is not a preference being
    # overruled: past it the ring holds fewer than min_ring clean apertures and
    # mawet_peak_snr stops using the ring at all, so the extra sources buy nothing and
    # cost every separation its noise estimate.  Callers that want to know it bound can
    # compare against what they asked for -- Runner._nsrc logs it.
    if inner_px is not None and fwhm:
        n = min(n, max_sources_for_noise(inner_px, fwhm, excl_fwhm=excl_fwhm,
                                         min_ring=min_ring, n_known=n_known,
                                         blocked_fraction=blocked_fraction))
    return max(1, int(n))
