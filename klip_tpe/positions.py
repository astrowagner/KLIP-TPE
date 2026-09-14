"""Injection-position samplers (``near2m_randpos`` and the variants' cousins).

Every evaluation, calibration trial and validation trial draws *fresh* positions so
the optimizer cannot garden speckles at fixed sites.  The default strategy keeps
the radii deterministic (a centred ladder across the band, so every evaluation of
an annulus injects at the same separations) and randomises only the azimuth
anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source

__all__ = ["PositionSampler"]


@dataclass
class PositionSampler:
    """Draw ``n`` sources in the radial band ``[r_lo, r_hi]`` (arcsec).

    strategy
        ``"spread"``   : deterministic radial ladder + random azimuth anchor with
                         even 360/n spacing (NEAR / LMIRCam / HPBoo);
        ``"fixed_pa"`` : all sources at the band centre radius, fully random PAs
                         subject to the exclusions (NIRCam variant).
    fwhm_as
        FWHM in arcsec; used for the inner floor (``min_r_fwhm``), the known-source
        exclusion radius (``excl_fwhm``) and the mutual-separation warning.
    known
        real sources ``[(rho, theta), ...]`` no injection may land within
        ``excl_fwhm`` FWHM of.
    forbidden_pa
        list of ``(centre_deg, half_width_deg)`` PA sectors to avoid (e.g. the
        JWST N/S axis).

    This one object is used by search, calibration, validation, param_verify and the
    final products alike, so every stage injects with identical geometry.  The
    two-source rule (both at the annulus' area-weighted mid radius, addendum 2 §2b) is
    applied by the *caller* collapsing the band to ``r_lo == r_hi`` -- the routine itself
    is unchanged (``span == 0`` puts every source at that radius, IWA clamp included),
    exactly as IDL's ``near2m_randpos``.
    """

    fwhm_as: float
    strategy: str = "spread"
    min_r_fwhm: float = 1.5
    excl_fwhm: float = 1.5
    known: Sequence[Tuple[float, float]] = ()
    forbidden_pa: Sequence[Tuple[float, float]] = ()
    min_mutual_fwhm: float = 2.0
    max_tries: int = 30

    def _xy(self, rho, theta):
        phi = np.deg2rad(np.asarray(theta, float) + 90.0)
        return np.asarray(rho, float) * np.cos(phi), np.asarray(rho, float) * np.sin(phi)

    def _bad(self, rho, theta) -> bool:
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
        return False

    def sample(self, n: int, r_lo: float, r_hi: float, rng: np.random.Generator,
               contrast: float = 0.0) -> List[Source]:
        n = max(int(n), 1)
        rmin = self.min_r_fwhm * self.fwhm_as
        r_lo = max(float(r_lo), rmin)
        r_hi = max(float(r_hi), r_lo)
        span = r_hi - r_lo
        rho = np.zeros(n)
        theta = np.zeros(n)
        for _ in range(self.max_tries):
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
        return {"strategy": self.strategy, "min_r_fwhm": self.min_r_fwhm, "excl_fwhm": self.excl_fwhm,
                "known": [list(k) for k in self.known], "forbidden_pa": [list(f) for f in self.forbidden_pa]}


def n_sources_rule(annulus_index: int, outer_as: float, override: Optional[int] = None) -> int:
    """Default per-annulus source count (``nsrc_a`` in IDL): 4 in the innermost
    annulus, 6 outside, reduced to 3 when the outer edge is < 2", to <= 2 when < 1"."""
    if override is not None and override >= 1:
        return int(override)
    n = 4 if annulus_index == 0 else 6
    if outer_as < 2.0:
        n = 3
    if outer_as < 1.0:
        n = min(n, 2)
    return n
