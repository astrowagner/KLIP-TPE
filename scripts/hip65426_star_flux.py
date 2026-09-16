#!/usr/bin/env python
"""The two model quantities the HIP 65426 contrast axis needs, and nothing else.

The JWST data are in MJy/sr with a coronagraphy-specific flux calibration, so the star's
brightness can be *imported* rather than measured off the frames (it cannot be measured off
them -- the star is behind MASK335R).  Two numbers are needed:

  EE(r)   the fraction of an UNOCCULTED point source's flux that lands within the radius
          the injection model normalises in.  Unocculted **through the Lyot stop**: the
          pipeline's PHOTMJSR for NRC_CORON is derived from standards observed in this same
          coronagraphic mode, so it already carries the Lyot stop and the mask substrate,
          and the only thing left for the model to supply is (a) the PSF shape and (b) the
          occulter's spatial transmission T(rho), which offaxis_grid measures separately.
          Using an IMAGING PSF here would double-count the Lyot stop's effect on the shape.

  S       the star's flux density in the F444W bandpass [Jy], by synthetic photometry of a
          stellar model normalised to 2MASS Ks -- the same route Carter et al. (2023) took
          (PHOENIX, Teff = 8600 +/- 200 K), reduced here to a Planck function because Ks and
          F444W are both deep in the Rayleigh-Jeans tail of an A2V star and the difference
          is a percent-level colour term, which this script quantifies rather than assumes.

Then, for a model whose stamps carry unit flux inside ``ee_radius_px``,

    flux_unit = S / (1e6 * PIXAR_SR) * EE(ee_radius_px)        [MJy/sr, summed over pixels]

and ``throughput(rho)`` is the grid's own transmission.  Run
``scripts/check_hip65426_contrast.py`` afterwards: that is what decides whether this is right.

Needs STPSF and its data files (STPSF_PATH).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klip_tpe.instruments.generic import aperture_sum

FILTER = "F444W"
PUPIL = "MASKRND"
#: 2MASS: HIP 65426 is A2V with J = 6.826 and J-K = 0.055 (Carter et al. 2023, sect. 2)
KS_MAG, KS_ERR = 6.771, 0.021
KS_ZP_JY = 666.7          # 2MASS Ks isophotal zero-point flux density (Cohen et al. 2003)
KS_LAM_UM = 2.159         # 2MASS Ks isophotal wavelength
TEFF, TEFF_ERR = 8600.0, 200.0    # Carter et al. 2023


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------- encircled energy

def unocculted_psf(fov_arcsec=20.0, oversample=2, nlambda=3):
    """The coronagraphic mode's PSF with the occulter removed but the Lyot stop in."""
    import stpsf
    inst = stpsf.NIRCam()
    inst.filter = FILTER
    inst.pupil_mask = PUPIL
    inst.image_mask = None
    p = inst.calc_psf(fov_arcsec=fov_arcsec, oversample=oversample, nlambda=nlambda)
    img = np.asarray(p["DET_SAMP"].data, float)
    return img, float(inst.pixelscale)


def ee_curve(img, radii_px):
    ny, nx = img.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    tot = float(np.nansum(img))
    return np.array([aperture_sum(img, cx, cy, r) for r in radii_px]), tot


# --------------------------------------------------------------- synthetic photometry

def bandpass(path):
    """(wavelength_um, throughput) from an STPSF filter file."""
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[1].data
        names = [c.upper() for c in d.columns.names]
        w = np.asarray(d[d.columns.names[names.index("WAVELENGTH")]], float)
        t = np.asarray(d[d.columns.names[names.index("THROUGHPUT")]], float)
    unit = str(h[1].header.get("TUNIT1", "")).strip().lower() if True else ""
    if unit in ("angstrom", "angstroms", "a"):
        w = w / 1e4
    elif unit in ("m", "meter", "meters"):
        w = w * 1e6
    elif unit in ("nm", "nanometer", "nanometers"):
        w = w / 1e3
    elif w.max() > 1e4:                     # Angstrom without a usable unit keyword
        w = w / 1e4
    elif w.max() > 100:                     # nm
        w = w / 1e3
    elif w.max() < 1e-3:                    # m
        w = w * 1e6
    ok = np.isfinite(w) & np.isfinite(t) & (t > 0)
    return w[ok], t[ok]


def planck_fnu(lam_um, teff):
    """B_nu(T) in arbitrary units, per unit frequency."""
    lam = np.asarray(lam_um, float) * 1e-6
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    nu = c / lam
    x = h * nu / (k * teff)
    return nu ** 3 / np.expm1(x)


def photon_weighted(lam_um, fnu, band_lam, band_t):
    """Photon-weighted mean flux density over a bandpass: int f_nu T dnu/nu / int T dnu/nu."""
    t = np.interp(band_lam, lam_um, np.ones_like(lam_um)) * 0 + band_t   # keep shape
    f = np.interp(band_lam, lam_um, fnu)
    # dnu/nu = -dlam/lam, so integrate in ln(lambda)
    x = np.log(band_lam)
    return float(np.trapezoid(f * t, x) / np.trapezoid(t, x))


def ks_bandpass():
    """2MASS Ks relative spectral response, as a compact tabulation (Cohen et al. 2003)."""
    lam = np.array([1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200, 2.250, 2.300,
                    2.350, 2.400, 2.450, 2.500], float)
    rsr = np.array([0.000, 0.019, 0.408, 0.795, 0.928, 0.966, 0.985, 0.973, 0.940,
                    0.760, 0.135, 0.005, 0.000], float)
    ok = rsr > 0
    return lam[ok], rsr[ok]


def main():
    if not (os.environ.get("STPSF_PATH") or os.environ.get("WEBBPSF_PATH")):
        raise SystemExit("set STPSF_PATH to the stpsf data directory first")
    import stpsf

    log("1. encircled energy of the UNOCCULTED coronagraphic-mode PSF (Lyot stop in)")
    best = None
    for fov in (10.0, 20.0, 30.0):
        img, px = unocculted_psf(fov_arcsec=fov)
        tot = float(np.nansum(img))
        log(f"   fov {fov:4.1f}\"  {img.shape}  pxscale {px:.5f}\"  sum {tot:.6f} "
            f"(normalize='first', so this is the Lyot stop's diffractive throughput)")
        best = (img, px, tot)
    img, px, tot = best

    rad = np.array([2, 4, 8, 12, 16.5, 20, 30, 50, 80, 120, 160, 200, 240], float)
    ee, _ = ee_curve(img, rad)
    log(f"\n   {'r px':>7} {'r arcsec':>9} {'EE (/total in FOV)':>20}")
    for r, v in zip(rad, ee):
        if r <= min(img.shape) / 2:
            log(f"   {r:7.1f} {r * px:9.3f} {v / tot:20.5f}")
    log(f"\n   EE converges to 1 by construction inside the FOV; the number that matters is")
    log(f"   the fraction inside the injection model's normalisation radius.")

    for r_ap in (7.5, 16.5):
        log(f"   EE({r_ap:4.1f} px = {r_ap * px:.3f}\") = {aperture_sum(img, (img.shape[1]-1)/2, (img.shape[0]-1)/2, r_ap) / tot:.5f}")

    log("\n2. HIP 65426 in F444W, by synthetic photometry")
    fpath = os.path.join(stpsf.utils.get_stpsf_data_path(), "NIRCam", "filters",
                         f"{FILTER}_throughput.fits")
    bl, bt = bandpass(fpath)
    log(f"   {FILTER} bandpass {bl.min():.3f}-{bl.max():.3f} um, peak throughput {bt.max():.3f}")
    kl, kt = ks_bandpass()
    lam = np.linspace(1.5, 6.0, 4000)
    f_ks = KS_ZP_JY * 10 ** (-0.4 * KS_MAG)
    log(f"   2MASS Ks = {KS_MAG:.3f} +/- {KS_ERR:.3f}  ->  F_Ks = {f_ks:.4f} Jy")
    for teff in (TEFF - TEFF_ERR, TEFF, TEFF + TEFF_ERR):
        fnu = planck_fnu(lam, teff)
        c_ks = photon_weighted(lam, fnu, kl, kt)
        c_f4 = photon_weighted(lam, fnu, bl, bt)
        s = f_ks * c_f4 / c_ks
        log(f"   Teff {teff:6.0f} K:  F444W/Ks colour {c_f4 / c_ks:.5f}  ->  S = {s:.5f} Jy "
            f"({-2.5 * np.log10(c_f4 / c_ks):+.4f} mag)")
    fnu = planck_fnu(lam, TEFF)
    S = f_ks * photon_weighted(lam, fnu, bl, bt) / photon_weighted(lam, fnu, kl, kt)
    log(f"\n   S(F444W) = {S:.5f} Jy = {S * 1e3:.2f} mJy "
        f"(+/- {100 * 0.4 * np.log(10) * KS_ERR:.1f}% from Ks alone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
