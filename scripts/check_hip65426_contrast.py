#!/usr/bin/env python
"""Verify the HIP 65426 contrast axis against HIP 65426 b's published brightness.

    STPSF_PATH=... python scripts/check_hip65426_contrast.py

The JWST set is the one case where the star's brightness cannot be measured off the frames
at all -- HIP 65426 sits behind MASK335R in every exposure and the reference star (phi Cen)
is behind it too, so there is no off-axis stellar image anywhere in the programme.  It has
to be imported, and the import has three parts, each of which can be wrong on its own:

  S        the star's flux density in F444W [Jy], by synthetic photometry of a stellar model
           normalised to 2MASS Ks (scripts/hip65426_star_flux.py);
  units    the frames are MJy/sr with the NRC_CORON flux calibration, so a point source of
           flux density S deposits S / (1e6 * PIXAR_SR) summed over its pixels;
  EE       only the fraction inside the injection model's normalisation radius is the
           `flux_unit` -- from the UNOCCULTED-through-the-Lyot-stop PSF, because that is the
           configuration the pipeline's PHOTMJSR refers to.

and the occulter's own transmission T(rho) is NOT in any of them -- it is the model's
`throughput(rho)`, measured by stpsf_psf.offaxis_grid against that same unocculted
reference.  Applying it twice, or folding it into flux_unit, is the classic way to get a
coronagraphic contrast axis wrong by 1/T.

The measurement is the one used for beta Pic: reduce clean, inject one fake at the
companion's separation and measure it in `inj - clean`, and solve for the contrast as a
fixed point so the fake and the companion are equally bright and their KLIP throughputs are
therefore identical.

Expected: dF444W ~ 8.7 against Carter et al. (2023)'s 8.703 +/- 0.055 (Table 3) for the same
data, with NOTHING anchored on the companion: optics_transmission is 1.0 because PHOTMJSR
for PUPIL=MASKRND already carries the coronagraphic optics.  Until 2026-09-16 the loader's
sigma-clip repair median-filtered the companion (and not the fakes, which are injected after
it), and a 0.561 'optics transmission' hid the 1.9x that cost; see docs/FLUX_CALIBRATION.md.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klip_tpe import datasets, stpsf_psf
from klip_tpe.backends import spaceklip as sk
from klip_tpe.instruments import generic
from klip_tpe.metrics import Source, mawet_peak_snr, radprof
from klip_tpe.reducer import ReductionRequest

PUB_DMAG = 8.703                     # Carter et al. 2023, Table 3, F444W (+/- 0.015 stat, 0.055 total)
PUB_REF = "Carter et al. 2023, ApJL 951, L20 -- the same programme (ERS 1386)"
PUB = 10.0 ** (-0.4 * PUB_DMAG)

RHO, PA = 0.820, 149.9               # Carter et al. 2023 Table 3, F444W (820 +/- 6 mas, 149.9 +/- 0.4 deg)
INRAD, OUTRAD = 6, 30
TEST_PA = [30.0, 90.0, 240.0, 300.0]      # clear of the companion
TOL_MAG = 0.45


def measure(img, red, rho, theta, *, known=(), model=None):
    ker = red.matched_filter_kernel(rho)
    snr, det = mawet_peak_snr(radprof(img), [rho], [theta], red.pxscale, red.fwhm,
                              kernel=ker, known=known, return_details=True)
    return float(det[0]["peak"]), float(snr[0]), det[0]


def _peak_near(img, xy, box=6):
    """Sub-pixel peak of a smoothed, radial-profile-subtracted image near ``xy``."""
    from scipy import ndimage
    a = ndimage.gaussian_filter(np.nan_to_num(radprof(img)), 1.2)
    x0, y0 = int(round(xy[0])), int(round(xy[1]))
    sub = a[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1]
    k = np.unravel_index(int(np.argmax(sub)), sub.shape)

    def par(f, i):
        if 0 < i < f.size - 1 and (f[i - 1] - 2 * f[i] + f[i + 1]) != 0:
            return 0.5 * (f[i - 1] - f[i + 1]) / (f[i - 1] - 2 * f[i] + f[i + 1])
        return 0.0
    return (x0 - box + k[1] + par(sub[k[0], :], k[1]),
            y0 - box + k[0] + par(sub[:, k[1]], k[0])), float(sub[k])


def solve_star_center(dsets, info, log):
    """The star's detector position, from where the companion lands in each roll.

    CRPIX is the aperture reference point and misses the star; there is no off-axis stellar
    image in the programme to centroid; and the coronagraphic residual is too speckle-
    dominated for a 180-degree symmetry fit (it moved the centre by 2 px between the two
    rolls of these very data).  What IS well determined is the companion: derotation about a
    centre that is wrong by ``delta`` puts it at ``u + R(PA_k) . delta`` in roll ``k``, so
    each roll gives ``delta = R(-PA_k) . (measured - expected)`` independently, and the two
    have to agree.

    This uses the companion's PUBLISHED astrometry to fix the centre, so it is not an
    astrometric measurement -- it is the geometry the photometry needs, and the roll-to-roll
    agreement is what says whether it is trustworthy.
    """
    from klip_tpe.metrics import source_xy, star_center
    ds, ws = [], []
    for name, dset in dsets.items():
        red = sk.make_reducer({name: dset}, mode="RDI", max_workers=1, log=lambda s: None)
        space = generic.make_space(red, k_klip_max=6, search_angles=False)
        p0 = dict(space.decode(space.default_vector()).params, inrad=INRAD, outrad=OUTRAD, k_klip=4)
        img = red.reduce(ReductionRequest(params=p0)).image
        cx, cy = star_center(img.shape)
        ux, uy = source_xy(np.array([RHO]), np.array([PA]), red.pxscale, cx, cy, "pa")
        m, v = _peak_near(img, (float(ux[0]), float(uy[0])), box=6)
        pa = float(np.median(dset.angles))
        a = np.deg2rad(-pa)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        delta = R @ np.array([m[0] - float(ux[0]), m[1] - float(uy[0])])
        ds.append(delta)
        ws.append(max(v, 0.0))
        log(f"   {name}: PA {pa:7.3f}  companion expected ({float(ux[0]):.2f}, {float(uy[0]):.2f}) "
            f"found ({m[0]:.2f}, {m[1]:.2f}) peak {v:.4g}  ->  delta ({delta[0]:+.2f}, {delta[1]:+.2f})")
    ds, ws = np.array(ds), np.array(ws)
    if ws.sum() <= 0:
        ws = np.ones(len(ws))
    delta = (ds * ws[:, None]).sum(0) / ws.sum()
    spread = float(np.hypot(*(ds.max(0) - ds.min(0)))) if len(ds) > 1 else float("nan")
    log(f"   weighted delta = ({delta[0]:+.2f}, {delta[1]:+.2f}) px, |{np.hypot(*delta):.2f}|; "
        f"roll-to-roll spread {spread:.2f} px")
    c0 = info["crpix"]
    return (c0[0] + float(delta[0]), c0[1] + float(delta[1])), delta, spread


def main(verbose=True):
    def log(m):
        if verbose:
            print(m, flush=True)

    if not (os.environ.get("STPSF_PATH") or os.environ.get("WEBBPSF_PATH")):
        raise SystemExit("set STPSF_PATH to the stpsf data directory first")

    d = os.path.join(datasets.data_dir(), "jwst_hip65426")
    files = sorted(glob.glob(os.path.join(d, "jw*calints.fits")))
    if not files:
        raise SystemExit(f"no calints in {d} -- run tutorials/fetch_jwst_hip65426.py")
    dsets, info = sk.load_calints(files, science_target="HIP65426", log=log)
    px, pixar_sr = info["pxscale"], info["pixar_sr"]
    log(f"  {info['filter']}, {info['bunit']}, PIXAR_SR {pixar_sr:.6e}, "
        f"{px * 1e3:.2f} mas/px")

    log("\nwhere is the star?  (CRPIX is the aperture reference, not the star)")
    center, delta, spread = solve_star_center(dsets, info, log)
    stored = datasets.PHOTOMETRY["hip65426_f444w"].get("star_center")
    if stored is not None:
        d = np.hypot(center[0] - stored[0], center[1] - stored[1])
        log(f"   datasets.PHOTOMETRY has ({stored[0]:.2f}, {stored[1]:.2f}); this solve is "
            f"{d:.2f} px away -- using the stored value so the runs are reproducible"
            + ("" if d < 0.3 else "  <-- MORE THAN 0.3 px: re-solve and update PHOTOMETRY"))
        center = tuple(stored)
    dsets, info = sk.load_calints(files, science_target="HIP65426", star_center=center,
                                  log=lambda s: None)
    log(f"   centred on ({center[0]:.2f}, {center[1]:.2f})")

    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    grid = stpsf_psf.offaxis_grid("NIRCam", info["filter"], image_mask="MASK335R",
                                  seps_as=np.arange(0.2, 3.01, 0.2), stamp_px=41, nlambda=3,
                                  log=log)
    sf = stpsf_psf.star_flux_from_flux_density(
        grid, phot["flux_density_jy"], pixar_sr,
        optics_transmission=phot.get("optics_transmission", 1.0), log=log)
    if "anchored" in str(phot.get("optics_transmission_source", "")):
        log("  NOTE: optics_transmission is anchored on this very companion, so the number "
            "below is a CALIBRATION, not an independent check.  What it still tests is "
            "everything else -- the units, the EE, T(rho), the centring and the recovery.")
    model = stpsf_psf.library(grid, star_flux=sf)
    log(f"  occulter transmission at {RHO}\": {model.throughput(RHO):.4f}   "
        f"(NOT in flux_unit -- it multiplies it)")

    red = sk.make_reducer(dsets, injection_model=model, mode="RDI", max_workers="auto",
                          log=lambda s: None)
    space = generic.make_space(red, k_klip_max=18, search_angles=False)
    space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
    p0 = dict(space.decode(space.default_vector()).params, inrad=INRAD, outrad=OUTRAD, k_klip=10)

    clean = red.reduce(ReductionRequest(params=p0)).image
    pk_p, snr_p, det_p = measure(clean, red, RHO, PA)
    log(f"\nHIP 65426 b: matched-filter peak {pk_p:.5g}, S/N {snr_p:.1f} "
        f"(at x={det_p['x']}, y={det_p['y']}, r={det_p['r']:.2f} px; expected "
        f"{RHO / px:.2f} px)")

    def throughput(c):
        t = []
        for th in TEST_PA:
            img = red.reduce(ReductionRequest(params=p0,
                                              injections=[Source(RHO, th, c)])).image
            pk_i, _, _ = measure(img - clean, red, RHO, th)
            t.append(pk_i / c)
        t = np.array(t, float)
        return float(np.median(t)), t

    log(f"\nsolving for the contrast at which the fake matches the companion:")
    log(f"{'iter':>5} {'c_inj':>12} {'peak_inj/c':>12} {'-> contrast':>13} {'dF444W':>9}")
    c, per_az = PUB, None
    for it in range(6):
        tp, per_az = throughput(c)
        got = pk_p / tp
        log(f"{it:5d} {c:12.4e} {tp:12.5g} {got:13.4e} {-2.5 * np.log10(got):9.3f}")
        if abs(np.log(got / c)) < 0.02:
            break
        c = got

    az = pk_p / per_az
    err = float(np.std(az) / np.median(az) / np.sqrt(len(az)))
    err = float(np.hypot(err, 1.0 / snr_p)) * 2.5 / np.log(10)
    dmag = -2.5 * np.log10(got)
    log(f"\nHIP 65426 b, fixed point over {len(az)} azimuths "
        f"({', '.join(f'{v:.3e}' for v in np.sort(az))}):")
    log(f"   contrast {got:.4e}  ->  dF444W = {dmag:.3f} +/- {err:.3f}")
    log(f"   published {PUB:.4e}  ->  dF444W = {PUB_DMAG}   ({PUB_REF})")
    dd = dmag - PUB_DMAG
    log(f"   difference {dd:+.3f} mag (ratio {got / PUB:.3f})")
    log(f"\n   the flux scale rests on: S = {phot['flux_density_jy']:.4f} Jy "
        f"(+/- {100 * phot.get('flux_density_err_frac', 0.03):.0f}%), "
        f"PIXAR_SR from the header, and the model EE -- see docs/FLUX_CALIBRATION.md")
    ok = abs(dd) <= TOL_MAG
    log(f"\n{'PASS' if ok else 'FAIL'}: the contrast axis is "
        f"{'on HIP 65426 b' if ok else f'off by more than {TOL_MAG} mag'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
