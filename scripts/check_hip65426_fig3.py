#!/usr/bin/env python
"""Reproduce Carter et al. (2023) Fig. 3 (F444W row) from the MAST calints, and check the
companion's flux against their Table 3 by injection.

    TQDM_DISABLE=1 STPSF_PATH=... python scripts/check_hip65426_fig3.py [--out DIR] [--sigma]

Same data, same recipe: the frames are cleaned by filling DQ pixels from their neighbours
(spaceKLIP's method) and registered to the median science frame; pyKLIP runs with ONE
annulus and ONE subsection over the whole image in ADI (2 modes), RDI (18) and ADI+RDI (20)
-- "the maximum number of KLIP PCA modes", as in their figure; the panels are shown on their
stretch (-1..5 MJy/sr, N up E left).  What should come out: a three-bar "hamburger" core with
six faint lobes around it and a ring of negative lobes -- "expected features that are related
to the Lyot stop design, and ... not indicative of discrete astrophysical sources".

Then the STPSF off-axis PSF is injected at Carter's published F444W flux density (127 uJy,
Table 3) at five other position angles and recovered through the same reductions.  The fakes
carry the same KLIP throughput and aperture fraction as the companion, so
sum(companion)/mean(sum(fakes)) is F_measured / F_Carter directly, with no flux-unit
bookkeeping in between.  Expected ~1.0-1.1.

``--sigma`` reruns everything with the pre-2026-09-16 sigma-clip repair, to see what it did:
one smeared blob at a third of the peak, different in the two rolls.

Writes ``hip65426_fig3.png`` (the four panels, one row per repair mode requested),
``hip65426_fig3_stamps.png`` (21 px stamps of the companion, per mode and per roll) and
``hip65426_fig3.txt`` (the numbers).  Needs the calints (tutorials/fetch_jwst_hip65426.py)
and STPSF with its data files.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klip_tpe import datasets, stpsf_psf
from klip_tpe.backends import spaceklip as sk
from klip_tpe.klip import derotate

# Carter et al. 2023, ApJL 951, L20, Table 3, F444W
CARTER = dict(rho=0.820, pa=149.9, dmag=8.703, dmag_err=0.055, flux_wm2um=1.97e-17, lam_um=4.397)
FAKE_DPA = (60.0, 120.0, 180.0, 240.0, 300.0)
H = 55                                   # 111 px crop, as load_calints


def carter_flux_mjy():
    """F_lambda [W m^-2 um^-1] -> F_nu [MJy] at the mean wavelength."""
    lam = CARTER["lam_um"] * 1e-6
    fnu_si = CARTER["flux_wm2um"] * 1e6 * lam ** 2 / 2.998e8          # W m^-2 Hz^-1
    return fnu_si / 1e-26 / 1e6


def run_pyklip(Sx, Rx, pas, mode, numbasis):
    import pyklip.parallelized as par
    from pyklip.instruments.Instrument import GenericData
    from pyklip.rdi import PSFLibrary
    n, ny, nx = Sx.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    centers = np.tile([cx, cy], (n, 1)).astype(float)
    kw = {}
    if "RDI" in mode:
        sci_names = np.array([f"sci{i}" for i in range(n)])
        lib_imgs = np.concatenate([Rx, Sx], axis=0)
        names = np.concatenate([np.array([f"ref{i}" for i in range(Rx.shape[0])]), sci_names])
        gdata = GenericData(Sx, centers, parangs=np.asarray(pas, float), filenames=sci_names)
        lib = PSFLibrary(lib_imgs, (cx, cy), names, compute_correlation=True)
        lib.prepare_library(gdata)
        kw = dict(psf_library=lib.master_library, psf_library_corr=lib.correlation, psf_library_good=lib.isgoodpsf)
    out = par.klip_parallelized(Sx, centers, np.asarray(pas, float), np.ones(n), np.arange(n), 1.0, OWA=H,
                                mode=mode, annuli=1, subsections=1, movement=1.0,
                                numbasis=np.asarray(numbasis), aligned_center=[cx, cy], numthreads=1,
                                minrot=0, maxrot=360, verbose=False, **kw)
    return np.asarray(out[0], np.float32)                          # (b, n, ny, nx): aligned, not derotated


def sky_xy(rho, pa, px):
    r = rho / px
    return H - r * np.sin(np.deg2rad(pa)), H + r * np.cos(np.deg2rad(pa))


def det_xy(rho, pa_sky, pa_frame, px):
    """Where a source at (rho, PA) sits in an un-derotated frame that derotate() will turn
    CCW by pa_frame.  Returns x, y and the CCW angle of the outward radial direction."""
    r = rho / px
    phi = 90.0 + pa_sky - pa_frame
    return H + r * np.cos(np.deg2rad(phi)), H + r * np.sin(np.deg2rad(phi)), phi


def place(template, xy, phi_ccw_deg, shape):
    """Template (source at centre, star along -x) rotated so its +x points along phi, at xy."""
    from scipy import ndimage
    t = ndimage.rotate(template, -phi_ccw_deg, reshape=False, order=3)   # scipy's + is clockwise
    t[t < 0] = 0.0
    n = t.shape[0]; c = (n - 1) // 2
    x, y = xy
    ix, iy = int(np.floor(x)), int(np.floor(y))
    ts = ndimage.shift(t, (y - iy, x - ix), order=3)
    out = np.zeros(shape, float)
    y0, x0 = iy - c, ix - c
    ys, xs = slice(max(y0, 0), min(y0 + n, shape[0])), slice(max(x0, 0), min(x0 + n, shape[1]))
    out[ys, xs] = ts[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=".", help="directory for the png/txt outputs")
    ap.add_argument("--sigma", action="store_true", help="also run the pre-fix sigma-clip repair, as a second row")
    ap.add_argument("--data", default=os.path.join(datasets.data_dir(), "jwst_hip65426"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(a.data, "jw*calints.fits")))
    if len(files) < 3:
        raise SystemExit(f"no calints in {a.data} -- run tutorials/fetch_jwst_hip65426.py")
    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    modes = ["dq"] + (["sigma"] if a.sigma else [])
    results = {}
    for rmode in modes:
        dsets, info = sk.load_calints(files, science_target="HIP65426", star_center=phot["star_center"],
                                      repair=rmode, keep_frames=True, log=log)
        px, pixar_sr = info["pxscale"], info["pixar_sr"]
        Sx, Rx = info["crop_sci"], info["crop_ref"]
        pas = np.array([f["pa"] for f in info["frames"] if f["role"] == "SCI"])
        res = {"unsub": np.median(Sx[pas == pas[0]], axis=0)}
        for m, nb in (("ADI", [2]), ("RDI", [18]), ("ADI+RDI", [20])):
            sub = run_pyklip(Sx, Rx, pas, m, nb)
            dr = derotate(sub[-1], pas, 0.0)
            res[m] = np.nanmean(dr, axis=0)
            res[m + "_roll1"] = np.nanmean(dr[pas == pas[0]], axis=0)
            res[m + "_roll2"] = np.nanmean(dr[pas != pas[0]], axis=0)
        results[rmode] = (res, Sx, Rx, pas, px, pixar_sr)

    # ---- Fig. 3 -------------------------------------------------------------------------
    xb, yb = sky_xy(CARTER["rho"], CARTER["pa"], results["dq"][4])
    fig, axes = plt.subplots(len(modes), 4, figsize=(15, 3.9 * len(modes)), squeeze=False)
    for r, rmode in enumerate(modes):
        res = results[rmode][0]
        for c, (t, im) in enumerate([("Unsub (roll 1 median, detector frame)", res["unsub"]), ("ADI, 2 modes", res["ADI"]),
                                     ("RDI, 18 modes", res["RDI"]), ("ADI+RDI, 20 modes", res["ADI+RDI"])]):
            ax = axes[r, c]
            ax.imshow(im, origin="lower", cmap="viridis", vmin=-1, vmax=5, interpolation="nearest")
            ax.plot(H, H, "w*", ms=9); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(("" if len(modes) == 1 else f"repair={rmode}: ") + t, fontsize=10)
            if c:
                ax.add_patch(plt.Circle((xb, yb), 6, ec="w", fc="none", lw=0.6, ls=":"))
    fig.suptitle("HIP 65426 F444W from the MAST calints, in the layout of Carter et al. (2023) Fig. 3 "
                 "(-1..5 MJy/sr; subtracted panels N up, E left)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hip65426_fig3.png"), dpi=130)

    # ---- stamps -------------------------------------------------------------------------
    fig, axes = plt.subplots(len(modes), 5, figsize=(15, 3.3 * len(modes)), squeeze=False)
    x0, y0 = int(round(xb)) - 10, int(round(yb)) - 10
    for r, rmode in enumerate(modes):
        res = results[rmode][0]
        for c, k in enumerate(("RDI", "ADI+RDI", "ADI", "RDI_roll1", "RDI_roll2")):
            st = res[k][y0:y0 + 21, x0:x0 + 21]
            ax = axes[r, c]
            ax.imshow(st, origin="lower", cmap="inferno", vmin=-1, vmax=max(np.nanmax(st), 1.0), interpolation="nearest")
            ax.plot(xb - x0, yb - y0, "c+", ms=12); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"repair={rmode}: {k.replace('_', ', ')}\npeak {np.nanmax(st):.1f} MJy/sr", fontsize=9)
    fig.suptitle("HIP 65426 b, 21 px stamps (N up, E left); + = Carter et al. Table 3 position", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hip65426_fig3_stamps.png"), dpi=130)

    # ---- photometry by injection at Carter's flux ---------------------------------------
    res, Sx, Rx, pas, px, pixar_sr = results["dq"]
    F_b = carter_flux_mjy()
    log(f"\nCarter et al. Table 3: F444W flux of b = {F_b*1e12:.1f} uJy -> {F_b/pixar_sr:.0f} MJy/sr summed if unocculted; "
        f"dF444W = {CARTER['dmag']} +/- {CARTER['dmag_err']}")
    g = stpsf_psf.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R", seps_as=[0.78, CARTER["rho"], 0.86],
                               stamp_px=41, oversample=2, nlambda=5, detector_position=(150, 173), log=log)
    sl = g["slices"][1]
    import stpsf
    nc = stpsf.NIRCam(); nc.filter = "F444W"; nc.pupil_mask = "MASKRND"; nc.image_mask = None
    nc.detector = "NRCA5"; nc.detector_position = (150, 173)
    L = float(np.asarray(nc.calc_psf(fov_pixels=121, oversample=2, nlambda=5)["DET_SAMP"].data).sum())
    amp = F_b / pixar_sr / L                     # the slice is normalised at the entrance pupil; L is the Lyot stop's share
    log(f"STPSF: Lyot-stop throughput L = {L:.3f}, occulter transmission at {CARTER['rho']}\" = {sl.sum()/L:.3f}; "
        f"injected peak {amp*sl.max():.1f} MJy/sr")
    inj = Sx.astype(float).copy()
    for i, pa_f in enumerate(pas):
        for d in FAKE_DPA:
            x, y, phi = det_xy(CARTER["rho"], CARTER["pa"] + d, pa_f, px)
            inj[i] += amp * place(sl, (x, y), phi, inj[i].shape)
    inj = inj.astype(np.float32)
    yy, xx = np.mgrid[0:2 * H + 1, 0:2 * H + 1]
    ap_sum = lambda im, x, y, r: float(np.nansum(im[np.hypot(xx - x, yy - y) <= r]))
    ratios = {}
    for m, nb in (("RDI", [18]), ("ADI+RDI", [20])):
        d = res[m]
        df = np.nanmean(derotate(run_pyklip(inj, Rx, pas, m, nb)[-1], pas, 0.0), axis=0)
        for r in (3.0, 4.0, 6.0):
            real = ap_sum(d, xb, yb, r)
            fk = np.array([ap_sum(df, *sky_xy(CARTER["rho"], CARTER["pa"] + dd, px), r)
                           - ap_sum(d, *sky_xy(CARTER["rho"], CARTER["pa"] + dd, px), r) for dd in FAKE_DPA])
            ratio = real / fk.mean(); err = ratio * fk.std() / fk.mean() / np.sqrt(len(fk))
            log(f"{m:8s} r<={r:.0f}px: companion {real:6.0f}  fakes {fk.mean():6.0f} +/- {fk.std():4.0f} MJy/sr  ->  "
                f"F_meas/F_Carter = {ratio:.2f} +/- {err:.2f}  (dF444W {CARTER['dmag'] - 2.5*np.log10(ratio):.2f})")
            if r == 4.0:
                ratios[m] = (ratio, err)
        pk = [np.nanmax((df - d)[int(sky_xy(CARTER['rho'], CARTER['pa'] + dd, px)[1]) - 3:int(sky_xy(CARTER['rho'], CARTER['pa'] + dd, px)[1]) + 4,
                                  int(sky_xy(CARTER['rho'], CARTER['pa'] + dd, px)[0]) - 3:int(sky_xy(CARTER['rho'], CARTER['pa'] + dd, px)[0]) + 4]) for dd in FAKE_DPA]
        log(f"{m:8s} peak: companion {np.nanmax(d[int(yb)-3:int(yb)+4, int(xb)-3:int(xb)+4]):.1f}, fakes {np.mean(pk):.1f} +/- {np.std(pk):.1f} MJy/sr")
    for rmode in modes:
        rr = results[rmode][0]
        log(f"repair={rmode}: RDI 18 companion peak {np.nanmax(rr['RDI'][int(yb)-3:int(yb)+4, int(xb)-3:int(xb)+4]):.1f} MJy/sr "
            f"(roll 1 {np.nanmax(rr['RDI_roll1'][int(yb)-3:int(yb)+4, int(xb)-3:int(xb)+4]):.1f}, "
            f"roll 2 {np.nanmax(rr['RDI_roll2'][int(yb)-3:int(yb)+4, int(xb)-3:int(xb)+4]):.1f})")
    ok = all(abs(v[0] - 1.0) < 0.25 for v in ratios.values())
    log(f"\n{'PASS' if ok else 'FAIL'}: the companion is {'at' if ok else 'NOT at'} Carter et al.'s flux "
        f"(RDI {ratios['RDI'][0]:.2f}, ADI+RDI {ratios['ADI+RDI'][0]:.2f})")
    with open(os.path.join(a.out, "hip65426_fig3.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
