#!/usr/bin/env python3
"""check_mwc758_paper_setup.py -- the LBTI/ALES reduction of Wagner et al. (2023, Nature Astronomy 7,
1208) run on tutorial 7's release of the same data, as a test of what the release and klip-tpe give
with the paper's own setup rather than a searched one.

  python3 scripts/check_mwc758_paper_setup.py [--data ~/.klip_tpe/data] [--out DIR] [--tutorial]

The paper's setup, from its Methods and from the IDL that made Fig. 1 (reduce_ales_mwc758.pro with
adiklip.pro, sdiklip.pro, get_klip_basis.pro, getzone.pro, nw_ang_comb.pro, destripe.pro):
  * the 2019-11-14 night only, channels 3.4-4.14 um.  The release has group means, not channels:
    here the groups whose mean wavelength is in 3.4-4.14 um (g09-g19, 3.41-4.15 um).
  * every frame destriped (its column medians, then its row medians, subtracted), the frames
    sorted by angle and averaged 15 at a time (the remainder dropped), then high-passed:
    frame - smooth(frame, 11), IDL's boxcar with the edges left unsmoothed.
  * ADI-KLIP in one annulus, 5-27 px from (nx/2, ny/2), cut into n_ang sectors: 6 sectors and 15
    KL modes for point sources (Fig. 1b), the whole annulus and 5 modes for the disk (Fig. 1a).
    References at least ANGSEP = 0.1 lambda/D of arc away at the annulus' mid radius (16 px;
    lambda/D with the 0.035"/px that the call passes), no maximum.  The Methods say "0.1 degree";
    the code's 0.1 lambda/D is 0.8-0.95 deg here.  Fewer than 4 references: all other frames.
  * each residual frame rotated by -(LBT_PARA + 180) with IDL's ROT (clockwise, bilinear), the
    frames of a channel combined with nw_ang_comb (inverse-variance per pixel), then the
    channels averaged: ADIKLIP.
  * SDI-KLIP on the aggressive ADI residuals, frame by frame: for each target channel the other
    channels magnified by lambda_target/lambda_channel (ROT with MAG), 2 KL modes per sector, from
    the channels with |lambda - lambda_target| >= 1.5 FWHM / 16 px.  The code compares microns with
    a unitless number, so that is |dlambda| >= 0.27-0.33 um; with fewer than 4 such channels
    (always the case for the central groups here) all other channels are used.  Then the same
    rotation and nw_ang_comb per channel, and the mean: SDIKLIP.
The same reduction through klip-tpe's ALES reducer (one reduce_config, no search, no destriping,
no SDI step) checks the package against the port.  --tutorial adds tutorial 7's default reduction
(ales.BASELINE: bin 5, high-pass 13, 5 modes, references 0.77 lambda/D to 45 deg away, the annulus
0.4-0.8") of each November night and of both, all 20 groups each.  --no-destripe runs the port
without the IDL's destriping, which neither the release cubes nor klip-tpe apply.

Every product also for the red groups alone (mean wavelength >= 3.8 um, g15-g19), where the
paper's spectrum of c reaches S/N ~3 per channel.  MWC 758c's S/N on every image is the
tutorial's: klip-tpe's MawetPeakSNR at 0.603", PA 228.5 deg (the paper's 2019-11-14 position);
also at that position rotated by 180 deg and mirrored east-west and north-south (an angle
convention error would move c there), and around the whole ring at 0.603" in 5 deg steps (where
c ranks among 73 positions).  Inputs are only read.  Writes to DIR (default
tutorials/runs/mwc758_paper_setup): paper_setup.fits (one extension per image), summary.json
and paper_setup.png.  Needs numpy, scipy, astropy, matplotlib and klip-tpe; Python >= 3.9.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RHO, PA = 0.603, 228.5                  # MWC 758c, 2019-11-14 ALES (Wagner et al. 2023)
NIGHT = "20191114"
LAM_BAND = (3.4, 4.14)                  # um, the paper's channels (start_wave / end_wave)
BIN, HPF = 15, 11                       # frames per bin, high-pass width (2019-11-14 values)
ANN = (5.0, 27.0)                       # px, annmode_inout
ANGSEP, ANGLEMAX = 0.1, 360.0           # lambda/D of arc at the mid radius; degrees
K_SDI, SDISEP = 2, 1.5                  # SDI-KLIP modes; separation in FWHM (as the code applies it)
DIAM = 8.4                              # m, as the IDL calls pass it
PIX_ADI, PIX = 0.035, 0.0345            # arcsec/px: adiklip's call, and the data's
SETUPS = {"aggressive": (15, 6), "nonaggressive": (5, 1)}   # (KL modes, sectors)
RED_UM = 3.8                            # the red sub-band, where the paper's spectrum of c has S/N ~3


# ------------------------------------------------------------------ the IDL, ported
def idl_rot(img, angle, mag=1.0):
    """IDL ROT(img, angle, mag, /INTERP): clockwise by ``angle`` degrees and magnified by ``mag``
    about ((nx-1)/2, (ny-1)/2), bilinear, the nearest edge pixel outside (POLY_2D's default)."""
    ny, nx = img.shape
    xc, yc = (nx - 1) / 2.0, (ny - 1) / 2.0
    th = np.deg2rad(angle)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    dx, dy = (x - xc) / mag, (y - yc) / mag
    xi = xc + dx * np.cos(th) - dy * np.sin(th)
    yi = yc + dx * np.sin(th) + dy * np.cos(th)
    return ndimage.map_coordinates(img, [yi, xi], order=1, mode="nearest")


def destripe(img):
    """destripe(img, 90) then destripe(img, 0) with clip_level=0 (no clipping): each column's
    median, then each row's, subtracted."""
    out = img - np.median(img, axis=0)[None, :]
    return out - np.median(out, axis=1)[:, None]


def idl_smooth(img, w):
    """IDL SMOOTH(img, w): the boxcar mean, the w//2-wide border copied unsmoothed."""
    h = int(w) // 2
    out = img.copy()
    sm = ndimage.uniform_filter(img, size=int(w), mode="nearest")
    out[h:img.shape[0] - h, h:img.shape[1] - h] = sm[h:img.shape[0] - h, h:img.shape[1] - h]
    return out


def zones(shape, ann, n_ang):
    """getzone: r and angle from (nx/2, ny/2), the angle atan2 + pi (angmask), both ends inclusive."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - nx / 2.0, yy - ny / 2.0)
    a = np.arctan2(yy - ny / 2.0, xx - nx / 2.0) + np.pi
    out = []
    for j in range(n_ang):
        a0, a1 = np.deg2rad(j * 360.0 / n_ang), np.deg2rad((j + 1) * 360.0 / n_ang)
        out.append(np.flatnonzero(((a >= a0) & (a <= a1) & (r >= ann[0]) & (r <= ann[1])).ravel()))
    return out


def klip_sub(t, R, k):
    """get_klip_basis + the projection: target and references mean-subtracted over the zone, the
    references' Gram matrix, its eigenvectors in decreasing order scaled by 1/sqrt(eigenvalue)
    (eigenvalues below 5e-5 zeroed), the first k modes."""
    t = t - t.mean()
    R = R - R.mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(R @ R.T)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    w = np.where(np.abs(w) < 5e-5, 0.0, w)
    s = np.sqrt(np.abs(w))
    v = v / np.where(s > 0, s, 1.0)
    Z = (v.T @ R)[:min(int(k), R.shape[0])]
    return t - (t @ Z.T) @ Z


DESTRIPE = True                         # --no-destripe: the port without the IDL's destriping


def prepare(cube, angles):
    """Destripe every frame, sort by angle, average BIN at a time, high-pass HPF."""
    d = np.asarray(cube, float)
    if DESTRIPE:
        d = np.stack([destripe(f) for f in d])
    o = np.argsort(angles, kind="stable")
    d, a = d[o], np.asarray(angles, float)[o]
    nb = len(a) // BIN
    b = d[:nb * BIN].reshape(nb, BIN, *d.shape[1:]).mean(axis=1)
    ba = a[:nb * BIN].reshape(nb, BIN).mean(axis=1)
    return np.stack([f - idl_smooth(f, HPF) for f in b]), ba


def adiklip(b, ba, lam_um, k, n_ang):
    """adiklip, annmode: every binned frame KLIP-subtracted zone by zone."""
    nb, ny, nx = b.shape
    lod = lam_um * 1e-6 / DIAM * 206265.0 / PIX_ADI
    arcdist = 360.0 * lod / (2.0 * np.pi * (ANN[0] + abs(ANN[1] - ANN[0]) / 2.0))
    posang = -ba
    flat = b.reshape(nb, -1)
    out = np.full((nb, ny * nx), np.nan)
    Z = zones((ny, nx), ANN, n_ang)
    for t in range(nb):
        dpa = np.abs(posang - posang[t])
        refs = np.flatnonzero((dpa >= ANGSEP * arcdist) & (dpa <= ANGLEMAX))
        if refs.size < 4:
            refs = np.flatnonzero(np.arange(nb) != t)
        for idx in Z:
            out[t, idx] = klip_sub(flat[t, idx], flat[refs][:, idx], k)
    return out.reshape(nb, ny, nx)


def nw_ang_comb(derot, angles):
    """nw_ang_comb: back to a common orientation, NaN -> 0, the per-pixel variance over the frames,
    rotated back with each frame; the inverse-variance weighted mean."""
    n = derot.shape[0]
    adi = np.stack([idl_rot(f, a) for f, a in zip(derot, angles)])
    adi[~np.isfinite(adi)] = 0.0
    var = adi.var(axis=0, ddof=1)
    adi = np.stack([idl_rot(f, -a) for f, a in zip(adi, angles)])
    varc = np.stack([idl_rot(var, -a) for a in angles])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / (n * np.mean(1.0 / varc, axis=0))) * (n * np.mean(adi / varc, axis=0))


def derotate(cube, angles, offset=180.0):
    return np.stack([idl_rot(f, -a - offset) for f, a in zip(cube, angles)])


def sdiklip(res, ba, lams, n_ang):
    """sdiklip, annmode, on the ADI residuals res (group, frame, y, x), frame by frame."""
    ng, nb, ny, nx = res.shape
    lams = np.asarray(lams, float)
    Z = zones((ny, nx), ANN, n_ang)
    mid = ANN[0] + abs(ANN[1] - ANN[0]) / 2.0
    out = np.full(res.shape, np.nan)
    nref = []
    for k in range(ng):
        fwhm = 1.22 * lams[k] * 1e-6 / DIAM / (np.pi / 180.0) * 3600.0 / PIX
        refs = np.flatnonzero(np.abs(lams - lams[k]) >= SDISEP * fwhm / mid)
        nref.append({"eligible": int(refs.size), "threshold_um": float(SDISEP * fwhm / mid)})
        if refs.size < 4:
            refs = np.flatnonzero(np.arange(ng) != k)
        nref[-1]["used"] = int(refs.size)
        for i in range(nb):
            sc = np.stack([idl_rot(res[kk, i], 0.0, lams[k] / lams[kk]) for kk in range(ng)])
            sc[~np.isfinite(sc)] = 0.0
            flat = sc.reshape(ng, -1)
            o = np.full(ny * nx, np.nan)
            for idx in Z:
                o[idx] = klip_sub(flat[k, idx], flat[refs][:, idx], K_SDI)
            out[k, i] = o.reshape(ny, nx)
    return out, nref


def annulus_mask(shape):
    """The IDL's final mask: NaN within 5 or beyond 27 px of (31, 31)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - 31.0, yy - 31.0)
    return (r < ANN[0]) | (r > ANN[1])


def band_mean(per, gl):
    """The mean of the groups gl's images, NaN outside the annulus (the IDL's final mask)."""
    img = np.mean([per[g] for g in gl], axis=0)
    img[annulus_mask(img.shape)] = np.nan
    return img


def idl_port(groups, lams, angles, log):
    """The paper's ADIKLIP (both setups) and SDIKLIP, group by group.  Returns the per-group
    images ``{product: {group: image}}``."""
    prepped = {}
    for g in groups:
        prepped[g] = prepare(groups[g], angles)
    ba = next(iter(prepped.values()))[1]
    per, residuals = {}, {}
    for name, (k, n_ang) in SETUPS.items():
        per["port_adi_" + name] = {}
        for g in groups:
            t0 = time.time()
            res = adiklip(prepped[g][0], ba, lams[g], k, n_ang)
            if name == "aggressive":
                residuals[g] = res
            per["port_adi_" + name][g] = nw_ang_comb(derotate(res, ba), ba)
            log("  port ADI %-13s %s: %d frames, %.1f s" % (name, g, res.shape[0], time.time() - t0))
    t0 = time.time()
    gl = list(groups)
    sdi, nref = sdiklip(np.stack([residuals[g] for g in gl]), ba, [lams[g] for g in gl], SETUPS["aggressive"][1])
    per["port_sdi_aggressive"] = {g: nw_ang_comb(derotate(sdi[i], ba), ba + 180.0) for i, g in enumerate(gl)}
    log("  port SDI: %.1f s; references per group %s" % (time.time() - t0, [r["used"] for r in nref]))
    return per, {g: r for g, r in zip(gl, nref)}, len(ba)


# ------------------------------------------------------------------ the data and klip-tpe
def release_files(data, nights):
    f = {}
    for d in nights:
        f["meta_%s" % d] = os.path.join(data, "mwc758_ales_%s_meta.fits" % d)
        for g in range(20):
            f["g%02d_%s" % (g, d)] = os.path.join(data, "mwc758_ales_%s_g%02d.fits" % (d, g))
    return f


def band_groups(meta_path):
    t = fits.getdata(meta_path, "GROUPS")
    return [str(r["NAME"]) for r in t if LAM_BAND[0] <= float(r["LAM_MEAN"]) <= LAM_BAND[1]]


def ktp_reduce(red, space, selected, params):
    cfg = space.decode(space.default_vector())
    for d in [cfg.params, *cfg.per_partition.values()]:
        d.update(params)
    return red.reduce_config(cfg, selected=selected).image


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=os.environ.get("KLIP_TPE_DATA") or os.path.expanduser("~/.klip_tpe/data"))
    ap.add_argument("--out", default=os.path.join(REPO, "tutorials", "runs", "mwc758_paper_setup"))
    ap.add_argument("--tutorial", action="store_true", help="also tutorial 7's default reduction")
    ap.add_argument("--no-destripe", action="store_true", help="the port without the IDL's destriping")
    args = ap.parse_args()
    global DESTRIPE
    DESTRIPE = not args.no_destripe
    os.makedirs(args.out, exist_ok=True)
    log = lambda s: print(s, flush=True)

    from klip_tpe.instruments import ales
    from klip_tpe.metrics import MawetPeakSNR
    import klip_tpe
    files = release_files(args.data, [NIGHT])
    groups = band_groups(files["meta_%s" % NIGHT])
    log("klip_tpe %s from %s; %s, groups %s" % (getattr(klip_tpe, "__version__", "?"),
                                                 os.path.dirname(klip_tpe.__file__), NIGHT, groups))
    angles = np.asarray(fits.getdata(files["meta_%s" % NIGHT], "ANGLES"), float)
    para = np.asarray(fits.getdata(files["meta_%s" % NIGHT], "FRAMES")["LBT_PARA"], float)
    log("ANGLES - LBT_PARA: max |diff| %.3g deg over %d frames" % (np.max(np.abs(angles - para)), len(angles)))
    gt = {str(r["NAME"]): r for r in fits.getdata(files["meta_%s" % NIGHT], "GROUPS")}
    lams = {g: float(gt[g]["LAM_MEAN"]) for g in groups}
    cubes = {g: fits.getdata(files["g%s_%s" % (g[1:], NIGHT)]) for g in groups}

    t0 = time.time()
    per, nref, nbin = idl_port(cubes, lams, para, log)
    log("IDL port: %.0f s, %d binned frames" % (time.time() - t0, nbin))
    del cubes
    red_band = [g for g in groups if lams[g] >= RED_UM]
    imgs = {}
    for band, gl in (("", groups), ("_red", red_band)):
        for name in per:
            imgs[name + band] = band_mean(per[name], gl)

    dsets, lam_m, fwhm, table = ales.load_release(files, nights=[NIGHT], only=groups, log=lambda s: None)
    red = ales.make_reducer(dsets, lam_m, fwhm, table=table, max_workers=1)
    space = ales.make_space(red, bin_range=(1, 30))
    sel = red.partitions()
    sel_red = [p for p in sel if lam_m[p] * 1e6 >= RED_UM]
    for name, (k, n_ang) in SETUPS.items():
        for band, ps in (("", sel), ("_red", sel_red)):
            t0 = time.time()
            imgs["ktp_adi_" + name + band] = ktp_reduce(red, space, ps, dict(
                bin=BIN, filter=HPF, k_klip=k, n_ang=n_ang, angsep=ANGSEP, anglemax=ANGLEMAX,
                inrad=ANN[0], outrad=ANN[1], pad_in=False, pad_out=False))
            log("klip-tpe ADI %s%s: %.0f s" % (name, band, time.time() - t0))
    metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None)
    del red, dsets

    if args.tutorial:
        tut = dict(ales.BASELINE, inrad=0.4 / PIX, outrad=0.8 / PIX, pad_in=False, pad_out=False)
        for label, nights in (("tutorial_default_nov13", ["20191113"]), ("tutorial_default_nov14", [NIGHT]),
                              ("tutorial_default_both", ["20191113", NIGHT])):
            t0 = time.time()
            f2 = release_files(args.data, nights)
            d2, l2, w2, t2 = ales.load_release(f2, log=lambda s: None)
            r2 = ales.make_reducer(d2, l2, w2, table=t2, max_workers=1)
            s2 = ales.make_space(r2, bin_range=(5, 30))
            imgs[label] = ktp_reduce(r2, s2, r2.partitions(), tut)
            log("%s (%d partitions): %.0f s" % (label, len(r2.partitions()), time.time() - t0))
            del r2, d2

    snr, where = {}, {}
    pas = np.arange(0.0, 360.0, 5.0)
    alt = {"c": PA, "rotated_180": (PA + 180.0) % 360.0, "mirrored_EW": (360.0 - PA) % 360.0,
           "mirrored_NS": (180.0 - PA) % 360.0}
    log("  %-30s %6s %8s %8s %8s   ring of 72 at %.3f\": max at PA, c's rank" % (
        "image", "c", "rot180", "mirEW", "mirNS", RHO))
    for name, img in imgs.items():
        im = np.nan_to_num(img)
        snr[name] = float(metric.per_source(im, None, [RHO], [PA])[0])
        a = {k: float(metric.per_source(im, None, [RHO], [v])[0]) for k, v in alt.items()}
        ring = np.array(metric.per_source(im, None, [RHO] * len(pas), list(pas)), float)
        rank = int(np.sum(ring > snr[name])) + 1
        where[name] = {"alternatives": a, "ring_max": float(ring.max()), "ring_max_pa": float(pas[np.argmax(ring)]),
                       "ring_median": float(np.median(ring)), "ring_std": float(ring.std()), "c_rank_of_73": rank}
        log("  %-30s %6.2f %8.2f %8.2f %8.2f   %5.2f at %5.1f, c %d of 73" % (
            name, snr[name], a["rotated_180"], a["mirrored_EW"], a["mirrored_NS"], ring.max(),
            pas[np.argmax(ring)], rank))

    hdus = [fits.PrimaryHDU()]
    for name, img in imgs.items():
        h = fits.ImageHDU(np.asarray(img, np.float32), name=name[:68])
        h.header["CSNR"] = (snr[name], "MWC 758c, MawetPeakSNR at 0.603 228.5")
        hdus.append(h)
    hdus[0].header["NIGHT"] = NIGHT
    hdus[0].header["DESTRIPE"] = (DESTRIPE, "the port destriped the frames")
    hdus[0].header["GROUPS"] = ",".join(groups)
    hdus[0].header["SCRIPT"] = "scripts/check_mwc758_paper_setup.py"
    with open(os.path.join(args.out, "paper_setup.fits"), "wb") as fh:
        fits.HDUList(hdus).writeto(fh)
    summary = {"night": NIGHT, "groups": groups, "lam_mean_um": lams, "binned_frames": nbin, "destripe": DESTRIPE,
               "setup": {"bin": BIN, "highpass": HPF, "annulus_px": ANN, "angsep_lod": ANGSEP,
                         "anglemax": ANGLEMAX, "setups": {k: {"k_klip": v[0], "n_ang": v[1]} for k, v in SETUPS.items()},
                         "sdi": {"k": K_SDI, "sep_fwhm": SDISEP}},
               "red_band": {"from_um": RED_UM, "groups": red_band},
               "sdi_references": nref, "c": {"rho": RHO, "pa": PA}, "snr": snr, "positions": where,
               "klip_tpe": os.path.dirname(klip_tpe.__file__)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    figure(os.path.join(args.out, "paper_setup.png"), imgs, snr)
    log("wrote %s" % args.out)


def figure(path, imgs, snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    surface, ink, muted, mark = "#fcfcfb", "#2b2b2a", "#8a8a86", "#d9d9d6"
    cmap = LinearSegmentedColormap.from_list("bw", ["#000000", "#ffffff"])
    cmap.set_bad(surface)
    titles = {"port_adi_nonaggressive": "IDL port, 3.4-4.14 µm: ADI, 5 modes (Fig. 1a)",
              "port_adi_aggressive": "IDL port, 3.4-4.14 µm: ADI, 15 modes, 6 sectors",
              "port_sdi_aggressive": "IDL port, 3.4-4.14 µm: then SDI (Fig. 1b)",
              "port_adi_nonaggressive_red": "IDL port, 3.8-4.14 µm: ADI, 5 modes",
              "port_adi_aggressive_red": "IDL port, 3.8-4.14 µm: ADI, 15 modes, 6 sectors",
              "port_sdi_aggressive_red": "IDL port, 3.8-4.14 µm: then SDI",
              "ktp_adi_nonaggressive": "klip-tpe, 3.4-4.14 µm: ADI, 5 modes",
              "ktp_adi_aggressive": "klip-tpe, 3.4-4.14 µm: ADI, 15 modes, 6 sectors",
              "ktp_adi_aggressive_red": "klip-tpe, 3.8-4.14 µm: ADI, 15 modes, 6 sectors",
              "tutorial_default_nov13": "tutorial 7 default: Nov 13, 20 groups",
              "tutorial_default_nov14": "tutorial 7 default: Nov 14, 20 groups",
              "tutorial_default_both": "tutorial 7 default: Nov 13 + 14, 40 groups"}
    rows = [["port_adi_nonaggressive", "port_adi_aggressive", "port_sdi_aggressive"],
            ["port_adi_nonaggressive_red", "port_adi_aggressive_red", "port_sdi_aggressive_red"],
            ["ktp_adi_nonaggressive", "ktp_adi_aggressive", "ktp_adi_aggressive_red"],
            ["tutorial_default_nov13", "tutorial_default_nov14", "tutorial_default_both"]]
    rows = [[n for n in r if n in imgs] for r in rows]
    rows = [r for r in rows if r]
    ncol, nrow = 3, len(rows)
    half = 31 * PIX
    ext = [half, -half, -half, half]
    cx, cy = RHO * np.sin(np.radians(PA)), RHO * np.cos(np.radians(PA))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.3 * nrow + 0.5), facecolor=surface, squeeze=False)
    yy, xx = np.mgrid[0:62, 0:62]
    rr = np.hypot(xx - 30.5, yy - 30.5)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, n in [(axes[i, j], n) for i, r in enumerate(rows) for j, n in enumerate(r)]:
        ax.set_visible(True)
        img = imgs[n]
        ring = np.isfinite(img) & (rr >= 6) & (rr <= 26)
        v = float(np.nanpercentile(np.abs(img[ring]), 99.5)) if ring.any() else 1.0
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0.0, vmax=v, interpolation="nearest")
        ax.add_patch(Circle((cx, cy), 0.08, fill=False, ec=mark, lw=0.8))
        ax.set_xlim(1.0, -1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("%s\nc at S/N %.1f" % (titles[n], snr[n]), fontsize=9, color=ink, loc="left")
        ax.tick_params(labelsize=7, colors=muted)
        for s in ax.spines.values():
            s.set_color(muted)
            s.set_linewidth(0.6)
    fig.suptitle("MWC 758, LBTI/ALES 2019-11-14: the Wagner et al. (2023) setup on the tutorial-7 release "
                 "(c circled, 0.603\", PA 228.5°)", fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "North up, east left; axes in arcsec.  Grey scale linear from 0 (negatives black) to "
             "each image's 99.5th percentile of |value| at 6-26 px.  S/N: klip-tpe MawetPeakSNR, as tutorial 7.",
             fontsize=7.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
