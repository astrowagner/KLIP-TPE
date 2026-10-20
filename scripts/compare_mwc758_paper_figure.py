#!/usr/bin/env python3
"""compare_mwc758_paper_figure.py -- the published Fig. 1 of Wagner et al. (2023, Nature Astronomy 7,
1208; MWC 758, LBTI/ALES, UT 2019-11-14) side by side with our reprocessing, and a pixel-by-pixel test
that our port of the paper's IDL reduction (scripts/mwc758_idl2022.py) is the same reduction.

  python3 scripts/compare_mwc758_paper_figure.py [--make-input] [--reduced DIR] [--idl-input DIR]
          [--release DIR] [--published PNG] [--out DIR] [--workers N]

Inputs (all read-only):
  --reduced    /Volumes/RAID36TB/LMIRCam/MWC758_6/reduced: the paper's products (FITS of 2022) and,
               for --make-input only, MWC758_cube_cen_clean.fits + MWC758_primary_angles_clean.sav
  --idl-input  the binned input, made by --make-input from those two files exactly as the IDL bins
               (frames sorted by angle, means of 15, remainder dropped): cube_bin15_ch*.npy (all 98
               channels in four chunks) and angles_bin15.npy
  --release    tutorial 7's release (mwc758_ales_20191114_meta.fits and _g09.._g19.fits)
  --published  a capture of the published Fig. 1 (springernature.com, 41550_2023_2028_Fig1_HTML.png,
               shown at 1400 px wide); its colour bar gives the colour map
What it runs (cached in OUT/cache, so a rerun only redraws):
  1. ADI-KLIP of channels 4-89 (2.9-4.2 um): bin 15, high-pass 11, 15 modes in 6 sectors of 5-27 px.
  2. the disk setup of Fig. 1a: 5 modes, the whole annulus, channels 31-84 (3.39-4.14 um).
  3. SDI-KLIP as the public script has it (Aug 2022): channels 31-84, the ADI residuals high-passed
     again with smooth(/nan), 2 modes, references 1.5 FWHM/16 px apart in lambda.
  4. SDI-KLIP as it ran on 2022-02-11, which made the published b, d, e, f: channels 4-89 (the
     script's commented-out start_wave 2.9 / end_wave 4.2 line), NO second high-pass.
  5. step 1+3 and the disk on the release's 11 group means (g09-g19) instead of the 54 channels.
  6. yesterday's check (scripts/check_mwc758_paper_setup.py: destriped, ANGSEP 0.1, zones about
     (31, 31), no second high-pass, release groups) for comparison.
Then: every surviving 2022 product compared with the port (correlation, rms of the difference); every
published panel compared with every row (rank correlation of the panel's brightness with the image,
geometry fitted to the capture); MWC 758c's S/N (Mawet et al. 2014, lambda/D aperture, this night
only); and two figures.  Writes OUT/side_by_side.png, OUT/identity.png, OUT/products.fits,
OUT/summary.json.  Needs numpy, scipy, astropy, matplotlib, Pillow; Python >= 3.9.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # parallel by process; set before numpy loads
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mwc758_idl2022 as M  # noqa: E402

RUN = os.path.join(REPO, "tutorials", "runs", "mwc758_paper_setup")
C_XY = (44.1, 19.4)            # MWC 758c, 0.603" at PA 228.5 (the paper's 2019-11-14 astrometry), data px
CH_LO, CH_HI = 31, 84          # 3.39-4.14 um: start_wave / end_wave for 3.4 and 4.14
FEB_LO, FEB_HI = M.wave_index(2.9), M.wave_index(4.2)          # 4, 89
BANDS = {"full": (3.4, 4.14), "red": (3.8, 4.14), "d": (3.4, 3.8), "e": (3.8, 3.97), "f": (3.97, 4.14), "all": (0.0, 9.9)}
COLS = ["a", "b", "d", "e", "f"]

# ------------------------------------------------------------------ the published figure
# capture boxes (x0, y0, x1, y1) and fitted geometry (capture px per data px; capture position of data
# pixel (31, 31)), data y up.  Fitted by maximising the rank correlation with the rows that match.
BOX = {"a": (0, 25, 462, 489), "b": (464, 25, 928, 489), "d": (0, 491, 466, 851),
       "e": (467, 491, 933, 851), "f": (934, 491, 1400, 851)}
GEOM = {"a": (7.549, 233.0, 254.2), "b": (7.523, 698.9, 255.1), "d": (7.501, 234.0, 673.0),
        "e": (7.504, 700.5, 673.3), "f": (7.486, 1167.1, 671.5)}
COL_TITLES = {"a": "a   ADI, disk setup, 3.4-4.14 µm", "b": "b   ASDI, point-source setup",
              "d": "d   ASDI, 3.4-3.8 µm", "e": "e   ASDI, 3.8-3.97 µm", "f": "f   ASDI, 3.97-4.14 µm"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Published:
    def __init__(self, path):
        from PIL import Image
        self.rgb = np.asarray(Image.open(path).convert("RGB"))
        self.L = (0.299 * self.rgb[..., 0] + 0.587 * self.rgb[..., 1] + 0.114 * self.rgb[..., 2]).astype(float)
        bar = self.rgb[110:467, 1336:1344].astype(float).mean(axis=1)[::-1] / 255.0     # colour bar of panel c
        self.lut = np.vstack([np.linspace([0.0, 0.0, 0.0], bar[0], 24, endpoint=False), bar])  # + black below

    def cmap(self):
        from matplotlib.colors import ListedColormap
        return ListedColormap(self.lut, name="fig1")

    def crop(self, col):
        x0, y0, x1, y1 = BOX[col]
        s, cx, cy = GEOM[col]
        ext = [31 + (x0 - 0.5 - cx) / s, 31 + (x1 + 0.5 - cx) / s, 31 - (y1 + 0.5 - cy) / s, 31 - (y0 - 0.5 - cy) / s]
        return self.rgb[y0:y1 + 1, x0:x1 + 1], ext

    def _grid(self, col, rlo, rhi):
        s, cx, cy = GEOM[col]
        yy, xx = np.mgrid[0:62, 0:62]
        px, py = cx + (xx - 31) * s, cy - (yy - 31) * s
        x0, y0, x1, y1 = BOX[col]
        r = np.hypot(xx - 31, yy - 31)
        return (r >= rlo) & (r <= rhi) & (py > y0 + 2) & (py < y1 - 2) & (px > x0 + 2) & (px < x1 - 2), px, py

    def rank_corr(self, col, img, rlo=7, rhi=21):
        from scipy import ndimage, stats
        ok, px, py = self._grid(col, rlo, rhi)
        ok &= np.isfinite(img)
        return float(stats.spearmanr(ndimage.map_coordinates(self.L, [py[ok], px[ok]], order=1), img[ok]).correlation)

    def fit_stretch(self, col, img, rlo=7.5, rhi=21.0):
        """(vmin, vmax, rms): the linear map onto the colour map that best reproduces the panel's colours
        from img (least squares in RGB over the ring; white overlays and c's circle excluded)."""
        from scipy import ndimage, optimize
        ok, px, py = self._grid(col, rlo, rhi)
        yy, xx = np.mgrid[0:62, 0:62]
        ok &= np.isfinite(img) & (np.abs(np.hypot(xx - C_XY[0], yy - C_XY[1]) - 3.6) > 1.2)
        pub = np.stack([ndimage.map_coordinates(self.rgb[..., k].astype(float), [py[ok], px[ok]], order=1)
                        for k in range(3)], 1) / 255.0
        keep = pub.min(axis=1) < 0.97
        v, pub = img[ok][keep], pub[keep]
        lut, n = self.lut, len(self.lut) - 1

        def cost(p):
            lo, hi = p
            if hi <= lo:
                return 1e9
            t = np.clip((v - lo) / (hi - lo), 0, 1) * n
            i0 = np.floor(t).astype(int)
            i1 = np.minimum(i0 + 1, n)
            f = (t - i0)[:, None]
            return float(np.mean(np.sum((lut[i0] * (1 - f) + lut[i1] * f - pub) ** 2, axis=1)))
        q = np.percentile(v, [1, 99])
        grid = [(cost((lo, hi)), lo, hi) for lo in np.linspace(2 * q[0], -0.1 * q[1], 15)
                for hi in np.linspace(0.4 * q[1], 1.6 * q[1], 15)]
        best = min(grid)
        res = optimize.minimize(cost, best[1:], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=800))
        return float(res.x[0]), float(res.x[1]), float(np.sqrt(res.fun / 3.0))


# ------------------------------------------------------------------ S/N of c
def aperture_sums(img, xs, ys, rad, sub=8):
    a = np.nan_to_num(np.asarray(img, float))
    o = (np.arange(sub) + 0.5) / sub - 0.5
    out = []
    for x0, y0 in zip(xs, ys):
        tot = 0.0
        for yy in range(max(int(np.floor(y0 - rad - 1)), 0), min(int(np.ceil(y0 + rad + 1)) + 1, a.shape[0])):
            for xx in range(max(int(np.floor(x0 - rad - 1)), 0), min(int(np.ceil(x0 + rad + 1)) + 1, a.shape[1])):
                gx, gy = np.meshgrid(xx + o, yy + o)
                tot += np.mean(np.hypot(gx - x0, gy - y0) <= rad) * a[yy, xx]
        out.append(tot)
    return np.array(out)


def snr_c(img, lam_um, cen=(31.0, 31.0)):
    """Mawet et al. (2014): an aperture of diameter lambda/D on c against the n-1 others around the ring
    at its separation (n = floor(2 pi rho / (lambda/D))), times the small-sample factor."""
    lod = lam_um * 1e-6 / M.DIAM * 206265.0 / 0.0345
    dx, dy = C_XY[0] - cen[0], C_XY[1] - cen[1]
    rho, th0 = np.hypot(dx, dy), np.arctan2(dy, dx)
    n = int(np.floor(2 * np.pi * rho / lod))
    th = th0 + 2 * np.pi * np.arange(n) / n
    f = aperture_sums(img, cen[0] + rho * np.cos(th), cen[1] + rho * np.sin(th), lod / 2.0)
    o = f[1:]
    return float((f[0] - o.mean()) / (o.std(ddof=1) * np.sqrt(1 + 1.0 / len(o))))


# ------------------------------------------------------------------ the reductions (parallel, cached)
_G = {}


def _init(d):
    """Worker initialiser: the shared arrays arrive once per worker (fork or spawn alike)."""
    _G.update(d)


def _adi_worker(args):
    j, k, n_ang = args
    res, km = M.adi_channel_products(_G["hp"][j], _G["ang"], float(_G["waves"][j]), k, n_ang)
    return j, res.astype(np.float32), km         # the IDL's final_cube is single precision too


def adi_cube(cube, ang, waves, k, n_ang, workers, log):
    """ADI-KLIP of every channel of cube (nwave, nframe, y, x): residual frames (float32) and images."""
    d = dict(hp=M.highpass(cube, 11), ang=np.asarray(ang, float), waves=np.asarray(waves))
    res = np.empty(cube.shape, np.float32)
    img = np.empty((cube.shape[0], 62, 62))
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(d,)) as ex:
        for j, r, km in ex.map(_adi_worker, [(j, k, n_ang) for j in range(cube.shape[0])]):
            res[j], img[j] = r, km
    log("    ADI %d channels, k %d, %d sectors: %.0f s" % (cube.shape[0], k, n_ang, time.time() - t0))
    return res, img


def _sdi_worker(kt):
    return kt, M.sdi_target(_G["adic"], _G["ang"], _G["waves"], kt, refsets=_G["refs"])


def sdi_cube(adic, ang, waves, targets, workers, log):
    """SDI-KLIP of the target channels (indices into adic's channels); unmasked images."""
    refs = M.sdi_refs(waves)
    d = dict(adic=np.asarray(adic, float), ang=np.asarray(ang, float), waves=np.asarray(waves, np.float32), refs=refs)
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(d,)) as ex:
        for kt, im in ex.map(_sdi_worker, list(targets)):
            out[kt] = im
    log("    SDI %d targets from %d channels: %.0f s" % (len(out), adic.shape[0], time.time() - t0))
    return out, [len(r) for r in refs]


def cached(path, fn, log):
    if os.path.exists(path):
        log("  cached: %s" % os.path.basename(path))
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    d = fn()
    np.savez(path, **d)
    return d


def make_input(reduced, out, log):
    """--make-input: the binned cube of all 98 channels from MWC758_cube_cen_clean.fits, as the IDL bins."""
    from scipy.io import readsav
    os.makedirs(out, exist_ok=True)
    ang = np.asarray(readsav(os.path.join(reduced, "MWC758_primary_angles_clean.sav"))["angles"], float)
    o = np.argsort(ang, kind="stable")
    nb = len(ang) // 15
    np.save(os.path.join(out, "angles_bin15.npy"), ang[o][:nb * 15].reshape(nb, 15).mean(axis=1))
    with fits.open(os.path.join(reduced, "MWC758_cube_cen_clean.fits"), memmap=True) as h:
        for c0, c1 in ((0, 31), (31, 49), (49, 85), (85, 98)):
            blk = np.empty((c1 - c0, nb, 62, 62), np.float32)
            for i, ch in enumerate(range(c0, c1)):
                d = np.asarray(h[0].data[ch], np.float64)[o]
                blk[i] = d[:nb * 15].reshape(nb, 15, 62, 62).mean(axis=1)
            np.save(os.path.join(out, "cube_bin15_ch%02d-%02d.npy" % (c0, c1 - 1)), blk)
            log("  binned channels %d-%d" % (c0, c1 - 1))


def load_input(d):
    chunks = sorted(f for f in os.listdir(d) if f.startswith("cube_bin15_ch") and f.endswith(".npy"))
    parts, have = [], []
    for f in chunks:
        a, b = [int(v) for v in f[len("cube_bin15_ch"):-4].split("-")]
        parts.append((a, np.load(os.path.join(d, f), mmap_mode="r")))
        have += list(range(a, b + 1))
    if sorted(have) != list(range(98)):
        raise SystemExit("idl-input must hold all 98 channels (run --make-input)")
    cube = np.empty((98,) + parts[0][1].shape[1:], np.float32)
    for a, p in parts:
        cube[a:a + p.shape[0]] = p
    return cube, np.load(os.path.join(d, "angles_bin15.npy")), {f: md5(os.path.join(d, f)) for f in chunks + ["angles_bin15.npy"]}


def release(rel):
    meta = os.path.join(rel, "mwc758_ales_20191114_meta.fits")
    G = fits.getdata(meta, "GROUPS")
    groups = [str(r["NAME"]) for r in G if 3.4 <= float(r["LAM_MEAN"]) <= 4.14]
    lams = np.array([float(r["LAM_MEAN"]) for r in G if str(r["NAME"]) in groups], np.float32)
    para = np.asarray(fits.getdata(meta, "FRAMES")["LBT_PARA"], float)
    cubes = np.stack([fits.getdata(os.path.join(rel, "mwc758_ales_20191114_%s.fits" % g)).astype(np.float32) for g in groups])
    return groups, lams, para, cubes


def corr_rms(a, b, m):
    a, b = a[m], b[m]
    return float(np.corrcoef(a, b)[0, 1]), float((a - b).std() / b.std())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--make-input", action="store_true", help="bin the IDL's cube into --idl-input first")
    ap.add_argument("--reduced", default="/Volumes/RAID36TB/LMIRCam/MWC758_6/reduced")
    ap.add_argument("--idl-input", default=os.path.join(RUN, "idl_input"))
    ap.add_argument("--release", default=os.environ.get("KLIP_TPE_DATA") or os.path.expanduser("~/.klip_tpe/data"))
    ap.add_argument("--published", default=os.path.join(RUN, "paper_figure", "published_fig1_capture.png"))
    ap.add_argument("--out", default=os.path.join(RUN, "paper_figure"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    args = ap.parse_args()
    t00 = time.time()
    log = lambda s: print(s, flush=True)
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    cache = lambda name: os.path.join(args.out, "cache", name)
    R = lambda name: os.path.join(args.reduced, name)
    if args.make_input:
        make_input(args.reduced, args.idl_input, log)

    cube98, ang, in_md5 = load_input(args.idl_input)
    W = M.WAVES98
    from scipy.io import readsav
    nw = readsav(R("MWC758_newwaves.sav"))["waves"].astype(np.float32)
    assert np.array_equal(nw, W[CH_LO:CH_HI + 1]), "MWC758_newwaves.sav != channels 31-84"
    log("input: %s, %d binned frames, angles %.2f to %.2f" % (cube98.shape, len(ang), ang.min(), ang.max()))

    # 1-2. ADI
    adi = cached(cache("adi_k15_6sect_ch04-89.npz"), lambda: dict(zip(
        ("res", "img"), adi_cube(cube98[FEB_LO:FEB_HI + 1], ang, W[FEB_LO:FEB_HI + 1], 15, 6, args.workers, log))), log)
    sl = slice(CH_LO - FEB_LO, CH_HI - FEB_LO + 1)                 # channels 31-84 within 4-89
    disk = cached(cache("adi_k5_1sect_ch31-84.npz"), lambda: {
        "img": adi_cube(cube98[CH_LO:CH_HI + 1], ang, W[CH_LO:CH_HI + 1], 5, 1, args.workers, log)[1]}, log)
    # 3. SDI, public script (Aug 2022)
    def run_final():
        adic = M.highpass(adi["res"][sl], 11, nan="2d")
        out, nref = sdi_cube(adic, ang, W[CH_LO:CH_HI + 1], range(CH_HI - CH_LO + 1), args.workers, log)
        return {"xyl": np.stack([out[k] for k in range(len(out))]), "nref": np.array(nref)}
    sdi_final = cached(cache("sdi_final_ch31-84.npz"), run_final, log)
    # 4. SDI as on 2022-02-11
    def run_feb():
        tg = range(CH_LO - FEB_LO, CH_HI - FEB_LO + 1)
        out, nref = sdi_cube(adi["res"], ang, W[FEB_LO:FEB_HI + 1], tg, args.workers, log)
        return {"xyl": np.stack([out[k] for k in tg]), "nref": np.array(nref)[list(tg)]}
    sdi_feb = cached(cache("sdi_feb2022_ch04-89_targets31-84.npz"), run_feb, log)
    # 5. the release groups
    def run_release():
        groups, lams, para, cubes = release(args.release)
        b, ba = M.bin_frames(cubes, para, 15)
        b = b.astype(np.float32)
        res, img = adi_cube(b, ba, lams, 15, 6, args.workers, log)
        out, nref = sdi_cube(M.highpass(res, 11, nan="2d"), ba, lams, range(len(groups)), args.workers, log)
        _, dimg = adi_cube(b, ba, lams, 5, 1, args.workers, log)
        return {"groups": np.array(groups), "lams": lams, "sdi": np.stack([out[k] for k in range(len(groups))]),
                "disk": dimg, "nref": np.array(nref), "binned_angles": ba}
    rel = cached(cache("release_final_code.npz"), run_release, log)
    # 6. yesterday's check
    def run_yesterday():
        spec = importlib.util.spec_from_file_location("chk", os.path.join(HERE, "check_mwc758_paper_setup.py"))
        chk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chk)
        chk.DESTRIPE = True
        groups, lams, para, cubes = release(args.release)
        per, nref, nbin = chk.idl_port({g: cubes[i] for i, g in enumerate(groups)},
                                       {g: float(lams[i]) for i, g in enumerate(groups)}, para, log)
        return {"groups": np.array(groups), "lams": lams,
                **{n: np.stack([per[n][g] for g in groups]) for n in per}}
    yest = cached(cache("release_yesterdays_check.npz"), run_yesterday, log)

    # ---------------------------------------------------------- identity with the 2022 products
    r = M.radius31()
    ring = (r >= 6) & (r <= 22)
    W54 = W[CH_LO:CH_HI + 1]
    adi54 = M.final_adi_masks(adi["img"][sl])
    sdi54 = M.mask_annulus(sdi_final["xyl"], 6, 22)
    feb54 = M.mask_annulus(sdi_feb["xyl"], 6, 22)
    disk_img = M.final_adi_masks(disk["img"]).mean(axis=0)
    ident = {}
    for name, fn, port in (("ADI cube, 15 modes", "MWC758_adiklip_xyl_3.4-4.14_ak15_sk2.fits", adi54),
                           ("ASDI cube, public script", "MWC758_sdiklip_xyl_3.4-4.14_ak15_sk2.fits", sdi54)):
        ref = fits.getdata(R(fn)).astype(float)
        st = [corr_rms(port[i], ref[i], ring) for i in range(54)]
        ident[name] = {"file": fn, "date": fits.getheader(R(fn)).get("DATE"), "md5": md5(R(fn)),
                       "corr": [s[0] for s in st], "rms_frac": [s[1] for s in st],
                       "channel_mean": corr_rms(port.mean(0), ref.mean(0), ring)}
    ref = fits.getdata(R("MWC758_adiklip_disk.fits")).astype(float)
    ident["disk image (Fig. 1a)"] = {"file": "MWC758_adiklip_disk.fits", "date": fits.getheader(R("MWC758_adiklip_disk.fits")).get("DATE"),
                                     "md5": md5(R("MWC758_adiklip_disk.fits")), "image": corr_rms(disk_img, ref, ring)}
    feb_bands = {}
    for band in ("3.4-3.5", "3.4-3.6", "3.4-3.8", "3.5-3.8", "3.6-3.8", "3.8-4.14"):
        lo, hi = [float(v) for v in band.split("-")]
        fn = "MWC758_sdiklip_%s.fits" % band
        if os.path.exists(R(fn)):
            feb_bands[band] = {"md5": md5(R(fn)), "image": corr_rms(feb54[M.band_select(W54, lo, hi)].mean(0), fits.getdata(R(fn)).astype(float), ring)}
    ident["ASDI bands of 2022-02-11"] = feb_bands
    if os.path.exists(R("MWC758_adiklip_xyl_2.9-4.3.fits")):
        ref = fits.getdata(R("MWC758_adiklip_xyl_2.9-4.3.fits")).astype(float)     # channels 4-97 (2022-03-23)
        port = M.final_adi_masks(adi["img"])
        st = [corr_rms(port[i], ref[i], ring) for i in range(FEB_HI - FEB_LO + 1)]
        ident["ADI cube 2.9-4.3 um (2022-03-23), channels 4-89"] = {"corr_min": min(s[0] for s in st),
                                                                   "rms_frac_median": float(np.median([s[1] for s in st]))}
    for k, v in ident.items():
        if "corr" in v:
            log("%-34s vs %s: corr per channel min %.6f, median %.6f; rms(diff)/rms median %.2e, max %.2e" % (
                k, v["file"], min(v["corr"]), np.median(v["corr"]), np.median(v["rms_frac"]), max(v["rms_frac"])))
        elif "image" in v:
            log("%-34s vs %s: corr %.7f, rms(diff)/rms %.2e" % (k, v["file"], *v["image"]))
        elif k.startswith("ASDI bands"):
            for b, s in v.items():
                log("%-34s %-9s: corr %.7f, rms(diff)/rms %.2e" % (k, b, *s["image"]))
        else:
            log("%-34s %s" % (k, v))

    # ---------------------------------------------------------- the rows of the figure
    def bands_of(cube, waves, keys):
        out = {}
        for col, key in keys:
            s = M.band_select(waves, *BANDS[key])
            out[col] = (cube[s].mean(axis=0), float(np.mean(np.asarray(waves)[s])), "%.2f-%.2f µm" % BANDS[key])
        return out
    rows = []
    disk_feb = fits.getdata(R("MWC758_adiklip_disk.fits")).astype(float)
    f38, f34 = fits.getdata(R("MWC758_sdiklip_3.8-4.14.fits")).astype(float), fits.getdata(R("MWC758_sdiklip_3.4-3.8.fits")).astype(float)
    rows.append(("The paper's FITS on disk\n(MWC758_6/reduced)",
                 {"a": (disk_feb, 3.77, "adiklip_disk.fits, 2022-07-19"), "b": (f38, 3.97, "sdiklip_3.8-4.14.fits, 2022-02-11"),
                  "d": (f34, 3.60, "sdiklip_3.4-3.8.fits, 2022-02-11"), "e": None, "f": None}))
    rows.append(("Our port, paper's input:\nthe runs the panels came from",
                 {"a": (disk_img, 3.77, "5 modes, whole annulus"),
                  **bands_of(feb54, W54, (("b", "red"), ("d", "d"), ("e", "e"), ("f", "f")))}))
    rows.append(("Our port, paper's input:\nthe public script (Aug 2022)",
                 {"a": (disk_img, 3.77, "5 modes, whole annulus"),
                  **bands_of(sdi54, W54, (("b", "full"), ("d", "d"), ("e", "e"), ("f", "f")))}))
    rsd = M.mask_annulus(rel["sdi"], 6, 22)
    rows.append(("Public script on the release:\n11 group means, not 54 channels",
                 {"a": (M.final_adi_masks(rel["disk"]).mean(axis=0), 3.77, "groups g09-g19"),
                  **bands_of(rsd, rel["lams"], (("b", "full"), ("d", "d"), ("e", "e"), ("f", "f")))}))
    # panel b's display: the script's _conv recipe (each channel of the UNMASKED cube convolved with its
    # lambda/D Gaussian, the band mean, zero within 5 and beyond 27 px) -- the published b has signal out to
    # ~27 px, which smoothing an image masked at 22 px cannot give; d-f: the masked band image, smoothed
    def conv_band(cube_unmasked, waves, key):
        s = np.flatnonzero(M.band_select(waves, *BANDS[key]))
        return M.mask_annulus(np.mean([M.lod_smooth(np.nan_to_num(cube_unmasked[i]), float(waves[i])) for i in s], axis=0), 5, 27)
    b_disp = {1: conv_band(sdi_feb["xyl"], W54, "red"), 2: conv_band(sdi_final["xyl"], W54, "all"),
              3: conv_band(rel["sdi"], rel["lams"], "all")}
    ysd = M.mask_annulus(yest["port_sdi_aggressive"], 5, 27)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        ya = M.mask_annulus(np.nanmean(yest["port_adi_nonaggressive"], axis=0), 5, 27)
    rows.append(("Yesterday's check:\nrelease groups, Jan-epoch steps",
                 {"a": (ya, 3.77, "destriped, ANGSEP 0.1"),
                  **bands_of(ysd, yest["lams"], (("b", "full"), ("d", "d"), ("e", "e"), ("f", "f")))}))
    pub = Published(args.published)
    smooth = lambda col, img, lam: img if col == "a" else M.lod_smooth(img, lam)
    disp = [{c: (None if v is None else (b_disp[i] if (c == "b" and i in b_disp) else smooth(c, v[0], v[1])))
             for c, v in row[1].items()} for i, row in enumerate(rows)]
    match = [{c: (None if v is None else pub.rank_corr(c, v)) for c, v in d.items()} for d in disp]
    snr = [{c: (None if v is None else snr_c(v[0], v[1])) for c, v in row[1].items()} for row in rows]
    stretch = {}
    for c in COLS:
        src = disp[0][c] if disp[0][c] is not None else disp[1][c]
        stretch[c] = pub.fit_stretch(c, src) + ("row 2" if disp[0][c] is not None else "row 3",)
    # which band panel b shows (the caption says 3.4-4.14 um)
    bfeb = {k: pub.rank_corr("b", M.lod_smooth(feb54[M.band_select(W54, *BANDS[k])].mean(0), float(np.mean(W54[M.band_select(W54, *BANDS[k])]))))
            for k in ("full", "red", "d")}
    log("panel b vs the 2022-02-11 run: 3.4-4.14 um %.3f, 3.8-4.14 um %.3f, 3.4-3.8 um %.3f" % (bfeb["full"], bfeb["red"], bfeb["d"]))
    bconv = {k: pub.rank_corr("b", conv_band(sdi_feb["xyl"], W54, k), 7, 26) for k in ("full", "red", "d")}
    bconv["red, masked 6-22 then smoothed"] = pub.rank_corr("b", M.lod_smooth(feb54[M.band_select(W54, *BANDS["red"])].mean(0), 3.97), 7, 26)
    log("  same at 7-26 px with the _conv recipe: 3.4-4.14 um %.3f, 3.8-4.14 um %.3f, 3.4-3.8 um %.3f; "
        "3.8-4.14 masked at 22 px then smoothed %.3f" % (bconv["full"], bconv["red"], bconv["d"], bconv["red, masked 6-22 then smoothed"]))
    for i, row in enumerate(rows):
        log("row %d %-40s match %s  S/N %s" % (i + 2, row[0].replace("\n", " "),
            " ".join("%s %.2f" % (c, match[i][c]) for c in COLS if match[i][c] is not None),
            " ".join("%s %.1f" % (c, snr[i][c]) for c in COLS if snr[i][c] is not None)))

    # ---------------------------------------------------------- outputs
    side_by_side(os.path.join(args.out, "side_by_side.png"), pub, rows, disp, match, snr, stretch)
    identity_figure(os.path.join(args.out, "identity.png"), pub, ident, W54,
                    [("Fig. 1a source\ndisk image\n(2022-07-19)", disk_feb, disk_img, stretch["a"], 3.77, False),
                     ("Fig. 1b source\nASDI 3.8-4.14 µm\n(2022-02-11)", f38, feb54[M.band_select(W54, *BANDS["red"])].mean(0), stretch["b"], 3.97, True),
                     ("public script\nASDI 3.4-4.14 µm\n(2022-04-05)", fits.getdata(R("MWC758_sdiklip_xyl_3.4-4.14_ak15_sk2.fits")).astype(float).mean(0),
                      sdi54.mean(0), stretch["b"], 3.77, True)])
    hdus = [fits.PrimaryHDU()]
    for i, row in enumerate(rows):
        for c in COLS:
            if row[1][c] is not None:
                h = fits.ImageHDU(np.asarray(row[1][c][0], np.float32), name="R%d_%s" % (i + 2, c.upper()))
                h.header["ROW"] = row[0].replace("\n", " ")[:68]
                h.header["WHAT"] = row[1][c][2].replace("µ", "u")[:68]
                h.header["LAMBDA"] = (row[1][c][1], "um, for the lambda/D smoothing of the display")
                h.header["CSNR"] = (snr[i][c], "MWC 758c, Mawet S/N, lambda/D aperture")
                h.header["MATCH"] = (match[i][c], "rank corr. with the published panel, 7-21 px")
                hdus.append(h)
    hdus[0].header["SCRIPT"] = "scripts/compare_mwc758_paper_figure.py"
    with open(os.path.join(args.out, "products.fits"), "wb") as fh:
        fits.HDUList(hdus).writeto(fh)
    summary = {"rows": [r[0].replace("\n", " ") for r in rows], "match_published": match, "snr_c": snr,
               "stretch": {c: list(v) for c, v in stretch.items()}, "panel_b_vs_feb2022_bands": bfeb, "panel_b_vs_feb2022_conv_7_26px": bconv,
               "identity": ident, "sdi_refs_final": sdi_final["nref"].tolist(), "sdi_refs_feb2022_targets": sdi_feb["nref"].tolist(),
               "inputs": {"idl_input_md5": in_md5, "published_rgb_md5": hashlib.md5(np.ascontiguousarray(pub.rgb).tobytes()).hexdigest(),
                          "reduced": args.reduced,
                          "release": args.release},
               "script_md5": {f: md5(os.path.join(HERE, f)) for f in ("compare_mwc758_paper_figure.py", "mwc758_idl2022.py", "check_mwc758_paper_setup.py")},
               "seconds": time.time() - t00}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log("wrote %s (%.0f s)" % (args.out, time.time() - t00))


def side_by_side(path, pub, rows, disp, match, snr, stretch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle
    ink, muted, rule = "#1f1f1e", "#62615c", "#d4d3cf"
    cmap = pub.cmap()
    nrow = len(rows) + 1
    H = 2.42 * nrow + 1.25
    fig = plt.figure(figsize=(14.2, H), facecolor="white")
    gs = fig.add_gridspec(nrow, 6, width_ratios=[1.0, 1, 1, 1, 1, 1], left=0.008, right=0.995,
                          top=1 - 0.95 / H, bottom=0.42 / H, wspace=0.035, hspace=0.12)
    labels = ["Published Fig. 1\nNature Astron. 7, 1208"] + [r[0] for r in rows]
    for i in range(nrow):
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        axl.text(0.0, 0.5, "%d" % (i + 1), fontsize=13, color=muted, va="center", ha="left", transform=axl.transAxes)
        axl.text(0.12, 0.5, labels[i], fontsize=8.6, color=ink, va="center", ha="left", transform=axl.transAxes, linespacing=1.4)
        for j, c in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j + 1])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(rule)
                sp.set_linewidth(0.6)
            if c in "ab":
                ylim = (-0.5, 61.5)
            else:
                s, cx, cy = GEOM[c]
                ylim = (31 - (BOX[c][3] - cy) / s, 31 + (cy - BOX[c][1]) / s)
            if i == 0:
                crop, ext = pub.crop(c)
                ax.imshow(crop, extent=ext, interpolation="antialiased")
                ax.set_title(COL_TITLES[c], fontsize=8.8, color=ink, loc="left", pad=5)
            elif disp[i - 1][c] is None:
                ax.set_facecolor("#f4f4f2")
                ax.text(0.5, 0.5, "file gone\n(overwritten in 2023)", ha="center", va="center", fontsize=7.8, color=muted, transform=ax.transAxes)
            else:
                lo, hi = stretch[c][:2]
                ax.imshow(disp[i - 1][c], origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest",
                          extent=[-0.5, 61.5, -0.5, 61.5])
                ax.add_patch(Circle((31, 31), 6.35, facecolor=(7 / 255, 72 / 255, 202 / 255), edgecolor="white", lw=0.9, ls=(0, (3, 2)), zorder=3))
                ax.plot([31], [31], marker="+", color="white", ms=4, mew=0.7, zorder=4)
                if c != "a":
                    ax.add_patch(Circle(C_XY, 3.6, fill=False, ec="white", lw=0.8, zorder=4))
                txt = "%s\nc: S/N %.1f   match %.2f" % (rows[i - 1][1][c][2], snr[i - 1][c], match[i - 1][c])
                ax.text(0.025, 0.025, txt, fontsize=6.4, color="white", ha="left", va="bottom", transform=ax.transAxes, linespacing=1.25,
                        zorder=5, path_effects=[pe.withStroke(linewidth=1.6, foreground=(0.0, 0.03, 0.30))])
            ax.set_xlim(-0.5, 61.5)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_anchor("N")
    fig.text(0.008, 1 - 0.32 / H, "MWC 758, LBTI/ALES, 2019-11-14: the published Fig. 1 next to our reprocessing",
             fontsize=13, color=ink, ha="left", va="center")
    fig.text(0.008, 1 - 0.62 / H, "Rows 2 and 3 are the same reduction as the published panels; rows 4-6 are not.  "
             "Panel b shows the 3.8-4.14 µm half of the 2022-02-11 run, although the caption gives 3.4-4.14 µm.",
             fontsize=9, color=muted, ha="left", va="center")
    fig.text(0.008, 0.2 / H, "North up, east left; 62 x 62 px at 0.0345\"/px; d-f cropped as published.  Published colour map (from panel c's bar); "
             "one linear stretch per column, fitted to the published panel.  b, d-f smoothed by a unit-peak Gaussian of FWHM lambda/D, "
             "as published; a not.  Circle: c at 0.603\", PA 228.5.  S/N: Mawet et al. 2014, lambda/D aperture, unsmoothed, this night only "
             "(the paper's 5.7 combines all LBTI data).  match: rank correlation with the published panel at 7-21 px.",
             fontsize=6.9, color=muted, ha="left", va="center", wrap=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def identity_figure(path, pub, ident, W54, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, rule, surf = "#1f1f1e", "#62615c", "#d4d3cf", "#ffffff"
    s1, s2, s3 = "#2a78d6", "#eb6834", "#1baf7a"
    fig = plt.figure(figsize=(12.6, 5.6), facecolor=surf)
    gs = fig.add_gridspec(3, 6, width_ratios=[2.45, 0.06, 0.62, 0.86, 0.86, 0.86], left=0.062, right=0.99, top=0.86, bottom=0.19,
                          wspace=0.06, hspace=0.10)
    ax = fig.add_subplot(gs[:, 0])
    a, s = ident["ADI cube, 15 modes"], ident["ASDI cube, public script"]
    ax.plot(W54, 100 * np.array(a["rms_frac"]), color=s1, lw=1.4, marker="o", ms=3.2, label="ADI cube, 54 channels (2022-04-05)")
    ax.plot(W54, 100 * np.array(s["rms_frac"]), color=s2, lw=1.4, marker="o", ms=3.2, label="ASDI cube, 54 channels (2022-04-05)")
    for k, v in ident["ASDI bands of 2022-02-11"].items():
        lo, hi = [float(x) for x in k.split("-")]
        ax.plot([lo, hi], [100 * v["image"][1]] * 2, color=s3, lw=2.0, solid_capstyle="butt")
    ax.plot([], [], color=s3, lw=2.0, label="ASDI band images (2022-02-11)")
    d = ident["disk image (Fig. 1a)"]["image"][1]
    ax.axhline(100 * d, color=muted, lw=0.8, ls=(0, (3, 2)))
    ax.text(4.135, 100 * d * 1.3, "disk image, Fig. 1a: %.4f %%" % (100 * d), color=muted, fontsize=8, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlim(3.37, 4.16)
    ax.set_yticks([0.001, 0.01, 0.1, 1.0])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("wavelength (µm)", fontsize=9, color=ink)
    ax.set_ylabel("rms(port - paper) / rms(paper) at 6-22 px  (%)", fontsize=9, color=ink)
    ax.grid(True, which="major", color=rule, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(rule)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=ink)
    ax.set_title("Our port vs the paper's 2022 FITS, channel by channel", fontsize=10.5, color=ink, loc="left", pad=8)
    cmap = pub.cmap()
    for i, (name, ref, port, st, lam, sm) in enumerate(pairs):
        f = (lambda im: M.lod_smooth(im, lam)) if sm else (lambda im: im)
        lo, hi = st[:2]
        axl = fig.add_subplot(gs[i, 2])
        axl.axis("off")
        axl.text(0.95, 0.5, name, fontsize=8.2, color=ink, ha="right", va="center", transform=axl.transAxes, linespacing=1.3)
        for j, (im, t) in enumerate(((f(ref), "paper"), (f(port), "our port"), (f(port - ref) * 100.0, "difference x 100"))):
            axi = fig.add_subplot(gs[i, 3 + j])
            if j < 2:
                axi.imshow(im, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            else:
                h = (hi - lo) / 2.0
                axi.imshow(im, origin="lower", cmap="RdBu_r", vmin=-h, vmax=h, interpolation="nearest")
            axi.set_xticks([])
            axi.set_yticks([])
            axi.set_xlim(4, 57)
            axi.set_ylim(4, 57)
            for sp in axi.spines.values():
                sp.set_color(rule)
                sp.set_linewidth(0.6)
            if i == 0:
                axi.set_title(t, fontsize=9, color=ink, loc="left")
    fig.text(0.062, 0.065, "Same input: MWC758_cube_cen_clean.fits, frames sorted by angle and averaged 15 at a time.  What is left is at the "
             "level of IDL's single-precision arithmetic (e.g. SMOOTH's running sums), not a different step.", fontsize=7.4, color=muted, ha="left")
    fig.text(0.062, 0.03, "Images: the published colour map and stretch; b and the public-script row smoothed by lambda/D as in Fig. 1.  "
             "Differences: (port - paper) x 100 on a diverging map spanning half the column's range.", fontsize=7.4, color=muted, ha="left")
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


if __name__ == "__main__":
    main()
