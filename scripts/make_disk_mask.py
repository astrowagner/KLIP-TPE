#!/usr/bin/env python3
"""make_disk_mask.py -- a disk mask from an image of the disk, for a run script's ``--disk-mask``: the pixels
:func:`klip_tpe.metrics.disk_mask_from_image` finds bright and extended (spiral arms, a ring), which the search
then keeps out of its noise estimate and its injections (``default_config(disk_mask=...)``).

  python3 scripts/make_disk_mask.py IMAGE.fits --fwhm PX --out MASK.fits [--nsig 2] [--r-range 6 26]
          [--min-beams 2] [--grow-fwhm 1] [--smooth-fwhm 1] [--neg-nsig S [--neg-reach-fwhm 2]]
          [--protect RHO,PA ...] [--pxscale AS] [--band RHO ...] [--preview PNG] [--ext N]

IMAGE is a disk-preserving reduction with the star at the frame centre, north up and east left as the
reducers' images are (a 2-D FITS; ``--ext`` picks the HDU; a cube is collapsed by its median).  ``--fwhm`` is
the PSF FWHM in pixels, in which the smoothing, the piece size and the growth are counted.  ``--neg-nsig S``
adds the disk's ADI negatives: the troughs S robust sigma deep that come within ``--neg-reach-fwhm`` FWHM of the
bright structure (the preview draws the bright-only mask dashed for comparison).  ``--protect RHO,PA``
keeps a companion's position (arcsec, PA in deg) unmasked and needs ``--pxscale``; ``--band RHO`` draws the
injections' noise band at RHO arcsec (+- 0.6 FWHM, the metric's radial band) on the preview and reports how
much of it the mask covers.  Writes MASK (uint8, 1 = disk) with the parameters, the image's name and its SHA-1
in the header, and the preview (default: MASK with .png).

Build it from an image of the disk made before the search -- a disk-preserving reduction, another epoch or
instrument -- never from a search's own winner, and look at the preview: it masks whatever is bright, and a
bright speckle group masked with the disk biases the noise estimate down."""
import argparse
import hashlib
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from klip_tpe.metrics import disk_mask_from_image, source_xy, star_center   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("image")
ap.add_argument("--fwhm", type=float, required=True, help="PSF FWHM, px")
ap.add_argument("--out", required=True)
ap.add_argument("--nsig", type=float, default=2.0)
ap.add_argument("--r-range", type=float, nargs=2, default=None, metavar=("R0", "R1"), help="px from the star")
ap.add_argument("--min-beams", type=float, default=2.0)
ap.add_argument("--grow-fwhm", type=float, default=1.0)
ap.add_argument("--smooth-fwhm", type=float, default=1.0)
ap.add_argument("--neg-nsig", type=float, default=None, help="also mask the disk's ADI negatives this deep (sigma)")
ap.add_argument("--neg-reach-fwhm", type=float, default=2.0, help="a trough counts within this many FWHM of the disk")
ap.add_argument("--protect", nargs="*", default=[], metavar="RHO,PA")
ap.add_argument("--pxscale", type=float, default=None, help="arcsec per px (for --protect and --band)")
ap.add_argument("--band", type=float, nargs="*", default=[], metavar="RHO")
ap.add_argument("--preview", default=None)
ap.add_argument("--ext", type=int, default=0)
a = ap.parse_args()
if (a.protect or a.band) and a.pxscale is None:
    ap.error("--protect and --band need --pxscale")
protect = [tuple(float(v) for v in p.split(",")) for p in a.protect]

img = np.asarray(fits.getdata(a.image, ext=a.ext), float)
img = np.squeeze(img)
if img.ndim == 3:
    img = np.nanmedian(img, axis=0)
if img.ndim != 2:
    sys.exit(f"{a.image}: want a 2-D image, got shape {img.shape}")
kw = dict(nsig=a.nsig, r_range=tuple(a.r_range) if a.r_range else None, min_beams=a.min_beams,
          grow_fwhm=a.grow_fwhm, smooth_fwhm=a.smooth_fwhm, protect=protect, pxscale=a.pxscale)
mask = disk_mask_from_image(img, a.fwhm, nsig_neg=a.neg_nsig, neg_reach_fwhm=a.neg_reach_fwhm, **kw)
bright = disk_mask_from_image(img, a.fwhm, **kw) if a.neg_nsig is not None else mask
ny, nx = img.shape
cx, cy = star_center(img.shape)
yy, xx = np.mgrid[0:ny, 0:nx]
rr = np.hypot(xx - cx, yy - cy)
print(f"{os.path.basename(a.image)} {img.shape}: {int(mask.sum())} px masked"
      + (f" ({int(bright.sum())} for the bright structure alone)" if a.neg_nsig is not None else ""))
bands = []
for rho in a.band:
    b = np.abs(rr - rho / a.pxscale) <= 0.6 * a.fwhm
    bands.append((rho, b))
    print(f"  noise band at {rho:.3f}\" (+- 0.6 FWHM): {100 * (mask & b).sum() / b.sum():.1f}% masked"
          + (f" ({100 * (bright & b).sum() / b.sum():.1f}% bright alone)" if a.neg_nsig is not None else ""))

sha = hashlib.sha1(open(a.image, "rb").read()).hexdigest()
hdr = fits.Header()
hdr["SRCIMG"] = (os.path.basename(a.image)[:60], "image the mask was built from")
hdr["SRCSHA1"] = (sha, "SHA-1 of that file")
hdr["MKFWHM"] = (float(a.fwhm), "PSF FWHM, px")
hdr["MKNSIG"] = (float(a.nsig), "threshold, robust sigma of the smoothed image")
hdr["MKSMOOTH"] = (float(a.smooth_fwhm), "smoothing Gaussian, FWHM")
hdr["MKBEAMS"] = (float(a.min_beams), "smallest piece kept, beams")
hdr["MKGROW"] = (float(a.grow_fwhm), "growth, FWHM")
if a.neg_nsig is not None:
    hdr["MKNEGSIG"] = (float(a.neg_nsig), "ADI negatives: depth, robust sigma")
    hdr["MKNEGRCH"] = (float(a.neg_reach_fwhm), "ADI negatives: reach from the disk, FWHM")
if a.r_range:
    hdr["MKRMIN"], hdr["MKRMAX"] = float(a.r_range[0]), float(a.r_range[1])
if a.pxscale:
    hdr["MKPXSCL"] = (float(a.pxscale), "arcsec per px")
for i, p in enumerate(protect):
    hdr[f"MKPROT{i}"] = (f"{p[0]:g},{p[1]:g}", "protected position: arcsec, PA deg")
hdr["NMASKED"] = int(mask.sum())
hdr.add_history("klip-tpe scripts/make_disk_mask.py: 1 = disk (kept out of the noise and the injections)")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fits.writeto(a.out, mask.astype(np.uint8), hdr, overwrite=True)
print("wrote", a.out)

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
fin = np.isfinite(img)
sg = a.fwhm / 2.3548
num = ndimage.gaussian_filter(np.where(fin, img, 0.0), sg)
den = ndimage.gaussian_filter(fin.astype(float), sg)
sm = np.where(fin & (den > 0.2), num / np.where(den > 0, den, 1.0), np.nan)
inr = np.isfinite(sm)
if a.r_range:
    inr &= (rr >= a.r_range[0]) & (rr <= a.r_range[1])
v = sm[inr]
s = 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else 1.0
fig, ax = plt.subplots(figsize=(5.6, 5.8))
ax.imshow(np.where(inr, sm, np.nan), origin="lower", cmap="inferno", vmin=np.median(v) - 2 * s, vmax=np.median(v) + 6 * s)
if mask.any():
    ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.1)
if a.neg_nsig is not None and bright.any():
    ax.contour(bright.astype(float), levels=[0.5], colors="cyan", linewidths=0.7, linestyles="--")
for rho, b in bands:
    ax.contour(b.astype(float), levels=[0.5], colors="w", linewidths=0.8, linestyles=":")
for p in protect:
    px, py = source_xy([p[0]], [p[1]], a.pxscale, cx, cy)
    ax.plot(px, py, "+", color="lime", ms=12, mew=1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"{os.path.basename(a.image)}\n{int(mask.sum())} px masked (cyan): > {a.nsig:g} sigma"
             + (f" and its ADI negatives < -{a.neg_nsig:g} sigma (bright alone dashed)" if a.neg_nsig is not None else "")
             + f",\npieces >= {a.min_beams:g} beams, grown {a.grow_fwhm:g} FWHM", fontsize=8)
fig.tight_layout()
png = a.preview or os.path.splitext(a.out)[0] + ".png"
fig.savefig(png, dpi=130, facecolor="white")
print("wrote", png)
