#!/usr/bin/env python
"""Write out the HIP 65426 cubes, as a file you can open.

    TQDM_DISABLE=1 python scripts/dump_hip65426_cubes.py [-o out.fits] [--params k=v ...]

There is normally no such file: `spaceklip.load_calints` reads the stage-2 `calints`,
fills the DQ pixels, registers and crops them in memory and hands the arrays to the reducer.
This writes that state to disk so it can be inspected, at the two levels that matter:

  SCI_ROLL1 / SCI_ROLL2 / REF   the REGISTERED, CROPPED cubes as the reducer receives them
                                -- `Dataset.cube` and `Dataset.ref_cube`.  This is the
                                alignment product: DQ pixels filled, cross-correlation registered on
                                the median science frame, and cropped about the star (NOT
                                about CRPIX -- see datasets.PHOTOMETRY).
  KLIP_ROLL1 / KLIP_ROLL2       what `pyklip.parallelized.klip_parallelized` is actually
                                handed for one parameter set, captured from the call itself
                                rather than reconstructed: the same frames after the
                                searched temporal binning, high-pass filter and frame
                                selection.  These differ per evaluation, because those are
                                search parameters; the header records which ones produced
                                this file.
  PSFLIB_ROLL1 / PSFLIB_ROLL2   the RDI library pyKLIP sees for each roll (reference frames
                                high-passed the same way, science frames appended; pyKLIP
                                excludes each science frame from its own library).

Angles are in the PA_ROLL1 / PA_ROLL2 / PA_REF extensions, in klip-tpe's convention
(counter-clockwise derotation angle, `ROLL_REF - V3I_YANG * VPARITY`).
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
from klip_tpe.instruments import generic
from klip_tpe.reducer import ReductionRequest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None,
                    help="output FITS (default: <data>/jwst_hip65426/hip65426_F444W_cubes.fits)")
    ap.add_argument("--params", nargs="*", default=[], metavar="K=V",
                    help="override the default reduction parameters, e.g. filter=5 k_klip=18")
    ap.add_argument("--frames", action="store_true",
                    help="also write the INDIVIDUAL frames and their provenance, as "
                         "<base>_frames_sci.fits and <base>_frames_ref.fits")
    ap.add_argument("--png", default=None, help="montage of the individual science frames")
    a = ap.parse_args(argv)

    from astropy.io import fits

    d = os.path.join(datasets.data_dir(), "jwst_hip65426")
    files = sorted(glob.glob(os.path.join(d, "jw*calints.fits")))
    if not files:
        raise SystemExit(f"no calints in {d}")
    out = a.out or os.path.join(d, "hip65426_F444W_cubes.fits")

    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    dsets, info = sk.load_calints(files, science_target="HIP65426", keep_frames=a.frames,
                                  star_center=tuple(phot["star_center"]), log=print)
    grid = stpsf_psf.offaxis_grid("NIRCam", info["filter"] or "F444W", image_mask="MASK335R",
                                  seps_as=np.arange(0.2, 3.01, 0.2), stamp_px=41, nlambda=3,
                                  log=lambda s: None)
    sf = stpsf_psf.star_flux_from_flux_density(
        grid, phot["flux_density_jy"], info["pixar_sr"],
        optics_transmission=phot["optics_transmission"], log=lambda s: None)
    model = stpsf_psf.library(grid, star_flux=sf)

    hdr = fits.Header()
    hdr["ORIGIN"] = "klip_tpe scripts/dump_hip65426_cubes.py"
    hdr["BUNIT"] = (info["bunit"], "unchanged from the calints")
    hdr["PIXSCALE"] = (info["pxscale"], "arcsec/px, from PIXAR_A2")
    hdr["PIXAR_SR"] = (info["pixar_sr"], "sr/px, for the flux scale")
    hdr["FILTER"] = info["filter"]
    hdr["STARCENX"] = (info["star_center"][0], "0-based star x used for the crop")
    hdr["STARCENY"] = (info["star_center"][1], "0-based star y used for the crop")
    hdr["CRPIXX"] = (info["crpix"][0], "0-based CRPIX - the APERTURE ref, 0.78 px off")
    hdr["CRPIXY"] = (info["crpix"][1], "")
    hdr["CROPPX"] = (info["crop_px"], "odd on purpose: star on the centre pixel")
    hdr["NSCI"] = info["n_sci"]
    hdr["NREF"] = info["n_ref"]
    hdr["STARFLUX"] = (float(sf), "flux_unit of the injection model")
    prim = fits.PrimaryHDU(header=hdr)
    hdus = [prim]

    # ---- level 1: what the reducer receives ------------------------------------------
    for name, ds in dsets.items():
        hdus.append(fits.ImageHDU(np.asarray(ds.cube, np.float32), name=f"SCI_{name.upper()}"))
        hdus.append(fits.ImageHDU(np.asarray(ds.angles, float), name=f"PA_{name.upper()}"))
    ref = next(iter(dsets.values())).ref_cube
    if ref is not None:
        hdus.append(fits.ImageHDU(np.asarray(ref, np.float32), name="REF"))

    # ---- level 2: what pyKLIP is actually handed -------------------------------------
    import pyklip.parallelized as par
    grabbed = {}
    real = par.klip_parallelized

    def spy(imgs, centers, parangs, *rest, **kw):
        """Record what pyKLIP is handed, then let it run as usual."""
        grabbed["cube"] = np.array(imgs, np.float32)
        grabbed["centers"] = np.array(centers, float)
        grabbed["parangs"] = np.array(parangs, float)
        if kw.get("psf_library") is not None:
            grabbed["psflib"] = np.array(kw["psf_library"], np.float32)
        return real(imgs, centers, parangs, *rest, **kw)

    over = {}
    for kv in a.params:
        k, _, v = kv.partition("=")
        try:
            over[k] = int(v)
        except ValueError:
            try:
                over[k] = float(v)
            except ValueError:
                over[k] = v

    par.klip_parallelized = spy
    try:
        for name, ds in dsets.items():
            red = sk.make_reducer({name: ds}, injection_model=model, mode="RDI",
                                  max_workers=1, log=lambda s: None)
            space = generic.make_space(red, k_klip_max=18, search_angles=False)
            p0 = dict(space.decode(space.default_vector()).params,
                      inrad=6, outrad=30, k_klip=10, **over)
            grabbed.clear()
            red.reduce(ReductionRequest(params=p0))
            if "cube" not in grabbed:
                print(f"  {name}: klip_parallelized was not called -- nothing captured")
                continue
            hdus.append(fits.ImageHDU(grabbed["cube"], name=f"KLIP_{name.upper()}"))
            if "psflib" in grabbed:
                hdus.append(fits.ImageHDU(grabbed["psflib"], name=f"PSFLIB_{name.upper()}"))
            for k in ("bin", "filter", "n_ang", "angsep", "anglemax", "k_klip", "mode",
                      "inrad", "outrad"):
                if k in p0:
                    prim.header[f"P_{k[:6].upper()}"] = (p0[k], f"reduction parameter {k}")
            print(f"  {name}: klip cube {grabbed['cube'].shape} from dataset cube "
                  f"{tuple(ds.cube.shape)}")
    finally:
        par.klip_parallelized = real

    fits.HDUList(hdus).writeto(out, overwrite=True)
    print(f"\nwrote {out}")
    for h in fits.open(out):
        if h.data is not None:
            print(f"   {h.name:<14} {np.shape(h.data)}")

    if a.frames:
        _write_frames(out, info, hdr, fits)
    if a.png:
        _montage(info, a.png)
    return 0


def _frame_table(rows, fits):
    """The per-frame provenance, as a FITS binary table."""
    def col(name, key, fmt, unit=None, default=0.0):
        vals = [r.get(key, default) for r in rows]
        if fmt.endswith("A"):
            vals = [str(v) for v in vals]
        return fits.Column(name=name, format=fmt, unit=unit, array=np.array(vals))
    w = max(len(str(r["file"])) for r in rows)
    return fits.BinTableHDU.from_columns([
        col("FILE", "file", f"{w}A"),
        col("INTEG", "integration", "J"),
        col("ROLE", "role", "4A"),
        col("TARGET", "target", "16A"),
        col("PA", "pa", "D", "deg"),
        col("N_DQ", "n_dq", "J"),
        col("N_REPAIR", "n_repaired", "J"),
        col("DX", "dx", "D", "pix"),
        col("DY", "dy", "D", "pix"),
        col("XOFFSET", "xoffset", "D", "arcsec"),
        col("YOFFSET", "yoffset", "D", "arcsec"),
    ], name="FRAMES")


def _write_frames(out, info, hdr, fits):
    """One file per role, with every individual frame at each processing stage.

    RAW is the archive frame with the DQ DO_NOT_USE pixels set to NaN and nothing else;
    ALIGNED is after the DQ fill and the cross-correlation shift, still full frame;
    CROP is the individual frame the reducer actually receives.  The FRAMES table says
    which archive file and integration each slice came from, the shift that was applied to
    it, and how many pixels the DQ flagged and the fill rewrote (equal, by construction).
    """
    base = out[:-5] if out.endswith(".fits") else out
    rows = info["frames"]
    for role, raw, aligned, crop in (("sci", info["raw_sci"], info["aligned_sci"], info["crop_sci"]),
                                     ("ref", info["raw_ref"], info["aligned_ref"], info["crop_ref"])):
        sel = [r for r in rows if r["role"].lower() == role]
        if not len(raw):
            continue
        path = f"{base}_frames_{role}.fits"
        h = fits.Header(hdr, copy=True)
        h["ROLE"] = (role.upper(), "SCI = science rolls, REF = RDI library")
        hl = fits.HDUList([fits.PrimaryHDU(header=h), _frame_table(sel, fits),
                           fits.ImageHDU(np.asarray(raw, np.float32), name="RAW"),
                           fits.ImageHDU(np.asarray(aligned, np.float32), name="ALIGNED"),
                           fits.ImageHDU(np.asarray(crop, np.float32), name="CROP")])
        hl.writeto(path, overwrite=True)
        print(f"\nwrote {path}")
        for x in fits.open(path):
            if x.data is not None:
                print(f"   {x.name:<10} {np.shape(x.data)}")
        print(f"   {'idx':>3} {'file':<34} {'int':>3} {'PA':>8} {'n_dq':>6} {'n_rep':>6} "
              f"{'dx':>6} {'dy':>6}")
        for i, r in enumerate(sel):
            print(f"   {i:>3} {r['file']:<34} {r['integration']:>3} {r['pa']:>8.3f} "
                  f"{r.get('n_dq', 0):>6} {r.get('n_repaired', 0):>6} "
                  f"{r.get('dx', 0):>6.2f} {r.get('dy', 0):>6.2f}")


def _montage(info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in info["frames"] if r["role"] == "SCI"]
    raw, crop = info["raw_sci"], info["crop_sci"]
    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(2.5 * n, 5.4))
    ax = np.atleast_2d(ax)
    for j, r in enumerate(rows):
        for i, (im, t) in enumerate(((raw[j], "raw (DQ masked)"), (crop[j], "cropped, aligned"))):
            v = np.nanpercentile(im[np.isfinite(im)], [5, 99.5])
            ax[i, j].imshow(im, origin="lower", cmap="inferno", vmin=v[0], vmax=v[1])
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f"{t}\n{r['file'][:18]} int {r['integration']}\n"
                               f"PA {r['pa']:.2f}, dx {r.get('dx', 0):+.2f} dy {r.get('dy', 0):+.2f}",
                               fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
