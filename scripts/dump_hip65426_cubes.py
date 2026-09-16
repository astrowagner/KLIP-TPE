#!/usr/bin/env python
"""Write out the HIP 65426 cubes, as a file you can open.

    TQDM_DISABLE=1 python scripts/dump_hip65426_cubes.py [-o out.fits] [--params k=v ...]

There is normally no such file: `spaceklip.load_calints` reads the stage-2 `calints`,
repairs, registers and crops them in memory and hands the arrays straight to the reducer.
This writes that state to disk so it can be inspected, at the two levels that matter:

  SCI_ROLL1 / SCI_ROLL2 / REF   the REGISTERED, CROPPED cubes as the reducer receives them
                                -- `Dataset.cube` and `Dataset.ref_cube`.  This is the
                                alignment product: repaired, cross-correlation registered on
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
    a = ap.parse_args(argv)

    from astropy.io import fits

    d = os.path.join(datasets.data_dir(), "jwst_hip65426")
    files = sorted(glob.glob(os.path.join(d, "jw*calints.fits")))
    if not files:
        raise SystemExit(f"no calints in {d}")
    out = a.out or os.path.join(d, "hip65426_F444W_cubes.fits")

    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    dsets, info = sk.load_calints(files, science_target="HIP65426",
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
    hdr["CRPIXX"] = (info["crpix"][0], "0-based CRPIX - the APERTURE ref, 1.48 px off")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
