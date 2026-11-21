#!/usr/bin/env python
"""Fit beta Pic b with the extraction step and compare it with its published position and brightness.

    python scripts/check_betapic_extract.py            # the step's defaults, about 6,000 reductions
    python scripts/check_betapic_extract.py --quick    # 300 trials and 3 fakes of 200, for a first look
    python scripts/check_betapic_extract.py --quick --display betapic_extract   # with the live dashboard

The VLT/NACO L' sequence of beta Pictoris that VIP distributes is public, and Absil et al. (2013,
A&A 559, L12) measured beta Pic b in these same data.  The script fits it with
:func:`klip_tpe.extract.fit_companion` in an ADI configuration, the default of the generic search
space over the annulus 8-26 px, and checks the fit with :func:`klip_tpe.extract.recovery_check`.
The star's flux is VIP's published ``starphot`` (``datasets.PHOTOMETRY``), as
``scripts/check_betapic_contrast.py`` takes it.

It prints the fit, both sets of 1-sigma uncertainties, the recovery check and the offsets from the
published values in units of the combined uncertainty.  The data download once (``klip-tpe fetch``).
``--display DIR`` draws the step's live dashboard into ``DIR/steps``
(:class:`klip_tpe.extract_display.ExtractDisplay`), with the published position as the reference.

Expected with the defaults (2026-11-21): rho 0.4516", pa 210.64 deg, dL' 7.86.  The quadratic gives
0.0007", 0.046 deg and 0.9 %, the profiles 0.0002", 0.013 deg and 0.2 %, and 5 fakes scatter by
0.0023", 0.15 deg and 4.3 %, with biases of -0.0001 +- 0.0010", -0.13 +- 0.07 deg and +3.0 +- 1.9 %.
The offsets from Absil et al. are -0.03, -0.42 and -0.89 sigma.
"""
from __future__ import annotations

import argparse
import math
import sys

from klip_tpe import datasets
from klip_tpe.extract import ExtractConfig, fit_companion, recovery_check
from klip_tpe.instruments import generic
from klip_tpe.metrics import Source
from klip_tpe.reducer import ReductionRequest

#: Absil et al. 2013: separation (arcsec), PA (deg) and L' magnitude difference, with 1-sigma errors
PUB = {"rho": (0.452, 0.010), "pa": (211.2, 1.3), "dmag": (8.01, 0.16)}
INRAD, OUTRAD = 8, 26                # the annulus that brackets 0.452" = 16.6 px


def main(argv=None) -> int:
    """Fit beta Pic b, print the comparison and return 0 when every offset is within 3 sigma.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command line, None for ``sys.argv``.

    Returns
    -------
    status : int
        0 when the separation, angle and contrast all lie within 3 combined sigma of the published
        values, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="300 trials and 3 fakes of 200 trials each")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--display", default=None, metavar="DIR", help="draw the live dashboard into DIR/steps")
    ap.add_argument("--display-every", type=int, default=1, metavar="N", help="trials between two panels")
    a = ap.parse_args(argv)
    files = datasets.fetch("naco_betapic", quiet=True)
    inst = datasets.INSTRUMENT["naco_betapic"]
    phot = datasets.PHOTOMETRY["naco_betapic"]
    ds = generic.load_cube(files["cube"], files["angles"], psf=files["psf"], name="betapic")
    sf = generic.star_flux_from_aperture_photometry(ds.meta["psf"], phot["starphot"], phot["aperture_px"])
    red = generic.make_reducer({"betapic": ds}, star_flux=sf, max_workers="auto", log=lambda s: None, **inst)
    space = generic.make_space(red, k_klip_max=30)
    p0 = dict(space.decode(space.default_vector()).params, inrad=INRAD, outrad=OUTRAD)
    print(f"configuration: {p0}")

    def reduce_fn(srcs):
        return red.reduce(ReductionRequest(dict(p0), list(srcs) if srcs else None)).image

    cfg = (ExtractConfig(n_iter=300, n_init=50, n_fakes=3, fake_iter=200, seed=a.seed) if a.quick
           else ExtractConfig(seed=a.seed))
    c0 = 10.0 ** (-0.4 * 7.9)
    disp = None
    if a.display:
        from klip_tpe.extract_display import ExtractDisplay
        from klip_tpe.injection import injected_model
        disp = ExtractDisplay(a.display, label="beta Pic b", every=a.display_every,
                              reference=(PUB["rho"][0], PUB["pa"][0]), log=print,
                              psf_fn=lambda rho, pa, shape: injected_model(red, [Source(rho, pa, 1.0)], shape))
    fit = fit_companion(reduce_fn, 0.45, 212.0, c0, fwhm_px=red.fwhm, pxscale=red.pxscale,
                        angle_convention=red.angle_convention, config=cfg, log=print, progress=disp)
    rec = recovery_check(reduce_fn, fit, fwhm_px=red.fwhm, pxscale=red.pxscale,
                         angle_convention=red.angle_convention, config=cfg, log=print, progress=disp)
    if disp is not None:
        disp.close()
    b = fit["best"]
    s1 = fit["quadratic"]["sigma"]
    s2 = [p["sigma"] for p in fit["profile_fits"]]
    dmag = -2.5 * math.log10(b["contrast"])
    print(f"\nbeta Pic b: rho {b['rho']:.4f}\"  pa {b['pa']:.3f} deg  contrast {b['contrast']:.4e} "
          f"(dL' {dmag:.3f})  residual {b['residual']:.3f}  after {fit['n_reductions']} reductions")
    print(f"  sigma #1 quadratic: rho {s1[0]:.4f}\"  pa {s1[1]:.3f} deg  contrast {100 * s1[2] / b['contrast']:.1f} %")
    print(f"  sigma #2 profiles : rho {s2[0]:.4f}\"  pa {s2[1]:.3f} deg  contrast {100 * s2[2] / b['contrast']:.1f} %")
    bi, se, sc = rec["bias"], rec["bias_se"], rec["scatter"]
    print(f"  recovery ({len(rec['fakes'])} fakes): bias rho {bi['rho']:+.4f} +- {se['rho']:.4f}\"  pa "
          f"{bi['pa']:+.3f} +- {se['pa']:.3f} deg  contrast {100 * bi['contrast']:+.1f} +- "
          f"{100 * se['contrast']:.1f} %, scatter rho {sc['rho']:.4f}\"  pa {sc['pa']:.3f} deg  contrast "
          f"{100 * sc['contrast']:.1f} %")
    # the fit's own error: the larger of the profile sigma and the fakes' scatter
    err = {"rho": max(s2[0], sc["rho"]), "pa": max(s2[1], sc["pa"]),
           "dmag": 2.5 / math.log(10) * max(s2[2] / b["contrast"], sc["contrast"])}
    got = {"rho": b["rho"], "pa": b["pa"], "dmag": dmag}
    ok = True
    print("\nagainst Absil et al. 2013 (these same data):")
    for k in ("rho", "pa", "dmag"):
        v, e = PUB[k]
        z = (got[k] - v) / math.hypot(e, err[k])
        ok &= abs(z) < 3.0
        print(f"  {k:>4s}: {got[k]:.4f} +- {err[k]:.4f}  published {v} +- {e}  ->  {z:+.2f} sigma")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
