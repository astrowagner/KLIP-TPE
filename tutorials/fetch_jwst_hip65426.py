#!/usr/bin/env python3
"""Download the public JWST/NIRCam coronagraphy of HIP 65426 (ERS programme 1386,
Carter et al. 2023) used by tutorial 3: the F444W / MASK335R science exposures (two
rolls) and the reference star HIP 68245 (9-point small-grid dither) as stage-2
``calints`` products, plus the F300M set when ``--filters`` includes it.

    python3 -m pip install astroquery
    python3 tutorials/fetch_jwst_hip65426.py [--out ~/.klip_tpe/data/jwst_hip65426] [--filters F444W F300M]

Files (~10-60 MB each, 320x320 SUB320A335R) come from MAST; no login needed.  The
tutorial then reads them with ``klip_tpe.backends.spaceklip.load_spaceklip(sci_files=...,
ref_files=...)``.  spaceKLIP's ImageTools products (better centring / cleaning) can be
substituted one for one.
"""
import argparse
import os

from astroquery.mast import Observations

TARGETS = {"sci": "HIP-65426", "ref": "HIP-68245"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.klip_tpe/data/jwst_hip65426"))
    ap.add_argument("--filters", nargs="+", default=["F444W"])
    ap.add_argument("--product", default="calints", help="calints (stage 2, default) | rateints (stage 1)")
    ap.add_argument("--dry-run", action="store_true", help="list the files that would be downloaded and stop")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for role, target in TARGETS.items():
        obs = Observations.query_criteria(proposal_id="1386", instrument_name="NIRCAM/CORON", target_name=target,
                                          filters=a.filters)
        if len(obs) == 0:
            print(f"no observations for {target} {a.filters}")
            continue
        prods = Observations.get_product_list(obs)
        prods = Observations.filter_products(prods, productSubGroupDescription=a.product.upper(),
                                             extension="fits")
        names = [str(f) for f in prods["productFilename"]]
        keep = [("_" + a.product) in n and "_ta" not in n and "tacq" not in n and n.endswith(".fits")
                for n in names]
        prods = prods[keep]
        print(f"{role}: {target} -> {len(prods)} {a.product} files")
        for n in sorted(str(f) for f in prods["productFilename"]):
            print("    ", n)
        if a.dry_run:
            continue
        # NB: older astroquery has no flat= keyword (unknown keywords are treated as product
        # filters and raise), so download into mastDownload/... and flatten afterwards
        man = Observations.download_products(prods, download_dir=a.out)
        for row in man:
            src = str(row["Local Path"])
            if row["Status"] != "COMPLETE" or not os.path.exists(src):
                print("  !", src, row["Status"], row.get("Message", ""))
                continue
            dst = os.path.join(a.out, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)
            print("  ", dst)
    n = len([f for f in os.listdir(a.out) if f.endswith(".fits")]) if os.path.isdir(a.out) else 0
    print(f"done -> {a.out}  ({n} FITS files)")


if __name__ == "__main__":
    main()
