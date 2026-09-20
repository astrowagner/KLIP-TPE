#!/usr/bin/env python3
"""Fetch the JWST coronagraphy this AR pilot needs, and count the archive for the proposal.

MAST is blocked from the Claude session (403 at the egress proxy) and from the Linux VM on
the Mac, so this has to run in your own Terminal.  Nothing here is Claude-specific: it is
astroquery.mast with the queries written down.

    pip install astroquery                       # if needed
    python3 fetch_jwst_ar.py --count             # just the archive census (fast, no download)
    python3 fetch_jwst_ar.py --targets rxj0534 aflep
    python3 fetch_jwst_ar.py --miri              # list public MIRI coronagraphy, pick one
    python3 fetch_jwst_ar.py --targets rxj0534 --download

Downloads land in  <outdir>/<name>/  (default ~/Data/JWST), calints + the association
files spaceKLIP wants.  Nothing is deleted or overwritten: existing files are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------- what to fetch
# exp_type is the reliable discriminator for coronagraphy; filters/masks vary by program.
CORON = {"nircam": ["NRC_CORON"], "miri": ["MIR_4QPM", "MIR_LYOT"]}

TARGETS = {
    # RX J0534.0-0221 -- Ferrer-Chavez, Wang, Wagner, Lawson et al. 2026 (arXiv:2609.20748),
    # "The JWST Sub-Jupiters Survey".  NIRCam F444W + F200W; the planet is at 0.41", 2.8 MJup,
    # with a debris disk at 79 au.  EV Lac (jw06122009) and Wolf 359 (jw06122001) are already
    # on disk from the same program, so the program ID is the safe way in if the name resolver
    # disagrees about this one.
    "rxj0534": dict(target_name="RX*J0534.0-0221", proposal_id="6122", instrument="nircam"),
    # AF Lep b -- confirm which program before downloading; the census below will show them.
    "aflep":   dict(target_name="AF*LEP", instrument="nircam"),
    # already on disk, here for completeness / re-fetch
    "mwc758":  dict(proposal_id="4014", instrument="nircam"),
}


def _obs():
    from astroquery.mast import Observations
    return Observations


def census(inst=("nircam", "miri"), public_only=True):
    """The two numbers the AR attachment's \\nProgs / \\nDatasets placeholders want."""
    Observations = _obs()
    progs, datasets = set(), 0
    for i in inst:
        for et in CORON[i]:
            kw = dict(obs_collection="JWST", instrument_name="*", dataproduct_type="image")
            try:
                t = Observations.query_criteria(obs_collection="JWST", exp_type=et, **{})
            except Exception as exc:
                print(f"  {et}: query failed ({exc})", file=sys.stderr)
                continue
            if public_only and "dataRights" in t.colnames:
                t = t[[str(v).upper() == "PUBLIC" for v in t["dataRights"]]]
            ids = {str(v) for v in t["proposal_id"]} if "proposal_id" in t.colnames else set()
            progs |= ids
            datasets += len(t)
            print(f"  {i:7s} {et:10s}  {len(t):5d} observations  {len(ids):3d} programs")
    print(f"\nTOTAL public coronagraphy: {datasets} observations across {len(progs)} programs")
    print(f"  \\newcommand{{\\nProgs}}{{{len(progs)}}}")
    print(f"  \\newcommand{{\\nDatasets}}{{{datasets}}}")
    return sorted(progs), datasets


def miri_list(public_only=True, limit=40):
    """Public MIRI coronagraphy, brightest-program-first, to pick a pilot target from."""
    Observations = _obs()
    rows = []
    for et in CORON["miri"]:
        try:
            t = Observations.query_criteria(obs_collection="JWST", exp_type=et)
        except Exception as exc:
            print(f"  {et}: {exc}", file=sys.stderr)
            continue
        if public_only and "dataRights" in t.colnames:
            t = t[[str(v).upper() == "PUBLIC" for v in t["dataRights"]]]
        for r in t:
            rows.append((str(r["proposal_id"]), str(r["target_name"]), str(r["filters"]), et))
    seen, out = set(), []
    for r in sorted(rows):
        key = r[:3]
        if key not in seen:
            seen.add(key)
            out.append(r)
    print(f"{'program':>8}  {'target':28s} {'filter':10s} exp_type")
    for p, tgt, filt, et in out[:limit]:
        print(f"{p:>8}  {tgt[:28]:28s} {filt[:10]:10s} {et}")
    print(f"\n{len(out)} distinct (program, target, filter) combinations; showing {min(limit, len(out))}")
    return out


def fetch(name, outdir, download=False, products=("CALINTS", "ASN")):
    Observations = _obs()
    spec = dict(TARGETS[name])
    inst = spec.pop("instrument", "nircam")
    crit = dict(obs_collection="JWST", **spec)
    print(f"\n=== {name}: {crit}")
    hits = []
    for et in CORON[inst]:
        try:
            t = Observations.query_criteria(exp_type=et, **crit)
        except Exception as exc:
            print(f"  {et}: {exc}", file=sys.stderr)
            continue
        if len(t):
            hits.append(t)
            for r in t:
                print(f"  {r['proposal_id']:>6}  {str(r['target_name'])[:26]:26s} "
                      f"{str(r['filters'])[:12]:12s} {et}  {r['obsid']}")
    if not hits:
        print("  no matches -- check the target name against the census output")
        return
    if not download:
        print("  (--download to pull the products)")
        return
    from astropy.table import vstack
    obs = vstack(hits)
    prod = Observations.get_product_list(obs)
    keep = Observations.filter_products(prod, productSubGroupDescription=list(products))
    dest = os.path.join(outdir, name)
    os.makedirs(dest, exist_ok=True)
    print(f"  downloading {len(keep)} products -> {dest}")
    Observations.download_products(keep, download_dir=dest, cache=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", action="store_true", help="archive census for the proposal")
    ap.add_argument("--miri", action="store_true", help="list public MIRI coronagraphy")
    ap.add_argument("--targets", nargs="*", default=[], choices=sorted(TARGETS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--outdir", default=os.path.expanduser("~/Data/JWST"))
    a = ap.parse_args()
    if a.count:
        print("Public JWST coronagraphy in MAST:")
        census()
    if a.miri:
        print("\nPublic MIRI coronagraphy:")
        miri_list()
    for name in a.targets:
        fetch(name, a.outdir, a.download)
    if not (a.count or a.miri or a.targets):
        ap.print_help()


if __name__ == "__main__":
    main()
