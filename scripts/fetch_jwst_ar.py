#!/usr/bin/env python3
"""Fetch the JWST coronagraphy this AR pilot needs, and count the archive for the proposal.

MAST is blocked from the Claude session (403 at the egress proxy) and from the Linux VM on
the Mac, so this has to run in your own Terminal.  Nothing here is Claude-specific: it is
astroquery.mast with the queries written down.

    pip install astroquery                       # if needed
    python3 fetch_jwst_ar.py --count             # archive census (one query, no download)
    python3 fetch_jwst_ar.py --miri              # public MIRI coronagraphy, pick a pilot
    python3 fetch_jwst_ar.py --find "AF Lep"     # which program holds a target
    python3 fetch_jwst_ar.py --targets rxj0534 --download

Downloads land in  <outdir>/<name>/  (default ~/Data/JWST), calints + the association
files spaceKLIP wants.  Nothing is deleted or overwritten: existing files are skipped.

WHY THE CRITERIA LOOK THE WAY THEY DO
-------------------------------------
``Observations.query_criteria`` accepts only the CAOM columns that
``Observations.get_metadata("observations")`` lists, and ``exp_type`` is not one of them.
An earlier version of this script filtered on ``exp_type="NRC_CORON"``; astroquery does
not raise on an unknown column, it emits

    InputWarning: Filter exp_type does not exist. This filter will be skipped.

and returns the query *without* that constraint -- which for ``obs_collection="JWST"``
is the entire public archive.  That is how a coronagraphy census came back as 338,230
observations in 1,943 programs, and the same number three times over for three
"different" exposure types.  Coronagraphy lives in ``instrument_name`` instead
("NIRCAM/CORON", "MIRI/CORON"), and :func:`_check_fields` now refuses to run a query
whose columns MAST does not recognise, so a skipped filter can never again be read as
an answer.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

# CAOM instrument_name is the reliable discriminator for coronagraphy.  Wildcarded so a
# spelling change on MAST's side (or a mode suffix) does not silently return nothing --
# and --count prints the distinct values it actually matched, so a miss is visible.
CORON_INSTRUMENT = "*CORON*"
NIRCAM_PREFIX, MIRI_PREFIX = "NIRCAM", "MIRI"

TARGETS = {
    # RX J0534.0-0221 -- Ferrer-Chavez, Wang, Wagner, Lawson et al. 2026 (arXiv:2609.20748),
    # "The JWST Sub-Jupiters Survey".  NIRCam F444W + F200W; the planet is at 0.41", 2.8 MJup,
    # with a debris disk at 79 au.  EV Lac (jw06122009) and Wolf 359 (jw06122001) are already
    # on disk from the same program, so the program ID is the safe way in if the name
    # resolver disagrees about this one.
    "rxj0534": dict(proposal_id="6122", target_name="RX*J0534*"),
    # AF Lep b -- GO 4558, the only public program with coronagraphy of it: NIRCam
    # F200W + F356W + F444W behind MASKRND (confirmed with --find "AF Lep", 2026-09-20).
    "aflep":   dict(proposal_id="4558", target_name="AF*LEP*"),
    # already on disk, here for completeness / re-fetch
    "mwc758":  dict(proposal_id="4014"),
}


def _obs():
    from astroquery.mast import Observations
    return Observations


def _check_fields(**criteria):
    """Refuse a query whose columns MAST does not know.

    astroquery warns and *drops* an unknown criterion, so a typo widens the query to the
    whole archive instead of failing.  Validating up front turns that into an error.
    """
    Observations = _obs()
    try:
        meta = Observations.get_metadata("observations")
        known = {str(v).strip() for v in meta["Column Name"]}
    except Exception as exc:                          # offline: fall back to warnings-as-errors
        print(f"  (could not fetch the column list: {exc}; relying on warnings)", file=sys.stderr)
        return
    bad = sorted(k for k in criteria if k not in known
                 and k not in {"objectname", "coordinates", "radius", "resolver"})
    if bad:
        raise SystemExit(f"not CAOM columns, MAST would silently ignore them: {bad}\n"
                         f"valid columns include: {sorted(known)[:25]} ...")


def _coron(public_only=True, **extra):
    """Every public JWST coronagraphic observation, optionally narrowed by ``extra``."""
    Observations = _obs()
    crit = dict(obs_collection="JWST", instrument_name=CORON_INSTRUMENT, **extra)
    if public_only:
        crit["dataRights"] = "PUBLIC"
    _check_fields(**crit)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)   # a skipped filter is a failure
        try:
            return Observations.query_criteria(**crit)
        except UserWarning as exc:
            raise SystemExit(f"MAST skipped a filter, so the result would not mean what it "
                             f"says: {exc}")


def census(public_only=True):
    """The two numbers the AR attachment's \\nProgs / \\nDatasets placeholders want."""
    t = _coron(public_only=public_only)
    if not len(t):
        raise SystemExit(f"no coronagraphy matched instrument_name={CORON_INSTRUMENT!r} -- "
                         f"check the spelling against Observations.get_metadata('observations')")
    inst = [str(v) for v in t["instrument_name"]]
    prog = [str(v) for v in t["proposal_id"]]
    print(f"  matched instrument_name values: {sorted(set(inst))}")
    for label, pref in (("nircam", NIRCAM_PREFIX), ("miri", MIRI_PREFIX)):
        rows = [k for k, v in enumerate(inst) if v.upper().startswith(pref)]
        ids = {prog[k] for k in rows}
        print(f"  {label:7s} {len(rows):5d} observations  {len(ids):3d} programs")
    progs = sorted(set(prog))
    print(f"\nTOTAL public coronagraphy: {len(t)} observations across {len(progs)} programs")
    print(f"  \\newcommand{{\\nProgs}}{{{len(progs)}}}")
    print(f"  \\newcommand{{\\nDatasets}}{{{len(t)}}}")
    return progs, len(t)


def _role(target: str) -> str:
    """``sci``, or what kind of supporting pointing this is.

    Background and PSF-reference exposures are real observations and a reduction needs
    them -- the reference pointing IS the RDI library -- but they are not things to point
    a search at, and undifferentiated they bury the science targets: of the first sixty
    rows of public MIRI coronagraphy, thirty-one were background or reference pointings.
    """
    t = target.lower()
    if "bkg" in t or "background" in t:
        return "bkg"
    if "reference" in t or "psfref" in t or t.endswith("-ref"):
        return "ref"
    return "sci"


def _rows(t):
    """(program, target, filter, instrument) tuples, de-duplicated.

    Sorted by programme NUMERICALLY.  ``proposal_id`` is a string, so the default sort put
    programme 10758 between 1046 and 1193 -- which looks like a listing bug and makes a
    programme's rows non-contiguous.
    """
    keys = ("proposal_id", "target_name", "filters", "instrument_name")
    out = {tuple(str(r[k]) for k in keys) for r in t}
    return sorted(out, key=lambda r: (int(r[0]) if r[0].isdigit() else 1 << 30, r[1], r[2]))


def miri_list(public_only=True, limit=60, filt_only=None, science_only=False):
    """Public MIRI coronagraphy to pick a pilot target from.

    ``science_only`` drops background and PSF-reference pointings from the listing; the
    count of each is still reported, because a target with no reference pointing in its
    programme has no RDI library and that is worth knowing before choosing it.
    """
    t = _coron(public_only=public_only)
    t = t[[str(v).upper().startswith(MIRI_PREFIX) for v in t["instrument_name"]]]
    rows = _rows(t)
    if filt_only:
        want = {f.upper() for f in filt_only}
        rows = [r for r in rows if any(w in r[2].upper() for w in want)]
    roles = {r: _role(r[1]) for r in rows}
    n_ref = sum(1 for v in roles.values() if v == "ref")
    n_bkg = sum(1 for v in roles.values() if v == "bkg")
    shown = [r for r in rows if roles[r] == "sci"] if science_only else rows
    # which programmes have a reference pointing at all -- i.e. where RDI is possible
    with_ref = {r[0] for r in rows if roles[r] == "ref"}
    print(f"{'program':>8}  {'role':4s} {'target':30s} {'filter':14s} RDI")
    for r in shown[:limit]:
        p, tgt, f, _ = r
        print(f"{p:>8}  {roles[r]:4s} {tgt[:30]:30s} {f.split(';')[0][:14]:14s} "
              f"{'yes' if p in with_ref else '--'}")
    print(f"\n{len(rows)} distinct (program, target, filter) combinations "
          f"({n_ref} reference, {n_bkg} background, {len(rows) - n_ref - n_bkg} science); "
          f"showing {min(limit, len(shown))}")
    print(f"{len(with_ref)} programmes include a PSF-reference pointing, so RDI is possible "
          f"in those; elsewhere the search has ADI only.")
    return rows


def find(name, public_only=True):
    """Which programs hold coronagraphy of a target, by name (wildcards allowed)."""
    pat = name if any(c in name for c in "*%") else f"*{name.replace(' ', '*')}*"
    t = _coron(public_only=public_only, target_name=pat)
    if not len(t):
        print(f"  nothing public matched target_name={pat!r}")
        return []
    rows = _rows(t)
    print(f"{'program':>8}  {'target':28s} {'filter':12s} instrument")
    for p, tgt, filt, inst in rows:
        print(f"{p:>8}  {tgt[:28]:28s} {filt[:12]:12s} {inst}")
    return rows


def fetch(name, outdir, download=False, products=("CALINTS", "ASN")):
    spec = dict(TARGETS[name])
    print(f"\n=== {name}: {spec}")
    t = _coron(**spec)
    if not len(t):
        print("  no matches -- try --find with part of the name, or drop target_name and "
              "keep proposal_id")
        return
    for p, tgt, filt, inst in _rows(t):
        print(f"  {p:>6}  {tgt[:26]:26s} {filt[:12]:12s} {inst}")
    if not download:
        print("  (--download to pull the products)")
        return
    Observations = _obs()
    prod = Observations.get_product_list(t)
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
    ap.add_argument("--limit", type=int, default=60, metavar="N",
                    help="rows to print (default 60; 0 for all)")
    ap.add_argument("--filters", nargs="*", default=None, metavar="F",
                    help="narrow the listing to these filters, e.g. --filters F1140C F1550C")
    ap.add_argument("--science-only", action="store_true",
                    help="hide background and PSF-reference pointings")
    ap.add_argument("--find", metavar="NAME", help="which programs observed this target")
    ap.add_argument("--targets", nargs="*", default=[], choices=sorted(TARGETS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--all-rights", action="store_true",
                    help="include proprietary data in the census (default: public only)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/Data/JWST"))
    a = ap.parse_args()
    pub = not a.all_rights
    if a.count:
        print(f"{'Public' if pub else 'All'} JWST coronagraphy in MAST:")
        census(public_only=pub)
    if a.miri:
        print(f"\n{'Public' if pub else 'All'} MIRI coronagraphy:")
        miri_list(public_only=pub, limit=a.limit or 10 ** 9, filt_only=a.filters,
                  science_only=a.science_only)
    if a.find:
        print(f"\nCoronagraphy of {a.find!r}:")
        find(a.find, public_only=pub)
    for name in a.targets:
        fetch(name, a.outdir, a.download)
    if not (a.count or a.miri or a.find or a.targets):
        ap.print_help()


if __name__ == "__main__":
    main()
