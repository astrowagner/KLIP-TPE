"""Stitch helpers and FITS/text product writers for the run protocol (A §9).

Everything here is a pure function of arrays + numbers; :class:`~klip_tpe.runner.Runner`
decides *what* to stitch (previous annuli's winners, the current running best, the
per-partition stacks) and calls in here for the *how*:

* :func:`tile_mask` / :func:`stitch_tiles` -- NaN-aware weighted mean over radial tiles
  ``edge_ia - pad <= r <= edge_ia+1 + pad`` (running stitch: equal weights; final
  stitch: the sensitivity weights of :func:`klip_tpe.products.stitch_annuli`).
* :func:`stitch_stacks` -- the same for per-partition stacks onto the union of
  partition ids (NaN where an annulus did not use a partition).
* :func:`seam_pixels` -- pixels inside the searched span still NaN after stitching.
* :func:`fits_header` / :func:`write_fits` -- provenance headers (``.fits`` or ``.fits.gz``).
* :func:`write_params_table`, :func:`write_contrast_curve`, :func:`write_setup_file`
  -- the IDL text products (``klip_stitched_params.txt``, ``contrast_curve.txt``,
  ``evalNNNN_setup.txt`` ...).
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .metrics import star_center

__all__ = ["tile_mask", "stitch_tiles", "stitch_stacks", "seam_pixels", "seam_trim",
           "fits_header", "write_fits", "read_fits", "write_params_table", "write_contrast_curve",
           "write_setup_file", "PARAM_COLUMNS"]

PARAM_COLUMNS = ("k_klip", "bin", "n_ang", "filter", "angsep", "anglemax", "corr_thresh", "noise_max",
                 "coronoise_max")


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def _radius(shape) -> np.ndarray:
    ny, nx = shape
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.hypot(xx - cx, yy - cy)


def tile_mask(shape, rin: float, rout: float, pad: float = 2.0) -> np.ndarray:
    """Boolean tile ``rin - pad <= r <= rout + pad``."""
    rr = _radius(shape)
    return (rr >= rin - pad) & (rr <= rout + pad)


def stitch_tiles(images: Sequence[Optional[np.ndarray]], zones: Sequence[Tuple[float, float]],
                 weights: Optional[Sequence[float]] = None, pad: float = 2.0) -> np.ndarray:
    """NaN-aware weighted mean of ``images`` over their tiles (``None`` entries skipped).
    ``zones`` are ``(rin, rout)`` in px; overlaps (±pad) are weighted means."""
    ref = next((im for im in images if im is not None), None)
    if ref is None:
        raise ValueError("no images to stitch")
    ny, nx = ref.shape
    w = np.ones(len(images)) if weights is None else np.asarray(weights, float)
    num = np.zeros((ny, nx))
    den = np.zeros((ny, nx))
    for im, (rin, rout), wi in zip(images, zones, w):
        if im is None or im.shape != (ny, nx):
            continue
        m = tile_mask((ny, nx), rin, rout, pad) & np.isfinite(im)
        num[m] += wi * im[m]
        den[m] += wi
    return num / np.where(den > 0, den, np.nan)


def stitch_stacks(stacks: Sequence[Optional[np.ndarray]], part_ids: Sequence[Optional[Sequence[Any]]],
                  zones: Sequence[Tuple[float, float]], weights: Optional[Sequence[float]] = None,
                  pad: float = 2.0) -> Tuple[Optional[np.ndarray], List[str]]:
    """Stitch per-partition stacks ``(npart_ia, ny, nx)`` onto the union of partition ids
    (sorted by first appearance).  Returns ``(cube (nunion, ny, nx), ids)``; NaN where an
    annulus did not use a partition.  ``(None, [])`` when nothing is available."""
    union: List[str] = []
    for ids, st in zip(part_ids, stacks):
        if st is None or ids is None:
            continue
        for p in ids:
            if str(p) not in union:
                union.append(str(p))
    if not union:
        return None, []
    ref = next(st for st in stacks if st is not None)
    ny, nx = ref.shape[-2:]
    w = np.ones(len(stacks)) if weights is None else np.asarray(weights, float)
    num = np.zeros((len(union), ny, nx))
    den = np.zeros((len(union), ny, nx))
    for st, ids, (rin, rout), wi in zip(stacks, part_ids, zones, w):
        if st is None or ids is None or st.shape[-2:] != (ny, nx):
            continue
        tile = tile_mask((ny, nx), rin, rout, pad)
        for j, p in enumerate(ids):
            if j >= st.shape[0]:
                break
            u = union.index(str(p))
            m = tile & np.isfinite(st[j])
            num[u][m] += wi * st[j][m]
            den[u][m] += wi
    return num / np.where(den > 0, den, np.nan), union


def seam_pixels(stitched: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """Mask of pixels with ``r0 <= r <= r1`` that are still NaN after stitching."""
    rr = _radius(stitched.shape)
    return (rr >= r0) & (rr <= r1) & ~np.isfinite(stitched)


def seam_trim(fwhm: float, pxscale: float) -> float:
    """Contrast-curve seam trim ``max(round(fwhm/3), 1) * pxscale`` (arcsec)."""
    return max(int(round(fwhm / 3.0)), 1) * pxscale


# ----------------------------------------------------------------------------
# FITS
# ----------------------------------------------------------------------------
def fits_header(d: Dict[str, Any], history: Sequence[str] = ()):
    """Build an ``astropy.io.fits.Header`` from a flat dict.  Keys are upper-cased and
    truncated to 8 characters (HIERARCH is avoided so the files stay readable by
    everything); values that FITS cannot hold are stringified; ``None`` / non-finite
    floats are skipped; ``history`` lines go into HISTORY cards.  ``PROG``, ``DATE``
    and ``KTPEVER`` (klip_tpe version) are always present."""
    from astropy.io import fits
    h = fits.Header()
    h["PROG"] = ("klip_tpe.runner", "generating package")
    h["KTPEVER"] = (str(__version__), "klip_tpe version")
    h["DATE"] = (_dt.datetime.now().isoformat(timespec="seconds"), "file creation time")
    for k, v in d.items():
        key = str(k).replace("-", "_").upper()[:8]
        if v is None:
            continue
        if isinstance(v, (np.floating, float)):
            if not np.isfinite(v):
                continue
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (bool, np.bool_)):
            v = bool(v)
        elif not isinstance(v, (int, str)):
            v = str(v)
        if isinstance(v, str) and len(v) > 68:
            v = v[:68]
        try:
            h[key] = v
        except Exception:
            h[key] = str(v)[:68]
    for line in history:
        h.add_history(str(line))
    return h


def write_fits(path: str, data: np.ndarray, header: Optional[Dict[str, Any]] = None,
               history: Sequence[str] = ()) -> str:
    """Write ``data`` (float32) with :func:`fits_header`.  ``.fits.gz`` paths are
    compressed by astropy.  Falls back to ``.npy`` without astropy.  Returns the path."""
    try:
        from astropy.io import fits
    except Exception:  # pragma: no cover
        np.save(path.replace(".fits.gz", "").replace(".fits", "") + ".npy", data)
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fits.PrimaryHDU(np.asarray(data, np.float32), header=fits_header(header or {}, history)).writeto(
        path, overwrite=True)
    return path


def read_fits(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        from astropy.io import fits
        return np.asarray(fits.getdata(path), float)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# text products
# ----------------------------------------------------------------------------
def _fmt(v, fmt: str, default: str = "nan") -> str:
    try:
        f = float(v)
        return fmt.format(f) if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def write_params_table(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """``klip_stitched_params.txt``: one row per annulus with the representative
    (median over partitions) values, then a per-partition block.

    Each row dict: ``ann, inner_px, outer_px, params (dict), best_SNR, partitions (list),
    per_partition ({pid: params})`` plus optional ``validated``, ``contrast``.
    Returns the text."""
    L = ["# Per-annulus optimal parameters used for klip_stitched.fits",
         f"# klip_tpe {__version__} -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         "# inner/outer in pixels; representative values = median over the selected partitions.",
         f"{'ann':>4}  {'inner_px':>8}  {'outer_px':>8}  {'k':>6}  {'bin':>7}  {'n_ang':>7}  {'filter':>7}  "
         f"{'angsep':>7}  {'anglemax':>8}  {'corr':>7}  {'noise':>7}  {'coro':>7}  {'best_SNR':>9}  "
         f"{'valid':>5}  {'contrast':>10}  partitions"]
    for r in rows:
        p = r.get("params", {})
        parts = r.get("partitions") or []
        L.append(f"{int(r['ann']):4d}  {float(r['inner_px']):8.2f}  {float(r['outer_px']):8.2f}  "
                 f"{_fmt(p.get('k_klip'), '{:6.0f}'):>6}  {_fmt(p.get('bin'), '{:7.2f}'):>7}  "
                 f"{_fmt(p.get('n_ang'), '{:7.2f}'):>7}  {_fmt(p.get('filter'), '{:7.2f}'):>7}  "
                 f"{_fmt(p.get('angsep'), '{:7.3f}'):>7}  {_fmt(p.get('anglemax'), '{:8.1f}'):>8}  "
                 f"{_fmt(p.get('corr_thresh'), '{:7.3f}'):>7}  {_fmt(p.get('noise_max'), '{:7.2f}'):>7}  "
                 f"{_fmt(p.get('coronoise_max'), '{:7.2f}'):>7}  {_fmt(r.get('best_SNR'), '{:9.3f}'):>9}  "
                 f"{int(bool(r.get('validated', True))):5d}  {_fmt(r.get('contrast'), '{:10.3E}'):>10}  "
                 + (",".join(str(v) for v in parts) if parts else "all"))
    if any(r.get("per_partition") for r in rows):
        L += ["", "# per-partition blocks (one row per annulus x selected partition)",
              f"{'ann':>4}  {'partition':>10}  " + "  ".join(f"{c:>13}" for c in PARAM_COLUMNS)]
        for r in rows:
            for pid, pp in (r.get("per_partition") or {}).items():
                if r.get("partitions") and str(pid) not in [str(v) for v in r["partitions"]]:
                    continue
                L.append(f"{int(r['ann']):4d}  {str(pid):>10}  " + "  ".join(
                    f"{_fmt(pp.get(c), '{:13.4g}'):>13}" for c in PARAM_COLUMNS))
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def write_contrast_curve(path: str, curves: Sequence[Optional[Dict[str, Any]]], edges_px: Sequence[float],
                         fwhm: float, pxscale: float, legacy: bool = False,
                         contrasts: Optional[Sequence[float]] = None,
                         fm_curves: Optional[Sequence[Optional[Dict[str, Any]]]] = None) -> Dict[str, np.ndarray]:
    """Run-level ``contrast_curve.txt``: per-annulus curves concatenated with the seam
    trim ``trim = max(round(fwhm/3), 1) * pxscale`` (annulus ``ia`` keeps ``r >=
    edge_ia + trim`` for ``ia > 0`` and ``r <= edge_ia+1 - trim`` for ``ia < nann-1``;
    ``legacy`` disables the trim), sorted by separation, followed by the per-sample
    ``5 * contrast / SNR`` points.  ``curves`` are the ``AnnulusResult.contrast_curve``
    dicts (``r_as, curve, sample_r, sample_c``; ``None`` entries skipped).  ``fm_curves``
    (the ``AnnulusResult.fm_curve`` dicts) append the ``# KLIP-FM`` cross-check section
    (A §6.4): two comment lines then ``sep_arcsec  contrast_5sig`` rows over all
    annuli, sorted by separation (no seam trim -- the test radii are inside the zone).
    Returns the concatenated arrays (``fm_r`` / ``fm_c`` when an FM section was written)."""
    nann = len(curves)
    trim = 0.0 if legacy else seam_trim(fwhm, pxscale)
    R, C, SR, SC, SA = [], [], [], [], []
    for ia, cc in enumerate(curves):
        if not cc or cc.get("r_as") is None or cc.get("curve") is None:
            continue
        r = np.array([np.nan if v is None else float(v) for v in cc["r_as"]])
        c = np.array([np.nan if v is None else float(v) for v in cc["curve"]])
        keep = np.isfinite(r) & np.isfinite(c)
        if ia > 0 and ia < len(edges_px):
            keep &= r >= edges_px[ia] * pxscale + trim
        if ia < nann - 1 and ia + 1 < len(edges_px):
            keep &= r <= edges_px[ia + 1] * pxscale - trim
        R.extend(r[keep]); C.extend(c[keep])
        sr = cc.get("sample_r") or []
        sc = cc.get("sample_c") or []
        for a, b in zip(sr, sc):
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                SR.append(float(a)); SC.append(float(b)); SA.append(ia + 1)
    R, C = np.asarray(R, float), np.asarray(C, float)
    o = np.argsort(R, kind="stable")
    L = ["# Injection-calibrated SNR=5 contrast curve (validated winners per annulus)",
         f"# klip_tpe {__version__} -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         f"# seam trim = {trim:.4f} arcsec ({'legacy_stitch: no trim' if legacy else 'max(round(fwhm/3),1) px'});"
         f" annulus edges (px) = {', '.join(f'{e:.1f}' for e in edges_px)}"]
    if contrasts:
        L.append("# injection contrast per annulus = " + ", ".join(f"{c:.3E}" for c in contrasts))
    L.append("# separation_arcsec   contrast_snr5")
    for i in o:
        L.append(f"{R[i]:10.3f}   {C[i]:12.4E}")
    if SR:
        L.append("# per-sample 5*contrast/SNR points (validation injections):")
        L.append("#   sep(\")    5sig_contrast_sample   annulus")
        so = np.argsort(SR, kind="stable")
        for i in so:
            L.append(f"{SR[i]:10.3f}   {SC[i]:12.4E}   {SA[i]:3d}")
    out = {"r_as": R[o], "curve": C[o], "sample_r": np.asarray(SR, float), "sample_c": np.asarray(SC, float)}
    FR, FC, FA = [], [], []
    for ia, fc in enumerate(fm_curves or []):
        if not fc or fc.get("r_as") is None or fc.get("curve") is None:
            continue
        for a, b in zip(fc["r_as"], fc["curve"]):
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and b > 0:
                FR.append(float(a)); FC.append(float(b)); FA.append(ia + 1)
    if FR:
        fo = np.argsort(FR, kind="stable")
        L.append(f"# KLIP-FM cross-check 5-sigma contrast (clean forward-model pass at each annulus' winner config; "
                 f"annuli {', '.join(str(a) for a in sorted(set(FA)))})")
        L.append("# sep_arcsec   contrast_5sig")
        for i in fo:
            L.append(f"{FR[i]:10.4f}   {FC[i]:12.4E}")
        out["fm_r"], out["fm_c"] = np.asarray(FR, float)[fo], np.asarray(FC, float)[fo]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return out


def write_setup_file(path: str, phase: str, annulus: int, step: int, params: Dict[str, Any],
                     inrad: float, outrad: float, partitions: Sequence[Any] = (),
                     sources: Sequence[Sequence[float]] = (), contrast: Optional[float] = None,
                     snr: Optional[float] = None, per_partition: Optional[Dict[Any, Dict[str, Any]]] = None,
                     k_used: Any = None, weights: Optional[Sequence[float]] = None,
                     extra: Optional[Dict[str, Any]] = None) -> str:
    """One step-setup text file in the ``near2m_write_setup`` layout (A §6.3)::

        # KLIP-TPE step setup -- <date>
        phase         : eval
        annulus       : 1
        step          : 12
        bin           : 29
        ...
        k_klip        : 6
        inrad/outrad  : 11.0 / 44.0 px
        nights        : n1,n2
        median_SNR    : 5.123
        # injected sources:  rho_arcsec   PA_deg     contrast
            0.700      123.40   3.000E-05

    followed (partitioned runs) by a ``# PER-PARTITION block`` table and, for a dict
    ``k_used``, a ``per-partition k*`` line.  Never raises (a text product must not
    kill a run); returns the text ('' on failure)."""
    try:
        L = [f"# KLIP-TPE step setup -- {_dt.datetime.now().isoformat(timespec='seconds')}  (klip_tpe {__version__})",
             f"{'phase':<14}: {phase}", f"{'annulus':<14}: {int(annulus) + 1}", f"{'step':<14}: {int(step)}"]
        p = params or {}
        for name, fmt in (("bin", "{:.0f}"), ("n_ang", "{:.0f}"), ("filter", "{:.0f}"), ("angsep", "{:.3f}"),
                          ("anglemax", "{:.1f}"), ("corr_thresh", "{:.3f}"), ("noise_max", "{:.3f}"),
                          ("coronoise_max", "{:.3f}"), ("k_klip", "{:.0f}")):
            if name in p and p[name] is not None:
                L.append(f"{name:<14}: {_fmt(p[name], fmt, str(p[name]))}")
        for name, v in p.items():
            if name not in ("bin", "n_ang", "filter", "angsep", "anglemax", "corr_thresh", "noise_max",
                            "coronoise_max", "k_klip", "inrad", "outrad", "width"):
                L.append(f"{name:<14}: {v}")
        L.append(f"{'inrad/outrad':<14}: {float(inrad):.1f} / {float(outrad):.1f} px")
        L.append(f"{'nights':<14}: {','.join(str(v) for v in partitions) if partitions else 'all'}")
        if isinstance(k_used, dict):
            L.append(f"{'per-partition k*':<14}: " + " ".join(f"{k}={v}" for k, v in k_used.items()))
        if snr is not None and np.isfinite(float(snr)):
            L.append(f"{'median_SNR':<14}: {float(snr):.3f}")
        for k, v in (extra or {}).items():
            L.append(f"{str(k):<14}: {v}")
        if sources:
            L.append("# injected sources:  rho_arcsec   PA_deg     contrast")
            for s in sources:
                c = s[2] if len(s) > 2 else (contrast if contrast is not None else 0.0)
                L.append(f"  {float(s[0]):10.3f}  {float(s[1]):10.2f}  {float(c):12.3E}")
        else:
            L.append("# injected sources: none (clean / aggregate step)")
        if per_partition:
            pids = [pp for pp in per_partition if not partitions or str(pp) in [str(v) for v in partitions]]
            cols = [c for c in PARAM_COLUMNS if any(c in per_partition[pp] for pp in pids)]
            L.append("# PER-PARTITION block")
            L.append(f"  {'partition':>10}  " + "  ".join(f"{c:>13}" for c in cols))
            for pp in pids:
                L.append(f"  {str(pp):>10}  " + "  ".join(f"{_fmt(per_partition[pp].get(c), '{:13.4g}'):>13}" for c in cols))
            if weights is not None:
                L.append("  per-partition combine weights: " + " ".join(f"{float(w):.4f}" for w in weights))
        text = "\n".join(L) + "\n"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return text
    except Exception:
        return ""
