"""Fit a chain of point sources to a finished run's final image, through the run's own winners.

A feature fixed on the sky, such as the line through alpha Cen A along the A-B direction in the
NEAR data, can be modelled as a chain of point sources whose contrasts follow a few node
values.  :func:`fit_chain` takes each node's part of the chain (its *basis*) through every
annulus' winning reduction with KLIP-FM, stitches the responses as the run stitched its final
image, and fits the node values to that image by noise-weighted least squares in a strip about
the line.  The fitted chain is written in the format ``--subtract-sources`` reads.

The chain file has one source per line, ``rho pa contrast [node]``: ``rho`` in arcsec, ``pa``
in degrees East of North, ``contrast`` in the injection model's flux unit, and an optional node
label.  A node's basis is its lines, each source at the contrast it has per unit node value, so
a source shared by two nodes (a linear interpolation between them) appears on two lines.  A
file without labels is one node, and the fit then scales the whole chain.  The fitted value of
a node multiplies its basis, and the fitted chain is the sum over the nodes.

A line can also be given by its nodes (:func:`read_line_nodes`): the PA of one half-line, the
step between chain sources, and the ridge's peak contrast ``a`` at a few separations on each
side, piecewise linear between them and held at the last value out to the crop edge.
:func:`line_chain` turns that into the chain, each source at ``a(s) step / L`` with ``L`` the
injection PSF's own line integral (:func:`psf_line_integral`), and :func:`line_basis` into one
basis per node and side for the fit.  ``klip-tpe fit-chain --line-nodes FILE`` fits the node
values, ``--subtract-line FILE`` subtracts the line a node file describes.

``klip-tpe fit-chain --run-dir RUN --chain FILE`` runs the fit on the command line.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source, line_mask, source_xy, star_center

__all__ = ["read_chain", "chain_pa", "read_candidates", "fit_chain", "read_line_nodes", "write_line_nodes",
           "psf_line_integral", "line_chain", "line_basis", "fit_per_night"]


def read_chain(path: str) -> Tuple[List[str], Dict[str, List[Source]]]:
    """``(labels, {label: [Source, ...]})`` from a chain file, the labels in the order they
    first appear.  Lines without a node label make the single node ``"all"``.  A file that
    labels some lines and not others is refused."""
    labels: List[str] = []
    nodes: Dict[str, List[Source]] = {}
    with_label = None
    with open(path) as f:
        for i, line in enumerate(f, 1):
            s = line.split("#", 1)[0].replace(",", " ").split()
            if not s:
                continue
            if len(s) < 3:
                raise ValueError(f"{path}:{i}: need rho, pa and contrast, got {line.strip()!r}")
            try:
                rho, pa, c = (float(v) for v in s[:3])
            except ValueError:
                raise ValueError(f"{path}:{i}: not three numbers: {line.strip()!r}") from None
            has = len(s) >= 4
            if with_label is None:
                with_label = has
            elif has != with_label:
                raise ValueError(f"{path}:{i}: some lines have a node label and some do not")
            lab = s[3] if has else "all"
            if lab not in nodes:
                labels.append(lab)
                nodes[lab] = []
            nodes[lab].append(Source(rho, pa, c))
    if not labels:
        raise ValueError(f"{path}: no sources")
    return labels, nodes


def read_line_nodes(path: str) -> Dict[str, Any]:
    """A line through the star, by its nodes.  The file holds keyword lines and node lines::

        pa 13            # PA of the first half-line (deg), the second is pa + 180
        step 0.5         # px between chain sources along each half-line (default 0.5)
        rho_max 3.4      # arcsec, the last value is held out to here (default: the crop edge)
        # |s| (arcsec)   a, first half-line   a, second half-line
        1.0  1.0e-6  0.5e-6
        1.5  2.0e-6  1.0e-6

    ``a`` is the ridge's peak contrast: across the line, the ridge peaks where a point source of
    contrast ``a`` would.  A node line with only the separation leaves both values 0, as a fit
    needs no values.  Returns ``{"pa", "step",
    "rho_max", "nodes", "a1", "a2"}`` (``rho_max`` None for the crop edge)."""
    spec: Dict[str, Any] = {"pa": None, "step": 0.5, "rho_max": None, "nodes": [], "a1": [], "a2": []}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            s = line.split("#", 1)[0].replace(",", " ").split()
            if not s:
                continue
            key = s[0].lower()
            if key in ("pa", "step", "rho_max"):
                if len(s) != 2:
                    raise ValueError(f"{path}:{i}: {key} takes one value")
                spec[key] = float(s[1])
                continue
            try:
                v = [float(x) for x in s]
            except ValueError:
                raise ValueError(f"{path}:{i}: neither a keyword (pa, step, rho_max) nor numbers: "
                                 f"{line.strip()!r}") from None
            if len(v) not in (1, 3):
                raise ValueError(f"{path}:{i}: a node is |s| alone, or |s| and a value for each side")
            spec["nodes"].append(v[0])
            spec["a1"].append(v[1] if len(v) == 3 else 0.0)
            spec["a2"].append(v[2] if len(v) == 3 else 0.0)
    if spec["pa"] is None:
        raise ValueError(f"{path}: no 'pa' line")
    n = np.asarray(spec["nodes"], float)
    if n.size < 1 or np.any(n <= 0) or np.any(np.diff(n) <= 0):
        raise ValueError(f"{path}: the nodes' separations must be positive and increasing")
    if spec["step"] <= 0:
        raise ValueError(f"{path}: step must be positive")
    return spec


def write_line_nodes(path: str, spec: Dict[str, Any], a1: Sequence[float], a2: Sequence[float],
                     e1: Optional[Sequence[float]] = None, e2: Optional[Sequence[float]] = None,
                     comment: str = "") -> None:
    """A node file (:func:`read_line_nodes`) with the values ``a1`` / ``a2``, and their errors
    as comments."""
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"pa {spec['pa']!r}\nstep {spec['step']!r}\n")
        if spec.get("rho_max") is not None:
            f.write(f"rho_max {spec['rho_max']!r}\n")
        f.write("# |s| (arcsec)  a, first half-line  a, second half-line\n")
        for j, s in enumerate(spec["nodes"]):
            err = "" if e1 is None else f"  # +- {e1[j]:.3g}  +- {e2[j]:.3g}"
            f.write(f"{float(s)!r} {float(a1[j])!r} {float(a2[j])!r}{err}\n")


def psf_line_integral(model, rho: float, fallback=None, step: float = 0.25) -> float:
    """``L = integral PSF(x, 0) dx / PSF(0, 0)`` (px) of the stamp a source at ``rho`` gets:
    the line integral through the source along the radial direction, over the value at the
    source.  A line through the star is radial everywhere, and an anisotropic model's stamp is
    turned so its reference axis is radial (``refpa_deg``), so the integral runs along that
    axis.  For a model without one it is the mean over directions.  A continuous line of
    ``lambda`` contrast per px has a ridge of peak ``lambda L``, so sources ``step`` px apart
    at ``a step / L`` make a ridge of peak ``a``."""
    from scipy import ndimage
    m = model
    st, c, ok = m.stamp(rho)
    if not ok and fallback is not None:
        m = fallback
        st, c, ok = m.stamp(rho)
    if not ok:
        raise ValueError(f"no PSF stamp at rho={rho:g} arcsec")
    st = np.where(np.isfinite(st), np.asarray(st, float), 0.0)
    cx, cy = float(c[0]), float(c[1])
    peak = float(ndimage.map_coordinates(st, [[cy], [cx]], order=3)[0])
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"the PSF stamp at rho={rho:g} arcsec has no positive value at its center")
    half = 0.5 * max(st.shape)
    t = np.arange(-half, half + 0.5 * step, step)
    dirs = [float(m.refpa_deg)] if getattr(m, "refpa_deg", None) is not None else list(np.arange(0.0, 180.0, 22.5))
    out = []
    for d in dirs:
        a = np.deg2rad(d)
        # a cubic spline: a bilinear profile across the pixel grid runs low near the peak
        prof = ndimage.map_coordinates(st, [cy + t * np.sin(a), cx + t * np.cos(a)], order=3, mode="constant", cval=0.0)
        out.append(float(np.sum(0.5 * (prof[1:] + prof[:-1])) * step) / peak)   # trapezoid rule
    return float(np.mean(out))


def _line_positions(spec: Dict[str, Any], pxscale: float, rho_max: float) -> np.ndarray:
    """The chain's separations (arcsec) along each half-line: every ``step`` px from the first
    node out to ``rho_max``."""
    d = float(spec["step"]) * float(pxscale)
    r0 = float(spec["nodes"][0])
    return np.round(r0 + d * np.arange(int(np.floor((rho_max - r0) / d + 1e-9)) + 1), 9)


def _rho_max(spec: Dict[str, Any], frame_shape: Tuple[int, int], pxscale: float) -> float:
    if spec.get("rho_max") is not None:
        return float(spec["rho_max"])
    return 0.5 * (min(frame_shape) - 1) * float(pxscale)       # the crop's inscribed edge


def line_chain(spec: Dict[str, Any], model, pxscale: float, frame_shape: Tuple[int, int], fallback=None,
               a1: Optional[Sequence[float]] = None, a2: Optional[Sequence[float]] = None) -> List[Source]:
    """The chain of point sources for the line ``spec`` (:func:`read_line_nodes`), or for the
    node values ``a1`` / ``a2`` in its place: a source every ``step`` px along each half-line
    from the first node out to ``rho_max`` (default: the crop's inscribed edge), each at
    ``a(s) step / L(s)``, with ``a`` piecewise linear between the nodes and held at the last
    value beyond them, and ``L`` the injection PSF's line integral there
    (:func:`psf_line_integral`).  The contrasts are the line's own, ready for
    :meth:`~klip_tpe.reducer.KLIPReducer.subtract_sources`."""
    nodes = np.asarray(spec["nodes"], float)
    v1 = np.asarray(spec["a1"] if a1 is None else a1, float)
    v2 = np.asarray(spec["a2"] if a2 is None else a2, float)
    rho = _line_positions(spec, pxscale, _rho_max(spec, frame_shape, pxscale))
    step = float(spec["step"])
    out: List[Source] = []
    for pa, v in ((float(spec["pa"]) % 360.0, v1), ((float(spec["pa"]) + 180.0) % 360.0, v2)):
        a = np.interp(rho, nodes, v)
        for r, ar in zip(rho, a):
            out.append(Source(float(r), pa, float(ar) * step / psf_line_integral(model, float(r), fallback)))
    return out


def line_basis(spec: Dict[str, Any], model, pxscale: float, frame_shape: Tuple[int, int],
               fallback=None) -> Tuple[List[str], Dict[str, List[Source]]]:
    """One basis chain per node and half-line for :func:`fit_chain`: the node at 1 and the
    others at 0, so a fitted value is the ridge's peak contrast ``a`` at that node.  Labels
    ``a1_<s>`` (first half-line, PA ``pa``) and ``a2_<s>`` (second, ``pa + 180``)."""
    nodes = np.asarray(spec["nodes"], float)
    rho = _line_positions(spec, pxscale, _rho_max(spec, frame_shape, pxscale))
    step = float(spec["step"])
    L = np.array([psf_line_integral(model, float(r), fallback) for r in rho])
    labels: List[str] = []
    basis: Dict[str, List[Source]] = {}
    for side, pa in (("a1", float(spec["pa"]) % 360.0), ("a2", (float(spec["pa"]) + 180.0) % 360.0)):
        for j, s in enumerate(nodes):
            unit = np.zeros(nodes.size)
            unit[j] = 1.0
            w = np.interp(rho, nodes, unit)
            lab = f"{side}_{s:g}"
            labels.append(lab)
            basis[lab] = [Source(float(r), pa, float(wi) * step / Li) for r, wi, Li in zip(rho, w, L) if wi > 0]
    return labels, basis


def chain_pa(nodes: Dict[str, Sequence[Source]], tol_deg: float = 5.0) -> float:
    """The position angle of the line the chain lies on, in [0, 180): the circular mean of
    the sources' PAs folded onto one side.  Refused when they spread more than ``tol_deg``
    about it, since the chain is then not one line (give ``line_pa`` instead)."""
    pas = np.array([s.theta for src in nodes.values() for s in src], float)
    z = np.exp(2j * np.deg2rad(pas))
    mean = 0.5 * np.rad2deg(np.angle(z.mean())) % 180.0
    dev = np.abs((pas - mean + 90.0) % 180.0 - 90.0)
    if dev.max() > tol_deg:
        raise ValueError(f"the chain's sources do not lie on one line through the star (PA spread up to "
                         f"{dev.max():.1f} deg about {mean:.1f}); give the line's PA")
    return float(mean)


def read_candidates(path: str) -> List[Dict[str, Any]]:
    """The rows of a run's ``cand/candidates.txt``: ``rho`` (arcsec), ``pa``, ``snr``, ``tier``."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s or line.lstrip().startswith("#") or s[0] == "rank":
                continue
            try:
                out.append({"rho": float(s[3]), "pa": float(s[4]), "snr": float(s[5]),
                            "tier": s[14] if len(s) > 14 else ""})
            except (ValueError, IndexError):
                continue
    return out


def _radial_sigma(img: np.ndarray, keep: np.ndarray, rr: np.ndarray, width: float) -> np.ndarray:
    """Robust scatter (1.4826 x MAD) of ``img`` over ``keep`` in rings ``width`` px wide,
    interpolated to every pixel's radius."""
    r = rr[keep]
    if r.size < 10:
        raise ValueError("too few pixels outside the strip and the masks to measure the noise")
    edges = np.arange(np.floor(r.min()), np.ceil(r.max()) + width, width)
    rc, sg = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = img[keep & (rr >= lo) & (rr < hi)]
        if v.size >= 10:
            rc.append(0.5 * (lo + hi))
            sg.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    if not rc:
        raise ValueError("no ring has enough pixels to measure the noise")
    return np.interp(rr, rc, sg)


def _span(rr: np.ndarray, edges: Sequence[float], rho_range: Optional[Tuple[float, float]],
          pxscale: float) -> np.ndarray:
    """The pixels inside the run's annuli (``edges`` in px) and inside ``rho_range`` (arcsec)."""
    span = (rr >= edges[0]) & (rr <= edges[-1])
    if rho_range is not None:
        lo, hi = float(rho_range[0]), float(rho_range[1])
        if not (hi > lo >= 0.0):
            raise ValueError(f"radial range {lo:g}-{hi:g} arcsec: need 0 <= rmin < rmax")
        span &= (rr >= lo / pxscale) & (rr <= hi / pxscale)
        if not span.any():
            raise ValueError(f"radial range {lo:g}-{hi:g} arcsec lies outside the run's annuli "
                             f"({edges[0] * pxscale:.3f}-{edges[-1] * pxscale:.3f} arcsec)")
    return span


def fit_chain(runner, nodes: Dict[str, Sequence[Source]], labels: Optional[Sequence[str]] = None, *,
              line_pa: Optional[float] = None, halfwidth_px: float = 12.0,
              masks: Sequence[Tuple[float, float, Optional[float]]] = (), mask_known: bool = True,
              candidate_tiers: str = "", out_dir: Optional[str] = None, line_spec: Optional[Dict[str, Any]] = None,
              rho_range: Optional[Tuple[float, float]] = None,
              log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Fit the node values of a chain to a finished run's final image.

    ``runner`` is a :class:`~klip_tpe.runner.Runner` on the finished run, built with the
    reducer the run used (:meth:`Runner.resume` checks the frames).  For every node, the
    node's sources go through each annulus' winning configuration as the KLIP-FM model
    (:meth:`Runner._reduce` with ``fm_sources``, so the zones, padding and known-source
    subtraction are the run's), and the FM images are stitched with the final stitch's
    weights and tiles.  The node values ``a`` then minimize
    ``sum w (image - sum_n a_n M_n)^2`` over the pixels within ``halfwidth_px`` of the line
    through the star at ``line_pa`` (default: the chain's own line, :func:`chain_pa`), inside
    the run's annuli (and inside ``rho_range``, ``(rmin, rmax)`` in arcsec, when given), with
    ``w = 1 / sigma(r)^2`` from the robust scatter of the image outside the strip.  A node
    that puts no signal in that region (beyond the run's annuli, say) is refused rather than
    reported as 0 +- 0.  Masked out: circles ``(rho_arcsec, pa_deg, radius_px)`` from ``masks``
    (``radius_px`` None: 1.5 FWHM), the run's known sources (``mask_known``), and the run's
    candidates of the tiers in ``candidate_tiers`` (``"A"``, ``"AB"``; the line's own peaks
    can be listed as candidates, so name only tiers that are not the line).

    The errors are the least-squares ones scaled by the square root of the pixels per beam,
    ``pi (FWHM / 2)^2``, since neighboring pixels are not independent.  Writes, under
    ``out_dir`` (default ``<run>_chainfit`` beside the run, which is left as it is):
    ``chainfit.json``, ``chain_fitted.txt`` (the fitted chain, ``rho pa contrast``),
    ``chainfit_model.fits``, ``chainfit_residual.fits``, ``chainfit_basis.fits`` (one stitched
    FM image per node, in the order of the labels) and ``chainfit_region.fits`` (1 where a
    pixel entered the fit).  With ``line_spec`` (the nodes the basis came from,
    :func:`line_basis`) also ``line_fitted.txt``, a node file with the fitted values.  Returns
    the JSON's content."""
    from .stitch import read_fits, stitch_tiles, write_fits
    t0 = time.time()
    labels = list(labels) if labels is not None else list(nodes)
    if not labels or any(not nodes.get(n) for n in labels):
        raise ValueError("every node needs at least one source")
    results = [r for r in (runner.results or []) if r is not None]
    if not results or len(results) < runner.cfg.nann:
        raise ValueError(f"{os.path.basename(runner.run_dir)} has not finished: {len(results)} of "
                         f"{runner.cfg.nann} annuli have a winner")
    if not getattr(runner.reducer, "supports_fm", False):
        raise ValueError(f"{type(runner.reducer).__name__} has no KLIP-FM, which the fit needs")
    items = [(r, runner._annulus_images(r.annulus)) for r in results]
    items = [(r, d) for r, d in items if d is not None and d.get("clean") is not None]
    if len(items) < len(results):
        raise ValueError("an annulus' winner images are missing (the run kept no FITS products?)")
    cleans = [d["clean"] for _, d in items]
    zones = [tuple(float(v) for v in d["zone"]) for _, d in items]
    edges = [z[0] for z in zones] + [zones[-1][1]]
    w = runner._stitch_weights(cleans, zones)
    pads = runner._tile_pads(zones)
    image = read_fits(os.path.join(runner.run_dir, "klip_stitched.fits"))
    if image is None:
        image = runner._maybe_flatten(stitch_tiles(cleans, zones, w, pads=pads))
    image = np.asarray(image, float)
    ny, nx = image.shape
    pa = chain_pa({n: nodes[n] for n in labels}) if line_pa is None else float(line_pa) % 180.0
    conv = getattr(runner.reducer, "angle_convention", "pa")
    fwhm, px = float(runner.fwhm), float(runner.pxscale)

    # each node's basis through every annulus' winner
    basis = []
    ia_keep = runner.ia
    try:
        for n in labels:
            tiles = []
            src = list(nodes[n])
            for r, _ in items:
                runner.ia = r.annulus
                cfg = runner._config_from_dict(r.winner_config)
                ev = runner._reduce(cfg, None, tag=f"chainfit_{n}_a{r.annulus + 1}", fm_sources=src)
                if ev.fm_image is None:
                    raise ValueError(f"annulus {r.annulus + 1}: the reduction returned no KLIP-FM image")
                tiles.append(np.asarray(ev.fm_image, float))
            basis.append(runner._maybe_flatten(stitch_tiles(tiles, zones, w, pads=pads)))
            log(f"  node {n}: {len(src)} source(s) through {len(items)} annulus winner(s)")
    finally:
        runner.ia = ia_keep
    basis = np.stack(basis)

    # the strip, the masks and the noise
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    span = _span(rr, edges, rho_range, px)
    strip = line_mask((ny, nx), [(pa, halfwidth_px)], angle_convention=conv)
    masked = np.zeros((ny, nx), bool)
    circles: List[Tuple[float, float, float, str]] = []
    for m in masks:
        rad = float(m[2]) if len(m) > 2 and m[2] is not None else 1.5 * fwhm
        circles.append((float(m[0]), float(m[1]), rad, "mask"))
    if mask_known:
        for k in runner._known():
            circles.append((float(k[0]), float(k[1]), 1.5 * fwhm, "known"))
    if candidate_tiers:
        for c in read_candidates(os.path.join(runner.run_dir, "cand", "candidates.txt")):
            if c["tier"] and c["tier"] in candidate_tiers:
                circles.append((c["rho"], c["pa"], 1.5 * fwhm, f"candidate {c['tier']}"))
    for rho, cpa, rad, _ in circles:
        x0, y0 = source_xy([rho], [cpa], px, cx, cy, conv)
        masked |= np.hypot(xx - x0[0], yy - y0[0]) <= rad
    finite = np.isfinite(image) & np.all(np.isfinite(basis), axis=0)
    fit = strip & span & finite & ~masked
    noise_px = ~strip & span & np.isfinite(image) & ~masked
    sigma = _radial_sigma(image, noise_px, rr, fwhm)
    fit &= np.isfinite(sigma) & (sigma > 0)
    k = len(labels)
    npix = int(fit.sum())
    if npix <= k:
        raise ValueError(f"{npix} pixels in the fit region for {k} node(s)")

    # noise-weighted least squares
    s = sigma[fit]
    A = basis[:, fit].T / s[:, None]
    b = image[fit] / s
    norms = np.sqrt(np.sum(A ** 2, axis=0))
    top = float(norms.max())
    dead = [n for n, v in zip(labels, norms) if not v > 1e-9 * top] if top > 0 else list(labels)
    if dead:
        lo, hi = float(rr[fit].min()) * px, float(rr[fit].max()) * px
        raise ValueError(f"node(s) {', '.join(dead)} put no signal in the fit region ({lo:.3f}-{hi:.3f} arcsec "
                         f"from the star): the run's annuli or the radial range do not reach them. Fit without "
                         f"them, or on a run whose annuli cover them")
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    resid = b - A @ coef
    chi2 = float(resid @ resid)
    beam = float(np.pi * (fwhm / 2.0) ** 2)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None) * beam)
    model = np.tensordot(coef, basis, axes=1)
    residual = image - model

    # the fitted chain, one line per position
    acc: Dict[Tuple[float, float], float] = {}
    for n, a_n in zip(labels, coef):
        for src in nodes[n]:
            key = (float(src.rho), float(src.theta))
            acc[key] = acc.get(key, 0.0) + float(a_n) * float(src.contrast)
    out_dir = out_dir or os.path.abspath(runner.run_dir).rstrip(os.sep) + "_chainfit"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chain_fitted.txt"), "w") as f:
        f.write(f"# fitted chain from {os.path.abspath(runner.run_dir)} (klip-tpe fit-chain)\n")
        f.write("# rho_arcsec  pa_deg  contrast\n")
        for (rho, cpa), c in acc.items():
            f.write(f"{rho!r} {cpa!r} {c!r}\n")
    hdr = {"RUNDIR": os.path.basename(os.path.abspath(runner.run_dir)), "LINEPA": pa, "HALFW": halfwidth_px,
           "NNODE": k, "NPIXFIT": npix}
    if rho_range is not None:
        hdr.update(RHOMIN=float(rho_range[0]), RHOMAX=float(rho_range[1]))
    write_fits(os.path.join(out_dir, "chainfit_model.fits"), model.astype(np.float32),
               dict(hdr, IMGTYPE="fitted chain, stitched KLIP-FM"))
    write_fits(os.path.join(out_dir, "chainfit_residual.fits"), residual.astype(np.float32),
               dict(hdr, IMGTYPE="final image minus the fitted chain"))
    write_fits(os.path.join(out_dir, "chainfit_basis.fits"), basis.astype(np.float32),
               dict(hdr, IMGTYPE="stitched KLIP-FM image per node", NODES=",".join(labels)))
    write_fits(os.path.join(out_dir, "chainfit_region.fits"), fit.astype(np.uint8),
               dict(hdr, IMGTYPE="pixels in the fit"))
    rec = {
        "run_dir": os.path.abspath(runner.run_dir),
        "nodes": [{"label": n, "value": float(a), "error": float(e), "error_unscaled": float(np.sqrt(max(c_, 0.0))),
                   "n_sources": len(nodes[n])} for n, a, e, c_ in zip(labels, coef, err, np.diag(cov))],
        "covariance": (cov * beam).tolist(),
        "line_pa_deg": pa, "halfwidth_px": float(halfwidth_px),
        "rho_range_arcsec": None if rho_range is None else [float(rho_range[0]), float(rho_range[1])],
        "annuli": [{"annulus": r.annulus + 1, "zone_px": list(z), "winner_eval": int(r.winner_index) + 1,
                    "stitch_weight": float(wi)} for (r, _), z, wi in zip(items, zones, w)],
        "pixels": npix, "chi2": chi2, "dof": npix - k, "chi2_per_dof": chi2 / max(npix - k, 1),
        "beam_px": beam, "error_note": "errors scaled by sqrt(pixels per beam), pi (FWHM/2)^2",
        "masks": [{"rho": c[0], "pa": c[1], "radius_px": c[2], "why": c[3]} for c in circles],
        "noise": "1.4826 x MAD of the final image outside the strip and the masks, in rings 1 FWHM wide",
        "image": "klip_stitched.fits", "seconds": round(time.time() - t0, 1),
    }
    if line_spec is not None:
        val = {n: (float(a_), float(e_)) for n, a_, e_ in zip(labels, coef, err)}
        try:
            a1 = [val[f"a1_{s:g}"] for s in line_spec["nodes"]]
            a2 = [val[f"a2_{s:g}"] for s in line_spec["nodes"]]
        except KeyError:
            raise ValueError("line_spec does not match the labels (line_basis makes them)") from None
        write_line_nodes(os.path.join(out_dir, "line_fitted.txt"), line_spec, [v[0] for v in a1], [v[0] for v in a2],
                         [v[1] for v in a1], [v[1] for v in a2],
                         comment=f"fitted to {os.path.abspath(runner.run_dir)} (klip-tpe fit-chain)")
        rec["line"] = {"pa": line_spec["pa"], "step": line_spec["step"], "rho_max": line_spec.get("rho_max"),
                       "nodes": list(line_spec["nodes"]), "a1": [v[0] for v in a1], "a2": [v[0] for v in a2],
                       "e1": [v[1] for v in a1], "e2": [v[1] for v in a2]}
    with open(os.path.join(out_dir, "chainfit.json"), "w") as f:
        json.dump(rec, f, indent=1)
    log(f"  fit: {npix} pixels within {halfwidth_px:g} px of the line at PA {pa:.2f} deg, chi2/dof "
        f"{rec['chi2_per_dof']:.2f}")
    for nd in rec["nodes"]:
        log(f"    node {nd['label']}: {nd['value']:.4g} +- {nd['error']:.2g}  ({nd['n_sources']} sources)")
    log(f"  wrote {out_dir}")
    return rec


def fit_per_night(runner, chain: Sequence[Source], *, line_pa: Optional[float] = None, halfwidth_px: float = 12.0,
                  masks: Sequence[Tuple[float, float, Optional[float]]] = (), mask_known: bool = True,
                  out_dir: Optional[str] = None, rho_range: Optional[Tuple[float, float]] = None,
                  log: Callable[[str], None] = print) -> Dict[str, Any]:
    """One amplitude per partition (night) for a chain whose shape is held fixed: does the line
    follow something that changes from night to night?

    The chain goes through every annulus' winner as the KLIP-FM model, as in :func:`fit_chain`,
    and each night's FM images and its own clean images (``best_clean_partitions.fits``) are
    stitched night by night with the final stitch's weights and tiles.  Each night's amplitude
    ``s`` minimizes ``sum w (clean - s FM)^2`` over the strip (inside ``rho_range`` when given,
    as in :func:`fit_chain`), with ``w`` from that night's scatter outside it.  1 means the
    night holds the chain as given.  Writes
    ``per_night.json`` under ``out_dir`` (default ``<run>_chainfit``) and returns it."""
    from .stitch import stitch_stacks
    results = [r for r in (runner.results or []) if r is not None]
    if not results or len(results) < runner.cfg.nann:
        raise ValueError(f"{os.path.basename(runner.run_dir)} has not finished")
    items = [(r, runner._annulus_images(r.annulus)) for r in results]
    if any(d is None or d.get("clean_stack") is None for _, d in items):
        raise ValueError("the run kept no per-partition images (best_clean_partitions.fits)")
    zones = [tuple(float(v) for v in d["zone"]) for _, d in items]
    edges = [z[0] for z in zones] + [zones[-1][1]]
    w = runner._stitch_weights([d["clean"] for _, d in items], zones)
    pads = runner._tile_pads(zones)
    fm_stacks, fm_ids = [], []
    ia_keep = runner.ia
    try:
        for r, _ in items:
            runner.ia = r.annulus
            ev = runner._reduce(runner._config_from_dict(r.winner_config), None,
                                tag=f"pernight_a{r.annulus + 1}", fm_sources=list(chain))
            if ev.fm_stack is None or not ev.partitions:
                raise ValueError("the reducer gives no per-partition KLIP-FM images (one partition only?)")
            fm_stacks.append(np.asarray(ev.fm_stack, float))
            fm_ids.append([str(p) for p in ev.partitions])
    finally:
        runner.ia = ia_keep
    fm, ids = stitch_stacks(fm_stacks, fm_ids, zones, w, pads=pads)
    cl, ids_c = stitch_stacks([np.asarray(d["clean_stack"], float) for _, d in items],
                              [d["parts"] for _, d in items], zones, w, pads=pads)
    if fm is None or cl is None:
        raise ValueError("nothing to stitch")
    ny, nx = fm.shape[-2:]
    pa = chain_pa({"chain": list(chain)}) if line_pa is None else float(line_pa) % 180.0
    conv = getattr(runner.reducer, "angle_convention", "pa")
    fwhm, px = float(runner.fwhm), float(runner.pxscale)
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    span = _span(rr, edges, rho_range, px)
    strip = line_mask((ny, nx), [(pa, halfwidth_px)], angle_convention=conv)
    masked = np.zeros((ny, nx), bool)
    circles = [(float(m[0]), float(m[1]), float(m[2]) if len(m) > 2 and m[2] is not None else 1.5 * fwhm) for m in masks]
    if mask_known:
        circles += [(float(k[0]), float(k[1]), 1.5 * fwhm) for k in runner._known()]
    for rho, cpa, rad in circles:
        x0, y0 = source_xy([rho], [cpa], px, cx, cy, conv)
        masked |= np.hypot(xx - x0[0], yy - y0[0]) <= rad
    beam = float(np.pi * (fwhm / 2.0) ** 2)
    rows = []
    for pid in ids:
        if pid not in ids_c:
            continue
        m_ = runner._maybe_flatten(fm[ids.index(pid)])
        c_ = runner._maybe_flatten(cl[ids_c.index(pid)])
        sig = _radial_sigma(c_, ~strip & span & np.isfinite(c_) & ~masked, rr, fwhm)
        fit = strip & span & np.isfinite(c_) & np.isfinite(m_) & ~masked & np.isfinite(sig) & (sig > 0)
        ww = 1.0 / sig[fit] ** 2
        den = float(np.sum(ww * m_[fit] ** 2))
        if den <= 0:
            continue
        s = float(np.sum(ww * c_[fit] * m_[fit]) / den)
        rows.append({"partition": pid, "amplitude": s, "error": float(np.sqrt(beam / den)), "pixels": int(fit.sum())})
        log(f"    {pid}: {s:.3f} +- {rows[-1]['error']:.3f}  ({rows[-1]['pixels']} px)")
    out_dir = out_dir or os.path.abspath(runner.run_dir).rstrip(os.sep) + "_chainfit"
    os.makedirs(out_dir, exist_ok=True)
    rec = {"run_dir": os.path.abspath(runner.run_dir), "line_pa_deg": pa, "halfwidth_px": float(halfwidth_px),
           "rho_range_arcsec": None if rho_range is None else [float(rho_range[0]), float(rho_range[1])],
           "n_sources": len(chain), "nights": rows,
           "note": "amplitude 1 = the chain as given; errors scaled by sqrt(pixels per beam)"}
    with open(os.path.join(out_dir, "per_night.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec
