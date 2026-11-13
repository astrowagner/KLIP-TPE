"""Stitching of the annuli's images, and the FITS and text products of a run (A §9).

Everything here is a pure function of arrays and numbers.  :class:`~klip_tpe.runner.Runner` decides
what to stitch (the previous annuli's winners, the current running best, the per-partition stacks)
and calls in here for how:

* :func:`tile_mask` and :func:`stitch_tiles`, the NaN-aware weighted mean over radial tiles
  ``edge_ia - pad <= r <= edge_ia+1 + pad``, with equal weights for the running stitch and the
  sensitivity weights of :func:`klip_tpe.products.stitch_annuli` for the final one.  A run pads
  only the edges where two annuli meet (``RunConfig.edge_padding``).
* :func:`stitch_stacks`, the same for per-partition stacks, onto the union of the partition ids.
* :func:`annulus_tiles` and :func:`open_ends`, the tiles of overlapping annuli and the tiles an
  image product is stitched over.
* :func:`tile_of`, :func:`tile_index` and :func:`stitch_by_tiles`, which tile holds a radius or a
  pixel, and maps put together tile by tile without blending, such as the annuli's own S/N maps.
* :func:`put_tile_in` and :func:`put_tile_in_stack`, the stitch with one annulus' tile put in.
* :func:`seam_pixels` and :func:`seam_trim`, the pixels of the searched span still NaN after
  stitching, and the contrast curve's seam trim.
* :func:`fits_header`, :func:`write_fits` and :func:`read_fits`, FITS files with provenance headers
  (``.fits`` or ``.fits.gz``).
* :func:`write_params_table`, :func:`write_contrast_curve` and :func:`write_setup_file`, the text
  products of the IDL predecessor (``klip_stitched_params.txt``, ``contrast_curve.txt``,
  ``evalNNNN_setup.txt``).
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .metrics import star_center

__all__ = ["tile_mask", "stitch_tiles", "stitch_stacks", "seam_pixels", "seam_trim", "annulus_tiles", "open_ends",
           "tile_of", "tile_index", "stitch_by_tiles", "put_tile_in", "put_tile_in_stack",
           "fits_header", "write_fits", "read_fits", "write_params_table", "write_contrast_curve",
           "write_setup_file", "PARAM_COLUMNS"]

PARAM_COLUMNS = ("k_klip", "bin", "n_ang", "filter", "angsep", "anglemax", "corr_thresh", "noise_max",
                 "coronoise_max")


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def _radius(shape) -> np.ndarray:
    """Return each pixel's distance from the star.

    Parameters
    ----------
    shape : tuple of 2 ints
        Shape of the image, ``(ny, nx)``.

    Returns
    -------
    r : ndarray, shape (ny, nx)
        The distance, in pixels.
    """
    ny, nx = shape
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.hypot(xx - cx, yy - cy)


def annulus_tiles(zones: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return each annulus' tile, the radii the stitch takes from it.

    Where two annuli next to each other overlap, the boundary between their tiles is the midpoint
    between their centers, kept within the overlap, so each radius comes from the annulus whose center
    is nearest.  Where they meet, or leave a gap, the tiles keep the zones' edges, so without overlaps
    the tiles are the zones.

    Parameters
    ----------
    zones : sequence of tuple of 2 floats
        Each annulus' zone, ``(rin, rout)`` in pixels, sorted by center.

    Returns
    -------
    tiles : list of tuple of 2 floats
        Each annulus' tile, in pixels.

    Notes
    -----
    Overlapping annuli come from ``--ann-extra`` (NEAR2 s6 note, section 5).
    """
    z = [(float(a), float(b)) for a, b in zones]
    tiles = [[a, b] for a, b in z]
    for k in range(len(z) - 1):
        (a0, b0), (a1, b1) = z[k], z[k + 1]
        if b0 > a1:
            m = min(max(0.25 * (a0 + b0 + a1 + b1), a1), b0)
            tiles[k][1] = m
            tiles[k + 1][0] = m
    return [(a, b) for a, b in tiles]


def open_ends(tiles: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return the tiles an image product is stitched over, opened in to the star and out past the frame.

    The stitch then takes the innermost and outermost annuli's images wherever they are finite, which
    is their whole KLIP zone with its pads and ``zone_in`` and ``zone_out``, since a reducer's image is
    NaN outside its zone.  A one-annulus run's stitch is then its winner's image.  The boundaries
    between annuli are kept.

    Parameters
    ----------
    tiles : sequence of tuple of 2 floats
        The scored tiles, in pixels.

    Returns
    -------
    tiles : list of tuple of 2 floats
        The tiles, the innermost from 0 and the outermost to infinity.

    Notes
    -----
    From the MWC 758 ALES session of 2026-11-07.
    """
    t = [(float(a), float(b)) for a, b in tiles]
    if not t:
        return t
    i_in = min(range(len(t)), key=lambda k: t[k][0])
    i_out = max(range(len(t)), key=lambda k: t[k][1])
    t[i_in] = (0.0, t[i_in][1])
    t[i_out] = (t[i_out][0], float("inf"))
    return t


def tile_mask(shape, rin: float, rout: float, pad: Any = 2.0) -> np.ndarray:
    """Return the mask of a tile, ``rin - pad_in <= r <= rout + pad_out``.

    Parameters
    ----------
    shape : tuple of 2 ints
        Shape of the image, ``(ny, nx)``.
    rin : float
        Inner radius, in pixels.
    rout : float
        Outer radius, in pixels.
    pad : float or tuple of 2 floats, default 2.0
        Pad of both edges, or ``(pad_in, pad_out)``, in pixels.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True inside the tile.
    """
    pin, pout = _pair(pad)
    rr = _radius(shape)
    return (rr >= rin - pin) & (rr <= rout + pout)


def _pair(pad: Any) -> Tuple[float, float]:
    """Return a pad as ``(pad_in, pad_out)``.

    Parameters
    ----------
    pad : float or sequence of 2 floats
        One pad for both edges, or the pair.

    Returns
    -------
    pad_in : float
        Pad of the inner edge.
    pad_out : float
        Pad of the outer edge.
    """
    if isinstance(pad, (tuple, list, np.ndarray)):
        return float(pad[0]), float(pad[1])
    return float(pad), float(pad)


def stitch_tiles(images: Sequence[Optional[np.ndarray]], zones: Sequence[Tuple[float, float]],
                 weights: Optional[Sequence[float]] = None, pad: float = 2.0,
                 pads: Optional[Sequence[Tuple[float, float]]] = None) -> np.ndarray:
    """Stitch images into their NaN-aware weighted mean over their tiles.

    Where padded tiles overlap, each pixel is the weighted mean of the images finite there.

    Parameters
    ----------
    images : sequence of ndarray or None
        One image per tile.  None is skipped, as is an image of another shape than the first.
    zones : sequence of tuple of 2 floats
        Each image's tile, ``(rin, rout)`` in pixels.
    weights : sequence of float or None, default None
        Weight of each image.  None weighs them equally.
    pad : float, default 2.0
        Pad of every edge, in pixels, when ``pads`` is None.
    pads : sequence of tuple of 2 floats or None, default None
        Each tile's ``(pad_in, pad_out)``, in pixels.  The runner pads only the edges between two annuli
        (``RunConfig.edge_padding``).

    Returns
    -------
    stitch : ndarray, shape (ny, nx)
        The stitched image, NaN where no image is finite.

    Raises
    ------
    ValueError
        If every image is None.
    """
    ref = next((im for im in images if im is not None), None)
    if ref is None:
        raise ValueError("no images to stitch")
    ny, nx = ref.shape
    w = np.ones(len(images)) if weights is None else np.asarray(weights, float)
    num = np.zeros((ny, nx))
    den = np.zeros((ny, nx))
    pads = [(pad, pad)] * len(images) if pads is None else list(pads)
    for im, (rin, rout), wi, pd in zip(images, zones, w, pads):
        if im is None or im.shape != (ny, nx):
            continue
        m = tile_mask((ny, nx), rin, rout, pd) & np.isfinite(im)
        num[m] += wi * im[m]
        den[m] += wi
    return num / np.where(den > 0, den, np.nan)


def stitch_stacks(stacks: Sequence[Optional[np.ndarray]], part_ids: Sequence[Optional[Sequence[Any]]],
                  zones: Sequence[Tuple[float, float]], weights: Optional[Sequence[float]] = None,
                  pad: float = 2.0, pads: Optional[Sequence[Tuple[float, float]]] = None
                  ) -> Tuple[Optional[np.ndarray], List[str]]:
    """Stitch per-partition stacks onto the union of their partition ids.

    Each partition's planes are stitched as :func:`stitch_tiles` stitches images.

    Parameters
    ----------
    stacks : sequence of ndarray or None
        Each annulus' per-partition images, ``(npart, ny, nx)``, None skipped.
    part_ids : sequence of sequence or None
        Each stack's partition ids, None skipped.
    zones : sequence of tuple of 2 floats
        Each annulus' tile, ``(rin, rout)`` in pixels.
    weights : sequence of float or None, default None
        Weight of each annulus.  None weighs them equally.
    pad : float, default 2.0
        Pad of every edge, in pixels, when ``pads`` is None.
    pads : sequence of tuple of 2 floats or None, default None
        Each tile's ``(pad_in, pad_out)``, in pixels.

    Returns
    -------
    cube : ndarray or None
        The stitched stack, ``(nunion, ny, nx)``, NaN where no annulus used a partition, None when
        nothing is available.
    ids : list of str
        The union of the ids, in the order they first appear, empty when nothing is available.
    """
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
    pads = [(pad, pad)] * len(stacks) if pads is None else list(pads)
    for st, ids, (rin, rout), wi, pd in zip(stacks, part_ids, zones, w, pads):
        if st is None or ids is None or st.shape[-2:] != (ny, nx):
            continue
        tile = tile_mask((ny, nx), rin, rout, pd)
        for j, p in enumerate(ids):
            if j >= st.shape[0]:
                break
            u = union.index(str(p))
            m = tile & np.isfinite(st[j])
            num[u][m] += wi * st[j][m]
            den[u][m] += wi
    return num / np.where(den > 0, den, np.nan), union


def tile_of(r_px: float, tiles: Sequence[Tuple[float, float]]) -> Optional[int]:
    """Return the index of the tile holding a radius, ``rin <= r < rout``, the last tile's outer edge included.

    Parameters
    ----------
    r_px : float
        The radius, in pixels.
    tiles : sequence of tuple of 2 floats
        The tiles, in pixels.

    Returns
    -------
    index : int or None
        The tile's index, None outside every tile.
    """
    r = float(r_px)
    for k, (a, b) in enumerate(tiles):
        if float(a) <= r < float(b):
            return k
    if len(tiles) and abs(r - float(tiles[-1][1])) < 1e-6:
        return len(tiles) - 1
    return None


def tile_index(shape, tiles: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Return each pixel's tile, as :func:`tile_of` gives it for the pixel's radius.

    Parameters
    ----------
    shape : tuple of 2 ints
        Shape of the image, ``(ny, nx)``.
    tiles : sequence of tuple of 2 floats
        The tiles, in pixels.

    Returns
    -------
    index : ndarray of int, shape (ny, nx)
        The tile's index, -1 outside every tile.
    """
    rr = _radius(shape)
    out = np.full(tuple(shape), -1, int)
    n = len(tiles)
    for k, (a, b) in enumerate(tiles):
        inside = (rr >= float(a)) & ((rr < float(b)) | ((k == n - 1) & (np.abs(rr - float(b)) < 1e-6)))
        out[inside & (out < 0)] = k
    return out


def stitch_by_tiles(images: Sequence[Optional[np.ndarray]], tiles: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Put maps together tile by tile, each pixel from the map of the tile holding it, without blending.

    It serves for maps that must not be averaged, such as the annuli's own S/N maps.

    Parameters
    ----------
    images : sequence of ndarray or None
        One map per tile.
    tiles : sequence of tuple of 2 floats
        The tiles, in pixels.

    Returns
    -------
    stitch : ndarray, shape (ny, nx)
        The map, NaN outside every tile, where a tile's map is missing and where it is NaN.

    Raises
    ------
    ValueError
        If every map is None.
    """
    ref = next((im for im in images if im is not None), None)
    if ref is None:
        raise ValueError("no images to stitch")
    shape = np.shape(ref)
    idx = tile_index(shape, tiles)
    out = np.full(shape, np.nan)
    for k, im in enumerate(images):
        if im is None or np.shape(im) != shape:
            continue
        m = idx == k
        out[m] = np.asarray(im, float)[m]
    return out


def put_tile_in(stitched: Optional[np.ndarray], image: Optional[np.ndarray], tile: np.ndarray
                ) -> Optional[np.ndarray]:
    """Return the stitch with one annulus' image put in on its tile.

    The result is ``image`` where ``tile`` holds and ``image`` is finite, and ``stitched`` everywhere
    else.  A matched filter and a noise ring near the tile's edge then take the neighbors' residuals,
    as on the stitch, where on ``image`` alone, NaN outside its zone, they take zeros.

    Parameters
    ----------
    stitched : ndarray or None
        The stitched image.
    image : ndarray or None
        The annulus' image.
    tile : ndarray of bool
        The tile's mask, such as ``tile_index(...) == k``.

    Returns
    -------
    image : ndarray or None
        The stitch with the tile put in, ``image`` alone when there is no stitch of its shape, None
        without ``image``.

    Notes
    -----
    From the NEAR2 note of 2026-11-04.
    """
    if image is None:
        return None
    img = np.asarray(image, float)
    if stitched is None or np.shape(stitched) != img.shape:
        return img
    out = np.array(stitched, dtype=float)
    take = np.asarray(tile, bool) & np.isfinite(img)
    out[take] = img[take]
    return out


def put_tile_in_stack(stitched: Optional[np.ndarray], stitched_ids: Optional[Sequence[Any]], stack: np.ndarray,
                      ids: Optional[Sequence[Any]], tile: np.ndarray) -> np.ndarray:
    """Put each plane of an annulus' per-partition stack into the stitched cube's plane of the same id.

    Each plane goes in as :func:`put_tile_in` puts an image in.  A plane whose id the stitched cube
    lacks, and every plane when either set of ids is unknown or the shapes disagree, stays as the
    annulus has it.

    Parameters
    ----------
    stitched : ndarray or None
        The stitched per-partition cube.
    stitched_ids : sequence or None
        The cube's partition ids.
    stack : ndarray
        The annulus' per-partition images, ``(n, ny, nx)`` or one ``(ny, nx)``.
    ids : sequence or None
        The stack's partition ids.
    tile : ndarray of bool
        The tile's mask.

    Returns
    -------
    stack : ndarray, shape (n, ny, nx)
        The planes with the tile put in.
    """
    st = np.asarray(stack, float)
    if st.ndim == 2:
        st = st[None]
    if stitched is None or ids is None or stitched_ids is None:
        return st
    cube = np.asarray(stitched, float)
    if cube.ndim == 2:
        cube = cube[None]
    if cube.shape[1:] != st.shape[1:] or len(stitched_ids) != cube.shape[0] or len(ids) != st.shape[0]:
        return st
    at = {str(p): k for k, p in enumerate(stitched_ids)}
    return np.stack([plane if str(p) not in at else put_tile_in(cube[at[str(p)]], plane, tile)
                     for p, plane in zip(ids, st)])


def seam_pixels(stitched: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """Return the mask of the pixels between two radii that are still NaN after stitching.

    Parameters
    ----------
    stitched : ndarray, shape (ny, nx)
        The stitched image.
    r0 : float
        Inner radius, in pixels.
    r1 : float
        Outer radius, in pixels.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True at the NaN pixels with ``r0 <= r <= r1``.
    """
    rr = _radius(stitched.shape)
    return (rr >= r0) & (rr <= r1) & ~np.isfinite(stitched)


def seam_trim(fwhm: float, pxscale: float) -> float:
    """Return the contrast curve's seam trim, ``max(round(fwhm / 3), 1) * pxscale``.

    Parameters
    ----------
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.

    Returns
    -------
    trim : float
        The trim, in arcsec.
    """
    return max(int(round(fwhm / 3.0)), 1) * pxscale


# ----------------------------------------------------------------------------
# FITS
# ----------------------------------------------------------------------------
def fits_header(d: Dict[str, Any], history: Sequence[str] = ()):
    """Build an ``astropy.io.fits.Header`` from a flat dict.

    Keys are upper-cased, with "-" as "_", and cut to 8 characters, which avoids HIERARCH so every
    reader can read the files.  Values FITS cannot hold become strings, strings are cut to 68
    characters, and None and non-finite floats are skipped.  ``PROG``, ``DATE`` and ``KTPEVER`` (the
    klip_tpe version) are always present.

    Parameters
    ----------
    d : dict
        The keys and values.
    history : sequence of str, default ()
        Lines of the HISTORY cards.

    Returns
    -------
    header : astropy.io.fits.Header
        The header.
    """
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
    """Write an array as float32 FITS with a :func:`fits_header`.

    A ``.fits.gz`` path is compressed by astropy, and the directory is made where needed.  Without
    astropy it saves a ``.npy`` file instead.

    Parameters
    ----------
    path : str
        Path of the file, ``.fits`` or ``.fits.gz``.
    data : ndarray
        The array.
    header : dict or None, default None
        Keys and values of the header.
    history : sequence of str, default ()
        Lines of the HISTORY cards.

    Returns
    -------
    path : str
        The path given, also without astropy.
    """
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
    """Read a FITS file's primary data as floats.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    data : ndarray or None
        The data, None when the file is missing or unreadable.
    """
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
    """Format a number, ``default`` where it is not finite or not a number.

    Parameters
    ----------
    v : object
        The value.
    fmt : str
        The format.
    default : str, default "nan"
        Text for a value that is not finite or not a number.

    Returns
    -------
    text : str
        The value formatted.
    """
    try:
        f = float(v)
        return fmt.format(f) if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _kept_median(r: Dict[str, Any]) -> Dict[str, Any]:
    """Return a table row's parameters, the columns of :data:`PARAM_COLUMNS` as medians over its kept blocks.

    The blocks are the per-partition blocks of the partitions the row keeps, all of them when it names
    none, and a column no block gives keeps the row's value.

    Parameters
    ----------
    r : dict
        The row, with ``params``, ``per_partition`` and ``partitions``.

    Returns
    -------
    params : dict
        The parameters.
    """
    p = dict(r.get("params") or {})
    per = r.get("per_partition") or {}
    keep = [str(v) for v in (r.get("partitions") or [])]
    blocks = [pp for pid, pp in per.items() if not keep or str(pid) in keep]
    for c in PARAM_COLUMNS:
        vals = []
        for b in blocks:
            try:
                v = float(b.get(c))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        if vals:
            p[c] = float(np.median(vals))
    return p


def write_params_table(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """Write ``klip_stitched_params.txt``, each annulus' representative parameters and its per-partition blocks.

    The representative value of each parameter is the median over the partitions the winner keeps of
    their per-partition blocks (:func:`_kept_median`), or the row's ``params`` where there are none.
    Values carry 6 significant digits, so they can be rounded again without the error of a second
    rounding.

    Parameters
    ----------
    path : str
        Path of the file.
    rows : sequence of dict
        One row per annulus, holding ``ann``, ``inner_px``, ``outer_px``, ``params``, ``best_SNR``,
        ``partitions`` and ``per_partition`` (partition id to parameters).  Optional keys are
        ``validated``, ``contrast``, and ``tile_in_px`` and ``tile_out_px`` for overlapping annuli
        (``RunConfig.ann_extra``), which add two tile columns after the zone's.

    Returns
    -------
    text : str
        The text written.
    """
    def g(v, w):
        return f"{_fmt(v, '{:.6g}'):>{w}}"
    tiles = any(r.get("tile_in_px") is not None for r in rows)
    L = ["# Per-annulus optimal parameters used for klip_stitched.fits",
         f"# klip_tpe {__version__} -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         "# inner/outer in pixels.  Representative values = the median over the partitions the winner keeps, "
         "6 significant digits."]
    if tiles:
        L.append("# inner/outer: each annulus' zone.  tile_in/tile_out: the radii the stitch takes from it "
                 "(nearest centre where annuli overlap).")
    L.append(f"{'ann':>4}  {'inner_px':>8}  {'outer_px':>8}  "
             + (f"{'tile_in':>8}  {'tile_out':>8}  " if tiles else "")
             + f"{'k':>6}  {'bin':>7}  {'n_ang':>7}  {'filter':>7}  "
             f"{'angsep':>9}  {'anglemax':>9}  {'corr':>9}  {'noise':>9}  {'coro':>9}  {'best_SNR':>9}  "
             f"{'valid':>5}  {'contrast':>10}  partitions")
    for r in rows:
        p = _kept_median(r)
        parts = r.get("partitions") or []
        tcol = (f"{_fmt(r.get('tile_in_px'), '{:8.2f}'):>8}  {_fmt(r.get('tile_out_px'), '{:8.2f}'):>8}  "
                if tiles else "")
        L.append(f"{int(r['ann']):4d}  {float(r['inner_px']):8.2f}  {float(r['outer_px']):8.2f}  {tcol}"
                 f"{g(p.get('k_klip'), 6)}  {g(p.get('bin'), 7)}  {g(p.get('n_ang'), 7)}  {g(p.get('filter'), 7)}  "
                 f"{g(p.get('angsep'), 9)}  {g(p.get('anglemax'), 9)}  {g(p.get('corr_thresh'), 9)}  "
                 f"{g(p.get('noise_max'), 9)}  {g(p.get('coronoise_max'), 9)}  {_fmt(r.get('best_SNR'), '{:9.3f}'):>9}  "
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
                    f"{_fmt(pp.get(c), '{:13.6g}'):>13}" for c in PARAM_COLUMNS))
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def _spans(nums: Sequence[int]) -> str:
    """Return integers as spans, such as ``[1, 2, 3, 5]`` as ``"1-3, 5"``.

    Parameters
    ----------
    nums : sequence of int
        The integers.

    Returns
    -------
    text : str
        The spans, sorted.
    """
    out, run = [], []
    for v in sorted(int(x) for x in nums):
        if run and v == run[-1] + 1:
            run.append(v)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [v]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ", ".join(out)


def write_contrast_curve(path: str, curves: Sequence[Optional[Dict[str, Any]]], edges_px: Sequence[float],
                         fwhm: float, pxscale: float, legacy: bool = False,
                         contrasts: Optional[Sequence[float]] = None,
                         fm_curves: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
                         grid_curves: Optional[Sequence[Optional[Dict[str, Any]]]] = None) -> Dict[str, np.ndarray]:
    """Write the run's ``contrast_curve.txt``, the annuli's curves joined at the seams, then their per-sample points.

    Each annulus' curve is trimmed at the seams by ``trim = max(round(fwhm / 3), 1) * pxscale``:
    annulus ``ia`` keeps ``r >= edge_ia + trim`` for ``ia > 0`` and ``r <= edge_ia+1 - trim`` for
    ``ia < nann - 1``.  The curves are joined and sorted by separation, followed by the per-sample
    ``5 * contrast / SNR`` points.  With ``fm_curves`` a ``# KLIP-FM`` cross-check section follows
    (A §6.4), two comment lines then ``sep_arcsec  contrast_5sig`` rows over all annuli, sorted by
    separation and not trimmed, since the test radii lie inside the zones.

    Parameters
    ----------
    path : str
        Path of the file.
    curves : sequence of dict or None
        Each annulus' ``AnnulusResult.contrast_curve`` (``r_as``, ``curve``, ``sample_r``,
        ``sample_c``), None skipped.
    edges_px : sequence of float
        The annuli's edges, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    legacy : bool, default False
        Whether to leave the curves untrimmed, as ``legacy_stitch`` does.
    contrasts : sequence of float or None, default None
        Injection contrast of each annulus, for the header.
    fm_curves : sequence of dict or None, default None
        Each annulus' ``AnnulusResult.fm_curve``, for the KLIP-FM section.
    grid_curves : sequence of dict or None, default None
        Each annulus' ``AnnulusResult.grid_curve`` (one injection per reduction), which takes the main
        section's place annulus by annulus where present.  When only some annuli have one, the header
        names the annuli of each method.

    Returns
    -------
    curve : dict of str to ndarray
        The joined ``r_as`` and ``curve``, ``sample_r`` and ``sample_c``, and ``fm_r`` and ``fm_c``
        where a KLIP-FM section was written.
    """
    nann = len(curves)
    trim = 0.0 if legacy else seam_trim(fwhm, pxscale)
    grid = list(grid_curves or [])
    use_grid = any(g for g in grid)
    R, C, SR, SC, SA = [], [], [], [], []
    by_method: Dict[str, List[int]] = {"grid": [], "validation": []}
    for ia, cc in enumerate(curves):
        main = grid[ia] if (use_grid and ia < len(grid) and grid[ia]) else cc
        if cc:
            for a, b in zip(cc.get("sample_r") or [], cc.get("sample_c") or []):
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    SR.append(float(a)); SC.append(float(b)); SA.append(ia + 1)
        if not main or main.get("r_as") is None or main.get("curve") is None:
            continue
        by_method["grid" if main is not cc else "validation"].append(ia + 1)
        cc = main
        r = np.array([np.nan if v is None else float(v) for v in cc["r_as"]])
        c = np.array([np.nan if v is None else float(v) for v in cc["curve"]])
        keep = np.isfinite(r) & np.isfinite(c)
        if ia > 0 and ia < len(edges_px):
            keep &= r >= edges_px[ia] * pxscale + trim
        if ia < nann - 1 and ia + 1 < len(edges_px):
            keep &= r <= edges_px[ia + 1] * pxscale - trim
        R.extend(r[keep]); C.extend(c[keep])
    R, C = np.asarray(R, float), np.asarray(C, float)
    o = np.argsort(R, kind="stable")
    if use_grid and by_method["validation"]:
        head = ("# SNR=5 contrast curve from two methods.  Injection grid (one injection per reduction at the winner "
                f"config, median over position angle): annuli {_spans(by_method['grid'])}.  Validation injections "
                f"(no grid curve): annuli {_spans(by_method['validation'])}")
    elif use_grid:
        head = ("# Injection-recovery SNR=5 contrast curve (one injection per reduction at each annulus' winner config, "
                "median over position angle)")
    else:
        head = "# Injection-calibrated SNR=5 contrast curve (validated winners per annulus)"
    L = [head,
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
    """Write one step's setup file, in the layout of the IDL predecessor's ``near2m_write_setup`` (A §6.3).

    The file reads::

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

    A partitioned run's file ends with a ``# PER-PARTITION block`` table.  It never raises, since a
    text product must not stop a run.

    Parameters
    ----------
    path : str
        Path of the file, whose directory is made where needed.
    phase : str
        Phase of the step, such as "eval".
    annulus : int
        The annulus, counted from 0 and written from 1.
    step : int
        The step.
    params : dict
        The step's parameters.
    inrad : float
        Inner radius of the zone, in pixels.
    outrad : float
        Outer radius of the zone, in pixels.
    partitions : sequence, default ()
        The partitions kept, all when empty.
    sources : sequence of sequence of float, default ()
        The injected sources, as ``(rho_arcsec, theta_deg)`` or ``(rho_arcsec, theta_deg, contrast)``.
    contrast : float or None, default None
        Contrast of the sources that give none.  None writes 0.
    snr : float or None, default None
        The step's median S/N.
    per_partition : dict or None, default None
        Each partition's parameters, for the per-partition block.
    k_used : dict or None, default None
        Each partition's KL modes, written as a ``per-partition k*`` line.  Other values are not written.
    weights : sequence of float or None, default None
        The per-partition combine weights, written under the block.
    extra : dict or None, default None
        More lines, as name and value.

    Returns
    -------
    text : str
        The text written, empty on a failure.
    """
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
