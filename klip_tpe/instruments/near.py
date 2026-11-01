"""VLT/VISIR NEAR (AGPM N-band) adapter -- the data layout and injection model of
``reduce_near_2.pro`` / ``near2_n4psf.pro``, wired onto the generic
:class:`~klip_tpe.reducer.KLIPReducer`.

Per night ``n<seq>/`` under ``root``::

    AB_cube_skysub_cen_clean.fits   [nf, 432, 432]  cleaned, centred frames (star at 215.5, 215.5)
    AB_parang_clean.sav             angles (deg)
    AB_frametags.sav                corrs, noises, coronoise      (optional)
    texp.sav                        texp (s)                      (optional)
    PSF_ACenB.fits                  16x16 B-star template, centre (7.5, 7.5)
    AB_cube_refstar_clean.fits      pre-registered PSF-reference-star frames (optional, RDI/ARDI)
    AB_truenorth_clean.txt          each frame's true north (deg), one number per line, ``#``
                                    comments (optional; :attr:`NEAR.truenorth` for every frame
                                    without it)
root/psflib/n4_psf_cube_EEnorm.fits    measured off-axis AGPM PSF library [14, 41, 41]
root/psflib/throughput_merged.csv      EE(1 lambda/D) throughput vs separation

Frames are cropped to 150x150 (``+-75`` px about the star) at load, as the IDL
reduction does before KLIP; all separations here are in the cropped frame.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..feasibility import FrameSelectionGuard, ReferenceCountGuard, compose
from ..injection import LibraryPSF, TemplatePSF, shift_bilinear
from ..metrics import MawetPeakSNR, Objective
from ..positions import PositionSampler
from ..klip import rotate_ccw
from ..reducer import Dataset, KLIPReducer, PartitionedReducer, reducer_class
from ..space import Param, SearchSpace, kgrid
# the production search space is every instrument's (generic and ales build theirs with it), so
# it lives in search_space; near.make_space and near.production_defaults are the same objects
from .search_space import make_space, pa_span_max, production_defaults  # noqa: E402,F401

__all__ = ["NEAR", "load_night", "N4Library", "make_reducer", "make_space", "make_guard",
           "default_config", "production_defaults", "legacy_masks", "near_constants", "NORTH_FILE",
           "NorthFileError", "read_north"]


class NEAR:
    """Instrument constants.  The pixel constants follow ``pxscale``: a new run takes this
    class's, and a run keeps the scale it was made with (:func:`near_constants`,
    ``PartitionedReducer.provenance``)."""
    #: arcsec / px, from the astrometry of the 2026 cubes (NEAR2 note, 2026-10-31): 0.04450
    #: +/- 0.00003 near the star.  Runs before 2026-11-01 used :attr:`legacy_pxscale`.
    pxscale = 0.04450
    legacy_pxscale = 0.0456          # the scale of every run made before 2026-11-01
    lam_klip = 11.25e-6              # m, used for lambda/D (angsep unit, binning rule)
    lam_metric = 11.0e-6             # m, the optimizer's FWHM (6.24 px)
    diam = 8.2                       # m
    truenorth = 36.5                 # deg, added to the parallactic angle at derotation, for every
                                     # frame of a night without a north file (NORTH_FILE)
    hsize = 216                      # full-frame half size (432 px)
    shsize = 75                      # KLIP crop half size (150 px)
    n4_refpa = -93.9                 # deg, distortion axis of the PSF library
    b_to_a = 129.0 / 57.0            # contrast is relative to alpha Cen A; template is B
    lam_over_d_px = (lam_klip / diam) * 206265.0 / pxscale          # 6.359 (6.206 at 0.0456)
    fwhm_px = 1.028 * (lam_metric / diam) * 206265.0 / pxscale       # 6.392 (6.238)
    fwhm_bin_px = 1.028 * lam_over_d_px                              # 6.537 (6.380)


def near_constants(pxscale: Optional[float] = None) -> Dict[str, float]:
    """NEAR's pixel constants at a plate scale (arcsec/px), by :class:`NEAR`'s own formulas, so
    the scale a run was made with gives it its constants to the bit.  None:
    :attr:`NEAR.pxscale`."""
    ps = float(NEAR.pxscale if pxscale is None else pxscale)
    lod = (NEAR.lam_klip / NEAR.diam) * 206265.0 / ps
    return {"pxscale": ps, "lam_over_d_px": lod,
            "fwhm_px": 1.028 * (NEAR.lam_metric / NEAR.diam) * 206265.0 / ps,
            "fwhm_bin_px": 1.028 * lod}


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def _readsav(path: str) -> Dict[str, Any]:
    from scipy.io import readsav
    return readsav(path)


class DropFramesError(ValueError):
    """A ``drop_frames`` list that cannot be applied: a night not loaded, an index past the
    night's cube, or every frame of a night."""


#: each frame's true north (deg) in a night's directory, from the astrometry of the cubes: one
#: number per line in the order of the cube's frames, ``#`` comments
NORTH_FILE = "AB_truenorth_clean.txt"


class NorthFileError(ValueError):
    """A north file that cannot be used: a line that is not one number, a count that is not
    the cube's, or not the file a run was made with."""


def read_north(path: str, n: Optional[int] = None, night: str = "") -> np.ndarray:
    """A night's true north per frame (deg) from ``path`` (:data:`NORTH_FILE`): one number per
    line, ``#`` and what follows it a comment, blank lines skipped.  ``n``: the cube's frame
    count, which the file must match."""
    vals = []
    tag = f"{night}: " if night else ""
    with open(path) as f:
        for k, line in enumerate(f, 1):
            t = line.split("#", 1)[0].strip()
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                raise NorthFileError(f"{tag}line {k} of {path} is not one number: {line.strip()!r}") from None
    v = np.asarray(vals, float)
    if not np.all(np.isfinite(v)):
        raise NorthFileError(f"{tag}{path} holds a value that is not finite")
    if n is not None and v.size != int(n):
        raise NorthFileError(f"{tag}{path} holds {v.size} values for the cube's {int(n)} frames")
    return v


def _sha1(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _north_choice(north) -> Tuple[str, Optional[str]]:
    """``(mode, sha1)`` from ``load_night``'s ``north``: ``"file"`` or ``"constant"``, and the
    recorded file's SHA-1 when a run's record asks for it."""
    if isinstance(north, dict):
        src = str(north.get("source", "constant"))
        return ("file", north.get("sha1")) if src == "file" else ("constant", None)
    m = str(north or "file").lower()
    if m in ("file", "files"):
        return "file", None
    if m == "constant":
        return "constant", None
    raise ValueError(f"north={north!r}: 'files', 'constant' or a run's record of a night")


def _drop_spec(drop: Optional[Dict[Any, Sequence[int]]]) -> Dict[str, List[int]]:
    """``drop_frames`` as ``{"n4": [551], ...}``, in night order.  ``4``, ``"4"`` and ``"n4"``
    all name night 4.  Each night's indices are sorted and unique, and a night with none is
    left out."""
    out: Dict[str, set] = {}
    for k, v in (drop or {}).items():
        s = str(k).strip().lower()
        s = s[1:] if s.startswith("n") else s
        if not s.isdigit():
            raise DropFramesError(f"{k!r} is not a night (n4, or 4)")
        idx = set()
        for i in ([v] if isinstance(v, (int, np.integer)) else (v or ())):
            if float(i) != int(i):
                raise DropFramesError(f"n{int(s)} lists {i!r}, and a frame index is a whole number")
            if int(i) < 0:
                raise DropFramesError(f"n{int(s)} lists {int(i)}, and the indices count from 0 in the night's cube")
            idx.add(int(i))
        if idx:
            out.setdefault(f"n{int(s)}", set()).update(idx)
    return {k: sorted(out[k]) for k in sorted(out, key=lambda x: int(x[1:]))}


def _frame_plan(n: int, frames: Optional[slice], drop: Optional[Sequence[int]], night: str):
    """Which of a night's ``n`` frames to load: the ``frames`` slice of them, less the indices
    in ``drop`` (counted from 0 in the whole cube).  Returns the indices kept, the mask of the
    dropped ones over the slice (None when none is dropped) and the dropped indices.  An index
    past the cube is refused, and so is a list that would leave no frame.  An index the slice
    leaves out anyway (``--max-frames``) drops nothing."""
    idx = np.arange(n)[slice(0, n) if frames is None else frames]
    want = sorted({int(i) for i in (drop or ())})
    if not want:
        return idx, None, []
    bad = [i for i in want if not 0 <= i < n]
    if bad:
        raise DropFramesError(f"{night}: frame{'s' if len(bad) > 1 else ''} {' '.join(str(i) for i in bad)} "
                              f"{'are' if len(bad) > 1 else 'is'} not in the night's cube, which holds frames "
                              f"0-{n - 1}")
    hit = np.isin(idx, want)
    if hit.all():
        raise DropFramesError(f"{night}: dropping frame{'s' if len(want) > 1 else ''} "
                              f"{' '.join(str(i) for i in want)} would leave no frame")
    if not hit.any():
        return idx, None, []
    return idx[~hit], hit, [int(i) for i in idx[hit]]


def _pick(values, sl, hit: Optional[np.ndarray]) -> np.ndarray:
    """A per-frame array, sliced as the cube is and less the dropped frames.  An array whose
    length does not match the slice is left as it is, and the length checks downstream say so."""
    a = np.asarray(values, float)[sl]
    return a if hit is None or a.shape[0] != hit.shape[0] else a[~hit]


def load_night(root: str, seq: int, crop_half: int = NEAR.shsize, frames: Optional[slice] = None,
               cube_name: Optional[str] = None, drop_frames: Optional[Sequence[int]] = None,
               north: Any = "file") -> Dataset:
    """Load one night as a cropped :class:`Dataset` (float32).  ``frames`` may
    subsample the sequence for quick tests.  ``cube_name`` defaults to the full
    cleaned cube, falling back to ``AB_cube_crop150.fits`` (a pre-cropped copy).

    ``drop_frames``: frames left out, by their index in the night's cube counted from 0, e.g.
    ``[551]`` for n4's last frame, whose parallactic angle is wrong.  They go before anything
    else sees them: the angles, the frame tags, and so the frame cut, the binning, the
    injections and KLIP-FM.  ``texp`` scales with the frames kept.  ``meta["frame_index"]``
    holds each loaded frame's index in the cube and ``meta["dropped_frames"]`` the indices
    dropped.  The reference star's cube is not touched.

    ``north``: ``"file"`` reads each frame's true north from :data:`NORTH_FILE` when the night
    has one (``Dataset.north``, sliced and dropped as the angles are), and leaves the frames
    on :attr:`NEAR.truenorth` when it has none.  ``"constant"``: :attr:`NEAR.truenorth` for
    every frame.  A run's record of the night (``{"source": "file", "sha1": ...}``, from
    ``PartitionedReducer.provenance``): the file it was made with, refused when it is gone or
    has changed (:class:`NorthFileError`), or the constant.  ``meta["north"]`` says which."""
    from astropy.io import fits
    d = os.path.join(root, f"n{seq}")
    if cube_name is None:
        cube_name = "AB_cube_skysub_cen_clean.fits"
        if not os.path.exists(os.path.join(d, cube_name)) and os.path.exists(os.path.join(d, "AB_cube_crop150.fits")):
            cube_name = "AB_cube_crop150.fits"      # pre-cropped copy (crop150.py)
    with fits.open(os.path.join(d, cube_name), memmap=True) as h:
        data = h[0].data
        n = data.shape[0]
        sl = slice(0, n) if frames is None else frames
        keep, hit, dropped = _frame_plan(n, sl, drop_frames, f"n{seq}")
        ny, nx = data.shape[1], data.shape[2]
        cy0, cx0 = ny // 2, nx // 2
        y0, y1 = cy0 - crop_half, cy0 + crop_half
        x0, x1 = cx0 - crop_half, cx0 + crop_half
        # the kept frames only, so a dropped frame is never read
        cube = np.ascontiguousarray(data[sl if hit is None else keep, y0:y1, x0:x1]).astype(np.float32)
    angles = _pick(_readsav(os.path.join(d, "AB_parang_clean.sav"))["angles"], sl, hit)
    mode, want_sha = _north_choice(north)
    nf = os.path.join(d, NORTH_FILE)
    tn, tn_meta = None, {"source": "constant"}
    if mode == "file" and (os.path.exists(nf) or want_sha):
        if not os.path.exists(nf):
            raise NorthFileError(f"n{seq}: the run was made with {nf}, which is not there")
        sha = _sha1(nf)
        if want_sha and sha != want_sha:
            raise NorthFileError(f"n{seq}: {nf} is not the file the run was made with (SHA-1 {want_sha[:12]} "
                                 f"recorded, {sha[:12]} now)")
        tn = _pick(read_north(nf, n, f"n{seq}"), sl, hit)
        tn_meta = {"source": "file", "file": NORTH_FILE, "path": os.path.abspath(nf), "sha1": sha, "n_file": int(n)}
    elif mode == "file":
        tn_meta["why"] = f"no {NORTH_FILE}"
    else:
        tn_meta["why"] = "--north constant" if not isinstance(north, dict) else "as the run was made"
    tags = None
    tf = os.path.join(d, "AB_frametags.sav")
    if os.path.exists(tf):
        t = _readsav(tf)
        tags = {k: _pick(t[k], sl, hit) for k in ("corrs", "noises", "coronoise") if k in t}
        if tags.get("corrs") is None or tags["corrs"].size != cube.shape[0]:
            tags = None
    texp = 0.0
    xf = os.path.join(d, "texp.sav")
    if os.path.exists(xf):
        texp = float(np.asarray(_readsav(xf)["texp"]).ravel()[0])
        if cube.shape[0] != n:
            texp *= cube.shape[0] / max(n, 1)
    psf = None
    pf = os.path.join(d, "PSF_ACenB.fits")
    if os.path.exists(pf):
        psf = np.asarray(fits.getdata(pf), float)
    ref_cube = None
    rf = os.path.join(d, "AB_cube_refstar_clean.fits")
    if os.path.exists(rf):
        # RDI / ARDI reference star (reduce_near_2 L1860-1866): must be at least the science
        # size; centre-cropped like the science cube.  Never frame-subsampled.
        with fits.open(rf, memmap=True) as h:
            rd = h[0].data
            if rd is not None and rd.ndim == 3 and rd.shape[1] >= 2 * crop_half and rd.shape[2] >= 2 * crop_half:
                ry0, rx0 = rd.shape[1] // 2, rd.shape[2] // 2
                ref_cube = np.ascontiguousarray(rd[:, ry0 - crop_half:ry0 + crop_half,
                                                   rx0 - crop_half:rx0 + crop_half]).astype(np.float32)
    return Dataset(cube, angles, tags, texp=texp, name=f"n{seq}",
                   meta={"psf_template": psf, "crop_half": crop_half, "nframes_total": n, "root": root,
                         "ref_file": rf if ref_cube is not None else None,
                         "frame_index": keep, "dropped_frames": dropped, "north": tn_meta},
                   ref_cube=ref_cube, north=tn)


# ----------------------------------------------------------------------------
# legacy NaN masks (reduce_near_2 L2147-2172; block_burn / block_airy, default off)
# ----------------------------------------------------------------------------
def legacy_masks(aa: int = 0, bb: int = 0, ab: int = 0, ba: int = 0, block_airy: bool = False,
                 truenorth: float = NEAR.truenorth, shsize: int = NEAR.shsize,
                 hsize: int = NEAR.hsize) -> Callable[[np.ndarray], np.ndarray]:
    """``mask_fn`` for :class:`~klip_tpe.reducer.KLIPReducer` reproducing the two
    legacy masks of the final image.

    ``block_burn`` (any of ``aa, bb, ab, ba > 0``, L2153-2159): rotate the image so the
    persistence streak is vertical (``fang = 54 - truenorth`` deg, IDL
    ``rot(img, -fang)`` = CCW by ``fang``), NaN the column band
    ``x in [shsize-aa, shsize+ab-1]`` for ``y <= shsize`` and ``x in [shsize-ba,
    shsize+bb-1]`` for ``y >= shsize``, rotate back (the double bilinear rotation
    of the whole image is part of the legacy behaviour).
    ``block_airy`` (L2162-2172): NaN two 85-px disks centred at the full-frame
    positions (57, 86) and (244, 215) expressed in the cropped frame.
    """
    fang = 54.0 - float(truenorth)
    burn = any(v > 0 for v in (aa, bb, ab, ba))

    def fn(img: np.ndarray) -> np.ndarray:
        out = np.asarray(img, float).copy()
        ny, nx = out.shape
        if burn:
            out = rotate_ccw(out, fang)
            if aa > 0 or ab > 0:
                out[0:min(shsize + 1, ny), max(shsize - aa, 0):min(shsize + ab, nx)] = np.nan
            if bb > 0 or ba > 0:
                out[shsize:min(2 * shsize, ny), max(shsize - ba, 0):min(shsize + bb, nx)] = np.nan
            out = rotate_ccw(out, -fang)
        if block_airy:
            yy, xx = np.mgrid[0:ny, 0:nx]
            for (xc_full, yc_full) in ((57.0, 86.0), (244.0, 215.0)):
                xc, yc = shsize + xc_full - hsize, shsize + yc_full - hsize
                out[np.hypot(xx - xc, yy - yc) <= 85.0] = np.nan
        return out.astype(np.float32)
    return fn


# ----------------------------------------------------------------------------
# injection model
# ----------------------------------------------------------------------------
def _library_separations(psflib: str, hdr, n: int) -> Tuple[Optional[List[float]], str]:
    """Each of a library's ``n`` slices' separation (arcsec) and where it came from: the v2
    index ``n4_psf_cube_index.csv`` (column ``sep_arcsec``, ordered by ``slice``), else the
    header's ``SEP00``.. keywords, else None (the caller's 75-mas grid).

    The v1 library's index has a ``sep_arcsec`` column too, with other values that klip-tpe
    never read, so reading it would change every run made with v1.  Only an index of the v2
    format, which records v1's grid beside the new values (``v1_grid_mas``), is read."""
    idx = os.path.join(psflib, "n4_psf_cube_index.csv")
    if os.path.exists(idx):
        with open(idx) as f:
            reader = csv.DictReader(f)
            v2 = "v1_grid_mas" in (reader.fieldnames or [])
            rows = [r for r in reader if (r.get("sep_arcsec") or "").strip()]
        if v2:
            if "slice" in (rows[0] if rows else {}):
                rows.sort(key=lambda r: int(float(r["slice"])))
            if len(rows) != n:
                raise ValueError(f"{idx}: {len(rows)} rows with sep_arcsec for {n} slices")
            return [float(r["sep_arcsec"]) for r in rows], "n4_psf_cube_index.csv"
    keys = [f"SEP{k:02d}" for k in range(n)]
    if all(k in hdr for k in keys):
        return [float(hdr[k]) for k in keys], "header SEPnn"
    return None, "75-mas grid (v1)"


def _throughput_table(psflib: str) -> Optional[Callable]:
    p = os.path.join(psflib, "throughput_merged.csv")
    if not os.path.exists(p):
        return None
    seps, thr = [], []
    with open(p) as f:
        for row in csv.reader(f):
            try:
                seps.append(float(row[0]))
                thr.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    seps, thr = np.array(seps), np.array(thr)
    o = np.argsort(seps)
    seps, thr = seps[o], thr[o]

    def fn(r):
        r = np.asarray(r, float)
        out = np.interp(r, seps, thr, right=1.0)
        return np.clip(out, 0.0, 1.0)
    return fn


def analytic_throughput(r):
    """Fallback ``T(r) = 1 - exp(-(r/0.742)^1.73)`` (AGPM-N4 EE within 1 lambda/D)."""
    r = np.asarray(r, float)
    return np.clip(1.0 - np.exp(-(np.maximum(r, 0) / 0.742) ** 1.73), 0.0, 1.0)


def maire_agpm_transmission(r):
    """Digitised Maire et al. (2020) AGPM-N4 curve (legacy ``near2_agpm_trans``)."""
    rr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 6.0]
    tt = [0.00, 0.05, 0.18, 0.337, 0.438, 0.513, 0.599, 0.734, 0.810, 0.86, 0.898, 0.94, 0.97, 1.00]
    return np.clip(np.interp(np.asarray(r, float), rr, tt), 0, 1)


class N4Library:
    """Measured off-axis AGPM-N4 PSF library + throughput (``near2_n4psf.pro``)."""

    def __init__(self, psflib: str):
        from astropy.io import fits
        self.dir = psflib
        with fits.open(os.path.join(psflib, "n4_psf_cube_EEnorm.fits")) as h:
            cube = np.asarray(h[0].data, float)          # (14, 41, 41)
            hdr = h[0].header
        self.platescale = float(hdr.get("PLATESCL", 0.0453))
        self.lambdad_mas = float(hdr.get("LAMBDAD", 283.0))
        self.rap = (self.lambdad_mas / 1000.0) / self.platescale    # 6.25 library px
        # slices 12 and 13 share the final offset -> average them
        sl = list(cube[:12]) + [0.5 * (cube[12] + cube[13])] if cube.shape[0] >= 14 else list(cube)
        stamps = []
        for s in sl:
            st = shift_bilinear(np.where(np.isfinite(s), s, 0.0), -0.5, -0.5)[:40, :40]   # star (20,20)->(19.5,19.5)
            stamps.append(st)
        # each slice's separation: the library's index, else its header, else the 75-mas grid
        # of v1 (0.150..1.050", which put the slices from the 4th on one step too far out)
        per, self.sep_source = _library_separations(psflib, hdr, cube.shape[0])
        if per is None:
            self.seps = (np.arange(len(stamps)) + 2) * 0.075
        else:
            per = list(per)
            self.seps = np.asarray(per[:12] + [0.5 * (per[12] + per[13])] if cube.shape[0] >= 14 else per, float)
        if np.any(np.diff(self.seps) <= 0):
            raise ValueError(f"{psflib}: the slices' separations do not increase ({self.sep_source}): {self.seps}")
        self.stamps = np.array(stamps)
        self.throughput = _throughput_table(psflib) or analytic_throughput

    def describe(self) -> Dict[str, Any]:
        """What a run records of the library: its path, the stamps' separations and their
        source."""
        return {"path": os.path.abspath(self.dir), "n_stamps": int(len(self.seps)),
                "seps_arcsec": [round(float(v), 4) for v in self.seps], "sep_source": self.sep_source,
                "platescale": self.platescale, "throughput": "throughput_merged.csv"
                if os.path.exists(os.path.join(self.dir, "throughput_merged.csv")) else "analytic"}

    def model(self, flux_unit: float = 1.0) -> LibraryPSF:
        return LibraryPSF(self.stamps, self.seps, center=(19.5, 19.5), ee_radius_px=self.rap,
                          throughput_fn=self.throughput, refpa_deg=NEAR.n4_refpa, flux_unit=flux_unit)


def _north_text(rec: Dict[str, Any]) -> str:
    """A night's true north for the log, from :meth:`KLIPReducer.north_record`."""
    if rec.get("source") == "file":
        same = rec["min"] == rec["max"]
        span = f"{rec['min']:.3f} deg for every frame" if same else \
            f"{rec['min']:.3f}-{rec['max']:.3f} deg (median {rec['median']:.3f})"
        return f"true north per frame from {rec.get('file', NORTH_FILE)}, {span}"
    why = rec.get("why")
    return f"true north {float(rec.get('deg', NEAR.truenorth)):g} deg for every frame" + (f" ({why})" if why else "")


def template_flux_unit(psf_template: np.ndarray, lam_over_d_px: Optional[float] = None) -> float:
    """Flux unit for a contrast relative to alpha Cen A: ``(129/57) * EE_B(1 lambda/D)`` of
    the B-star template (16x16, centre 7.5, 7.5), within ``lam_over_d_px`` (default
    :attr:`NEAR.lam_over_d_px`; the run's own, from :func:`near_constants`)."""
    t = np.where(np.isfinite(psf_template), psf_template, 0.0)
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    c = ((t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0)
    m = np.hypot(xx - c[0], yy - c[1]) <= (NEAR.lam_over_d_px if lam_over_d_px is None else float(lam_over_d_px))
    return NEAR.b_to_a * float(t[m].sum())


# ----------------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------------


def make_reducer(root: str, nights: Sequence[int] = (1, 2, 3, 4, 5, 6), frames: Optional[slice] = None,
                 use_library: bool = True, weighting: str = "equal", max_workers="auto", pool: str = "auto",
                 fast: bool = False, n_min_ref: int = 10, datasets: Optional[Dict[int, Dataset]] = None,
                 log: Callable[[str], None] = print, use_rdi: bool = False, rdi_mode: str = "ardi",
                 do_destripe: bool = False, mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 backend: str = "klip", cube_name: Optional[str] = None,
                 subtract_sources: Optional[Sequence[Any]] = None,
                 inject_sources: Optional[Sequence[Any]] = None,
                 drop_frames: Optional[Dict[Any, Sequence[int]]] = None,
                 pxscale: Optional[float] = None, psflib: Optional[str] = None,
                 north: Any = "files") -> PartitionedReducer:
    """Build the per-night :class:`PartitionedReducer` for the NEAR campaign.
    ``backend``: ``'klip'`` (built-in) | ``'pyklip'`` | ``'vip'`` PSF subtraction.
    ``use_rdi``/``rdi_mode`` use ``n<seq>/AB_cube_refstar_clean.fits`` when present
    (nights without one fall back to ADI); ``mask_fn`` e.g. :func:`legacy_masks`.
    ``cube_name``: the cube file read in each night's directory (:func:`load_night`), e.g. a
    reduction's line-subtracted ``AB_cube_skysub_cen_clean_abline.fits``.
    ``subtract_sources``: point sources ``(rho_arcsec, pa_deg, contrast)`` taken out of every
    night's frames once, at load (:meth:`~klip_tpe.reducer.KLIPReducer.subtract_sources`),
    with each night's own injection model.  ``inject_sources``: point sources added the same
    way, after the subtraction (:meth:`~klip_tpe.reducer.KLIPReducer.inject_sources`).
    ``drop_frames``: frames left out of a night at load, ``{"n4": [551]}``, by their index in
    the night's cube counted from 0 (:func:`load_night`), before the subtraction and anything
    else.  Each night named has to be among ``nights``.  All four go into the reducer's
    ``data_edits``.

    ``pxscale`` (arcsec/px, default :attr:`NEAR.pxscale`) sets the pixel constants
    (:func:`near_constants`), and ``psflib`` the PSF library's directory (default
    ``root/psflib``).  Both go into the reducer's ``provenance``, which ``run_setup.json``
    records and a resume reads back.

    ``north``: ``"files"`` gives each night's frames their own true north from the night's
    :data:`NORTH_FILE` where there is one (:func:`load_night`), ``"constant"``
    :attr:`NEAR.truenorth` for every frame, and a run's record (``provenance["north"]``, one
    entry per night) what the run was made with.  The north is used at derotation and in the
    injection azimuth.  The frame selection, the binning and the reference selection read the
    parallactic angles alone, as before.  ``provenance["north"]`` records each night's
    source, the file's SHA-1 and the values' range."""
    drop = _drop_spec(drop_frames)
    labels = [f"n{int(seq)}" for seq in nights]
    extra = [k for k in drop if k not in labels]
    if extra:
        raise DropFramesError(f"{' '.join(extra)} {'are' if len(extra) > 1 else 'is'} not among the nights loaded "
                              f"({' '.join(labels)})")
    pre = [k for k in drop if datasets and int(k[1:]) in datasets]
    if pre:
        raise DropFramesError(f"frames are dropped at load, and the dataset given for {' '.join(pre)} is loaded "
                              f"already: load it with load_night(..., drop_frames=...)")
    const = near_constants(pxscale)
    lib, lib_dir = None, (psflib or os.path.join(root, "psflib"))
    if use_library:
        try:
            lib = N4Library(lib_dir)
            log(f"  psflib: {lib_dir}, {len(lib.seps)} stamps at {lib.seps[0]:.3f}-{lib.seps[-1]:.3f}\" "
                f"(separations from the {lib.sep_source})")
        except Exception as exc:
            log(f"  psflib unavailable ({exc!r}); using the B template + analytic throughput")
    reducers = {}
    rec_north = north if isinstance(north, dict) else None
    for seq in nights:
        nn = (rec_north.get(f"n{seq}") or "files") if rec_north is not None else north
        ds = datasets[seq] if datasets and seq in datasets else load_night(root, seq, frames=frames,
                                                                             cube_name=cube_name,
                                                                             drop_frames=drop.get(f"n{seq}"),
                                                                             north=nn)
        psf = ds.meta.get("psf_template")
        fu = template_flux_unit(psf, const["lam_over_d_px"]) if psf is not None else 1.0
        thru = lib.throughput if lib is not None else analytic_throughput
        fallback = TemplatePSF(psf, center=(7.5, 7.5), ee_radius_px=const["lam_over_d_px"],
                               star_flux=fu, throughput_fn=thru) if psf is not None else None
        model = lib.model(flux_unit=fu) if lib is not None else fallback
        red = reducer_class(backend)(ds, pxscale=const["pxscale"], lam_m=NEAR.lam_klip, diam_m=NEAR.diam,
                          injection_model=model, fallback_model=fallback, truenorth=NEAR.truenorth,
                          fwhm_px=const["fwhm_px"], zone_pad=2.0, outrad_cap=70.0,
                          defaults={"fast": fast, "n_min_ref": n_min_ref, "comb_type": "nwadi",
                                    "use_rdi": bool(use_rdi), "rdi_mode": rdi_mode,
                                    "do_destripe": bool(do_destripe)},
                          mask_fn=mask_fn)
        red.name = f"near_n{seq}"
        reducers[f"n{seq}"] = red
        nref = 0 if ds.ref_cube is None else int(ds.ref_cube.shape[0])
        log(f"  loaded n{seq}: {ds.nframes} frames, PA span {ds.angles.min():.1f}..{ds.angles.max():.1f} deg, "
            f"texp {ds.texp:.0f} s, tags {'yes' if ds.tags else 'no'}, ref-star frames {nref}")
        log(f"    n{seq}: {_north_text(red.north_record())}")
        if f"n{seq}" in drop:
            gone = list(ds.meta.get("dropped_frames") or [])
            past = [i for i in drop[f"n{seq}"] if i not in gone]
            if gone:
                log(f"    n{seq}: frame{'s' if len(gone) > 1 else ''} {' '.join(str(i) for i in gone)} left out at "
                    f"load (drop_frames), {ds.nframes} of {ds.meta.get('nframes_total', '?')} kept")
            if past:
                log(f"    n{seq}: frame{'s' if len(past) > 1 else ''} {' '.join(str(i) for i in past)} "
                    f"{'lie' if len(past) > 1 else 'lies'} past the frames loaded (--max-frames), nothing to drop "
                    f"there")
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool, log=log)
    pr.name = "near"
    pr.provenance = {"instrument": "NEAR", **const, "psflib": None if lib is None else lib.describe(),
                     "north": {k: r.north_record() for k, r in reducers.items()}}
    if const["pxscale"] != NEAR.pxscale:
        log(f"  pixel scale {const['pxscale']:g}\"/px (lambda/D {const['lam_over_d_px']:.3f} px, FWHM "
            f"{const['fwhm_px']:.3f} px), not the default {NEAR.pxscale:g}")
    if cube_name:
        pr.data_edits = dict(getattr(pr, "data_edits", None) or {}, cube_name=str(cube_name))
        for r in reducers.values():
            r.data_edits = dict(getattr(r, "data_edits", None) or {}, cube_name=str(cube_name))
        log(f"  cube: {cube_name} in every night's directory")
    if drop:
        pr.data_edits = dict(getattr(pr, "data_edits", None) or {}, dropped_frames={k: list(v) for k, v in drop.items()})
        for k, r in reducers.items():
            if k in drop:
                r.data_edits = dict(getattr(r, "data_edits", None) or {}, dropped_frames={k: list(drop[k])})
    if subtract_sources:
        src = list(subtract_sources)
        n = pr.subtract_sources(src)
        c = [float(s[2] if not hasattr(s, "contrast") else s.contrast) for s in src]
        r_ = [float(s[0] if not hasattr(s, "rho") else s.rho) for s in src]
        log(f"  subtracted {n} source(s) from every night's frames, once, at load: rho {min(r_):.3f}-"
            f"{max(r_):.3f}\", contrast {min(c):.3g} to {max(c):.3g}")
        from ..reducer import _taken_text
        took = _taken_text(pr)
        if took:
            log(f"    the fraction of their flux that left the frames, on a sample of frames: {took}")
    if inject_sources:
        src = list(inject_sources)
        n = pr.inject_sources(src)
        c = [float(s[2] if not hasattr(s, "contrast") else s.contrast) for s in src]
        r_ = [float(s[0] if not hasattr(s, "rho") else s.rho) for s in src]
        log(f"  added {n} source(s) to every night's frames, once, at load: rho {min(r_):.3f}-"
            f"{max(r_):.3f}\", contrast {min(c):.3g} to {max(c):.3g}")
        from ..reducer import _taken_text
        took = _taken_text(pr, "injected_kept")
        if took:
            log(f"    the fraction of their flux that reached the frames, on a sample of frames: {took}")
    return pr


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95,
               k_max: int = 100, frame_selection: bool = True):
    """Feasibility projection: frame-selection no-cut snapping (addendum 2) followed
    by the reference-count guardrail evaluated on the executed frame set."""
    const = near_constants(getattr(reducer, "pxscale", None))
    ref = ReferenceCountGuard(angles_fn=reducer.frame_angles, tags_fn=reducer.frame_tags,
                              fwhm_px=const["fwhm_bin_px"], lam_over_d_px=const["lam_over_d_px"],
                              n_min_ref=n_min_ref, ref_frac=ref_frac, k_max=k_max)
    if not frame_selection:
        return ref
    fs = FrameSelectionGuard(tags_fn=reducer.frame_tags,
                             nframes_fn=lambda pid: reducer.reducers[pid].data.nframes)
    return compose(fs, ref)


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   measured_kernel: bool = True, metric: str = "mawet", forbidden_pa=(),
                   pixel_mask: Optional[np.ndarray] = None):
    """Objective + position sampler with the production metric settings.  (The two-source
    area-midpoint rule lives in ``RunConfig.pair_area_midpoint``, applied by the Runner.)
    ``metric='fmmf'`` swaps the matched filter for the forward-modelled one.

    ``forbidden_pa`` = ``[(centre_pa_deg, half_width_deg), ...]``: wedges the injections keep
    out of.  ``pixel_mask`` (True = exclude, the cropped frame's shape) keeps pixels out of
    the noise apertures, e.g. :func:`klip_tpe.metrics.line_mask` for the A-B line.  Use the
    two together, as :func:`klip_tpe.instruments.generic.default_config` explains."""
    kfn = reducer.matched_filter_kernel if measured_kernel else None
    const = near_constants(getattr(reducer, "pxscale", None))       # the reducer's scale
    kw = dict(pxscale=const["pxscale"], fwhm=const["fwhm_px"], kernel_fn=kfn, known=list(known))
    if pixel_mask is not None:
        kw["pixel_mask"] = np.asarray(pixel_mask, bool)
    if str(metric).lower() == "fmmf":
        from ..fmmf import FMMFSNR
        metric = FMMFSNR(**kw)
    else:
        metric = MawetPeakSNR(**kw)
    objective = Objective(metric, clean_subtract=clean_subtract)
    sampler = PositionSampler(fwhm_as=const["fwhm_px"] * const["pxscale"], known=list(known),
                              forbidden_pa=[tuple(float(v) for v in b) for b in forbidden_pa])
    return objective, sampler
