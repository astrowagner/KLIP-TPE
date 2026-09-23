"""pyKLIP backend (https://pyklip.readthedocs.io).

* :class:`PyKLIPReducer` -- the optimizer's pre-processed cube goes through
  ``pyklip.parallelized.klip_parallelized`` (ADI, RDI or ADI+RDI; ``algo`` klip / nmf /
  empca); derotation and combination are done here with the package's own routines so
  the conventions match the other backends.  Parameter mapping (searched -> pyKLIP):

  ==============  ==========================================================
  ``k_klip``       ``numbasis`` (k-scan: ``numbasis = 1..k`` in one call)
  ``n_ang``        ``subsections``
  ``inrad/outrad`` ``IWA / OWA`` with ``annuli = n_annuli`` (default 1)
  ``angsep``       ``movement = max(angsep * lambda/D, 1e-6) [px]`` (0 = the frame itself only)
  ``anglemax``     ``maxrot``
  ``bin, filter, corr_thresh, noise_max, coronoise_max``  handled upstream
  ==============  ==========================================================

* :func:`dataset_from_pyklip` -- turn any ``pyklip.instruments.Instrument.Data`` object
  (GPI, CHARIS, JWST from spaceKLIP, GenericData ...) into :class:`Dataset`\\ s: frames
  are re-centred on a common pixel, cropped, and split into partitions (default: one per
  ``filenums`` group, i.e. per exposure / roll; ``partition_by=None`` for one dataset).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..klip import derotate, nw_ang_comb
from ..reducer import Dataset, KLIPParams, KLIPReducer

__all__ = ["PyKLIPReducer", "dataset_from_pyklip", "MIN_MOVEMENT_PX"]

#: Smallest ``movement`` handed to pyKLIP.  Its reference selection is ``moves >= movement``
#: with no other exclusion, so 0 puts the target frame in its own basis (see ``_subtract``);
#: 1e-6 px excludes only frames with no motion at all.
MIN_MOVEMENT_PX = 1e-6


def _require_pyklip():
    try:
        import pyklip.parallelized as par  # noqa: F401
        import pyklip.klip as pk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyKLIPReducer needs pyklip: pip install pyklip") from exc
    _check_pyklip_numpy(pk)
    _quiet_pyklip_progress(par)


def _quiet_pyklip_progress(par) -> None:
    """Turn off pyklip's per-call progress bar.

    ``klip_parallelized`` waits for its workers inside ``trange(len(outputs))`` regardless
    of ``verbose``, so an optimizer that calls it hundreds of times draws hundreds of bars:
    in a terminal that is scrolling noise, and in a notebook with ipywidgets installed each
    bar is a widget output with its state saved into the file -- tutorial 03 grew from 2 to
    7 MB (426 widgets) the day ipywidgets appeared.  The bars are replaced by disabled ones
    on the pyklip module itself, once; ``KLIP_TPE_PYKLIP_PROGRESS=1`` keeps them.
    """
    if getattr(par, "_klip_tpe_quiet", False) or os.environ.get("KLIP_TPE_PYKLIP_PROGRESS"):
        return
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:                                           # pragma: no cover
        return

    def _trange(*a, **k):
        k.setdefault("disable", True)
        return _tqdm(range(*a), **k)

    def _tq(*a, **k):
        k.setdefault("disable", True)
        return _tqdm(*a, **k)
    par.trange, par.tqdm = _trange, _tq
    par._klip_tpe_quiet = True


def _check_pyklip_numpy(pk=None) -> None:
    """pyklip >= 2.9 calls ``numpy.reshape(..., copy=False)`` -- a keyword that exists only
    in numpy >= 2.1 -- and declares no numpy version.  With an older numpy every rotation
    and alignment raises ``TypeError("reshape() got an unexpected keyword argument 'copy'")``,
    which inside a search is not one error but every evaluation failing the same way: paper
    run D on 2026-09-17 spent 350 evaluations on it and ended with "run complete" and no
    winner.  Probe once, at construction, and say what to do."""
    try:
        np.reshape(np.zeros(2), (2,), copy=False)
        return                                                    # numpy >= 2.1: nothing to check
    except TypeError:
        pass
    if pk is None:
        import pyklip.klip as pk
    try:
        import inspect
        uses = "copy=" in inspect.getsource(pk)
    except (OSError, TypeError):                                  # no source (zipapp, frozen): assume the worst
        uses = True
    if uses:
        import pyklip
        raise ImportError(
            f"pyklip {getattr(pyklip, '__version__', '?')} calls numpy.reshape(copy=...), which needs "
            f"numpy >= 2.1, and this environment has numpy {np.__version__}: every reduction would fail "
            f"with \"reshape() got an unexpected keyword argument 'copy'\".  Either upgrade numpy "
            f"(pip install 'numpy>=2.1') or use pyklip <= 2.8.4 (pip install 'pyklip<2.9')."
        )


class PyKLIPReducer(KLIPReducer):
    """KLIP/NMF/empca subtraction by pyKLIP on the optimizer's pre-processed cube.

    Extra defaults (fixed unless you add them to the search space): ``algo``
    (``'klip'`` | ``'nmf'`` | ``'empca'``), ``mode`` (``'ADI'`` | ``'RDI'`` | ``'ADI+RDI'``;
    RDI needs ``Dataset.ref_cube``), ``n_annuli`` (radial subdivisions of the zone),
    ``annuli_spacing``, ``corr_smooth``, ``numthreads``.  ``comb_type`` ``'nwadi'`` (default,
    the noise-weighted combine of the built-in reducer) | ``'mean'`` | ``'median'``.
    """

    name = "pyklip"
    supports_kscan = True
    supports_fm = False

    def __init__(self, data: Dataset, pxscale: float, lam_m: float, diam_m: float, **kw):
        _require_pyklip()
        super().__init__(data, pxscale=pxscale, lam_m=lam_m, diam_m=diam_m, **kw)
        self.defaults.setdefault("algo", "klip")
        self.defaults.setdefault("mode", "ADI")
        self.defaults.setdefault("n_annuli", 1)
        self.defaults.setdefault("annuli_spacing", "constant")
        self.defaults.setdefault("corr_smooth", 1)
        self.defaults.setdefault("numthreads", 1)
        self.defaults.setdefault("minrot", 0.0)
        #: pyKLIP's OWN reference selection, searched (:meth:`set_native_library`)
        self._native_lib: Optional[Dict[str, Any]] = None

    # -- pyKLIP's own library, searched ---------------------------------------------
    def set_native_library(self, *, partition, modes: Sequence[str] = ("ADI", "RDI", "ADI+RDI"),
                           default_mode: str = "ADI+RDI") -> None:
        """Search pyKLIP's own reference selection instead of a fixed one.

        pyKLIP 2.10 already ranks references: per target frame and per sector, its ADI+RDI
        keeps the ``maxnumbasis`` most-correlated frames of the union of the other roll and
        the reference library (``parallelized._klip_section_multifile_perfile``).  Left
        unset, ``maxnumbasis`` is ``max(numbasis)`` -- ``k_klip`` -- so one number used to set
        both the KL truncation and the library size.  This makes the library two searched
        dimensions of its own:

        * ``mode`` -- which pools (``ADI``: the other roll only, ``RDI``: the reference star
          only, ``ADI+RDI``: both, ranked together);
        * ``maxnumbasis`` -- how many of the most-correlated frames of those pools each
          target keeps, decoupled from ``k_klip``.

        It is the pyKLIP-native counterpart of :meth:`KLIPReducer.set_reference_library`,
        which this backend cannot apply: one count across the chosen pools rather than one
        per pool, but ranked per sector rather than per annulus.  ``partition`` labels the
        roll of each science frame and sizes the other-roll pool.
        """
        part = np.asarray(list(partition))
        n_sci = int(self.data.cube.shape[0])
        if part.size != n_sci:
            raise ValueError(f"native library: {part.size} partition labels for {n_sci} science frames")
        _, counts = np.unique(part, return_counts=True)
        n_ref = 0 if self.data.ref_cube is None else int(np.asarray(self.data.ref_cube).shape[0])
        n_alt = int(min(n_sci - c for c in counts)) if counts.size > 1 else 0
        ok = [m for m in modes
              if not (("RDI" in m and n_ref == 0) or (m.startswith("ADI") and n_alt == 0))]
        if not ok:
            raise ValueError("native library: no mode has references (no reference star and one roll)")
        if default_mode not in ok:
            default_mode = ok[-1]
        self._native_lib = {"n_alt": n_alt, "n_ref": n_ref, "modes": list(ok),
                            "default_mode": str(default_mode)}

    def reference_params(self):
        """``mode`` and ``maxnumbasis`` when the native library is searched; never the
        ``nkeep_*`` counts, which this backend cannot apply."""
        s = getattr(self, "_native_lib", None)
        if s is None:
            return []
        from ..space import Param, kgrid
        pool = max(int(s["n_alt"] + s["n_ref"]), 1)
        grid = sorted({float(v) for v in kgrid(pool)} | {float(pool)})
        return [Param("mode", 0, len(s["modes"]) - 1, "categorical", choices=list(s["modes"]),
                      default=s["default_mode"], doc="pyKLIP pools: ADI / RDI / ADI+RDI"),
                Param("maxnumbasis", 1, pool, "int", grid=grid, default=None,
                      doc="most-correlated frames kept per target and sector (pyKLIP maxnumbasis)")]

    def _subtract(self, bcube, bang, kp: KLIPParams, p, filt, req, mcube, ref_basis, fm_ref, meta):
        import pyklip.parallelized as par
        n, ny, nx = bcube.shape
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        centers = np.tile([cx, cy], (n, 1)).astype(float)
        numbasis = np.arange(1, kp.k_klip + 1) if kp.k_scan else np.array([kp.k_klip])
        mode = str(p["mode"]).upper()
        psflib = None
        if "RDI" in mode:
            ref = self._reference_cube(filt, meta)            # Dataset.ref_cube, high-passed like the science
            if ref is None:
                mode = "ADI"
                meta["rdi_note"] = "no reference cube: mode fell back to ADI"
            else:
                from pyklip.instruments.Instrument import GenericData
                from pyklip.rdi import PSFLibrary
                ref = np.asarray(ref, np.float32)
                sci_names = np.array([f"sci{i}" for i in range(n)])
                lib_imgs = np.concatenate([ref, bcube], axis=0)
                names = np.concatenate([np.array([f"ref{i}" for i in range(ref.shape[0])]), sci_names])
                gdata = GenericData(np.asarray(bcube, np.float32), centers, parangs=np.asarray(bang, float),
                                    filenames=sci_names)
                lib = PSFLibrary(lib_imgs, (cx, cy), names, compute_correlation=True)
                lib.prepare_library(gdata)          # science frames excluded from their own library
                psflib = dict(psf_library=lib.master_library, psf_library_corr=lib.correlation,
                              psf_library_good=lib.isgoodpsf)
        # pyKLIP selects reference frames with ``moves >= movement`` and nothing else, so at
        # movement = 0 the target frame -- and every frame at the same PA -- is in its own KL
        # basis: in ADI and ADI+RDI the frame is then subtracted from itself and a companion
        # comes back at the level of round-off (paper run D: peak 3e-7 in ADI, 2e-6 in
        # ADI+RDI, against 2.8 in RDI, which made the search's "election" of RDI a foregone
        # conclusion).  The built-in reducer's angsep = 0 means "exclude the target frame
        # only", so give pyKLIP the same: a floor just above zero keeps out exactly the
        # frames with no motion at all -- the frame itself and, on a roll pair, its
        # same-roll twins -- and no others.
        movement = max(float(kp.angsep) * self.lam_over_d_px, MIN_MOVEMENT_PX)
        # pyKLIP keeps at most maxnumbasis references and clips numbasis to what it kept, so a
        # maxnumbasis below k_klip would quietly make k_klip inert; the guard keeps it above,
        # and so does this.  None/0 = pyKLIP's own default, max(numbasis).
        mnb = p.get("maxnumbasis")
        mnb = None if mnb in (None, 0) else int(max(round(float(mnb)), int(kp.k_klip)))
        out = par.klip_parallelized(np.asarray(bcube, np.float32), centers, np.asarray(bang, float),
                                       np.ones(n), np.zeros(n, int), kp.inrad, OWA=kp.outrad, mode=mode,
                                       annuli=int(p["n_annuli"]), subsections=int(kp.n_ang), movement=movement,
                                       numbasis=numbasis, aligned_center=[cx, cy],
                                       numthreads=int(p["numthreads"]) or None, minrot=float(p["minrot"]),
                                       maxrot=float(kp.anglemax), annuli_spacing=str(p["annuli_spacing"]),
                                       corr_smooth=float(p["corr_smooth"]), algo=str(p["algo"]), verbose=False,
                                       maxnumbasis=mnb, **(psflib or {}))
        sub = np.asarray(out[0], np.float32)                  # (b, n, ny, nx), aligned, not derotated
        ct = p["comb_type"]

        def _combine(dr):
            if ct == "median":
                return np.nanmedian(dr, axis=0)
            if ct == "mean":
                return np.nanmean(dr, axis=0)
            return nw_ang_comb(dr, bang)

        imgs = [_combine(derotate(sub[b], bang, self.truenorth)) for b in range(sub.shape[0])]
        img = np.stack(imgs) if kp.k_scan else imgs[0]
        return img, None, {"backend": "pyklip", "algo": p["algo"], "mode": mode, "movement_px": movement,
                           "maxnumbasis": mnb}

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(backend="pyklip")
        return d


def dataset_from_pyklip(data, crop_half: Optional[int] = None, partition_by: Optional[str] = "filenums",
                        name: str = "pyklip", texp_per_frame: Optional[float] = None,
                        tags_fn: Optional[Callable[[Any, np.ndarray], Optional[Dict[str, np.ndarray]]]] = None,
                        include_psflib: bool = True, wv_index: Optional[int] = None) -> Dict[str, Dataset]:
    """Convert a pyKLIP ``Data`` object into :class:`Dataset`\\ s.

    Frames (``data.input``) are shifted (bilinear) so every star centre (``data.centers``)
    lands on the common pixel ``((nx-1)/2, (ny-1)/2)`` of a ``2*crop_half`` cut-out
    (default: the largest square that fits); ``data.PAs`` are the parallactic angles
    (pyKLIP derotates CCW by PA, like this package).  Multi-wavelength data: pass
    ``wv_index`` to pick one channel (spectral collapse is not done here).
    ``partition_by``: ``'roll'`` (one partition per distinct position angle -- the natural
    unit for JWST, where each exposure of a roll is its own ``filenum``), ``'filenums'``
    (one partition per exposure), ``'filenames'``, any other attribute of ``data``, or
    ``None`` (one dataset).
    ``tags_fn(data, index) -> {'corrs','noises','coronoise'}`` optionally supplies frame
    quality tags.  ``include_psflib`` attaches ``data.psflib`` master frames as
    ``Dataset.ref_cube`` for RDI when present.
    """
    from scipy import ndimage
    imgs = np.asarray(data.input, np.float32)
    centers = np.asarray(data.centers, float)
    pas = np.asarray(data.PAs, float)
    wvs = np.asarray(getattr(data, "wvs", np.ones(len(imgs))), float)
    sel = np.arange(len(imgs))
    if wv_index is not None:
        uw = np.unique(wvs)
        sel = np.flatnonzero(wvs == uw[int(wv_index)])
    ny, nx = imgs.shape[-2:]
    if crop_half is None:
        cxs, cys = centers[sel, 0], centers[sel, 1]
        crop_half = int(np.floor(min(cxs.min(), cys.min(), nx - 1 - cxs.max(), ny - 1 - cys.max())))
    h = int(crop_half)
    c = h - 0.5

    def _recentre(frame, cx, cy):
        out = np.empty((2 * h, 2 * h), np.float32)
        ix, iy = int(np.floor(cx - c)), int(np.floor(cy - c))          # integer origin of the cut-out
        fx, fy = (cx - c) - ix, (cy - c) - iy                           # residual sub-pixel shift
        pad = 1
        y0, x0 = iy - pad, ix - pad
        sub = np.full((2 * h + 2 * pad, 2 * h + 2 * pad), np.nan, np.float32)
        ys, xs = slice(max(y0, 0), min(y0 + sub.shape[0], ny)), slice(max(x0, 0), min(x0 + sub.shape[1], nx))
        sub[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = frame[ys, xs]
        fin = np.isfinite(sub)
        sub0 = np.where(fin, sub, 0.0)
        shifted = ndimage.shift(sub0, (-fy, -fx), order=1, mode="constant", cval=0.0)
        w = ndimage.shift(fin.astype(np.float32), (-fy, -fx), order=1, mode="constant", cval=0.0)
        shifted = np.where(w > 0.999, shifted / np.maximum(w, 1e-6), np.nan)
        out[:] = shifted[pad:pad + 2 * h, pad:pad + 2 * h]
        return out

    key = None
    if partition_by == "roll":                       # group by position angle (rounded to 0.1 deg)
        key = np.round(pas[sel], 1)
    elif partition_by:
        key = np.asarray(getattr(data, partition_by))[sel]
    if key is None:
        groups = [("", sel)]
    else:
        uniq = np.unique(key)
        # short, stable names: roll1, roll2, ... for the roll grouping, the key itself otherwise
        labels = [f"roll{i + 1}" for i in range(len(uniq))] if partition_by == "roll" else [str(k) for k in uniq]
        groups = [(lab, sel[key == k]) for lab, k in zip(labels, uniq)]
    out: Dict[str, Dataset] = {}
    ref_cube = None
    if include_psflib and getattr(data, "psflib", None) is not None:
        try:
            lib = data.psflib
            ref_imgs = np.asarray(lib.master_library, np.float32)
            rc = np.asarray(lib.aligned_center, float)
            ref_cube = np.stack([_recentre(f, rc[0], rc[1]) for f in ref_imgs])
        except Exception:
            ref_cube = None
    for gname, idx in groups:
        cube = np.stack([_recentre(imgs[i], centers[i, 0], centers[i, 1]) for i in idx])
        cube = np.where(np.isfinite(cube), cube, 0.0).astype(np.float32)
        tags = tags_fn(data, idx) if tags_fn else None
        pid = f"{name}{gname}" if gname else name
        meta = {"source": "pyklip", "partition_by": partition_by, "crop_half": h,
                "filenames": [str(f) for f in np.asarray(getattr(data, "filenames", np.array([""] * len(imgs))))[idx][:5]]}
        texp = float(len(idx) * texp_per_frame) if texp_per_frame else float(len(idx))
        out[pid] = Dataset(cube, pas[idx], tags, texp=texp, name=pid, meta=meta, ref_cube=ref_cube)
    return out
