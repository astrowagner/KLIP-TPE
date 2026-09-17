"""spaceKLIP / JWST backend (https://spaceklip.readthedocs.io).

spaceKLIP's own PSF-subtraction step hands the stage-2 / ImageTools products to pyKLIP
(``pyklip.instruments.JWST.JWSTData`` -> ``klip_dataset``).  This module does the same
hand-over into the optimizer: the science rolls become partitions, the reference-star
exposures become the RDI library, and :class:`~klip_tpe.backends.pyklip.PyKLIPReducer`
(mode ``'ADI+RDI'`` by default, as in spaceKLIP) does the subtraction, so the parameters
the optimizer searches (``numbasis``, ``subsections``, ``movement``/``angsep``, annulus,
frame selection, binning, high-pass) are exactly the knobs of a spaceKLIP reduction.

Entry points
------------
:func:`load_spaceklip`
    from a ``spaceKLIP.Database`` (or its ``obs`` table dict) and a concatenation key, or
    directly from lists of science / reference FITS files.  Uses pyKLIP's ``JWSTData``
    when importable (needs pyklip + astroquery); otherwise reads the files with the same
    header arithmetic (``STARCENX/Y`` or ``CRPIX``, ``PA = ROLL_REF - V3I_YANG*VPARITY``).
:func:`make_reducer`
    ``PyKLIPReducer`` per roll with a JWST PSF template for injection: pass a webbpsf /
    ``webbpsf_ext`` offset PSF (2-D array or FITS path, e.g. spaceKLIP's
    ``analysistools.get_offsetpsf`` output or a ``JWST_PSF`` model) as ``psf_template``,
    else a Gaussian of the filter's ``1.028 lambda/D`` is used (calibrate contrasts!).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..injection import GaussianPSF, InjectionModel, TemplatePSF
from ..reducer import Dataset, PartitionedReducer
from .pyklip import PyKLIPReducer, dataset_from_pyklip

__all__ = ["load_spaceklip", "load_calints", "make_reducer", "read_jwst_files", "JWST_DIAM",
           "fill_dq_neighbours", "sigma_clip_repair"]

JWST_DIAM = 6.5   # m (effective; lambda/D for the angsep unit and the default FWHM)


def fill_dq_neighbours(im: np.ndarray, maxit: int = 20) -> Tuple[np.ndarray, int]:
    """Replace every non-finite pixel by the median of its finite orthogonal and diagonal
    neighbours -- spaceKLIP's bad-pixel treatment (Carter et al. 2023, section 2.2) -- and
    nothing else.  Clusters are closed from their rims inward, ``maxit`` passes at most.
    Returns ``(filled image, number of pixels filled)``.

    This touches ONLY the pixels the pipeline flagged (DQ ``DO_NOT_USE``, which
    :func:`load_calints` has already turned into NaN).  A coronagraphic PSF is *supposed*
    to be full of sharp, isolated blobs -- the six-lobed Lyot-stop pattern and, for a
    companion behind MASK335R, a "hamburger" core of three bars -- and any filter that
    decides from the pixel values what is an outlier will eat those.  See
    :func:`sigma_clip_repair` for the one that did.
    """
    out = np.array(im, float)
    n0 = int((~np.isfinite(out)).sum())
    ny, nx = out.shape
    for _ in range(int(maxit)):
        bad = ~np.isfinite(out)
        if not bad.any():
            break
        p = np.pad(out, 1, constant_values=np.nan)
        st = np.stack([p[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(st, axis=0)
        out[bad] = med[bad]
    return out, n0


def sigma_clip_repair(im: np.ndarray, size: int = 5, nsig: float = 7.0) -> Tuple[np.ndarray, int, int]:
    """The pre-1.1 ``load_calints`` repair: a ``size x size`` median filter, and every pixel
    that differs from it by more than ``nsig`` times the *frame-wide* robust scatter of that
    difference is replaced by the median.  Returns ``(image, n_rewritten, n_of_those_flagged)``.

    **Do not use this on coronagraphic frames.**  The scatter is set by the empty sky, so
    the threshold is a few counts, and every pixel of the structured PSF -- star and
    companion alike -- exceeds it: on ERS 1386 F444W it rewrites ~4,500-5,500 pixels per
    320x320 frame, of which only ~1,560 are DQ-flagged; the rest is the PSF, median-filtered.
    That blurred HIP 65426 b from Carter et al. (2023)'s three-bar core into one blob at
    39% of its peak, and because the set of rewritten pixels differs from frame to frame it
    left a roll-dependent residual that was mistaken for a speckle.  Kept only so that runs
    made before the fix can be reproduced (``repair='sigma'``).
    """
    from scipy import ndimage
    im = np.asarray(im, float)
    bad = ~np.isfinite(im)
    med = ndimage.median_filter(np.where(bad, np.nanmedian(im), im), size=int(size))
    r = np.abs(im - med)
    s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    hit = bad | (r > float(nsig) * s)
    return np.where(hit, med, im), int(hit.sum()), int(bad.sum())


def _files_from_database(database, key: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    obs = database.obs if hasattr(database, "obs") else database
    tab = obs[key]
    sci = [str(f) for t, f in zip(tab["TYPE"], tab["FITSFILE"]) if t == "SCI"]
    ref = [str(f) for t, f in zip(tab["TYPE"], tab["FITSFILE"]) if t == "REF"]
    info = {}
    for col in ("FILTER", "CWAVEL", "PIXSCALE", "TELESCOP", "INSTRUME", "APERNAME"):
        if col in tab.colnames:
            info[col] = tab[col][0]
    return sci, ref, info


def read_jwst_files(filepaths: Sequence[str], center_keywords: Tuple[str, str] = ("STARCENX", "STARCENY")
                    ) -> Dict[str, np.ndarray]:
    """Minimal reader for spaceKLIP stage-2 / ImageTools ``*_calints.fits`` products
    (what ``pyklip.instruments.JWST.JWSTData.readdata`` does, without the SVO lookup)."""
    from astropy.io import fits
    imgs, centers, pas, nums, names, wvs, pxs = [], [], [], [], [], [], []
    for i, f in enumerate(filepaths):
        with fits.open(f) as h:
            ph, sh = h[0].header, h["SCI"].header
            data = np.asarray(h["SCI"].data, np.float32)
            if data.ndim == 2:
                data = data[None]
            nints = data.shape[0]
            pxscale = float(np.sqrt(sh["PIXAR_A2"])) if "PIXAR_A2" in sh else float(ph.get("PIXSCALE", np.nan))
            cx = sh.get(center_keywords[0], sh.get("CRPIX1")) - 1
            cy = sh.get(center_keywords[1], sh.get("CRPIX2")) - 1
            if "XOFFSET" in ph and np.isfinite(pxscale):
                cx += ph["XOFFSET"] / pxscale
                cy += ph["YOFFSET"] / pxscale
            pa = float(sh["ROLL_REF"]) - float(sh.get("V3I_YANG", 0.0)) * float(sh.get("VPARITY", 1))
            imgs.append(data)
            centers.append(np.tile([cx, cy], (nints, 1)))
            pas.append(np.full(nints, pa))
            nums.append(np.full(nints, i))
            names.append(np.array([os.path.basename(f)] * nints))
            wvs.append(np.full(nints, _wavelength_m(ph)))
            pxs.append(pxscale)
    return {"input": np.concatenate(imgs), "centers": np.concatenate(centers), "PAs": np.concatenate(pas),
            "filenums": np.concatenate(nums), "filenames": np.concatenate(names), "wvs": np.concatenate(wvs),
            "pxscale": float(np.nanmedian(pxs))}


def _wavelength_m(ph) -> float:
    """Filter central wavelength (m) from the primary header: ``CWAVEL`` (micron) when
    present, else pyKLIP's JWST filter tables (NIRCam PUPIL/FILTER, NIRISS, MIRI)."""
    if "CWAVEL" in ph:
        return 1e-6 * float(ph["CWAVEL"])
    try:
        from pyklip.instruments.JWST import JWSTData
        inst = str(ph.get("INSTRUME", "")).upper()
        tabs = {"NIRCAM": "wave_nircam", "NIRISS": "wave_niriss", "MIRI": "wave_miri"}
        import ast, inspect, re                    # the offline tables live as literals in __init__
        src = inspect.getsource(JWSTData.__init__)
        ms = re.findall(r"self\." + tabs.get(inst, "x") + r" = (\{[^\n]*\})", src)
        lit = [m for m in ms if len(m) > 2]              # skip the empty '{}' initialisers
        if lit:
            table = ast.literal_eval(lit[0])
            key = ph.get("PUPIL") if inst == "NIRCAM" and ph.get("PUPIL") in table else ph.get("FILTER")
            if key in table:
                return 1e-6 * float(table[key])
    except Exception:
        pass
    return float("nan")


class _ArrData:
    """Duck-typed stand-in for a pyKLIP Data object (what :func:`dataset_from_pyklip` needs)."""

    def __init__(self, d: Dict[str, np.ndarray], psflib=None):
        self.input, self.centers, self.PAs = d["input"], d["centers"], d["PAs"]
        self.filenums, self.filenames, self.wvs = d["filenums"], d["filenames"], d["wvs"]
        self.psflib = psflib


class _ArrLib:
    def __init__(self, master_library: np.ndarray, aligned_center: Tuple[float, float]):
        self.master_library, self.aligned_center = master_library, aligned_center


def load_spaceklip(database=None, key: Optional[str] = None, sci_files: Optional[Sequence[str]] = None,
                   ref_files: Optional[Sequence[str]] = None, crop_half: Optional[int] = None,
                   partition_by: Optional[str] = "roll", name: Optional[str] = None,
                   use_pyklip_reader: bool = True, log: Callable[[str], None] = print) -> Dict[str, Dataset]:
    """Datasets (one per science roll by default) + RDI reference cube from a spaceKLIP
    database concatenation or explicit file lists.  ``Dataset.meta`` carries
    ``pxscale`` and ``wavelength_m`` when the headers provide them.

    ``partition_by='roll'`` (default) groups the integrations by position angle, which is
    what you want for JWST: every exposure of a roll is a separate ``filenum``, so
    ``'filenums'`` would give one partition per exposure (often a single frame).
    ``None`` puts everything in one dataset."""
    if database is not None:
        if key is None:
            keys = list((database.obs if hasattr(database, "obs") else database).keys())
            if len(keys) != 1:
                raise ValueError(f"pass key= (available: {keys})")
            key = keys[0]
        sci_files, ref_files, info = _files_from_database(database, key)
        if name is None:
            name = str(key)
    else:
        info = {}
    if not sci_files:
        raise ValueError("no science files")
    name = name or "jwst"
    data = None
    if use_pyklip_reader:
        try:
            from pyklip.instruments.JWST import JWSTData
            data = JWSTData(list(sci_files), list(ref_files) if ref_files else None)
        except Exception as exc:
            log(f"  pyklip JWSTData unavailable ({exc!r}); using the built-in reader")
    if data is None:
        d = read_jwst_files(sci_files)
        psflib = None
        if ref_files:
            r = read_jwst_files(ref_files)
            # each reference frame is first shifted to one common centre; dataset_from_pyklip
            # then treats the library like pyKLIP's (single aligned_center)
            target = tuple(np.median(r["centers"], axis=0))
            psflib = _ArrLib(_recentre_stack(r["input"], r["centers"], target), target)
        data = _ArrData(d, psflib)
        pxscale = d["pxscale"]
        wave = float(np.nanmedian(d["wvs"])) if np.isfinite(d["wvs"]).any() else None
    else:
        pxscale = float(getattr(data, "pxscale", np.nan)) if hasattr(data, "pxscale") else float(info.get("PIXSCALE", np.nan))
        if not np.isfinite(pxscale):                  # pyKLIP's JWSTData does not carry it -> read a header
            try:
                from astropy.io import fits
                with fits.open(sci_files[0]) as h:
                    sh = h["SCI"].header
                    pxscale = float(np.sqrt(sh["PIXAR_A2"])) if "PIXAR_A2" in sh else float(h[0].header.get("PIXSCALE", np.nan))
            except Exception:
                pxscale = float("nan")
        wave = float(np.nanmedian(np.asarray(data.wvs, float))) if len(data.wvs) else None
    out = dataset_from_pyklip(data, crop_half=crop_half, partition_by=partition_by, name=name)
    for ds in out.values():
        ds.meta.update(source="spaceklip", pxscale=pxscale, wavelength_m=wave, filter=info.get("FILTER"))
        log(f"  {ds.name}: {ds.nframes} frames, PA {ds.angles.min():.1f}..{ds.angles.max():.1f} deg, "
            f"refs {0 if ds.ref_cube is None else ds.ref_cube.shape[0]}")
    return out


def load_calints(files: Sequence[str], science_target: Optional[str] = None, half_px: int = 55,
                 align: bool = True, repair: Union[bool, str] = True,
                 star_center: Optional[Tuple[float, float]] = None, keep_frames: bool = False,
                 partition: str = "roll",
                 log: Callable[[str], None] = print) -> Tuple[Dict[str, Dataset], Dict[str, Any]]:
    """Stage-2 ``*_calints.fits`` straight into ``{name: Dataset}``, without spaceKLIP.

    ``partition='roll'`` (default): one partition per unique roll angle, the other target's
    exposures as the RDI library of each.  ``partition='all'``: ONE partition holding every
    science frame with its own PA, plus the library.  **The choice decides what pyKLIP's
    ``mode`` can mean.**  A partition is reduced on its own, so with one per roll every frame
    in it has the same PA: ADI has no reference frames at all (pyKLIP returns NaN) and
    ADI+RDI is RDI -- the other roll is never in the basis, because it is in the other
    partition.  Only with ``'all'`` does ADI subtract the other roll, and only then is a
    searched ``mode`` a real choice (on ERS 1386 F444W at k = 10: injected S/N 6.2 ADI,
    6.9 RDI, 7.2 ADI+RDI; companion peak 1.7 / 2.9 / 2.6 MJy/sr).  The price of ``'all'``
    is one ``k_klip`` block for both rolls instead of one per roll.
    ``PA = ROLL_REF - V3I_YANG * VPARITY``; DQ ``DO_NOT_USE`` pixels are replaced by the
    median of their neighbours (:func:`fill_dq_neighbours`, spaceKLIP's treatment) and
    **nothing else is touched**; frames are registered to the median science frame by
    cross-correlation and cropped about the star to ``2*half_px + 1`` pixels.

    ``repair``: ``True`` / ``'dq'`` (default) fills the flagged pixels only; ``False`` leaves
    them NaN (the built-in reducers cope, pyKLIP's RDI library does not); ``'sigma'`` is the
    pre-1.1 behaviour, :func:`sigma_clip_repair`, which median-filters the whole PSF and is
    kept only to reproduce old runs -- it is logged loudly when used.  The per-frame
    ``n_repaired`` in ``info['frames']`` says how many pixels were rewritten either way.

    That crop is **odd on purpose**.  :func:`klip_tpe.metrics.star_center` puts the star at
    ``((nx-1)/2, (ny-1)/2)``, so an even crop with the star on an integer pixel leaves it
    half a pixel off in each axis -- 0.7 px radially, which at HIP 65426 b's 13 px
    separation is a 3 degree error in position angle and a throughput mismatch between the
    companion and the fakes injected to calibrate it.

    ``star_center`` (0-based detector pixels) overrides ``CRPIX`` as the point the crop is
    centred on.  **Use it.**  ``CRPIX`` is the aperture reference point, not a measured star
    position -- it is identical in every file of a programme, dithers included -- and on
    ERS 1386 it misses HIP 65426 by about 0.8 px, which throws the companion 0.8 px inside
    its own separation and mismatches its KLIP throughput against the fakes injected to
    calibrate it.  The right source is spaceKLIP's star-centring step (``STARCENX/Y``);
    failing that, ``scripts/check_hip65426_contrast.py`` shows how to solve for it from the
    companion's position in each roll.

    ``info`` carries ``pxscale`` (arcsec/px from ``PIXAR_A2``), ``pixar_sr``, ``bunit``,
    ``filter``, ``star_center`` and the first science ``SCI`` header, which is what the flux
    calibration in :mod:`klip_tpe.stpsf_psf` needs.  It also carries ``frames``: one dict per
    individual integration with the archive file it came from, its integration index, role,
    position angle, commanded dither offset, how many pixels the DQ and the outlier repair
    touched, the shift the registration measured and removed, and the integration's own
    clock -- ``effinttm`` (its exposure time, the JWST analogue of a DIT), ``int_mid_mjd``
    from ``INT_TIMES``, and the ``readpatt``/``ngroups``/``nframes``/``groupgap``/``tframe``
    that build it.  There is nothing below an integration to recover: the ``nframes`` frames
    of each group are averaged on the detector, and the ``ngroups`` groups are cumulative
    samples of one charge ramp, already collapsed to a slope in a calints file.

    ``keep_frames=True``
    additionally attaches the image stacks themselves (``raw_sci``/``raw_ref``,
    ``aligned_sci``/``aligned_ref``, ``crop_sci``/``crop_ref``) -- about 18 MB for this
    programme, which is why it is off by default.
    """
    from astropy.io import fits
    from scipy import ndimage

    files = sorted(files)
    if not files:
        raise ValueError("load_calints got no files")

    def targ(f):
        return str(fits.getheader(f).get("TARGPROP", "")).replace("-", "").replace("_", "").upper()

    want = (science_target or targ(files[0])).replace("-", "").replace("_", "").upper()
    sci = [f for f in files if targ(f).startswith(want)]
    ref = [f for f in files if f not in sci]
    if not sci:
        raise ValueError(f"no science files matching TARGPROP {want!r} among {len(files)} files")

    def read(fs, role):
        ims, pas, prov = [], [], []
        for f in fs:
            with fits.open(f) as h:
                d = np.asarray(h["SCI"].data, float)
                dq = np.asarray(h["DQ"].data, int)
                s = h["SCI"].header
                ph = h[0].header
                pa = float(s["ROLL_REF"]) - float(s.get("V3I_YANG", 0)) * float(s.get("VPARITY", 1))
                bad = (dq & 1).astype(bool)
                d = np.where(bad, np.nan, d)
                # One plane of a calints cube is ONE integration -- a single up-the-ramp
                # charge accumulation, already fitted to a slope.  EFFINTTM is its
                # exposure time, the JWST analogue of a DIT; the NFRAMES frames that
                # build each group were averaged on the detector and no longer exist
                # separately.  int_mid_mjd comes from INT_TIMES when the extension is
                # there, so each frame carries its own clock rather than the exposure's.
                mids = None
                if "INT_TIMES" in h and h["INT_TIMES"].data is not None:
                    t = h["INT_TIMES"].data
                    if "int_mid_MJD_UTC" in t.columns.names:
                        mids = np.asarray(t["int_mid_MJD_UTC"], float)
                for i in range(d.shape[0]):
                    ims.append(d[i])
                    pas.append(pa)
                    prov.append({"file": os.path.basename(f), "integration": i, "role": role,
                                 "target": str(ph.get("TARGPROP", "")), "pa": pa,
                                 "n_dq": int(bad[i].sum()),
                                 "effinttm": float(ph.get("EFFINTTM", np.nan)),
                                 "readpatt": str(ph.get("READPATT", "")),
                                 "ngroups": int(ph.get("NGROUPS", 0)),
                                 "nframes": int(ph.get("NFRAMES", 0)),
                                 "groupgap": int(ph.get("GROUPGAP", 0)),
                                 "tframe": float(ph.get("TFRAME", np.nan)),
                                 "int_mid_mjd": (float(mids[i]) if mids is not None
                                                 and i < mids.size else float("nan")),
                                 # the small-grid dither offsets live in the PRIMARY
                                 # header, not SCI -- read_jwst_files gets this right too
                                 "xoffset": float(ph.get("XOFFSET", 0.0)),
                                 "yoffset": float(ph.get("YOFFSET", 0.0))})
        return np.array(ims), np.array(pas), prov

    rmode = {True: "dq", False: "none", None: "none"}.get(repair, repair)
    rmode = str(rmode).lower()
    if rmode not in ("dq", "sigma", "none"):
        raise ValueError(f"repair must be True/'dq', False or 'sigma', got {repair!r}")

    unflagged_rewritten: List[int] = []

    def _repair(cube, prov=None):
        out = np.array(cube, float)
        for i, im in enumerate(out):
            if rmode == "sigma":
                out[i], n_hit, n_bad = sigma_clip_repair(im)
                unflagged_rewritten.append(n_hit - n_bad)
            else:
                out[i], n_hit = fill_dq_neighbours(im)
            if prov is not None:
                prov[i]["n_repaired"] = int(n_hit)
                prov[i]["repair"] = rmode
        return out

    def _xs(im, ref_, search=6):
        cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(im) * np.conj(np.fft.rfft2(ref_)), s=im.shape))
        c = np.array(im.shape) // 2
        sub = cc[c[0] - search:c[0] + search + 1, c[1] - search:c[1] + search + 1]
        k = np.unravel_index(np.argmax(sub), sub.shape)
        dy, dx = k[0] - search, k[1] - search
        p = lambda a, b, c_: 0.0 if (a - 2 * b + c_) == 0 else 0.5 * (a - c_) / (a - 2 * b + c_)
        if 0 < k[0] < sub.shape[0] - 1:
            dy += p(sub[k[0] - 1, k[1]], sub[k[0], k[1]], sub[k[0] + 1, k[1]])
        if 0 < k[1] < sub.shape[1] - 1:
            dx += p(sub[k[0], k[1] - 1], sub[k[0], k[1]], sub[k[0], k[1] + 1])
        return dx, dy

    S, pas, prov_s = read(sci, "SCI")
    if ref:
        R, _, prov_r = read(ref, "REF")
    else:
        R, prov_r = np.zeros((0,) + S.shape[1:]), []
    raw_s = np.array(S, np.float32) if keep_frames else None
    raw_r = np.array(R, np.float32) if keep_frames else None
    if rmode != "none":
        S = _repair(S, prov_s)
        R = _repair(R, prov_r) if R.size else R
    if unflagged_rewritten:
        log(f"  calints: repair='sigma' rewrote a median of {int(np.median(unflagged_rewritten))} UNFLAGGED "
            f"pixels per frame on top of the DQ ones -- that is the PSF being median-filtered, star and "
            f"companion alike (HIP 65426 b peaks at 7 instead of 19 MJy/sr).  Use repair='dq'.")
    if align:
        a0 = np.median(S, axis=0)

        def _shift_all(cube, prov):
            out = []
            for i, im in enumerate(cube):
                dx, dy = _xs(im, a0)
                if prov is not None:
                    prov[i]["dx"], prov[i]["dy"] = float(dx), float(dy)
                out.append(ndimage.shift(im, (-dy, -dx), order=3))
            return np.array(out)

        S = _shift_all(S, prov_s)
        if R.size:
            R = _shift_all(R, prov_r)

    hdr = fits.getheader(sci[0], "SCI")
    px = float(np.sqrt(hdr["PIXAR_A2"]))
    if star_center is not None:
        cx, cy = float(star_center[0]), float(star_center[1])
    else:
        cx, cy = hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1
        log(f"  calints: centring on CRPIX ({cx:.2f}, {cy:.2f}) -- the APERTURE reference "
            f"point, not a measured star position; pass star_center= if you have one")
    H = int(half_px)
    n = 2 * H + 1                                        # ODD: see the docstring

    def crop(cube):
        if not cube.size:
            return cube
        fx, fy = cx - round(cx), cy - round(cy)
        x0, y0 = int(round(cx)) - H, int(round(cy)) - H
        return np.array([ndimage.shift(im, (-fy, -fx), order=3)[y0:y0 + n, x0:x0 + n]
                         for im in cube], np.float32)

    Sx, Rx = crop(S), crop(R)
    meta = {"pxscale": px, "wavelength_m": _wavelength_m(fits.getheader(sci[0])),
            "header": dict(fits.getheader(sci[0])), "pixar_sr": float(hdr.get("PIXAR_SR", np.nan)),
            "bunit": str(hdr.get("BUNIT", "")).strip()}
    if str(partition).lower() not in ("roll", "all"):
        raise ValueError(f"partition must be 'roll' or 'all', got {partition!r}")
    dsets: Dict[str, Dataset] = {}
    if str(partition).lower() == "all":
        dsets["sci"] = Dataset(Sx, pas, name="sci", ref_cube=Rx if Rx.size else None, meta=dict(meta))
    else:
        for k, pa in enumerate(np.unique(np.round(pas, 1))):
            m = np.round(pas, 1) == pa
            dsets[f"roll{k + 1}"] = Dataset(Sx[m], pas[m], name=f"roll{k + 1}",
                                            ref_cube=Rx if Rx.size else None, meta=dict(meta))
    info = dict(meta, n_sci=int(S.shape[0]), n_ref=int(R.shape[0]), crop_px=n,
                frames=prov_s + prov_r, repair=rmode, partition=str(partition).lower(),
                filter=str(fits.getheader(sci[0]).get("FILTER", "")),
                star_center=(float(cx), float(cy)),
                crpix=(float(hdr["CRPIX1"] - 1), float(hdr["CRPIX2"] - 1)),
                rolls=[float(v) for v in np.unique(np.round(pas, 1))])
    if keep_frames:
        info.update(raw_sci=raw_s, raw_ref=raw_r,
                    aligned_sci=np.asarray(S, np.float32), aligned_ref=np.asarray(R, np.float32),
                    crop_sci=Sx, crop_ref=Rx)
    nrep = [p.get("n_repaired", 0) for p in prov_s + prov_r]
    log(f"  calints: {len(sci)} science / {len(ref)} reference files -> {S.shape[0]} + "
        f"{R.shape[0]} frames, {n}x{n} px at {px*1e3:.2f} mas, rolls {info['rolls']}, "
        f"{info['bunit']!r}; repair={rmode!r}"
        + (f" ({int(np.median(nrep))} px/frame)" if nrep and rmode != "none" else ""))
    return dsets, info


def _recentre_stack(imgs: np.ndarray, centers: np.ndarray, target: Tuple[float, float]) -> np.ndarray:
    from scipy import ndimage
    out = np.empty_like(imgs, dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        f = np.where(np.isfinite(imgs[i]), imgs[i], 0.0)
        out[i] = ndimage.shift(f, (target[1] - cy, target[0] - cx), order=1, mode="constant", cval=0.0)
    return out


def make_reducer(datasets: Dict[str, Dataset], pxscale: Optional[float] = None, wavelength_m: Optional[float] = None,
                 diam_m: float = JWST_DIAM, psf: str = "auto",
                 psf_template: Union[None, str, np.ndarray] = None,
                 psf_center: Optional[Tuple[float, float]] = None, star_flux: Optional[float] = None,
                 injection_model: Optional[InjectionModel] = None,
                 mode: str = "ADI+RDI", weighting: str = "equal", max_workers=1, pool: str = "threads",
                 n_min_ref: int = 10,
                 fwhm_px: Optional[float] = None, outrad_cap: Optional[float] = None,
                 stpsf_kw: Optional[Dict[str, Any]] = None,
                 log: Callable[[str], None] = print, **kw) -> PartitionedReducer:
    """``PyKLIPReducer`` per roll.

    The injection model, in order of precedence: an ``injection_model`` you built
    yourself; ``psf='stpsf'``, which computes the **off-axis coronagraphic PSF of the
    actual mask** with STPSF (:func:`klip_tpe.stpsf_psf.model_for_datasets`) -- separation
    dependent, with the mask throughput, and the right template to forward-model for
    :class:`klip_tpe.fmmf.FMMFSNR`; a ``psf_template`` webbpsf(_ext) offset PSF (array or
    FITS path) in the same pixel scale; else a Gaussian of ``1.028 lambda/D``.
    ``star_flux`` puts the template on the data's flux scale (if None the template's own
    flux is the unit, i.e. contrast is relative to the template).

    ``pool='threads'`` (default) maps the partitions onto threads rather than forked
    worker processes: pyKLIP starts its own process pool inside ``klip_parallelized``, and
    a forked (daemonic) worker may not have children.
    """
    if injection_model is None and str(psf).lower() == "stpsf":
        from ..stpsf_psf import model_for_datasets
        injection_model = model_for_datasets(list(datasets.values()), star_flux=star_flux or 1.0,
                                             pxscale=pxscale, log=log, **(stpsf_kw or {}))
    reducers = {}
    for pid, ds in datasets.items():
        px = pxscale or ds.meta.get("pxscale")
        lam = wavelength_m or ds.meta.get("wavelength_m")
        if not px or not np.isfinite(px):
            raise ValueError(f"{pid}: pixel scale unknown -- pass pxscale=")
        if not lam or not np.isfinite(lam):
            raise ValueError(f"{pid}: wavelength unknown -- pass wavelength_m=")
        lod = (lam / diam_m) * 206265.0 / px
        fw = fwhm_px or 1.028 * lod
        if injection_model is not None:
            model: InjectionModel = injection_model
        elif psf_template is not None:
            t = psf_template
            if isinstance(t, str):
                from astropy.io import fits
                t = np.asarray(fits.getdata(t), float)
            model = TemplatePSF(t, center=psf_center, ee_radius_px=2.0 * lod, star_flux=star_flux)
        else:
            model = GaussianPSF(fw, star_flux=star_flux or 1.0)
        ny = ds.cube.shape[-1]
        cap = outrad_cap or (ny / 2.0 - 2.0)
        red = PyKLIPReducer(ds, pxscale=px, lam_m=lam, diam_m=diam_m, injection_model=model, fwhm_px=fw,
                            outrad_cap=cap, defaults={"mode": mode if ds.ref_cube is not None else "ADI",
                                                      "n_min_ref": n_min_ref, **kw})
        red.name = f"spaceklip_{pid}"
        reducers[pid] = red
        log(f"  {pid}: pyklip {red.defaults['mode']}, lambda/D {lod:.2f} px, fwhm {fw:.2f} px, "
            f"injection {model.name}")
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool)
    pr.name = "spaceklip"
    pr.partition_label = "roll"                              # display wording only
    return pr
