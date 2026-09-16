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

__all__ = ["load_spaceklip", "load_calints", "make_reducer", "read_jwst_files", "JWST_DIAM"]

JWST_DIAM = 6.5   # m (effective; lambda/D for the angsep unit and the default FWHM)


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
                 align: bool = True, repair: bool = True,
                 star_center: Optional[Tuple[float, float]] = None,
                 log: Callable[[str], None] = print) -> Tuple[Dict[str, Dataset], Dict[str, Any]]:
    """Stage-2 ``*_calints.fits`` straight into ``{roll: Dataset}``, without spaceKLIP.

    One partition per unique roll angle, the other target's exposures as the RDI library.
    ``PA = ROLL_REF - V3I_YANG * VPARITY``; DQ-flagged and deviant pixels are replaced by a
    median filter; frames are registered to the median science frame by cross-correlation
    and cropped about ``CRPIX`` to ``2*half_px + 1`` pixels.

    That crop is **odd on purpose**.  :func:`klip_tpe.metrics.star_center` puts the star at
    ``((nx-1)/2, (ny-1)/2)``, so an even crop with the star on an integer pixel leaves it
    half a pixel off in each axis -- 0.7 px radially, which at HIP 65426 b's 13 px
    separation is a 3 degree error in position angle and a throughput mismatch between the
    companion and the fakes injected to calibrate it.

    ``star_center`` (0-based detector pixels) overrides ``CRPIX`` as the point the crop is
    centred on.  **Use it.**  ``CRPIX`` is the aperture reference point, not a measured star
    position -- it is identical in every file of a programme, dithers included -- and on
    ERS 1386 it misses HIP 65426 by about 1.4 px, which throws the companion 1.5 px inside
    its own separation and mismatches its KLIP throughput against the fakes injected to
    calibrate it.  The right source is spaceKLIP's star-centring step (``STARCENX/Y``);
    failing that, ``scripts/check_hip65426_contrast.py`` shows how to solve for it from the
    companion's position in each roll.

    ``info`` carries ``pxscale`` (arcsec/px from ``PIXAR_A2``), ``pixar_sr``, ``bunit``,
    ``filter``, ``star_center`` and the first science ``SCI`` header, which is what the flux
    calibration in :mod:`klip_tpe.stpsf_psf` needs.
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

    def read(fs):
        ims, pas = [], []
        for f in fs:
            with fits.open(f) as h:
                d = np.asarray(h["SCI"].data, float)
                dq = np.asarray(h["DQ"].data, int)
                s = h["SCI"].header
                pa = float(s["ROLL_REF"]) - float(s.get("V3I_YANG", 0)) * float(s.get("VPARITY", 1))
                d = np.where((dq & 1).astype(bool), np.nan, d)
                for i in range(d.shape[0]):
                    ims.append(d[i])
                    pas.append(pa)
        return np.array(ims), np.array(pas)

    def _repair(cube, size=5, nsig=7.0):
        out = np.array(cube, float)
        for i, im in enumerate(out):
            bad = ~np.isfinite(im)
            med = ndimage.median_filter(np.where(bad, np.nanmedian(im), im), size=size)
            r = np.abs(im - med)
            s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
            out[i] = np.where(bad | (r > nsig * s), med, im)
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

    S, pas = read(sci)
    R, _ = read(ref) if ref else (np.zeros((0,) + S.shape[1:]), np.zeros(0))
    if repair:
        S = _repair(S)
        R = _repair(R) if R.size else R
    if align:
        a0 = np.median(S, axis=0)
        S = np.array([ndimage.shift(im, (-d[1], -d[0]), order=3)
                      for im, d in ((im, _xs(im, a0)) for im in S)])
        if R.size:
            R = np.array([ndimage.shift(im, (-d[1], -d[0]), order=3)
                          for im, d in ((im, _xs(im, a0)) for im in R)])

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
    dsets: Dict[str, Dataset] = {}
    for k, pa in enumerate(np.unique(np.round(pas, 1))):
        m = np.round(pas, 1) == pa
        dsets[f"roll{k + 1}"] = Dataset(Sx[m], pas[m], name=f"roll{k + 1}",
                                        ref_cube=Rx if Rx.size else None, meta=dict(meta))
    info = dict(meta, n_sci=int(S.shape[0]), n_ref=int(R.shape[0]), crop_px=n,
                filter=str(fits.getheader(sci[0]).get("FILTER", "")),
                star_center=(float(cx), float(cy)),
                crpix=(float(hdr["CRPIX1"] - 1), float(hdr["CRPIX2"] - 1)),
                rolls=[float(v) for v in np.unique(np.round(pas, 1))])
    log(f"  calints: {len(sci)} science / {len(ref)} reference files -> {S.shape[0]} + "
        f"{R.shape[0]} frames, {n}x{n} px at {px*1e3:.2f} mas, rolls {info['rolls']}, "
        f"{info['bunit']!r}")
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
