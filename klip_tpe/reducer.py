"""Reducers, the black box the search optimizes.

A :class:`Reducer` turns a parameter dictionary, with optional synthetic injections, into a
derotated, combined image.  :class:`KLIPReducer` is the reference implementation on top of
:mod:`klip_tpe.klip`.  Another engine (pyKLIP, VIP, a shell script) is wrapped by subclassing
:class:`Reducer` and implementing :meth:`Reducer.reduce`.

:class:`PartitionedReducer` handles a dataset made of independent partitions, such as nights,
epochs, detector sides or rolls.  Each selected partition is reduced with its own parameter
block, and the results are combined with ``equal`` or ``sqrt_texp`` weights.

Notes
-----
The partitions generalize the multi-night machinery of the IDL predecessor, without its
night-specific plumbing.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .injection import (InjectionModel, _check_float32_headroom, add_crop, cube_peak, inject_frames,
                        place_source, prepare_sources)
from .klip import (BLOCK_BYTES, KLIPParams, _highpass_frames, arcdist_deg, bin_by_groups, bin_frames,
                   bin_groups, bin_values, bins_with_data, derotate, destripe, frame_blocks, frame_selection_mask, group_bounds,
                   group_mean, highpass_stack, klip_annular, nw_ang_comb, run_blocks)
from .metrics import Source
from .space import Config

__all__ = ["reducer_class", "set_klip_gram", "set_zone_center", "KLIP_GRAM_MODES", "ReductionRequest", "ReductionResult", "Reducer", "Dataset", "KLIPReducer",
           "PartitionedReducer", "EvalImages"]


@dataclass
class ReductionRequest:
    """One reduction asked of a reducer.

    Attributes
    ----------
    params : dict
        Parameters of the reduction, a configuration of the search space.
    injections : sequence of Source or None, default None
        Sources to inject before the reduction.  None injects nothing.
    k_scan : bool, default False
        Whether to return the images for every number of KL modes from 1 to ``k_klip``.
    tag : str, default ""
        Label of the request, for logs.
    fm_sources : sequence of Source or None, default None
        KLIP-FM test sources, put in the model cube and never added to the data.
    extras : bool, default False
        Whether to also make the reference images, ``nosub`` for the clean pass and ``cadi`` for
        the injected pass.
    threads : int, default 1
        Threads of this partition job, for the per-target KLIP loop and the per-frame steps, set by
        :class:`PartitionedReducer`.
    stage : {"reduce", "prepare"}, default "reduce"
        What to do.  "prepare" returns the prepared frames as :class:`SiblingFrames`, for the other
        wavelength groups' SDI pools, and reduces nothing.
    siblings : object, default None
        The other groups' prepared frames for the SDI pool, ``{pid: SiblingFrames}`` or the
        shared-memory descriptor :class:`klip_tpe.parallel.SharedArrays` of them.
    card : bool, default False
        Whether the job may go to the card workers (:class:`klip_tpe.parallel.CardSpec`).  The
        runner marks the search's injected draws, and the pool sends the share of them its slot
        picks.
    """
    params: Dict[str, Any]
    injections: Optional[Sequence[Source]] = None
    k_scan: bool = False
    tag: str = ""
    fm_sources: Optional[Sequence[Source]] = None   # KLIP-FM test sources (model cube, never added to the data)
    extras: bool = False                            # also produce the nosub (clean pass) / cadi (injected pass) reference image
    threads: int = 1                                # this partition job's threads: the per-target KLIP loop and the per-frame steps (set by PartitionedReducer)
    #: ``"reduce"`` (the default) or ``"prepare"``: the latter returns the prepared frames
    #: (selected, injected, high-passed, binned) as :class:`SiblingFrames`, for the SDI pool
    #: of the other wavelength groups, and reduces nothing
    stage: str = "reduce"
    #: the other groups' prepared frames for the SDI pool: ``{pid: SiblingFrames}``, or the
    #: shared-memory descriptor :class:`klip_tpe.parallel.SharedArrays` of them
    siblings: Any = None
    #: the job may go to the card workers (:class:`klip_tpe.parallel.CardSpec`): the runner
    #: marks the search's injected draws, and the pool sends the share of them its slot picks
    card: bool = False


@dataclass
class SiblingFrames:
    """One wavelength group's frames as another group's reduction needs them for its SDI pool.

    They are prepared exactly as the group prepares its own, with the same selection, injections,
    high-pass and binning, and are not yet rescaled.

    Attributes
    ----------
    pid : object
        Identifier of the group's partition.
    lam_m : float
        Wavelength of the group, in meters.
    bcube : ndarray, shape (nb, ny, nx)
        The nb binned frames.
    bang : ndarray, shape (nb,)
        Parallactic angle of each binned frame, in degrees.
    mcube : ndarray or None, default None
        The KLIP-FM model cube through the same chain, when the request asked for KLIP-FM.
    """
    pid: Any
    lam_m: float
    bcube: np.ndarray                 # (n_binned, ny, nx)
    bang: np.ndarray                  # the binned angles
    mcube: Optional[np.ndarray] = None


@dataclass
class PreparedFrames:
    """The first half of :meth:`KLIPReducer.reduce`, everything up to the binned frames.

    Attributes
    ----------
    p : dict
        The request's parameters over the reducer's defaults.
    k : int
        Number of KL modes, at least 1.
    bin_ : int
        Temporal binning factor.
    filt : int
        Width of the high-pass box, in pixels, 1 or less for none.
    inrad : float
        Inner radius of the KLIP zone, in pixels.
    outrad : float
        Outer radius of the KLIP zone, in pixels.
    dth : float
        Maximum span of parallactic angle in a bin, in degrees (:meth:`KLIPReducer.dth_max_deg`).
    keep : ndarray of bool, shape (n,)
        The frame selection over the n raw frames.
    bcube : ndarray, shape (nb, ny, nx)
        The nb binned frames kept.
    bang : ndarray, shape (nb,)
        Parallactic angle of each binned frame, in degrees.
    grp : ndarray of int
        Bin number of each selected frame.
    bkeep : ndarray of bool
        True for each bin kept, one value per bin before the bins without data are dropped.
    mcube : ndarray or None
        The KLIP-FM model cube through the same chain, or None without KLIP-FM.
    hp_nan : bool
        Whether the high-pass leaves NaN pixels out (``hp_nan_aware``).
    hp_binned : bool
        Whether the frames were binned before the high-pass (``hp_order="binned"``).
    nth : int
        Threads of the job.
    meta : dict
        Notes on the reduction, for its result's ``meta``.
    ann : tuple of 2 floats or None, default None
        The ``(inner, outer)`` annulus in pixels the zone was made from, with the runner's edge
        padding.  ``angsep``'s arc and the bin's PA span are measured on it, and None makes the
        zone the annulus.
    bnorth : ndarray or None, default None
        True north of each kept bin, in degrees, the mean of its frames'.  None takes the reducer's
        ``truenorth`` for every bin.
    """
    p: Dict[str, Any]
    k: int
    bin_: int
    filt: int
    inrad: float
    outrad: float
    dth: float
    keep: np.ndarray
    bcube: np.ndarray
    bang: np.ndarray
    grp: np.ndarray
    bkeep: np.ndarray
    mcube: Optional[np.ndarray]
    hp_nan: bool
    hp_binned: bool
    nth: int
    meta: Dict[str, Any]
    #: the annulus the zone was made from, ``(inner, outer)`` px with the runner's edge
    #: padding: ``inrad``/``outrad`` are the KLIP zone, which ``zone_in``/``zone_out`` widen
    #: beyond it.  ``angsep``'s arc and the bin's PA span are measured on the annulus.  None
    #: (a caller that builds one itself): the zone is the annulus.
    ann: Optional[Tuple[float, float]] = None
    #: each kept bin's true north (deg), its frames' mean, when the frames carry their own
    #: (``Dataset.north``); None: the reducer's ``truenorth`` for every bin
    bnorth: Optional[np.ndarray] = None

    def siblings_payload(self, pid: Any, lam_m: float) -> SiblingFrames:
        """Package the binned frames for another group's SDI pool.

        Parameters
        ----------
        pid : object
            Identifier of this group's partition.
        lam_m : float
            Wavelength of this group, in meters.

        Returns
        -------
        frames : SiblingFrames
            Contiguous float32 copies of ``bcube`` and ``mcube``, with the angles.
        """
        return SiblingFrames(pid, float(lam_m), np.ascontiguousarray(self.bcube, np.float32),
                             np.asarray(self.bang, float),
                             None if self.mcube is None else np.ascontiguousarray(self.mcube, np.float32))


@dataclass
class ReductionResult:
    """The image a reduction returns, with its weight and notes.

    Attributes
    ----------
    image : ndarray, shape (ny, nx) or (nk, ny, nx)
        The derotated, combined image, or one per number of KL modes with ``k_scan``.
    weight : float, default 1.0
        Weight of the image in a combination of partitions.
    meta : dict
        Notes on how the reduction ran.
    fm_image : ndarray or None, default None
        KLIP-FM forward-modeled planet image, shape (ny, nx).
    nosub : ndarray or None, default None
        Derotated median stack without PSF subtraction, of the clean pass with ``extras``.
    cadi : ndarray or None, default None
        Classical ADI reference image of the injected pass with ``extras``.
    """
    image: np.ndarray                # (ny, nx) or (nk, ny, nx) when k_scan
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    fm_image: Optional[np.ndarray] = None    # KLIP-FM forward-modeled planet response (ny, nx)
    nosub: Optional[np.ndarray] = None       # derotate + median stack, no PSF subtraction (clean pass, extras=True)
    cadi: Optional[np.ndarray] = None        # classical ADI reference (injected pass, extras=True)


class Reducer:
    """Interface every reduction backend implements.

    A reducer turns a request's parameters and optional injections into a derotated, combined image
    (:meth:`reduce`).

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel, for the metric.
    fwhm : float
        FWHM of the PSF, in pixels, for the metric.
    lam_over_d_px : float
        lambda/D in pixels, the unit of ``angsep``.
    angle_convention : {"pa", "math"}
        How source angles are read, "pa" East of North or "math" counter-clockwise from +x.
    supports_kscan : bool
        Whether :meth:`reduce` can return a cube over k = 1 to ``k_klip`` in one call.
    supports_fm : bool
        Whether :meth:`reduce` can run KLIP-FM.
    name : str
        Name of the backend, for a run's record.
    """

    pxscale: float = 1.0
    fwhm: float = 4.0
    lam_over_d_px: float = 4.0
    angle_convention: str = "pa"
    supports_kscan: bool = False
    supports_fm: bool = False
    name: str = "reducer"

    def reduce(self, req: ReductionRequest) -> ReductionResult:  # pragma: no cover
        """Reduce the data under one request.

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult
            The image and its notes.
        """
        raise NotImplementedError

    def partitions(self) -> List[Any]:
        """Return the identifiers of the dataset's partitions, empty for one sequence.

        Returns
        -------
        pids : list
            The identifiers.
        """
        return []

    def partition_weight(self, pid) -> float:
        """Return a partition's weight in the combination.

        Parameters
        ----------
        pid : object
            Identifier of the partition.

        Returns
        -------
        weight : float
            The weight, 1 for this base class.
        """
        return 1.0

    def frame_angles(self, pid=None) -> Optional[np.ndarray]:
        """Return the frames' parallactic angles, for the reference-count check.

        Parameters
        ----------
        pid : object or None, default None
            Identifier of a partition, or None for the dataset.

        Returns
        -------
        angles : ndarray or None
            Angle of each frame in degrees, or None when unknown.
        """
        return None

    def frame_tags(self, pid=None) -> Optional[Dict[str, np.ndarray]]:
        """Return the frames' quality tags.

        Parameters
        ----------
        pid : object or None, default None
            Identifier of a partition, or None for the dataset.

        Returns
        -------
        tags : dict of str to ndarray or None
            The tags, or None when there are none.
        """
        return None

    def matched_filter_kernel(self, rho_as: float):
        """Return a measured matched filter for the metric, or None for its Gaussian.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        kernel : ndarray or None
            The kernel, None for this base class.
        """
        return None

    def cost_estimate(self, params: Dict[str, Any], k_scan: bool = False, n_inject: int = 0,
                      fm: bool = False) -> Optional[float]:
        """Estimate the single-thread seconds of one :meth:`reduce`, to schedule partition jobs.

        The jobs run the longest first, with threads by cost.

        Parameters
        ----------
        params : dict
            The request's parameters.
        k_scan : bool, default False
            Whether the reduction scans the number of KL modes.
        n_inject : int, default 0
            Number of injected sources.
        fm : bool, default False
            Whether the reduction runs KLIP-FM.

        Returns
        -------
        seconds : float or None
            The estimate, or None when unknown, and the jobs then keep a flat thread count in
            submission order.
        """
        return None

    def describe(self) -> Dict[str, Any]:
        """Return the reducer's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, ``pxscale``, ``fwhm``, ``angle_convention``, ``supports_kscan`` and
            ``supports_fm``.
        """
        return {"name": self.name, "pxscale": self.pxscale, "fwhm": self.fwhm,
                "angle_convention": self.angle_convention, "supports_kscan": self.supports_kscan,
                "supports_fm": self.supports_fm}


# ----------------------------------------------------------------------------
# reference KLIP reducer
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    """One pupil-tracking sequence, centered on the star.

    Attributes
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each, in float32.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    tags : dict of str to ndarray or None, default None
        Per-frame quality tags, ``corrs``, ``noises`` and ``coronoise``.
    texp : float, default 0.0
        Total on-source exposure time, in seconds, for the partition weights.
    name : str, default "data"
        Name of the sequence.
    meta : dict
        The loader's record of the data.
    ref_cube : ndarray or None, default None
        Registered frames of a PSF reference star, shape (nref, ny, nx), for RDI and ARDI.
    north : ndarray or None, default None
        Each frame's own true north, in degrees, used at derotation and in the injection azimuth.
        None takes the reducer's one angle.
    """

    cube: np.ndarray                       # (n, ny, nx) float32
    angles: np.ndarray                     # parallactic angle per frame (deg)
    tags: Optional[Dict[str, np.ndarray]] = None   # corrs / noises / coronoise per frame
    texp: float = 0.0                      # total on-source exposure (s), for weights
    name: str = "data"
    meta: Dict[str, Any] = field(default_factory=dict)
    ref_cube: Optional[np.ndarray] = None  # (nref, ny, nx) pre-registered PSF-reference-star frames (RDI/ARDI)
    #: each frame's true north (deg), the frames' own in place of the reducer's ``truenorth``,
    #: at derotation and in the injection azimuth.  None: the reducer's one angle.
    north: Optional[np.ndarray] = None

    @property
    def nframes(self) -> int:
        """Number of frames.
        """
        return int(self.cube.shape[0])


class _FrameCache:
    """One set of sources injected into a cube frame by frame, as reductions ask for frames.

    The lock covers filling frames and copying out a selection, so two reductions of the same
    sources on threads never read a frame another is still writing.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The cube the injected frames are written into.

    Attributes
    ----------
    done : ndarray of bool, shape (n,)
        True for each frame injected so far.
    lock : threading.Lock
        Lock over filling frames and copying them out.
    """

    __slots__ = ("cube", "done", "lock")

    def __init__(self, cube: np.ndarray):
        self.cube = cube
        self.done = np.zeros(cube.shape[0], bool)
        self.lock = threading.Lock()


class _CropCache:
    """What one set of sources adds to each frame, kept once the set comes back.

    A set seen for the first time is placed without keeping anything, since the search draws new
    sources for every reduction.  A set that comes back, such as param-verify's fixed set or a
    fixed-source search's set per annulus, keeps each frame's stamps from its second reduction on,
    so a later selection places only the frames it adds.  One set is kept at a time, the latest that
    came back.

    Attributes
    ----------
    seen : tuple or None
        The set seen last.
    key : tuple or None
        The set whose stamps are kept.
    frames : dict of int to list
        Each frame's stamps for the kept set.
    lock : threading.Lock
        Lock over the lookups.
    """

    __slots__ = ("seen", "key", "frames", "lock")

    def __init__(self):
        self.seen = None
        self.key = None
        self.frames: Dict[int, list] = {}
        self.lock = threading.Lock()

    def lookup(self, key) -> Tuple[bool, Optional[Dict[int, list]]]:
        """Look up a set of sources.

        Parameters
        ----------
        key : tuple
            The set's sources, as tuples.

        Returns
        -------
        first : bool
            True when the set is seen for the first time.
        frames : dict of int to list or None
            The dict the set's per-frame stamps go in, or None when they are not kept.
        """
        with self.lock:
            if key == self.key:
                return False, self.frames
            if key == self.seen:
                self.key, self.frames = key, {}
                return False, self.frames
            self.seen = key
            return True, None


def _flux_taken(orig: np.ndarray, edited: np.ndarray, angles: np.ndarray, sources: Sequence[Source], model,
                pxscale: float, *, truenorth, fallback, angle_convention: str,
                n_frames: int = 32) -> Tuple[float, int]:
    """Measure the fraction of sources' flux that left the frames when they were subtracted.

    It is ``sum(sign(m) (orig - edited)) / sum(|m|)``, with ``m`` the sources injected into zeros, on
    up to ``n_frames`` frames spread over the cube.  float32 rounding loses the part of a source that
    sits below half a unit in the last place of the pixel it lands on.  For sources added, pass the
    frames the other way round.

    Parameters
    ----------
    orig : ndarray, shape (n, ny, nx)
        The frames before the edit.
    edited : ndarray, shape (n, ny, nx)
        The frames after it.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    sources : sequence of Source
        The sources, with the contrasts they were subtracted at.
    model : InjectionModel
        The injection model.
    pxscale : float
        Plate scale, in arcsec per pixel.
    truenorth : float or ndarray
        True-north offset, in degrees, one value or one per frame.
    fallback : InjectionModel or None
        Model that supplies the stamp where ``model`` has none.
    angle_convention : {"pa", "math"}
        How the sources' angles are read.
    n_frames : int, default 32
        Maximum number of frames measured.

    Returns
    -------
    fraction : float
        Fraction of the flux that moved, 1 for all of it, NaN when the model puts no flux there.
    n : int
        Number of frames measured.
    """
    n = int(orig.shape[0])
    idx = np.unique(np.linspace(0, n - 1, min(n, max(int(n_frames), 1))).round().astype(np.intp))
    m = np.zeros((idx.size,) + tuple(orig.shape[-2:]), np.float32)
    tn = truenorth if np.ndim(truenorth) == 0 else np.asarray(truenorth, float)[idx]
    inject_frames(m, m, np.asarray(angles, float)[idx], list(sources), model, pxscale, None, truenorth=tn,
                  fallback=fallback, angle_convention=angle_convention)
    took = (np.asarray(orig[idx], np.float32).astype(np.float64)
            - np.asarray(edited[idx], np.float32).astype(np.float64))
    m64 = m.astype(np.float64)
    ok = np.isfinite(took) & (m64 != 0.0)
    den = float(np.sum(np.abs(m64[ok])))
    if not den > 0.0:
        return float("nan"), int(idx.size)
    return float(np.sum(np.sign(m64[ok]) * took[ok]) / den), int(idx.size)


def _taken_text(red, attr: str = "subtracted_kept") -> str:
    """Format what :func:`_flux_taken` measured for a reducer's subtracted or added sources.

    Parameters
    ----------
    red : Reducer
        The reducer.
    attr : {"subtracted_kept", "injected_kept"}, default "subtracted_kept"
        Attribute to read, for the sources subtracted or the sources added.

    Returns
    -------
    text : str
        Such as "n4 0.9999998, n6 0.9999997" for partitions, or one number, empty when nothing was
        measured.
    """
    kept = getattr(red, attr, None)
    items = list(kept.items()) if isinstance(kept, dict) else [(None, kept)]
    out = [(k, v) for k, v in items if v is not None and np.isfinite(v)]
    return ", ".join(f"{k} {v:.7f}" if k is not None else f"{v:.7f}" for k, v in out)


def _flag(v) -> bool:
    """Read a yes-or-no request parameter.

    Parameters
    ----------
    v : object
        The value.

    Returns
    -------
    flag : bool
        True for a true value, or for the strings "1", "yes", "true", "y" and "on" in any case.
    """
    if isinstance(v, str):
        return v.strip().lower() in ("1", "yes", "true", "y", "on")
    return bool(v)


def _zone_ext(p: Dict[str, Any]) -> Tuple[float, float]:
    """Return how far a request's KLIP zone reaches inside and outside the annulus.

    Parameters
    ----------
    p : dict
        The request's parameters.

    Returns
    -------
    zone_in : float
        Reach inside the annulus, in pixels, at least 0.
    zone_out : float
        Reach outside the annulus, in pixels, at least 0.
    """
    return (max(float(p.get("zone_in", 0.0) or 0.0), 0.0), max(float(p.get("zone_out", 0.0) or 0.0), 0.0))


def _zone_center(p: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return a request's ``zone_center`` as an ``(x, y)`` pixel, or None.

    Parameters
    ----------
    p : dict
        The request's parameters.

    Returns
    -------
    center : tuple of 2 floats or None
        The center, or None, which draws the zones about ``(nx/2, ny/2)``
        (:func:`klip_tpe.klip.zone_indices`).

    Raises
    ------
    ValueError
        If ``zone_center`` does not have two values.
    """
    c = p.get("zone_center")
    if c is None:
        return None
    c = [float(v) for v in c]
    if len(c) != 2:
        raise ValueError(f"zone_center is (x, y), not {p.get('zone_center')!r}")
    return (c[0], c[1])


class KLIPReducer(Reducer):
    """Reference annular KLIP/ADI reducer, built on :mod:`klip_tpe.klip`.

    The chain is frame selection, injection into the kept frames, high-pass, optional destriping at
    90 and 0 degrees, angle-aware binning, annular KLIP (with an RDI or ARDI basis and KLIP-FM when
    asked), a second high-pass, derotation, combination and the optional ``mask_fn``.  The selection
    reads only the frame tags, so injecting the kept frames alone changes no number.  Everything
    before KLIP runs one bin group at a time (:meth:`_prep_fused`), which reads the cube once per
    reduction and gives the whole-cube steps' numbers to the bit.

    Parameters
    ----------
    data : Dataset
        The sequence to reduce.
    pxscale : float
        Plate scale, in arcsec per pixel.
    lam_m : float
        Wavelength, in meters.
    diam_m : float
        Telescope diameter, in meters.
    injection_model : InjectionModel or None, default None
        Model of the injected companions.  None allows no injection or KLIP-FM.
    fallback_model : InjectionModel or None, default None
        Model that supplies the stamp where the injection model has none.
    truenorth : float, default 0.0
        True-north offset, in degrees.  Frames that carry their own (``data.north``) override it.
    fwhm_px : float or None, default None
        FWHM of the PSF, in pixels.  None takes ``1.028`` lambda/D.
    zone_pad : float, default 2.0
        Padding of the KLIP zone beyond each annulus edge the runner marks, in pixels.
    outrad_cap : float, default 70.0
        Maximum outer radius of a zone, in pixels.
    defaults : dict or None, default None
        Request parameters that replace the built-in defaults.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read.
    mask_fn : callable or None, default None
        Legacy post-mask hook ``image -> image`` (``block_burn``, ``block_airy``), applied to the
        final science image only.

    Attributes
    ----------
    lam_over_d_px : float
        lambda/D in pixels, ``(lam_m / diam_m) * 206265 / pxscale``.
    fwhm : float
        FWHM of the PSF, in pixels.
    truenorth : float
        True-north offset, in degrees.
    north_frames : ndarray or None
        Each frame's own true north, in degrees, when the frames carry values that differ.
    defaults : dict
        Default request parameters.

    Raises
    ------
    ValueError
        If ``data.north`` does not hold one finite value per frame.

    Notes
    -----
    The request's parameters, all optional, default to the behavior of the IDL predecessor's
    ``reduce_near_2``:

    ``k_klip``, ``n_ang``, ``inrad``, ``outrad``, ``angsep``, ``anglemax``, ``fast``, ``spat_mean``, ``temp_mean``, ``n_min_ref``
        The KLIP settings (:class:`~klip_tpe.klip.KLIPParams`).
    ``bin``, ``filter``
        Temporal binning factor and width of the high-pass box in pixels.
    ``corr_thresh``, ``noise_max``, ``coronoise_max``
        The frame selection (:func:`~klip_tpe.klip.frame_selection_mask`).
    ``comb_type``
        Combination of the derotated frames, "nwadi", "mean" or "median".
    ``use_rdi``, ``rdi_mode``
        Whether to build the basis from the reference cube, as "rdi" or "ardi".
    ``do_destripe``, ``destripe_clip``
        Destriping of the frames and its clipping level (:func:`~klip_tpe.klip.destripe`).
    ``fm_selfsub``
        Whether KLIP-FM includes its self-subtraction terms, True by default.
    ``hp_nan_aware``
        False by default, the predecessor's ``smooth`` without ``/nan``, under which a frame with a
        NaN pixel comes out NaN all over.  True leaves each NaN pixel NaN and filters the rest, for
        frames with NaN at their registration edges, such as ALES's.
    ``prep_fused``
        True by default, the fused path.  False runs a whole cube per step, the reference the fused
        path is tested against.
    ``klip_gram``
        "target" by default, or "zone" (:attr:`klip_tpe.klip.KLIPParams.gram`), about twice as fast
        per target at small bins and equal to float64 rounding (:func:`set_klip_gram`).
    ``zone_in``, ``zone_out``
        How far the KLIP zone reaches inside and outside the annulus, in pixels, 0 by default.  The
        ALES reduction of MWC 758 fits one 5-27 px annulus this way, while ``angsep``'s arc and the
        bin's PA span stay those of the annulus.
    ``post_hp``
        True by default, the residuals high-passed again before derotation, as ``reduce_near_2``
        does.  False derotates them as KLIP leaves them, as the ALES reduction's ADI images do.
    ``zone_center``, ``zone_angles``
        Center and angle convention of the zones (:func:`set_zone_center`,
        :func:`~klip_tpe.klip.zone_indices`).
    """

    name = "klip"
    supports_kscan = True
    supports_fm = True
    #: A backend that overrides :meth:`_subtract` builds its own basis and does NOT apply a
    #: searched reference library unless it declares it here.  pyKLIP, VIP and the custom
    #: ``FunctionReducer`` all override it; pyKLIP in particular hands the WHOLE reference
    #: cube to ``klip_parallelized`` and lets it pick.
    _honours_reference_library = False

    @property
    def supports_reference_library(self) -> bool:
        """Whether ``nkeep_<group>`` changes the reduction, for the built-in KLIP or a backend that declares ``_honours_reference_library``.
        """
        return (type(self)._subtract is KLIPReducer._subtract
                or bool(getattr(type(self), "_honours_reference_library", False)))

    def __init__(self, data: Dataset, pxscale: float, lam_m: float, diam_m: float,
                 injection_model: Optional[InjectionModel] = None,
                 fallback_model: Optional[InjectionModel] = None, truenorth: float = 0.0,
                 fwhm_px: Optional[float] = None, zone_pad: float = 2.0, outrad_cap: float = 70.0,
                 defaults: Optional[Dict[str, Any]] = None, angle_convention: str = "pa",
                 mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.data = data
        self.pxscale = float(pxscale)
        self.lam_m, self.diam_m = float(lam_m), float(diam_m)
        self.lam_over_d_px = (lam_m / diam_m) * 206265.0 / pxscale
        self.fwhm = float(fwhm_px) if fwhm_px else 1.028 * self.lam_over_d_px
        self.model = injection_model
        self.fallback = fallback_model
        # the frames' own true north (Dataset.north) where it varies; a list of one value is
        # that one angle, so it reduces as the constant does, to the bit
        north = getattr(data, "north", None)
        self.north_frames: Optional[np.ndarray] = None
        if north is not None:
            north = np.asarray(north, float).ravel()
            if north.shape[0] != data.nframes:
                raise ValueError(f"{getattr(data, 'name', 'data')}: {north.shape[0]} true-north values for "
                                 f"{data.nframes} frames")
            if not np.all(np.isfinite(north)):
                raise ValueError(f"{getattr(data, 'name', 'data')}: a true-north value is not finite")
            if north.size and np.all(north == north[0]):
                truenorth = float(north[0])
            else:
                self.north_frames = north
        self.truenorth = float(truenorth)
        self.zone_pad = float(zone_pad)
        self.outrad_cap = float(outrad_cap)
        self.angle_convention = angle_convention
        self.mask_fn = mask_fn
        self.defaults = {"k_klip": 10, "bin": 1, "filter": 0, "n_ang": 1, "inrad": 0.0,
                         "outrad": outrad_cap, "angsep": 0.0, "anglemax": 360.0,
                         "corr_thresh": 0.95, "noise_max": 2.0, "coronoise_max": 2.0,   # reduce_near_2 defaults; 2.0 = historical no-op
                         "fast": False, "spat_mean": False, "temp_mean": False,
                         "n_min_ref": 10, "comb_type": "nwadi",
                         "use_rdi": False, "rdi_mode": "ardi", "do_destripe": False, "destripe_clip": 0.0,
                         "fm_selfsub": True, "hp_nan_aware": False, "prep_fused": True,
                         "klip_gram": "target"}
        self.defaults.update(defaults or {})
        self._inj_cache: Dict[Tuple, _FrameCache] = {}
        self._fm_cache: Dict[Tuple, _FrameCache] = {}
        #: the fused path's stamps, per kind: the injected sources and the KLIP-FM model's
        self._crops: Dict[str, _CropCache] = {"inj": _CropCache(), "fm": _CropCache()}
        #: the level of the cube's brightest pixels for the float32 headroom check, computed once
        self._peak: Optional[float] = None
        self._inj_lock = threading.Lock()
        self._cost_cache: Dict[Tuple, float] = {}
        #: optional searched reference library (:mod:`klip_tpe.reflib`).  Set through
        #: :meth:`set_reference_library`; when it is None every RDI/ARDI reduction uses
        #: the whole reference cube as one shared basis, which is the previous behavior.
        self._reflib_spec: Optional[Dict[str, Any]] = None
        self._reflib_cache: Dict[Tuple, Any] = {}

    # -- helpers -------------------------------------------------------------
    def partitions(self) -> List[Any]:
        """Return no partitions, the dataset being one sequence.

        Returns
        -------
        pids : list
            Empty.
        """
        return []

    def frame_angles(self, pid=None):
        """Return the frames' parallactic angles.

        Parameters
        ----------
        pid : object or None, default None
            Identifier of a partition, ignored.

        Returns
        -------
        angles : ndarray
            Angle of each frame, in degrees.
        """
        return self.data.angles

    def frame_tags(self, pid=None):
        """Return the frames' quality tags.

        Parameters
        ----------
        pid : object or None, default None
            Identifier of a partition, ignored.

        Returns
        -------
        tags : dict of str to ndarray or None
            The tags of ``data``.
        """
        return self.data.tags

    def partition_weight(self, pid=None) -> float:
        """Return the dataset's weight, the square root of its exposure time.

        Parameters
        ----------
        pid : object or None, default None
            Identifier of a partition, ignored.

        Returns
        -------
        weight : float
            ``sqrt(texp)``, or 1 without an exposure time.
        """
        return float(np.sqrt(self.data.texp)) if self.data.texp > 0 else 1.0

    def matched_filter_kernel(self, rho_as: float):
        """Return the injection model's measured matched filter at a separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        kernel : ndarray or None
            The kernel, or None without an injection model or a measured filter.
        """
        if self.model is None:
            return None
        return self.model.matched_filter_kernel(rho_as, self.fwhm)

    def dth_max_deg(self, outrad: float) -> float:
        """Return the maximum span of parallactic angle in one temporal bin.

        Parameters
        ----------
        outrad : float
            Outer radius of the annulus, in pixels.

        Returns
        -------
        dth : float
            Span in degrees that moves a source by half a FWHM, ``0.5 * 1.028`` lambda/D, at
            ``outrad``.
        """
        return float(np.rad2deg(0.5 * 1.028 * self.lam_over_d_px / max(outrad, 1.0)))

    def cost_estimate(self, params: Dict[str, Any], k_scan: bool = False, n_inject: int = 0,
                      fm: bool = False) -> Optional[float]:
        """Estimate the single-thread seconds of one :meth:`reduce`, for scheduling only.

        The estimate counts the injection and high-pass of every kept frame, the derotation and
        combination of every binned frame, and the per-target KL bases (binned frames times references
        squared times zone pixels).  The coefficients come from 2024-frame, 150 x 150 timings on one
        2.8 GHz core, and only the ratios between jobs are used.

        Parameters
        ----------
        params : dict
            The request's parameters.
        k_scan : bool, default False
            Whether the reduction scans the number of KL modes.
        n_inject : int, default 0
            Number of injected sources.
        fm : bool, default False
            Whether the reduction runs KLIP-FM.

        Returns
        -------
        seconds : float or None
            The estimate, or None when it fails.
        """
        try:
            p = dict(self.defaults)
            p.update({k: v for k, v in params.items() if v is not None})
            k = int(max(round(p["k_klip"]), 1))
            bin_ = int(max(round(p["bin"]), 1))
            filt = int(round(p["filter"]))
            n_ang = int(max(round(p["n_ang"]), 1))
            inrad = max(float(p["inrad"]), 0.0)
            outrad = min(float(p["outrad"]), self.outrad_cap)
            z_in, z_out = _zone_ext(p)                  # the KLIP zone beyond the annulus
            zin, zout = max(inrad - z_in, 0.0), min(outrad + z_out, self.outrad_cap)
            # a searched library's counts: more rows in every target's basis, and the SDI
            # pool's rescaling of every sibling frame
            spec = self._reflib_spec or {}
            extra = 0
            n_sdi = 0
            if spec:
                for name in ["sdi"] + list((spec.get("groups") or {spec.get("ref_group", "psfref"): None}).keys()):
                    v = p.get(f"nkeep_{name}")
                    if v is not None:
                        extra += int(max(round(float(v)), 0))
                        if name == "sdi":
                            n_sdi = int(max(round(float(v)), 0))
            key = (k, bin_, filt > 1, n_ang, round(inrad, 2), round(outrad, 2), p["corr_thresh"], p["noise_max"],
                   p["coronoise_max"], round(float(p["angsep"]), 3), round(float(p["anglemax"]), 2),
                   bool(p["fast"]), bool(k_scan), int(n_inject), bool(fm), int(extra), int(n_sdi),
                   round(zin, 2), round(zout, 2))
            hit = self._cost_cache.get(key)
            if hit is not None:
                return hit
            ds = self.data
            keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"],
                                        p["coronoise_max"], bin_, k)
            ang = np.asarray(ds.angles, float)[keep]
            n_kept = int(ang.size)
            m = int(bin_groups(ang, bin_, self.dth_max_deg(outrad)).max()) + 1 if n_kept else 0
            npix = int(np.prod(ds.cube.shape[1:]))
            zone = np.pi * max(zout ** 2 - zin ** 2, 1.0)
            span = float(ang.max() - ang.min()) if n_kept > 1 else 0.0
            sep = float(p["angsep"]) * abs(arcdist_deg(inrad, outrad, self.lam_over_d_px))
            amax = float(p["anglemax"])
            if (bool(p["fast"]) or (sep <= 0 and amax >= span)) and extra == 0:
                klip = 2.2e-10 * m * m * zone                        # one basis per sector
            else:
                frac = 1.0 if span <= 0 else min(1.0, max(2.0 * (amax - sep), 0.0) / span)
                mref = max(min(float(m), m * frac), 4.0) + extra
                klip = 2.2e-10 * m * mref * mref * zone + 2e-4 * m * n_ang
            if n_sdi > 0 and spec.get("sdi"):
                lenders = max(len(spec["sdi"].get("lenders") or []) - 1, 1)
                klip += 1.3e-7 * lenders * m * npix * (2 if fm else 1)   # the rescaling of every sibling frame
            raw = (1.5e-8 if filt > 1 else 2e-9) * n_kept * npix     # high-pass, copies, binning
            post = 3.6e-7 * m * npix                                 # post high-pass, derotate, combine
            cost = raw + post * (k if k_scan else 1) + klip
            if fm:
                cost += raw + post + klip                            # the model cube follows the data
            if n_inject:
                cost += 1.2e-4 * n_inject * n_kept                    # each source into each kept frame
            if len(self._cost_cache) > 4096:
                self._cost_cache.clear()
            self._cost_cache[key] = float(cost)
            return float(cost)
        except Exception:
            return None

    def _inject(self, sources: Sequence[Source], keep: Optional[np.ndarray] = None,
                threads: int = 1) -> np.ndarray:
        """Return the cube with sources injected, every frame or the kept frames only.

        Only the frames asked for are injected, since the selection depends on the tags and a frame it
        drops never needs its companions.  The latest set of sources stays cached with the frames
        injected so far, so a reduction with the same sources and another selection injects only the
        frames it adds.  Each frame gets the numbers :func:`klip_tpe.injection.inject_sources` gives it.

        Parameters
        ----------
        sources : sequence of Source
            The sources.
        keep : ndarray of bool or None, default None
            The frames to return, ``cube[keep]`` in order.  None returns every frame.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        cube : ndarray
            The injected frames, in float32.

        Raises
        ------
        ValueError
            If the reducer has no injection model.

        Warns
        -----
        RuntimeWarning
            When a new set of sources is too faint to survive float32 rounding.
        """
        key = tuple(s.as_tuple() for s in sources)
        with self._inj_lock:
            ent = self._inj_cache.get(key)
            if ent is None:
                if self.model is None:
                    raise ValueError("KLIPReducer needs an injection_model to inject sources")
                # frames are written as they are asked for: a frame never asked for costs no
                # memory, and no frame is read before it is written
                ent = _FrameCache(np.empty(self.data.cube.shape, np.float32))
                self._inj_cache = {key: ent}       # keep only the latest set of sources
                _check_float32_headroom(self.data.cube, sources, self.model, self.fallback,
                                        peak=self.cube_peak())
        return self._fill(ent, self.data.cube, sources, keep, threads)

    def model_cube(self, sources: Sequence[Source], keep: Optional[np.ndarray] = None,
                   threads: int = 1) -> np.ndarray:
        """Return the KLIP-FM planet-model cube, the sources injected into zero frames.

        The injection code and conventions are those of the data.

        Parameters
        ----------
        sources : sequence of Source
            The KLIP-FM test sources.
        keep : ndarray of bool or None, default None
            The frames to return, as for :meth:`_inject`.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        mcube : ndarray
            The model frames, in float32.

        Raises
        ------
        ValueError
            If the reducer has no injection model.

        Notes
        -----
        It does what ``reduce_near_2`` L1441-1494 does in the IDL predecessor.
        """
        key = tuple(s.as_tuple() for s in sources)
        with self._inj_lock:
            ent = self._fm_cache.get(key)
            if ent is None:
                if self.model is None:
                    raise ValueError("KLIPReducer needs an injection_model to build the FM model cube")
                ent = _FrameCache(np.zeros(self.data.cube.shape, np.float32))
                self._fm_cache = {key: ent}
        return self._fill(ent, ent.cube, sources, keep, threads)

    def _fill(self, ent: _FrameCache, base: np.ndarray, sources: Sequence[Source],
              keep: Optional[np.ndarray], threads: int) -> np.ndarray:
        """Inject the frames a cache entry does not hold yet, and return the frames asked for.

        Parameters
        ----------
        ent : _FrameCache
            The cache entry, filled in place.
        base : ndarray
            Frames the sources are added to.
        sources : sequence of Source
            The sources.
        keep : ndarray of bool or None
            The frames asked for, or None for all.
        threads : int
            Number of threads.

        Returns
        -------
        cube : ndarray
            The entry's cube, or its frames in ``keep``.
        """
        want = np.ones(ent.done.size, bool) if keep is None else np.asarray(keep, bool)
        with ent.lock:
            todo = np.flatnonzero(want & ~ent.done)
            if todo.size:
                inject_frames(ent.cube, base, self.data.angles, sources, self.model, self.pxscale, todo,
                              truenorth=self.injection_north(), fallback=self.fallback,
                              angle_convention=self.angle_convention, threads=threads)
                ent.done[todo] = True
            return ent.cube if want.all() else ent.cube[want]

    def __getstate__(self):
        # the injection caches are scratch, and a lock does not pickle
        """Return the state to pickle, without the injection caches and the lock.

        Returns
        -------
        state : dict
            The reducer's attributes.
        """
        d = self.__dict__.copy()
        d.update(_inj_cache={}, _fm_cache={})
        d.pop("_inj_lock", None)
        d.pop("_crops", None)
        return d

    def __setstate__(self, d):
        """Restore a pickled state, with new caches and a new lock.

        Parameters
        ----------
        d : dict
            The state.
        """
        self.__dict__.update(d)
        self._inj_lock = threading.Lock()
        self._crops = {"inj": _CropCache(), "fm": _CropCache()}
        self.__dict__.setdefault("_peak", None)
        self.__dict__.setdefault("north_frames", None)

    # -- the true north ------------------------------------------------------
    def injection_north(self):
        """Return the true north the injection takes, each frame's own or the one angle.

        Returns
        -------
        north : float or ndarray
            ``north_frames``, indexed as ``data.angles``, or ``truenorth``.
        """
        nf = getattr(self, "north_frames", None)
        return self.truenorth if nf is None else nf

    def binned_north(self, keep: np.ndarray, grp: np.ndarray, bkeep: np.ndarray, bin_: int) -> Optional[np.ndarray]:
        """Return each kept bin's true north, the mean of its frames'.

        Parameters
        ----------
        keep : ndarray of bool
            The frame selection.
        grp : ndarray of int
            Bin number of each selected frame.
        bkeep : ndarray of bool
            The bins kept.
        bin_ : int
            Binning factor.

        Returns
        -------
        north : ndarray or None
            True north of each kept bin, in degrees, or None when the frames share ``truenorth``.
        """
        nf = getattr(self, "north_frames", None)
        if nf is None:
            return None
        keep = np.asarray(keep, bool)
        n_ = nf if keep.all() else nf[keep]
        return bin_values(n_, grp, bin_)[np.asarray(bkeep, bool)]

    @property
    def supports_frame_north(self) -> bool:
        """Whether the derotation takes each bin's own north, for the built-in KLIP or a backend that declares ``_honours_frame_north``.
        """
        return (type(self)._subtract is KLIPReducer._subtract
                or bool(getattr(type(self), "_honours_frame_north", False)))

    def north_record(self) -> Dict[str, Any]:
        """Return what a run records of the true north this reducer used.

        Returns
        -------
        rec : dict
            The loader's record (``data.meta["north"]``) with the values' range over the frames, or the
            one angle.
        """
        rec = dict((getattr(self.data, "meta", None) or {}).get("north") or {})
        north = getattr(self.data, "north", None)
        if north is None:
            rec.setdefault("source", "constant")
            rec["deg"] = self.truenorth
            return rec
        v = np.asarray(north, float)
        rec.setdefault("source", "per frame")
        rec.update(frames=int(v.size), min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
        return rec

    def _check_frame_north(self) -> None:
        """Refuse frames with their own true north on a backend that derotates by one angle.

        Raises
        ------
        ValueError
            If the frames carry their own north and the backend does not take it.
        """
        if getattr(self, "north_frames", None) is not None and not self.supports_frame_north:
            raise ValueError(f"{type(self).__name__} derotates every frame by one angle, and "
                             f"{getattr(self.data, 'name', 'the data')} has a true north per frame: reduce it "
                             f"with the built-in KLIP, or with the frames' north set to one angle")

    def cube_peak(self) -> float:
        """Return the level of the cube's brightest pixels, computed once.

        The float32 headroom check runs for every new set of sources, and its strided sample touches
        most of the cube's cache lines (:func:`~klip_tpe.injection.cube_peak`).

        Returns
        -------
        peak : float
            The level, in data units.
        """
        if getattr(self, "_peak", None) is None:
            self._peak = cube_peak(self.data.cube)
        return self._peak

    def subtract_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Take point sources out of the data once, before any reduction.

        Each source goes into every frame of ``data.cube`` with the negative of its contrast, through
        :func:`~klip_tpe.injection.inject_frames` with this reducer's model, fallback, true north,
        center and angle convention.  Every reduction then works on the edited frames, except the
        KLIP-FM model cube, which is built from zeros (:meth:`model_cube`).  Call it before the worker
        pool starts (:meth:`PartitionedReducer.subtract_sources` checks), since the workers keep the
        reducers they were forked with.

        Parameters
        ----------
        sources : sequence
            The sources, each a :class:`~klip_tpe.metrics.Source` or ``(rho_arcsec, pa_deg,
            contrast)``, such as :func:`~klip_tpe.injection.read_sources` gives.  A source of contrast 0
            is skipped.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        n : int
            Number of sources subtracted.

        Raises
        ------
        ValueError
            If a source lies beyond the frame or the reducer has no injection model.

        Warns
        -----
        RuntimeWarning
            When float32 rounding kept more than 0.1 % of the sources' flux in the frames.

        Notes
        -----
        The edited frames are a copy, so a dataset the caller still holds is not changed.  The edit is
        kept in ``data_edits`` and in :meth:`describe`, and the fraction of the flux that left the
        frames in ``subtracted_kept``.  In a chain whose contrasts pass through 0, the faintest sources
        lose part of their own flux and carry almost none of the chain's.
        """
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n, kept = self._edit_frames(src, -1.0, threads, "subtract_sources")
        if n:
            self.subtracted_kept = kept
        rec = dict(getattr(self, "data_edits", None) or {})
        rec["subtracted_sources"] = list(rec.get("subtracted_sources") or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    def inject_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Add point sources to the data once, before any reduction, to see what a reduction does to them.

        It is :meth:`subtract_sources` with the opposite sign.  From then on the sources are part of the
        data, for the clean pass, each draw, the reference ranking, validation and param-verify, but
        not for the KLIP-FM model cube.  It runs after :meth:`subtract_sources` when both are asked
        for.

        Parameters
        ----------
        sources : sequence
            The sources, each a :class:`~klip_tpe.metrics.Source` or ``(rho_arcsec, pa_deg,
            contrast)``.  A source of contrast 0 is skipped.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        n : int
            Number of sources added.

        Raises
        ------
        ValueError
            If a source lies beyond the frame or the reducer has no injection model.

        Warns
        -----
        RuntimeWarning
            When float32 rounding lost more than 0.1 % of the sources' flux.

        Notes
        -----
        The edit is kept in ``data_edits`` as ``injected_sources``, and the fraction of the flux that
        reached the frames in ``injected_kept``.
        """
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n, kept = self._edit_frames(src, 1.0, threads, "inject_sources")
        if n:
            self.injected_kept = kept
        rec = dict(getattr(self, "data_edits", None) or {})
        rec["injected_sources"] = list(rec.get("injected_sources") or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    def _edit_frames(self, src: Sequence[Source], sign: float, threads: int, what: str) -> Tuple[int, float]:
        """Add sources to every frame at a sign times their contrast, on a float32 copy of the data.

        The reducer's data is replaced by the edited copy, and every cache built from the old frames is
        cleared.

        Parameters
        ----------
        src : sequence of Source
            The sources.
        sign : float
            -1 to take them out, +1 to put them in.
        threads : int
            Number of threads.
        what : str
            Name of the caller, for messages.

        Returns
        -------
        n : int
            Number of sources applied, those with a contrast other than 0.
        fraction : float
            Fraction of their flux that moved (:func:`_flux_taken`).

        Raises
        ------
        ValueError
            If a source lies beyond the frame or the reducer has no injection model.

        Warns
        -----
        RuntimeWarning
            When float32 rounding kept or lost more than 0.1 % of the flux.
        """
        ny, nx = self.data.cube.shape[-2:]
        far = [s for s in src if s.rho / self.pxscale > max(ny, nx)]
        if far:
            raise ValueError(f"source at rho={far[0].rho:g} arcsec is {far[0].rho / self.pxscale:.0f} px from the "
                             f"star, beyond the {nx} x {ny} px frame: rho is in arcsec (pixel scale "
                             f"{self.pxscale:g} arcsec/px)")
        own = [s for s in src if float(s.contrast) != 0.0]
        todo = [Source(s.rho, s.theta, sign * float(s.contrast)) for s in own]
        if not todo:
            return 0, float("nan")
        if self.model is None:
            raise ValueError(f"{what} needs an injection_model")
        from dataclasses import replace
        orig = self.data.cube
        cube = np.array(orig, dtype=np.float32, order="C", copy=True)
        inject_frames(cube, cube, self.data.angles, todo, self.model, self.pxscale, None,
                      truenorth=self.injection_north(), fallback=self.fallback,
                      angle_convention=self.angle_convention, threads=max(int(threads), 1))
        before, after = (orig, cube) if sign < 0 else (cube, orig)
        kept, k = _flux_taken(before, after, self.data.angles, own, self.model, self.pxscale,
                              truenorth=self.injection_north(), fallback=self.fallback,
                              angle_convention=self.angle_convention)
        if np.isfinite(kept) and abs(kept - 1.0) > 1e-3:
            warnings.warn(f"{what}: {kept:.5f} of the sources' flux {'left' if sign < 0 else 'reached'} the frames "
                          f"(measured on {k} of {orig.shape[0]} frames), float32 rounding "
                          f"{'kept' if sign < 0 else 'lost'} the rest.  flux_unit is {self.model.flux_unit:.4g} -- if "
                          f"that is a normalised template's own sum rather than the star's flux in the science "
                          f"frames' units, set star_flux (see generic.star_flux_from_aperture_photometry).",
                          RuntimeWarning, stacklevel=3)
        self.data = replace(self.data, cube=cube)
        # everything worked out from the old frames goes
        self._peak = None
        self._inj_cache, self._fm_cache = {}, {}
        self._crops = {"inj": _CropCache(), "fm": _CropCache()}
        self._reflib_cache = {}
        return len(todo), kept

    def set_reference_library(self, *, partition, groups=None, metric: str = "cc",
                             shift_px: int = 1, n_min_ref: int = 2,
                             min_keep: Optional[Dict[str, int]] = None,
                             ref_group: str = "psfref",
                             similarity: Optional[np.ndarray] = None,
                             rotation: bool = False,
                             searched: Optional[Sequence[str]] = None,
                             sdi: Optional[Dict[str, Any]] = None) -> None:
        """Make the reference library a searched object rather than a fixed basis.

        The similarity matrix that ranks each target's references depends on the binned cube, the
        high-pass width and the annulus, so it is built when first needed and cached per combination.
        With ``bin`` fixed, as in the MWC 758 run, that is one matrix for the whole search.

        Parameters
        ----------
        partition : sequence
            One label per unbinned science frame, such as its roll, night or channel.  A target never
            draws references that share its label, which stops a roll subtracting its own companion.
        groups : dict of str to sequence of int or None, default None
            Named external pools, as rows of ``data.ref_cube``.  None makes the whole cube one pool,
            named ``ref_group``.
        metric : str, default "cc"
            Similarity measure of the ranking (:func:`klip_tpe.reflib.similarity_matrix`).
        shift_px : int, default 1
            Range of the similarity's shift search, in pixels.
        n_min_ref : int, default 2
            Minimum number of references for a target.
        min_keep : dict of str to int or None, default None
            Minimum count of each pool.
        ref_group : str, default "psfref"
            Name of the single external pool when ``groups`` is None.
        similarity : ndarray or None, default None
            Matrix of shape (n_library, n_target) used instead of computing one, such as the IDL
            predecessor's ``F430M_refmetrics_cc.fits``.  A difference in the result is then a difference
            in the selection, not in how the two code bases measure correlation.
        rotation : bool, default False
            Whether the ADI rule selects from the science pool per target and per annulus, by
            ``angsep`` and ``anglemax`` over the binned angles, for one continuously rotating sequence.
            With every eligible science frame kept and ``nkeep_<ref_group> = 0`` the reduction is plain
            ADI to the bit.
        searched : sequence of str or None, default None
            Pools whose counts become search dimensions.  A pool left out puts every eligible frame in
            the basis, and None searches every pool.
        sdi : dict or None, default None
            The pool of the other wavelength groups' frames, rescaled to this group's wavelength and
            counted by ``nkeep_sdi``, as ``{"n_frames": frames the groups hold, "cap": maximum count,
            "lenders": the groups' identifiers}``.  The frames arrive with each request
            (``ReductionRequest.siblings``), so this only declares the pool.

        Raises
        ------
        NotImplementedError
            If the backend does not apply a searched library.
        ValueError
            If ``sdi`` has no frames or no cap, or ``searched`` names an unknown pool.

        Notes
        -----
        ``searched=("psfref",)`` with ``rotation`` is ADI plus a searched number of reference-star
        frames, the ALES form.  A target's eligible SDI rows are those whose planet has moved the arc
        ``angsep`` asks for, by rotation and scaling together (:func:`klip_tpe.rescale.displacement_px`),
        within ``anglemax`` of rotation.
        """
        if not self.supports_reference_library:
            # This used to succeed on every backend: the counts were added to the search
            # space from here, and a backend with its own basis then ignored them.  GO 1386
            # MIRI run v6 searched nkeep_altroll / nkeep_psfref for 600 evaluations on
            # pyKLIP, and four libraries from pure ADI to pure RDI gave bit-identical scores
            # on 40 common draws -- the "elected" counts were where TPE left two dead
            # dimensions.  A searched library that is not applied must not be searchable.
            raise NotImplementedError(
                f"{type(self).__name__} ({self.name!r}) does not apply a searched reference "
                f"library: its _subtract builds its own basis from the whole reference cube, "
                f"so nkeep_* would be searched and change nothing.  Use the built-in engine "
                f"(backend 'klip') for a searched library, or search this backend's own "
                f"selection instead (pyKLIP: its ADI / RDI / ADI+RDI mode).")
        pools = ["altroll"] + (["sdi"] if sdi else []) + list((groups or {ref_group: None}).keys())
        if sdi:
            sdi = {"n_frames": int(sdi.get("n_frames", 0)), "cap": int(sdi.get("cap", 100)),
                   "lenders": list(sdi.get("lenders", []))}
            if sdi["n_frames"] < 1 or sdi["cap"] < 1:
                raise ValueError(f"an SDI pool needs frames and a cap above 0, got {sdi}")
        if searched is not None:
            searched = tuple(str(s) for s in searched)
            bad = [s for s in searched if s not in pools]
            if bad:
                raise ValueError(f"searched pools {bad} are not among {pools}")
        self._reflib_spec = {"partition": np.asarray(list(partition)), "groups": groups,
                             "metric": metric, "shift_px": int(shift_px),
                             "n_min_ref": int(n_min_ref), "min_keep": dict(min_keep or {}),
                             "ref_group": ref_group, "rotation": bool(rotation), "searched": searched,
                             "sdi": sdi or None,
                             "similarity": None if similarity is None else np.asarray(similarity, float)}
        self._reflib_cache.clear()

    def reference_params(self):
        """Return the searched counts this reducer's library adds to the search space.

        Returns
        -------
        params : list
            One parameter per searched pool (:meth:`set_reference_library`), empty without a library.
        """
        if self._reflib_spec is None or not self.supports_reference_library:
            return []
        searched = self._reflib_spec.get("searched")
        return [p for p in self._reflib_stub().params()
                if searched is None or p.name[len("nkeep_"):] in searched]

    def _reflib_stub(self):
        """Build a library over the unbinned geometry, for the search space before any reduction runs.

        Only the group sizes matter here, so the similarity is a placeholder of the right shape.

        Returns
        -------
        lib : ReferenceLibrary
            The placeholder library.
        """
        from .reflib import ReferenceGroup, ReferenceLibrary
        spec = self._reflib_spec
        n_sci = int(self.data.cube.shape[0])
        ref = self.data.ref_cube
        n_ref = 0 if ref is None else int(np.asarray(ref).shape[0])
        groups = [ReferenceGroup("altroll", np.arange(n_sci), partition=spec["partition"],
                                 min_keep=spec["min_keep"].get("altroll", 0),
                                 rotation=bool(spec.get("rotation", False)))]
        n_sdi = 0
        if spec.get("sdi"):
            n_sdi = int(spec["sdi"]["n_frames"])
            groups.append(ReferenceGroup("sdi", n_sci + np.arange(n_sdi), min_keep=spec["min_keep"].get("sdi", 0),
                                         max_keep=min(int(spec["sdi"]["cap"]), n_sdi), scales=np.ones(n_sdi)))
        for name, rows in (spec["groups"] or {spec["ref_group"]: np.arange(n_ref)}).items():
            rows = np.asarray(rows, int)
            if rows.size:
                groups.append(ReferenceGroup(name, n_sci + n_sdi + rows,
                                             min_keep=spec["min_keep"].get(name, 0)))
        ruled = bool(spec.get("rotation")) or n_sdi > 0
        return ReferenceLibrary(groups, np.zeros((n_sci + n_sdi + n_ref, n_sci)),
                                target_partition=spec["partition"], metric=spec["metric"],
                                n_min_ref=spec["n_min_ref"],
                                angles=np.concatenate([np.asarray(self.data.angles, float), np.zeros(n_sdi)])
                                if ruled else None,
                                mid_px=1.0 if n_sdi else None)

    def _reflib(self, bcube, ref, bkeep_partition, filt, inrad, outrad, key_extra=(), sim_cube=None,
                bang=None, sdi=None, mid_px=None):
        """Return this evaluation's reference library and its basis rows.

        The selection, the groups and the similarity matrix that ranks each target's references, is
        built once per binning, frame selection, filter and annulus, from the bins without injections.
        The ranking then depends neither on what was injected nor on which evaluation came first.  The
        basis rows are always this evaluation's own frames, ``[bcube | SDI rows | ref]``.

        Parameters
        ----------
        bcube : ndarray, shape (nb, ny, nx)
            This evaluation's binned science frames.
        ref : ndarray or None
            The reference-star frames, or None.
        bkeep_partition : ndarray
            Partition label of each kept bin.
        filt : int
            Width of the high-pass box, in pixels.
        inrad : float
            Inner radius of the zone, in pixels.
        outrad : float
            Outer radius of the zone, in pixels.
        key_extra : tuple, default ()
            The binning, the frame selection and the bins kept, which join the cache key.
        sim_cube : ndarray or None, default None
            The same bins without injections, for the similarity.  None takes ``bcube``.
        bang : ndarray or None, default None
            Parallactic angle of each bin, in degrees, which a rotation pool selects by.
        sdi : dict or None, default None
            This evaluation's SDI pool (:meth:`_sdi_rows`), ranked here against its own target frames at
            zero lag and never cached.
        mid_px : float or None, default None
            Radius the SDI pool's displacement rule measures at, in pixels.  None takes the zone's
            middle radius.

        Returns
        -------
        lib : ReferenceLibrary
            The selection.
        library : ndarray
            The basis rows.

        Raises
        ------
        ValueError
            If a rotation pool lacks one angle per bin, or a similarity has the wrong shape.

        Notes
        -----
        Until 2026-09-27 the cache held the basis rows too, keyed without the injections.  Every later
        reduction with the same key took its other-roll references from whichever reduction filled the
        cache first, so an injected source was not self-subtracted by the other roll and a result
        depended on the order of the reductions.  It inflated every configuration that used the other
        roll on the built-in engine's searched library (NIRCam DK, D2K, H2K, and MIRI on the built-in
        engine).
        """
        from .reflib import ReferenceGroup, ReferenceLibrary, similarity_matrix
        spec = self._reflib_spec
        n_sci = int(bcube.shape[0])
        n_ref = 0 if ref is None else int(ref.shape[0])
        library = bcube if ref is None else np.concatenate([bcube, ref], axis=0)
        key = (n_sci, int(filt), round(float(inrad), 3), round(float(outrad), 3),
               spec["metric"], spec["shift_px"]) + tuple(key_extra)
        lib = self._reflib_cache.get(key)
        if lib is None:
            rot = bool(spec.get("rotation", False))
            groups = [ReferenceGroup("altroll", np.arange(n_sci), partition=bkeep_partition,
                                     min_keep=spec["min_keep"].get("altroll", 0), rotation=rot)]
            for name, rows in (spec["groups"] or {spec["ref_group"]: np.arange(n_ref)}).items():
                rows = np.asarray(rows, int)
                if rows.size:
                    groups.append(ReferenceGroup(name, n_sci + rows,
                                                 min_keep=spec["min_keep"].get(name, 0)))
            if rot and (bang is None or np.asarray(bang).size != n_sci):
                raise ValueError("a rotation library needs one binned angle per science frame")
            if spec.get("similarity") is not None:
                sim = spec["similarity"]
                if sim.shape != (library.shape[0], n_sci):
                    raise ValueError(f"supplied similarity is {sim.shape}, expected "
                                     f"{(library.shape[0], n_sci)} (library x target)")
            else:
                sc = bcube if sim_cube is None else sim_cube
                if sc.shape != bcube.shape:
                    raise ValueError(f"similarity cube {sc.shape} does not match the binned cube {bcube.shape}")
                sim = similarity_matrix(sc, sc if ref is None else np.concatenate([sc, ref], axis=0),
                                        inrad=inrad, outrad=outrad, filt=0.0,
                                        metric=spec["metric"], shift_px=spec["shift_px"])
            lib = ReferenceLibrary(groups, sim, target_partition=bkeep_partition,
                                   metric=spec["metric"], n_min_ref=spec["n_min_ref"],
                                   angles=np.asarray(bang, float) if rot else None)
            self._reflib_cache[key] = lib
        if sdi is None:
            return lib, library
        # this evaluation's SDI rows between the science frames and the reference star's
        n_sdi = int(sdi["rows"].shape[0])
        cap = int(spec["sdi"]["cap"]) if spec.get("sdi") else n_sdi
        groups = []
        for g in lib.groups:
            if g.name == "altroll":
                groups.append(ReferenceGroup(g.name, g.rows, partition=g.partition, min_keep=g.min_keep,
                                             max_keep=g.max_keep, rotation=g.rotation))
            else:
                groups.append(ReferenceGroup(g.name, g.rows + n_sdi, partition=g.partition, min_keep=g.min_keep,
                                             max_keep=g.max_keep))
        groups.insert(1, ReferenceGroup("sdi", n_sci + np.arange(n_sdi), min_keep=spec["min_keep"].get("sdi", 0),
                                        max_keep=min(cap, n_sdi), scales=sdi["scales"]))
        sim_sdi = similarity_matrix(bcube, sdi["rows"], inrad=inrad, outrad=outrad, filt=0.0,
                                    metric=spec["metric"], shift_px=0)
        sim = np.concatenate([lib.similarity[:n_sci], sim_sdi, lib.similarity[n_sci:]], axis=0)
        ang = np.asarray(bang, float) if lib.angles is None else np.asarray(lib.angles, float)[:n_sci]
        full = ReferenceLibrary(groups, sim, target_partition=bkeep_partition, metric=spec["metric"],
                                n_min_ref=spec["n_min_ref"],
                                angles=np.concatenate([ang, np.asarray(sdi["angles"], float)]),
                                mid_px=max(float(inrad) + abs(float(outrad) - float(inrad)) / 2.0, 1.0)
                                if mid_px is None else max(float(mid_px), 1.0))
        library = np.concatenate([bcube, np.asarray(sdi["rows"], bcube.dtype)] + ([ref] if ref is not None else []),
                                 axis=0)
        return full, library

    def _binned_partition(self, keep, grp, bkeep):
        """Return the partition label of each kept binned frame.

        A bin takes the label of its first frame.  Binning never spans partitions, since a roll's frames
        are contiguous, and a mislabeled bin would let a target draw references from its own roll.

        Parameters
        ----------
        keep : ndarray of bool
            The frame selection over the raw frames.
        grp : ndarray of int
            Bin number of each selected frame.
        bkeep : ndarray of bool
            The bins kept.

        Returns
        -------
        labels : ndarray
            Label of each kept bin.

        Raises
        ------
        ValueError
            If the labels do not match the frames or a bin mixes partitions.
        """
        lab = np.asarray(self._reflib_spec["partition"])
        if lab.size != int(np.asarray(keep).size):
            raise ValueError(f"reference library has {lab.size} partition labels for "
                             f"{np.asarray(keep).size} science frames")
        lab = lab[np.asarray(keep, bool)]
        grp = np.asarray(grp, int)
        ng = int(grp.max()) + 1 if grp.size else 0
        first = np.full(ng, -1, int)
        for i in range(grp.size - 1, -1, -1):          # last write wins -> the first frame
            first[grp[i]] = i
        out = lab[first]
        mixed = [b for b in range(ng) if len(set(lab[grp == b].tolist())) > 1]
        if mixed:
            raise ValueError(f"bins {mixed[:4]} mix partitions; bin within a partition or "
                             f"fix bin so the exclusion stays meaningful")
        return out[np.asarray(bkeep, bool)]

    def _reference_cube(self, filt: int, meta: Dict[str, Any], threads: int = 1) -> Optional[np.ndarray]:
        """Return the RDI reference frames, cropped to the science frames and high-passed.

        The frames are center-cropped to the science field and given the first high-pass, NaN-aware.
        They are neither selected nor binned, so the file must be registered and binned already.

        Parameters
        ----------
        filt : int
            Width of the high-pass box, in pixels, 1 or less for none.
        meta : dict
            The reduction's notes, which get ``rdi_note`` when the cube is missing or does not fit.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        ref : ndarray or None
            The frames in float32, or None, which falls back to ADI.

        Notes
        -----
        It does what ``reduce_near_2`` L1864-1867 does in the IDL predecessor, ``smooth(/nan)``.
        """
        ref = self.data.ref_cube
        ny, nx = self.data.cube.shape[1:]
        if ref is None:
            meta["rdi_note"] = "no reference cube -> ADI"
            return None
        ref = np.asarray(ref, np.float32)
        if ref.ndim != 3 or ref.shape[1] < ny or ref.shape[2] < nx:
            meta["rdi_note"] = "reference cube shape mismatch -> ADI"
            return None
        if ref.shape[1] > ny or ref.shape[2] > nx:
            oy, ox = (ref.shape[1] - ny) // 2, (ref.shape[2] - nx) // 2
            ref = ref[:, oy:oy + ny, ox:ox + nx]
        if filt > 1:
            ref = highpass_stack(ref, filt, nan_aware=True, threads=threads)
        return ref

    # -- main entry ----------------------------------------------------------
    def prepare(self, req: ReductionRequest) -> PreparedFrames:
        """Run the first half of :meth:`reduce`, up to the binned frames.

        It makes the frame selection, the injections into the kept frames, the KLIP-FM model cube, the
        high-pass, the destriping and the angle-aware binning.  The KLIP zone is the annulus padded by
        ``zone_pad`` on each edge the runner marks (``pad_in``, ``pad_out``), then widened by ``zone_in``
        and ``zone_out``.  Another wavelength group's SDI pool needs exactly this, so the SDI mode runs
        it once per group (``ReductionRequest.stage = "prepare"``).

        Parameters
        ----------
        req : ReductionRequest
            The request.  KLIP-FM is skipped with ``k_scan``.

        Returns
        -------
        prep : PreparedFrames
            The prepared frames.

        Raises
        ------
        ValueError
            If every frame is dropped as empty after the high-pass and the binning, as happens to a
            padded or masked cube with NaN.
        """
        p = dict(self.defaults)
        p.update({k: v for k, v in req.params.items() if v is not None})
        k = int(max(round(p["k_klip"]), 1))
        bin_ = int(max(round(p["bin"]), 1))
        filt = int(round(p["filter"]))
        # the KLIP zone is the annulus padded by zone_pad on each edge the runner marks: by
        # default only an edge shared with another annulus (RunConfig.edge_padding).  A
        # request without the marks pads both, as the IDL predecessor does.
        pad_in = self.zone_pad if _flag(p.get("pad_in", True)) else 0.0
        pad_out = self.zone_pad if _flag(p.get("pad_out", True)) else 0.0
        ann_in = max(float(p["inrad"]) - pad_in, 0.0)
        ann_out = min(float(p["outrad"]) + pad_out, self.outrad_cap)
        ann_in = min(ann_in, ann_out - 1.0)
        # zone_in / zone_out widen the zone beyond the annulus (0, 0: the zone is the annulus).
        # The bin's PA span and angsep's arc stay the annulus', where the sources are.
        z_in, z_out = _zone_ext(p)
        inrad = max(ann_in - z_in, 0.0)
        outrad = min(ann_out + z_out, self.outrad_cap)
        inrad = min(inrad, outrad - 1.0)
        ds = self.data
        meta: Dict[str, Any] = {}
        # the job's threads: the per-target KLIP loop, and the per-frame steps around it
        nth = int(max(getattr(req, "threads", 1) or 1, 1))
        do_fm = bool(req.fm_sources)
        if do_fm and req.k_scan:
            meta["fm_note"] = "KLIP-FM skipped: not supported with k_scan"
            do_fm = False
        keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"],
                                    p["coronoise_max"], bin_, k)
        dth = self.dth_max_deg(ann_out)
        # ``hp_order="binned"`` (under test, off by default): bin first, then high-pass the
        # binned frames.  Both are linear, so the images agree to rounding, and the high-pass
        # costs 1/bin as much.  The means are taken in double precision: binned before the
        # high-pass, the frames still carry their background.  The IDL predecessor high-passes
        # every frame before binning, and a destriped cube keeps that order (not linear).
        hp_binned = str(p.get("hp_order", "raw")).lower() == "binned" and filt > 1 and not p["do_destripe"]
        hp_nan = _flag(p.get("hp_nan_aware", False))
        if _flag(p.get("prep_fused", True)):
            # one bin group at a time, the frames read once (_prep_fused)
            bcube, bang, grp, bkeep, mcube = self._prep_fused(
                p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, req.injections or None,
                req.fm_sources if do_fm else None)
        else:
            # the reference: a whole cube per step.  The selection reads the tags, not the
            # pixels, so it comes first.  Only the kept frames are injected, and the injected
            # cubes come back as cube[keep].
            cube = self._inject(req.injections, keep, threads=nth) if req.injections else ds.cube
            mcube = self.model_cube(req.fm_sources, keep, threads=nth) if do_fm else None
            if mcube is not None and filt > 1 and not hp_binned:
                mcube = highpass_stack(mcube, filt, nan_aware=hp_nan, threads=nth)
            bcube, bang, grp, bkeep = self._prep_cube(cube, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth,
                                                      selected=bool(req.injections))
            if mcube is not None and bcube.shape[0]:
                if hp_binned:
                    mb = bin_by_groups(mcube, grp, threads=nth, dtype=np.float64) if bin_ > 1 else mcube
                    mb = highpass_stack(mb, filt, nan_aware=hp_nan, threads=nth)
                else:
                    mb = bin_by_groups(mcube, grp, threads=nth) if bin_ > 1 else np.asarray(mcube, np.float32)
                mcube = mb[bkeep]
        if bcube.shape[0] == 0:
            # the binning drops the bins without a pixel that is neither NaN nor zero
            # (bins_with_data), so a cube that is mostly NaN -- a subarray padded into a
            # larger grid, say -- is wiped out entirely by the high-pass, which at
            # nan_aware=False spreads NaN over the whole frame.  Every downstream shape then goes to zero and the failure
            # surfaces somewhere far away and unrecognizable.
            nan_frac = float(np.mean(~np.isfinite(np.asarray(ds.cube, float))))
            raise ValueError(
                f"every frame was dropped as empty after filtering and binning "
                f"(filter={filt}, bin={bin_}); {100 * nan_frac:.0f}% of the input pixels are "
                f"NaN. A high-pass filter spreads NaN across the whole frame, so a padded or "
                f"masked cube must be cropped to its finite region before it reaches the "
                f"reducer.")
        return PreparedFrames(p, k, bin_, filt, inrad, outrad, dth, keep, bcube, bang, grp, bkeep, mcube,
                              hp_nan, hp_binned, nth, meta, ann=(ann_in, ann_out),
                              bnorth=self.binned_north(keep, grp, bkeep, bin_))

    # ------------------------------------------------------------------ the fused path
    def _prep_fused(self, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, injections=None, fm_sources=None,
                    bkeep=None):
        """Make the frames before KLIP one bin group at a time, as the whole-cube steps make them, to the bit.

        A group's kept frames are copied out of the cube (in float32 when they take sources, as the
        injected cube held them), given each source's stamp in the order of the sources
        (:func:`klip_tpe.injection.place_source`), high-passed a block at a time, destriped when asked
        and averaged by :func:`~klip_tpe.klip.group_mean`.  With ``hp_order="binned"`` the group is
        averaged in double precision first and the mean high-passed.  The cube is read once and nothing
        the size of the cube is written, and the groups go to the job's threads.

        Parameters
        ----------
        p : dict
            The request's parameters over the defaults.
        keep : ndarray of bool
            The frame selection over the raw frames.
        bin_ : int
            Temporal binning factor.
        filt : int
            Width of the high-pass box, in pixels, 1 or less for none.
        dth : float
            Maximum span of parallactic angle in a bin, in degrees.
        hp_nan : bool
            Whether the high-pass leaves NaN pixels out.
        hp_binned : bool
            Whether to bin before the high-pass (``hp_order="binned"``).
        nth : int
            Number of threads.
        injections : sequence of Source or None, default None
            Sources to inject.
        fm_sources : sequence of Source or None, default None
            KLIP-FM test sources, for a model cube made the same way from zero frames.
        bkeep : ndarray of bool or None, default None
            Bins to keep in place of the test for data (:func:`~klip_tpe.klip.bins_with_data`).

        Returns
        -------
        bcube : ndarray, shape (nb, ny, nx)
            The nb binned frames kept.
        bang : ndarray, shape (nb,)
            Mean parallactic angle of each kept bin, in degrees.
        grp : ndarray of int
            Bin number of each selected frame.
        bkeep : ndarray of bool
            True for each bin kept.
        mcube : ndarray or None
            The binned KLIP-FM model cube, or None without ``fm_sources``.

        Raises
        ------
        ValueError
            If ``bkeep`` does not have one value per bin.
        """
        ds = self.data
        kidx = np.flatnonzero(keep)
        a_ = np.asarray(ds.angles, float)[kidx]
        grp = bin_groups(a_, bin_, dth)
        bounds = group_bounds(grp)
        ng = bounds.size - 1
        if bin_ <= 1:
            bang = a_.copy()
        else:
            bang = np.bincount(grp, weights=a_, minlength=ng) / np.bincount(grp, minlength=ng)
        clip = float(p["destripe_clip"]) if (p["do_destripe"] and not hp_binned) else None
        place = self._placer(injections, "inj") if injections else None
        bc = self._fused_bins(ds.cube, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth, place, clip)
        if bkeep is None:
            bkeep = bins_with_data(bc)
        else:
            bkeep = np.asarray(bkeep, bool)
            if bkeep.shape != (bc.shape[0],):
                raise ValueError(f"bkeep has {bkeep.size} bins, the binning {bc.shape[0]}")
        mcube = None
        if fm_sources:
            mb = self._fused_bins(None, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth,
                                  self._placer(fm_sources, "fm"), None)
            mcube = mb[bkeep]
        return bc[bkeep], bang[bkeep], grp, bkeep, mcube

    def _placer(self, sources: Sequence[Source], kind: str):
        """Return a function that adds every source's stamp to a batch of frames.

        The function is ``place(frames, J)``, which adds to ``frames[i]``, frame ``J[i]`` of the cube,
        every source's stamp in the order of the sources, as :func:`inject_frames` adds them.  The
        stamps of a set that comes back are kept (:class:`_CropCache`).

        Parameters
        ----------
        sources : sequence of Source
            The sources.
        kind : {"inj", "fm"}
            Whether the sources are injections or KLIP-FM test sources, which have caches of their own.

        Returns
        -------
        place : callable
            The function, which changes ``frames`` in place.

        Raises
        ------
        ValueError
            If the reducer has no injection model.

        Warns
        -----
        RuntimeWarning
            When a new set of injections is too faint to survive float32 rounding.
        """
        if self.model is None:
            raise ValueError("KLIPReducer needs an injection_model to inject sources" if kind == "inj" else
                             "KLIPReducer needs an injection_model to build the FM model cube")
        key = tuple(s.as_tuple() for s in sources)
        first, cache = self._crops[kind].lookup(key)
        if first and kind == "inj":
            _check_float32_headroom(self.data.cube, sources, self.model, self.fallback, peak=self.cube_peak())
        prepared = prepare_sources(sources, self.model, self.fallback, self.pxscale)
        ny, nx = self.data.cube.shape[1:]
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
        angles = np.asarray(self.data.angles, float)
        model, truenorth, conv = self.model, self.injection_north(), self.angle_convention

        def place(frames: np.ndarray, J: np.ndarray) -> None:
            todo = []
            for i, j in enumerate(J):
                got = None if cache is None else cache.get(int(j))
                if got is None:
                    todo.append(i)
                else:
                    for c in got:
                        add_crop(frames[i], c)
            if not todo:
                return
            Jt = [int(J[i]) for i in todo]
            kept = None if cache is None else [[] for _ in todo]
            for pp in prepared:
                def sink(k, c):
                    add_crop(frames[todo[k]], c)
                    if kept is not None:
                        kept[k].append((c[0], c[1], c[2], c[3], c[4].copy()))
                place_source(Jt, pp, model, angles, truenorth, center, conv, (ny, nx), sink)
            if kept is not None:
                for k, j in enumerate(Jt):
                    cache[j] = kept[k]
        return place

    def _fused_bins(self, base, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth, place, clip) -> np.ndarray:
        """Make every bin group's frame before the test for data, a batch of whole groups at a time.

        Each group goes through ``place``, the high-pass, the destriping and the mean (see
        :meth:`_prep_fused`).  A batch holds whole groups and, past its first group, no more frames than
        one block of :func:`~klip_tpe.klip.highpass_stack` takes.

        Parameters
        ----------
        base : ndarray or None
            The cube, or None for zero frames, as for the KLIP-FM model cube.
        kidx : ndarray of int
            Indices of the kept frames.
        bounds : ndarray of int
            First frame of each group among the kept frames, then their number
            (:func:`~klip_tpe.klip.group_bounds`).
        bin_ : int
            Temporal binning factor.
        filt : int
            Width of the high-pass box, in pixels, 1 or less for none.
        hp_nan : bool
            Whether the high-pass leaves NaN pixels out.
        hp_binned : bool
            Whether to average before the high-pass.
        nth : int
            Number of threads.
        place : callable or None
            Function that adds the sources' stamps (:meth:`_placer`), or None for none.
        clip : float or None
            Clipping level of the destriping, or None not to destripe.

        Returns
        -------
        bins : ndarray, shape (ng, ny, nx)
            The frame of each of the ng groups, in float32.
        """
        ny, nx = self.data.cube.shape[1:]
        npix = int(ny) * int(nx)
        ng = int(bounds.size) - 1
        out = np.empty((ng, ny, nx), np.float32)
        # a batch holds whole groups and, past its first group, no more frames than one block
        # of highpass_stack (its float64 copy within BLOCK_BYTES): bin 1 goes five frames of
        # 150 x 150 at a time, a bin of 29 one group at a time
        cap = max(1, BLOCK_BYTES // (8 * npix))

        def hp(fr):
            if filt <= 1:
                return fr
            if fr.dtype != np.float32:
                return highpass_stack(fr, filt, nan_aware=hp_nan, threads=1)
            for lo, hi in frame_blocks(fr.shape[0], 1, npix):    # in place, a block at a time
                fr[lo:hi] = _highpass_frames(fr[lo:hi], filt, hp_nan)
            return fr

        def block(g0, g1):
            g = g0
            while g < g1:
                h = g + 1
                while h < g1 and bounds[h + 1] - bounds[g] <= cap:
                    h += 1
                s0, s1 = int(bounds[g]), int(bounds[h])
                J = kidx[s0:s1]
                if base is None:
                    fr = np.zeros((s1 - s0, ny, nx), np.float32)
                else:
                    fr = base[J]                     # a copy, in the cube's own precision
                    if place is not None and fr.dtype != np.float32:
                        fr = fr.astype(np.float32)   # as the injected cube held the frames
                if place is not None:
                    place(fr, J)
                if not hp_binned:
                    fr = hp(fr)
                    if clip is not None:
                        # reduce_near_2 L1672-1677: 90 deg (columns) then 0 deg (rows); science only
                        fr = np.stack([destripe(destripe(f, 90.0, clip), 0.0, clip) for f in fr])
                    for q in range(g, h):
                        a, b = int(bounds[q]) - s0, int(bounds[q + 1]) - s0
                        out[q] = fr[a] if bin_ <= 1 else group_mean(fr[a:b], np.float32)
                else:
                    for q in range(g, h):
                        a, b = int(bounds[q]) - s0, int(bounds[q + 1]) - s0
                        m = np.asarray(fr[a:b], np.float32) if bin_ <= 1 else group_mean(fr[a:b], np.float64)[None]
                        out[q] = highpass_stack(m, filt, nan_aware=hp_nan, threads=1)[0]
                g = h
        run_blocks(block, frame_blocks(ng, nth), nth)
        return out

    def _clean_bins(self, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, bkeep=None) -> np.ndarray:
        """Return the binned frames without injections, the reference library's ranking cube.

        They are made by the path :meth:`prepare` takes.

        Parameters
        ----------
        p : dict
            The request's parameters over the defaults.
        keep : ndarray of bool
            The frame selection over the raw frames.
        bin_ : int
            Temporal binning factor.
        filt : int
            Width of the high-pass box, in pixels, 1 or less for none.
        dth : float
            Maximum span of parallactic angle in a bin, in degrees.
        hp_nan : bool
            Whether the high-pass leaves NaN pixels out.
        hp_binned : bool
            Whether to bin before the high-pass (``hp_order="binned"``).
        nth : int
            Number of threads.
        bkeep : ndarray of bool or None, default None
            The bins to take, those the reduction's own cube kept, so the ranking cube matches it bin
            for bin.

        Returns
        -------
        bcube : ndarray, shape (nb, ny, nx)
            The binned frames.
        """
        if _flag(p.get("prep_fused", True)):
            return self._prep_fused(p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, bkeep=bkeep)[0]
        return self._prep_cube(self.data.cube, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, bkeep=bkeep)[0]

    def _prep_cube(self, c, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, selected=False, bkeep=None):
        """Select, high-pass, destripe and bin a science cube a whole cube per step.

        It is the reference the fused path (:meth:`_prep_fused`) is tested against.

        Parameters
        ----------
        c : ndarray, shape (n, ny, nx)
            The science cube.
        p : dict
            The request's parameters over the defaults.
        keep : ndarray of bool
            The frame selection over the raw frames.
        bin_ : int
            Temporal binning factor.
        filt : int
            Width of the high-pass box, in pixels, 1 or less for none.
        dth : float
            Maximum span of parallactic angle in a bin, in degrees.
        hp_nan : bool
            Whether the high-pass leaves NaN pixels out.
        hp_binned : bool
            Whether to bin before the high-pass (``hp_order="binned"``).
        nth : int
            Number of threads.
        selected : bool, default False
            Whether ``c`` holds only the kept frames already.
        bkeep : ndarray of bool or None, default None
            Bins to keep in place of the test for data.

        Returns
        -------
        bcube : ndarray
            The binned frames kept.
        bang : ndarray
            Mean parallactic angle of each kept bin, in degrees.
        grp : ndarray of int
            Bin number of each selected frame.
        bkeep : ndarray of bool
            True for each bin kept.

        Notes
        -----
        The destriping runs at 90 degrees (columns), then 0 degrees (rows), on the science frames only,
        as ``reduce_near_2`` L1672-1677 does in the IDL predecessor.
        """
        a_ = self.data.angles
        if not keep.all():
            a_ = a_[keep]
            if not selected:
                c = c[keep]
        if hp_binned:
            return bin_frames(c, a_, bin_, dth, return_groups=True, threads=nth, dtype=np.float64,
                              then=lambda bc: highpass_stack(bc, filt, nan_aware=hp_nan, threads=nth), bkeep=bkeep)
        if filt > 1:
            c = highpass_stack(c, filt, nan_aware=hp_nan, threads=nth)
        if p["do_destripe"]:
            # reduce_near_2 L1672-1677: 90 deg (columns) then 0 deg (rows); science only
            clip = float(p["destripe_clip"])
            c = np.stack([destripe(destripe(f, 90.0, clip), 0.0, clip) for f in c])
        return bin_frames(c, a_, bin_, dth, return_groups=True, threads=nth, bkeep=bkeep)

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        """Reduce the data under one request.

        With ``req.stage = "prepare"`` it returns the prepared frames for another group's SDI pool
        instead (:meth:`prepare`).

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult or SiblingFrames
            The image and its notes, or the prepared frames.
        """
        if str(getattr(req, "stage", "reduce")) == "prepare":
            pr = self.prepare(req)
            return pr.siblings_payload(getattr(self, "partition_id", None), self.lam_m)
        return self._reduce_prepared(self.prepare(req), req)

    def _sdi_rows(self, siblings, prep: PreparedFrames):
        """Build this group's SDI pool from the other groups' prepared frames.

        Each sibling's frames are magnified about the star by ``lambda_this / lambda_sibling``
        (:mod:`klip_tpe.rescale`), and its model cube the same way under KLIP-FM, so a planet sits at
        ``r * s`` in the FM rows as in the data rows.

        Parameters
        ----------
        siblings : dict, SharedArrays or None
            The other groups' prepared frames, ``{pid: SiblingFrames}`` or their shared-memory
            descriptor.
        prep : PreparedFrames
            This group's prepared frames.

        Returns
        -------
        sdi : dict or None
            ``rows``, ``angles``, ``scales``, ``fm_rows``, ``lenders`` and ``counts``, or None without
            sibling frames.

        Raises
        ------
        ValueError
            If KLIP-FM is on and a sibling has no model cube.
        """
        from .parallel import SharedArrays
        from .rescale import rescale_frames
        if siblings is None:
            return None
        closer = None
        if isinstance(siblings, SharedArrays):
            arrays, closer = siblings.attach()
            siblings = {pid: SiblingFrames(pid, float(ex["lam_m"]), arrays[f"{pid}/bcube"], np.asarray(ex["bang"], float),
                                           arrays.get(f"{pid}/mcube"))
                        for pid, ex in siblings.extra.items()}
        try:
            rows, angs, scales, fms, lenders, counts = [], [], [], [], [], []
            for pid, sib in siblings.items():
                if getattr(sib, "pid", pid) == getattr(self, "partition_id", object()) or sib.bcube.shape[0] == 0:
                    continue
                s = float(self.lam_m) / float(sib.lam_m)
                rows.append(rescale_frames(sib.bcube, s, threads=prep.nth))
                angs.append(np.asarray(sib.bang, float))
                scales.append(np.full(int(sib.bcube.shape[0]), s))
                if prep.mcube is not None:
                    if sib.mcube is None:
                        raise ValueError(f"KLIP-FM with an SDI pool needs the siblings' model cubes ({pid} has none)")
                    fms.append(rescale_frames(sib.mcube, s, threads=prep.nth))
                lenders.append(pid)
                counts.append(int(sib.bcube.shape[0]))
        finally:
            if closer is not None:
                closer()
        if not rows:
            return None
        return {"rows": np.concatenate(rows, axis=0), "angles": np.concatenate(angs),
                "scales": np.concatenate(scales), "fm_rows": np.concatenate(fms, axis=0) if fms else None,
                "lenders": lenders, "counts": counts}

    def _reduce_prepared(self, prep: PreparedFrames, req: ReductionRequest) -> ReductionResult:
        """Run the second half of :meth:`reduce`, from the prepared frames to the image.

        It chooses the basis (ADI, RDI, ARDI, a searched library or its ADI point, and the SDI pool),
        builds the :class:`~klip_tpe.klip.KLIPParams`, subtracts, derotates and combines
        (:meth:`_subtract`), and makes the ``extras`` reference images when asked.

        Parameters
        ----------
        prep : PreparedFrames
            The prepared frames.
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult
            The image, its weight and its notes.
        """
        p, k, bin_, filt = prep.p, prep.k, prep.bin_, prep.filt
        inrad, outrad, dth, keep = prep.inrad, prep.outrad, prep.dth, prep.keep
        bcube, bang, grp, bkeep, mcube = prep.bcube, prep.bang, prep.grp, prep.bkeep, prep.mcube
        meta, nth = prep.meta, prep.nth
        ds = self.data
        spec = self._reflib_spec
        # the frames' own north, per bin, at every derotation (None: the one angle)
        bnorth = getattr(prep, "bnorth", None)
        if bnorth is not None:
            self._check_frame_north()
        dnorth = self.truenorth if bnorth is None else bnorth
        # a zone reaching beyond its annulus (zone_in / zone_out) keeps angsep's arc, and the
        # SDI displacement rule's radius, on the annulus
        ann = None if prep.ann is None else (float(prep.ann[0]), float(prep.ann[1]))
        widened = ann is not None and (ann[0] != float(inrad) or ann[1] != float(outrad))
        # the SDI pool: the other groups' frames, rescaled, when the count asks for any
        sdi = None
        if spec is not None and spec.get("sdi") and int(round(float(p.get("nkeep_sdi", 0) or 0))) > 0:
            if req.siblings is None:
                # a bare KLIPReducer.reduce: nobody prepared the other groups' frames
                meta["sdi_note"] = "nkeep_sdi > 0 but no sibling frames came with the request: no SDI rows"
            else:
                sdi = self._sdi_rows(req.siblings, prep)
        ref_basis = fm_ref = None
        ref_lib = ref_keep = None
        rdi_mode = str(p["rdi_mode"]).lower()
        adi_point = self._library_at_adi_point(p, int(bcube.shape[0]))
        if adi_point is not None:
            # the ADI point of a searched library -- every count the space searches at 0 and the
            # rotating science pool whole -- IS the ADI reduction, and klip_annular would run it
            # as such (library_as_adi) after the reference cube's high-pass and the similarity
            # matrix had been paid for.  Neither is needed to reduce nothing but the sequence.
            meta.update(rdi_mode="searched", ref_keep=dict(adi_point),
                        n_ref_frames=0 if ds.ref_cube is None else int(np.asarray(ds.ref_cube).shape[0]),
                        ref_metric=spec["metric"])
        elif p["use_rdi"] or sdi is not None:
            ref = self._reference_cube(filt, meta, threads=nth) if p["use_rdi"] else None
            if spec is not None:
                # Searched library: the basis cube is [science | SDI rows | references] in that
                # order, because the group row indices the library hands back address it that
                # way, and every target picks its own rows out of it.  ARDI and RDI stop being
                # separate modes here -- an all-zero altroll count IS pure RDI, and the
                # optimizer can elect it rather than being told.
                bpart = self._binned_partition(keep, grp, bkeep)
                # the key: the binning, the frame selection and the bins kept, so that a
                # ranking is reused only for the same bins
                kx = (bin_, round(float(dth), 6), hash(np.asarray(keep, bool).tobytes()),
                      hash(np.asarray(bkeep, bool).tobytes()))
                sim_cube = None
                if req.injections:
                    # the ranking is computed once per key, on the frames WITHOUT injections
                    # at this cube's bins -- rebuilt here only if the key is new
                    k0 = (int(bcube.shape[0]), int(filt), round(float(inrad), 3), round(float(outrad), 3),
                          spec["metric"], spec["shift_px"]) + kx
                    if k0 not in self._reflib_cache:
                        sim_cube = self._clean_bins(p, keep, bin_, filt, dth, prep.hp_nan, prep.hp_binned, nth,
                                                    bkeep=bkeep)
                ref_lib, ref_basis = self._reflib(bcube, ref, bpart, filt, inrad, outrad, key_extra=kx,
                                                  sim_cube=sim_cube, bang=bang, sdi=sdi,
                                                  mid_px=0.5 * (ann[0] + ann[1]) if widened else None)
                ref_keep = ref_lib.keep_from_config(p)
                if mcube is not None:
                    fm_ref = np.concatenate(
                        [mcube] + ([sdi["fm_rows"]] if sdi is not None else [])
                        + ([np.zeros_like(ref)] if ref is not None else []), axis=0)
                meta.update(rdi_mode="searched", n_ref_frames=0 if ref is None else int(ref.shape[0]),
                            ref_keep=dict(ref_keep), ref_metric=ref_lib.metric)
                if sdi is not None:
                    meta.update(n_sdi_rows=int(sdi["rows"].shape[0]), sdi_lenders=[str(x) for x in sdi["lenders"]])
            elif ref is not None:
                if rdi_mode == "rdi":
                    ref_basis = ref
                    fm_ref = np.zeros_like(ref) if mcube is not None else None
                else:                                   # ARDI: reference + science
                    ref_basis = np.concatenate([ref, bcube], axis=0)
                    fm_ref = np.concatenate([np.zeros_like(ref), mcube], axis=0) if mcube is not None else None
                meta.update(rdi_mode=rdi_mode, n_ref_frames=int(ref.shape[0]))

        kp = KLIPParams(k_klip=k, inrad=inrad, outrad=outrad, n_ang=int(max(round(p["n_ang"]), 1)),
                        fast=bool(p["fast"]), angsep=float(p["angsep"]), anglemax=float(p["anglemax"]),
                        n_min_ref=int(p["n_min_ref"]), spat_mean=bool(p["spat_mean"]),
                        temp_mean=bool(p["temp_mean"]), k_scan=bool(req.k_scan),
                        threads=int(max(getattr(req, "threads", 1), 1)),
                        ref_lib=ref_lib, ref_keep=ref_keep, gram=str(p.get("klip_gram", "target") or "target"),
                        arc_radii=ann if widened else None, zone_center=_zone_center(p),
                        zone_angles=str(p.get("zone_angles") or "ccw"))
        if widened:
            meta["annulus"] = [ann[0], ann[1]]
        fm_img = nosub = cadi = None
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # nanmedian of all-NaN corners
            img, fm_img, info = self._subtract(bcube, bang, kp, p, filt, req, mcube, ref_basis, fm_ref, meta,
                                               **({} if bnorth is None else {"north": bnorth}))
            if req.extras:
                if req.injections:
                    # classical ADI (L2072-2086): 2nd high-pass, median PSF in the detector frame, derotate, median
                    c2 = highpass_stack(bcube, filt, nan_aware=True, threads=nth) if filt > 1 else bcube
                    c2 = c2 - np.nanmedian(c2, axis=0)
                    cadi = np.asarray(np.nanmedian(derotate(c2, bang, dnorth, threads=nth), axis=0),
                                      np.float32)
                else:
                    # no subtraction (L2087-2096): derotate + median stack of the selected/filtered/binned cube
                    nosub = np.asarray(np.nanmedian(derotate(bcube, bang, dnorth, threads=nth), axis=0),
                                       np.float32)
        if self.mask_fn is not None:
            img = np.stack([self.mask_fn(im) for im in img]) if req.k_scan else self.mask_fn(img)
        meta.update(info)
        if adi_point is not None:
            meta["library_as_adi"] = True
        meta.update(n_kept=int(keep.sum()), n_binned=int(bcube.shape[0]), inrad=inrad, outrad=outrad,
                    params=p, tag=req.tag)
        return ReductionResult(np.asarray(img, np.float32), self.partition_weight(), meta,
                               fm_image=fm_img, nosub=nosub, cadi=cadi)

    def _library_at_adi_point(self, p: Dict[str, Any], n_binned: int) -> Optional[Dict[str, int]]:
        """Return a searched library's counts when they leave only the rotating sequence itself.

        That is every pool the space searches at 0 and the one-label science pool whole, which is the
        ADI reduction.  A pool the space does not search is at its maximum, so it counts as present.

        Parameters
        ----------
        p : dict
            The request's parameters.
        n_binned : int
            Number of binned frames.

        Returns
        -------
        keep : dict of str to int or None
            The count of each pool, or None when the counts are not the ADI point.
        """
        spec = self._reflib_spec
        if spec is None or not spec.get("rotation") or not self.supports_reference_library:
            return None
        if int(np.unique(np.asarray(spec["partition"])).size) > 1:
            return None
        searched = spec.get("searched")
        stub = self._reflib_stub()
        keep: Dict[str, int] = {}
        for g in stub.groups:
            is_searched = searched is None or g.name in searched
            v = p.get(g.param_name)
            if g.name == "altroll":
                n = int(round(float(v))) if (is_searched and v is not None) else max(n_binned - 1, 0)
                if n < max(n_binned - 1, 0):
                    return None
                keep[g.name] = n                      # the count as the record has it (the config's own when searched)
                continue
            if not is_searched or v is None or int(round(float(v))) != 0:
                return None
            keep[g.name] = 0
        return keep

    # ------------------------------------------------------------------ backend hook
    def _subtract(self, bcube: np.ndarray, bang: np.ndarray, kp: KLIPParams, p: Dict[str, Any], filt: int,
                  req: ReductionRequest, mcube: Optional[np.ndarray], ref_basis: Optional[np.ndarray],
                  fm_ref: Optional[np.ndarray], meta: Dict[str, Any], north: Optional[np.ndarray] = None):
        """Subtract the PSF, derotate and combine the prepared cube.

        Backends (pyKLIP, VIP, a user pipeline) override this and inherit the rest of the chain.  The
        reference implementation is the annular KLIP of :mod:`klip_tpe.klip`, with the residuals
        high-passed again unless ``post_hp`` is False.

        Parameters
        ----------
        bcube : ndarray, shape (nb, ny, nx)
            The prepared frames, injected, selected, high-passed and binned.
        bang : ndarray, shape (nb,)
            Parallactic angle of each frame, in degrees.
        kp : KLIPParams
            The KLIP settings.
        p : dict
            The request's parameters.
        filt : int
            Width of the high-pass box, in pixels.
        req : ReductionRequest
            The request.
        mcube : ndarray or None
            The KLIP-FM model cube, or None.
        ref_basis : ndarray or None
            Reference frames for the basis, or None for ADI.
        fm_ref : ndarray or None
            The planet model in ``ref_basis``.
        meta : dict
            The reduction's notes.
        north : ndarray or None, default None
            Each bin's true north (:meth:`binned_north`), or None to derotate every bin by
            ``bang + truenorth``.  A backend that overrides this method without ``_honours_frame_north``
            is never handed frames with their own north.

        Returns
        -------
        image : ndarray, shape (ny, nx) or (k_klip, ny, nx)
            The combined image, or one per number of KL modes with ``kp.k_scan``.
        fm_image : ndarray or None
            The KLIP-FM image, combined with the science frames' weights, or None.
        info : dict
            How the subtraction ran (:func:`~klip_tpe.klip.klip_annular`).

        Notes
        -----
        The noise-weighted combination turns the frames back by the angles it is given to measure each
        pixel's variance, so with each bin's own north it gets ``bang + (north - truenorth)``, which
        lines the frames up again to one angle.
        """
        out = klip_annular(bcube, bang, kp, self.lam_over_d_px, fm_cube=mcube, ref_cube=ref_basis,
                           fm_ref_cube=fm_ref, fm_selfsub=bool(p["fm_selfsub"]))
        res, info = out[0], out[1]
        fm_res = out[2] if len(out) > 2 else None
        ct = p["comb_type"]
        nth = int(max(getattr(req, "threads", 1) or 1, 1))
        # the residuals' second high-pass (reduce_near_2's); post_hp = False leaves them as KLIP
        # made them, as the ALES reduction's ADI images of MWC 758 are
        post_hp = filt > 1 and _flag(p.get("post_hp", True))

        dnorth = self.truenorth if north is None else np.asarray(north, float)
        wang = bang if north is None else np.asarray(bang, float) + (dnorth - self.truenorth)

        def _post(rc: np.ndarray) -> np.ndarray:
            if post_hp:
                rc = highpass_stack(rc, filt, nan_aware=True, threads=nth)
            return derotate(rc, bang, dnorth, threads=nth)

        def _combine(dr: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
            if ct == "median":
                return np.nanmedian(dr, axis=0)
            if ct == "mean":
                return np.nanmean(dr, axis=0)
            return nw_ang_comb(dr, wang, ref_cube=ref, threads=nth)

        fm_img = None
        if req.k_scan:
            img = np.stack([_combine(_post(res[kk])) for kk in range(res.shape[0])])
        else:
            dr = _post(res)
            img = _combine(dr)
            if fm_res is not None:
                # FM frames post-processed like the data, combined with the SCIENCE weights (nw_ang_comb_ref)
                fm_img = np.asarray(_combine(_post(fm_res), ref=dr), np.float32)
        return img, fm_img, info

    def describe(self) -> Dict[str, Any]:
        """Return the reducer's settings, for a run's record.

        Returns
        -------
        info : dict
            The base keys, the optics, the defaults, the injection model, the frames' provenance, any
            edits of the data, the true north and the reference library.
        """
        d = super().describe()
        d.update(lam_m=self.lam_m, diam_m=self.diam_m, truenorth=self.truenorth, zone_pad=self.zone_pad,
                 outrad_cap=self.outrad_cap, defaults=dict(self.defaults), nframes=self.data.nframes,
                 has_ref_cube=self.data.ref_cube is not None, mask_fn=self.mask_fn is not None,
                 injection_model=None if self.model is None else self.model.describe(),
                 # which frames these are, when the loader said (load_calints does): the run's
                 # record of its input, beside the record of what it did with it
                 frames=(getattr(self.data, "meta", None) or {}).get("provenance"))
        if getattr(self, "data_edits", None):
            d["data_edits"] = dict(self.data_edits)
        if getattr(self.data, "north", None) is not None:
            d["north"] = self.north_record()
        spec = getattr(self, "_reflib_spec", None)
        if spec is not None:
            n_ref = 0 if self.data.ref_cube is None else int(np.asarray(self.data.ref_cube).shape[0])
            searched = spec.get("searched")
            d["reference_library"] = {
                "metric": spec["metric"], "shift_px": spec["shift_px"], "n_min_ref": spec["n_min_ref"],
                "ref_group": spec["ref_group"], "n_ref_frames": n_ref, "rotation": bool(spec.get("rotation", False)),
                "n_labels": int(np.unique(np.asarray(spec["partition"])).size),
                "pools": [g.param_name + (f" {g.min_keep}-{g.max_keep} searched"
                                          if searched is None or g.name in searched
                                          else " all eligible frames (not searched)")
                          for g in self._reflib_stub().groups]}
        return d


# ----------------------------------------------------------------------------
# multi-partition wrapper
# ----------------------------------------------------------------------------
@dataclass
class EvalImages:
    """Everything one evaluation produced.

    Attributes
    ----------
    image : ndarray, shape (ny, nx)
        The combined image, injected or clean.
    stack : ndarray or None, default None
        The per-partition images, shape (npart, ny, nx).
    weights : ndarray or None, default None
        Weight of each partition in the combination.
    partitions : list or None, default None
        Identifiers of the partitions, in the order of ``stack``.
    meta : dict
        Notes on the evaluation.
    fm_image : ndarray or None, default None
        The KLIP-FM image, combined with the same partition weights.
    fm_stack : ndarray or None, default None
        The per-partition KLIP-FM images, shape (npart, ny, nx).
    nosub : ndarray or None, default None
        The combined reference image without subtraction, of the clean pass with ``extras``.
    cadi : ndarray or None, default None
        The combined classical-ADI reference image, of the injected pass with ``extras``.
    """

    image: np.ndarray                              # combined injected (or clean) image
    stack: Optional[np.ndarray] = None             # (npart, ny, nx) per-partition images
    weights: Optional[np.ndarray] = None
    partitions: Optional[List[Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    fm_image: Optional[np.ndarray] = None          # KLIP-FM response, combined with the SAME partition weights
    fm_stack: Optional[np.ndarray] = None          # (npart, ny, nx) per-partition FM images
    nosub: Optional[np.ndarray] = None             # extras: combined no-subtraction reference (clean pass)
    cadi: Optional[np.ndarray] = None              # extras: combined classical-ADI reference (injected pass)


def combine_stack(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Take the NaN-aware weighted mean over the first axis.

    Parameters
    ----------
    stack : ndarray, shape (m, ...)
        The m images.
    weights : ndarray, shape (m,)
        Weight of each image.

    Returns
    -------
    mean : ndarray
        The weighted mean of the finite values, NaN where none is finite.

    Notes
    -----
    It does what ``near2m_avg`` and ``near2m_pncombine`` do in the IDL predecessor.
    """
    stack = np.asarray(stack, float)
    w = np.asarray(weights, float).reshape((-1,) + (1,) * (stack.ndim - 1))
    fin = np.isfinite(stack)
    num = np.nansum(np.where(fin, stack, 0.0) * w, axis=0)
    den = np.sum(fin * w, axis=0)
    out = num / np.where(den > 0, den, np.nan)
    return out


def _run_one(args):
    """Reduce one job, for a worker pool.

    Parameters
    ----------
    args : tuple
        The ``(reducer, request)`` pair.

    Returns
    -------
    result : ReductionResult
        What ``reducer.reduce(request)`` returns.
    """
    reducer, req = args
    return reducer.reduce(req)


def _gpu_card_factory() -> Optional[Callable[[Any], Any]]:
    """Return the factory of GPU reducers for the KLIP partitions, when :mod:`klip_tpe.gpu_reducer` is available.

    CuPy loads only in the card workers.

    Returns
    -------
    factory : callable or None
        ``GPUKLIPReducer.from_cpu`` for a :class:`KLIPReducer` and None for other reducers, or None
        when the module is missing.
    """
    try:
        from .gpu_reducer import GPUKLIPReducer         # type: ignore[import-not-found]
    except Exception:                                  # noqa: BLE001
        return None
    return lambda r: GPUKLIPReducer.from_cpu(r) if isinstance(r, KLIPReducer) else None


class PartitionedReducer(Reducer):
    """Reducer of a dataset made of partitions, each reduced with its own parameter block and combined.

    Parameters
    ----------
    reducers : dict
        The reducer of each partition, ``{partition_id: Reducer}``.
    weighting : {"equal", "sqrt_texp"}, default "equal"
        Weights of the combination.  "equal" takes the uniform mean, the production per-night
        combination, and "sqrt_texp" each reducer's :meth:`Reducer.partition_weight`.
    max_workers : int or str, default 1
        The thread budget (:func:`klip_tpe.parallel.resolve_workers`).  More than 1 sends partition
        jobs to forked worker processes, or to a thread pool when forking is unavailable.
    pool : {"auto", "processes", "threads"}, default "auto"
        Kind of worker pool.
    log : callable or None, default None
        Function called with each log line.  None stays silent.
    threads_per_job : int, default 2
        Average threads of a partition job.  The process pool gets ``max_workers // threads_per_job``
        processes, never fewer than ``min(max_workers, 2 x partitions)``, and a job expected to take
        long gets up to four times as many threads (:meth:`_plan`).
    representative : object or None, default None
        The partition whose plate scale, FWHM, lambda/D and matched filter stand for the combined
        image.  None takes the first, which does not suit the wavelength groups of an IFS cube, where
        an adapter names the group at the mean wavelength.

    Attributes
    ----------
    injection_scale : dict or None
        Factor on each partition's injected contrasts (:meth:`set_injection_spectrum`), or None for
        the contrast given in every partition.
    card_share : float or None
        Percent of the search's injected draws the card workers take
        (:class:`klip_tpe.parallel.CardSpec`), set before the pool starts.  None or 0 uses no card.
    card_workers : int
        Number of card worker processes, sharing the card's queue.
    card_factory : callable or None
        Function ``cpu_reducer -> card reducer or None``.  None takes ``GPUKLIPReducer.from_cpu`` for
        the KLIP partitions when :mod:`klip_tpe.gpu_reducer` is present.
    sdi_lenders : {"all", "selected"}
        Which partitions lend their frames to the others' SDI pools.
    partition_label : str
        What the partitions are, such as "night", "channel" or "roll", for the display's panels.
    provenance : dict or None
        Record of what the partitions are, written to the run's setup file by :meth:`describe`.
    cap_threads : bool
        Whether to hold the threads of the jobs running together to ``max_workers``, off by default.

    Raises
    ------
    ValueError
        If ``reducers`` is empty.
    KeyError
        If ``representative`` is not a partition.

    Notes
    -----
    Inverse-variance weights are pathological here, because an over-subtracted night has low scatter
    and low throughput.  On patrick, 48 jobs of 2 threads at a budget of 96 reduced NEAR2's nights
    1.25x faster than 48 jobs of 4 at 192.  Capping the threads (``cap_threads``) took a six-night
    reduction from 0.205 s to 0.339 s there, since the few jobs sent first took 8 threads each and
    nearly every later one ran on 1.
    """

    name = "partitioned"
    # class defaults, so an instance made without __init__ still works; __init__ sets its own
    threads_per_job = 2
    #: hold the threads of the jobs running together to ``max_workers``
    #: (:meth:`klip_tpe.parallel.ProcessPool._fit`).  Off: on patrick (96 / 2 / batch 4,
    #: NEAR2's annulus 5) it took a six-night reduction from 0.205 s to 0.339 s, 1.65x
    #: slower.  With 48 slots and a budget of 96, the few jobs sent first took 8 threads each
    #: and nearly every later one, the heavy nights included, ran on 1, so each trial waited on
    #: one-thread stragglers and the pool sat 25 % busy.  The plan's own excess (a load average
    #: of 133 on 96 cores) is absorbed by the 192 logical CPUs and is the fastest layout
    #: measured.  True turns the cut on, for measuring a bound.
    cap_threads = False
    _threads_fallback = False
    _start_lock = threading.Lock()
    _recent_costs = None
    #: ``{pid: factor}``: partition ``pid`` gets every injected source at its contrast times
    #: ``factor`` (:meth:`set_injection_spectrum`); None, the default, injects the contrast
    #: given in every partition, a source with the star's spectrum
    injection_scale: Optional[Dict[Any, float]] = None
    injection_spectrum_info: Optional[Dict[str, Any]] = None
    #: the percentage of the search's injected draws the card workers reduce
    #: (:class:`klip_tpe.parallel.CardSpec`), set before the pool starts; None or 0: no card
    card_share: Optional[float] = None
    #: card worker processes, sharing the card's queue
    card_workers: int = 3
    #: ``factory(cpu_reducer) -> card reducer or None``; None: ``GPUKLIPReducer.from_cpu`` for
    #: the KLIP partitions, when :mod:`klip_tpe.gpu_reducer` is there
    card_factory: Optional[Callable[[Any], Any]] = None

    def __init__(self, reducers: Dict[Any, Reducer], weighting: str = "equal", max_workers=1,
                 pool: str = "auto", log=None, threads_per_job: int = 2, representative: Any = None):
        if not reducers:
            raise ValueError("need at least one partition")
        if representative is not None and representative not in reducers:
            raise KeyError(f"representative partition {representative!r} is not one of {list(reducers)}")
        from .parallel import resolve_workers
        self.reducers = dict(reducers)
        self.weighting = weighting
        self.max_workers = resolve_workers(max_workers)
        self.threads_per_job = max(int(threads_per_job or 2), 1)
        self.pool_kind = pool            # auto | processes | threads
        self._log = log or (lambda s: None)
        self._procs = None
        self._threads_fallback = False   # worker processes could not be started
        self._start_lock = threading.Lock()    # the runner keeps several reductions in flight
        self._recent_costs = None        # costs of the latest partition jobs (see _plan)
        if self.max_workers > 1:
            from .parallel import pin_blas
            pin_blas()                      # one BLAS thread per worker thread (no oversubscription)
        self.representative = representative if representative is not None else next(iter(self.reducers))
        for pid, r in self.reducers.items():
            if getattr(r, "partition_id", None) is None:
                r.partition_id = pid
        #: which partitions lend their frames to the others' SDI pools (``"all"``, or
        #: ``"selected"``: only the partitions the configuration includes); an IFS adapter sets it
        self.sdi_lenders: str = "all"
        first = self.reducers[self.representative]
        self.pxscale, self.fwhm = first.pxscale, first.fwhm
        self.lam_over_d_px = first.lam_over_d_px
        self.angle_convention = first.angle_convention
        #: a record of what the partitions are (an IFS adapter's wavelength groups, say),
        #: written to the run's setup file by :meth:`describe`
        self.provenance: Optional[Dict[str, Any]] = None
        self.supports_kscan = all(r.supports_kscan for r in self.reducers.values())
        self.supports_fm = all(getattr(r, "supports_fm", False) for r in self.reducers.values())
        #: what the partitions ARE -- only the wording of the display's partition panels
        #: ("night" for a multi-epoch campaign, "channel", "roll", "group", ...)
        self.partition_label = "night"
        self._pool: Optional[cf.ThreadPoolExecutor] = None

    def partitions(self) -> List[Any]:
        """Return the identifiers of the partitions.

        Returns
        -------
        pids : list
            The identifiers, in order.
        """
        return list(self.reducers)

    def frame_angles(self, pid=None):
        """Return one partition's frame angles.

        Parameters
        ----------
        pid : object, default None
            Identifier of the partition.

        Returns
        -------
        angles : ndarray or None
            Parallactic angle of each frame, in degrees.
        """
        return self.reducers[pid].frame_angles()

    def frame_tags(self, pid=None):
        """Return one partition's frame quality tags.

        Parameters
        ----------
        pid : object, default None
            Identifier of the partition.

        Returns
        -------
        tags : dict of str to ndarray or None
            The tags.
        """
        return self.reducers[pid].frame_tags()

    def partition_weight(self, pid) -> float:
        """Return a partition's weight in the combination.

        Parameters
        ----------
        pid : object
            Identifier of the partition.

        Returns
        -------
        weight : float
            1 for equal weights, or the partition's own weight.
        """
        return 1.0 if self.weighting == "equal" else self.reducers[pid].partition_weight()

    def matched_filter_kernel(self, rho_as: float):
        """Return the representative partition's matched filter at a separation.

        Parameters
        ----------
        rho_as : float
            Separation, in arcsec.

        Returns
        -------
        kernel : ndarray or None
            The kernel, or None for the metric's Gaussian.
        """
        return self.reducers[getattr(self, "representative", next(iter(self.reducers)))].matched_filter_kernel(rho_as)

    # -- the injected sources' spectrum ----------------------------------------------
    def set_injection_spectrum(self, spectrum: Any, norm: Any = None, label: Optional[str] = None
                               ) -> Optional[Dict[Any, float]]:
        """Give every injected source a spectrum across the partitions.

        A source of contrast ``c`` goes into partition ``pid`` at ``c * factor[pid]`` in every reduction
        this reducer makes for the run, the search's, calibration's and validation's injections, the
        KLIP-FM sources and the frames an SDI pool borrows.  ``c`` is then the contrast where the factor
        is 1, at ``norm``.  A source of negative contrast, a known companion taken out of the frames
        (``RunConfig.subtract_known``), keeps the contrast given.  Set it before the run starts, since
        the run's calibrated contrast is in its units.

        Parameters
        ----------
        spectrum : dict, callable, tuple or None
            None for a flat spectrum with the star's colors, ``{pid: factor}``, a function
            ``f(lambda_um) -> relative contrast`` read at each partition's ``lam_m``, or
            ``(lambda_um, values)``, interpolated linearly and held at its end values outside.  The
            factors must be finite and at least 0, and 0 puts no source in that partition.
        norm : object, float or None, default None
            Where the factor is 1, a partition identifier or a wavelength in meters.  None takes the
            representative partition, whose template is the matched filter.
        label : str or None, default None
            What the spectrum is, for :meth:`describe` and the run's setup file.

        Returns
        -------
        factors : dict or None
            The factor of each partition, or None for a flat spectrum.

        Raises
        ------
        KeyError
            If a per-partition spectrum misses a partition.
        ValueError
            If a table is malformed, a partition has no wavelength, ``norm`` cannot be read there, the
            spectrum is not positive at ``norm``, or a factor is negative or not finite.
        """
        if spectrum is None:
            self.injection_scale = None
            self.injection_spectrum_info = None
            return None
        lam_um = {pid: (float(getattr(r, "lam_m", np.nan)) * 1e6) for pid, r in self.reducers.items()}
        if isinstance(spectrum, dict):
            raw = {pid: float(spectrum[pid]) for pid in self.reducers if pid in spectrum}
            missing = [pid for pid in self.reducers if pid not in raw]
            if missing:
                raise KeyError(f"the spectrum has no factor for partitions {missing[:4]}")
            fn = None
            kind = "per partition"
        else:
            if callable(spectrum):
                fn = spectrum
                kind = "function of wavelength"
            else:
                wl, val = (np.asarray(v, float).ravel() for v in spectrum)
                if wl.size != val.size or wl.size < 1:
                    raise ValueError(f"spectrum table: {wl.size} wavelengths and {val.size} values")
                o = np.argsort(wl)
                wl, val = wl[o], val[o]
                fn = lambda x, wl=wl, val=val: float(np.interp(float(x), wl, val))
                kind = f"table of {wl.size} wavelengths, {wl[0]:.3f}-{wl[-1]:.3f} um, linear"
            bad = [pid for pid, v in lam_um.items() if not np.isfinite(v)]
            if bad:
                raise ValueError(f"partitions {bad[:4]} have no wavelength (lam_m) to read the spectrum at")
            raw = {pid: float(fn(lam_um[pid])) for pid in self.reducers}
        rep = getattr(self, "representative", next(iter(self.reducers)))
        if norm is None or norm in self.reducers:
            at = rep if norm is None else norm
            ref = raw[at]
            at_txt = f"partition {at} ({lam_um[at]:.3f} um)" if np.isfinite(lam_um[at]) else f"partition {at}"
        else:
            if fn is None:
                raise ValueError(f"norm={norm!r} is not a partition, and a per-partition spectrum has no "
                                 f"value at a wavelength")
            ref = float(fn(float(norm) * 1e6))
            at_txt = f"{float(norm) * 1e6:.3f} um"
        if not (np.isfinite(ref) and ref > 0):
            raise ValueError(f"the spectrum is {ref!r} at its normalisation ({at_txt}); it must be > 0 there")
        fac = {pid: v / ref for pid, v in raw.items()}
        bad = {pid: v for pid, v in fac.items() if not (np.isfinite(v) and v >= 0)}
        if bad:
            raise ValueError(f"spectrum factors must be finite and >= 0: {dict(list(bad.items())[:4])}")
        self.injection_scale = fac
        self.injection_spectrum_info = {
            "label": label or kind, "kind": kind, "normalised_at": at_txt,
            "factors": {str(pid): round(float(v), 6) for pid, v in fac.items()},
            "lam_um": {str(pid): round(float(v), 5) for pid, v in lam_um.items() if np.isfinite(v)}}
        vals = np.array(list(fac.values()))
        getattr(self, "_log", lambda s: None)(
            f"  injected spectrum ({label or kind}): contrast x {vals.min():.3g}-{vals.max():.3g} across "
            f"{len(fac)} partitions, 1 at {at_txt}")
        return dict(fac)

    def scaled_sources(self, sources: Optional[Sequence[Source]], pid: Any) -> Optional[Sequence[Source]]:
        """Return sources as a partition injects them, each positive contrast times its spectrum factor.

        Parameters
        ----------
        sources : sequence of Source or None
            The sources.
        pid : object
            Identifier of the partition.

        Returns
        -------
        scaled : sequence of Source or None
            The sources, unchanged without a spectrum.
        """
        scale = getattr(self, "injection_scale", None)
        if not sources or not scale:
            return sources
        f = float(scale.get(pid, 1.0))
        if f == 1.0:
            return sources
        return [Source(s.rho, s.theta, float(s.contrast) * f) if float(s.contrast) > 0 else s for s in sources]

    def subtract_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Take point sources out of every partition's frames (:meth:`KLIPReducer.subtract_sources`).

        It must run before the worker processes start, since they keep the reducers they were started
        with.

        Parameters
        ----------
        sources : sequence
            The sources, each a :class:`~klip_tpe.metrics.Source` or ``(rho_arcsec, pa_deg, contrast)``.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        n : int
            Number of sources subtracted per partition.

        Raises
        ------
        RuntimeError
            If the worker pool has started.
        TypeError
            If a partition's reducer cannot subtract sources.
        """
        return self._edit_partitions(sources, threads, "subtract_sources", "subtracted_kept", "subtracted_sources")

    def inject_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Add point sources to every partition's frames (:meth:`KLIPReducer.inject_sources`).

        It must run before the worker processes start, since they keep the reducers they were started
        with.

        Parameters
        ----------
        sources : sequence
            The sources, each a :class:`~klip_tpe.metrics.Source` or ``(rho_arcsec, pa_deg, contrast)``.
        threads : int, default 1
            Number of threads.

        Returns
        -------
        n : int
            Number of sources added per partition.

        Raises
        ------
        RuntimeError
            If the worker pool has started.
        TypeError
            If a partition's reducer cannot inject sources.
        """
        return self._edit_partitions(sources, threads, "inject_sources", "injected_kept", "injected_sources")

    def _edit_partitions(self, sources: Sequence[Any], threads: int, method: str, attr: str, key: str) -> int:
        """Apply a reducer's data edit to every partition and record it.

        Parameters
        ----------
        sources : sequence
            The sources.
        threads : int
            Number of threads.
        method : {"subtract_sources", "inject_sources"}
            The edit.
        attr : {"subtracted_kept", "injected_kept"}
            Attribute that collects each partition's measured flux fraction.
        key : {"subtracted_sources", "injected_sources"}
            Key of the edit in ``data_edits``.

        Returns
        -------
        n : int
            Number of sources applied per partition.

        Raises
        ------
        RuntimeError
            If the worker pool has started.
        TypeError
            If a partition's reducer lacks the method.
        """
        if self._procs is not None:
            raise RuntimeError(f"{method} has to run before the worker pool starts: the worker "
                               "processes keep the frames they were started with")
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n = 0
        kept: Dict[Any, Optional[float]] = {}
        for pid, r in self.reducers.items():
            f = getattr(r, method, None)
            if f is None:
                raise TypeError(f"partition {pid}: a {type(r).__name__} cannot {method.replace('_', ' ')}")
            n = f(src, threads=threads)
            kept[pid] = getattr(r, attr, None)
        setattr(self, attr, kept)
        rec = dict(getattr(self, "data_edits", None) or {})
        rec[key] = list(rec.get(key) or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    _pool_logged = False

    def start_workers(self) -> str:
        """Start the worker pool once the data are loaded, and log which kind the run got.

        The pool kind goes to the run's log here, so a script that builds its own runner records it as
        the command line does.

        Returns
        -------
        kind : {"processes", "threads", "serial"}
            The pool in use.  "processes" runs :meth:`n_procs` forked workers, and "threads" is the
            fallback or the request.
        """
        kind = self._pool_kind_now()
        if not self._pool_logged:
            self._pool_logged = True
            from .parallel import describe
            procs = (getattr(self._procs, "n", None) or self.n_procs()) if kind == "processes" else None
            per = self.job_threads()
            costed = bool(procs) and all(getattr(type(r), "cost_estimate", Reducer.cost_estimate)
                                         is not Reducer.cost_estimate for r in self.reducers.values())
            cap = min(4 * per, int(self.max_workers)) if costed else None
            held = getattr(self._procs, "budget", None) if kind == "processes" else None
            self._log(f"  parallel: {describe(self.max_workers, len(self.reducers), procs, per, cap, held)}"
                      f"; pool = {kind}")
            cs = self.card_summary()
            if cs:
                self._log(f"  card: {cs['workers']} worker{'s' * (cs['workers'] != 1)} starting, for "
                          f"{cs['share']:g} % of the search's injected draws")
            if kind == "threads" and self.pool_kind in ("auto", "processes"):
                self._log("    threads share the GIL, so expect several times less throughput than "
                          "worker processes: worth fixing rather than living with.")
        return kind

    def _pool_kind_now(self) -> str:
        """Return the pool kind in use, starting the worker processes on the first call.

        Returns
        -------
        kind : {"processes", "threads", "serial"}
            The pool in use.
        """
        if self.max_workers <= 1:
            return "serial"
        with self._start_lock:
            return self._pool_kind_locked()

    def _pool_kind_locked(self) -> str:
        """Return the pool kind, starting the worker processes when asked for, with the start lock held.

        Returns
        -------
        kind : {"processes", "threads"}
            "processes", or "threads" when they cannot start or are not asked for.
        """
        if self._procs is not None:
            return "processes"
        if self.pool_kind in ("auto", "processes") and not self._threads_fallback:
            try:
                from .parallel import ProcessPool
                self._procs = ProcessPool(self.reducers, self.n_procs(), log=self._log,
                                          budget=int(self.max_workers) if self.cap_threads else None,
                                          card=self._card_spec())
                return "processes"
            except Exception as exc:
                self._log(f"  parallel: worker processes unavailable ({exc!r}); using threads")
                self._threads_fallback = True
        return "threads"

    def _card_spec(self):
        """Build the card workers' specification for the pool.

        The card reducers are built in the parent before the fork, and load nothing of CUDA until their
        worker warms them up.

        Returns
        -------
        spec : CardSpec or None
            The specification, or None without a share, a card reducer or a partition the card can
            reduce.
        """
        share = float(self.card_share or 0)
        if share <= 0:
            return None
        factory = self.card_factory or _gpu_card_factory()
        if factory is None:
            self._log(f"  card: a share of {share:g} % was asked, but klip_tpe.gpu_reducer is not there: CPU only")
            return None
        reds = {}
        for pid, r in self.reducers.items():
            try:
                c = factory(r)
            except Exception as exc:              # noqa: BLE001
                self._log(f"  card: partition {pid} stays on the CPU ({exc!r})")
                c = None
            if c is not None:
                reds[pid] = c
        if not reds:
            self._log("  card: no partition the card can reduce: CPU only")
            return None
        from .parallel import CardSpec
        return CardSpec(reds, share, self.card_workers)

    def take_card_notes(self) -> List[str]:
        """Return the card's log lines since the last call, for the caller to write from its own thread.

        Returns
        -------
        notes : list of str
            Lines such as the card coming up, a job failing on it or a partition left out.
        """
        take = getattr(self._procs, "take_notes", None)
        return take() if callable(take) else []

    def card_summary(self) -> Optional[Dict[str, Any]]:
        """Return the card workers' summary (:meth:`klip_tpe.parallel.ProcessPool.card_summary`).

        Returns
        -------
        summary : dict or None
            The summary, or None without card workers.
        """
        f = getattr(self._procs, "card_summary", None)
        return f() if callable(f) else None

    def n_procs(self) -> int:
        """Return the number of worker processes for the budget.

        Returns
        -------
        n : int
            One per ``threads_per_job`` threads, never fewer than ``min(max_workers, 2 x partitions)``.
        """
        b = int(self.max_workers)
        return max(b // self.threads_per_job, min(b, 2 * len(self.reducers)), 1)

    def job_threads(self, n_sel: Optional[int] = None) -> int:
        """Return the per-target threads of one partition job.

        With worker processes the budget is shared out by process, so every job gets
        ``max_workers // processes``.  The thread-pool fallback shares the budget between the injected
        and the clean reduction of ``n_sel`` partitions.

        Parameters
        ----------
        n_sel : int or None, default None
            Number of partitions reduced, for the thread-pool fallback.  None takes all.

        Returns
        -------
        threads : int
            Threads per job, at least 1.
        """
        if self._procs is not None or (self.max_workers > 1 and self.pool_kind != "threads"
                                       and not self._threads_fallback):
            return max(int(self.max_workers) // self.n_procs(), 1)
        n = len(self.reducers) if n_sel is None else int(n_sel)
        conc = 2 if self.max_workers >= 2 else 1
        return max(int(self.max_workers) // max(conc * n, 1), 1)

    def job_slots(self) -> int:
        """Return how many partition jobs can run at once.

        Returns
        -------
        n : int
            The worker processes, the thread pool's size, or 1 when serial.
        """
        if self.max_workers <= 1:
            return 1
        if self._procs is not None:
            return int(getattr(self._procs, "n", None) or self.n_procs())
        if self.pool_kind != "threads" and not self._threads_fallback:
            return self.n_procs()
        return int(self.max_workers)

    def memory_note(self) -> Optional[str]:
        """Return a log line on the worker processes' memory, for sizing the pool against the machine.

        Returns
        -------
        note : str or None
            Their peak resident sets and what they hold together now, or None without worker processes
            or off Linux.
        """
        mem = getattr(self._procs, "memory", None)
        m = mem() if callable(mem) else None
        if not m:
            return None
        from .parallel import mem_available
        gb = 1024.0 ** 3
        pk = sorted(m["peak"])
        avail = mem_available()
        return (f"worker memory: {len(pk)} processes peaked at {pk[len(pk) // 2] / gb:.2f} GB each (largest "
                f"{pk[-1] / gb:.2f} GB, each counting the data it shares), {m['pss'] / gb:.1f} GB together now "
                f"(shared data counted once)" + (f", {avail / gb:.0f} GB available" if avail else ""))

    def _map(self, jobs, keys=None):
        """Run partition jobs on the worker pool, or in turn when serial.

        A lone job goes to the worker processes too, since the runner keeps several reductions in flight.
        The thread pool's wait keeps the GUI event loop serviced.

        Parameters
        ----------
        jobs : list of tuple
            The ``(reducer, request)`` pairs.
        keys : list of float or None, default None
            Priority of each job in the process pool, the lowest first.

        Returns
        -------
        results : list
            Each job's result, in order.
        """
        if self.max_workers > 1 and (len(jobs) > 1 or self._procs is not None):
            if self.pool_kind != "threads" and self._procs is None:
                self.start_workers()
            if self._procs is not None:
                # a lone job goes to the workers too: the runner keeps several reductions in
                # flight, and run here they would share this process's GIL
                pid_of = {id(r): pid for pid, r in self.reducers.items()}
                return self._procs.map([(pid_of[id(r)], req) for r, req in jobs], keys)
            with self._start_lock:
                if self._pool is None:
                    self._pool = cf.ThreadPoolExecutor(max_workers=self.max_workers)
            # pump_wait, not map(): the main thread must keep servicing the GUI event loop
            # while the partitions reduce, or the live window is marked unresponsive
            from .parallel import pump_wait
            return pump_wait([self._pool.submit(_run_one, j) for j in jobs])
        return [_run_one(j) for j in jobs]

    def close(self):
        """Shut down the worker processes and the thread pool.
        """
        if self._procs is not None:
            self._procs.close()
            self._procs = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def sdi_lender_ids(self, selected: Sequence[Any]) -> Optional[List[Any]]:
        """Return the partitions whose frames feed the others' SDI pools.

        Parameters
        ----------
        selected : sequence
            The partitions the configuration includes.

        Returns
        -------
        ids : list or None
            All partitions or the selected ones (``sdi_lenders``), or None when no partition has an SDI
            pool.
        """
        if not any((getattr(r, "_reflib_spec", None) or {}).get("sdi") for r in self.reducers.values()):
            return None
        return list(self.reducers) if str(self.sdi_lenders) == "all" else list(selected)

    def _prepare_siblings(self, config: Config, sel, injections, k_scan, tag, fm_sources, per, kind):
        """Run the SDI mode's first round, the lending partitions' frames prepared for this configuration.

        The frames are prepared in parallel as each partition would prepare its own
        (``ReductionRequest.stage = "prepare"``), then handed to the partition jobs, in one shared-memory
        block with worker processes.  Nothing is done when the configuration's ``nkeep_sdi`` is 0, the
        point without an SDI pool.

        Parameters
        ----------
        config : Config
            The configuration.
        sel : list
            The partitions it includes.
        injections : sequence of Source or None
            Sources to inject.
        k_scan : bool
            Whether the reduction scans the number of KL modes.
        tag : str
            Label of the request.
        fm_sources : sequence of Source or None
            KLIP-FM test sources.
        per : int
            Threads per job.
        kind : str
            The pool kind in use.

        Returns
        -------
        siblings : dict, SharedArrays or None
            The prepared frames for the requests, or None.
        shared : SharedArrays or None
            The shared-memory block to release afterward, or None.
        """
        lenders = self.sdi_lender_ids(sel)
        if lenders is None or int(round(float(config.params.get("nkeep_sdi", 0) or 0))) <= 0:
            return None, None
        jobs = [(self.reducers[pid], ReductionRequest(config.params_for(pid), self.scaled_sources(injections, pid),
                                                      k_scan, f"{tag}_{pid}_prep",
                                                      fm_sources=self.scaled_sources(fm_sources, pid),
                                                      threads=per, stage="prepare"))
                for pid in lenders]
        payload = {pid: sf for pid, sf in zip(lenders, self._map(jobs))}
        if kind != "processes":
            return payload, None
        from .parallel import SharedArrays
        arrays, extra = {}, {}
        for pid, sf in payload.items():
            arrays[f"{pid}/bcube"] = sf.bcube
            if sf.mcube is not None:
                arrays[f"{pid}/mcube"] = sf.mcube
            extra[pid] = {"lam_m": float(sf.lam_m), "bang": np.asarray(sf.bang, float)}
        if getattr(self, "_no_shared_memory", False):
            return payload, None
        try:
            shared = SharedArrays.create(arrays, extra)
        except Exception as exc:          # no shared memory here: the arrays travel with the jobs
            self._no_shared_memory = True
            self._log(f"  (shared memory unavailable for the SDI frames, {exc!r}; passing them with the jobs, "
                      f"which is slower)")
            return payload, None
        return shared, shared

    def reduce_config(self, config: Config, injections: Optional[Sequence[Source]] = None,
                      k_scan: bool = False, tag: str = "", selected: Optional[Sequence[Any]] = None,
                      fm_sources: Optional[Sequence[Source]] = None, extras: bool = False,
                      card: bool = False) -> EvalImages:
        """Reduce every selected partition with its own parameters and combine the images.

        Each partition injects the sources at its own point of their spectrum (:meth:`scaled_sources`).
        With worker processes the jobs get threads by expected cost (:meth:`_plan`), and the job expected
        to run longest goes first among those submitted together.

        Parameters
        ----------
        config : Config
            The configuration, with each partition's parameters.
        injections : sequence of Source or None, default None
            Sources to inject.
        k_scan : bool, default False
            Whether to return images for every number of KL modes.
        tag : str, default ""
            Label of the requests.
        selected : sequence or None, default None
            Partitions to reduce.  None takes ``config.selected``.
        fm_sources : sequence of Source or None, default None
            KLIP-FM test sources, for ``fm_image`` combined with the same partition weights.
        extras : bool, default False
            Whether to make the ``nosub`` and ``cadi`` reference images.
        card : bool, default False
            Whether the partition jobs may go to the card workers (``card_share``).

        Returns
        -------
        ev : EvalImages
            The combined images, the per-partition stack, the weights and each partition's notes.
        """
        sel = list(config.selected if selected is None else selected)
        kind = self.start_workers() if self.max_workers > 1 else "serial"
        per = self.job_threads(len(sel))          # threads inside each partition job
        siblings, shared = self._prepare_siblings(config, sel, injections, k_scan, tag, fm_sources, per, kind)
        # each partition injects the sources at its own point of their spectrum, when they have one
        jobs = [(self.reducers[pid], ReductionRequest(config.params_for(pid), self.scaled_sources(injections, pid),
                                                      k_scan, f"{tag}_{pid}",
                                                      fm_sources=self.scaled_sources(fm_sources, pid), extras=extras,
                                                      threads=per, siblings=siblings, card=bool(card)))
                for pid in sel]
        keys = None
        plan = self._plan(jobs, per) if kind == "processes" else None
        if plan is not None:
            threads, walls = plan
            for (_, req), t in zip(jobs, threads):
                req.threads = t
            now = time.monotonic()
            # the job expected to run longest goes first among those submitted together;
            # one submitted later goes ahead of an earlier one only when it is longer by
            # more than the time between them, so a trial's jobs are not held back for long
            keys = [now - w for w in walls]
        try:
            results = self._map(jobs, keys)
        finally:
            if shared is not None:
                shared.release()
        stack = np.stack([r.image for r in results])
        w = np.array([self.partition_weight(pid) for pid in sel], float)
        img = combine_stack(stack, w)
        meta = {str(pid): r.meta for pid, r in zip(sel, results)}
        ev = EvalImages(img, stack, w, sel, meta)

        def _comb(attr):
            ims = [getattr(r, attr) for r in results]
            if any(im is None for im in ims):
                return None, None
            st = np.stack(ims)
            return combine_stack(st, w), st
        ev.fm_image, ev.fm_stack = _comb("fm_image")
        ev.nosub, _ = _comb("nosub")
        ev.cadi, _ = _comb("cadi")
        return ev

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        """Reduce every partition with the same parameters, the single-block interface.

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult
            The combined images, with the partitions in ``meta``.
        """
        cfg = Config(params=dict(req.params), per_partition={p: dict(req.params) for p in self.reducers},
                     selected=list(self.reducers), x=np.zeros(0))
        ev = self.reduce_config(cfg, req.injections, req.k_scan, req.tag, fm_sources=req.fm_sources,
                                extras=req.extras, card=bool(getattr(req, "card", False)))
        return ReductionResult(ev.image, 1.0, {"partitions": ev.partitions}, fm_image=ev.fm_image,
                               nosub=ev.nosub, cadi=ev.cadi)

    def _plan(self, jobs, base: int):
        """Plan each partition job's threads and expected wall time from the partitions' cost estimates.

        Threads scale with a job's expected cost against the mean of the latest jobs, rounded to a power
        of two from 1 to ``4 x base``, so a big night at a small bin gets many, a light job one, and the
        mean stays near ``base``.  The jobs running at a moment can hold more than ``max_workers``, since
        the long ones run longest.  The images do not depend on the thread count.

        Parameters
        ----------
        jobs : list of tuple
            The ``(reducer, request)`` pairs.
        base : int
            Average threads per job (``--threads-per-job``).

        Returns
        -------
        plan : tuple of 2 lists or None
            Threads and expected wall time of each job, or None when a partition has no estimate.
        """
        costs = []
        for r, req in jobs:
            f = getattr(r, "cost_estimate", None)
            c = f(req.params, k_scan=req.k_scan, n_inject=len(req.injections or ()),
                  fm=bool(req.fm_sources)) if callable(f) else None
            if c is None or not np.isfinite(c) or c <= 0:
                return None
            costs.append(float(c))
        with self._start_lock:
            if self._recent_costs is None:
                self._recent_costs = collections.deque(maxlen=max(64, 4 * self.n_procs()))
            self._recent_costs.extend(costs)
            mean = float(np.mean(self._recent_costs))
        cap = max(1, min(4 * int(base), int(self.max_workers)))
        threads = []
        for c in costs:
            t = base * c / max(mean, 1e-12)
            t = 1 if t <= 1.0 else 2 ** int(round(np.log2(t)))
            threads.append(int(min(max(t, 1), cap)))
        walls = [c / t ** 0.85 for c, t in zip(costs, threads)]
        return threads, walls

    def describe(self) -> Dict[str, Any]:
        """Return the reducer's settings, for a run's record.

        Returns
        -------
        info : dict
            The base keys, the weighting, the threads, the representative, each partition's settings,
            and the injected spectrum, data edits, SDI lenders and card share when set.
        """
        d = super().describe()
        d.update(weighting=self.weighting, max_workers=self.max_workers, threads_per_job=self.threads_per_job,
                 representative=str(getattr(self, "representative", next(iter(self.reducers)))),
                 partition_label=getattr(self, "partition_label", "night"),
                 provenance=getattr(self, "provenance", None),
                 partitions={str(k): r.describe() for k, r in self.reducers.items()})
        if getattr(self, "injection_spectrum_info", None):
            d["injection_spectrum"] = dict(self.injection_spectrum_info)
        if getattr(self, "data_edits", None):
            d["data_edits"] = dict(self.data_edits)
        if self.sdi_lender_ids(list(self.reducers)) is not None:
            d["sdi_lenders"] = str(self.sdi_lenders)
        if float(self.card_share or 0) > 0:
            d["card"] = {"share": float(self.card_share), "workers": int(self.card_workers)}
        return d


KLIP_GRAM_MODES = ("target", "zone")


def set_zone_center(reducer: Reducer, center: Optional[Sequence[float]], angles: Optional[str] = None) -> List[Any]:
    """Set the zones' center and angle convention as defaults of every KLIP reducer in a reducer.

    The ALES paper reduction of MWC 758 is ``set_zone_center(red, (30, 30), "getzone2015")``.  Set it
    before the worker pool starts, since the workers keep the reducers they were forked with.

    Parameters
    ----------
    reducer : Reducer
        A :class:`PartitionedReducer`, whose partitions are set, or a single reducer.
    center : sequence of 2 floats or None
        The ``(x, y)`` pixel the zones' radii and sector angles are measured from.  None goes back to
        ``(nx/2, ny/2)`` and "ccw".
    angles : {"ccw", "getzone2015"} or None, default None
        How the sector angles are counted (:func:`klip_tpe.klip.zone_indices`).  None and "ccw" take
        the default.

    Returns
    -------
    done : list
        The partitions it was set on, or ``[None]`` for a single reducer.

    Raises
    ------
    ValueError
        If ``center`` does not have two values or ``angles`` is unknown.
    """
    from .klip import ZONE_ANGLES
    c = None if center is None else _zone_center({"zone_center": center})
    if angles is not None and angles not in ZONE_ANGLES:
        raise ValueError(f"zone angles are one of {ZONE_ANGLES}, not {angles!r}")
    subs = getattr(reducer, "reducers", None)
    items = list(subs.items()) if isinstance(subs, dict) and subs else [(None, reducer)]
    done = []
    for pid, r in items:
        d = getattr(r, "defaults", None)
        if isinstance(d, dict):
            if c is None:
                d.pop("zone_center", None)
                d.pop("zone_angles", None)
            else:
                d["zone_center"] = c
                if angles is None or angles == "ccw":
                    d.pop("zone_angles", None)
                else:
                    d["zone_angles"] = angles
            done.append(pid)
    return done


def set_klip_gram(reducer: Reducer, mode: str) -> List[Any]:
    """Set ``klip_gram`` as the default of every KLIP reducer in a reducer.

    Set it before the worker pool starts, since the workers keep the reducers they were forked with.

    Parameters
    ----------
    reducer : Reducer
        A :class:`PartitionedReducer`, whose partitions are set, or a single reducer.
    mode : {"target", "zone"}
        How the per-target KLIP forms its projection (:attr:`klip_tpe.klip.KLIPParams.gram`).  None
        or empty takes "target".

    Returns
    -------
    done : list
        The partitions it was set on, or ``[None]`` for a single reducer.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    mode = str(mode or "target").lower()
    if mode not in KLIP_GRAM_MODES:
        raise ValueError(f"klip_gram is one of {KLIP_GRAM_MODES}, not {mode!r}")
    subs = getattr(reducer, "reducers", None)
    items = list(subs.items()) if isinstance(subs, dict) and subs else [(None, reducer)]
    done = []
    for pid, r in items:
        d = getattr(r, "defaults", None)
        if isinstance(d, dict) and "klip_gram" in d:
            d["klip_gram"] = mode
            done.append(pid)
    return done


def reducer_class(backend: str = "klip"):
    """Return the reducer class for a backend name.

    Parameters
    ----------
    backend : {"klip", "pyklip", "vip"}, default "klip"
        The backend, "klip" (or "builtin", "default") for the built-in annular KLIP.  pyKLIP's and
        VIP's are imported when asked for (:mod:`klip_tpe.backends`).

    Returns
    -------
    cls : type
        The reducer class.

    Raises
    ------
    ValueError
        If ``backend`` is unknown.
    """
    b = (backend or "klip").lower()
    if b in ("klip", "builtin", "default"):
        return KLIPReducer
    if b == "pyklip":
        from .backends.pyklip import PyKLIPReducer
        return PyKLIPReducer
    if b == "vip":
        from .backends.vip import VIPReducer
        return VIPReducer
    raise ValueError(f"unknown backend {backend!r} (klip | pyklip | vip)")
