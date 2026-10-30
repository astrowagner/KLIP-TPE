"""The run protocol: per annulus **calibrate -> search -> validate -> products**,
with a checkpoint after every evaluation (written at most every
``Runner.CHECKPOINT_EVERY_S``) and crash-resume as a first-class
feature (``config lives in the checkpoint``: ``Runner.resume`` takes the run
directory and the un-serialisable objects -- reducer, objective, sampler -- and
nothing else, so a resumed run can never silently mix configurations).

Outputs in ``run_dir``::

    run_setup.txt / run_setup.json   full configuration, search space, defaults
    results.txt                      one row per evaluation (IDL-compatible layout)
    results.jsonl                    one JSON record per evaluation (positions, per-source S/N ...)
    checkpoint.json                  search state (taken every evaluation, written at most every 10 s and
                                     on any exit; atomic and fsynced, so it survives a machine crash;
                                     resume also reads a sync client's copies of it)
    checkpoint.prev.json             the save before the newest (a crash mid-write leaves one of the two)
    annulusNN/                       calibration log, validation table, winner record, FITS products,
                                     evalNNNN_setup.txt / calibNNNN_setup.txt / valid_candNN_setup.txt /
                                     final_setup.txt, verify_report.txt, verify_curve.txt, param_verify/
    klip_stitched_running*.fits      equal-weight running stitch (every ``stitch_every`` evals)
    klip_stitched[_inj|_snr|_snr_inj].fits, klip_stitched_nights[_inj].fits.gz,
    klip_stitched_params.txt, contrast_curve.txt, verify_curve.txt, cand/
    final_results.json               validated winners of every annulus (+ stitch weights)

Protocol extensions over the plain loop (see the IDL spec, notes A/C):

* ``Runner.extend`` (A §7.3) -- reopen a finished/interrupted run with new per-annulus
  budgets: validate-only or continue the search with the contrast frozen.
* ``opt_width`` (A §3.1) -- greedy adaptive annuli: ``width`` is a searched dim, the
  winner's outer edge commits the next annulus' inner edge until ``r_cap``.
* ``k_mode='scan'`` (A §5.9 b/d) -- legacy k-scan per eval with ``near2m_pnkpick`` for
  partitioned reducers (per-partition k by greedy coordinate ascent).
* verification hooks (C) -- ``verify`` / ``param_verify`` / ``candidates`` after each
  annulus and on the final stitch; every hook is exception-guarded.
"""
from __future__ import annotations

import collections
import contextlib
import glob
import json
import os
import pickle
import shutil
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import __version__, durable
from .heartbeat import Heartbeat
from .liveness import liveness_mode
from .metrics import (Objective, Source, mawet_peak_snr, nanmedian_even, radprof, source_xy,
                      star_center)
from .contrast_grid import ContrastGridConfig
from .optimizers import GridSearch, History, Optimizer, RandomSearch, TPE
from .positions import PositionSampler, n_sources_rule
from .reducer import EvalImages, PartitionedReducer, Reducer, ReductionRequest
from .space import Config, SearchSpace

__all__ = ["RunConfig", "CalibrationConfig", "ValidationConfig", "EvalRecord", "AnnulusResult", "Runner", "RunCallback",
           "load_checkpoint", "checkpoint_candidates"]

SCAN_MODES = ("scan", "scan_rescore")

_UNSYNCED_LOCK = threading.Lock()      # guards Runner._unsynced


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def _from_dict_filtered(cls, d: Optional[Dict[str, Any]]):
    """Build a dataclass from a dict, ignoring unknown keys (forward/backward
    compatible checkpoints)."""
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class CalibrationConfig:
    """Injection-contrast calibration (trial 0).

    target
        acceptable window for the default config's median S/N; ``aim`` is the value
        the contrast is rescaled toward (``fac = aim / S/N`` clipped to ``step_clip``).
    forced
        per-annulus list; an entry > 0 bypasses calibration for that annulus.
    ceiling
        upper bound on the calibrated contrast (``cmax_cal``); None = off.
    n_remeasure
        fresh-position measurements per trial (median taken).
    scan_k
        pick the default config's k by a k-scan once (needs ``supports_kscan``).
    overrides
        parameter overrides during calibration only (e.g. ``{'bin': 64}``).
    recal_check / recal_ntop / recal_budget
        re-calibration revisit: after ``recal_check`` search evaluations, if the median
        of the top ``recal_ntop`` scores is outside ``target`` the contrast is rescaled
        and the annulus search restarted (at most ``recal_budget`` times).
    warm_start_previous
        start the calibration of annulus i from annulus i-1's contrast.
    max_contrast
        hard ceiling on the injection contrast, the one IDL's ``cmax_cal`` was for, on by
        default: a source brighter than a tenth of the star is not a high-contrast
        injection.  Without it a default configuration whose S/N does not follow the
        injected flux walks the contrast up by x10 a trial for ever -- run A2's [6, 12] px
        annulus (three sources 4 FWHM apart on a 9-px ring, every reference frame holding
        the source) scored S/N 0-1 from 3e-5 to 76 and the search then ran at contrast
        4.6e+03, a planet 4,600 times its star.  At the cap the k-scan is asked whether any
        k detects the sources (at that radius k = 1 does, at S/N 7-9) and the walk starts
        again there; failing that the contrast stays AT the cap, the annulus is marked
        ``uncalibrated`` in ``calibration.json`` and the log says so, and the search runs
        anyway -- a default configuration that cannot see an injection is a statement about
        the default, not about the problem, and finding a configuration that can is the
        whole point.  ``None`` removes the ceiling (and with it the k rescue).
    """

    target: Tuple[float, float] = (4.0, 6.0)
    aim: float = 5.0
    max_trials: int = 8
    step_clip: Tuple[float, float] = (0.1, 10.0)
    forced: Optional[List[float]] = None
    ceiling: Optional[float] = None
    n_remeasure: int = 2
    scan_k: bool = True
    overrides: Dict[str, Any] = field(default_factory=dict)
    recal_check: int = 10
    recal_ntop: int = 5
    recal_budget: int = 4
    warm_start_previous: bool = False
    max_contrast: Optional[float] = 0.1


@dataclass
class ValidationConfig:
    n_top: int = 3
    n_valid: int = 8
    distinct_tol: float = 1e-4
    commit: str = "median_trial"
    #: how two candidates count as the same.  With ``"image"`` a configuration whose clean
    #: reduction equals an earlier candidate's, bit for bit, is skipped for the next in the
    #: search's order.  Parameters that cannot change the reduction (a deselected night's,
    #: frame cuts with no frame between them, an even high-pass width, which runs as the next
    #: odd one) made two parameter vectors distinct.  In NEAR2's run_20261027_222223 the 25
    #: candidates of an annulus were 7 to 20 distinct reductions.  ``"vector"`` compares the
    #: parameter vectors only (``distinct_tol``), as the IDL does and as runs saved before
    #: 2026-10-28 resume.
    dedup: str = "image"
    #: how deep in the search's order ``"image"`` looks, in units of ``n_top``
    dedup_depth: int = 8
    #: fresh trials for the elected winner, whose median is then its validated score.  The
    #: election picks the largest of noisy medians, so the elected one is high on its own
    #: draws (in NEAR2 by 0.22 S/N on average, up to 0.94).  The median of fresh draws is not.
    #: The election's median stays as ``AnnulusResult.election_score``.  A lone candidate's
    #: median was not selected on, and it stands.  None means ``n_valid``.  0 reports the
    #: election's median, as the IDL does and as runs saved before 2026-10-28 resume.
    n_rescore: Optional[int] = None
    #: a final round for a sharper election.  The ``n_final`` candidates with the largest
    #: medians are scored on ``n_final_draws`` common draws, the same injections for each, and
    #: the largest median there is elected.  With 0 or 1 the first round elects.  The first
    #: round's medians are each on their own ``n_valid`` draws and scatter by a few tenths of
    #: S/N, about as far as the top candidates lie apart.  On common draws the candidates'
    #: scores move together, so their differences scatter far less.  In NEAR2's
    #: run_20261027_222223 the list held a better reduction than the elected one in four of
    #: seven annuli, by 1 to 3 %.
    n_final: int = 0
    n_final_draws: int = 200


@dataclass
class RunConfig:
    """Everything that defines a run except the data objects.

    New protocol fields (all default to the pre-existing behaviour):

    opt_width / width_range / r_cap
        adaptive annuli (A §3.1): ``ann_edges`` holds the first inner edge only, the
        space carries a searched integer ``width`` dim, each eval reduces
        ``[inrad, min(inrad + width, r_cap)]`` with the sources at the zone mid-radius,
        and the winner's outer edge becomes the next annulus' inner edge.
    k_mode
        ``search`` (k searched), ``scan_rescore`` (per-eval k-scan, one common k) or
        ``scan`` (per-eval k-scan; partitioned reducers pick per-partition k with
        ``near2m_pnkpick``); both scan modes re-score honestly at fresh positions.
    stitch_every
        running-stitch cadence in evaluations (0 = only at annulus end).
    write_setup_files
        write the IDL-style ``evalNNNN_setup.txt`` family.
    verify / verify_n_boot
        ``klip_tpe.verify`` on the winner's per-partition stacks after each annulus
        (``verify_report.txt``) plus the per-eval limit curve (``verify_curve.txt``).
    param_verify / n_pv / pv_divmin
        parameter-ensemble persistence stage (``annulusNN/param_verify/``); ``None`` =
        on when the space has partitions.
    candidates / cand_snrmin / cand_verify_top
        blind candidate search on the running and final stitches (``cand/``).
    legacy_stitch
        equal-weight final stitch and no seam trim of the contrast curve.
    edge_padding
        where annulus edges are padded.  ``"between"`` (the default, and unlike the IDL):
        only an edge shared by two annuli.  The reduction zone there takes the reducer's
        ``zone_pad``, the stitch tile 2 px and a seam re-run 3 px, while the run's
        innermost inner edge and outermost outer edge are reduced and stitched as given.
        ``"all"`` pads every edge, as the IDL does.  A run saved before this field existed
        resumes with ``"all"``, so its evaluations stay comparable.
    batch / batch_mode
        ``batch > 1`` keeps that many trials in flight.  ``"pipeline"`` (the default): when
        the oldest trial is in, the next proposal is asked from the history so far and goes
        out, so ``batch`` trials are always reducing.  ``"sync"``: the optimizer proposes
        ``batch`` trials from one posterior and the next batch waits for all of them.  Either
        way the history is appended in index order and a seed gives one trajectory, but not
        the trajectory of ``batch = 1``.  A run saved before ``batch_mode`` existed resumes
        with ``"sync"``.  The k-scan modes always use ``"sync"``.
    partition_weighting
        ``'equal'`` | ``'sqrt_texp'`` pass-through to a :class:`PartitionedReducer`.
    fm_curve
        KLIP-FM cross-check curve (A §9.1): after validation one clean forward-model
        pass at the winner config on the golden-angle test spiral -> ``AnnulusResult.
        fm_curve``, ``annulusNN/best_fm.fits`` and a ``# KLIP-FM`` section in the
        contrast-curve files.  Needs ``reducer.supports_fm``; skipped in the scan modes.
    fm_preview
        the §9.3 live preview: at every new search best one extra clean FM pass at
        that eval's config with its own sources (``best_images['fm']``, shown by the
        display).  Off by default -- it costs a reduction per new best.
    contrast_grid
        the injection-recovery contrast curve after validation
        (:mod:`klip_tpe.contrast_grid`): one injected source per reduction over a grid of
        separations and position angles at the winner config, each iterated to S/N 5 ->
        ``AnnulusResult.grid_curve``, ``annulusNN/contrast_grid.json`` and ``.txt``, and the
        main curve of the run-level ``contrast_curve.txt``.  A run saved before it existed
        resumes without it.
    """

    ann_edges: List[float] = field(default_factory=lambda: [11.0, 44.0])   # px
    n_iter: Any = 200                     # int or per-annulus list (total evaluations incl. seed)
    n_init: Any = 40                      # int or per-annulus list (random warm-up rows incl. seed)
    search_mode: str = "tpe"              # tpe | random | grid
    gamma: float = 0.25
    ncand: int = 48
    prior_weight: float = 0.25
    bw_floor: float = 0.08
    pbest: float = 0.0                    # best-anchored candidates; harmful under objective noise (addendum 2) -- keep 0
    p_local: float = 0.15
    n_elite: int = 5
    explore_frac: float = 0.15
    blocks: Any = "univariate"          # TPE density model: univariate (reference) | partitions | full | [[names], ...]
    warmstart_tie: Optional[bool] = None  # None -> True if partitions
    link_params: List[str] = field(default_factory=lambda: ["bin", "filter"])
    grid_axes: Optional[List[str]] = None
    seed: Optional[int] = None
    contrast0: float = 3e-5
    n_sources: Any = None                 # None / 0 -> per-annulus rule; an int everywhere; a list per annulus (0 / None entries -> the rule)
    #: No injection inside this many FWHM of the star, in every stage (search, calibration,
    #: validation, param_verify, products): it sets ``min_r_fwhm`` on the run's one sampler,
    #: here and on a resume.  None leaves the sampler's own floor (``PositionSampler``: 1.5).
    #: On NEAR 1.0 FWHM is 1.0 lambda/D to half a percent, and the AGPM reaches 0.9 lambda/D.
    min_r_fwhm: Optional[float] = None
    #: Fresh-position measurements per SEARCH trial, averaged into one score (1 = the IDL's
    #: single draw).  Not to be confused with ``CalibrationConfig.n_remeasure``, which is the
    #: same idea applied to the calibration trials and the k-scan only.
    #:
    #: The objective's only stochastic input is the azimuth anchor of the injected sources,
    #: and it is not a small effect: on HIP 65426 F1140C, repeated evaluations of *identical*
    #: configurations scatter with sd 0.84 against a total useful range of 0.86-7.11.  A
    #: search that cannot tell 0.84 of S/N apart is ranking noise for much of its budget.
    #: Averaging 3 draws measured 0.48 -- a factor 1.74, i.e. the sqrt(3) an independent
    #: draw predicts, which is itself the evidence that the draws are independent.
    #:
    #: The MEAN, not the median: on the same data median-of-3 reached only 0.68, because the
    #: residuals are mildly left-skewed (-0.85) with no excess kurtosis (-0.03) -- there are
    #: no heavy tails for a median to earn its efficiency back on.
    #:
    #: Costs n + 1 reductions per trial against 2 for a single draw (the clean image depends
    #: on the configuration alone and is reduced once; the later draws run in pairs), so at
    #: fixed wall clock it buys ranking fidelity with trials: about twice the time per trial
    #: at n = 3.  Raising the source count is the cheaper lever where the noise ring allows it,
    #: since one reduction serves every source; a fresh draw additionally re-randomises the
    #: azimuth anchor, which extra sources in a single draw do not.
    n_remeasure: int = 1
    inject_inset_fwhm: float = 1.0        # injection band inset from the annulus edges
    pair_area_midpoint: bool = True       # 2 sources -> both at sqrt((r_in^2+r_out^2)/2), 180 deg apart (IDL 2026-09-05)
    #: Every injection of every stage (search, k-scan re-score, calibration, validation,
    #: param_verify) at this one separation, arcsec, instead of the band the annulus gives
    #: (``pair_area_midpoint``, ``inject_inset_fwhm``): a companion's own radius, say, so the
    #: search is scored where it matters.  Two sources are still 180 deg apart; the sampler's
    #: floor (``min_r_fwhm``) still applies; it must lie in every annulus the run scores.
    #: None (the default): the band, as before.
    inject_rho: Optional[float] = None
    #: Freeze the search's injected sources -- radii AND azimuths -- for a whole annulus, so
    #: every configuration is scored on one speckle realisation.  OFF by default, because it
    #: is NOT what ``optimize_near_2_tpe`` does and it defeats something the IDL designed in
    #: deliberately.
    #:
    #: What the IDL does (``near2m_randpos``, and its own comment says so): the source
    #: SEPARATIONS are a deterministic ladder, identical for every evaluation of an annulus,
    #: so the score is never biased by radial throughput luck -- but the AZIMUTH anchor
    #: ``th0 = randomu(seed)*360.`` is re-drawn on every evaluation, the comment's reason
    #: being "so sources still rotate eval-to-eval (anti-gaming)".  ``near2m_randpos`` is
    #: called from inside ``for it=it_start, n_iter-1``, at four call sites.
    #:
    #: :class:`PositionSampler`'s ``"spread"`` strategy already reproduces that exactly --
    #: same ``r0 = r_lo + 0.5*span/n`` ladder, same random ``th0`` per call -- so the port was
    #: faithful before this flag existed.
    #:
    #: The flag was added on 2026-09-13 on a misreading: a NEAR run showed the *same*
    #: configuration scoring over a spread of 6.6 (sd ~1.0), which was taken for a porting
    #: bug.  The variance is real, and the IDL accepts it on purpose: its defence is the
    #: validated election, which re-scores the top candidates at FRESH positions, not a
    #: frozen injection pattern.  Freezing the azimuths lets the optimizer tune to the
    #: speckle realisation at those particular position angles, which is the gaming the IDL
    #: comment names -- a search score that rises without sensitivity improving.  A NEAR run
    #: spent 2037 evaluations injecting at PA 87.39 / 267.39 before this was caught.
    #:
    #: Left in place, off, because pairing is a legitimate experiment to be able to run (it
    #: is how you measure how much of a search's gain is realisation luck); it is a
    #: documented departure from the reference, never the default.
    fixed_sources: bool = False

    k_mode: str = "search"                # search | scan_rescore | scan
    k_scan_max: int = 30
    seed_default: bool = True             # evaluate the default config as trial 0
    defaults: Dict[str, Any] = field(default_factory=dict)   # default-config overrides
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    contrast_grid: ContrastGridConfig = field(default_factory=ContrastGridConfig)
    kfit_order: int = 2
    save_fits: bool = True
    save_eval_images: bool = True         # per-eval injected/clean crops (annulusNN/evals/, IDL evalNNNN_score_inj)
    batch: int = 1                        # trials in flight (>1: another trajectory than 1, see batch_mode)
    batch_mode: str = "pipeline"          # "pipeline" (a new proposal when the oldest trial is in) | "sync"
    bench_tag: str = "none"
    notes: str = ""
    # --- protocol extensions (defaults reproduce the previous behaviour) ---
    opt_width: bool = False
    width_range: Tuple[float, float] = (15.0, 30.0)
    r_cap: float = 70.0
    stitch_every: int = 10
    write_setup_files: bool = True
    verify: bool = False
    verify_n_boot: int = 100
    param_verify: Optional[bool] = None
    n_pv: int = 20
    pv_divmin: float = 0.05
    candidates: bool = False
    cand_snrmin: float = 2.0
    cand_verify_top: int = 5
    legacy_stitch: bool = False
    edge_padding: str = "between"         # "between" (only where annuli meet) | "all" (every edge, the IDL)
    partition_weighting: Optional[str] = None
    fm_curve: bool = True
    fm_preview: bool = True              # live KLIP-FM preview at each new best (IDL: 'after 1st best')
    #: Before a new run's first evaluation, move every searched reduction dimension alone
    #: and see whether it changes the clean reduction
    #: (:func:`klip_tpe.liveness.check_live_dimensions`, about ``1 + ndim`` reductions; a
    #: resumed run is not re-checked).  ``"warn"`` logs a dimension that never does and goes
    #: on, ``"strict"`` (or ``True``) stops the run, ``False`` / ``"off"`` skips the check.
    #: Off here; ``klip-tpe near`` / ``generic`` warn (``--strict-liveness``,
    #: ``--no-liveness-check``).  A dead dimension is the bug class of the IDL ``parstr``
    #: omission, which the search itself cannot see.
    liveness_check: Union[bool, str] = False
    #: Known sources to take out of the frames before EVERY reduction the run makes --
    #: ``[(rho_arcsec, pa_deg, contrast), ...]``, each injected with the negative of its
    #: contrast (:mod:`klip_tpe.companion`; fit the values there, in a reference-star
    #: configuration).  Declaring a companion ``known`` keeps injections 1.5 FWHM away and
    #: drops noise apertures within 1.5 FWHM of it, which is enough on the ground, where
    #: speckles dominate its outer light.  It is not enough when the speckle floor is far
    #: below the companion: on JWST/NIRCam, HIP 65426 b's light, spread by the reduction over
    #: its KLIP sector, set the metric's ring scatter at 5-14x the companion-free value
    #: across the whole inner annulus and ~2x at 2.2" -- the search then learned to suppress
    #: the companion rather than the star.  Removing it first keeps it out of the basis, the
    #: rings and the score.  Only the positive injections are ever scored; the known-source
    #: exclusions still apply, to the residual.  Empty (the default) changes nothing.
    subtract_known: List[Tuple[float, float, float]] = field(default_factory=list)
    #: What the frames are, when they are not the instrument's default: ``cube_name`` (the
    #: cube file read for each night), ``dropped_frames`` (``{"n4": [551], ...}``, frames left
    #: out of a night at load, by their index in its cube), ``subtracted_sources`` (``[[rho,
    #: pa, contrast], ...]`` taken out of every frame once, at load,
    #: :meth:`KLIPReducer.subtract_sources`) and ``injected_sources`` (added the same way), as
    #: the reducer reports them in its ``data_edits``.  A new run records them here, so the
    #: checkpoint holds them, and a resume refuses a reducer whose frames differ.  The command
    #: line fills the reducer from here on ``klip-tpe resume`` and ``extend`` when
    #: ``--cube-name`` / ``--drop-frames`` / ``--subtract-sources`` / ``--inject-sources`` are
    #: not typed.  Empty: the default frames.
    data_edits: Dict[str, Any] = field(default_factory=dict)

    def per_annulus(self, val, ia: int) -> int:
        if isinstance(val, (list, tuple, np.ndarray)):
            return int(val[min(ia, len(val) - 1)])
        return int(val)

    @property
    def nann(self) -> int:
        return max(len(self.ann_edges) - 1, 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        d = dict(d)
        # a run saved before edge_padding existed padded every edge: it resumes that way
        d.setdefault("edge_padding", "all")
        # and one saved before batch_mode evaluated its batches whole
        d.setdefault("batch_mode", "sync")
        d["calibration"] = _from_dict_filtered(CalibrationConfig, d.get("calibration", {}))
        # one saved before 2026-10-28 compared candidates by their vectors and reported the
        # election's median: it goes on that way, so its annuli stay alike
        v = dict(d.get("validation") or {})
        v.setdefault("dedup", "vector")
        v.setdefault("n_rescore", 0)
        d["validation"] = _from_dict_filtered(ValidationConfig, v)
        # and one saved before the injection grid existed made no grid
        cg = d.get("contrast_grid")
        d["contrast_grid"] = (_from_dict_filtered(ContrastGridConfig, cg) if isinstance(cg, dict)
                              else ContrastGridConfig(mode="off"))
        return _from_dict_filtered(cls, d)


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------
@dataclass
class EvalRecord:
    annulus: int
    index: int                                # 0-based slot in the annulus history
    phase: str
    x: List[float]
    config: Dict[str, Any]
    sources: List[Tuple[float, float, float]]
    score: Optional[float]                    # search objective (None = failed)
    raw_score: Optional[float]                # uncorrected median S/N
    per_source: List[Optional[float]]
    raw_per_source: List[Optional[float]]
    clean_per_source: Optional[List[Optional[float]]]
    #: each selected partition's own raw S/N.  Empty since 2026-10-25: the search scores the
    #: combined image only, and measuring every night's image of every draw as well took two
    #: thirds of the main thread's time per trial (NEAR2, 5 draws, 6 nights).  Records
    #: written before keep their values, and the readers take either.
    partition_snr: Dict[str, Optional[float]]
    k_used: Optional[Any]
    contrast: float
    wall_s: float
    meta: Dict[str, Any] = field(default_factory=dict)
    #: from the trial's dispatch until the last of its reductions was back (s).  ``wall_s``
    #: runs on to its scoring, which with several trials in flight also holds the wait for
    #: workers and for the main thread to reach it.  ``meta['t_dispatch']`` is the dispatch
    #: time itself (epoch s), from which the display counts the trial's whole time.
    reduce_s: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_json_default)


@dataclass
class AnnulusResult:
    annulus: int
    inrad: float
    outrad: float
    contrast: float
    search_best_index: int
    search_best_score: float
    validated: bool
    winner_index: int
    winner_score: float
    winner_x: List[float]
    winner_config: Dict[str, Any]
    validation_table: List[Dict[str, Any]]
    validation_samples: Dict[str, List[float]]     # r_as, snr pooled over the winner's trials
    contrast_curve: Optional[Dict[str, List[float]]] = None
    distance_to_bounds: Optional[Dict[str, float]] = None
    n_evaluations: int = 0
    winner_sources: List[Tuple[float, float, float]] = field(default_factory=list)   # committed trial's injections
    partitions: List[str] = field(default_factory=list)                              # selected partition ids
    k_used: Any = None                                                               # representative / per-partition k
    fm_curve: Optional[Dict[str, List[float]]] = None                                # KLIP-FM cross-check (A §9.1)
    per_source: Optional[List[Optional[float]]] = None                               # raw per-source S/N, committed trial
    k_default: Optional[int] = None                                                  # calibration's default-config k
    #: the elected candidate's median on the election's own draws, when ``winner_score`` is
    #: its median on fresh draws (``ValidationConfig.n_rescore``), else None.  The election
    #: picks the largest of noisy medians, so this one is high.
    election_score: Optional[float] = None
    #: the fresh trials' scores behind ``winner_score``, their injections (as the
    #: ``trial_sources`` of ``validation.json``) and the committed one, whose sources are
    #: ``winner_sources``
    rescore_trials: Optional[List[Optional[float]]] = None
    rescore_sources: Optional[List[List[List[float]]]] = None
    rescore_committed: Optional[int] = None
    #: ``[[eval, same_as_eval], ...]`` (0-based): configurations the validation skipped because
    #: their clean reduction equals an earlier candidate's (``ValidationConfig.dedup``)
    validation_duplicates: List[List[int]] = field(default_factory=list)
    #: the injection-recovery curve (``RunConfig.contrast_grid``): per separation ``r_as``,
    #: the median contrast at the target S/N over the settled positions (``curve``), its 16th
    #: and 84th percentiles and the counts; ``annulusNN/contrast_grid.json`` has every position
    grid_curve: Optional[Dict[str, Any]] = None
    #: the final round (``ValidationConfig.n_final``): ``n_draws`` and per entrant its
    #: ``candidate``, ``eval_index``, first-round ``validated_score``, ``final_score`` and
    #: ``trials``, with ``winner_candidate``.  ``validation_final.json`` adds the draws.
    final_round: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=_json_default))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnnulusResult":
        return _from_dict_filtered(cls, d)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in o.tolist()]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, Source):
        return o.as_tuple()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def _secs(v: Optional[float]) -> str:
    """A duration for a log line: ``1.6s``, ``212s``, or ``--`` when unknown."""
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:.1f}s" if v < 100 else f"{v:.0f}s"


def _error_why(exc: BaseException) -> str:
    """A failed evaluation's ``meta['why']``: the error, shortened for a record."""
    s = f"error: {exc!r}"
    return s if len(s) <= 300 else s[:297] + "..."


def _whys(counts: "collections.Counter", top: int = 3) -> str:
    """The commonest reasons of a count of failed draws, for a log line."""
    items = counts.most_common()
    out = ", ".join(f"{n} {w}" for w, n in items[:top])
    rest = sum(n for _, n in items[top:])
    return out + (f", {rest} other" if rest else "")


def _atomic_write(path: str, text: Union[str, bytes], keep: Optional[str] = None) -> None:
    """Replace ``path`` atomically and durably (:func:`klip_tpe.durable.write`).  After a
    crash of the machine it holds the old contents or all of the new, never nothing.
    ``keep`` moves the version being replaced to that name first."""
    durable.write(path, text, keep=keep)


#: the save before the newest, kept beside checkpoint.json (see :meth:`Runner._write_checkpoint`)
CHECKPOINT_PREV = "checkpoint.prev.json"


def _read_checkpoint(path: str) -> Dict[str, Any]:
    """The state in a checkpoint file.  Raises when the file cannot be read or is not a
    klip-tpe checkpoint (a 0-byte file left by a crash, a sync client's placeholder)."""
    with open(path) as f:
        st = json.load(f)
    if not (isinstance(st, dict) and all(k in st for k in ("config", "space", "rng_state", "ia"))):
        raise ValueError("not a klip-tpe checkpoint")
    return st


def checkpoint_candidates(run_dir: str) -> List[str]:
    """``checkpoint.json`` and every copy of it a file-sync client has made alongside --
    ``checkpoint (<name>'s conflicted copy <date> N).json`` (Dropbox), ``checkpoint 2.json``
    (macOS, iCloud), ``checkpoint (1).json`` (Google Drive), ``checkpoint.sync-conflict-...``
    (Syncthing).  ``checkpoint.json`` first when it exists."""
    main = os.path.join(run_dir, "checkpoint.json")
    others = sorted(p for p in glob.glob(os.path.join(run_dir, "checkpoint*.json")) if p != main)
    return ([main] if os.path.exists(main) else []) + others


def _checkpoint_progress(st: Dict[str, Any]) -> Tuple[int, int, int, int, float]:
    """How far a checkpoint had got: evaluations recorded, then annuli finished, then
    post-annulus hooks run, then wall time -- each only ever grows within a run.

    Ahead of all of these, the number of :meth:`Runner.extend` calls behind it.  An extend
    that re-validates an annulus records no new evaluation and goes back to an earlier
    annulus, so by the other counts its saves trail the finished run's.  They come later
    all the same, and ``checkpoint.prev.json`` (or a backup copy) of the finished run must
    not outrank them."""
    hooks = st.get("hooks_done") or {}
    n_hooks = sum(len(v) for v in hooks.values()) if isinstance(hooks, dict) else 0
    return (int(st.get("extends") or 0), int(st.get("records_count") or 0),
            int(st.get("ia") or 0) + (1 if st.get("annulus_done") else 0), n_hooks, float(st.get("wall_s") or 0.0))


def load_checkpoint(run_dir: str, log: Optional[Callable[[str], None]] = None
                    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The most advanced readable checkpoint of ``run_dir``: ``(state, path)``, or
    ``(None, None)`` when there is none.

    Not simply ``checkpoint.json``.  A synced folder can hand the real name to a stale
    version: on 2026-09-23 Dropbox moved H2K's newest save (800 evaluations, annulus done,
    param_verify recorded) into ``checkpoint (... conflicted copy ... 48).json`` and put an
    older, online-only version back as ``checkpoint.json``.  A resume that trusted the name
    would have gone back in time; one that could not read the placeholder would have started
    the run over.  So every copy is read and the one furthest along wins -- which is also
    right if a stale resume had happened in between, since its evaluations replay the same
    configurations from the same RNG state.  Ties go to ``checkpoint.json``.

    The run keeps the save before the newest as ``checkpoint.prev.json``, and it is one of
    the copies.  It wins only when ``checkpoint.json`` is missing or unreadable, which a
    crash of the machine in the middle of replacing it can leave.
    """
    say = log or (lambda s: None)
    main = os.path.join(run_dir, "checkpoint.json")
    best, bad, main_st = None, [], None
    for p in checkpoint_candidates(run_dir):
        try:
            st = _read_checkpoint(p)
        except Exception as exc:
            bad.append((os.path.basename(p), exc))
            continue
        key = _checkpoint_progress(st)
        if p == main:
            main_st = st
        if best is None or key > best[0]:
            best = (key, st, p)
    for name, exc in bad:
        say(f"  checkpoint: {name} unreadable ({exc!r}) -- skipped")
    if best is None:
        return None, None
    n_best = int(best[1].get("records_count") or 0)
    if best[2] != main:
        if main_st is not None:
            was = f"at {int(main_st.get('records_count') or 0)} evaluations"
        else:
            was = "unreadable" if os.path.exists(main) else "missing"
        if os.path.basename(best[2]) == CHECKPOINT_PREV and main_st is None:
            say(f"  checkpoint: resuming from {CHECKPOINT_PREV} ({n_best} evaluations), the save before "
                f"the newest -- checkpoint.json is {was}, as the machine stopping while it was replaced leaves it.")
        else:
            say(f"  checkpoint: resuming from {os.path.basename(best[2])} ({n_best} evaluations) -- "
                f"checkpoint.json is {was}.  A file-sync client (Dropbox?) moved the newest save aside.")
    res = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(res):
        try:
            with open(res, "rb") as f:            # complete lines: a crash can leave half of one
                n_lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
            if n_lines < n_best:
                say(f"  checkpoint: it counts {n_best} evaluations but results.jsonl holds {n_lines} lines -- "
                    f"that file may have been replaced by a sync client; the search resumes from the checkpoint")
        except Exception:
            pass
    return best[1], best[2]


def _write_fits(path: str, data: np.ndarray, header: Optional[Dict[str, Any]] = None,
                history: Sequence[str] = ()) -> None:
    """FITS writer with the provenance header of :func:`klip_tpe.stitch.fits_header`."""
    from .stitch import write_fits
    try:
        write_fits(path, data, header, history)
    except Exception:  # pragma: no cover  (no astropy)
        np.save(path.replace(".fits.gz", "").replace(".fits", "") + ".npy", data)


class _ReductionQueue:
    """Reductions submitted over time and taken back in the order submitted: the stream a
    pipelined search feeds its trials' reductions into and :meth:`Runner._trial_finish`
    reads.  ``window`` threads run them, and the rest wait their turn.  With a window of 1
    each reduction runs on this thread when its result is taken.  A reduction that raises
    yields its exception as the result."""

    def __init__(self, window: int):
        self.serial = int(window) <= 1
        self._ex = None
        if not self.serial:
            import concurrent.futures as _cf
            self._ex = _cf.ThreadPoolExecutor(max_workers=int(window), thread_name_prefix="klip-red")
        self._items: collections.deque = collections.deque()
        self._n = 0

    def submit(self, fn: Callable[[], Any]) -> None:
        self._items.append(fn if self.serial else self._ex.submit(fn))

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, Any]:
        if not self._items:
            raise StopIteration
        it = self._items.popleft()
        i, self._n = self._n, self._n + 1
        try:
            if self.serial:
                return i, it()
            from .parallel import pump_wait
            return i, pump_wait([it])[0]
        except Exception as exc:          # noqa: BLE001 - handed to the caller
            return i, exc

    def close(self) -> None:
        """Drop what has not started; what is running finishes on its own."""
        if self._ex is not None:
            self._ex.shutdown(wait=False, cancel_futures=True)


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
class RunCallback:
    """Hook interface for live display / extra products.  Subclass and override; every
    method receives the runner and may read ``runner.history``, ``runner.results``...
    Exceptions raised by callbacks are logged and swallowed so a plotting problem can
    never kill a multi-day run."""

    def on_setup(self, runner) -> None: ...
    def on_calibration(self, runner, info: Dict[str, Any], images: Optional[Dict[str, Any]]) -> None:
        """``images`` = ``{'inj': EvalImages, 'clean': EvalImages | None, 'sources': [Source],
        'per_source': [...], 'clean_per_source': [...] | None}`` of the last calibration
        measurement at the default config (None when no measurement succeeded)."""
        ...
    def on_calibration_trial(self, runner, ia: int, trial: int, contrast: float, msnr: float,
                             images: Optional[Dict[str, Any]]) -> None:
        """After every calibration trial (IDL ``near2m_calshow``): ``images`` as in
        :meth:`on_calibration` for the last measurement of that trial."""
        ...
    def on_eval(self, runner, record: "EvalRecord", inj: EvalImages, clean: Optional[EvalImages], is_best: bool) -> None: ...
    def on_draw(self, runner, draw: int, n_draws: int, record: "EvalRecord", inj: EvalImages,
                clean: Optional[EvalImages]) -> None:
        """One remeasurement of a trial under ``RunConfig.n_remeasure > 1``, before the draws
        are averaged into the trial's score.  ``record`` is that draw alone.

        Deliberately NOT ``on_eval``: the display times every ``on_eval`` to derive the
        completion rate and both ETAs, and counts them for its panel cadence, so feeding
        draws through it would report the interval between draws as the cost of a trial and
        divide the ETA by ``n_remeasure``.  A callback that ignores this hook loses nothing -- the averaged
        trial still arrives through ``on_eval``.
        """
        ...
    def on_validation_trial(self, runner, ia: int, ci: int, n_cand: int, eval_index: int, trial: int, n_valid: int,
                            inj: EvalImages, clean: Optional[EvalImages], sources, score_result, trials) -> None:
        """After every validation trial: candidate ``ci`` (of ``n_cand``, = search eval
        ``eval_index``), trial ``trial`` (of ``n_valid``), its images, sources, the
        :class:`ScoreResult` and the trial scores so far."""
        ...
    def on_validation(self, runner, table: List[Dict[str, Any]], winner: Dict[str, Any]) -> None: ...
    def on_annulus_done(self, runner, result: "AnnulusResult", winner: Dict[str, Any]) -> None: ...
    def on_stitch(self, runner, kind: str, files: Dict[str, str]) -> None: ...
    def on_verify(self, runner, annulus: int, kind: str, result: Any) -> None: ...
    def on_finish(self, runner) -> None: ...


def _same_reduction(a, b) -> bool:
    """Two reductions (:class:`~klip_tpe.reducer.EvalImages`) that are the same, bit for bit:
    the combined image and, where both have one, the per-partition stack.  A reduction is a
    pure function of its configuration and the data, on any number of threads, so a
    configuration that differs only in parameters the reduction never reads gives exactly
    this, and one that differs in anything it reads does not."""
    ia, ib = getattr(a, "image", None), getattr(b, "image", None)
    if ia is None or ib is None or np.shape(ia) != np.shape(ib):
        return False
    if list(getattr(a, "partitions", None) or []) != list(getattr(b, "partitions", None) or []):
        return False
    if not np.array_equal(np.asarray(ia), np.asarray(ib), equal_nan=True):
        return False
    sa, sb = getattr(a, "stack", None), getattr(b, "stack", None)
    if sa is not None and sb is not None:
        return np.shape(sa) == np.shape(sb) and bool(np.array_equal(np.asarray(sa), np.asarray(sb), equal_nan=True))
    return True


def _dropped(d: Optional[Dict[str, Any]]) -> Dict[str, List[int]]:
    """A ``dropped_frames`` record, ``{"n4": [551], ...}``, with each partition's indices sorted
    and unique, partitions with none left out, in natural order (n4 before n10)."""
    out = {str(k): sorted({int(i) for i in (v or [])}) for k, v in dict(d or {}).items()}
    return {k: out[k] for k in sorted(out, key=lambda x: (len(x), x)) if out[k]}


def _dropped_text(d: Dict[str, List[int]]) -> str:
    """``{"n4": [551]}`` as ``n4 frame 551 dropped``."""
    d = _dropped(d)
    if not d:
        return "no frames dropped"
    return ", ".join(f"{k} frame{'s' if len(v) > 1 else ''} {' '.join(str(i) for i in v)}"
                     for k, v in d.items()) + " dropped"


def _core_edits(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The part of a ``data_edits`` record that says what the frames are, for comparison:
    the cube name, the frames dropped and the sources subtracted and added (a file name is
    only where they came from)."""
    d = dict(d or {})
    out: Dict[str, Any] = {}
    if d.get("cube_name"):
        out["cube_name"] = str(d["cube_name"])
    drop = _dropped(d.get("dropped_frames"))
    if drop:
        out["dropped_frames"] = drop
    for key in ("subtracted_sources", "injected_sources"):
        src = [[float(v) for v in s] for s in (d.get(key) or [])]
        if src:
            out[key] = src
    return out


def _describe_edits(d: Dict[str, Any]) -> str:
    if not d:
        return "the default frames"
    bits = []
    if d.get("cube_name"):
        bits.append(f"cube {d['cube_name']}")
    if d.get("dropped_frames"):
        bits.append(_dropped_text(d["dropped_frames"]))
    for key, what in (("subtracted_sources", "subtracted"), ("injected_sources", "added")):
        src = d.get(key) or []
        if src:
            c = [s[2] for s in src]
            bits.append(f"{len(src)} {what} source(s), contrast {min(c):.3g} to {max(c):.3g}")
    return ", ".join(bits)


class Runner:
    """Drive the optimizer against a reducer.

    Parameters
    ----------
    reducer
        any :class:`~klip_tpe.reducer.Reducer`; a :class:`PartitionedReducer` for
        multi-partition data.
    space
        the :class:`~klip_tpe.space.SearchSpace` (its ``project`` hook is applied
        to every proposal).  With ``config.opt_width`` it must carry an integer
        ``width`` dim (``Param('width', w0, w1, 'int')``).
    objective
        :class:`~klip_tpe.metrics.Objective` (search metric + clean-subtraction rule).
    sampler
        :class:`~klip_tpe.positions.PositionSampler`.
    config
        :class:`RunConfig`.
    run_dir
        output directory (created).  Use :meth:`Runner.resume` to continue a run and
        :meth:`Runner.extend` to reopen one with a bigger budget.
    """

    CKPT_VERSION = 2

    def __init__(self, reducer: Reducer, space: SearchSpace, objective: Objective,
                 sampler: PositionSampler, config: RunConfig, run_dir: str,
                 throughput_fn: Optional[Callable] = None, log: Callable[[str], None] = print,
                 callbacks: Sequence[RunCallback] = (), resume: str = "auto",
                 _from_checkpoint: bool = False):
        self.reducer, self.space, self.objective = reducer, space, objective
        #: "auto" -- continue a run already check-pointed in ``run_dir`` (every run is
        #: resumable, however it was built: a script that constructs a Runner directly gets
        #: the same recovery as the command line).  "never" starts a fresh search over it.
        self.resume_mode = str(resume or "auto").lower()
        self.sampler, self.cfg, self.run_dir = sampler, config, run_dir
        self.throughput_fn = throughput_fn
        self.log = log
        self._apply_sampler_config()
        self.callbacks: List[RunCallback] = list(callbacks)
        self._last_images: Dict[str, Any] = {}
        os.makedirs(run_dir, exist_ok=True)
        if self.cfg.seed is None:
            self.cfg.seed = int(time.time()) % (2 ** 31)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.results: List[Optional[AnnulusResult]] = []
        self.ia = 0
        self.history: Optional[History] = None
        self.contrast = float(self.cfg.contrast0)
        self.records_count = 0
        self.wall0 = time.time()
        self.wall_prev = 0.0
        self._recal_done = 0
        self._best_images: Dict[str, Any] = {}
        self._nsrc_warned: Dict[int, int] = {}     # annulus -> count already reported
        self._nsrc_cache: Dict[int, int] = {}      # annulus -> resolved count (the probe is not free)
        self._draw_tally: Dict[int, Dict[str, Any]] = {}   # annulus -> search draws, failed draws, reasons
        self._calib_images: Optional[Dict[str, Any]] = None
        self._k_default = int(self.cfg.defaults.get("k_klip", 10) or 10)
        self._resumed = False
        self._extend_mode = False
        self._inflight = None              # trials in flight in a pipelined search (checkpointed)
        self._resume_pending = None        # ... and those a checkpoint had in flight
        self._vt_on_disk: Dict[Tuple[int, int], Optional[Dict[str, int]]] = {}   # validation trial records written
        self._ann_images: Dict[int, Dict[str, Any]] = {}       # per-annulus winner images (memory cache)
        self._val_dups: Dict[int, List[List[int]]] = {}         # per annulus: [eval, same_as_eval] skipped
        self._ann_histories: Dict[int, History] = {}           # completed annuli's histories
        self._pv_results: Dict[int, Dict[str, Any]] = {}       # param_verify outputs per annulus
        self._hooks_done: Dict[int, List[str]] = {}             # post-annulus hooks completed (checkpointed)
        self._last_final = False
        self._ckpt_written = 0.0                                # when checkpoint.json was last written
        self._ckpt_pending: Optional[Dict[str, Any]] = None     # a routine checkpoint held back (see checkpoint)
        self._ckpt_main_ok: Optional[bool] = None               # checkpoint.json worth keeping as .prev? (None: look)
        self._unsynced: set = set()                             # files the next checkpoint write fsyncs first
        self._extends = 0                                       # Runner.extend calls behind this state
        #: proof of life, stamped from its own thread: the run's own files advance once per
        #: completed evaluation, which on a loaded machine cannot tell a slow reduction from
        #: a dead process.  See :mod:`klip_tpe.heartbeat`.
        self.hb = Heartbeat(run_dir, log=self.log)
        if self.cfg.opt_width and not _from_checkpoint and len(self.cfg.ann_edges) > 1:
            self.log(f"opt_width: keeping the first inner edge only ({self.cfg.ann_edges[0]} px)")
            self.cfg.ann_edges = [float(self.cfg.ann_edges[0])]
        if self.cfg.opt_width and "width" not in self.space.bases:
            raise ValueError("opt_width needs a searched 'width' dim in the space "
                             "(SearchSpace.add(Param('width', w0, w1, 'int')))")
        if self.cfg.partition_weighting and isinstance(self.reducer, PartitionedReducer):
            self.reducer.weighting = self.cfg.partition_weighting

    def _emit(self, event: str, *args) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is None:
                continue
            try:
                fn(self, *args)
            except Exception as exc:      # never let a display problem kill a run
                self.log(f"  callback {type(cb).__name__}.{event} failed: {exc!r}")

    def _on_draw(self, ia: int, j: int, n: int, rec: "EvalRecord", inj, clean) -> None:
        """One remeasurement finished: log it and offer it to the display."""
        s = (f"failed ({rec.meta.get('why') or 'no reason given'})" if rec.score is None
             else f"{rec.score:.3f}")
        self.log(f"[ann {ia+1}]   draw {j+1}/{n} score={s}  ({rec.wall_s:.0f}s)")
        self._emit("on_draw", j, n, rec, inj, clean)

    def _guard(self, what: str, fn: Callable, *args, **kw):
        """Run an optional stage; any exception is logged and swallowed."""
        try:
            return fn(*args, **kw)
        except Exception as exc:
            self.log(f"  {what} skipped: {exc!r}")
            return None

    # ------------------------------------------------------------------ setup
    @property
    def fwhm(self) -> float:
        return float(self.reducer.fwhm)

    @property
    def pxscale(self) -> float:
        return float(self.reducer.pxscale)

    @property
    def has_partitions(self) -> bool:
        return bool(self.space.partitions)

    @property
    def best_images(self) -> Dict[str, Any]:
        """Images of the current annulus' running search best: ``inj`` / ``clean``
        arrays, ``inj_images`` / ``clean_images`` (:class:`EvalImages`), ``record``,
        ``zone``, ``sources`` and, with ``fm_preview``, ``fm`` = ``{'fm_image',
        'curve', 'sources', 'clean'}`` (A §9.3).  Empty before the first evaluation."""
        return self._best_images

    @property
    def do_fm(self) -> bool:
        """KLIP-FM passes are possible (reducer support).  In the scan k-modes the FM pass
        runs at the k the record / winner actually used (never as a k-scan cube)."""
        return bool(getattr(self.reducer, "supports_fm", False))

    @property
    def do_param_verify(self) -> bool:
        pv = self.cfg.param_verify
        return self.has_partitions if pv is None else bool(pv)

    def _annulus(self, ia: int) -> Tuple[float, float]:
        """Nominal zone of annulus ``ia`` (px).  With ``opt_width`` the outer edge is the
        committed one when known, else the width-range midpoint default."""
        e = self.cfg.ann_edges
        if self.cfg.opt_width:
            a_in = float(e[min(ia, len(e) - 1)]) if e else 0.0
            if ia + 1 < len(e):
                return a_in, float(e[ia + 1])
            w0, w1 = self.cfg.width_range
            return a_in, float(min(a_in + round(0.5 * (w0 + w1)), self.cfg.r_cap))
        if len(e) >= 2:
            return float(e[ia]), float(e[ia + 1])
        return 0.0, float(e[0]) if e else 70.0

    def _zone(self, ia: int, cfg: Optional[Config] = None) -> Tuple[float, float]:
        """Zone actually reduced for ``cfg`` (opt_width: ``outrad = min(inrad + width, r_cap)``)."""
        a_in, a_out = self._annulus(ia)
        if self.cfg.opt_width and cfg is not None and cfg.params.get("width") is not None:
            a_out = float(min(a_in + float(cfg.params["width"]), self.cfg.r_cap))
        return a_in, a_out

    def _pad_edges(self, a_in: float, a_out: float) -> Tuple[bool, bool]:
        """Whether a zone's inner and outer edges are padded (:attr:`RunConfig.edge_padding`).

        ``"between"`` pads only an edge another annulus shares.  The run's innermost edge is
        the first annulus' inner edge.  Its outermost edge is the last annulus' outer edge,
        or with ``opt_width`` an outer edge that leaves no room for another annulus (at
        ``r_cap``, or closer to it than the smallest width, where the run stops)."""
        if str(self.cfg.edge_padding).lower() == "all":
            return True, True
        e = [float(v) for v in (self.cfg.ann_edges or [])]
        if self.cfg.opt_width:
            lo = e[0] if e else 0.0
            w_lo = float(self.cfg.width_range[0]) if self.cfg.width_range else 0.0
            outer_end = a_out >= self.cfg.r_cap - 0.5 or self.cfg.r_cap - a_out < w_lo
        else:
            lo, hi = (e[0], e[-1]) if len(e) >= 2 else (0.0, e[0] if e else 70.0)
            outer_end = a_out >= hi - 1e-6
        return a_in > lo + 1e-6, not outer_end

    def _tile_pads(self, zones: Sequence[Tuple[float, float]], pad: float = 2.0) -> List[Tuple[float, float]]:
        """``(pad_in, pad_out)`` in px for each stitch tile: ``pad`` on an edge between
        annuli, 0 at the run's innermost and outermost edges (:meth:`_pad_edges`)."""
        out = []
        for a_in, a_out in zones:
            pin, pout = self._pad_edges(float(a_in), float(a_out))
            out.append((pad if pin else 0.0, pad if pout else 0.0))
        return out

    def _band(self, ia: int, cfg: Optional[Config] = None) -> Tuple[float, float]:
        """Injection band (arcsec), inset by ``inject_inset_fwhm`` FWHM from each edge
        (opt_width: both bounds at the zone mid-radius, A §3.1)."""
        return self._band_for(ia, self._nsrc(ia), cfg)

    def _band_for(self, ia: int, n: int, cfg: Optional[Config] = None) -> Tuple[float, float]:
        """:meth:`_band` for an explicit source count, so the packing check can ask what a
        candidate ``n`` would place without calling back into :meth:`_nsrc`."""
        a_in, a_out = self._zone(ia, cfg)
        if self.cfg.inject_rho is not None:
            r = float(self.cfg.inject_rho)
            if not a_in * self.pxscale <= r <= a_out * self.pxscale:
                raise ValueError(f"RunConfig.inject_rho {r:.3f}\" lies outside annulus {ia + 1}, "
                                 f"{a_in * self.pxscale:.3f}-{a_out * self.pxscale:.3f}\"")
            return r, r
        if self.cfg.opt_width:
            rmid = 0.5 * (a_in + a_out)
            return rmid * self.pxscale, rmid * self.pxscale
        if self.cfg.pair_area_midpoint and int(n) == 2:
            # addendum 2 §2b: a pair sits at the annulus' area-weighted mid radius (half the
            # annulus area inside, half outside), 180 deg apart; the band collapses to that
            # radius BEFORE the placement routine, whose 1.5-FWHM IWA clamp still applies.
            r_area = float(np.sqrt(0.5 * (a_in ** 2 + a_out ** 2)))
            return r_area * self.pxscale, r_area * self.pxscale
        rlo = a_in + self.cfg.inject_inset_fwhm * self.fwhm
        rhi = a_out - self.cfg.inject_inset_fwhm * self.fwhm
        if rhi <= rlo:
            rlo = rhi = 0.5 * (a_in + a_out)
        return rlo * self.pxscale, rhi * self.pxscale

    def _known(self) -> List[Tuple[float, float]]:
        """Real companions the objective was told about [(rho_as, pa_deg), ...]; they are
        excluded from the noise rings of the contrast curve."""
        try:
            return [tuple(k) for k in getattr(self.objective.metric, "known", ()) or ()]
        except Exception:
            return []

    def _excl_fwhm(self) -> float:
        """The metric's keep-out around a source (FWHM), which the verify's noise rings use too."""
        return float(getattr(getattr(self.objective, "metric", None), "excl_fwhm", 1.5) or 1.5)

    def _nsrc_asked(self, ia: int) -> Optional[int]:
        """The count ``RunConfig.n_sources`` asks for in annulus ``ia``, or None for the rule."""
        ns = self.cfg.n_sources
        if isinstance(ns, (list, tuple, np.ndarray)):
            # per annulus, the last entry repeating (like n_iter / n_init)
            ns = ns[min(ia, len(ns) - 1)] if len(ns) else None
        # 0 or None hands the annulus back to the rule -- in a list, as documented, and as a
        # single value too.  A 0 used to be taken literally: the annulus asked for no sources,
        # the ring-survival loop below had nothing to try, and none were injected.
        return int(ns) if ns else None

    def _nsrc(self, ia: int) -> int:
        ns = self._nsrc_asked(ia)
        if ia in self._nsrc_cache:
            return self._nsrc_cache[ia]
        a_in, a_out = self._zone(ia, None)
        asked = int(ns) if ns is not None else n_sources_rule(ia, a_out * self.pxscale, None)
        n = asked
        for cand in range(asked, 0, -1):
            if self._ring_survives(ia, cand):
                n = cand
                break
        mr = self._min_ring()
        if n < asked and self._nsrc_warned.get(ia) != n:
            self._nsrc_warned[ia] = n
            self.log(f"  annulus {ia+1}: {asked} sources requested, {n} used -- more would leave "
                     f"some separation's Mawet ring fewer than {mr} clean apertures, and it would "
                     f"fall back to the radial band there")
        # The count used can still starve a ring: one source close to the star, whose ring
        # holds only a handful of apertures, is never capped.  Asked at the sources themselves
        # and over more azimuths than the cap asks: where a source falls on the pixel grid
        # moves its count by one.
        got = self._ring_clean(ia, n, ndraw=12, at_sources=True)
        if got is not None and got[0] < mr:
            few, rho, most = got
            who = "one source" if n == 1 else f"with {n} sources, the one"
            keeps = f"{few}" if most == few else f"{few} to {most}"
            self.log(f"  annulus {ia+1}: {who} at {rho:.3f}\" keeps {keeps} clean aperture"
                     f"{'s' * (most != 1)} in its Mawet ring, fewer than {mr}"
                     + (", so its S/N comes from the radial band instead" if most < mr else
                        f" at some azimuths, and there its S/N comes from the radial band instead"))
        self._nsrc_cache[ia] = n
        return n

    def _apply_sampler_config(self) -> None:
        """``RunConfig.min_r_fwhm`` onto the run's sampler (the base of a replay), so the run
        injects with the floor it was started with, whatever sampler the caller built.  Then
        the metric's inner guard is fitted to that floor (:meth:`_fit_inner_guard`)."""
        v = getattr(self.cfg, "min_r_fwhm", None)
        smp = getattr(self.sampler, "base", None) or self.sampler
        if v is not None and hasattr(smp, "min_r_fwhm"):
            smp.min_r_fwhm = float(v)
        self._fit_inner_guard(smp)

    def _fit_inner_guard(self, smp) -> None:
        """Lower the metric's inner guard (``inner_fwhm``, no S/N inside it) to the injection
        floor when the floor is closer in, so every source the sampler places is scored.

        A guard outside the floor fails every draw between the two, in every stage.  The
        guard is never raised: inside the floor nothing is injected, and other measurements
        (a candidate's S/N, the known companions') keep the metric's own guard."""
        m = getattr(self.objective, "metric", None)
        g = getattr(m, "inner_fwhm", None)
        try:
            # the floor in the metric's own FWHM, which need not be the sampler's
            floor = float(smp.min_r_fwhm) * float(smp.fwhm_as) / (float(m.pxscale) * float(m.fwhm))
            g = float(g)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return
        if not np.isfinite(floor) or floor >= g * (1.0 - 1e-9):
            return
        m.inner_fwhm = max(floor, 0.0)
        self.log(f"  injection floor {float(smp.min_r_fwhm):g} FWHM: the metric scores sources from "
                 f"{m.inner_fwhm:.3f} FWHM (its inner guard was {g:g})"
                 + (f".  Inside {5 / (2 * np.pi):.3f} FWHM a ring holds fewer than 5 apertures and "
                    f"a source there still gets no S/N" if m.inner_fwhm < 5 / (2 * np.pi) else ""))

    def _ring_survives(self, ia: int, n: int) -> bool:
        """Would ``n`` sources leave every one of their rings ``min_ring`` clean apertures?

        Asked of the REAL geometry -- the sampler's own placement, scored by
        :func:`mawet_peak_snr` itself -- rather than a closed form.  A closed form has to
        assume the sources share one ring, and the default sampler does not place them that
        way: ``rho[i]`` steps across the injection band, so sources sit at different radii
        and mostly stay out of each other's exclusion zones.  Assuming co-radial sources
        understates what a wide annulus holds by a wide margin -- on MIRI's 6.7-36 px band
        it says 2 where the measured answer is past 14.

        The co-radial cases are the ones where it really does bite (``fixed_pa``,
        ``opt_width``, and the ``pair_area_midpoint`` pair), and they are covered here too
        because ``_band_for`` collapses the band for them.
        """
        if n <= 1:
            return True
        got = self._ring_clean(ia, n)
        return got is None or got[0] >= self._min_ring()    # a guard that cannot measure must not veto

    def _min_ring(self) -> int:
        return int(getattr(getattr(self.objective, "metric", None), "min_ring", 6) or 6)

    def _inner_fwhm(self) -> float:
        """The metric's inner guard (FWHM), for the places that call ``mawet_peak_snr`` directly."""
        g = getattr(getattr(self.objective, "metric", None), "inner_fwhm", None)
        return 1.0 if g is None else float(g)

    def _ring_clean(self, ia: int, n: int, ndraw: int = 2,
                    at_sources: bool = False) -> Optional[Tuple[int, float, int]]:
        """The fewest clean noise apertures any of ``n`` sources keeps in its Mawet ring, over
        ``ndraw`` probe draws of the run's own sampler, that source's separation (arcsec), and
        the most any draw's worst source keeps.  None when it cannot be measured.  The first
        two draws are the ones the source-count cap asks (:meth:`_ring_survives`).

        The cap probes a blank frame, on which the metric takes the first pixel of its search
        window as each source's peak, one up and one to the left of it, and lays the ring
        through that.  ``at_sources`` puts a spike on each source instead, so the metric finds
        it where it was put, as it finds an injected source.  Close to the star the two
        differ: on NEAR at 0.312" the blank probe keeps 0 apertures where the source keeps 3
        or 4."""
        m = getattr(self.objective, "metric", None)
        mr = self._min_ring()
        try:
            lo, hi = self._band_for(ia, n, None)
            shape = self._probe_shape()
            if shape is None:
                return None
            conv = getattr(self.reducer, "angle_convention", "pa")
            cx, cy = star_center(shape)
            probe = np.zeros(shape, float)
            worst, most = None, None
            # the geometry's own sampler: a replay of recorded draws must not spend them here
            smp = getattr(self.sampler, "base", None) or self.sampler
            seeds = [0x5E3D, 0xC0FFEE] + list(range(1, int(ndraw) - 1))
            for seed in seeds[:max(int(ndraw), 1)]:
                src = smp.sample(n, lo, hi, np.random.default_rng([int(self.cfg.seed or 0),
                                                                   int(ia), int(n), seed]))
                kw = {}
                if at_sources:
                    probe = np.zeros(shape, float)
                    xs, ys = source_xy([s.rho for s in src], [s.theta for s in src], self.pxscale, cx, cy, conv)
                    for x, y in zip(xs, ys):
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
                            probe[iy, ix] = 1.0
                    kw["declip_nsig"] = None          # the spikes must stay
                _, det = mawet_peak_snr(probe, [s.rho for s in src], [s.theta for s in src],
                                        self.pxscale, self.fwhm, known=self._known(),
                                        excl_fwhm=float(getattr(m, "excl_fwhm", 1.5) or 1.5),
                                        pixel_mask=getattr(m, "pixel_mask", None),
                                        inner_fwhm=self._inner_fwhm(),
                                        angle_convention=conv, min_ring=mr, return_details=True, **kw)
                fewest = None
                for s, d in zip(src, det):
                    c = int(d.get("nclean", 0))
                    if worst is None or c < worst[0]:
                        worst = (c, float(s.rho))
                    fewest = c if fewest is None else min(fewest, c)
                if fewest is not None:
                    most = fewest if most is None else max(most, fewest)
            return None if worst is None else (worst[0], worst[1], most)
        except Exception:
            return None

    def _probe_shape(self) -> Optional[Tuple[int, int]]:
        """Frame shape for the packing probe, from whichever reducer holds data."""
        for r in [self.reducer] + list(getattr(self.reducer, "reducers", {}).values() or []):
            c = getattr(getattr(r, "data", None), "cube", None)
            if c is not None and np.ndim(c) == 3:
                return tuple(int(v) for v in np.shape(c)[1:])
        pm = getattr(getattr(self.objective, "metric", None), "pixel_mask", None)
        return None if pm is None else tuple(int(v) for v in np.shape(pm))

    def search_sources(self, ia: int, contrast: float) -> List[Source]:
        """One frozen set of injected sources for annulus ``ia``.

        Only used when ``RunConfig.fixed_sources`` is on, which is NOT the default and is a
        documented departure from ``optimize_near_2_tpe`` -- see that field.  The reference
        re-draws the azimuths every evaluation on purpose; this is here for the experiment of
        pairing every comparison on one speckle realisation, not for production.

        Drawn from a generator seeded only by the run seed and the annulus index -- never
        from ``self.rng``, whose state advances with the search.  That independence is what
        makes them reproducible: a resumed run recovers the same positions, so the
        evaluations either side of the restart stay comparable.
        """
        cached = getattr(self, "_fixed_src", None)
        if cached is None or cached[0] != int(ia):
            rng = np.random.default_rng([int(self.cfg.seed or 0), int(ia), 0x5E3D])
            rlo, rhi = self._band(ia, None)          # the annulus' own band, not a config's
            self._fixed_src = (int(ia), self.sampler.sample(self._nsrc(ia), rlo, rhi, rng, contrast))
        return [Source(s.rho, s.theta, float(contrast)) for s in self._fixed_src[1]]

    def _make_optimizer(self, ia: int) -> Optimizer:
        c = self.cfg
        has_parts = bool(self.space.partitions)
        pbest = float(c.pbest or 0.0)
        wtie = has_parts if c.warmstart_tie is None else c.warmstart_tie
        n_init = c.per_annulus(c.n_init, ia)
        if c.search_mode == "tpe":
            return TPE(self.space, n_init=n_init, gamma=c.gamma, ncand=c.ncand, prior_weight=c.prior_weight,
                       bw_floor=c.bw_floor, pbest=pbest, p_local=c.p_local, n_elite=c.n_elite,
                       explore_frac=c.explore_frac, blocks=c.blocks, warmstart_tie=wtie, link_params=c.link_params)
        if c.search_mode == "random":
            return RandomSearch(self.space, n_init=0, warmstart_tie=wtie, link_params=c.link_params)
        if c.search_mode == "grid":
            return GridSearch(self.space, budget=c.per_annulus(c.n_iter, ia) - 1, axes=c.grid_axes,
                              base_vector=self._default_vector(), link_params=c.link_params)
        raise ValueError(c.search_mode)

    def _default_vector(self, k: Optional[int] = None) -> np.ndarray:
        ov = dict(self.cfg.defaults)
        if k is not None:
            ov["k_klip"] = k
        return self.space.default_vector(ov)

    def _zone_overrides(self, ia: int, cfg: Optional[Config] = None) -> Dict[str, Any]:
        a_in, a_out = self._zone(ia, cfg)
        return {"inrad": a_in, "outrad": a_out}

    def _config_from_dict(self, d: Dict[str, Any]) -> Config:
        """Rebuild a :class:`Config` from its ``to_dict`` form (partition ids matched by str)."""
        per_s = d.get("per_partition", {})
        per = {}
        for pid in self.space.partitions:
            if str(pid) in per_s:
                per[pid] = dict(per_s[str(pid)])
        sel_s = [str(s) for s in d.get("selected", [])]
        sel = [pid for pid in self.space.partitions if str(pid) in sel_s] or list(self.space.partitions)
        x = np.asarray(d.get("x", self.space.default_vector()), float)
        return Config(dict(d.get("params", {})), per, sel, x)

    # -------------------------------------------------------------- reduction
    def _known_negatives(self) -> List[Source]:
        """``RunConfig.subtract_known`` as negative injections."""
        return [Source(float(r), float(p), -abs(float(c))) for r, p, c in (self.cfg.subtract_known or [])]

    def _reduce(self, cfg: Config, sources: Optional[Sequence[Source]], k_scan: bool = False,
                tag: str = "", zone: Optional[Tuple[float, float]] = None,
                fm_sources: Optional[Sequence[Source]] = None, extras: bool = False) -> EvalImages:
        """One reduction of ``cfg`` at the current annulus' zone.  ``fm_sources`` requests
        the KLIP-FM pass (``EvalImages.fm_image``), ``extras`` the nosub / cadi reference
        images; both are forwarded to partitioned and plain reducers alike."""
        ia_over = self._zone_overrides(self.ia, cfg) if zone is None else {"inrad": zone[0], "outrad": zone[1]}
        # zone geometry is a run property, not a searched parameter (unless searched)
        params = dict(cfg.params)
        params.pop("width", None)
        for k, v in ia_over.items():
            if k not in self.space.bases or zone is not None:
                params[k] = v
        per = {}
        for pid, d in cfg.per_partition.items():
            dd = dict(d)
            dd.pop("width", None)
            dd.update({k: v for k, v in ia_over.items() if k not in self.space.bases or zone is not None})
            per[pid] = dd
        # which zone edges the reducer pads: by default only those shared with another annulus
        z_in = float(params.get("inrad", ia_over["inrad"]))
        z_out = float(params.get("outrad", ia_over["outrad"]))
        pad_in, pad_out = self._pad_edges(z_in, z_out)
        params.update(pad_in=pad_in, pad_out=pad_out)
        for dd in per.values():
            dd.update(pad_in=pad_in, pad_out=pad_out)
        cfg2 = Config(params, per, cfg.selected, cfg.x)
        if (fm_sources is None and sources and not k_scan and self.reducer.supports_fm
                and getattr(self.objective.metric, "needs_fm", False)):
            # a forward-modelled matched filter needs this configuration's KLIP-FM response
            # of the very sources it is about to score (KLIP-FM is undefined under k_scan;
            # a reducer without it falls back to the numerical FM, see Runner._fm_for)
            fm_sources = sources
        # known sources taken out of the frames (RunConfig.subtract_known): appended here, the
        # one place every reduction of the run passes through, and never seen by a caller --
        # each scores the list it passed in
        neg = self._known_negatives()
        if neg:
            sources = list(sources or []) + neg
        if isinstance(self.reducer, PartitionedReducer):
            return self.reducer.reduce_config(cfg2, sources, k_scan, tag, fm_sources=fm_sources, extras=extras)
        res = self.reducer.reduce(ReductionRequest(cfg2.params, sources, k_scan, tag, fm_sources=fm_sources,
                                                   extras=extras))
        return EvalImages(res.image, None, None, None, res.meta, fm_image=getattr(res, "fm_image", None),
                          nosub=getattr(res, "nosub", None), cadi=getattr(res, "cadi", None))

    # -- KLIP-FM (A §9.1 / §9.3) -------------------------------------------------
    def _fm_pass(self, cfg: Config, zone: Tuple[float, float], contrast: float, tag: str,
                 sources: Optional[Sequence[Source]] = None,
                 clean_img: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """One CLEAN reduction of ``cfg`` with the KLIP-FM model cube of ``sources``
        (default: the golden-angle test spiral of ``products.fm_test_sources``) and the
        FM 5-sigma curve against ``clean_img`` (default: the pass' own clean image).
        Returns ``{'images': EvalImages, 'fm_image', 'clean', 'sources', 'curve'}`` with
        ``curve`` a JSON-safe dict of lists, or None when the reducer produced no FM image."""
        from .products import fm_contrast_curve, fm_test_sources
        a_in, a_out = zone
        rin = max(float(a_in), self.fwhm)
        srcs = list(sources) if sources else fm_test_sources(rin, float(a_out), self.pxscale, contrast)
        srcs = [Source(s.rho, s.theta, contrast) for s in srcs]
        ev = self._reduce(cfg, None, fm_sources=srcs, tag=tag)
        if ev.fm_image is None:
            return None
        clean = ev.image if clean_img is None else clean_img
        cc = fm_contrast_curve(ev.fm_image, clean, contrast, self.fwhm, self.pxscale, rin, float(a_out), srcs,
                               kernel_fn=self.reducer.matched_filter_kernel,
                               angle_convention=self.reducer.angle_convention,
                               flatten=self.flatten_products, known=self._known())
        curve = {k: [None if not np.isfinite(v) else float(v) for v in np.atleast_1d(np.asarray(v_, float))]
                 for k, v_ in cc.items()}
        return {"images": ev, "fm_image": ev.fm_image, "clean": clean, "sources": srcs, "curve": curve}

    @staticmethod
    def _fm_section(curve: Optional[Dict[str, Any]], ia: int, contrast: float) -> List[str]:
        """The ``# KLIP-FM`` block of ``contrast_curve.txt`` (A §6.4): two comment lines
        then ``sep_arcsec  contrast_5sig`` rows (empty list when the curve has no points)."""
        if not curve:
            return []
        rows = [(r, c) for r, c in zip(curve.get("r_as") or [], curve.get("curve") or [])
                if r is not None and c is not None and np.isfinite(r) and np.isfinite(c) and c > 0]
        if not rows:
            return []
        L = [f"# KLIP-FM cross-check 5-sigma contrast  (annulus {ia+1}, clean forward-model pass at the winner "
             f"config, {len(rows)} test sources on the golden-angle spiral, contrast_inj={contrast:.3e})",
             "# sep_arcsec   contrast_5sig"]
        L += [f"{r:10.4f}   {c:12.4e}" for r, c in rows]
        return L

    # -- injection-recovery contrast curve -----------------------------------------
    def _rotation_deg(self, cfg: Config) -> Optional[float]:
        """The field rotation the configuration's references can span (deg): the largest PA
        range of the selected partitions' frames, capped by the largest ``anglemax`` among them.
        None when the reducer does not give its frames' angles."""
        pids = list(cfg.selected) if cfg.selected else [None]
        spans = []
        for pid in pids:
            try:
                ang = (self.reducer.frame_angles(pid) if isinstance(self.reducer, PartitionedReducer)
                       else self.reducer.frame_angles())
            except Exception:                 # noqa: BLE001
                ang = None
            if ang is not None and np.size(ang) > 1:
                spans.append(float(np.ptp(np.asarray(ang, float))))
        if not spans:
            return None
        rot = max(spans)
        am = []
        for pid in pids:
            v = (cfg.params_for(pid) if pid is not None else cfg.params).get("anglemax")
            if v is not None and np.isfinite(float(v)):
                am.append(float(v))
        return min(rot, max(am)) if am else rot

    def _contrast_grid(self, ia: int, cfg: Config, zone: Tuple[float, float],
                       fm_curve: Optional[Dict[str, Any]], curve: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Annulus ``ia``'s injection-recovery contrast curve at the winner config ``cfg``
        (:mod:`klip_tpe.contrast_grid`): one injected source per reduction, each test position
        iterated on its contrast to the target S/N, the noise rings without the injected source,
        the known companions and their sidelobes.  The first guesses come from the KLIP-FM curve,
        else the validated one.  Writes ``contrast_grid.json`` (every position and trial),
        ``contrast_grid.txt`` (the curve) and a section of ``contrast_curve.txt``, and returns the
        curve (:func:`klip_tpe.contrast_grid.summarize`).  The positions of ``mode="random"`` come
        from :meth:`_side_rng`, so the run's generator stands where it did."""
        from . import contrast_grid as CG
        from .metrics import aperture_mawet_snr
        gc = self.cfg.contrast_grid
        if not gc.enabled:
            return None
        a_in, a_out = float(zone[0]), float(zone[1])
        metric = getattr(self.objective, "metric", None)

        def mget(k, dflt):
            v = getattr(metric, k, dflt) if metric is not None else dflt
            return dflt if v is None and dflt is not None else v
        fwhm, pxscale = float(mget("fwhm", self.fwhm)), float(mget("pxscale", self.pxscale))
        inner = float(mget("inner_fwhm", 1.0)) * fwhm
        seps = CG.separations_px(a_in, a_out, fwhm, gc, floor_px=max(inner, fwhm))
        conv = mget("angle_convention", None) or getattr(self.reducer, "angle_convention", "pa")
        rot = self._rotation_deg(cfg)

        def hw(r_px: float) -> float:
            return CG.footprint_deg(r_px, fwhm, gc, rot)
        known = [(float(k[0]), float(k[1])) for k in self._known()]
        known_arcs = [(rk, tk, hw(rk / pxscale), gc.known_radial_fwhm) for rk, tk in known]
        if str(gc.mode).lower() == "random":
            positions = CG.random_positions(seps, gc.sep_step_fwhm * fwhm, gc.n_samples,
                                            self._side_rng("contrast_grid", ia))
        else:
            positions = CG.grid_positions(seps, gc.n_pa)
        for p in positions:
            for rk, tk in known:
                if CG.in_footprint(p.r_px, p.theta, rk / pxscale, tk, hw(rk / pxscale), gc.known_radial_fwhm * fwhm):
                    p.status, p.why = "excluded", "in a known companion's sidelobes"
                    break
        # first guesses: a 5-sigma curve already in hand, scaled to the target
        guide = None
        for cc in (fm_curve, curve):
            if cc and cc.get("r_as") and cc.get("curve"):
                r_ = np.array([np.nan if v is None else v for v in cc["r_as"]], float)
                c_ = np.array([np.nan if v is None else v for v in cc["curve"]], float)
                ok = np.isfinite(r_) & np.isfinite(c_) & (c_ > 0)
                if ok.any():
                    o = np.argsort(r_[ok])
                    guide = (r_[ok][o], np.log(c_[ok][o]))
                    break
        scale = float(gc.target_snr) / 5.0

        def start(p) -> float:
            if guide is None:
                return float(self.contrast) * scale
            return float(np.exp(np.interp(p.r_px * pxscale, guide[0], guide[1]))) * scale
        window = self._reduction_window(len(cfg.selected))

        def reduce_many(batch):
            calls = [lambda p=p, c=c: self._reduce(cfg, [Source(p.r_px * pxscale, p.theta, c)],
                                                   tag=f"a{ia+1}_cg{p.index}_{len(p.trials)}") for p, c in batch]
            with contextlib.closing(self._run_ordered(calls, window)) as stream:
                for _, res in stream:
                    yield res
        min_ring = int(mget("min_ring", 6))
        aperture = str(gc.snr).lower() == "aperture"

        def measure(res, p, c):
            img = np.asarray(res.image, float)
            rho = p.r_px * pxscale
            arcs = [(rho, p.theta, hw(p.r_px), max(gc.known_radial_fwhm, 1.0))] + known_arcs
            if aperture:
                snr, det = aperture_mawet_snr(img, [rho], [p.theta], pxscale, fwhm, excl_fwhm=float(mget("excl_fwhm", 1.5)),
                                              angle_convention=conv, known=known, excl_arcs=arcs, return_details=True)
            else:
                kern = metric.kernel(rho) if callable(getattr(metric, "kernel", None)) else None
                im = radprof(img) if mget("flatten", False) else img
                snr, det = mawet_peak_snr(im, [rho], [p.theta], pxscale, fwhm, kernel=kern,
                                          search_px=float(mget("search_px", 1.5)), excl_fwhm=float(mget("excl_fwhm", 1.5)),
                                          known=known, declip_nsig=mget("declip_nsig", 6.0),
                                          band_fwhm=float(mget("band_fwhm", 0.6)), min_ring=min_ring,
                                          penalty=str(mget("penalty", "geometric")), pixel_mask=mget("pixel_mask", None),
                                          inner_fwhm=float(mget("inner_fwhm", 1.0)), angle_convention=conv,
                                          return_details=True, excl_arcs=arcs)
            d = det[0]
            if d.get("why"):
                return np.nan, str(d["why"])
            if int(d.get("nclean", 0)) < min_ring:
                return np.nan, f"{int(d.get('nclean', 0))} noise apertures left after the exclusions (fewer than {min_ring})"
            return float(snr[0]), None
        t0 = time.time()
        n_test = sum(p.status != "excluded" for p in positions)
        self.log(f"  contrast grid: {n_test} position(s) on {seps.size} separation(s), one injection per reduction, "
                 f"S/N {gc.target_snr:g} +/- {100 * gc.precision:g} % ({'aperture' if aperture else 'matched-filter'} S/N, "
                 f"sidelobes +/- {hw(float(np.median(seps))):.1f} deg at {np.median(seps) * pxscale:.2f}\")")
        positions, n_red = CG.run_grid(
            positions, start, reduce_many, measure, gc, log=self.log,
            stage=lambda s_: self.hb.stage(f"annulus {ia + 1} contrast grid: {s_}", write=False, annulus=ia))
        out = CG.summarize(positions, seps, pxscale, gc, [hw(r) for r in seps], n_red)
        out["rotation_deg"] = rot
        d = self._ann_dir(ia)
        _atomic_write(os.path.join(d, "contrast_grid.json"),
                      json.dumps(dict(out, positions=[p.to_dict(pxscale) for p in positions]), indent=1,
                                 default=_json_default))
        n_ok = sum(p.status in CG.STATUS_DONE for p in positions)
        hdr = (f"annulus {ia + 1}  injection-recovery {gc.target_snr:g}-sigma contrast  (one injection per reduction, "
               f"{n_test} positions, {gc.n_pa if str(gc.mode).lower() != 'random' else gc.n_samples} "
               f"{'per separation' if str(gc.mode).lower() != 'random' else 'random'}, S/N {gc.target_snr:g} +/- "
               f"{100 * gc.precision:g} %, {'aperture' if aperture else 'matched-filter'} S/N, median over position angle)")
        rows = [(r, c, lo, hi, n_, np_) for r, c, lo, hi, n_, np_ in zip(out["r_as"], out["curve"], out["p16"],
                                                                          out["p84"], out["n"], out["n_pos"])]
        with open(os.path.join(d, "contrast_grid.txt"), "w") as f:
            f.write(f"# {hdr}\n# sep_arcsec   contrast_median   p16   p84   n_settled   n_tested\n")
            for r, c, lo, hi, n_, np_ in rows:
                f.write(f"{r:10.4f}   {np.nan if c is None else c:12.4e}   {np.nan if lo is None else lo:12.4e}   "
                        f"{np.nan if hi is None else hi:12.4e}   {n_:4d}   {np_:4d}\n")
        with open(os.path.join(d, "contrast_curve.txt"), "a") as f:
            f.write(f"# {hdr}\n# sep_arcsec   contrast_5sig\n")
            for r, c, *_ in rows:
                if c is not None:
                    f.write(f"{r:10.4f}   {c:12.4e}\n")
        self._durable_later(os.path.join(d, "contrast_grid.json"), os.path.join(d, "contrast_grid.txt"),
                            os.path.join(d, "contrast_curve.txt"))
        self.log(f"  contrast grid: {n_ok} of {n_test} position(s) settled in {n_red} reductions "
                 f"({time.time() - t0:.0f} s), {sum(out['n_blocked'])} blocked"
                 + (f", {out['n_excluded']} excluded for the known companions" if out["n_excluded"] else ""))
        return out if n_ok else None

    def _fmkw(self, fm) -> Dict[str, Any]:
        """``fm=`` is passed to the objective only for a metric that asked for the
        forward model, so every other objective keeps its original signature."""
        return {"fm": fm} if getattr(self.objective.metric, "needs_fm", False) else {}

    def _fm_for(self, inj: Optional[EvalImages], clean: Optional[EvalImages] = None) -> Dict[str, Any]:
        """The forward model to score this pair of reductions with.

        The analytic KLIP-FM image when the reducer produced one; otherwise the
        **numerical** forward model ``injected - clean``, which is the same quantity to
        first order (both reductions see the same data and the same configuration, so the
        speckle field cancels and what is left is the pipeline's response to the injected
        sources, self-subtraction and all).  That fallback is what gives a matched filter
        to the backends with no analytic FM -- this package's pyKLIP, VIP and spaceKLIP
        ones.  Both scoring calls of :meth:`Objective.score_search` get the *same* array, so
        the injected and clean terms are always filtered with identical kernels.
        """
        if not getattr(self.objective.metric, "needs_fm", False) or inj is None:
            return {}
        fm = getattr(inj, "fm_image", None)
        if fm is None and clean is not None and getattr(self.objective.metric, "fm_from_difference", True):
            a, b = np.asarray(inj.image, float), np.asarray(clean.image, float)
            if a.ndim == 2 and a.shape == b.shape:
                fm = a - b
        return {"fm": fm}

    def _score_stack(self, stack: Optional[np.ndarray], parts, sources,
                     fm_stack: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
        """Each partition's own raw S/N, ``score_raw`` on its image.  The search has not called
        it since 2026-10-25 (:attr:`EvalRecord.partition_snr`), and it stays for scripts."""
        out: Dict[str, Optional[float]] = {}
        if stack is None:
            return out
        for i, (pid, img) in enumerate(zip(parts, stack)):
            try:
                fm = None if fm_stack is None or i >= len(fm_stack) else fm_stack[i]
                r = self.objective.score_raw(img, sources, **self._fmkw(fm))
                out[str(pid)] = None if not np.isfinite(r.score) else float(r.score)
            except Exception:
                out[str(pid)] = None
        return out

    def _project(self, x: np.ndarray, is_random: bool) -> np.ndarray:
        x = self.space.sanitize(x)
        if self.space.project is not None:
            zone = self._zone(self.ia, self.space.decode(x)) if self.cfg.opt_width else self._annulus(self.ia)
            x = self.space.project(x, self.space, rng=self.rng, is_random=is_random, zone=zone)
            x = self.space.sanitize(x)
        return x

    # -- k handling ------------------------------------------------------------
    def _with_k(self, cfg: Config, k: int) -> Config:
        params = dict(cfg.params, k_klip=k)
        per = {p: dict(d, k_klip=k) for p, d in cfg.per_partition.items()}
        x = cfg.x.copy()
        for i in self.space.dims_of_base("k_klip"):
            x[i] = self.space.params[i].sanitize(k)
        return Config(params, per, cfg.selected, x)

    def _with_k_map(self, cfg: Config, kmap: Dict[Any, int]) -> Config:
        """Per-partition k (``kpn``); the global ``k_klip`` becomes the rounded median."""
        km = {str(p): int(v) for p, v in kmap.items() if v is not None}
        per = {}
        for p, d in cfg.per_partition.items():
            per[p] = dict(d, k_klip=km[str(p)]) if str(p) in km else dict(d)
        vals = [km[str(p)] for p in cfg.selected if str(p) in km] or list(km.values())
        params = dict(cfg.params, k_klip=int(max(round(float(np.median(vals))), 1)) if vals else cfg.params.get("k_klip"))
        x = cfg.x.copy()
        for i in self.space.dims_of_base("k_klip"):
            p = self.space.params[i]
            if p.partition is not None and str(p.partition) in km:
                x[i] = p.sanitize(km[str(p.partition)])
            elif p.partition is None and vals:
                x[i] = p.sanitize(params["k_klip"])
        return Config(params, per, cfg.selected, x)

    def _apply_k_flag(self, cfg: Config, k_hist) -> Config:
        """Re-apply the k recorded in a history flag (scan modes only)."""
        if self.cfg.k_mode not in SCAN_MODES:
            return cfg
        if isinstance(k_hist, dict):
            return self._with_k_map(cfg, k_hist)
        if isinstance(k_hist, (int, float)) and np.isfinite(k_hist) and k_hist >= 1:
            return self._with_k(cfg, int(k_hist))
        return cfg

    def _k_scan_max(self) -> int:
        kmax = int(self.cfg.k_scan_max)
        dims = self.space.dims_of_base("k_klip")
        if len(dims):
            kmax = min(kmax, int(max(self.space.params[i].hi for i in dims)))
        return max(kmax, 1)

    @staticmethod
    def _nanmean_stack(stack: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        stack = np.asarray(stack, float)
        w = np.ones(stack.shape[0]) if w is None else np.asarray(w, float)
        fin = np.isfinite(stack)
        num = np.tensordot(w, np.where(fin, stack, 0.0), axes=(0, 0))
        den = np.tensordot(w, fin.astype(float), axes=(0, 0))
        return num / np.where(den > 0, den, np.nan)

    def _pnkpick(self, stack: np.ndarray, weights: Optional[np.ndarray], sources: Sequence[Source],
                 cstack: Optional[np.ndarray] = None) -> Tuple[List[int], Dict[str, Any]]:
        """``near2m_pnkpick``: per-partition k over a ``(npart, nk, ny, nx)`` scan cube.
        Init = each partition's standalone argmax, then two passes of greedy coordinate
        ascent on the combined image's median raw metric.  Returns 1-based k per
        partition and the S/N-vs-k curves."""
        npart, nk = stack.shape[:2]
        w = np.ones(npart) if weights is None else np.asarray(weights, float)

        def med(img, cl=None):
            try:
                return float(self.objective.score_raw(img, sources, cl).score)
            except Exception:
                return np.nan

        snrk = np.full((npart, nk), np.nan)
        kk = []
        for j in range(npart):
            for k in range(nk):
                snrk[j, k] = med(stack[j, k], None if cstack is None else cstack[j, k])
            kk.append(int(np.nanargmax(snrk[j])) if np.any(np.isfinite(snrk[j])) else nk // 2)

        def comb(kidx, cube):
            return self._nanmean_stack(np.stack([cube[j, kidx[j]] for j in range(npart)]), w)

        for _ in range(2):
            for j in range(npart):
                bk, bs = kk[j], -np.inf
                for k in range(nk):
                    kk[j] = k
                    s = med(comb(kk, stack), None if cstack is None else comb(kk, cstack))
                    if np.isfinite(s) and s > bs:
                        bs, bk = s, k
                kk[j] = bk
        return [k + 1 for k in kk], {"snrk_pn": snrk.tolist()}

    def _kscan_pick(self, cfg: Config, sources: Sequence[Source], tag: str) -> Tuple[Config, Any, Dict[str, Any]]:
        """Per-eval k-scan (A §5.9 b/d): returns the config with the chosen k(s), the
        ``k_used`` value (int, or ``{pid: k}`` for ``k_mode='scan'`` on partitions) and
        diagnostics (``kbest``, ``kpn``, the scan curve)."""
        kmax = self._k_scan_max()
        cfgk = self._with_k(cfg, kmax)
        calls = [lambda: self._reduce(cfgk, sources, k_scan=True, tag=tag + "_scan")]
        if self.objective.metric.needs_clean:
            calls.append(lambda: self._reduce(cfgk, None, k_scan=True, tag=tag + "_cscan"))
        got = [r for _, r in self._run_ordered(calls, len(calls) if self._parallel_reducer() else 1)]
        for r in got:                                  # the first failure, in the old order
            if isinstance(r, BaseException):
                raise r
        scan, cscan = got[0], (got[1] if len(got) > 1 else None)
        nk = int(scan.image.shape[0])
        info: Dict[str, Any] = {}
        stack = scan.stack
        if (self.cfg.k_mode == "scan" and stack is not None and stack.ndim == 4 and stack.shape[0] >= 1
                and scan.partitions is not None):
            kk, extra = self._pnkpick(stack, scan.weights, sources, None if cscan is None else cscan.stack)
            kmap = {str(p): int(k) for p, k in zip(scan.partitions, kk)}
            cfg = self._with_k_map(cfg, kmap)
            k_used: Any = {str(p): kmap[str(p)] for p in cfg.selected if str(p) in kmap}
            info.update(extra, kpn=k_used, kbest=int(cfg.params.get("k_klip") or 1))
        else:
            sc = [self.objective.score_raw(scan.image[k], sources, None if cscan is None else cscan.image[k]).score
                  for k in range(nk)]
            kbest = int(np.nanargmax(sc)) + 1 if np.any(np.isfinite(sc)) else max(1, nk // 2)
            cfg = self._with_k(cfg, kbest)
            k_used = kbest
            info.update(kbest=kbest, snrk=[None if not np.isfinite(v) else float(v) for v in sc])
        return cfg, k_used, info

    # -- evaluation ------------------------------------------------------------
    def _draw_sources(self, cfg: Config, contrast: float) -> List[Source]:
        """The sources one evaluation of ``cfg`` injects when the caller supplies none.

        A fresh draw per evaluation, from the band this configuration defines (with
        opt_width that band is the configuration's own mid-radius, which is why freezing is
        meaningless there and stays off).  The non-default ``fixed_sources`` returns the
        annulus' frozen set instead, which does not touch ``self.rng``.
        """
        if self.cfg.fixed_sources and not self.cfg.opt_width:
            return self.search_sources(self.ia, contrast)
        rlo, rhi = self._band(self.ia, cfg)
        return self.sampler.sample(self._nsrc(self.ia), rlo, rhi, self.rng, contrast)

    def _kscan_active(self) -> bool:
        """Each evaluation picks its own k by a k-scan (``k_mode`` scan / scan_rescore)."""
        return bool(self.cfg.k_mode in SCAN_MODES and getattr(self.reducer, "supports_kscan", False))

    def _needs_clean(self, raw_only: bool = False) -> bool:
        return bool((self.objective.needs_clean and not raw_only) or self.objective.metric.needs_clean)

    # -- reductions in flight ----------------------------------------------------
    def _parallel_reducer(self) -> bool:
        """Whether the reducer can serve several reductions at once (a partitioned reducer
        with a worker budget of 2 or more)."""
        return getattr(self.reducer, "max_workers", 1) >= 2

    def _reduction_window(self, n_sel: Optional[int] = None) -> int:
        """How many independent reductions to keep in flight: enough partition jobs to fill
        the reducer's worker slots, plus one so the next is queued while a result is being
        scored.  1 on a serial reducer."""
        if not self._parallel_reducer():
            return 1
        slots = int(getattr(self.reducer, "job_slots", lambda: 2)() or 2)
        n = max(int(n_sel or len(getattr(self.reducer, "reducers", {}) or {}) or 1), 1)
        return max(2, -(-slots // n) + 1)

    def _memory_note(self) -> None:
        """The reducer's line on its worker processes' memory, when it has one (Linux):
        what a pool of this size needs, for choosing ``--workers`` / ``--threads-per-job``."""
        note = getattr(self.reducer, "memory_note", None)
        line = self._guard("worker memory", note) if callable(note) else None
        if line:
            self.log(f"  {line}")

    def _start_workers(self) -> None:
        """Start the reducer's worker processes, if it has any, from this thread.  A forked
        worker inherits every lock the parent's other threads held at the fork, held, so
        the pool is started before any reduction thread exists (and, from :meth:`run`,
        before the heartbeat thread).  Starting it again is a no-op."""
        start = getattr(self.reducer, "start_workers", None)
        if callable(start):
            self._guard("worker pool", start)

    def _run_ordered(self, calls: Sequence[Callable[[], Any]], window: int):
        """Run ``calls`` (no-argument callables, each one reduction) with at most ``window``
        in flight, and yield ``(i, result)`` in the order given.  A call that raises yields
        its exception as the result, so the caller decides what a failure means.  The
        reductions are pure functions of their arguments, so the schedule changes when
        they finish, never what they return.  Waiting services the live window."""
        calls = list(calls)
        if window <= 1 or len(calls) <= 1:
            for i, fn in enumerate(calls):
                try:
                    res = fn()
                except Exception as exc:          # noqa: BLE001 - handed to the caller
                    res = exc
                yield i, res
            return
        import concurrent.futures as _cf
        from .parallel import pump_wait
        self._start_workers()
        ex = _cf.ThreadPoolExecutor(max_workers=min(int(window), len(calls)), thread_name_prefix="klip-red")
        finished = False
        try:
            futs: Dict[int, Any] = {}
            nxt = 0
            for i in range(len(calls)):
                while nxt < len(calls) and nxt - i < window:
                    futs[nxt] = ex.submit(calls[nxt])
                    nxt += 1
                f = futs.pop(i)
                try:
                    res = pump_wait([f])[0]
                except Exception as exc:          # noqa: BLE001
                    res = exc
                yield i, res
            finished = True
        finally:
            # left early (an interrupt, a caller that has what it needs): drop what has not
            # started instead of waiting for it
            ex.shutdown(wait=finished, cancel_futures=not finished)

    def evaluate(self, x: np.ndarray, phase: str, contrast: Optional[float] = None,
                 sources: Optional[Sequence[Source]] = None, raw_only: bool = False,
                 tag: str = "eval", clean: Optional[EvalImages] = None
                 ) -> Tuple[EvalRecord, EvalImages, Optional[EvalImages]]:
        """Reduce + score one (already projected) vector.

        Sources are drawn fresh for this evaluation, as ``optimize_near_2_tpe`` does: the
        radial ladder is deterministic for the annulus, the azimuth anchor is not.  The
        caller may supply its own instead (calibration, validation and param-verify all do),
        and the non-default ``fixed_sources`` freezes them for the annulus instead.

        ``clean``: this configuration's clean reduction, already made.  Only the injected
        image is then reduced.  The draws of one averaged trial share it
        (:meth:`evaluate_mean`): the clean image depends on the configuration alone, never
        on where the sources went.  Ignored in the k-scan modes, where every evaluation
        picks its own k and so reduces its own clean image.
        """
        t0 = time.time()
        contrast = self.contrast if contrast is None else float(contrast)
        cfg = self.space.decode(x)
        if sources is None:
            sources = self._draw_sources(cfg, contrast)
        else:
            sources = [Source(s.rho, s.theta, contrast) for s in sources]
        k_used = None
        given_clean, clean = clean, None
        kinfo: Dict[str, Any] = {}
        sr = None
        why = None
        try:
            if self._kscan_active():
                given_clean = None
                cfg, k_used, kinfo = self._kscan_pick(cfg, sources, tag)
                # re-score at the chosen k(s) on fresh positions, as IDL does: its k-scan
                # picks on one set (inj_rho/inj_theta) and then draws hrho/htheta for the
                # re-score reduction, so the scan's selection bias does not carry into the
                # recorded score.  Only the non-default fixed_sources keeps the one set.
                if not self.cfg.fixed_sources:
                    rlo, rhi = self._band(self.ia, cfg)
                    sources = self.sampler.sample(self._nsrc(self.ia), rlo, rhi, self.rng, contrast)
            need_clean = self._needs_clean(raw_only)
            t_reduced = None
            if need_clean and given_clean is not None:
                inj = self._reduce(cfg, sources, tag=tag + "_inj")
                clean = given_clean
            elif need_clean and getattr(self.reducer, "max_workers", 1) >= 2:
                # injected and clean reductions concurrently (IDL: clean on the 2nd worker per night)
                import concurrent.futures as _cf
                from .parallel import pump_wait
                with _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="klip-eval") as ex:
                    f_inj = ex.submit(self._reduce, cfg, sources, tag=tag + "_inj")
                    f_cln = ex.submit(self._reduce, cfg, None, tag=tag + "_clean")
                    inj, clean = pump_wait([f_inj, f_cln])
            else:
                inj = self._reduce(cfg, sources, tag=tag + "_inj")
                if need_clean:
                    clean = self._reduce(cfg, None, tag=tag + "_clean")
            t_reduced = time.time()
            sr = self._score_images(inj, clean, sources, raw_only)
            failed = False
        except Exception as exc:  # a failed reduction scores as failed, never as a stale image
            self.log(f"  evaluation failed: {exc!r}")
            inj = EvalImages(np.full((2, 2), np.nan))
            sr = None
            failed = True
            why = _error_why(exc)
            t_reduced = time.time()
        rec = self._eval_record(cfg, phase, contrast, sources, inj, sr, failed, k_used, kinfo, t0, t_reduced,
                                why=why)
        return rec, inj, clean

    def _score_images(self, inj: EvalImages, clean: Optional[EvalImages], sources: Sequence[Source],
                      raw_only: bool = False):
        """The objective on one injected image (and the clean one), as :meth:`evaluate`
        scores it."""
        fmkw = self._fm_for(inj, clean)
        if raw_only:
            return self.objective.score_raw(inj.image, sources, None if clean is None else clean.image, **fmkw)
        return self.objective.score_search(inj.image, sources, None if clean is None else clean.image, **fmkw)

    def _eval_record(self, cfg: Config, phase: str, contrast: float, sources: Sequence[Source],
                     inj: EvalImages, sr, failed: bool, k_used: Any = None,
                     kinfo: Optional[Dict[str, Any]] = None, t0: Optional[float] = None,
                     t_reduced: Optional[float] = None, why: Optional[str] = None) -> EvalRecord:
        """The :class:`EvalRecord` of one scored (or failed) evaluation.  ``t0``: when its
        reductions were dispatched, ``t_reduced``: when the last of them was back.  A record
        without a score says why in ``meta['why']``: ``why`` for a failed one (the error),
        the objective's reason for a NaN score."""
        kinfo = kinfo or {}
        t0 = time.time() if t0 is None else t0
        if k_used is None:
            k_used = cfg.params.get("k_klip")
            if cfg.per_partition:
                k_used = {str(p): cfg.per_partition[p].get("k_klip") for p in cfg.selected}
        meta = {"failed": failed, "selected": [str(s) for s in cfg.selected],
                "zone": list(self._zone(self.ia, cfg)),
                "angle_convention": str(getattr(self.reducer, "angle_convention", "pa")),
                "t_dispatch": float(t0)}
        if failed:
            meta["why"] = why or "error"
        elif sr is not None and not np.isfinite(sr.score):
            meta["why"] = getattr(sr, "why", None) or "score not finite"
        if kinfo:
            meta["kbest"] = kinfo.get("kbest")
            if "kpn" in kinfo:
                meta["kpn"] = kinfo["kpn"]
        rec = EvalRecord(
            annulus=self.ia, index=-1, phase=phase, x=[float(v) for v in cfg.x], config=cfg.to_dict(),
            sources=[s.as_tuple() for s in sources],
            score=None if (failed or not np.isfinite(sr.score)) else float(sr.score),
            raw_score=None if failed else (None if not np.isfinite(sr.raw_score) else float(sr.raw_score)),
            per_source=[] if failed else [None if not np.isfinite(v) else float(v) for v in sr.per_source],
            raw_per_source=[] if failed else [None if not np.isfinite(v) else float(v) for v in sr.raw_per_source],
            clean_per_source=None if (failed or sr.clean_per_source is None)
            else [None if not np.isfinite(v) else float(v) for v in sr.clean_per_source],
            partition_snr={},          # the combined score only (EvalRecord.partition_snr)
            k_used=k_used, contrast=contrast, wall_s=time.time() - t0, meta=meta,
            reduce_s=None if t_reduced is None else float(t_reduced - t0))
        return rec

    def replay_draws(self, recorded) -> None:
        """From now on, inject ``recorded`` draws in order wherever this runner would draw
        fresh positions (search, validation, remeasurement), so a configuration can be
        re-scored on other frames at the same injections.  ``recorded``: a list of draws,
        each ``[(rho, theta[, contrast]), ...]`` -- ``library_ablation.json``'s ``draws``, an
        evaluation's ``meta['draw_sources']``, a validation row's ``trial_sources``.  The
        injected contrast is the runner's.  See :class:`~klip_tpe.positions.ReplaySampler`."""
        from .positions import ReplaySampler
        base = getattr(self.sampler, "base", None) if isinstance(self.sampler, ReplaySampler) else self.sampler
        self.sampler = ReplaySampler(recorded, base=base)

    def evaluate_mean(self, x: np.ndarray, phase: str, tag: str = "eval",
                      n: Optional[int] = None, on_draw=None
                      ) -> Tuple[EvalRecord, EvalImages, Optional[EvalImages]]:
        """:meth:`evaluate` repeated on ``n`` fresh source draws, scored as their MEAN.

        One trial, one history entry: the draws are an estimator of this configuration's
        score, not separate configurations.  Appending them individually would let the
        optimizer read one trial as ``n`` and would triple the apparent budget.

        The returned record carries the MEAN in ``score`` / ``raw_score`` and everything
        positional -- ``sources``, ``per_source``, the images -- from the LAST draw, which is
        what the panel then shows.  So the panel is one real measurement and the number
        beside it is the average of ``n``; ``meta['draw_scores']`` and ``meta['draw_sd']``
        record the spread so the run reports its own noise as it goes.

        The clean image is reduced ONCE per trial and shared by its draws: it depends on the
        configuration, not on where the sources went, so reducing it per draw only repeated
        it -- ``2n`` reductions where ``n + 1`` do (6 where 4 do at ``n = 3``).  Every draw's
        positions are drawn first, in order, by the same generator calls the per-draw loop
        made, so positions and scores are the ones that loop produced.  On a parallel
        reducer the clean and all ``n`` injected reductions then go out together, one round
        (``(n + 1) x partitions`` jobs), and each draw is scored, in order, as soon as its
        image and the clean are back.  The schedule changes when a reduction finishes, not
        what it returns.  The k-scan modes keep the per-draw loop: each draw picks its own k
        there, and with it its own clean image.
        """
        n = int(self.cfg.n_remeasure if n is None else n)
        if n <= 1:
            return self.evaluate(x, phase, tag=tag)
        if not self.SHARE_CLEAN or self._kscan_active():
            return self._evaluate_draws_each(x, phase, tag, n, on_draw)
        t0 = time.time()
        cfg = self.space.decode(x)
        draws = [self._draw_sources(cfg, self.contrast) for _ in range(n)]
        back: List[float] = []
        stream = self._run_ordered(self._timed(self._trial_calls(cfg, draws, tag), back),
                                   len(draws) + 1 if self._parallel_reducer() else 1)
        try:
            return self._trial_finish(x, cfg, phase, tag, draws, stream, on_draw, t0, back=back)
        finally:
            stream.close()

    @staticmethod
    def _timed(calls: Sequence[Callable[[], Any]], back: List[float]) -> List[Callable[[], Any]]:
        """``calls``, each noting in ``back`` when it returned: when its reduction was back."""
        def one(fn):
            def run():
                try:
                    return fn()
                finally:
                    back.append(time.time())
            return run
        return [one(fn) for fn in calls]

    def _trial_calls(self, cfg: Config, draws: Sequence[Sequence[Source]], tag: str) -> List[Callable[[], Any]]:
        """The reductions of one trial, in the order :meth:`_trial_finish` reads them: the
        clean image (when the objective needs one), then each draw's injected image.  Tags
        as the per-draw evaluations named them (a single draw as :meth:`evaluate` does)."""
        one = len(draws) == 1
        calls: List[Callable[[], Any]] = []
        if self._needs_clean():
            ctag = f"{tag}_clean" if one else f"{tag}_d1_clean"
            calls.append(lambda: self._reduce(cfg, None, tag=ctag))
        for j, d in enumerate(draws):
            srcs = [Source(s.rho, s.theta, self.contrast) for s in d]
            itag = f"{tag}_inj" if one else f"{tag}_d{j + 1}_inj"
            calls.append(lambda srcs=srcs, itag=itag: self._reduce(cfg, srcs, tag=itag))
        return calls

    def _trial_finish(self, x: np.ndarray, cfg: Config, phase: str, tag: str,
                      draws: Sequence[Sequence[Source]], stream, on_draw, t0: float, combine: bool = True,
                      back: Optional[List[float]] = None
                      ) -> Tuple[EvalRecord, EvalImages, Optional[EvalImages]]:
        """Score one trial from ``stream`` (:meth:`_run_ordered` over :meth:`_trial_calls`),
        draw by draw, in order.  ``combine=False`` with a single draw returns that draw as
        :meth:`evaluate` would (no ``draw_*`` entries in its record).

        Failures behave as the per-draw schedule did.  A draw whose injected reduction
        fails is a failed draw.  When the clean reduction fails, draw 1 fails with it and
        the later draws reduce their own clean images, one draw at a time.  Either way the
        trial takes exactly its own reductions off ``stream``, which a concurrent batch
        shares between its proposals.  ``back``: when each of this trial's reductions came
        back (:meth:`_timed`), for the record's ``reduce_s``."""
        n = len(draws)
        contrast = self.contrast
        recs: List[EvalRecord] = []
        last: Tuple[Any, Any, Any] = (None, None, None)

        def _done(j, out):
            nonlocal last
            rec, inj, clean = out
            recs.append(rec)
            last = out
            if on_draw is not None:
                on_draw(j, n, rec, inj, clean)

        shared, clean_exc = None, None
        if self._needs_clean():
            _, res = next(stream)
            if isinstance(res, BaseException):
                clean_exc = res
            else:
                shared = res
        for j in range(n):
            srcs = [Source(s.rho, s.theta, contrast) for s in draws[j]]
            _, res = next(stream)
            if clean_exc is not None:
                # draw 1 fails with its clean image (the injected failure is reported first,
                # as it was), and the later draws reduce their own
                err = res if isinstance(res, BaseException) else clean_exc
                self.log(f"  evaluation failed: {err!r}")
                rec = self._eval_record(cfg, phase, contrast, srcs, EvalImages(np.full((2, 2), np.nan)),
                                        None, True, t0=t0, why=_error_why(err))
                _done(0, (rec, EvalImages(np.full((2, 2), np.nan)), None))
                for _ in range(1, n):                 # the other draws' images, made without a clean one
                    next(stream)
                for jj in range(1, n):
                    _done(jj, self.evaluate(x, phase, sources=draws[jj], tag=f"{tag}_d{jj + 1}"))
                if back is not None:
                    back.append(time.time())          # their reductions, and their scoring with them
                break
            clean = shared
            why = None
            try:
                if isinstance(res, BaseException):
                    clean = None                      # as evaluate() returned it for a failed reduction
                    raise res
                inj = res
                sr = self._score_images(inj, shared, srcs)
                failed = False
            except Exception as exc:                  # noqa: BLE001
                self.log(f"  evaluation failed: {exc!r}")
                inj, sr, failed, why = EvalImages(np.full((2, 2), np.nan)), None, True, _error_why(exc)
            _done(j, (self._eval_record(cfg, phase, contrast, srcs, inj, sr, failed, t0=t0, why=why), inj, clean))
        rec, inj, clean = last
        t_red = max(back) if back else None
        if not combine and n == 1:
            if t_red is not None:
                rec.reduce_s = float(t_red - t0)
            return rec, inj, clean
        return self._combine_draws(recs, rec, wall=time.time() - t0,
                                   reduce=None if t_red is None else t_red - t0), inj, clean

    #: share one clean reduction between the draws of an averaged trial (see
    #: :meth:`evaluate_mean`).  False restores the per-draw loop -- a switch for the test
    #: that the two give the same trial, not a setting.
    SHARE_CLEAN = True

    def _evaluate_draws_each(self, x: np.ndarray, phase: str, tag: str, n: int, on_draw=None
                             ) -> Tuple[EvalRecord, EvalImages, Optional[EvalImages]]:
        """Every draw a complete :meth:`evaluate` of its own, clean image included."""
        recs, last = [], (None, None, None)
        for j in range(n):
            rec, inj, clean = self.evaluate(x, phase, tag=f"{tag}_d{j+1}")
            recs.append(rec)
            last = (rec, inj, clean)
            if on_draw is not None:
                on_draw(j, n, rec, inj, clean)
        rec, inj, clean = last
        return self._combine_draws(recs, rec), inj, clean

    @staticmethod
    def _combine_draws(recs: List[EvalRecord], last: EvalRecord, wall: Optional[float] = None,
                       reduce: Optional[float] = None) -> EvalRecord:
        """Average the draws' scores onto the last draw's record.  ``wall`` and ``reduce``:
        the trial's elapsed and reduction times, when its draws overlapped (otherwise the
        draws' sums)."""
        def _mean(vals):
            v = [float(s) for s in vals if s is not None and np.isfinite(s)]
            return float(np.mean(v)) if v else None

        scores = [r.score for r in recs]
        good = [float(s) for s in scores if s is not None and np.isfinite(s)]
        meta = dict(last.meta)
        meta["draw_scores"] = [None if s is None or not np.isfinite(s) else float(s) for s in scores]
        # every draw's injections and raw score, not just the last one's: with these a trial
        # can be re-scored on other frames (positions.ReplaySampler), which a sequential RNG
        # stream cannot reproduce once the frames' forbidden sectors differ
        meta["draw_sources"] = [[list(s) for s in r.sources] for r in recs]
        meta["draw_raw_scores"] = [None if r.raw_score is None or not np.isfinite(r.raw_score)
                                   else float(r.raw_score) for r in recs]
        meta["draw_n"] = len(recs)
        meta["draw_failed"] = int(len(recs) - len(good))
        # the last draw's reason is not the trial's: the trial fails only when every draw does
        meta.pop("why", None)
        if meta["draw_failed"]:
            whys = [None if (s is not None and np.isfinite(s)) else (r.meta.get("why") or "score not finite")
                    for s, r in zip(scores, recs)]
            meta["draw_why"] = whys
            if not good:
                meta["why"] = collections.Counter(w for w in whys if w).most_common(1)[0][0]
        meta["draw_sd"] = float(np.std(good, ddof=1)) if len(good) > 1 else None
        meta["draw_spread"] = float(max(good) - min(good)) if len(good) > 1 else None
        if reduce is None:
            parts = [getattr(r, "reduce_s", None) for r in recs]
            reduce = float(sum(parts)) if parts and all(v is not None for v in parts) else None
        return replace(last, score=_mean(scores), raw_score=_mean([r.raw_score for r in recs]),
                       wall_s=float(sum(r.wall_s for r in recs)) if wall is None else float(wall), meta=meta,
                       reduce_s=None if reduce is None else float(reduce))

    # ------------------------------------------------------------ calibration
    def _scan_k(self, cfg0: Config, rlo: float, rhi: float, nsrc: int, contrast: float,
                info: Dict[str, Any], k_now: int) -> Optional[int]:
        """The k-scan of the default configuration at ``contrast``: the search metric of
        every k up to :meth:`_k_scan_max`, median over ``n_remeasure`` fresh source sets
        (IDL scanned one set; one draw's argmax over a 30-point noisy curve is mostly the
        draw).  Records the curve in ``info`` and returns the best k, or None when nothing
        finite came back (the caller then keeps ``k_now``)."""
        try:
            kmax = self._k_scan_max()
            curves = []
            cfgk = self._with_k(cfg0, kmax)
            needs = bool(self.objective.metric.needs_clean)
            # every draw first, in order, keeping the generator state after each: a failure
            # in draw m puts it back where the draw-by-draw loop stopped
            plan = []
            for _ in range(max(int(self.cfg.calibration.n_remeasure), 1)):
                src = self.sampler.sample(nsrc, rlo, rhi, self.rng, contrast)
                plan.append((src, self.rng.bit_generator.state))
            calls: List[Callable[[], Any]] = []
            for src, _ in plan:
                calls.append(lambda src=src: self._reduce(cfgk, src, k_scan=True, tag="calib_scan"))
                if needs:
                    calls.append(lambda: self._reduce(cfgk, None, k_scan=True, tag="calib_cscan"))
            per = 2 if needs else 1
            with contextlib.closing(self._run_ordered(calls, len(calls) if self._parallel_reducer() else 1)) as st:
                for m, (src, state) in enumerate(plan):
                    got = [next(st)[1] for _ in range(per)]
                    bad = next((r for r in got if isinstance(r, BaseException)), None)
                    if bad is not None:
                        self.rng.bit_generator.state = state
                        raise bad
                    scan, cscan = got[0], (got[1] if needs else None)
                    curves.append([self.objective.score_raw(scan.image[kk], src,
                                                            None if cscan is None else cscan.image[kk]).score
                                   for kk in range(scan.image.shape[0])])
            n = min(len(c) for c in curves)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN slices are a legal answer
                sc = np.nanmedian(np.array([c[:n] for c in curves], float), axis=0)
            # The spread ACROSS draws, kept rather than thrown away with the median: it is the
            # only estimate available of how much of this curve is the draw, and the margin
            # the argmax wins by has to be read against it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                arr = np.array([c[:n] for c in curves], float)
                spread = float(np.nanmedian(np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0))) \
                    if arr.shape[0] > 1 else float("nan")
            info["kscan"] = [None if not np.isfinite(v) else float(v) for v in sc]
            info["kscan_contrast"] = float(contrast)
            info["kscan_draw_spread"] = None if not np.isfinite(spread) else spread
            if not np.any(np.isfinite(sc)):
                self.log(f"  calibration k-scan at contrast {contrast:.3e}: no finite score at any k; keeping k = {k_now}")
                return None
            k = int(np.nanargmax(sc)) + 1
            now = sc[k_now - 1] if 1 <= k_now <= sc.size else np.nan
            self.log(f"  calibration k-scan at contrast {contrast:.3e}: k_default = {k} "
                     f"(S/N {sc[k - 1]:.2f}; k = {k_now} scores {now:.2f})")
            # An argmax at either end of the scanned range is not an interior optimum, and
            # saying so is the same guard locate_boundaries already applies to its own scan:
            # a extremum at the edge of the window means the window is wrong.  Measured on
            # HIP 65426 F1140C: with a background-mismatched RDI library the curve was flat to
            # 1.5% over k = 4..20 with k = 1 on top by 1.6% (the leading KL mode was the sky
            # pedestal, so removing it was the only subtraction that helped); with the
            # background fixed the argmax moved to k = 20 -- the other end -- winning by 0.1%.
            # Neither was a measurement of anything.
            edge = "bottom" if k == 1 else ("top" if k == sc.size else "")
            if edge:
                self.log(f"  calibration k-scan: WARNING k_default = {k} is at the {edge} of the "
                         f"scanned range (1..{sc.size}), so this is an edge hit and not an "
                         f"interior optimum"
                         + (f" -- the data still wants more modes at k = {sc.size}; raise "
                            f"k_klip_max, which caps the SEARCH as well as this scan"
                            if edge == "top" else
                            " -- k = 1 keeps only the single dominant mode, which is the "
                            "signature of a basis whose first component is not the star "
                            "(an unsubtracted background, say)"))
            info["kscan_edge"] = edge or None
            # And when the win is inside the draw-to-draw spread, it is not a win: keep the
            # incumbent rather than moving the seed of the whole run on noise.
            if (np.isfinite(spread) and np.isfinite(now) and 1 <= k_now <= sc.size
                    and (sc[k - 1] - now) < spread):
                self.log(f"  calibration k-scan: keeping k = {k_now} -- k = {k} beats it by "
                         f"{sc[k - 1] - now:.3f} in S/N, inside the {spread:.3f} spread between "
                         f"the {arr.shape[0]} draws, so the argmax is not resolved")
                info["kscan_unresolved"] = True
                return None
            return k
        except Exception as exc:
            self.log(f"  calibration k-scan failed ({exc!r}); keeping k = {k_now}")
            return None

    def calibrate(self, ia: int) -> Tuple[float, int, Dict[str, Any]]:
        """Choose the injection contrast for annulus ``ia`` (see :class:`CalibrationConfig`).
        Returns ``(contrast, k_default, info)``.

        Order of operations (since 2026-09-17): the contrast is walked into the target S/N
        window at the configured default ``k`` first; the k-scan that picks the seed's ``k``
        runs only once the sources sit at that S/N, and the window is then re-measured at
        the k it chose (the contrast keeps moving if that took it out of the window).
        ``optimize_near_2_tpe`` scanned k once at the *starting* contrast and calibrated at
        that k -- fine when the starting contrast is close, which NEAR's per-annulus
        ``use_contrast`` made it, but the generic ``contrast0 = 3e-5`` is orders of magnitude
        from the calibrated value on every public data set here.  At 3e-5 the sources scored
        S/N ~ 0 on beta Pic (and ~ 40 on HD 95086), ``argmax_k`` of that curve was the noise,
        and the seed k -- also the k the contrast was calibrated at -- came out 4/6/13, 8,
        4/1 and 18/6 across the annuli of runs A2, B2, C and D (2026-09-16/17): a different
        "default" per annulus, chosen by a coin.  With a forced contrast the scan runs at
        that contrast, as before, and the window is not enforced.
        """
        cc = self.cfg.calibration
        self.hb.stage(f"annulus {ia + 1} calibration", annulus=ia)
        forced = 0.0
        if cc.forced is not None and len(cc.forced) > ia and cc.forced[ia] and cc.forced[ia] > 0:
            forced = float(cc.forced[ia])
        contrast = forced if forced > 0 else self.contrast
        c_start = contrast
        info: Dict[str, Any] = {"forced": forced, "trials": []}
        x0 = self._default_vector()
        x0 = self._project(x0, is_random=False)
        cfg0 = self.space.decode(x0)
        for k, v in cc.overrides.items():
            cfg0.params[k] = v
            for d in cfg0.per_partition.values():
                d[k] = v
        kdef0 = int(cfg0.params.get("k_klip", self.cfg.defaults.get("k_klip", 10)) or 10)
        kdef = kdef0
        info["k_default_initial"] = kdef0
        rlo, rhi = self._band(ia, cfg0)
        nsrc = self._nsrc(ia)
        scan_wanted = bool(cc.scan_k and ("k_klip" in self.space.bases or self.cfg.k_mode in SCAN_MODES)
                           and self.reducer.supports_kscan)
        scanned = False
        rescued = False
        c_at_cap = False
        cfgk = self._with_k(cfg0, kdef)
        info["k_default"] = kdef
        self._calib_images = None
        msnr = np.nan
        budget = int(cc.max_trials)
        trial = 0
        while trial < budget:
            vals = []
            whys: List[Optional[str]] = []       # per draw: why it has no score (None: it has one)
            src = []
            errors: List[BaseException] = []
            # the remeasurements' positions first, in order (every draw happens whatever
            # fails), then their reductions together, scored in order
            draws = [self.sampler.sample(nsrc, rlo, rhi, self.rng, contrast) for _ in range(cc.n_remeasure)]
            need_c = bool(self.objective.needs_clean)
            calls: List[Callable[[], Any]] = []
            for d in draws:
                calls.append(lambda d=d, cfgk=cfgk: self._reduce(cfgk, d, tag="calib_inj"))
                if need_c:
                    calls.append(lambda cfgk=cfgk: self._reduce(cfgk, None, tag="calib_clean"))
            with contextlib.closing(self._run_ordered(calls, len(calls) if self._parallel_reducer() else 1)) \
                    as stream:
                for src in draws:
                    got_inj = next(stream)[1]
                    got_cln = next(stream)[1] if need_c else None
                    try:
                        if isinstance(got_inj, BaseException):
                            raise got_inj
                        inj = got_inj
                        if isinstance(got_cln, BaseException):
                            raise got_cln
                        clean = got_cln
                        r = self.objective.score_search(inj.image, src, None if clean is None else clean.image,
                                                        **self._fm_for(inj, clean))
                        vals.append(r.score)
                        whys.append(None if np.isfinite(r.score) else (getattr(r, "why", None) or "score not finite"))
                        # keep the last measurement for the display (on_calibration images)
                        self._calib_images = {
                            "inj": inj, "clean": clean, "sources": list(src), "contrast": contrast, "config": cfgk,
                            "per_source": [None if not np.isfinite(v) else float(v) for v in r.per_source],
                            "clean_per_source": None if r.clean_per_source is None
                            else [None if not np.isfinite(v) else float(v) for v in r.clean_per_source]}
                    except Exception as exc:
                        self.log(f"  calibration reduction failed: {exc!r}")
                        vals.append(np.nan)
                        whys.append(_error_why(exc))
                        errors.append(exc)
            if trial == 0 and errors and len(errors) == len(vals):
                # Not a calibration problem: the DEFAULT reduction of this annulus does not run
                # at all -- and every evaluation of the search uses the same reducer on the
                # same zone with the same injection ladder, so every one of them would fail
                # the same way.  Left alone, that is what happened: paper run D on 2026-09-17
                # spent 350 evaluations on a pyklip/numpy mismatch and ended with "run
                # complete" and no winner.  Stop here, with the reason.
                raise RuntimeError(
                    f"the default reduction of annulus {ia + 1} failed on all {len(errors)} attempts "
                    f"({errors[-1]!r}); nothing in the search can succeed until that is fixed") from errors[-1]
            msnr = nanmedian_even(vals)
            info["trials"].append({"contrast": contrast, "k": kdef,
                                   "snr": None if not np.isfinite(msnr) else float(msnr),
                                   "values": [None if not np.isfinite(v) else float(v) for v in vals]})
            nfail = sum(w is not None for w in whys)
            if nfail:
                info["trials"][-1]["why"] = whys
            self.log(f"  calibration trial {trial+1}: contrast={contrast:.3e}  k={kdef}  median S/N={msnr:.2f}"
                     + (f"  ({nfail} of {len(whys)} draws failed: "
                        f"{_whys(collections.Counter(w for w in whys if w))})" if nfail else ""))
            self._emit("on_calibration_trial", ia, trial + 1, contrast, msnr, self._calib_images)
            if self.cfg.write_setup_files:
                self._write_setup_file(os.path.join(self._ann_dir(ia), f"calib{trial+1:04d}_setup.txt"), "calib",
                                       ia, trial + 1, cfgk, src, contrast, msnr, kdef)
            in_window = bool(np.isfinite(msnr) and cc.target[0] <= msnr <= cc.target[1])
            settled = forced > 0 or not np.isfinite(msnr) or in_window
            if scan_wanted and not scanned and (forced > 0 or in_window):
                # The sources now sit where the search will see them: pick the seed's k here.
                scanned = True
                knew = self._scan_k(cfg0, rlo, rhi, nsrc, contrast, info, kdef)
                if knew is not None and knew != kdef:
                    kdef = knew
                    cfgk = self._with_k(cfg0, kdef)
                    info["k_default"] = kdef
                    if forced <= 0:
                        # the window was measured at the old k; re-measure at the one the seed
                        # will use, with a full budget to walk the contrast back into the window
                        # (a scan's best k is by construction the one that raised the S/N most).
                        # A forced contrast cannot move, so it keeps its one trial (the seed
                        # evaluation reports the S/N at the chosen k anyway).
                        budget = trial + 1 + int(cc.max_trials)
                        trial += 1
                        continue
            if settled:
                break
            fac = float(np.clip(cc.aim / max(msnr, 0.5), cc.step_clip[0], cc.step_clip[1]))
            contrast *= fac
            caps = [v for v in (cc.ceiling, cc.max_contrast) if v is not None]
            cap = min(caps) if caps else None
            if cap is not None and contrast > cap:
                # The S/N is not following the injected flux at this k.  Before giving up,
                # ask the k-scan whether ANY k detects the sources at the last measured
                # contrast: at small radii the configured k over-subtracts a source that
                # every reference frame contains (run A2's [6, 12] px annulus at k = 10:
                # S/N ~ 1 from 3e-3 to 1e-1, while k = 1 sees the same sources at S/N 10).
                # If one does, the walk continues at that k with a fresh budget; the seed's
                # k is still chosen by the scan at the calibrated contrast afterwards.
                c_last = float(info["trials"][-1]["contrast"])
                if scan_wanted and not rescued and cc.max_contrast is not None:
                    rescued = True
                    knew = self._scan_k(cfg0, rlo, rhi, nsrc, c_last, info, kdef)
                    best = np.nanmax(np.array([np.nan if v is None else v for v in info.get("kscan", [])], float)) \
                        if info.get("kscan") else np.nan
                    if knew is not None and knew != kdef and np.isfinite(best) and best >= 2.0:
                        # walk again from the starting contrast, upward, so the first contrast
                        # inside the window is the one adopted: the S/N of a bright source on
                        # a small ring saturates, and a walk resumed at c_last would stop there
                        self.log(f"  calibration: S/N did not follow the contrast at k = {kdef}; walking again "
                                 f"from {c_start:.2e} at k = {knew} (scan S/N {best:.2f} at {c_last:.2e})")
                        kdef = knew
                        cfgk = self._with_k(cfg0, kdef)
                        info["k_default"] = kdef
                        info["rescue_k"] = kdef
                        contrast = c_start
                        budget = trial + 1 + int(cc.max_trials)
                        trial += 1
                        continue
                # No k sees them either.  Stay AT the cap rather than walking past it: one more
                # trial measures the S/N the search will actually be scored against there, and
                # if that is still outside the window the annulus is uncalibrated -- said
                # loudly below, not raised, because a default configuration that cannot see an
                # injection is a statement about the default and not about the problem.
                if c_at_cap:
                    contrast = cap
                    break
                contrast = cap
                c_at_cap = True
            trial += 1
        if scan_wanted and not scanned and np.isfinite(msnr) and msnr >= 2.0:
            # the window was never reached within the budget, but the sources are detected:
            # a scan here is still about the right regime (there is no budget left to
            # re-measure at the chosen k, so the recorded S/N is the one at k = kdef0)
            knew = self._scan_k(cfg0, rlo, rhi, nsrc, contrast, info, kdef)
            if knew is not None:
                kdef = knew
                info["k_default"] = kdef
        info["contrast"] = contrast
        info["snr"] = None if not np.isfinite(msnr) else float(msnr)
        info["k_default"] = kdef
        # Did the calibration do its job?  A forced contrast is not calibrated by definition;
        # otherwise the window is the job.  Saying so here is the difference between a run
        # whose injected S/N means "5 sigma" and one whose axis is a guess: the old code
        # printed nothing and the contrast kept climbing.
        info["uncalibrated"] = bool(forced <= 0 and not (np.isfinite(msnr)
                                                        and cc.target[0] <= msnr <= cc.target[1]))
        if info["uncalibrated"]:
            hist = ", ".join(f"{t['contrast']:.1e} -> {t['snr'] if t['snr'] is None else round(t['snr'], 2)}"
                             for t in info["trials"])
            caps = [v for v in (cc.ceiling, cc.max_contrast) if v is not None]
            at_cap = bool(caps) and contrast >= 0.999 * min(caps)
            self.log(f"  ** annulus {ia + 1} could NOT be calibrated: the default configuration's median S/N "
                     f"is {msnr:.2f}, outside {cc.target[0]:g}-{cc.target[1]:g}, at contrast {contrast:.3e}"
                     + (f" -- the cap (CalibrationConfig.max_contrast)" if at_cap else "")
                     + f".  Trials: {hist}.  The search will run at that contrast, so its injected S/N is "
                     f"not on the usual scale and neither is this annulus' 5-sigma curve.  An S/N that does "
                     f"not follow the injected flux usually means the {nsrc} injected sources are limiting "
                     f"each other's noise ring (their KLIP wings land in each other's apertures): inject "
                     f"fewer in this annulus (RunConfig.n_sources takes a per-annulus list; the rule gives "
                     f"{n_sources_rule(ia, self._annulus(ia)[1] * self.pxscale, None)} here) or widen it.")
        return contrast, kdef, info

    # ----------------------------------------------------------------- search
    def _append(self, rec: EvalRecord) -> int:
        rec.index = len(self.history)
        i = self.history.append(np.array(rec.x), np.nan if rec.score is None else rec.score,
                                {"phase": rec.phase, "raw": rec.raw_score, "wall": rec.wall_s,
                                 "sources": rec.sources, "k": rec.k_used})
        self._log_row(rec)
        if self.cfg.write_setup_files and not rec.meta.get("failed"):
            cfg = self._config_from_dict(rec.config)
            self._write_setup_file(os.path.join(self._ann_dir(rec.annulus), f"eval{rec.index+1:04d}_setup.txt"),
                                   "eval", rec.annulus, rec.index + 1, cfg, [Source(*s) for s in rec.sources],
                                   rec.contrast, rec.score, rec.k_used)
        return i

    def _log_row(self, rec: EvalRecord) -> None:
        names = self.space.names
        rpath = os.path.join(self.run_dir, "results.txt")
        new = not os.path.exists(rpath) or os.path.getsize(rpath) == 0
        with open(rpath, "a") as f:
            if new:
                f.write("# annulus iter phase " + " ".join(names) + "   k_klip   score   raw_score   wall_s\n")
            kstr = rec.k_used if not isinstance(rec.k_used, dict) else "[" + ",".join(
                str(v) for v in rec.k_used.values()) + "]"
            sc = "nan" if rec.score is None else f"{rec.score:.4f}"
            rw = "nan" if rec.raw_score is None else f"{rec.raw_score:.4f}"
            f.write(f"{rec.annulus+1:3d} {rec.index+1:5d} {rec.phase:<7s} "
                    + " ".join(f"{v:9.4f}" for v in rec.x) + f"   {kstr}   {sc}   {rw}   {rec.wall_s:.1f}\n")
        jpath = os.path.join(self.run_dir, "results.jsonl")
        with open(jpath, "a") as f:
            f.write(rec.to_json() + "\n")
        self._durable_later(rpath, jpath)          # on disk before a checkpoint counts this line
        self.records_count += 1

    def _durable_later(self, *paths: str) -> None:
        """Files the next checkpoint write makes durable before it writes the checkpoint
        (:meth:`_sync_files`): what the checkpoint counts or points to, written since."""
        with _UNSYNCED_LOCK:
            self.__dict__.setdefault("_unsynced", set()).update(p for p in paths if p)

    def _sync_files(self) -> None:
        """fsync every file :meth:`_durable_later` collected, then their directories, then
        the run directory (it holds the annulus directories' own entries)."""
        with _UNSYNCED_LOCK:
            paths = self.__dict__.get("_unsynced") or set()
            self._unsynced = set()
        if not paths:
            return
        dirs = {os.path.abspath(self.run_dir)}
        for p in sorted(paths):
            if durable.fsync_file(p):
                dirs.add(os.path.dirname(os.path.abspath(p)))
        for d in sorted(dirs, key=len, reverse=True):          # deepest first
            durable.fsync_dir(d)

    def _set_best_images(self, rec: EvalRecord, inj: EvalImages, clean: Optional[EvalImages]) -> None:
        cfg = self._apply_k_flag(self._config_from_dict(rec.config), rec.k_used)
        sources = [Source(*s) for s in rec.sources]
        self._best_images = {"inj": inj.image, "clean": None if clean is None else clean.image,
                             "record": rec, "inj_images": inj, "clean_images": clean,
                             "zone": self._zone(self.ia, cfg), "sources": sources}
        # A §9.3: live KLIP-FM preview at each new best (one extra clean FM pass; opt-in)
        if self.cfg.fm_preview and self.do_fm and not rec.meta.get("failed") and sources:
            try:
                fm = self._fm_pass(cfg, self._zone(self.ia, cfg), rec.contrast, f"a{self.ia+1}_e{rec.index+1}_fm",
                                   sources=sources, clean_img=None if clean is None else clean.image)
                if fm is not None:
                    self._best_images["fm"] = fm
                    inj.fm_image = fm["fm_image"]          # visible to on_eval through the images
                    if clean is None and self._best_images.get("clean") is None:
                        self._best_images["clean"] = fm["clean"]
            except Exception as exc:
                self.log(f"  KLIP-FM preview failed: {exc!r}")

    def _search_annulus(self, ia: int, start_index: int) -> None:
        """Run evaluations ``start_index .. n_iter-1`` of annulus ``ia``."""
        c = self.cfg
        n_iter = c.per_annulus(c.n_iter, ia)
        opt = self._make_optimizer(ia)
        n_init = opt.n_init
        if self._pipelined():
            if self._search_pipeline(ia, n_iter, opt):
                return self._search_annulus(ia, 0)
            return
        if int(c.batch) > 1:
            why = " (each trial picks its k as it runs, so batches go whole)" \
                if str(c.batch_mode).lower() == "pipeline" else ""
            self.log(f"[ann {ia+1}] batches of {int(c.batch)} trials, each batch whole{why}")
        while len(self.history) < n_iter:
            nb = max(1, min(c.batch, n_iter - len(self.history)))
            t_loop = time.time()
            props = opt.ask(self.history, self.rng, nb)
            if nb > 1 and self._batch_concurrent():
                if self._search_batch(ia, n_iter, props, t_loop):
                    return self._search_annulus(ia, 0)
                continue
            for b, pr in enumerate(props):
                phase = pr.flags.get("phase", "tpe")
                is_random = phase in ("warmup", "explore", "random")
                x = self._project(pr.x, is_random=is_random)   # optimizer already tied link_params
                self._hb_eval(ia, len(self.history) + 1, n_iter, phase)
                rec, inj, clean = self.evaluate_mean(
                    x, phase, tag=f"a{ia+1}_e{len(self.history)+1}",
                    on_draw=lambda j, nd, r, i_, c_: self._on_draw(ia, j, nd, r, i_, c_))
                if self._after_eval(ia, n_iter, phase, rec, inj, clean, t_loop, checkpoint=b == len(props) - 1):
                    return self._search_annulus(ia, 0)
                t_loop = time.time()

    def _batch_concurrent(self) -> bool:
        """Whether a batch of proposals (``RunConfig.batch > 1``) is evaluated concurrently:
        on a parallel reducer, outside the k-scan modes (where each draw picks its own k and
        the draws stay one after another)."""
        return self._parallel_reducer() and self.SHARE_CLEAN and not self._kscan_active()

    def _search_batch(self, ia: int, n_iter: int, props, t_loop: float) -> bool:
        """Evaluate a batch of proposals concurrently (``RunConfig.batch``, ``--batch``).

        Each proposal is projected, its eval index assigned and its draws' positions drawn
        at ask time, in proposal order -- the generator calls a one-at-a-time batch makes.
        Then every proposal's clean and injected reductions go out together, and the trials
        are scored and appended in index order, so the history does not depend on which
        finishes first.  The same batch evaluated one proposal at a time gives the same
        history.  Returns True when a re-calibration restarted the annulus.

        The checkpoint is taken once the whole batch is in, as for a batch evaluated one
        proposal at a time: a run stopped part way through a batch resumes at its start,
        asks for the same proposals and reduces them again, so the resumed run is the one
        that never stopped.  ``wall_s`` of each trial counts from the batch's dispatch."""
        n = max(int(self.cfg.n_remeasure), 1)
        prepared = []
        for b, pr in enumerate(props):
            phase = pr.flags.get("phase", "tpe")
            x = self._project(pr.x, is_random=phase in ("warmup", "explore", "random"))
            cfg = self.space.decode(x)
            draws = [self._draw_sources(cfg, self.contrast) for _ in range(n)]
            prepared.append((phase, x, cfg, draws, f"a{ia+1}_e{len(self.history) + 1 + b}",
                             self.rng.bit_generator.state))
        calls: List[Callable[[], Any]] = []
        backs: List[List[float]] = [[] for _ in prepared]
        for (_, _, cfg, draws, tag, _), back in zip(prepared, backs):
            calls += self._timed(self._trial_calls(cfg, draws, tag), back)
        t0 = time.time()
        on_draw = (lambda j, nd, r, i_, c_: self._on_draw(ia, j, nd, r, i_, c_)) if n > 1 else None
        with contextlib.closing(self._run_ordered(calls, len(calls))) as stream:
            for b, (phase, x, cfg, draws, tag, rng_after) in enumerate(prepared):
                self._hb_eval(ia, len(self.history) + 1, n_iter, phase)
                rec, inj, clean = self._trial_finish(x, cfg, phase, tag, draws, stream, on_draw, t0,
                                                     combine=n > 1, back=backs[b])
                if self._after_eval(ia, n_iter, phase, rec, inj, clean, t_loop, rng_state=rng_after,
                                    checkpoint=b == len(prepared) - 1):
                    return True
                t_loop = time.time()
        return False

    def _after_eval(self, ia: int, n_iter: int, phase: str, rec: EvalRecord, inj: EvalImages,
                    clean: Optional[EvalImages], t_loop: float,
                    rng_state: Optional[Dict[str, Any]] = None, checkpoint: bool = True) -> bool:
        """Record one search trial: history, best images, files, display, log, checkpoint,
        running stitch, and the re-calibration revisit.  Returns True when the revisit
        reset the annulus (the caller restarts its search).  ``rng_state``: the generator
        state right after this trial's draws, restored before such a reset when later
        proposals of a concurrent batch have drawn already.  ``checkpoint=False`` inside a
        batch: the batch is checkpointed once, after its last trial."""
        i = self._append(rec)
        self._eval_bookkeeping(ia, n_iter, i, phase, rec, inj, clean, t_loop, checkpoint)
        rc = self._recal_check(ia, i)
        if rc is None:
            return False
        if rc[0] == "reset" and rng_state is not None:
            self.rng.bit_generator.state = rng_state   # as if the batch had stopped here
        return self._recal_apply(ia, rc)

    def _eval_bookkeeping(self, ia: int, n_iter: int, i: int, phase: str, rec: EvalRecord, inj: EvalImages,
                          clean: Optional[EvalImages], t_loop: float, checkpoint: bool = True) -> None:
        """Everything after a trial enters the history at index ``i``: best images, its
        files, the checkpoint, the running stitch, the display and the log line.

        The display comes last, so the trial is processed but for its own panel when the
        display counts its time from the dispatch.  The log line then gives the same two
        numbers: ``reduce`` from the dispatch until its last reduction was back, ``eval``
        until now.  With several trials in flight neither the interval between two
        evaluations nor ``wall_s`` is either of them (NEAR2 note, 2026-10-12).  ``t_loop``
        is no longer used."""
        c = self.cfg
        bi, bs = self.history.best()
        if i == bi and inj.image is not None and inj.image.ndim == 2:
            self._set_best_images(rec, inj, clean)
        self._save_eval_images(rec, inj, clean)
        if checkpoint:
            self.checkpoint(routine=True)
        if c.stitch_every and c.stitch_every > 0 and (i + 1) % int(c.stitch_every) == 0:
            self._guard("running stitch", self._running_stitch, ia)
        self._emit("on_eval", rec, inj, clean, i == bi)
        t_disp = rec.meta.get("t_dispatch")
        eval_s = None if t_disp is None else time.time() - float(t_disp)
        elapsed = time.time() - self.wall0 + self.wall_prev
        _sd = rec.meta.get("draw_sd")
        _dn = rec.meta.get("draw_n")
        _df = int(rec.meta.get("draw_failed") or 0)
        self.log(f"[ann {ia+1}] eval {i+1}/{n_iter} {phase:<7s} score={rec.score if rec.score is None else round(rec.score,3)}"
                 + (f" (mean of {_dn}" + (f", sd {_sd:.2f}" if _sd is not None else "")
                    + (f", {_df} failed" if _df else "") + ")" if _dn else "")
                 + (f" [{rec.meta['why']}]" if rec.score is None and rec.meta.get("why") else "")
                 + f" raw={rec.raw_score if rec.raw_score is None else round(rec.raw_score,3)}"
                 f" best={bs:.3f}@{bi+1}  contrast={self.contrast:.2e}"
                 f"  reduce {_secs(rec.reduce_s)} / eval {_secs(eval_s)}  ({elapsed/3600:.2f} h)")
        self._tally_draws(ia, rec)

    #: the share of failed draws, after at least ``FAILED_DRAWS_AFTER`` of them, at which the
    #: log warns once per annulus (the end of the search gives the count either way)
    FAILED_DRAWS_WARN = 0.1
    FAILED_DRAWS_AFTER = 20

    def _tally_draws(self, ia: int, rec: EvalRecord) -> None:
        """Count a search trial's draws and failed draws, with their reasons, and warn once
        when failures are common.  Each failed draw also says why on its own line, but a
        rate is what shows a systematic cause: at 1.1 FWHM the inner guard failed half the
        draws for 400 evaluations before the run was stopped (NEAR2 note, 2026-10-13)."""
        t = self._draw_tally.setdefault(ia, {"n": 0, "failed": 0, "why": collections.Counter(),
                                             "warned": False, "since_resume": False})
        if rec.meta.get("draw_n"):
            n, whys = int(rec.meta["draw_n"]), [w for w in (rec.meta.get("draw_why") or []) if w]
        else:
            n, whys = 1, ([rec.meta.get("why") or "score not finite"] if rec.score is None else [])
        t["n"] += n
        t["failed"] += len(whys)
        t["why"].update(whys)
        if (not t["warned"] and t["n"] >= self.FAILED_DRAWS_AFTER
                and t["failed"] >= self.FAILED_DRAWS_WARN * t["n"]):
            t["warned"] = True
            self.log(f"  ** annulus {ia+1}: {t['failed']} of the first {t['n']} draws failed "
                     f"({_whys(t['why'])}).  A failed draw does not count toward its trial's mean.  "
                     f"Each one's reason is on its log line and in results.jsonl (meta 'why', 'draw_why')")

    def _draw_note(self, ia: int) -> None:
        """The end of a search: how many of its draws failed, and why."""
        t = self._draw_tally.get(ia)
        if not t or not t["failed"]:
            return
        since = " since the resume" if t["since_resume"] else ""
        self.log(f"  note: {t['failed']} of {t['n']} draws{since} failed ({_whys(t['why'])})")

    def _recal_check(self, ia: int, i: int) -> Optional[Tuple[str, float, Optional[float], Optional[float]]]:
        """The re-calibration revisit after the trial at index ``i``, decided but not acted
        on: None, ``("capped", S/N, cap, None)`` (a note only) or ``("reset", S/N, cap,
        new contrast)``."""
        c = self.cfg
        cc = c.calibration
        if not (not self._extend_mode and i + 1 == cc.recal_check and self._recal_done < cc.recal_budget
                and cc.recal_budget > 0
                and not (cc.forced and len(cc.forced) > ia and cc.forced[ia] and cc.forced[ia] > 0)):
            return None
        yv = self.history.y[1:cc.recal_check] if c.seed_default else self.history.y[:cc.recal_check]
        yv = yv[np.isfinite(yv)]
        if yv.size < 3:
            return None
        top = np.sort(yv)[::-1][:cc.recal_ntop]
        mwu = float(np.median(top))
        if cc.target[0] <= mwu <= cc.target[1]:
            return None
        # the cap is the smaller of the two: `ceiling` (IDL cmax_cal, off by default) and
        # `max_contrast` (the physical one, 0.1 by default)
        caps = [v for v in (cc.ceiling, cc.max_contrast) if v is not None]
        cap = min(caps) if caps else None
        if mwu < cc.target[0] and cap is not None and self.contrast >= 0.999 * cap:
            return ("capped", mwu, cap, None)
        fac = float(np.clip(cc.aim / max(mwu, 0.1), cc.step_clip[0], cc.step_clip[1]))
        newc = self.contrast * fac
        if cap is not None:
            newc = min(newc, cap)
        return ("reset", mwu, cap, newc)

    def _recal_apply(self, ia: int, rc) -> bool:
        """Act on :meth:`_recal_check`: log the note, or rescale the contrast and restart the
        annulus (True)."""
        kind, mwu, cap, newc = rc
        cc = self.cfg.calibration
        if kind == "capped":
            self.log(f"  re-cal: median top-{cc.recal_ntop} S/N {mwu:.2f} below {cc.target}, but the "
                     f"contrast is already at its cap {cap:.2e} -- a source brighter than that is "
                     f"not a calibration; continuing at {self.contrast:.2e}")
            return False
        self.log(f"  re-cal: median top-{cc.recal_ntop} S/N {mwu:.2f} outside "
                 f"{cc.target}; contrast {self.contrast:.2e} -> {newc:.2e}; restarting annulus")
        self.contrast = newc
        self._recal_done += 1
        self._reset_history(ia)
        return True

    # -- pipelined batches (RunConfig.batch_mode "pipeline") ----------------------
    def _pipelined(self) -> bool:
        """Whether this annulus search keeps ``batch`` trials in flight
        (:meth:`_search_pipeline`).  Not in the k-scan modes, where each draw picks its own k
        while it is evaluated, and so draws positions mid-trial."""
        c = self.cfg
        return (int(c.batch) > 1 and str(getattr(c, "batch_mode", "sync")).lower() == "pipeline"
                and self.SHARE_CLEAN and not self._kscan_active())

    def _search_pipeline(self, ia: int, n_iter: int, opt) -> bool:
        """The annulus search with ``batch`` trials always in flight.

        The first ``batch`` proposals are asked together.  From then on, each time the
        oldest trial is scored it enters the history, the next proposal is asked from that
        history (the pending trials count toward the warm-up and the grid, but cannot train
        the model), its positions are drawn and its reductions go out, and only then are the
        finished trial's files, display, log line and checkpoint done.  Every generator call
        happens on this thread in index order, so a seed gives one run whatever the machine
        and however the reductions finish.  The checkpoint records the trials in flight, and
        a resume reduces them again before it asks for anything new: the resumed run is the
        run that never stopped.  A re-calibration drops the trials in flight and puts the
        generator back where it was after the trial that triggered it.  Returns True when a
        re-calibration restarted the annulus."""
        c = self.cfg
        b = int(c.batch)
        n = max(int(c.n_remeasure), 1)
        per_trial = n + (1 if self._needs_clean() else 0)
        q = _ReductionQueue(b * per_trial if self._parallel_reducer() else 1)
        if not q.serial:
            self._start_workers()
        pend: collections.deque = collections.deque()
        self._inflight = pend
        on_draw = (lambda j, nd, r, i_, c_: self._on_draw(ia, j, nd, r, i_, c_)) if n > 1 else None

        def dispatch(item):
            item["t0"], item["back"] = time.time(), []
            for fn in self._timed(self._trial_calls(item["cfg"], item["draws"], item["tag"]), item["back"]):
                q.submit(fn)
            pend.append(item)

        def ask(k_new):
            for pr in opt.ask(self._ask_history(pend), self.rng, k_new):
                phase = pr.flags.get("phase", "tpe")
                x = self._project(pr.x, is_random=phase in ("warmup", "explore", "random"))
                cfg = self.space.decode(x)
                draws = [self._draw_sources(cfg, self.contrast) for _ in range(n)]
                dispatch({"phase": phase, "x": x, "cfg": cfg, "draws": draws,
                          "tag": f"a{ia+1}_e{len(self.history) + len(pend) + 1}",
                          "rng_after": self.rng.bit_generator.state})

        try:
            for item in self._take_resume_pending():
                dispatch(item)
            if not pend and len(self.history) < n_iter:
                self.log(f"[ann {ia+1}] {b} trials in flight: the next goes out as soon as the oldest is in")
                ask(min(b, n_iter - len(self.history)))
            t_loop = time.time()
            while pend:
                item = pend[0]
                self._hb_eval(ia, len(self.history) + 1, n_iter, item["phase"])
                rec, inj, clean = self._trial_finish(item["x"], item["cfg"], item["phase"], item["tag"],
                                                     item["draws"], q, on_draw, item["t0"], combine=n > 1,
                                                     back=item["back"])
                pend.popleft()
                i = self._append(rec)
                rc = self._recal_check(ia, i)
                if rc is not None and rc[0] == "reset":
                    pend.clear()
                    q.close()                       # the dropped trials' reductions that have not started
                    self.rng.bit_generator.state = item["rng_after"]
                    self._eval_bookkeeping(ia, n_iter, i, item["phase"], rec, inj, clean, t_loop)
                    return self._recal_apply(ia, rc)
                if len(self.history) + len(pend) < n_iter:
                    ask(1)                          # the pool keeps working through what follows
                self._eval_bookkeeping(ia, n_iter, i, item["phase"], rec, inj, clean, t_loop)
                if rc is not None:
                    self._recal_apply(ia, rc)
                t_loop = time.time()
            return False
        finally:
            self._inflight = None
            q.close()

    def _ask_history(self, pend) -> History:
        """The history the optimizer proposes from: the trials scored so far, then the
        pending ones as unscored rows.  They advance the warm-up count and the grid, and
        cannot train the model."""
        if not pend:
            return self.history
        h = History(self.history.ndim, self.history.X.copy(), self.history.y.copy(), list(self.history.flags))
        for it in pend:
            h.append(it["x"], np.nan, {"phase": it["phase"], "pending": True})
        return h

    def _pending_state(self) -> Optional[List[Dict[str, Any]]]:
        """The trials in flight, as the checkpoint stores them (JSON)."""
        pend = getattr(self, "_inflight", None)
        if not pend:
            return None
        return [{"phase": it["phase"], "x": [float(v) for v in it["x"]], "tag": it["tag"],
                 "draws": [[list(s.as_tuple()) for s in d] for d in it["draws"]],
                 "rng_after": it["rng_after"]} for it in pend]

    def _take_resume_pending(self) -> List[Dict[str, Any]]:
        """The trials a checkpoint had in flight, rebuilt for :meth:`_search_pipeline` (once).
        They must continue this annulus' history: the first is its next evaluation."""
        raw, self._resume_pending = getattr(self, "_resume_pending", None), None
        nxt = f"a{self.ia + 1}_e{len(self.history) + 1}"
        if raw and raw[0].get("tag") != nxt:
            self.log(f"  (checkpoint's trials in flight start at {raw[0].get('tag')}, not {nxt}: asked again)")
            raw = None
        items = []
        for d in raw or []:
            x = np.asarray(d["x"], float)
            items.append({"phase": d["phase"], "x": x, "cfg": self.space.decode(x), "tag": d["tag"],
                          "draws": [[Source(*s) for s in dr] for dr in d["draws"]],
                          "rng_after": d["rng_after"]})
        return items

    #: a stage is worth a log line once it has run this many times the recent median
    #: evaluation -- generous, because the honest spread of an evaluation on a shared
    #: machine is already wide (6 s to 479 s within one RX J0534 run)
    STALL_FACTOR = 8.0
    STALL_FLOOR = 300.0

    def _hb_eval(self, ia: int, ev: int, n_iter: int, phase: str) -> None:
        """Name the evaluation about to start in ``heartbeat.json``, and set the duration
        past which it is worth saying in the log that this one is taking a while.

        The threshold comes from this run's own recent evaluations: a benchmark evaluates in
        seconds and a four-night stack in minutes, so any fixed number is either silent when
        it matters or crying wolf.  It is a note, never an abort -- the run is checkpointed
        after every evaluation and the reduction may simply be slow.
        """
        try:
            w = [float(f["wall"]) for f in (self.history.flags[-20:] if self.history else [])
                 if isinstance(f, dict) and isinstance(f.get("wall"), (int, float))
                 and np.isfinite(f["wall"]) and f["wall"] > 0]
            med = float(np.median(w)) if w else None
            self.hb.stall_after(max(self.STALL_FLOOR, self.STALL_FACTOR * med) if med else None)
            self.hb.stage(f"annulus {ia + 1} eval {ev}/{n_iter} ({phase})", write=False,
                          annulus=ia, eval=ev, n_eval=self.records_count)
        except Exception:
            pass

    def _reset_history(self, ia: int) -> None:
        self.history = History(self.space.ndim)
        if self.cfg.seed_default:
            x0 = self._project(self._default_vector(self._k_default), is_random=False)
            rec, inj, clean = self.evaluate_mean(
                x0, "seed", tag=f"a{ia+1}_seed",
                on_draw=lambda j, nd, r, i_, c_: self._on_draw(ia, j, nd, r, i_, c_))
            self._append(rec)
            self._tally_draws(ia, rec)
            self._set_best_images(rec, inj, clean)
            self.checkpoint()          # not routine: a new segment (annulus start, re-calibration)
            self._save_eval_images(rec, inj, clean)
            self._emit("on_eval", rec, inj, clean, True)

    # ------------------------------------------------------------- validation
    def validate(self, ia: int) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """Re-score the top ``n_top`` distinct configs on ``n_valid`` fresh injection
        sets with the RAW metric, elect the winner on the validated median, then score the
        winner on ``n_rescore`` fresh draws for its reported median (:meth:`_rescore_winner`)
        when two or more candidates had one.  Distinct means distinct reductions
        (:meth:`_validation_candidates`)."""
        vc = self.cfg.validation
        self.hb.stage(f"annulus {ia + 1} validation", annulus=ia)
        # The search's state, written now rather than held by the routine throttle: the
        # trial records carry the RNG from here on, and they pair with this history.
        self.checkpoint()
        cands, cleans, dups = self._validation_candidates(ia, vc)
        self._val_dups[int(ia)] = [list(map(int, d)) for d in dups]
        table: List[Dict[str, Any]] = []
        vbest, vwin = -np.inf, -1
        winner: Dict[str, Any] = {}
        n_final = int(vc.n_final or 0)
        entrants: List[Tuple[float, int, Dict[str, Any]]] = []     # the final round's: (median, ci, cand)

        def enter(vscore: float, ci: int, cand: Dict[str, Any]) -> None:
            if n_final > 1 and np.isfinite(vscore):
                entrants.append((float(vscore), int(ci), cand))
                entrants.sort(key=lambda t: (-t[0], t[1]))
                del entrants[n_final:]
        # --- per-candidate resume: candidates already validated in an interrupted pass are
        #     reloaded (row from validation.json, images from val_candNN.pkl) instead of
        #     re-reduced.  Each finished candidate also refreshes checkpoint.json (RNG state),
        #     so a resumed validation continues exactly where the interrupted one stopped.
        done: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = self._load_validation_partial(ia, cands)
        for ci, e in enumerate(cands):
            x = self.history.X[e]
            cfg = self.space.decode(x)
            k_hist = self.history.flags[e].get("k")
            cfg = self._apply_k_flag(cfg, k_hist)
            if ci in done:
                row, cand = done[ci]
                table.append(row)
                vscore = row["validated_score"] if row["validated_score"] is not None else np.nan
                self.log(f"  validation cand {ci+1}/{len(cands)} (eval {e+1}) -> validated {vscore:.3f}  [resumed]")
                full = dict(cand, x=np.asarray(row["x"], float), config=cfg)
                if np.isfinite(vscore) and vscore > vbest:
                    vbest, vwin = vscore, ci
                    winner = full
                enter(vscore, ci, full)
                continue
            trials, trial_imgs, trial_src, trial_ps, samples_r, samples_s = [], [], [], [], [], []
            vwhy: Dict[int, str] = {}           # trial -> why it has no score (this session's trials)
            clean = None
            rng_before = self.rng.bit_generator.state   # before this candidate's draws: see _load_validation_partial
            part = self._load_validation_trials(ia, ci, cands)
            calls: List[Callable[[], Any]] = []
            if part is not None:       # interrupted inside this candidate: continue after its last trial
                clean = part["clean"]
                trials, trial_imgs, trial_src, trial_ps = part["trials"], part["trial_imgs"], part["trial_src"], part["trial_ps"]
                samples_r, samples_s = part["samples_r"], part["samples_s"]
                self.log(f"  validation cand {ci+1}/{len(cands)}: resuming after trial {len(trials)}/{vc.n_valid}")
            elif cleans[ci] is not None:   # reduced when the candidates were picked
                clean = cleans[ci]
            else:
                calls.append(lambda cfg=cfg, ci=ci: self._reduce(cfg, None, tag=f"a{ia+1}_val{ci}_clean"))
            # Every remaining trial's positions first, in order, by the same generator calls
            # the trial-by-trial loop made; each trial's record keeps the generator state
            # right after its own draw.  Then the clean image and the trials go out together,
            # a window at a time, and are scored in trial order: the same trials, sooner.
            plan = []
            for t in range(len(trials), vc.n_valid):
                rlo, rhi = self._band(ia, cfg)
                src = self.sampler.sample(self._nsrc(ia), rlo, rhi, self.rng, self.contrast)
                plan.append((t, src, self.rng.bit_generator.state))
                calls.append(lambda cfg=cfg, ci=ci, t=t, src=src: self._reduce(cfg, src, tag=f"a{ia+1}_val{ci}_t{t}"))
            with contextlib.closing(self._run_ordered(calls, self._reduction_window(len(cfg.selected)))) as stream:
                if part is None and clean is None:
                    _, res = next(stream)
                    if isinstance(res, BaseException):
                        self.log(f"  validation clean reduction failed: {res!r}")
                    else:
                        clean = res
                for t, src, rng_state in plan:
                    _, res = next(stream)
                    self.hb.stage(f"annulus {ia + 1} validation: candidate {ci + 1}/{len(cands)} "
                                  f"trial {t + 1}/{vc.n_valid}", write=False, annulus=ia)
                    try:
                        if isinstance(res, BaseException):
                            raise res
                        inj = res
                        r = self.objective.score_raw(inj.image, src, None if clean is None else clean.image,
                                                     **self._fm_for(inj, clean))
                        if not np.isfinite(r.score):
                            vwhy[t] = getattr(r, "why", None) or "score not finite"
                        trials.append(r.score)
                        trial_imgs.append(inj)
                        trial_src.append(src)
                        trial_ps.append([None if not np.isfinite(v) else float(v) for v in r.per_source])
                        for s, v in zip(src, r.per_source):
                            samples_r.append(s.rho)
                            samples_s.append(v)
                        self._emit("on_validation_trial", ia, ci, len(cands), e, t, vc.n_valid, inj, clean, list(src),
                                   r, list(trials))
                    except Exception as exc:
                        self.log(f"  validation trial failed: {exc!r}")
                        vwhy[t] = _error_why(exc)
                        trials.append(np.nan)
                        trial_imgs.append(None)
                        trial_src.append(src)
                        trial_ps.append([])
                    self._save_validation_trials(ia, ci, cands, clean, trials, trial_imgs, trial_src, trial_ps,
                                                 samples_r, samples_s, rng_state)
            vs = np.array(trials, float)
            vscore = nanmedian_even(vs)
            commit = -1
            if np.isfinite(vscore):
                fin = np.flatnonzero(np.isfinite(vs))
                commit = int(fin[np.argmin(np.abs(vs[fin] - vscore))])
            row = {"candidate": ci, "eval_index": e, "search_score": float(self.history.y[e]),
                   "validated_score": None if not np.isfinite(vscore) else float(vscore),
                   "trials": [None if not np.isfinite(v) else float(v) for v in vs],
                   "trial_sources": [[list(s.as_tuple()) for s in src] for src in trial_src],
                   "x": [float(v) for v in x], "config": cfg.to_dict(), "committed_trial": commit}
            nbad = int(np.count_nonzero(~np.isfinite(vs)))
            if nbad:
                # trials resumed from an interrupted validation carry no reason
                row["trial_why"] = [None if np.isfinite(v) else vwhy.get(t, "not recorded") for t, v in enumerate(vs)]
            table.append(row)
            self.log(f"  validation cand {ci+1}/{len(cands)} (eval {e+1}, search {self.history.y[e]:.3f}) "
                     f"-> validated {vscore:.3f}"
                     + (f"  ({nbad} of {vs.size} trials failed: "
                        f"{_whys(collections.Counter(w for w in row['trial_why'] if w))})" if nbad else ""))
            if self.cfg.write_setup_files:
                self._write_setup_file(os.path.join(self._ann_dir(ia), f"valid_cand{ci+1:02d}_setup.txt"), "valid",
                                       ia, e + 1, cfg, trial_src[commit] if commit >= 0 else [], self.contrast,
                                       vscore, k_hist if self.cfg.k_mode in SCAN_MODES else None,
                                       extra={"search_score": f"{self.history.y[e]:.3f}",
                                              "trials": " ".join(f"{v:.3f}" for v in vs)})
            cand = {"eval_index": e, "score": vscore,
                    "clean": None if clean is None else clean, "inj": trial_imgs[commit] if commit >= 0 else None,
                    "sources": trial_src[commit] if commit >= 0 else [],
                    "per_source": list(trial_ps[commit]) if commit >= 0 else [],
                    "k_default": int(self._k_default) if self._k_default is not None else None,
                    "samples_r": list(samples_r), "samples_s": list(samples_s)}
            full = dict(cand, x=x.copy(), config=cfg)
            if np.isfinite(vscore) and vscore > vbest:
                vbest, vwin = vscore, ci
                winner = full
            enter(vscore, ci, full)
            # durable before the checkpoint that pairs with it (_save_validation_partial)
            _atomic_write(os.path.join(self._ann_dir(ia), "validation.json"),
                          json.dumps(table, indent=1, default=_json_default))
            self._save_validation_partial(ia, ci, cands, cand, rng_before)
            self._vt_on_disk.pop((int(ia), int(ci)), None)
            try:
                os.remove(self._val_trials_pkl(ia, ci))
            except OSError:
                pass
        if len(entrants) > 1 and int(vc.n_final_draws or 0) > 0:
            winner, vwin = self._final_round(ia, entrants, int(vc.n_final_draws), winner, vwin)
        # An election among two or more medians picks the largest, so the winner is scored again
        # on fresh draws.  A lone candidate's median was not selected on: its draws are fresh.
        n_re = vc.n_valid if vc.n_rescore is None else int(vc.n_rescore)
        n_med = sum(1 for row in table if row.get("validated_score") is not None)
        if vwin >= 0 and n_re > 0 and n_med > 1:
            winner = self._rescore_winner(ia, winner, n_re)
        return table, vwin, winner

    def _validation_candidates(self, ia: int, vc: "ValidationConfig"):
        """The configurations validation re-scores, best search score first: ``(cands, cleans,
        dups)``, with each candidate's clean reduction (None where none was made) and
        ``[[eval, same_as_eval], ...]`` for the configurations skipped as copies.

        Two candidates must be ``distinct_tol`` apart as parameter vectors.  With
        ``vc.dedup == "image"`` they must also reduce differently: each configuration's clean
        image is made (the candidate's first reduction anyway), and one equal to an earlier
        candidate's, bit for bit, is skipped for the next in the order.  Parameters that cannot
        change a reduction set two vectors apart without changing what they reduce: a
        deselected night's, frame cuts with no frame between them, an even high-pass width
        (``klip.highpass`` runs it as the next odd one), k past the modes a target has.  Once
        the search has settled on the live parameters it samples those freely, so its top
        holds copies.  In NEAR2's run_20261027_222223 the 25 candidates of an annulus were 7
        to 20 distinct reductions, 74 of the run's 175 validation slots re-scored a reduction
        already in the list, and the election among copies was decided by draw noise.  The
        cost is one clean reduction per copy, and no draw is taken, so the run's generator
        stands where it did.  The look goes ``vc.dedup_depth x n_top`` deep."""
        order = [int(e) for e in self.history.order()]
        X = self.history.X

        def far(e: int, others: Sequence[int]) -> bool:
            return all(self.space.distinct(X[e], X[c], vc.distinct_tol) for c in others)

        if str(vc.dedup).lower() != "image":
            cands: List[int] = []
            for e in order:
                if far(e, cands):
                    cands.append(e)
                if len(cands) >= vc.n_top:
                    break
            return cands, [None] * len(cands), []
        pool = order[:max(int(vc.dedup_depth or 1), 1) * max(int(vc.n_top), 1)]
        cands, cleans, dups = [], [], []
        pos = examined = 0
        self.hb.stage(f"annulus {ia + 1} validation: distinct candidates", write=False, annulus=ia)
        while len(cands) < vc.n_top and pos < len(pool):
            batch: List[int] = []
            while pos < len(pool) and len(batch) < vc.n_top - len(cands):
                e = pool[pos]
                pos += 1
                if far(e, cands + batch):
                    batch.append(e)
            if not batch:
                break
            cfgs = [self._apply_k_flag(self.space.decode(X[e]), self.history.flags[e].get("k")) for e in batch]
            calls = [lambda cfg=cfg, e=e: self._reduce(cfg, None, tag=f"a{ia+1}_valpick_e{e+1}_clean")
                     for e, cfg in zip(batch, cfgs)]
            window = self._reduction_window(max(len(c.selected) for c in cfgs))
            with contextlib.closing(self._run_ordered(calls, window)) as stream:
                got = [res for _, res in stream]
            examined += len(batch)
            for e, res in zip(batch, got):
                if isinstance(res, BaseException):
                    self.log(f"  validation: the clean reduction of eval {e + 1} failed ({res!r}), so it is "
                             f"validated without the comparison")
                    res = None
                same = None if res is None else next(
                    (c for c, cl in zip(cands, cleans) if cl is not None and _same_reduction(cl, res)), None)
                if same is not None:
                    dups.append([e, same])
                    continue
                cands.append(e)
                cleans.append(res)
                if len(cands) >= vc.n_top:
                    break
        if dups:
            self.log(f"  validation: {len(dups)} of the {examined} configurations examined reduce as an earlier "
                     f"candidate does (parameters that do not change the reduction), so {len(cands)} distinct "
                     f"reductions are validated")
        if len(cands) < vc.n_top:
            self.log(f"  validation: {len(cands)} distinct reduction(s) among the first {len(pool)} of the search's "
                     f"order")
        return cands, cleans, dups

    def _final_round(self, ia: int, entrants: List[Tuple[float, int, Dict[str, Any]]], n_draws: int,
                     winner: Dict[str, Any], vwin: int) -> Tuple[Dict[str, Any], int]:
        """The election again among ``entrants``, the first round's best ``(median, ci, cand)``,
        on ``n_draws`` common draws.  Draw ``t`` injects the same sources into every entrant.

        A hard draw is hard for every configuration (the GPU session found the candidates'
        scores correlated at 0.92 to 1.00 across draws), so on common draws the differences
        between their medians scatter far less than between medians on draws of their own.
        The largest median here is elected, the better first-round median on a tie.  The
        draws come from :meth:`_side_rng`, so the run's generator stands where it did, and a
        resumed run, whose first round is reloaded, makes the same round again.  Returns the
        winner, with ``score`` its median here and ``final_round`` the round's record, and its
        first-round index.  When no trial scores, the first round's winner stands."""
        sampler = getattr(self.sampler, "base", None) or self.sampler     # a replaying run draws fresh here too
        n_src = self._nsrc(ia)
        plans: List[List[List[Source]]] = []
        calls: List[Callable[[], Any]] = []
        what: List[Tuple[int, Optional[int]]] = []
        cleans: List[Any] = []
        for k, (_, ci, cand) in enumerate(entrants):
            cfg = cand["config"]
            rlo, rhi = self._band(ia, cfg)
            # one generator per draw, made afresh for each entrant: the same sources wherever the
            # entrants' injection bands agree (they differ only under opt_width)
            srcs = [list(sampler.sample(n_src, rlo, rhi, self._side_rng("final", ia, t), self.contrast))
                    for t in range(n_draws)]
            plans.append(srcs)
            cleans.append(cand.get("clean"))
            if cleans[k] is None:
                calls.append(lambda cfg=cfg, ci=ci: self._reduce(cfg, None, tag=f"a{ia+1}_valfin{ci}_clean"))
                what.append((k, None))
            for t, src in enumerate(srcs):
                calls.append(lambda cfg=cfg, ci=ci, t=t, src=src: self._reduce(cfg, src, tag=f"a{ia+1}_valfin{ci}_t{t}"))
                what.append((k, t))
        scores = np.full((len(entrants), n_draws), np.nan)
        window = self._reduction_window(min(len(c["config"].selected) for _, _, c in entrants))
        self.hb.stage(f"annulus {ia + 1} validation: the best {len(entrants)} on {n_draws} common draws",
                      write=False, annulus=ia)
        with contextlib.closing(self._run_ordered(calls, window)) as stream:
            for (k, t), (_, res) in zip(what, stream):
                if t is None:
                    if isinstance(res, BaseException):
                        self.log(f"  validation final round: a clean reduction failed: {res!r}")
                    else:
                        cleans[k] = res
                    continue
                try:
                    if isinstance(res, BaseException):
                        raise res
                    clean = cleans[k]
                    r = self.objective.score_raw(res.image, plans[k][t], None if clean is None else clean.image,
                                                 **self._fm_for(res, clean))
                    scores[k, t] = float(r.score)
                except Exception as exc:          # noqa: BLE001
                    self.log(f"  validation final round: a trial failed: {exc!r}")
                if t == n_draws - 1:
                    vs1, ci, cand = entrants[k]
                    self.log(f"  validation final round {k + 1}/{len(entrants)} (cand {ci + 1}, eval "
                             f"{int(cand['eval_index']) + 1}, first round {vs1:.3f}) -> "
                             f"{nanmedian_even(scores[k]):.3f} on {n_draws} common draws")
        meds = [nanmedian_even(row) for row in scores]
        rows = [{"candidate": int(ci), "eval_index": int(cand["eval_index"]), "validated_score": float(vs1),
                 "final_score": None if not np.isfinite(m) else float(m),
                 "trials": [None if not np.isfinite(v) else float(v) for v in sc]}
                for (vs1, ci, cand), m, sc in zip(entrants, meds, scores)]
        fin = [k for k, m in enumerate(meds) if np.isfinite(m)]
        if not fin:
            self.log("  validation final round: no trial scored, so the first round's winner stands")
            return winner, vwin
        kb = max(fin, key=lambda k: (meds[k], -k))
        vs1, ci, cand = entrants[kb]
        info = {"n_draws": int(n_draws), "rows": rows, "winner_candidate": int(ci), "first_round_winner": int(vwin)}
        _atomic_write(os.path.join(self._ann_dir(ia), "validation_final.json"),
                      json.dumps(dict(info, trial_sources=[[[list(s_.as_tuple()) for s_ in src] for src in srcs]
                                                           for srcs in plans]), indent=1, default=_json_default))
        e1 = int(entrants[0][2]["eval_index"])
        self.log(f"  validation final round: eval {int(cand['eval_index']) + 1} elected at {meds[kb]:.3f} on "
                 f"{n_draws} common draws"
                 + ("" if int(ci) == int(vwin) else
                    f" (the first round's largest median was eval {e1 + 1}'s, {entrants[0][0]:.3f})"))
        new = dict(cand, score=float(meds[kb]), final_round=info)
        if new.get("clean") is None and cleans[kb] is not None:
            new["clean"] = cleans[kb]
        return new, int(ci)

    def _side_rng(self, *tag) -> np.random.Generator:
        """A generator for draws that must leave the run's own where it stands, seeded from that
        one's state and ``tag``.  A resumed run, whose generator is restored, draws the same,
        and every later draw of the run is the one it would have been without these."""
        import hashlib
        st = json.dumps(self.rng.bit_generator.state, sort_keys=True,
                        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        h = hashlib.sha256((st + "|" + "|".join(str(t) for t in tag)).encode()).digest()
        return np.random.default_rng(np.frombuffer(h, dtype=np.uint32))

    def _rescore_winner(self, ia: int, winner: Dict[str, Any], n: int) -> Dict[str, Any]:
        """The elected candidate on ``n`` fresh draws, whose median becomes its validated score.

        The election takes the largest of the candidates' medians, and a median of ``n_valid``
        draws scatters by a few tenths of S/N about its configuration's own, while the top
        candidates' own medians lie within a few tenths of each other.  So the largest is
        largely the luckiest, and the winner's median on the draws it was elected on is high:
        by 0.22 S/N on average over the seven annuli of NEAR2's run_20261027_222223 and up to
        0.94 in run_20261024_153342, against 200 common draws on the GPU (the medians of the
        candidates not elected were unbiased, +0.005).  Fresh draws are not selected on.  Their
        median is ``winner_score``, the trial nearest it is the committed one (the products,
        the verify's injections) and their per-source S/N give the contrast curve.  The
        election's median stays as ``election_score``.  The draws come from :meth:`_side_rng`,
        so the run's generator stands where it did.  No trial scored: the election's median
        stands."""
        cfg: Config = winner["config"]
        clean = winner.get("clean")
        rng = self._side_rng("rescore", ia, int(winner.get("eval_index", -1)))
        sampler = getattr(self.sampler, "base", None) or self.sampler     # a replaying run draws fresh here too
        plan = []
        for _ in range(int(n)):
            rlo, rhi = self._band(ia, cfg)
            plan.append(sampler.sample(self._nsrc(ia), rlo, rhi, rng, self.contrast))
        calls: List[Callable[[], Any]] = []
        if clean is None:
            calls.append(lambda: self._reduce(cfg, None, tag=f"a{ia+1}_valwin_clean"))
        calls += [lambda t=t, src=src: self._reduce(cfg, src, tag=f"a{ia+1}_valwin_t{t}") for t, src in enumerate(plan)]
        trials: List[float] = []
        imgs: List[Any] = []
        ps: List[List[Optional[float]]] = []
        samples_r: List[float] = []
        samples_s: List[float] = []
        self.hb.stage(f"annulus {ia + 1} validation: the winner on {n} fresh draws", write=False, annulus=ia)
        with contextlib.closing(self._run_ordered(calls, self._reduction_window(len(cfg.selected)))) as stream:
            if clean is None:
                _, res = next(stream)
                clean = None if isinstance(res, BaseException) else res
            for src in plan:
                _, res = next(stream)
                try:
                    if isinstance(res, BaseException):
                        raise res
                    r = self.objective.score_raw(res.image, src, None if clean is None else clean.image,
                                                 **self._fm_for(res, clean))
                    trials.append(float(r.score))
                    imgs.append(res)
                    ps.append([None if not np.isfinite(v) else float(v) for v in r.per_source])
                    for s_, v in zip(src, r.per_source):
                        samples_r.append(s_.rho)
                        samples_s.append(v)
                except Exception as exc:          # noqa: BLE001
                    self.log(f"  validation: a fresh trial of the winner failed: {exc!r}")
                    trials.append(np.nan)
                    imgs.append(None)
                    ps.append([])
        vs = np.array(trials, float)
        med = nanmedian_even(vs) if vs.size else np.nan
        old = float(winner["score"])
        if not np.isfinite(med):
            self.log(f"  validation: no fresh trial of the winner scored, so its election median {old:.3f} stands")
            return winner
        fin = np.flatnonzero(np.isfinite(vs))
        commit = int(fin[np.argmin(np.abs(vs[fin] - med))])
        self.log(f"  validation: the winner (eval {int(winner.get('eval_index', -1)) + 1}) on {len(plan)} fresh "
                 f"draws: median {med:.3f} (elected at {old:.3f} on its own draws)")
        return dict(winner, election_score=old, score=float(med),
                    rescore_trials=[None if not np.isfinite(v) else float(v) for v in vs],
                    rescore_sources=[[list(s_.as_tuple()) for s_ in src] for src in plan], rescore_committed=commit,
                    inj=imgs[commit], sources=list(plan[commit]), per_source=list(ps[commit]),
                    samples_r=samples_r, samples_s=samples_s,
                    clean=clean if clean is not None else winner.get("clean"))

    # validation resume helpers -------------------------------------------------
    def _val_pkl(self, ia: int, ci: int) -> str:
        return os.path.join(self._ann_dir(ia), f"val_cand{ci+1:02d}.pkl")

    def _save_validation_partial(self, ia: int, ci: int, cands: List[int], cand: Dict[str, Any],
                                 rng_before: Optional[Dict[str, Any]] = None) -> None:
        """The finished candidate, durably, then the checkpoint (RNG state after this
        candidate's fresh injection draws).  The pickle carries the RNG state from before
        and after those draws as well (:meth:`_load_validation_partial`)."""
        try:
            _atomic_write(self._val_pkl(ia, ci),
                          pickle.dumps({"cands": list(cands), "cand": cand, "rng_before": rng_before,
                                        "rng": self.rng.bit_generator.state}, protocol=pickle.HIGHEST_PROTOCOL))
            self.checkpoint()
        except Exception as exc:       # scratch only -- never fatal
            self.log(f"  (validation checkpoint not written: {exc!r})")

    def _val_trials_pkl(self, ia: int, ci: int) -> str:
        return os.path.join(self._ann_dir(ia), f"val_cand{ci+1:02d}_trials.pkl")

    #: the per-trial lists a validation candidate accumulates, in the order they are kept
    _VT_LISTS = ("trials", "trial_imgs", "trial_src", "trial_ps", "samples_r", "samples_s")

    def _save_validation_trials(self, ia, ci, cands, clean, trials, trial_imgs, trial_src, trial_ps, samples_r,
                                samples_s, rng_state: Optional[Dict[str, Any]] = None) -> None:
        """Intra-candidate checkpoint (one validation trial can take minutes).

        ``val_candNN_trials.pkl`` is an append-only stream of pickles: a header with the
        candidate list and the clean reduction, written once, then one record per finished
        trial holding that trial's entries and the RNG state after its draw (``rng_state``,
        taken when the positions were drawn; validation draws every trial's positions before
        it reduces any).  Until
        2026-10-08 the file was rewritten whole after every trial -- the clean image and
        every trial image so far, ~1,300 image sets per candidate at 50 trials -- and
        checkpoint.json with it, the whole search history once per trial (found by the
        NEAR2 / IDL session).  The RNG state in the records is what checkpoint.json was
        rewritten for; :meth:`_load_validation_trials` restores it.  A write that fails
        marks the stream for a full rewrite at the next trial, so an error cannot leave a
        gap in it.
        """
        path = self._val_trials_pkl(ia, ci)
        key = (int(ia), int(ci))
        lists = dict(zip(self._VT_LISTS, (trials, trial_imgs, trial_src, trial_ps, samples_r, samples_s)))
        on_disk = self._vt_on_disk.get(key)          # {list name: entries in the file}, or None
        try:
            if on_disk is None or any(on_disk[k] > len(lists[k]) for k in self._VT_LISTS):
                with open(path, "wb") as f:
                    pickle.dump({"format": 2, "cands": list(cands), "clean": clean}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                on_disk = {k: 0 for k in self._VT_LISTS}
            rec = {k: list(lists[k][on_disk[k]:]) for k in self._VT_LISTS}
            rec["rng"] = self.rng.bit_generator.state if rng_state is None else rng_state
            with open(path, "ab") as f:
                pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._vt_on_disk[key] = {k: len(lists[k]) for k in self._VT_LISTS}
        except Exception as exc:
            self._vt_on_disk[key] = None
            self.log(f"  (validation trial checkpoint not written: {exc!r})")

    def _load_validation_trials(self, ia: int, ci: int, cands: List[int]) -> Optional[Dict[str, Any]]:
        """The trials an interrupted pass finished for this candidate, or None.

        Reads the record stream :meth:`_save_validation_trials` writes, and the single
        pickle it wrote before 2026-10-08 (that whole state, possibly followed by records
        appended after a resume).  A record cut short by the interruption is dropped and
        cut off the file, so the next trial appends after the last complete one.  The RNG
        is set to the state after the last trial kept: the search generator then stands
        where it stood when that trial finished.
        """
        if not getattr(self, "_resumed", False):
            return None
        pk = self._val_trials_pkl(ia, ci)
        if not os.path.exists(pk):
            return None
        d: Optional[Dict[str, Any]] = None
        rng_state = None
        good_end = 0
        try:
            with open(pk, "rb") as f:
                while True:
                    try:
                        rec = pickle.load(f)
                    except EOFError:
                        break
                    except Exception:
                        if d is None:
                            return None         # not even the header: start the candidate over
                        break                   # a record cut short: keep what came before it
                    if d is None:               # header (or the old whole-state pickle)
                        d = {"cands": list(rec.get("cands", [])), "clean": rec.get("clean")}
                        for k in self._VT_LISTS:
                            d[k] = list(rec.get(k, []))
                    else:
                        for k in self._VT_LISTS:
                            d[k].extend(rec.get(k, []))
                        rng_state = rec.get("rng", rng_state)
                    good_end = f.tell()
            if os.path.getsize(pk) > good_end:
                os.truncate(pk, good_end)
        except Exception:
            return None
        if d is None or list(d["cands"]) != list(cands) or not d["trials"]:
            return None
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self._vt_on_disk[(int(ia), int(ci))] = {k: len(d[k]) for k in self._VT_LISTS}
        return d

    def _load_validation_partial(self, ia: int, cands: List[int]) -> Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """The candidates an interrupted pass finished, ``{ci: (row, cand)}``.

        The checkpoint written after a candidate holds the RNG state after its draws, and
        a resume normally starts from it.  A resume from the save before it
        (``checkpoint.prev.json``, after a crash while that checkpoint was being written)
        holds the state from before them, which would draw the same positions again for the
        next candidate.  The pickles record both states.  When the RNG stands where one of
        the loaded candidates began, it moves on to where the last of them ended."""
        out: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        if not getattr(self, "_resumed", False):
            return out
        vpath = os.path.join(self._ann_dir(ia), "validation.json")
        if not os.path.exists(vpath):
            return out
        try:
            with open(vpath) as f:
                rows = json.load(f)
        except Exception:
            return out
        now, began, ended = self.rng.bit_generator.state, False, None
        for ci, e in enumerate(cands):
            pk = self._val_pkl(ia, ci)
            if ci >= len(rows) or not os.path.exists(pk):
                break                  # candidates are validated in order; stop at the first gap
            row = rows[ci]
            try:
                with open(pk, "rb") as f:
                    d = pickle.load(f)
            except Exception:
                break
            if int(row.get("eval_index", -1)) != int(e) or list(d.get("cands", [])) != list(cands):
                break                  # a different candidate list (history changed) -> redo
            out[ci] = (row, d["cand"])
            began = began or (d.get("rng_before") is not None and d["rng_before"] == now)
            ended = d.get("rng")
        if began and ended is not None:
            self.rng.bit_generator.state = ended
        return out

    # ------------------------------------------------------------- annulus loop
    def _save_eval_images(self, rec: "EvalRecord", inj: Optional[EvalImages], clean: Optional[EvalImages]) -> None:
        """IDL ``evalNNNN_score_inj.fits.gz``: compressed float32 crops of this eval's
        injected and clean images (box = outer radius + 1.5 FWHM) so the step display
        can be rebuilt with the real per-eval images (``display.render_steps``)."""
        if not self.cfg.save_eval_images or rec.meta.get("failed"):
            return
        try:
            from astropy.io import fits
            d = os.path.join(self._ann_dir(rec.annulus), "evals")
            os.makedirs(d, exist_ok=True)
            _, outrad = self._zone(rec.annulus, self._config_from_dict(rec.config))
            for tag, ev in (("inj", inj), ("clean", clean)):
                img = None if ev is None else ev.image
                if img is None or np.ndim(img) != 2:
                    continue
                ny, nx = img.shape
                cx, cy = star_center(img.shape)
                h = int(np.ceil(outrad + 1.5 * self.fwhm)) + 1
                x0, x1 = max(int(np.floor(cx)) - h, 0), min(int(np.ceil(cx)) + h + 1, nx)
                y0, y1 = max(int(np.floor(cy)) - h, 0), min(int(np.ceil(cy)) + h + 1, ny)
                hdr = fits.Header()
                hdr["CROPX0"], hdr["CROPY0"], hdr["FULLNX"], hdr["FULLNY"] = x0, y0, nx, ny
                hdr["EVAL"], hdr["ANNULUS"], hdr["IMGTYPE"] = rec.index + 1, rec.annulus + 1, f"eval {tag} image crop"
                fits.PrimaryHDU(np.asarray(img[y0:y1, x0:x1], np.float32), header=hdr).writeto(
                    os.path.join(d, f"eval{rec.index + 1:04d}_{tag}.fits.gz"), overwrite=True)
        except Exception as exc:  # never fail a run over a display product
            self.log(f"[warn] eval image save failed: {exc!r}")

    def _ann_dir(self, ia: int) -> str:
        d = os.path.join(self.run_dir, f"annulus{ia+1:02d}")
        os.makedirs(d, exist_ok=True)
        return d

    def run_annulus(self, ia: int, resume_index: Optional[int] = None,
                    extend: Optional[Dict[str, Any]] = None) -> AnnulusResult:
        """Calibrate, search, validate and write the products of annulus ``ia``.

        ``resume_index`` continues an interrupted search from the checkpointed history;
        ``extend`` (used by :meth:`extend`) is ``{'mode': 'validate' | 'continue',
        'history': History, 'contrast': float, 'k_default': int}``: the spliced history
        replaces calibration, the contrast is frozen, and in ``continue`` mode the prior
        best is re-scored first before the search continues to the (new) ``n_iter``."""
        self.ia = ia
        if resume_index is None:
            self._hooks_done.pop(ia, None)      # a fresh or extended pass re-runs its post-annulus hooks
        if resume_index is None or extend is not None:
            self._resume_pending = None         # trials in flight belong to the search that was stopped
        a_in, a_out = self._annulus(ia)
        cc = self.cfg.calibration
        if extend is not None:
            self.history = extend["history"]
            self.contrast = float(extend["contrast"])
            self._k_default = int(extend.get("k_default") or self.cfg.defaults.get("k_klip", 10) or 10)
            self._recal_done = 0
            self._best_images = {}
            self._extend_mode = True
            if extend["mode"] == "continue":
                n_iter = self.cfg.per_annulus(self.cfg.n_iter, ia)
                self.log(f"=== annulus {ia+1}: EXTEND {len(self.history)} -> {n_iter} evals "
                         f"(contrast frozen at {self.contrast:.3e}; prior best re-scored first)")
                bi, _ = self.history.best()
                if bi >= 0:
                    rec, inj, clean = self.evaluate(self.history.X[bi].copy(), "reseed", tag=f"a{ia+1}_reseed")
                    self._append(rec)
                    if inj.image is not None and inj.image.ndim == 2:
                        self._set_best_images(rec, inj, clean)
                    nb, nbs = self.history.best()
                    self.log(f"[ann {ia+1}] eval {len(self.history)}/{n_iter} reseed  "
                             f"score={rec.score if rec.score is None else round(rec.score,3)}"
                             f" raw={rec.raw_score if rec.raw_score is None else round(rec.raw_score,3)}"
                             f" best={nbs:.3f}@{nb+1}  (prior best eval {bi+1} re-scored)  {rec.wall_s:.0f}s")
                    self.checkpoint(routine=True)
                    self._save_eval_images(rec, inj, clean)
                    self._emit("on_eval", rec, inj, clean, True)
                self._search_annulus(ia, len(self.history))
            else:
                self.log(f"=== annulus {ia+1}: RE-VALIDATE {len(self.history)} existing evals in place "
                         f"(contrast {self.contrast:.3e})")
        elif resume_index is None:
            label = f"{ia+1}" if self.cfg.opt_width else f"{ia+1}/{self.cfg.nann}"
            n, asked = self._nsrc(ia), self._nsrc_asked(ia)
            how = "the rule" if asked is None else f"n_sources {asked}"
            self.log(f"=== annulus {label}: [{a_in:.1f}, {a_out:.1f}] px, {n} source{'s' * (n != 1)} ({how}), "
                     f"band {self._band(ia)[0]:.3f}-{self._band(ia)[1]:.3f}\"")
            if ia > 0 and not cc.warm_start_previous:
                self.contrast = float(self.cfg.contrast0)
            self._draw_tally.pop(ia, None)
            contrast, kdef, cinfo = self.calibrate(ia)
            self.contrast = contrast
            self._k_default = kdef
            self._recal_done = 0
            cpath = os.path.join(self._ann_dir(ia), "calibration.json")
            with open(cpath, "w") as f:
                json.dump(cinfo, f, indent=1, default=_json_default)
            self._durable_later(cpath)
            self._emit("on_calibration", cinfo, self._calib_images)
            self._reset_history(ia)
            self._search_annulus(ia, len(self.history))
        else:
            self._draw_tally[ia] = {"n": 0, "failed": 0, "why": collections.Counter(), "warned": False,
                                    "since_resume": True}
            self._restore_best_images(ia)
            self._search_annulus(ia, resume_index)
        return self._finish_annulus(ia)

    def _restore_best_images(self, ia: int) -> None:
        """Resume: the incumbent's images are not checkpointed, so reduce the best-so-far
        evaluation once more (same config, same injected positions, same contrast) and
        take that as the best images -- the live display's Best cells, the KLIP-FM
        preview and the fallback products need them.  One evaluation's cost."""
        if self._best_images or self.history is None or len(self.history) == 0:
            return
        bi, bs = self.history.best()
        if bi < 0 or not np.isfinite(bs):
            return
        fl = self.history.flags[bi] if bi < len(self.history.flags) else {}
        srcs = [Source(*s) for s in (fl.get("sources") or [])] or None
        contrast = self.contrast
        st = self.rng.bit_generator.state                  # the re-reduction must not move the search RNG
        try:
            self.log(f"[ann {ia+1}] resume: re-reducing the best-so-far eval {bi+1} (score {bs:.3f}) "
                     f"to restore its images")
            rec, inj, clean = self.evaluate(self.history.X[bi].copy(), str(fl.get("phase", "tpe")),
                                            contrast=contrast, sources=srcs, tag=f"a{ia+1}_e{bi+1}_rebest")
            rec.index = int(bi)
            if fl.get("k") is not None:
                rec.k_used = fl["k"]
            if inj.image is not None and inj.image.ndim == 2 and not rec.meta.get("failed"):
                self._set_best_images(rec, inj, clean)
                self.log(f"[ann {ia+1}] resume: best images restored (re-scored {rec.score if rec.score is None else round(rec.score, 3)}"
                         f" vs logged {bs:.3f}; the logged score stands)  {rec.wall_s:.0f}s")
        except Exception as exc:
            self.log(f"  resume: could not restore the best images: {exc!r}")
        finally:
            self.rng.bit_generator.state = st

    def _finish_annulus(self, ia: int) -> AnnulusResult:
        """Validation, products, edge commit (opt_width), checkpoint and the
        verification hooks of annulus ``ia`` (search already complete)."""
        bi, bs = self.history.best()
        # A search in which EVERY evaluation failed still reached this point quietly: the
        # winner index came back -1, no best_*.fits were written, and the first thing to
        # read one of those files was what finally raised -- a notebook cell, several
        # minutes and one confusing traceback later.  Tutorial 3's STPSF section spent all
        # 50 evaluations this way, injecting at 2.40" into a library built only out to
        # 2.00", and nothing said so.  Say it here, where it happened.
        ntot = len(self.history)
        nok = int(self.history.valid.sum()) if ntot else 0
        if ntot and nok == 0:
            # Logging it was not enough: the run went on to write final_results.json with
            # winner_index -1 and score -inf, the driver called it "done in 3 min", and its
            # resume logic then treated the directory as finished.  A search with no valid
            # evaluation is not a result -- stop, and say how to start over.
            raise RuntimeError(
                f"every one of the {ntot} evaluations of annulus {ia+1} FAILED -- there is "
                f"no winner and no best image.  The usual cause is an injection model that does not "
                f"span the annulus; the reason is on the 'evaluation failed:' lines above.  This run "
                f"directory now holds only failed evaluations: fix the cause, move {self.run_dir} aside "
                f"and start again (a resume would just find the same {ntot} failures and stop here).")
        elif ntot and nok < ntot:
            self.log(f"  note: {ntot - nok}/{ntot} evaluations failed")
        self._draw_note(ia)
        self.log(f"  search done: best {bs:.3f} at eval {bi+1}")
        self._memory_note()
        table, vwin, winner = ([], -1, {})
        if self.cfg.validation.n_valid >= 1 and self.cfg.validation.n_top >= 1:
            table, vwin, winner = self.validate(ia)
            self._emit("on_validation", table, winner)
            self._memory_note()
        validated = vwin >= 0
        if not validated:
            x = self.history.X[bi]
            cfg = self._apply_k_flag(self.space.decode(x), self.history.flags[bi].get("k"))
            winner = {"eval_index": bi, "score": bs, "x": x, "config": cfg,
                      "clean": None, "inj": None, "sources": [], "samples_r": [], "samples_s": [],
                      "per_source": [], "k_default": int(self._k_default) if self._k_default is not None else None}
            brec = self._best_images.get("record")
            if brec is not None and brec.index == bi:
                winner["per_source"] = list(brec.raw_per_source)
            if self._best_images.get("clean_images") is not None:
                winner["clean"] = self._best_images["clean_images"]
            elif self._best_images.get("clean") is not None:
                winner["clean"] = EvalImages(self._best_images["clean"])
            if self._best_images.get("inj_images") is not None:
                winner["inj"] = self._best_images["inj_images"]
                winner["sources"] = list(self._best_images.get("sources", []))
            elif self._best_images.get("inj") is not None:
                winner["inj"] = EvalImages(self._best_images["inj"])
        res = self._products(ia, winner, table, validated, bi, bs)
        while len(self.results) < ia:
            self.results.append(None)
        if len(self.results) > ia:
            self.results[ia] = res
        else:
            self.results.append(res)
        self._ann_histories[ia] = self.history
        if self.cfg.opt_width:
            e = list(self.cfg.ann_edges)
            if len(e) > ia + 1:
                e[ia + 1] = float(res.outrad)
            else:
                e.append(float(res.outrad))
            self.cfg.ann_edges = e
            w_lo = float(self.cfg.width_range[0]) if self.cfg.width_range else 0.0
            note = ""
            if e[ia + 1] >= self.cfg.r_cap - 0.5:
                note = "  (reached r_cap: done)"
            elif self.cfg.r_cap - e[ia + 1] < w_lo:
                note = f"  (remaining ring {self.cfg.r_cap - e[ia+1]:.1f} px < w_lo {w_lo:.0f}: done)"
            self.log(f"  committed annulus edge: [{e[ia]:.1f}, {e[ia+1]:.1f}] px" + note)
        self.checkpoint(final_annulus=True)
        # post-annulus products and verification hooks (all exception-guarded, each one
        # recorded in the checkpoint when it completes so a resumed run picks up at the
        # first unfinished hook instead of skipping or repeating them)
        self._run_hooks(ia, winner)
        # The display's books and panels for the annulus are drawn here, on this thread.  They
        # get a stage of their own, since under the last hook's name minutes of them read as a
        # candidate search in heartbeat.json and in the stall note.
        self.hb.stage(f"annulus {ia + 1} display books", annulus=ia)
        self._emit("on_annulus_done", res, winner)
        return res

    HOOK_ORDER = ("stitch", "verify", "param_verify", "candidates")

    def _hooks_wanted(self) -> List[str]:
        return ["stitch"] + (["verify"] if self.cfg.verify else []) + \
               (["param_verify"] if self.do_param_verify else []) + (["candidates"] if self.cfg.candidates else [])

    def hooks_pending(self, ia: int) -> List[str]:
        done = self._hooks_done.get(ia, [])
        return [h for h in self._hooks_wanted() if h not in done]

    def _run_hooks(self, ia: int, winner: Dict[str, Any]) -> None:
        fns = {"stitch": ("running stitch", self._running_stitch, (ia, True)),
               "verify": ("verify", self._verify_hook, (ia, winner)),
               "param_verify": ("param_verify", self._param_verify_hook, (ia, winner)),
               "candidates": ("candidate search", self._candidates_hook, (ia,))}
        # An evaluation's stall threshold says nothing about these.  Each takes as long as it
        # takes, as the final products do, and the next evaluation sets the threshold again.
        self.hb.stall_after(None)
        for name in self.hooks_pending(ia):
            what, fn, args = fns[name]
            self.hb.stage(f"annulus {ia + 1} {what}", annulus=ia)
            self._guard(what, fn, *args)
            self._hooks_done.setdefault(ia, []).append(name)
            self.checkpoint(final_annulus=self._last_final)

    def _winner_from_disk(self, ia: int) -> Optional[Dict[str, Any]]:
        """Rebuild the ``winner`` dict of a completed annulus from ``winner.json`` and the
        FITS products, for re-running the post-annulus hooks after an interruption."""
        from .stitch import read_fits
        res = self.results[ia] if ia < len(self.results) else None
        if res is None:
            return None
        d = self._ann_dir(ia)
        x = np.asarray(res.winner_x, float)
        cfg = self.space.decode(x)
        h = self.annulus_history(ia)
        if h is not None and 0 <= res.winner_index < len(h):
            cfg = self._apply_k_flag(cfg, h.flags[res.winner_index].get("k"))
        parts = list(res.partitions) or [str(s) for s in cfg.selected]

        def _ev(key):
            img = read_fits(os.path.join(d, f"best_{key}.fits"))
            if img is None:
                return None
            stack = read_fits(os.path.join(d, f"best_{key}_partitions.fits"))
            return EvalImages(np.asarray(img, np.float32), None if stack is None else np.asarray(stack, np.float32),
                              None, parts if stack is not None else None)
        return {"eval_index": int(res.winner_index), "score": float(res.winner_score), "x": x, "config": cfg,
                "clean": _ev("clean"), "inj": _ev("inj"),
                "sources": [Source(*t) for t in res.winner_sources],
                "per_source": list(res.per_source or []), "k_default": res.k_default,
                "samples_r": list(res.validation_samples.get("r_as", [])),
                "samples_s": list(res.validation_samples.get("snr", []))}

    # ----------------------------------------------------------------- products
    def _write_setup_file(self, path: str, phase: str, ia: int, step: int, cfg: Config,
                          sources: Sequence[Source], contrast: float, snr, k_used=None,
                          extra: Optional[Dict[str, Any]] = None, weights=None) -> None:
        from .stitch import write_setup_file
        a_in, a_out = self._zone(ia, cfg)
        write_setup_file(path, phase, ia, step, cfg.params, a_in, a_out, [str(s) for s in cfg.selected],
                         [s.as_tuple() if isinstance(s, Source) else tuple(s) for s in sources], contrast,
                         None if snr is None else (snr if isinstance(snr, float) else float(snr)),
                         per_partition={str(k): v for k, v in cfg.per_partition.items()} if cfg.per_partition else None,
                         k_used=k_used, weights=weights, extra=extra)

    def _fits_header(self, ia: int, cfg: Config, score: float, validated: bool, phase: str = "final",
                     sources: Sequence[Source] = (), zone: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        a_in, a_out = self._zone(ia, cfg) if zone is None else zone
        p = cfg.params
        h: Dict[str, Any] = {"ANNULUS": ia + 1, "INRAD": a_in, "OUTRAD": a_out, "SCORE": float(score),
                             "VALID": int(validated), "CONTRAST": float(self.contrast), "PHASE": phase,
                             "KKLIP": p.get("k_klip"), "BIN": p.get("bin"), "NANG": p.get("n_ang"),
                             "FILTER": p.get("filter"), "ANGSEP": p.get("angsep"), "ANGLEMAX": p.get("anglemax"),
                             "CORRTHR": p.get("corr_thresh"), "NOISEMAX": p.get("noise_max"),
                             "CORONMAX": p.get("coronoise_max"),
                             "NPART": len(cfg.selected), "PARTS": ",".join(str(s) for s in cfg.selected),
                             "METRIC": getattr(self.objective.metric, "name", "metric"),
                             "CLEANSUB": int(bool(getattr(self.objective, "clean_subtract", False))),
                             "KMODE": self.cfg.k_mode, "SEED": self.cfg.seed, "RUNDIR": os.path.basename(
                                 os.path.abspath(self.run_dir))}
        for j, s in enumerate(sources):
            h[f"INJRHO{j+1}"] = float(s.rho)
            h[f"INJPA{j+1}"] = float(s.theta)
            h[f"INJCON{j+1}"] = float(s.contrast)
        if sources:
            h["INJNSRC"] = len(sources)
        return h

    def _products(self, ia: int, winner: Dict[str, Any], table, validated: bool, bi: int, bs: float) -> AnnulusResult:
        from .products import contrast_curve
        d = self._ann_dir(ia)
        cfg: Config = winner["config"]
        a_in, a_out = self._zone(ia, cfg)
        curve = None
        clean_img = None if winner.get("clean") is None else winner["clean"].image
        inj_img = None if winner.get("inj") is None else winner["inj"].image
        if clean_img is not None and len(winner.get("samples_r", [])) >= 3:
            try:
                cc = contrast_curve(clean_img, winner["samples_r"], winner["samples_s"], self.contrast,
                                    self.fwhm, self.pxscale, max(a_in, self.fwhm), a_out,
                                    self.throughput_fn, self.cfg.kfit_order, known=self._known())
                curve = {k: [None if not np.isfinite(v) else float(v) for v in np.atleast_1d(v_)]
                         for k, v_ in cc.items() for v in [np.asarray(v_, float)]}
                self._durable_later(os.path.join(d, "contrast_curve.txt"))
                with open(os.path.join(d, "contrast_curve.txt"), "w") as f:
                    f.write(f"# annulus {ia+1}  injection-calibrated 5-sigma contrast  (contrast_inj={self.contrast:.3e})\n")
                    f.write("# sep_arcsec   contrast_5sig\n")
                    for r, c_ in zip(cc["r_as"], cc["curve"]):
                        if np.isfinite(c_):
                            f.write(f"{r:10.4f}   {c_:12.4e}\n")
            except Exception as exc:
                self.log(f"  contrast curve failed: {exc!r}")
        sources = list(winner.get("sources") or [])
        k_used = cfg.params.get("k_klip")
        if cfg.per_partition:
            k_used = {str(p): cfg.per_partition[p].get("k_klip") for p in cfg.selected}
        hdr = self._fits_header(ia, cfg, float(winner["score"]), validated, "final", zone=(a_in, a_out))
        hdr_inj = self._fits_header(ia, cfg, float(winner["score"]), validated, "final", sources, zone=(a_in, a_out))
        # KLIP-FM cross-check curve (A §9.1): one clean forward-model pass at the winner config
        fm_curve = None
        fm = None
        if self.cfg.fm_curve and self.do_fm:
            try:
                fm = self._fm_pass(cfg, (a_in, a_out), self.contrast, f"a{ia+1}_fm", clean_img=clean_img)
                if fm is None:
                    self.log("  KLIP-FM cross-check: reducer returned no FM image; skipped")
                else:
                    fm_curve = fm["curve"]
                    nfin = int(np.sum([c is not None and c > 0 for c in fm_curve.get("curve", [])]))
                    self.log(f"  KLIP-FM cross-check: {len(fm['sources'])} test sources, {nfin} curve points")
                    sec = self._fm_section(fm_curve, ia, self.contrast)
                    if sec:
                        with open(os.path.join(d, "contrast_curve.txt"), "a") as f:
                            f.write("\n".join(sec) + "\n")
                    if self.cfg.save_fits:
                        hfm = self._fits_header(ia, cfg, float(winner["score"]), validated, "fm", fm["sources"],
                                                zone=(a_in, a_out))
                        hfm["IMGTYPE"] = "KLIP-FM forward-model response (clean pass, test spiral)"
                        _write_fits(os.path.join(d, "best_fm.fits"), np.asarray(fm["fm_image"], np.float32), hfm)
                        self._durable_later(os.path.join(d, "best_fm.fits"))
                    if clean_img is None:
                        clean_img = fm["clean"]
                    winner["fm"] = fm
            except Exception as exc:
                self.log(f"  KLIP-FM cross-check failed: {exc!r}")
                fm_curve = None
        grid_curve = None
        if self.cfg.contrast_grid.enabled:
            try:
                grid_curve = self._contrast_grid(ia, cfg, (a_in, a_out), fm_curve, curve)
            except Exception as exc:          # noqa: BLE001 - a product, never fatal
                self.log(f"  contrast grid failed: {exc!r}")
        if self.cfg.save_fits:
            if clean_img is not None:
                _write_fits(os.path.join(d, "best_clean.fits"), clean_img, hdr)
                self._durable_later(os.path.join(d, "best_clean.fits"))
                self._best_images["clean"] = clean_img
            if inj_img is not None:
                _write_fits(os.path.join(d, "best_inj.fits"), inj_img, hdr_inj)
                self._durable_later(os.path.join(d, "best_inj.fits"))
                self._best_images["inj"] = inj_img
                # ``_best_images`` is a bundle: the image and the sources that were injected
                # into it belong together, and every consumer treats them that way.  This
                # picture is the winner's COMMITTED trial, whose injections are validation's
                # fresh draw -- not the search evaluation the bundle was built from -- so the
                # sources have to move with it.  Leaving them behind made the annulus-done
                # panel circle the search azimuths on the committed image: every circle
                # rotated off its blob by one azimuth step (19.5 deg, ~5 px at 0.4"), on the
                # frame that stays on screen and is the one a notebook shows at the end.
                self._best_images["sources"] = list(sources)
                self._best_images["per_source"] = list(winner.get("per_source") or [])
            for key, hh in (("clean", hdr), ("inj", hdr_inj)):
                ev = winner.get(key)
                if ev is not None and getattr(ev, "stack", None) is not None:
                    h2 = dict(hh)
                    for j, pid in enumerate(ev.partitions or []):
                        h2[f"NIGHT{j+1}"] = str(pid)
                    _write_fits(os.path.join(d, f"best_{key}_partitions.fits"), ev.stack, h2)
                    self._durable_later(os.path.join(d, f"best_{key}_partitions.fits"))
        # memory cache for the stitches / hooks
        ev_c, ev_i = winner.get("clean"), winner.get("inj")
        self._ann_images[ia] = {
            "clean": clean_img, "inj": inj_img, "zone": (a_in, a_out),
            "clean_stack": None if ev_c is None else getattr(ev_c, "stack", None),
            "inj_stack": None if ev_i is None else getattr(ev_i, "stack", None),
            "parts": [str(p) for p in (getattr(ev_c, "partitions", None) or cfg.selected)],
            "sources": [s.as_tuple() for s in sources], "contrast": self.contrast,
            "weights": None if ev_c is None else getattr(ev_c, "weights", None)}
        res = AnnulusResult(
            annulus=ia, inrad=a_in, outrad=a_out, contrast=self.contrast, search_best_index=int(bi),
            search_best_score=float(bs), validated=validated, winner_index=int(winner["eval_index"]),
            winner_score=float(winner["score"]), winner_x=[float(v) for v in winner["x"]],
            winner_config=cfg.to_dict(), validation_table=table,
            validation_samples={"r_as": [float(v) for v in winner.get("samples_r", [])],
                                "snr": [None if not np.isfinite(v) else float(v) for v in winner.get("samples_s", [])]},
            contrast_curve=curve, distance_to_bounds=self.space.distance_to_bounds(np.asarray(winner["x"])),
            n_evaluations=len(self.history), winner_sources=[s.as_tuple() for s in sources],
            partitions=[str(s) for s in cfg.selected], k_used=k_used, fm_curve=fm_curve,
            per_source=[None if (v is None or not np.isfinite(v)) else float(v) for v in (winner.get("per_source") or [])],
            k_default=winner.get("k_default"),
            election_score=None if winner.get("election_score") is None else float(winner["election_score"]),
            rescore_trials=winner.get("rescore_trials"), rescore_sources=winner.get("rescore_sources"),
            rescore_committed=winner.get("rescore_committed"),
            validation_duplicates=list(self._val_dups.get(int(ia), [])), final_round=winner.get("final_round"),
            grid_curve=grid_curve)
        # the products above and this record are what a resume reads back for a finished
        # annulus (_winner_from_disk, _annulus_images): on disk before the checkpoint that
        # marks the annulus done
        _atomic_write(os.path.join(d, "winner.json"), json.dumps(res.to_dict(), indent=1, default=_json_default))
        if self.cfg.write_setup_files:
            self._write_setup_file(os.path.join(d, "final_setup.txt"), "final", ia, int(winner["eval_index"]) + 1,
                                   cfg, sources, self.contrast, float(winner["score"]), k_used,
                                   extra={"validated": int(validated), "search_score": f"{bs:.3f}"},
                                   weights=None if ev_c is None else getattr(ev_c, "weights", None))
        self.log(f"  annulus {ia+1} winner: eval {res.winner_index+1}, "
                 f"{'validated' if validated else 'search'} score {res.winner_score:.3f}"
                 + (f" on fresh draws (elected at {res.election_score:.3f})" if res.election_score is not None else ""))
        return res

    def _annulus_images(self, ia: int) -> Optional[Dict[str, Any]]:
        """Winner images of a completed annulus: memory cache, else the FITS products on disk."""
        d = self._ann_images.get(ia)
        if d is not None and d.get("clean") is not None:
            return d
        from .stitch import read_fits
        ad = os.path.join(self.run_dir, f"annulus{ia+1:02d}")
        clean = read_fits(os.path.join(ad, "best_clean.fits"))
        if clean is None:
            return None
        res = self.results[ia] if ia < len(self.results) else None
        parts, sources, zone, contrast = [], [], None, None
        if res is not None:
            parts = list(res.partitions) or [str(s) for s in res.winner_config.get("selected", [])]
            sources = list(res.winner_sources)
            zone = (res.inrad, res.outrad)
            contrast = res.contrast
        else:
            try:
                w = json.load(open(os.path.join(ad, "winner.json")))
                parts = list(w.get("partitions") or w["winner_config"].get("selected", []))
                sources = list(w.get("winner_sources", []))
                zone = (w["inrad"], w["outrad"])
                contrast = w.get("contrast")
            except Exception:
                zone = self._annulus(ia)
        d = {"clean": clean, "inj": read_fits(os.path.join(ad, "best_inj.fits")), "zone": zone,
             "clean_stack": read_fits(os.path.join(ad, "best_clean_partitions.fits")),
             "inj_stack": read_fits(os.path.join(ad, "best_inj_partitions.fits")),
             "parts": [str(p) for p in parts], "sources": sources, "contrast": contrast, "weights": None}
        self._ann_images[ia] = d
        return d

    def _stitch_weights(self, images: Sequence[np.ndarray], zones: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Sensitivity weights of the final stitch (``products.stitch_annuli`` recipe)."""
        from .products import noise_profile
        n = len(images)
        w = np.ones(n)
        if self.cfg.legacy_stitch or n <= 1:
            return w
        for ia, (img, (rin, rout)) in enumerate(zip(images, zones)):
            sig, rp = noise_profile(img, self.fwhm, rin, rout)
            gi = (rp >= rin + self.fwhm) & (rp <= rout - self.fwhm) & np.isfinite(sig) & (sig > 0)
            if gi.sum() == 0:
                gi = np.isfinite(sig) & (sig > 0)
            sg = float(np.median(sig[gi])) if gi.any() else 1.0
            if not np.isfinite(sg) or sg <= 0:
                sg = 1.0
            w[ia] = 1.0 / sg ** 2
        return w / w.max()

    def _inj_xy(self, sources: Sequence[Sequence[float]], shape) -> List[Tuple[float, float]]:
        if not sources:
            return []
        cx, cy = star_center(shape)
        xs, ys = source_xy([s[0] for s in sources], [s[1] for s in sources], self.pxscale, cx, cy,
                           getattr(self.reducer, "angle_convention", "pa"))
        return list(zip(np.atleast_1d(xs), np.atleast_1d(ys)))

    @property
    def flatten_products(self) -> bool:
        """Whether the saved stitches carry :func:`radprof`.

        Follows the metric rather than being a knob of its own, so the FITS on disk is
        the image the run actually scored.  These products used to flatten
        unconditionally, which meant a run could be optimised on one image and shipped
        with another -- and the header said ``radprof-flattened`` either way.
        """
        return bool(getattr(getattr(self.objective, "metric", None), "flatten", False))

    def _maybe_flatten(self, img: np.ndarray) -> np.ndarray:
        return radprof(img) if self.flatten_products else np.asarray(img, float)

    @property
    def _flat_tag(self) -> str:
        return " (radprof-flattened)" if self.flatten_products else ""

    def _running_stitch(self, ia: int, at_end: bool = False) -> Optional[Dict[str, Any]]:
        """Equal-weight NaN-aware running stitch (A §9.1): tiles of the previous annuli's
        winners plus the current running best (or, ``at_end``, this annulus' winner).
        Writes ``klip_stitched_running[_inj|_snr|_snr_inj].fits`` (+ the per-partition
        ``klip_stitched_running_nights.fits.gz`` at annulus end) and returns the arrays."""
        if not self.cfg.save_fits:
            return None
        from .products import snr_map
        from .stitch import stitch_tiles, stitch_stacks
        cleans, injs, zones, stacks, istacks, parts, sources = [], [], [], [], [], [], []
        for ja in range(ia + (1 if at_end else 0)):
            dj = self._annulus_images(ja)
            if dj is None or dj.get("clean") is None:
                continue
            cleans.append(dj["clean"]); injs.append(dj.get("inj")); zones.append(tuple(dj["zone"]))
            stacks.append(dj.get("clean_stack")); istacks.append(dj.get("inj_stack")); parts.append(dj.get("parts"))
            sources.extend(dj.get("sources") or [])
        if not at_end and self._best_images.get("clean") is not None:
            cleans.append(self._best_images["clean"]); injs.append(self._best_images.get("inj"))
            zones.append(tuple(self._best_images.get("zone") or self._annulus(ia)))
            ci, ii = self._best_images.get("clean_images"), self._best_images.get("inj_images")
            stacks.append(None if ci is None else ci.stack); istacks.append(None if ii is None else ii.stack)
            parts.append(None if ci is None else [str(p) for p in (ci.partitions or [])])
            sources.extend([s.as_tuple() for s in self._best_images.get("sources", [])])
        if not cleans:
            return None
        neval = 0 if self.history is None else len(self.history)
        hdr = {"ANNULUS": ia + 1, "NEVAL": neval, "RUNNING": 1, "NANN": len(cleans), "PHASE": "final" if at_end else "eval",
               "IMGTYPE": f"running equal-weight stitch{self._flat_tag}", "EDGES": ",".join(f"{z[0]:.1f}" for z in zones)
               + f",{zones[-1][1]:.1f}"}
        pads = self._tile_pads(zones)
        st = self._maybe_flatten(stitch_tiles(cleans, zones, pads=pads))
        files = {"clean": os.path.join(self.run_dir, "klip_stitched_running.fits")}
        _write_fits(files["clean"], st, hdr)
        snr = snr_map(st, self.fwhm)
        files["snr"] = os.path.join(self.run_dir, "klip_stitched_running_snr.fits")
        _write_fits(files["snr"], snr, dict(hdr, IMGTYPE="running stitched S/N map (Mawet)"))
        out = {"clean": st, "snr": snr, "zones": zones}
        if any(i is not None for i in injs):
            sti = self._maybe_flatten(stitch_tiles([i if i is not None else c for i, c in zip(injs, cleans)], zones,
                                                   pads=pads))
            hi = dict(hdr, IMGTYPE="running equal-weight stitch (injected)", INJNSRC=len(sources))
            for j, s in enumerate(sources):
                hi[f"INJRHO{j+1}"], hi[f"INJPA{j+1}"] = float(s[0]), float(s[1])
            files["inj"] = os.path.join(self.run_dir, "klip_stitched_running_inj.fits")
            _write_fits(files["inj"], sti, hi)
            files["snr_inj"] = os.path.join(self.run_dir, "klip_stitched_running_snr_inj.fits")
            _write_fits(files["snr_inj"], snr_map(sti, self.fwhm, exclude_xy=self._inj_xy(sources, sti.shape)),
                        dict(hi, IMGTYPE="running stitched S/N map (injected)"))
            out["inj"] = sti
        if at_end and any(s is not None for s in stacks):
            w = self._stitch_weights(cleans, zones)
            cube, ids = stitch_stacks(stacks, parts, zones, w, pads=pads)
            if cube is not None:
                hc = dict(hdr, IMGTYPE="running stitched per-partition cube (clean)", NNIGHT=len(ids))
                for j, p in enumerate(ids):
                    hc[f"NIGHT{j+1}"] = p
                files["nights"] = os.path.join(self.run_dir, "klip_stitched_running_nights.fits.gz")
                _write_fits(files["nights"], cube, hc)
                out.update(nights=cube, night_ids=ids, weights=w)
        self._emit("on_stitch", "running", files)
        out["files"] = files
        return out

    # ---------------------------------------------------------- verification hooks
    def _eval_injection_rows(self, annuli: Sequence[int]) -> List[Dict[str, Any]]:
        """Per-eval injected-source S/N samples (raw metric) of the given annuli from
        ``results.jsonl`` (last re-calibration segment of each), as ``verify`` injection rows."""
        path = os.path.join(self.run_dir, "results.jsonl")
        if not os.path.exists(path):
            return []
        by_ann: Dict[int, List[Dict[str, Any]]] = {}
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("annulus") in annuli:
                    by_ann.setdefault(int(r["annulus"]), []).append(r)
        rows = []
        for ia, recs in by_ann.items():
            starts = [i for i, r in enumerate(recs) if r.get("index") == 0]
            seg = recs[starts[-1]:] if starts else recs
            for r in seg:
                if r.get("meta", {}).get("failed"):
                    continue
                for s, v in zip(r.get("sources", []), r.get("raw_per_source", [])):
                    if v is None or not np.isfinite(v):
                        continue
                    rho = float(s[0])
                    rows.append({"kind": "injection", "index": len(rows) + 1, "rho": rho, "theta": float(s[1]),
                                 "contrast": float(r.get("contrast", np.nan)), "snr": float(v),
                                 "nap": int(np.floor(2 * np.pi * (rho / self.pxscale) / self.fwhm)), "annulus": ia})
        return rows

    def _verify_hook(self, ia: int, winner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """``klip_tpe.verify`` on the winner's per-partition stacks (report) and the
        per-eval injection limit curve (``verify_curve.txt``, per annulus and run-level)."""
        from .verify import limit_tables, verify_candidates, write_curve_table, write_verify_table
        d = self._ann_dir(ia)
        out: Dict[str, Any] = {}
        ev_c, ev_i = winner.get("clean"), winner.get("inj")
        src = [s.as_tuple() for s in winner.get("sources") or []]
        known = [tuple(k) for k in getattr(self.sampler, "known", ()) or ()]
        rng = np.random.default_rng((int(self.cfg.seed or 0) + 1013 * (ia + 1)) % (2 ** 32))
        if ev_c is not None and ev_i is not None and src:
            stack = ev_c.stack if ev_c.stack is not None else ev_c.image[None]
            istack = ev_i.stack if ev_i.stack is not None else ev_i.image[None]
            ids = [str(p) for p in (ev_c.partitions or list(range(stack.shape[0])))]
            rows, maps = verify_candidates(stack, ids, list(known), self.fwhm, self.pxscale, known=known,
                                           injected=(istack, src), n_boot=int(self.cfg.verify_n_boot), rng=rng,
                                           weights=ev_c.weights, angle_convention=self.reducer.angle_convention,
                                           excl_fwhm=self._excl_fwhm(), return_maps=True,
                                           combined=ev_c.image, combined_inj=ev_i.image,
                                           path_a="the annulus' combined image (best_clean.fits, injected best_inj.fits)")
            path = os.path.join(d, "verify_report.txt")
            write_verify_table(path, rows, maps.get("limits"), header={"run": os.path.abspath(self.run_dir)})
            self._durable_later(path)              # the hook's checkpoint records it as done
            out.update(rows=rows, limits=maps.get("limits"), report=path)
        # per-eval injection limit curve of this annulus + the run-level accumulation
        rows_a = self._eval_injection_rows([ia])
        lim = limit_tables(rows_a, self.fwhm, self.pxscale) if rows_a else None
        if lim is not None:
            write_curve_table(os.path.join(d, "verify_curve.txt"), lim,
                              header=f"annulus {ia+1}: {len(rows_a)} injections from {self.cfg.per_annulus(self.cfg.n_iter, ia)} search evals")
            self._durable_later(os.path.join(d, "verify_curve.txt"))
            out["curve"] = lim
        done = [r.annulus for r in self.results if r is not None]
        rows_all = self._eval_injection_rows(done)
        lim_all = limit_tables(rows_all, self.fwhm, self.pxscale) if rows_all else None
        if lim_all is not None:
            write_curve_table(os.path.join(self.run_dir, "verify_curve.txt"), lim_all,
                              header=f"annuli {','.join(str(a+1) for a in done)}: {len(rows_all)} search-eval injections")
            self._durable_later(os.path.join(self.run_dir, "verify_curve.txt"))
        self._emit("on_verify", ia, "verify", out)
        return out

    def _param_verify_hook(self, ia: int, winner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """``klip_tpe.param_verify`` on this annulus' history with a reduce_fn through the runner."""
        from .param_verify import param_verify
        cfg_w: Config = winner["config"]
        rlo, rhi = self._zone(ia, cfg_w)
        out_dir = os.path.join(self._ann_dir(ia), "param_verify")
        hist = self.annulus_history(ia)
        if hist is None:
            return None
        os.makedirs(out_dir, exist_ok=True)

        def reduce_fn(x, sources):
            # per-config reductions are cached on disk so an interrupted param_verify resumes
            # (the fixed injection set is deterministic, so the cache key is the config alone)
            import hashlib
            key = hashlib.sha1(np.asarray(x, float).round(9).tobytes()).hexdigest()[:12]
            cpath = os.path.join(out_dir, f"cache_{key}.npz")
            if os.path.exists(cpath):
                try:
                    with np.load(cpath) as z:
                        return z["clean"], z["inj"]
                except Exception:
                    pass                           # unreadable (cut short by a crash): reduce it again
            cfg = self.space.decode(np.asarray(x, float))
            if self.cfg.k_mode in SCAN_MODES:
                m = np.flatnonzero(np.all(np.abs(hist.X - np.asarray(x, float)[None, :]) < 1e-9, axis=1))
                if m.size:
                    cfg = self._apply_k_flag(cfg, hist.flags[int(m[-1])].get("k"))
            calls = [lambda: self._reduce(cfg, None, tag=f"a{ia+1}_pv_clean", zone=(rlo, rhi)),
                     lambda: self._reduce(cfg, sources, tag=f"a{ia+1}_pv_inj", zone=(rlo, rhi))]
            clean, inj = [r for _, r in self._run_ordered(calls, 2 if self._parallel_reducer() else 1)]
            for r in (clean, inj):
                if isinstance(r, BaseException):
                    raise r
            try:                                   # written whole, then renamed: configs run concurrently
                tmp = cpath + f".{os.getpid()}.{threading.get_ident()}.tmp.npz"
                np.savez(tmp, clean=clean.image, inj=inj.image)
                durable.fsync_file(tmp)            # the data on disk before the name
                os.replace(tmp, cpath)
                durable.fsync_dir(out_dir)
            except Exception:
                pass
            return clean.image, inj.image

        rng = np.random.default_rng((int(self.cfg.seed or 0) + 7919 * (ia + 1)) % (2 ** 32))
        # the fixed set comes from THE run's sampler (same geometry as search / calibration /
        # validation -- addendum 2 §2b: one placement function for every stage)
        blo, bhi = self._band(ia, cfg_w)
        fixed = self.sampler.sample(self._nsrc(ia), blo, bhi, rng, self.contrast)
        res = param_verify(hist, self.space, reduce_fn, self.fwhm, self.pxscale, rlo, rhi, out_dir,
                           fixed_sources=fixed,
                           ccal=self.contrast, tag=f"{ia+1:02d}", n_pv=int(self.cfg.n_pv),
                           pv_divmin=float(self.cfg.pv_divmin), n_init=self.cfg.per_annulus(self.cfg.n_init, ia),
                           nsrc=self._nsrc(ia), rng=rng, known=list(getattr(self.sampler, "known", ()) or ()),
                           angle_convention=self.reducer.angle_convention,
                           flatten=self.flatten_products, log=self.log, inner_fwhm=self._inner_fwhm(),
                           # several configurations in flight, each its clean + injected pair
                           run_calls=lambda cs: self._run_ordered(
                               cs, max(1, self._reduction_window(len(cfg_w.selected)) // 2)))
        if res is not None:
            res["zone"] = (rlo, rhi)
            self._pv_results[ia] = res
            # the maps the final stitch reads back after a resume (_pv_stitched)
            self._durable_later(*[p for p in (res.get("files") or {}).values() if isinstance(p, str)])
        self._emit("on_verify", ia, "param_verify", res)
        return res

    def _pv_stitched(self) -> Optional[Dict[str, np.ndarray]]:
        """Merge the per-annulus param_verify maps (memory or disk) into run-level maps."""
        from .param_verify import PV_KINDS, stitch_param_maps
        from .stitch import read_fits
        maps, zones = [], []
        for r in self.results:
            if r is None:
                continue
            pv = self._pv_results.get(r.annulus)
            if pv is None:
                pdir = os.path.join(self.run_dir, f"annulus{r.annulus+1:02d}", "param_verify")
                pv = {k: read_fits(os.path.join(pdir, f"paramverify_{k}_{r.annulus+1:02d}.fits")) for k in PV_KINDS}
                pv = {k: v for k, v in pv.items() if v is not None}
                if not pv:
                    continue
            maps.append(pv)
            zones.append(tuple(pv.get("zone", (r.inrad, r.outrad))))
        if not maps:
            return None
        return stitch_param_maps(maps, zones, out_dir=self.run_dir, pads=self._tile_pads(zones, pad=1.0))

    def _candidates_hook(self, ia: int, final: bool = False, stitched: Optional[Dict[str, Any]] = None
                         ) -> Optional[List[Dict[str, Any]]]:
        """Blind candidate search (``klip_tpe.candidates``) on the running stitch after
        annulus ``ia`` (quick: ranked table only) or on the final stitch (top-N verified,
        plus the injection-recovery search on the injected stitch)."""
        from .candidates import find_candidates, write_candidates
        st = stitched if stitched is not None else self._running_stitch(ia, True)
        if not st or st.get("clean") is None:
            return None
        cdir = os.path.join(self.run_dir, "cand")
        os.makedirs(cdir, exist_ok=True)
        edges = [z[0] for z in st["zones"]] + [st["zones"][-1][1]]
        ny, nx = st["clean"].shape
        iwa = max(edges[0], 1.2 * self.fwhm)
        owa = min(edges[-1], min(nx, ny) / 2.0 - 2.0)
        known = [tuple(k) for k in getattr(self.sampler, "known", ()) or ()]
        rng = np.random.default_rng((int(self.cfg.seed or 0) + 4241 * (ia + 1)) % (2 ** 32))
        pmap = None
        if final:
            pvs = self._guard("param_verify stitch", self._pv_stitched)
            pmap = None if not pvs else pvs.get("stim")
        allsrc = []
        for ja in range(ia + 1):
            dj = self._annulus_images(ja)
            if dj:
                allsrc.extend(dj.get("sources") or [])
        kw = dict(iwa_px=iwa, owa_px=owa, snrmin=float(self.cfg.cand_snrmin), known=known, persistence_map=pmap,
                  angle_convention=self.reducer.angle_convention)
        results = {}
        for label, img_key, snr_key, inj_key in (("", "clean", "snr", None), ("inj_blind_", "inj", "snr_inj", "inj")):
            img = st.get(img_key)
            if img is None:
                continue
            stack = st.get("nights_inj" if label else "nights")
            if stack is None:
                stack = img[None]
            ids = st.get("night_ids") if stack.ndim == 3 and stack.shape[0] > 1 else None
            rows = find_candidates(img, stack, self.fwhm, self.pxscale, snrmap=st.get(snr_key),
                                   injected=allsrc if label else (), night_ids=ids,
                                   verify_top=int(self.cfg.cand_verify_top) if (final and not label) else 0,
                                   verify_kwargs={"n_boot": int(self.cfg.verify_n_boot), "rng": rng},
                                   flatten=self.flatten_products, **kw)
            write_candidates(os.path.join(cdir, f"{label}candidates.txt"), rows,
                             source=("final " if final else f"running (annulus {ia+1}) ") + img_key + " stitch")
            results[label or "clean"] = rows
            if not final:
                break
        self._emit("on_verify", ia, "candidates", results)
        return results.get("clean")

    # --------------------------------------------------------------- run / io
    def write_setup(self) -> None:
        d = {
            "klip_tpe_version": __version__, "run_dir": os.path.abspath(self.run_dir),
            "config": self.cfg.to_dict(), "space": self.space.to_dict(),
            "reducer": self.reducer.describe(), "objective": self.objective.describe(),
            "sampler": self.sampler.describe(),
            "projection": getattr(self.space.project, "describe", lambda: None)() if self.space.project else None,
            "fwhm_px": self.fwhm, "pxscale": self.pxscale,
            "partition_label": getattr(self.reducer, "partition_label", "night"),
        }
        with open(os.path.join(self.run_dir, "run_setup.json"), "w") as f:
            json.dump(d, f, indent=1, default=_json_default)
        # written once: a resume does not write them again, so on disk with the first checkpoint
        self._durable_later(os.path.join(self.run_dir, "run_setup.json"), os.path.join(self.run_dir, "run_setup.txt"))
        with open(os.path.join(self.run_dir, "run_setup.txt"), "w") as f:
            f.write(f"# KLIP-TPE run setup -- {time.ctime()}  (klip_tpe {__version__})\n")
            for k, v in self.cfg.to_dict().items():
                f.write(f"{k:<19s}: {v}\n")
            f.write(f"{'pxscale / fwhm(px)':<19s}: {self.pxscale} / {self.fwhm:.3f}\n")
            f.write(f"{'reducer':<19s}: {self.reducer.name}  angle_convention={self.reducer.angle_convention}\n")
            f.write(f"{'objective':<19s}: {self.objective.describe()}\n\n")
            f.write(self.space.setup_text() + "\n")
            f.write("\n# default (seed) configuration\n  " + " ".join(
                f"{n}={v:g}" for n, v in zip(self.space.names, self._default_vector())) + "\n")

    def _done(self, next_ia: int) -> bool:
        """True when no further annulus is due (fixed edges: ``nann`` reached; opt_width:
        the last committed edge reached ``r_cap``)."""
        if self.cfg.opt_width:
            e = self.cfg.ann_edges
            if next_ia < 1 or len(e) <= next_ia:
                return False
            w_lo = float(self.cfg.width_range[0]) if self.cfg.width_range else 0.0
            return (e[next_ia] >= self.cfg.r_cap - 0.5 or e[next_ia] <= e[next_ia - 1]   # cap reached / no progress
                    or self.cfg.r_cap - e[next_ia] < w_lo)                               # remaining ring < w_lo: stop
        return next_ia >= self.cfg.nann

    def run(self) -> List[AnnulusResult]:
        self._auto_resume()            # any run is resumable, however it was constructed
        if not self._resumed:
            self._check_data_edits()   # a resume checked them as it adopted the checkpoint
        self._register()
        self._start_workers()          # before the heartbeat thread: the fork copies one thread
        self.hb.start()                # from here on the directory says whether this is alive
        from .profiling import sampler_from_env
        prof = sampler_from_env(self.run_dir, log=self.log, status=lambda: (
            f"annulus {self.ia + 1}, evaluation {0 if self.history is None else len(self.history)}"))
        try:
            out = self._run()
        except BaseException:
            self.flush_checkpoint()    # the last FINISHED evaluation, not the interrupted one
            self.hb.stop()             # keep the stage it died in: that is the diagnosis
            if prof is not None:
                prof.stop()
            raise
        self.flush_checkpoint()
        self.hb.stop("finished")
        if prof is not None:
            prof.stop()
        return out

    def _liveness_preflight(self) -> None:
        """:attr:`RunConfig.liveness_check`: is every searched dimension doing something?
        Dead = moved alone and never changed the clean reduction -> logged as a warning, or
        SystemExit before any evaluation in strict mode.  Untested = the guards never let it
        move alone at the points tried -> logged, not fatal (not evidence of a bug).  The
        search RNG is put back, so a seeded run is the same run with or without the check."""
        from .liveness import check_live_dimensions, liveness_mode
        strict = liveness_mode(self.cfg.liveness_check) == "strict"
        self.hb.stage("liveness pre-flight", annulus=0)
        n_red = sum(1 for p in self.space.params if p.role == "reduction")
        self.log(f"liveness: moving each of the {n_red} searched reduction dimension(s) alone, annulus 1")
        st = self.rng.bit_generator.state
        try:
            status = check_live_dimensions(self, log=self.log)
        finally:
            self.rng.bit_generator.state = st
        dead = sorted(n for n, s in status.items() if not s["live"] and s["tested"])
        untested = sorted(n for n, s in status.items() if not s["live"] and not s["tested"])
        if untested:
            self.log(f"liveness: {len(untested)} dimension(s) could not be moved alone at the points tried "
                     f"(guards): {', '.join(untested)} -- not evidence of a bug; continuing")
        if dead and strict:
            raise SystemExit(
                f"liveness: searched dimension(s) that change nothing when moved alone: {', '.join(dead)} -- "
                f"not starting a search that would spend its budget on them.  Check that the reducer "
                f"receives them (docs/IDL_FINDINGS.md section 1 shows what this looks like), or drop "
                f"--strict-liveness to search anyway.")
        if dead:
            self.log(f"liveness: WARNING -- searched dimension(s) that changed nothing when moved alone: "
                     f"{', '.join(dead)}.  The search will spend part of its budget on them.  If the "
                     f"reducer should act on them, check that it receives them (docs/IDL_FINDINGS.md "
                     f"section 1); --strict-liveness stops a run here.")
            return
        self.log(f"liveness: all {n_red - len(untested)} tested reduction dimension(s) live")

    def _run(self) -> List[AnnulusResult]:
        from .version import describe
        self.log(describe())           # which code this is: a log then says if it matches the repository
        if not self._resumed:
            if liveness_mode(self.cfg.liveness_check) != "off":
                self._liveness_preflight()
            self.write_setup()
            self._emit("on_setup")
            start_ia, resume_index = 0, None
        else:
            self._emit("on_setup")          # callbacks set up their window / backend on a resume too
            start_ia, resume_index = self.ia, len(self.history) if self.history is not None else None
        ia = start_ia
        if self._resumed:
            for ja, res in enumerate(self.results):
                if resume_index is not None and ja >= start_ia:
                    continue            # this annulus is being re-run (resumed search / extend): its hooks follow
                if res is not None and self.hooks_pending(ja):
                    w = self._winner_from_disk(ja)
                    if w is not None:
                        self.log(f"  annulus {ja+1}: finishing interrupted hooks {self.hooks_pending(ja)}")
                        self._run_hooks(ja, w)
        while not self._done(ia) or (ia == start_ia and resume_index is not None):
            self.run_annulus(ia, resume_index if ia == start_ia else None)
            resume_index = None
            ia += 1
        self.finish()
        return [r for r in self.results if r is not None]

    def finish(self) -> None:
        """Final products: sensitivity-weighted stitch with seam patching, injected /
        per-partition / S/N stitches, ``klip_stitched_params.txt``, run-level
        ``contrast_curve.txt``, ``final_results.json``, the bench summary row and the
        final verification / candidate hooks."""
        self.hb.stall_after(None)                 # final products take as long as they take
        self.hb.stage("final products")
        results = [r for r in self.results if r is not None]
        out = {"annuli": [r.to_dict() for r in results], "seed": self.cfg.seed, "bench_tag": self.cfg.bench_tag}
        stitched = None
        if results and self.cfg.save_fits:
            stitched = self._guard("final stitch", self._final_stitch, results)
            if stitched is not None and stitched.get("weights") is not None:
                out["stitch_weights"] = [float(v) for v in stitched["weights"]]
                out["stitch_files"] = stitched.get("files", {})
        if results:
            from .stitch import write_contrast_curve
            edges = [r.inrad for r in results] + [results[-1].outrad]
            self._guard("contrast_curve.txt", write_contrast_curve, os.path.join(self.run_dir, "contrast_curve.txt"),
                        [r.contrast_curve for r in results], edges, self.fwhm, self.pxscale,
                        legacy=self.cfg.legacy_stitch, contrasts=[r.contrast for r in results],
                        fm_curves=[r.fm_curve for r in results],
                        grid_curves=[getattr(r, "grid_curve", None) for r in results])
            self._durable_later(os.path.join(self.run_dir, "contrast_curve.txt"))
        # final_results.json says the run is finished (bench, klip-tpe runs): the products
        # above reach the disk first, and it is never left empty by a crash
        if stitched is not None:
            self._durable_later(*[p for p in (stitched.get("files") or {}).values() if isinstance(p, str)])
        self._sync_files()
        _atomic_write(os.path.join(self.run_dir, "final_results.json"), json.dumps(out, indent=1, default=_json_default))
        # append to the benchmark summary (run-tagged, one row per annulus)
        with open(os.path.join(self.run_dir, "bench_summary.txt"), "a") as f:
            for r in results:
                f.write(f"{self.cfg.search_mode:<8s} {r.annulus+1:3d} {self.cfg.nann:3d} "
                        f"{self.cfg.per_annulus(self.cfg.n_iter, r.annulus):5d} "
                        f"{self.cfg.per_annulus(self.cfg.n_init, r.annulus):4d} "
                        f"{r.search_best_score:9.3f} {r.winner_score:9.3f} {int(r.validated)} "
                        f"{self.cfg.bench_tag} {os.path.basename(os.path.abspath(self.run_dir))}\n")
        # final-layer hooks (guarded)
        if results and stitched is not None:
            if self.do_param_verify:
                self._guard("param_verify stitch", self._pv_stitched)
            if self.cfg.verify:
                self._guard("final verify", self._final_verify, stitched)
            if self.cfg.candidates:
                self._guard("final candidate search", self._candidates_hook, results[-1].annulus, True, stitched)
        self.log(f"run complete: {len(results)} annuli, {self.records_count} evaluations")
        self._emit("on_finish")

    def _final_stitch(self, results: Sequence[AnnulusResult], weights: Optional[Sequence[float]] = None,
                      extra_header: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """The final stitch of the annuli' winner images.  ``weights`` replaces the
        sensitivity weights (:meth:`rereduce` keeps the run's), and ``extra_header`` goes into
        every FITS header."""
        from .products import snr_map
        from .stitch import seam_pixels, stitch_stacks, stitch_tiles, write_params_table
        items = [(r, self._annulus_images(r.annulus)) for r in results]
        items = [(r, d) for r, d in items if d is not None and d.get("clean") is not None]
        if not items:
            return None
        cleans = [d["clean"] for _, d in items]
        zones = [tuple(d["zone"]) for _, d in items]
        edges = [z[0] for z in zones] + [zones[-1][1]]
        w = self._stitch_weights(cleans, zones) if weights is None else np.asarray(weights, float)
        pads = self._tile_pads(zones)
        st = stitch_tiles(cleans, zones, w, pads=pads)
        # seam patching: pixels inside the searched span left NaN -> a re-run of that annulus'
        # winner, its zone padded by 3 px on each edge shared with another annulus
        seam = seam_pixels(st, edges[0], edges[-1])
        npatch = 0
        t_seam = time.time()
        if seam.any():
            rr = np.hypot(*np.meshgrid(np.arange(st.shape[1]) - star_center(st.shape)[0],
                                       np.arange(st.shape[0]) - star_center(st.shape)[1]))
            ia_keep = self.ia
            for j, (r, d) in enumerate(items):
                lo, hi = edges[j], edges[j + 1]
                m = seam & (rr >= lo) & ((rr <= hi) if j == len(items) - 1 else (rr < hi))
                if not m.any():
                    continue
                try:
                    self.ia = r.annulus
                    cfg = self._config_from_dict(r.winner_config)
                    pin, pout = self._pad_edges(*zones[j])
                    zone = (max(zones[j][0] - (3.0 if pin else 0.0), 0.0),
                            min(zones[j][1] + (3.0 if pout else 0.0), self.cfg.r_cap))
                    img_p = self._reduce(cfg, None, tag=f"a{r.annulus+1}_seam", zone=zone).image
                    if img_p is not None and img_p.shape == st.shape:
                        st[m] = img_p[m]
                        npatch += int(m.sum())
                except Exception as exc:
                    self.log(f"  seam patch of annulus {r.annulus+1} failed: {exc!r}")
            self.ia = ia_keep
            # The cost is one full re-reduction per annulus that has ANY seam pixel, so it
            # is set by the number of annuli, not by the number of pixels being filled --
            # a one-pixel seam costs as much as a five-hundred-pixel one.  Cheap on a
            # single annulus (0.1 s measured); worth seeing on a run with eight of them,
            # where the same line would hide eight reductions.
            self.log(f"  seam patching: {int(seam.sum())} pixel(s), {npatch} filled "
                     f"({time.time() - t_seam:.1f} s; one re-reduction per annulus with a seam)")
        st_f = self._maybe_flatten(st)
        hist = ["per-annulus best (edges px | k bin nang filt angsep anglemax | SNR | partitions):"]
        for r, d in items:
            p = r.winner_config.get("params", {})
            ptxt = " ".join(f"{ab}{p[k]}" for k, ab in (("k_klip", "k"), ("bin", "b"), ("n_ang", "na"), ("filter", "f"),
                                                       ("angsep", "as"), ("anglemax", "am")) if p.get(k) is not None)
            hist.append(f"  a{r.annulus+1} [{r.inrad:5.1f}-{r.outrad:5.1f}] {ptxt} "
                        f"SNR{r.winner_score:6.2f} c{r.contrast:.2E} n={','.join(r.partitions) or 'all'}")
        hdr = {"NANN": len(items), "PHASE": "final", "IMGTYPE": f"radially-optimized stitch{self._flat_tag}",
               "LEGACY": int(self.cfg.legacy_stitch), "NSEAM": int(seam.sum()), "NPATCH": npatch,
               "EDGES": ",".join(f"{e:.1f}" for e in edges), "WEIGHTS": ",".join(f"{v:.3f}" for v in w),
               "METRIC": getattr(self.objective.metric, "name", "metric"), "KMODE": self.cfg.k_mode,
               "NITER": str(self.cfg.n_iter), "NINIT": str(self.cfg.n_init), "NVALID": self.cfg.validation.n_valid,
               "NTOP": self.cfg.validation.n_top, "GAMMA": self.cfg.gamma, "NCAND": self.cfg.ncand,
               "SEED": self.cfg.seed, "RUNDIR": os.path.basename(os.path.abspath(self.run_dir)),
               "OPTWIDTH": int(self.cfg.opt_width)}
        hdr.update(extra_header or {})
        files = {"clean": os.path.join(self.run_dir, "klip_stitched.fits")}
        _write_fits(files["clean"], st_f, hdr, hist)
        snr = snr_map(st_f, self.fwhm)
        files["snr"] = os.path.join(self.run_dir, "klip_stitched_snr.fits")
        _write_fits(files["snr"], snr, dict(hdr, IMGTYPE="final stitched S/N map (per-pixel Mawet)"), hist)
        out: Dict[str, Any] = {"clean": st_f, "snr": snr, "weights": w, "zones": zones, "files": files}
        # validated-winner injected stitch
        allsrc = []
        for _, d in items:
            allsrc.extend(d.get("sources") or [])
        if any(d.get("inj") is not None for _, d in items):
            sti = self._maybe_flatten(stitch_tiles([d["inj"] if d.get("inj") is not None else d["clean"]
                                                    for _, d in items], zones, w, pads=pads))
            hi = dict(hdr, IMGTYPE=f"validated-winner injected stitch{self._flat_tag}", INJNSRC=len(allsrc))
            for j, s in enumerate(allsrc):
                hi[f"INJRHO{j+1}"], hi[f"INJPA{j+1}"] = float(s[0]), float(s[1])
                if len(s) > 2:
                    hi[f"INJCON{j+1}"] = float(s[2])
            files["inj"] = os.path.join(self.run_dir, "klip_stitched_inj.fits")
            _write_fits(files["inj"], sti, hi, hist)
            files["snr_inj"] = os.path.join(self.run_dir, "klip_stitched_snr_inj.fits")
            _write_fits(files["snr_inj"], snr_map(sti, self.fwhm, exclude_xy=self._inj_xy(allsrc, sti.shape)),
                        dict(hi, IMGTYPE="validated injected stitched S/N map (per-pixel Mawet)"), hist)
            out["inj"] = sti
        # per-partition stacks on the union of partition ids
        for key, fname, label in (("clean_stack", "klip_stitched_nights.fits.gz", "nights"),
                                  ("inj_stack", "klip_stitched_nights_inj.fits.gz", "nights_inj")):
            stacks = [d.get(key) for _, d in items]
            if not any(s is not None for s in stacks):
                continue
            cube, ids = stitch_stacks(stacks, [d.get("parts") for _, d in items], zones, w, pads=pads)
            if cube is None:
                continue
            hc = dict(hdr, IMGTYPE=f"stitched per-partition cube ({'injected' if 'inj' in key else 'clean'})",
                      NNIGHT=len(ids), INJECTED=int("inj" in key))
            for j, p in enumerate(ids):
                hc[f"NIGHT{j+1}"] = p
            if "inj" in key:
                hc["INJNSRC"] = len(allsrc)
                for j, s in enumerate(allsrc):
                    hc[f"INJRHO{j+1}"], hc[f"INJPA{j+1}"] = float(s[0]), float(s[1])
            files[label] = os.path.join(self.run_dir, fname)
            _write_fits(files[label], cube, hc)
            out[label] = cube
            out["night_ids"] = ids
        # parameters table
        rows = []
        for r, d in items:
            rows.append({"ann": r.annulus + 1, "inner_px": r.inrad, "outer_px": r.outrad,
                         "params": r.winner_config.get("params", {}), "best_SNR": r.winner_score,
                         "partitions": r.partitions, "per_partition": r.winner_config.get("per_partition", {}),
                         "validated": r.validated, "contrast": r.contrast})
        files["params"] = os.path.join(self.run_dir, "klip_stitched_params.txt")
        write_params_table(files["params"], rows)
        self._emit("on_stitch", "final", files)
        return out

    def _final_verify(self, stitched: Dict[str, Any], header: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """``near2_verify`` with the known sources as candidates and all validated injections
        as the calibration block.  Path A measures the final stitch itself (``klip_stitched.fits``
        and ``klip_stitched_inj.fits``), Path B the stitched per-partition cubes.  ``header``
        adds lines to the report's header."""
        from .verify import verify_candidates, write_verify_table
        stack = stitched.get("nights")
        if stack is None:
            stack = stitched["clean"][None]
        istack = stitched.get("nights_inj")
        if istack is None and stitched.get("inj") is not None:
            istack = stitched["inj"][None]
        allsrc = []
        for r in self.results:
            if r is not None:
                allsrc.extend(r.winner_sources)
        known = [tuple(k) for k in getattr(self.sampler, "known", ()) or ()]
        rng = np.random.default_rng((int(self.cfg.seed or 0) + 31337) % (2 ** 32))
        injected = None if (istack is None or not allsrc) else (istack, allsrc)
        # Path A on the image the run made, not on the mean of the per-partition stitches: at an
        # edge between annuli those differ where the annuli keep different partitions (NEAR2,
        # 2026-10-20)
        clean_img, inj_img = stitched.get("clean"), stitched.get("inj")
        if injected is not None and inj_img is None:
            clean_img = None
        rows, maps = verify_candidates(stack, stitched.get("night_ids"), known, self.fwhm, self.pxscale, known=known,
                                       injected=injected, n_boot=int(self.cfg.verify_n_boot), rng=rng,
                                       angle_convention=self.reducer.angle_convention,
                                       excl_fwhm=self._excl_fwhm(), return_maps=True,
                                       combined=clean_img, combined_inj=None if injected is None else inj_img,
                                       path_a=None if clean_img is None else
                                       "the stitched image (klip_stitched.fits, injected klip_stitched_inj.fits)")
        path = os.path.join(self.run_dir, "stitched_verify_report.txt")
        write_verify_table(path, rows, maps.get("limits"), header=dict({"run": os.path.abspath(self.run_dir)},
                                                                       **(header or {})))
        out = {"rows": rows, "limits": maps.get("limits"), "report": path}
        self._emit("on_verify", -1, "verify", out)
        return out

    def rereduce(self, out_dir: str, *, verify_sources: Optional[Sequence[Any]] = None, verify: bool = True,
                 log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """A finished run's winners reduced again, on this runner's reducer, with the final
        stitch, its S/N map and the final verify written to ``out_dir``.  The run directory is
        only read.

        The point is other frames: a line subtracted, planets added (the reducer's
        ``data_edits``), without a new search.  Build the runner with :meth:`resume` and
        ``same_frames=False``.  Each annulus' winning configuration is reduced clean, and with
        the injections of the verify's calibration block: the winner's own validation sources
        (each in its annulus, as in the run), or ``verify_sources`` (``(rho_arcsec, pa_deg,
        contrast)``, all of them in every annulus' injected reduction, each listed under the
        annulus that holds it).  The tiles are stitched with the run's weights
        (``final_results.json``, else worked out from its winner images as the run did) and
        the run's seam patching, so on the run's own frames the products are the run's.

        Writes ``klip_stitched.fits``, ``klip_stitched_snr.fits``, the injected and
        per-partition stitches, ``klip_stitched_params.txt``, ``stitched_verify_report.txt``
        (unless ``verify`` is False) and ``rereduce.json``.  Returns that record."""
        log = log or self.log
        t0 = time.time()
        results = [r for r in (self.results or []) if r is not None]
        src_run = os.path.abspath(self.run_dir)
        if not results or len(results) < self.cfg.nann:
            raise ValueError(f"{os.path.basename(src_run)} has not finished: {len(results)} of {self.cfg.nann} "
                             f"annuli have a winner")
        out_dir = os.path.abspath(out_dir)
        if out_dir == src_run or out_dir.startswith(src_run + os.sep):
            raise ValueError(f"{out_dir} is inside the run: rereduce writes beside it")
        # the run's own winner images, for the zones and the stitch weights
        items = [(r, self._annulus_images(r.annulus)) for r in results]
        if any(d is None or d.get("clean") is None for _, d in items):
            raise ValueError("an annulus' winner images are missing (the run kept no FITS products?)")
        zones = [tuple(float(v) for v in d["zone"]) for _, d in items]
        weights, w_from = None, ""
        try:
            with open(os.path.join(src_run, "final_results.json")) as f:
                wl = json.load(f).get("stitch_weights")
            if wl is not None and len(wl) == len(items):
                weights, w_from = [float(v) for v in wl], "final_results.json"
        except (OSError, ValueError):
            pass
        if weights is None:
            weights = [float(v) for v in self._stitch_weights([d["clean"] for _, d in items], zones)]
            w_from = "the run's winner images"
        # the injections: per annulus, what goes in and what the verify lists under it
        if verify_sources is None:
            listed = {r.annulus: [Source(*(float(v) for v in s)) for s in r.winner_sources] for r, _ in items}
            put_in = dict(listed)
            v_from = "the winners' validation injections"
        else:
            vs = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in verify_sources]
            listed = {r.annulus: [] for r, _ in items}
            for s in vs:
                rp = float(s.rho) / self.pxscale
                hold = [r.annulus for (r, _), z in zip(items, zones) if z[0] <= rp < z[1]]
                if not hold and abs(rp - zones[-1][1]) < 1e-6:
                    hold = [items[-1][0].annulus]
                if not hold:
                    raise ValueError(f"verify source at rho={s.rho:g}\" ({rp:.1f} px) lies outside the run's annuli "
                                     f"({zones[0][0]:.1f}-{zones[-1][1]:.1f} px)")
                listed[hold[0]].append(s)
            put_in = {r.annulus: list(vs) for r, _ in items}
            v_from = f"{len(vs)} given source(s)"
        edits = dict(getattr(self.reducer, "data_edits", None) or {})
        frames, run_frames = _describe_edits(_core_edits(edits)), _describe_edits(_core_edits(self.cfg.data_edits))
        log(f"rereduce {os.path.basename(src_run)}: {len(items)} annulus winner(s) on {frames} (the run was made on "
            f"{run_frames}), stitch weights from {w_from}, verify injections: {v_from}")
        self._start_workers()
        new: Dict[int, Dict[str, Any]] = {}
        per_ann = []
        ia_keep = self.ia
        try:
            for (r, d), z in zip(items, zones):
                t1 = time.time()
                self.ia = r.annulus
                cfg = self._config_from_dict(r.winner_config)
                ev_c = self._reduce(cfg, None, tag=f"rereduce_a{r.annulus + 1}_clean")
                src = put_in[r.annulus]
                ev_i = self._reduce(cfg, src, tag=f"rereduce_a{r.annulus + 1}_inj") if src else None
                if ev_c.image is None or (ev_i is not None and ev_i.image is None):
                    raise ValueError(f"annulus {r.annulus + 1}: the winner's reduction returned no image")
                new[r.annulus] = {
                    "clean": ev_c.image, "inj": None if ev_i is None else ev_i.image, "zone": z,
                    "clean_stack": ev_c.stack, "inj_stack": None if ev_i is None else ev_i.stack,
                    "parts": [str(p) for p in (ev_c.partitions or r.partitions or cfg.selected)],
                    "sources": [s.as_tuple() for s in listed[r.annulus]], "contrast": d.get("contrast"),
                    "weights": ev_c.weights}
                per_ann.append({"annulus": r.annulus + 1, "zone_px": list(z), "winner_eval": int(r.winner_index) + 1,
                                "injected": len(src), "listed": len(listed[r.annulus]),
                                "seconds": round(time.time() - t1, 1)})
                log(f"  annulus {r.annulus + 1}: the winner (eval {r.winner_index + 1}) reduced clean"
                    + (f" and with {len(src)} injection(s)" if src else "") + f" ({time.time() - t1:.1f} s)")
        finally:
            self.ia = ia_keep
        os.makedirs(out_dir, exist_ok=True)
        mod = [replace(r, winner_sources=[s.as_tuple() for s in listed[r.annulus]]) for r in results]
        keep = (self.run_dir, self._ann_images, self.results)
        nsub, nadd = len(edits.get("subtracted_sources") or []), len(edits.get("injected_sources") or [])
        ndrop = sum(len(v) for v in _dropped(edits.get("dropped_frames")).values())
        try:
            self.run_dir, self._ann_images, self.results = out_dir, new, list(mod)
            stitched = self._final_stitch(mod, weights=weights, extra_header={
                "REREDUCE": 1, "FROMRUN": os.path.basename(src_run)[:60], "NSUBSRC": nsub, "NADDSRC": nadd,
                "NDROPFR": ndrop})
            ver = None
            if verify and stitched is not None:
                ver = self._final_verify(stitched, header={
                    "run": f"{out_dir} (the winners of {src_run} reduced again on {frames})"})
        finally:
            self.run_dir, self._ann_images, self.results = keep
        rec = {"run_dir": src_run, "out_dir": out_dir, "frames": frames, "run_frames": run_frames,
               "data_edits": edits, "run_data_edits": dict(self.cfg.data_edits or {}),
               "stitch_weights": weights, "weights_from": w_from, "verify_injections": v_from,
               "verify_sources": None if verify_sources is None else [s.as_tuple() for s in vs],
               "annuli": per_ann, "files": dict((stitched or {}).get("files") or {}),
               "verify_report": None if ver is None else ver.get("report"),
               "seconds": round(time.time() - t0, 1)}
        _atomic_write(os.path.join(out_dir, "rereduce.json"), json.dumps(rec, indent=1, default=_json_default))
        log(f"  wrote {out_dir} ({rec['seconds']:.0f} s)")
        return rec

    # ---------------------------------------------------------------- checkpoint
    #: Seconds between ROUTINE checkpoints (the save after each evaluation).  Rewriting the
    #: whole state after every evaluation of a fast run outran Dropbox: H2K (~1 s per
    #: evaluation, 2026-09-23) collected 49 "conflicted copies" of a 200 kB checkpoint.json in
    #: 20 minutes, and Dropbox once put its own older version back under the real name.
    CHECKPOINT_EVERY_S = 10.0

    def checkpoint(self, final_annulus: bool = False, routine: bool = False) -> None:
        """Save the search state to ``checkpoint.json``.

        ``routine`` marks the save after each evaluation.  It is taken every time, so it is
        exactly the state between two evaluations, but written at most every
        :attr:`CHECKPOINT_EVERY_S` seconds; one still held when the run stops, by Ctrl-C or
        an exception as much as by finishing, is written then (:meth:`flush_checkpoint`), so
        a resume is as exact as it always was.  Only a hard kill loses the held one, and
        then a resume replays those few seconds of evaluations from the same RNG state --
        the same configurations and draws -- while the readers of ``results.jsonl`` keep the
        first record of each index.

        Taking a save is cheap (:meth:`_checkpoint_snapshot`).  It is serialised only when
        it is written: ``json.dumps`` of the whole history after every evaluation cost a
        time that grew with the history, on the thread that sets a pipelined search's pace
        (4.2 MB at 5040 evaluations of NEAR2, 2026-10-12).

        Every other save is written at once: the end of an annulus, each post-annulus hook
        (they are how a resume knows which hooks have run), and each validation candidate,
        whose pickles pair with the RNG state saved here.
        """
        self._last_final = bool(final_annulus)
        snap = self._checkpoint_snapshot(final_annulus)
        if routine and time.time() - getattr(self, "_ckpt_written", 0.0) < self.CHECKPOINT_EVERY_S:
            self._ckpt_pending = snap
            return
        self._write_checkpoint(self._serialize_checkpoint(snap))

    def _checkpoint_snapshot(self, final_annulus: bool) -> Dict[str, Any]:
        """The state between two evaluations, taken without serialising it.

        Scalars, the RNG state and the small lists are copied.  The history and the
        finished annuli are kept by reference, with the history's length: until the next
        save written at once, which drops a held snapshot, the history only grows and the
        finished annuli stay as they are, so its first ``n_hist`` rows and those objects
        are this moment's state when :meth:`_serialize_checkpoint` reads them.
        """
        return {"config": self.cfg.to_dict(), "ia": self.ia, "contrast": self.contrast,
                "k_default": getattr(self, "_k_default", None), "recal_done": self._recal_done,
                "history": self.history, "n_hist": None if self.history is None else len(self.history),
                "annulus_done": final_annulus, "results": list(self.results),
                "rng_state": self.rng.bit_generator.state,
                "wall_s": time.time() - self.wall0 + self.wall_prev,
                "records_count": self.records_count,
                "hooks_done": {str(k): list(v) for k, v in self._hooks_done.items()},
                # proposals asked and dispatched but not yet in the history (RunConfig.batch_mode
                # "pipeline"): a resume reduces them again before it asks for anything new
                "pending": self._pending_state(),
                "extends": int(getattr(self, "_extends", 0))}

    def _serialize_checkpoint(self, snap: Dict[str, Any]) -> str:
        """The JSON text of a snapshot from :meth:`_checkpoint_snapshot`."""
        h = snap["history"]
        st = {
            "version": self.CKPT_VERSION, "klip_tpe_version": __version__,
            "config": snap["config"], "space": self.space.to_dict(),
            "ia": snap["ia"], "contrast": snap["contrast"], "k_default": snap["k_default"],
            "recal_done": snap["recal_done"],
            "history": None if h is None else h.to_dict(snap["n_hist"]),
            "annulus_done": snap["annulus_done"],
            "results": [None if r is None else r.to_dict() for r in snap["results"]],
            "rng_state": snap["rng_state"], "wall_s": snap["wall_s"], "records_count": snap["records_count"],
            "hooks_done": snap["hooks_done"], "pending": snap["pending"], "extends": snap["extends"],
        }
        return json.dumps(st, default=_json_default)

    def _write_checkpoint(self, text: str) -> None:
        """Write a serialized checkpoint so that a crash of the machine, not only of the
        process, leaves one to resume from.

        What it counts goes to disk first (:meth:`_sync_files`): the lines of
        ``results.jsonl`` and ``results.txt`` behind ``records_count``, and the products of
        a finished annulus or hook.  Then ``checkpoint.json`` is replaced durably, and the
        save it replaces moves to ``checkpoint.prev.json`` -- when it is a readable
        checkpoint, so that a damaged one never displaces a good copy.
        """
        self._sync_files()
        main = os.path.join(self.run_dir, "checkpoint.json")
        ok = getattr(self, "_ckpt_main_ok", None)
        if ok is None:                 # first write of this process: look at what is there
            try:
                ok = bool(_read_checkpoint(main))
            except Exception:
                ok = False
        _atomic_write(main, text, keep=os.path.join(self.run_dir, CHECKPOINT_PREV) if ok else None)
        self._ckpt_main_ok = True
        self._ckpt_written = time.time()
        self._ckpt_pending = None

    def flush_checkpoint(self) -> None:
        """Write the routine checkpoint the throttle is holding, if any.  Never raises: it
        runs on the way out of an interrupted run."""
        snap = getattr(self, "_ckpt_pending", None)
        if snap is None:
            return
        try:
            self._write_checkpoint(snap if isinstance(snap, str) else self._serialize_checkpoint(snap))
        except Exception as exc:
            try:
                self.log(f"  (last checkpoint not written: {exc!r})")
            except Exception:
                pass

    def _load_history_jsonl(self, ia: int) -> Tuple[Optional[History], Optional[float]]:
        """Rebuild the search history of annulus ``ia`` from ``results.jsonl`` (the last
        re-calibration segment, i.e. from the last record with ``index == 0``).  Returns
        ``(history, contrast)`` or ``(None, None)``."""
        path = os.path.join(self.run_dir, "results.jsonl")
        if not os.path.exists(path):
            return None, None
        recs = []
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("annulus") == ia and len(r.get("x", [])) == self.space.ndim:
                    recs.append(r)
        if not recs:
            return None, None
        starts = [i for i, r in enumerate(recs) if r.get("index") == 0]
        seg = recs[starts[-1]:] if starts else recs
        # One evaluation must enter the history once.  results.jsonl is append-only and
        # nothing guarantees an index appears in it once: two processes on the same run
        # directory both append (the registry guard is best-effort, and a stale entry lets a
        # second one start), and a resume can re-log what it replayed.  A duplicated point
        # is not harmless -- TPE fits a density to these points, so a repeat doubles that
        # configuration's weight and pulls later proposals towards it.  Keep the FIRST
        # record of each index: it is the one whose wall_s was measured without the second
        # process competing for cores.
        seen, uniq, ndup = set(), [], 0
        for r in seg:
            key = r.get("index")
            if key in seen:
                ndup += 1
                continue
            seen.add(key)
            uniq.append(r)
        if ndup:
            try:
                self.log(f"  results.jsonl: annulus {ia + 1} had {ndup} duplicated evaluation(s); "
                         f"kept the first of each ({len(uniq)} unique). "
                         f"scripts/dedup_results.py rewrites the file itself.")
            except Exception:
                pass
        seg = uniq
        h = History(self.space.ndim)
        for r in seg:
            h.append(np.asarray(r["x"], float), np.nan if r.get("score") is None else float(r["score"]),
                     {"phase": r.get("phase", "tpe"), "raw": r.get("raw_score"), "wall": r.get("wall_s"),
                      "sources": r.get("sources"), "k": r.get("k_used")})
        return h, float(seg[-1].get("contrast", self.cfg.contrast0))

    def annulus_history(self, ia: int) -> Optional[History]:
        """History of annulus ``ia``: the live one, a completed one kept in memory, or the
        one reconstructed from ``results.jsonl`` (resumed runs)."""
        if ia == self.ia and self.history is not None:
            return self.history
        h = self._ann_histories.get(ia)
        if h is None:
            h, _ = self._load_history_jsonl(ia)
            if h is not None:
                self._ann_histories[ia] = h
        return h

    def _register(self) -> None:
        """Put this run on the record, whoever started it.

        ``klip-tpe runs`` and a bare ``klip-tpe resume`` read that record, so a run built by
        a script rather than by the command line was invisible to both -- there was no way
        to see it was going, and no way to restart it without retyping how.  Best-effort:
        bookkeeping never stands between the user and a run.
        """
        try:
            from .registry import register
            register(self.run_dir)
        except Exception:
            pass

    def _apply_checkpoint(self, st: Dict[str, Any], project=None, same_frames: bool = True) -> None:
        """Adopt a checkpoint's state -- and its configuration -- onto this runner.

        The search configuration and space come from the checkpoint rather than from
        whatever the caller built, by design: the recorded history was produced under those
        settings and would not be comparable with evaluations made under any others.  A
        caller that wants different settings wants a new run directory.  ``same_frames=False``
        skips the check that the reducer reduces the run's frames, for a runner that only
        reads the run (:meth:`rereduce`).
        """
        self.cfg = RunConfig.from_dict(st["config"])
        self._apply_sampler_config()
        self.space = SearchSpace.from_dict(
            st["space"], project=project if project is not None else getattr(self.space, "project", None))
        self.rng.bit_generator.state = st["rng_state"]
        self.contrast = float(st["contrast"])
        self._k_default = st.get("k_default") or self.cfg.defaults.get("k_klip", 10)
        self._recal_done = int(st.get("recal_done", 0))
        self.records_count = int(st.get("records_count", 0))
        self._extends = int(st.get("extends") or 0)
        self.wall_prev = float(st.get("wall_s", 0.0))
        # A crash can leave half a line at the end of the logs, past every line a checkpoint
        # counted.  The next record would fuse with it, so it goes before anything appends.
        for name in ("results.jsonl", "results.txt"):
            try:
                said = durable.repair_tail(os.path.join(self.run_dir, name))
            except OSError as exc:
                said = f"could not check its last line ({exc!r})"
            if said:
                self.log(f"  {name}: {said}")
        self.results = [None if d is None else AnnulusResult.from_dict(d) for d in st.get("results", [])]
        self._hooks_done = {int(k): list(v) for k, v in (st.get("hooks_done") or {}).items()}
        self.ia = int(st["ia"])
        self.history = None if st["history"] is None else History.from_dict(st["history"])
        self._resume_pending = None if st.get("annulus_done") else st.get("pending")
        if st.get("annulus_done"):
            if self.history is not None:
                self._ann_histories[self.ia] = self.history
            self.ia += 1
            self.history = None
        for ia in range(self.ia):
            if ia not in self._ann_histories:
                h, _ = self._load_history_jsonl(ia)
                if h is not None:
                    self._ann_histories[ia] = h
        self._resumed = True
        self.log(f"resumed {self.run_dir}: annulus {self.ia + 1}, "
                 f"{0 if self.history is None else len(self.history)} evaluations, "
                 f"contrast {self.contrast:.3e}")
        if same_frames:
            self._check_data_edits()

    def _check_data_edits(self) -> None:
        """The frames this runner reduces, against the run's record of them
        (:attr:`RunConfig.data_edits`).  A new run records what its reducer reports, so the
        checkpoint holds it.  A resumed run has to reduce the same frames its evaluations
        were made on, and a reducer whose ``data_edits`` differ is refused."""
        have = _core_edits(getattr(self.reducer, "data_edits", None))
        want = _core_edits(self.cfg.data_edits)
        if not self._resumed:
            if want and want != have:
                raise ValueError(f"RunConfig.data_edits asks for {_describe_edits(want)}, and the reducer holds "
                                 f"{_describe_edits(have)}: build the reducer with them "
                                 f"(near.make_reducer(cube_name=..., drop_frames=..., subtract_sources=..., "
                                 f"inject_sources=...), or "
                                 f"reducer.subtract_sources(...) and reducer.inject_sources(...))")
            if have:
                self.cfg.data_edits = dict(getattr(self.reducer, "data_edits", None) or {})
                self.log(f"  data: {_describe_edits(have)} (recorded in the run's configuration)")
            return
        if want != have:
            raise ValueError(f"this run's evaluations were made on {_describe_edits(want)}, and the reducer here "
                             f"holds {_describe_edits(have)}.  A resume cannot change the frames.  Build the "
                             f"reducer as the run was started (klip-tpe resume and extend read --cube-name, "
                             f"--drop-frames, --subtract-sources and --inject-sources from the checkpoint when "
                             f"they are not typed), or start a new run directory.  klip-tpe rereduce reduces a "
                             f"finished run's winners on other frames.")

    def _auto_resume(self) -> None:
        """Continue a check-pointed run in this directory, unless told not to.

        Every run is resumable, however it was built.  ``Runner.resume`` covers the command
        line, but a script that constructs a Runner directly used to start a fresh search
        over an interrupted one -- which looks like a resume, writes into the same
        directory, and throws the history away.
        """
        if self._resumed or self.resume_mode == "never":
            return
        if not checkpoint_candidates(self.run_dir):
            return
        st, _ = load_checkpoint(self.run_dir, log=self.log)
        if st is None:                                 # a truncated checkpoint is not fatal
            self.log(f"  no readable checkpoint in {self.run_dir}; starting fresh")
            return
        self._apply_checkpoint(st, project=getattr(self.space, "project", None))

    @classmethod
    def resume(cls, run_dir: str, reducer: Reducer, objective: Objective, sampler: PositionSampler,
               project=None, throughput_fn=None, log=print, callbacks: Sequence[RunCallback] = (),
               same_frames: bool = True) -> "Runner":
        """Rebuild a runner from ``run_dir/checkpoint.json``.  The search configuration,
        space, contrast, RNG state and history come from the checkpoint -- no other
        keywords are accepted, by design.  Histories of already-completed annuli are
        reconstructed from ``results.jsonl`` on demand (:meth:`annulus_history`), so the
        final stitch / verification of a resumed multi-annulus run is complete.  The
        checkpoint is the furthest-along copy (:func:`load_checkpoint`).  The reducer has to
        reduce the frames the run was made on (:attr:`RunConfig.data_edits`), unless
        ``same_frames=False``: a runner that only reads a finished run, to reduce its winners
        on other frames (:meth:`rereduce`)."""
        st, _ = load_checkpoint(run_dir, log=log)
        if st is None:
            raise FileNotFoundError(f"no readable checkpoint in {run_dir} (checkpoint.json or a synced copy of it)")
        cfg = RunConfig.from_dict(st["config"])
        space = SearchSpace.from_dict(st["space"], project=project)
        r = cls(reducer, space, objective, sampler, cfg, run_dir, throughput_fn=throughput_fn, log=log,
                callbacks=callbacks, _from_checkpoint=True)
        r._apply_checkpoint(st, project=project, same_frames=same_frames)
        return r

    @classmethod
    def extend(cls, run_dir: str, reducer: Reducer, objective: Objective, sampler: PositionSampler,
               new_n_iter: Sequence[int], project=None, throughput_fn=None, log=print,
               callbacks: Sequence[RunCallback] = ()) -> "Runner":
        """Reopen a run with new per-annulus budgets (IDL ``extend_ann``, A §7.3).

        ``new_n_iter[ia]``: ``<= 0`` leaves annulus ``ia`` untouched; ``<= existing``
        re-validates the existing history in place (products rewritten); ``> existing``
        continues the search to the new total with the contrast frozen at the annulus'
        original value, splicing the history from the checkpoint (entry annulus) or
        from ``results.jsonl`` (last re-calibration segment) and re-scoring the prior
        best as the first continued evaluation.  The annulus' old contrast-curve samples
        are replaced by the re-validation (the run-level curve is rebuilt from the new
        :class:`AnnulusResult`).  Not available with ``opt_width``.  Finishes with the
        full stitch and returns the runner."""
        r = cls.resume(run_dir, reducer, objective, sampler, project=project, throughput_fn=throughput_fn,
                       log=log, callbacks=callbacks)
        if r.cfg.opt_width:
            raise ValueError("extend is not supported for opt_width runs")
        r._extends += 1                # its saves outrank the finished run's (_checkpoint_progress)
        st, _ = load_checkpoint(run_dir)               # the copy resume() just chose
        ck_ia = int(st["ia"])
        ck_hist = None if st.get("history") is None else History.from_dict(st["history"])
        nann = max(r.cfg.nann, len(new_n_iter))
        targets = [int(v) for v in new_n_iter] + [0] * (nann - len(new_n_iter))
        n_iter_list = [r.cfg.per_annulus(r.cfg.n_iter, ia) for ia in range(r.cfg.nann)]
        forced = list(r.cfg.calibration.forced or [])
        forced += [0.0] * (r.cfg.nann - len(forced))
        r._extend_mode = True
        try:
            for ia in range(r.cfg.nann):
                tgt = targets[ia]
                if tgt <= 0:
                    log(f"----- annulus {ia+1}: left untouched (extend target <= 0) -----")
                    continue
                if ia == ck_ia and ck_hist is not None:
                    hist, contrast = ck_hist, float(st["contrast"])
                else:
                    hist, contrast = r._load_history_jsonl(ia)
                if hist is None or len(hist) == 0:
                    log(f"----- annulus {ia+1}: no prior evaluations -- running it fresh to {tgt} -----")
                    n_iter_list[ia] = tgt
                    r.cfg.n_iter = list(n_iter_list)
                    r._extend_mode = False
                    r.run_annulus(ia)
                    r._extend_mode = True
                    continue
                res_old = r.results[ia] if ia < len(r.results) else None
                if res_old is not None and res_old.contrast > 0:
                    contrast = float(res_old.contrast)
                if contrast is None or contrast <= 0:
                    contrast = float(r.cfg.contrast0)
                forced[ia] = float(contrast)
                r.cfg.calibration.forced = list(forced)
                kdef = None
                if res_old is not None:
                    kdef = res_old.winner_config.get("params", {}).get("k_klip")
                if not kdef and isinstance(st.get("k_default"), (int, float)) and ia == ck_ia:
                    kdef = st["k_default"]
                nprev = len(hist)
                mode = "validate" if tgt <= nprev else "continue"
                n_iter_list[ia] = max(nprev, tgt)
                r.cfg.n_iter = list(n_iter_list)
                r.run_annulus(ia, extend={"mode": mode, "history": hist, "contrast": contrast,
                                          "k_default": kdef or r.cfg.defaults.get("k_klip", 10)})
            r.finish()
        except BaseException:
            r.flush_checkpoint()   # a held routine save: the last finished evaluation
            raise
        return r
