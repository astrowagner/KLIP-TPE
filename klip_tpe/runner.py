"""The run protocol: per annulus **calibrate -> search -> validate -> products**,
with a checkpoint after every evaluation and crash-resume as a first-class
feature (``config lives in the checkpoint``: ``Runner.resume`` takes the run
directory and the un-serialisable objects -- reducer, objective, sampler -- and
nothing else, so a resumed run can never silently mix configurations).

Outputs in ``run_dir``::

    run_setup.txt / run_setup.json   full configuration, search space, defaults
    results.txt                      one row per evaluation (IDL-compatible layout)
    results.jsonl                    one JSON record per evaluation (positions, per-source S/N ...)
    checkpoint.json                  search state (atomic; rewritten every evaluation)
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

import json
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .heartbeat import Heartbeat
from .metrics import Objective, Source, nanmedian_even, radprof, source_xy, star_center
from .optimizers import GridSearch, History, Optimizer, RandomSearch, TPE
from .positions import PositionSampler, n_sources_rule
from .reducer import EvalImages, PartitionedReducer, Reducer, ReductionRequest
from .space import Config, SearchSpace

__all__ = ["RunConfig", "CalibrationConfig", "ValidationConfig", "EvalRecord", "AnnulusResult", "Runner", "RunCallback"]

SCAN_MODES = ("scan", "scan_rescore")


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


@dataclass
class ValidationConfig:
    n_top: int = 3
    n_valid: int = 8
    distinct_tol: float = 1e-4
    commit: str = "median_trial"


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
    n_sources: Optional[int] = None       # None -> per-annulus rule
    inject_inset_fwhm: float = 1.0        # injection band inset from the annulus edges
    pair_area_midpoint: bool = True       # 2 sources -> both at sqrt((r_in^2+r_out^2)/2), 180 deg apart (IDL 2026-09-05)
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
    kfit_order: int = 2
    save_fits: bool = True
    save_eval_images: bool = True         # per-eval injected/clean crops (annulusNN/evals/, IDL evalNNNN_score_inj)
    batch: int = 1                        # proposals per ask (>1 = batched, same posterior)
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
    partition_weighting: Optional[str] = None
    fm_curve: bool = True
    fm_preview: bool = True              # live KLIP-FM preview at each new best (IDL: 'after 1st best')

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
        d["calibration"] = _from_dict_filtered(CalibrationConfig, d.get("calibration", {}))
        d["validation"] = _from_dict_filtered(ValidationConfig, d.get("validation", {}))
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
    partition_snr: Dict[str, Optional[float]]
    k_used: Optional[Any]
    contrast: float
    wall_s: float
    meta: Dict[str, Any] = field(default_factory=dict)

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


def _atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_fits(path: str, data: np.ndarray, header: Optional[Dict[str, Any]] = None,
                history: Sequence[str] = ()) -> None:
    """FITS writer with the provenance header of :func:`klip_tpe.stitch.fits_header`."""
    from .stitch import write_fits
    try:
        write_fits(path, data, header, history)
    except Exception:  # pragma: no cover  (no astropy)
        np.save(path.replace(".fits.gz", "").replace(".fits", "") + ".npy", data)


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
        self._calib_images: Optional[Dict[str, Any]] = None
        self._k_default = int(self.cfg.defaults.get("k_klip", 10) or 10)
        self._resumed = False
        self._extend_mode = False
        self._ann_images: Dict[int, Dict[str, Any]] = {}       # per-annulus winner images (memory cache)
        self._ann_histories: Dict[int, History] = {}           # completed annuli's histories
        self._pv_results: Dict[int, Dict[str, Any]] = {}       # param_verify outputs per annulus
        self._hooks_done: Dict[int, List[str]] = {}             # post-annulus hooks completed (checkpointed)
        self._last_final = False
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

    def _band(self, ia: int, cfg: Optional[Config] = None) -> Tuple[float, float]:
        """Injection band (arcsec), inset by ``inject_inset_fwhm`` FWHM from each edge
        (opt_width: both bounds at the zone mid-radius, A §3.1)."""
        a_in, a_out = self._zone(ia, cfg)
        if self.cfg.opt_width:
            rmid = 0.5 * (a_in + a_out)
            return rmid * self.pxscale, rmid * self.pxscale
        if self.cfg.pair_area_midpoint and self._nsrc(ia) == 2:
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

    def _nsrc(self, ia: int) -> int:
        return n_sources_rule(ia, self._annulus(ia)[1] * self.pxscale, self.cfg.n_sources)

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
        cfg2 = Config(params, per, cfg.selected, cfg.x)
        if (fm_sources is None and sources and not k_scan and self.reducer.supports_fm
                and getattr(self.objective.metric, "needs_fm", False)):
            # a forward-modelled matched filter needs this configuration's KLIP-FM response
            # of the very sources it is about to score (KLIP-FM is undefined under k_scan;
            # a reducer without it falls back to the numerical FM, see Runner._fm_for)
            fm_sources = sources
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
                               angle_convention=self.reducer.angle_convention, known=self._known())
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
        to the backends with no analytic FM -- pyKLIP, VIP, spaceKLIP.  Both scoring calls
        of :meth:`Objective.score_search` get the *same* array, so the injected and clean
        terms are always filtered with identical kernels.
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
        scan = self._reduce(cfgk, sources, k_scan=True, tag=tag + "_scan")
        cscan = None
        if self.objective.metric.needs_clean:
            cscan = self._reduce(cfgk, None, k_scan=True, tag=tag + "_cscan")
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
    def evaluate(self, x: np.ndarray, phase: str, contrast: Optional[float] = None,
                 sources: Optional[Sequence[Source]] = None, raw_only: bool = False,
                 tag: str = "eval") -> Tuple[EvalRecord, EvalImages, Optional[EvalImages]]:
        """Reduce + score one (already projected) vector.

        Sources are drawn fresh for this evaluation, as ``optimize_near_2_tpe`` does: the
        radial ladder is deterministic for the annulus, the azimuth anchor is not.  The
        caller may supply its own instead (calibration, validation and param-verify all do),
        and the non-default ``fixed_sources`` freezes them for the annulus instead.
        """
        t0 = time.time()
        contrast = self.contrast if contrast is None else float(contrast)
        cfg = self.space.decode(x)
        if sources is None:
            # The default path: a fresh draw per evaluation, from the band this
            # configuration defines (with opt_width that band is the configuration's own
            # mid-radius, which is why freezing is meaningless there and stays off).
            if self.cfg.fixed_sources and not self.cfg.opt_width:
                sources = self.search_sources(self.ia, contrast)
            else:
                rlo, rhi = self._band(self.ia, cfg)
                sources = self.sampler.sample(self._nsrc(self.ia), rlo, rhi, self.rng, contrast)
        else:
            sources = [Source(s.rho, s.theta, contrast) for s in sources]
        k_used = None
        clean = None
        kinfo: Dict[str, Any] = {}
        try:
            if self.cfg.k_mode in SCAN_MODES and self.reducer.supports_kscan:
                cfg, k_used, kinfo = self._kscan_pick(cfg, sources, tag)
                # re-score at the chosen k(s) on fresh positions, as IDL does: its k-scan
                # picks on one set (inj_rho/inj_theta) and then draws hrho/htheta for the
                # re-score reduction, so the scan's selection bias does not carry into the
                # recorded score.  Only the non-default fixed_sources keeps the one set.
                if not self.cfg.fixed_sources:
                    rlo, rhi = self._band(self.ia, cfg)
                    sources = self.sampler.sample(self._nsrc(self.ia), rlo, rhi, self.rng, contrast)
            need_clean = (self.objective.needs_clean and not raw_only) or self.objective.metric.needs_clean
            if need_clean and getattr(self.reducer, "max_workers", 1) >= 2:
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
            fmkw = self._fm_for(inj, clean)
            if raw_only:
                sr = self.objective.score_raw(inj.image, sources, None if clean is None else clean.image, **fmkw)
            else:
                sr = self.objective.score_search(inj.image, sources, None if clean is None else clean.image, **fmkw)
            failed = False
        except Exception as exc:  # a failed reduction scores as failed, never as a stale image
            self.log(f"  evaluation failed: {exc!r}")
            inj = EvalImages(np.full((2, 2), np.nan))
            sr = None
            failed = True
        if k_used is None:
            k_used = cfg.params.get("k_klip")
            if cfg.per_partition:
                k_used = {str(p): cfg.per_partition[p].get("k_klip") for p in cfg.selected}
        meta = {"failed": failed, "selected": [str(s) for s in cfg.selected],
                "zone": list(self._zone(self.ia, cfg)),
                "angle_convention": str(getattr(self.reducer, "angle_convention", "pa"))}
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
            partition_snr={} if failed else self._score_stack(inj.stack, inj.partitions, sources,
                                                             getattr(inj, "fm_stack", None)),
            k_used=k_used, contrast=contrast, wall_s=time.time() - t0, meta=meta)
        return rec, inj, clean

    # ------------------------------------------------------------ calibration
    def calibrate(self, ia: int) -> Tuple[float, int, Dict[str, Any]]:
        """Choose the injection contrast for annulus ``ia`` (see :class:`CalibrationConfig`).
        Returns ``(contrast, k_default, info)``."""
        cc = self.cfg.calibration
        self.hb.stage(f"annulus {ia + 1} calibration", annulus=ia)
        forced = 0.0
        if cc.forced is not None and len(cc.forced) > ia and cc.forced[ia] and cc.forced[ia] > 0:
            forced = float(cc.forced[ia])
        contrast = forced if forced > 0 else self.contrast
        info: Dict[str, Any] = {"forced": forced, "trials": []}
        x0 = self._default_vector()
        x0 = self._project(x0, is_random=False)
        cfg0 = self.space.decode(x0)
        for k, v in cc.overrides.items():
            cfg0.params[k] = v
            for d in cfg0.per_partition.values():
                d[k] = v
        kdef = int(cfg0.params.get("k_klip", self.cfg.defaults.get("k_klip", 10)) or 10)
        rlo, rhi = self._band(ia, cfg0)
        nsrc = self._nsrc(ia)
        # optional k-scan for the default config
        if cc.scan_k and ("k_klip" in self.space.bases or self.cfg.k_mode in SCAN_MODES) and self.reducer.supports_kscan:
            try:
                src = self.sampler.sample(nsrc, rlo, rhi, self.rng, contrast)
                kmax = self._k_scan_max()
                scan = self._reduce(self._with_k(cfg0, kmax), src, k_scan=True, tag="calib_scan")
                cscan = self._reduce(self._with_k(cfg0, kmax), None, k_scan=True, tag="calib_cscan") \
                    if self.objective.metric.needs_clean else None
                sc = np.array([self.objective.score_raw(scan.image[kk], src, None if cscan is None else cscan.image[kk]).score
                               for kk in range(scan.image.shape[0])])
                if np.any(np.isfinite(sc)):
                    kdef = int(np.nanargmax(sc)) + 1
                info["kscan"] = [None if not np.isfinite(v) else float(v) for v in sc]
                info["k_default"] = kdef
                self.log(f"  calibration k-scan: k_default = {kdef}")
            except Exception as exc:
                self.log(f"  calibration k-scan failed ({exc!r}); keeping k = {kdef}")
        cfgk = self._with_k(cfg0, kdef)
        info["k_default"] = kdef
        self._calib_images = None
        msnr = np.nan
        for trial in range(cc.max_trials):
            vals = []
            src = []
            for _ in range(cc.n_remeasure):
                src = self.sampler.sample(nsrc, rlo, rhi, self.rng, contrast)
                try:
                    inj = self._reduce(cfgk, src, tag="calib_inj")
                    clean = self._reduce(cfgk, None, tag="calib_clean") if self.objective.needs_clean else None
                    r = self.objective.score_search(inj.image, src, None if clean is None else clean.image,
                                                    **self._fm_for(inj, clean))
                    vals.append(r.score)
                    # keep the last measurement for the display (on_calibration images)
                    self._calib_images = {
                        "inj": inj, "clean": clean, "sources": list(src), "contrast": contrast, "config": cfgk,
                        "per_source": [None if not np.isfinite(v) else float(v) for v in r.per_source],
                        "clean_per_source": None if r.clean_per_source is None
                        else [None if not np.isfinite(v) else float(v) for v in r.clean_per_source]}
                except Exception as exc:
                    self.log(f"  calibration reduction failed: {exc!r}")
                    vals.append(np.nan)
            msnr = nanmedian_even(vals)
            info["trials"].append({"contrast": contrast, "snr": None if not np.isfinite(msnr) else float(msnr),
                                   "values": [None if not np.isfinite(v) else float(v) for v in vals]})
            self.log(f"  calibration trial {trial+1}: contrast={contrast:.3e}  median S/N={msnr:.2f}")
            self._emit("on_calibration_trial", ia, trial + 1, contrast, msnr, self._calib_images)
            if self.cfg.write_setup_files:
                self._write_setup_file(os.path.join(self._ann_dir(ia), f"calib{trial+1:04d}_setup.txt"), "calib",
                                       ia, trial + 1, cfgk, src, contrast, msnr, kdef)
            if forced > 0 or not np.isfinite(msnr):
                break
            if cc.target[0] <= msnr <= cc.target[1]:
                break
            fac = float(np.clip(cc.aim / max(msnr, 0.5), cc.step_clip[0], cc.step_clip[1]))
            contrast *= fac
            if cc.ceiling is not None:
                contrast = min(contrast, cc.ceiling)
        info["contrast"] = contrast
        info["snr"] = None if not np.isfinite(msnr) else float(msnr)
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
        new = not os.path.exists(rpath)
        with open(rpath, "a") as f:
            if new:
                f.write("# annulus iter phase " + " ".join(names) + "   k_klip   score   raw_score   wall_s\n")
            kstr = rec.k_used if not isinstance(rec.k_used, dict) else "[" + ",".join(
                str(v) for v in rec.k_used.values()) + "]"
            sc = "nan" if rec.score is None else f"{rec.score:.4f}"
            rw = "nan" if rec.raw_score is None else f"{rec.raw_score:.4f}"
            f.write(f"{rec.annulus+1:3d} {rec.index+1:5d} {rec.phase:<7s} "
                    + " ".join(f"{v:9.4f}" for v in rec.x) + f"   {kstr}   {sc}   {rw}   {rec.wall_s:.1f}\n")
        with open(os.path.join(self.run_dir, "results.jsonl"), "a") as f:
            f.write(rec.to_json() + "\n")
        self.records_count += 1

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
        while len(self.history) < n_iter:
            nb = max(1, min(c.batch, n_iter - len(self.history)))
            t_loop = time.time()
            props = opt.ask(self.history, self.rng, nb)
            for pr in props:
                phase = pr.flags.get("phase", "tpe")
                is_random = phase in ("warmup", "explore", "random")
                x = self._project(pr.x, is_random=is_random)   # optimizer already tied link_params
                self._hb_eval(ia, len(self.history) + 1, n_iter, phase)
                rec, inj, clean = self.evaluate(x, phase, tag=f"a{ia+1}_e{len(self.history)+1}")
                i = self._append(rec)
                bi, bs = self.history.best()
                if i == bi and inj.image is not None and inj.image.ndim == 2:
                    self._set_best_images(rec, inj, clean)
                self._save_eval_images(rec, inj, clean)
                self._emit("on_eval", rec, inj, clean, i == bi)
                elapsed = time.time() - self.wall0 + self.wall_prev
                loop_s = time.time() - t_loop                    # propose + reduce + score + display
                self.log(f"[ann {ia+1}] eval {i+1}/{n_iter} {phase:<7s} score={rec.score if rec.score is None else round(rec.score,3)}"
                         f" raw={rec.raw_score if rec.raw_score is None else round(rec.raw_score,3)}"
                         f" best={bs:.3f}@{bi+1}  contrast={self.contrast:.2e}  reduce {rec.wall_s:.0f}s / loop {loop_s:.0f}s"
                         f"  ({elapsed/3600:.2f} h)")
                self.checkpoint()
                if c.stitch_every and c.stitch_every > 0 and (i + 1) % int(c.stitch_every) == 0:
                    self._guard("running stitch", self._running_stitch, ia)
                t_loop = time.time()
                # re-calibration revisit
                cc = c.calibration
                if (not self._extend_mode and i + 1 == cc.recal_check and self._recal_done < cc.recal_budget
                        and cc.recal_budget > 0
                        and not (cc.forced and len(cc.forced) > ia and cc.forced[ia] and cc.forced[ia] > 0)):
                    yv = self.history.y[1:cc.recal_check] if c.seed_default else self.history.y[:cc.recal_check]
                    yv = yv[np.isfinite(yv)]
                    if yv.size >= 3:
                        top = np.sort(yv)[::-1][:cc.recal_ntop]
                        mwu = float(np.median(top))
                        if not (cc.target[0] <= mwu <= cc.target[1]):
                            if mwu < cc.target[0] and cc.ceiling is not None and self.contrast >= 0.999 * cc.ceiling:
                                self.log("  re-cal: contrast already at ceiling; continuing")
                            else:
                                fac = float(np.clip(cc.aim / max(mwu, 0.1), cc.step_clip[0], cc.step_clip[1]))
                                newc = self.contrast * fac
                                if cc.ceiling is not None:
                                    newc = min(newc, cc.ceiling)
                                self.log(f"  re-cal: median top-{cc.recal_ntop} S/N {mwu:.2f} outside "
                                         f"{cc.target}; contrast {self.contrast:.2e} -> {newc:.2e}; restarting annulus")
                                self.contrast = newc
                                self._recal_done += 1
                                self._reset_history(ia)
                                return self._search_annulus(ia, 0)

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
            self.hb.stage(f"annulus {ia + 1} eval {ev}/{n_iter} ({phase})",
                          annulus=ia, eval=ev, n_eval=self.records_count)
        except Exception:
            pass

    def _reset_history(self, ia: int) -> None:
        self.history = History(self.space.ndim)
        if self.cfg.seed_default:
            x0 = self._project(self._default_vector(self._k_default), is_random=False)
            rec, inj, clean = self.evaluate(x0, "seed", tag=f"a{ia+1}_seed")
            self._append(rec)
            self._set_best_images(rec, inj, clean)
            self.checkpoint()
            self._save_eval_images(rec, inj, clean)
            self._emit("on_eval", rec, inj, clean, True)

    # ------------------------------------------------------------- validation
    def validate(self, ia: int) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """Re-score the top ``n_top`` distinct configs on ``n_valid`` fresh injection
        sets with the RAW metric; elect the winner on the validated median."""
        vc = self.cfg.validation
        self.hb.stage(f"annulus {ia + 1} validation", annulus=ia)
        order = self.history.order()
        cands: List[int] = []
        for e in order:
            if all(self.space.distinct(self.history.X[e], self.history.X[c], vc.distinct_tol) for c in cands):
                cands.append(int(e))
            if len(cands) >= vc.n_top:
                break
        table: List[Dict[str, Any]] = []
        vbest, vwin = -np.inf, -1
        winner: Dict[str, Any] = {}
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
                if np.isfinite(vscore) and vscore > vbest:
                    vbest, vwin = vscore, ci
                    winner = dict(cand, x=np.asarray(row["x"], float), config=cfg)
                continue
            trials, trial_imgs, trial_src, trial_ps, samples_r, samples_s = [], [], [], [], [], []
            clean = None
            part = self._load_validation_trials(ia, ci, cands)
            if part is not None:       # interrupted inside this candidate: continue after its last trial
                clean = part["clean"]
                trials, trial_imgs, trial_src, trial_ps = part["trials"], part["trial_imgs"], part["trial_src"], part["trial_ps"]
                samples_r, samples_s = part["samples_r"], part["samples_s"]
                self.log(f"  validation cand {ci+1}/{len(cands)}: resuming after trial {len(trials)}/{vc.n_valid}")
            else:
                try:
                    clean = self._reduce(cfg, None, tag=f"a{ia+1}_val{ci}_clean")
                except Exception as exc:
                    self.log(f"  validation clean reduction failed: {exc!r}")
            for t in range(len(trials), vc.n_valid):
                rlo, rhi = self._band(ia, cfg)
                src = self.sampler.sample(self._nsrc(ia), rlo, rhi, self.rng, self.contrast)
                self.hb.stage(f"annulus {ia + 1} validation: candidate {ci + 1}/{len(cands)} "
                              f"trial {t + 1}/{vc.n_valid}", annulus=ia)
                try:
                    inj = self._reduce(cfg, src, tag=f"a{ia+1}_val{ci}_t{t}")
                    r = self.objective.score_raw(inj.image, src, None if clean is None else clean.image,
                                                 **self._fm_for(inj, clean))
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
                    trials.append(np.nan)
                    trial_imgs.append(None)
                    trial_src.append(src)
                    trial_ps.append([])
                self._save_validation_trials(ia, ci, cands, clean, trials, trial_imgs, trial_src, trial_ps, samples_r, samples_s)
            vs = np.array(trials, float)
            vscore = nanmedian_even(vs)
            commit = -1
            if np.isfinite(vscore):
                fin = np.flatnonzero(np.isfinite(vs))
                commit = int(fin[np.argmin(np.abs(vs[fin] - vscore))])
            row = {"candidate": ci, "eval_index": e, "search_score": float(self.history.y[e]),
                   "validated_score": None if not np.isfinite(vscore) else float(vscore),
                   "trials": [None if not np.isfinite(v) else float(v) for v in vs],
                   "x": [float(v) for v in x], "config": cfg.to_dict(), "committed_trial": commit}
            table.append(row)
            self.log(f"  validation cand {ci+1}/{len(cands)} (eval {e+1}, search {self.history.y[e]:.3f}) "
                     f"-> validated {vscore:.3f}")
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
            if np.isfinite(vscore) and vscore > vbest:
                vbest, vwin = vscore, ci
                winner = dict(cand, x=x.copy(), config=cfg)
            with open(os.path.join(self._ann_dir(ia), "validation.json"), "w") as f:
                json.dump(table, f, indent=1, default=_json_default)
            self._save_validation_partial(ia, ci, cands, cand)
            try:
                os.remove(self._val_trials_pkl(ia, ci))
            except OSError:
                pass
        return table, vwin, winner

    # validation resume helpers -------------------------------------------------
    def _val_pkl(self, ia: int, ci: int) -> str:
        return os.path.join(self._ann_dir(ia), f"val_cand{ci+1:02d}.pkl")

    def _save_validation_partial(self, ia: int, ci: int, cands: List[int], cand: Dict[str, Any]) -> None:
        try:
            with open(self._val_pkl(ia, ci), "wb") as f:
                pickle.dump({"cands": list(cands), "cand": cand}, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.checkpoint()          # RNG state after this candidate's fresh injection draws
        except Exception as exc:       # scratch only -- never fatal
            self.log(f"  (validation checkpoint not written: {exc!r})")

    def _val_trials_pkl(self, ia: int, ci: int) -> str:
        return os.path.join(self._ann_dir(ia), f"val_cand{ci+1:02d}_trials.pkl")

    def _save_validation_trials(self, ia, ci, cands, clean, trials, trial_imgs, trial_src, trial_ps, samples_r, samples_s) -> None:
        """Intra-candidate checkpoint (one validation trial can take minutes): the clean
        reduction and every finished trial, plus checkpoint.json for the RNG state."""
        try:
            with open(self._val_trials_pkl(ia, ci), "wb") as f:
                pickle.dump({"cands": list(cands), "clean": clean, "trials": list(trials), "trial_imgs": list(trial_imgs),
                             "trial_src": list(trial_src), "trial_ps": list(trial_ps),
                             "samples_r": list(samples_r), "samples_s": list(samples_s)}, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.checkpoint()
        except Exception as exc:
            self.log(f"  (validation trial checkpoint not written: {exc!r})")

    def _load_validation_trials(self, ia: int, ci: int, cands: List[int]) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_resumed", False):
            return None
        pk = self._val_trials_pkl(ia, ci)
        if not os.path.exists(pk):
            return None
        try:
            with open(pk, "rb") as f:
                d = pickle.load(f)
        except Exception:
            return None
        if list(d.get("cands", [])) != list(cands) or not d.get("trials"):
            return None
        return d

    def _load_validation_partial(self, ia: int, cands: List[int]) -> Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]]:
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
                    self.checkpoint()
                    self._save_eval_images(rec, inj, clean)
                    self._emit("on_eval", rec, inj, clean, True)
                self._search_annulus(ia, len(self.history))
            else:
                self.log(f"=== annulus {ia+1}: RE-VALIDATE {len(self.history)} existing evals in place "
                         f"(contrast {self.contrast:.3e})")
        elif resume_index is None:
            label = f"{ia+1}" if self.cfg.opt_width else f"{ia+1}/{self.cfg.nann}"
            self.log(f"=== annulus {label}: [{a_in:.1f}, {a_out:.1f}] px, "
                     f"{self._nsrc(ia)} sources, band {self._band(ia)[0]:.3f}-{self._band(ia)[1]:.3f}\"")
            if ia > 0 and not cc.warm_start_previous:
                self.contrast = float(self.cfg.contrast0)
            contrast, kdef, cinfo = self.calibrate(ia)
            self.contrast = contrast
            self._k_default = kdef
            self._recal_done = 0
            with open(os.path.join(self._ann_dir(ia), "calibration.json"), "w") as f:
                json.dump(cinfo, f, indent=1, default=_json_default)
            self._emit("on_calibration", cinfo, self._calib_images)
            self._reset_history(ia)
            self._search_annulus(ia, len(self.history))
        else:
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
            self.log(f"  ** every one of the {ntot} evaluations of annulus {ia+1} FAILED -- there "
                     f"is no winner and no best image.  The usual cause is an injection model "
                     f"that does not span the annulus; the reason is on the 'evaluation failed:' "
                     f"lines above.")
        elif ntot and nok < ntot:
            self.log(f"  note: {ntot - nok}/{ntot} evaluations failed")
        self.log(f"  search done: best {bs:.3f} at eval {bi+1}")
        table, vwin, winner = ([], -1, {})
        if self.cfg.validation.n_valid >= 1 and self.cfg.validation.n_top >= 1:
            table, vwin, winner = self.validate(ia)
            self._emit("on_validation", table, winner)
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
                    if clean_img is None:
                        clean_img = fm["clean"]
                    winner["fm"] = fm
            except Exception as exc:
                self.log(f"  KLIP-FM cross-check failed: {exc!r}")
                fm_curve = None
        if self.cfg.save_fits:
            if clean_img is not None:
                _write_fits(os.path.join(d, "best_clean.fits"), clean_img, hdr)
                self._best_images["clean"] = clean_img
            if inj_img is not None:
                _write_fits(os.path.join(d, "best_inj.fits"), inj_img, hdr_inj)
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
            k_default=winner.get("k_default"))
        with open(os.path.join(d, "winner.json"), "w") as f:
            json.dump(res.to_dict(), f, indent=1, default=_json_default)
        if self.cfg.write_setup_files:
            self._write_setup_file(os.path.join(d, "final_setup.txt"), "final", ia, int(winner["eval_index"]) + 1,
                                   cfg, sources, self.contrast, float(winner["score"]), k_used,
                                   extra={"validated": int(validated), "search_score": f"{bs:.3f}"},
                                   weights=None if ev_c is None else getattr(ev_c, "weights", None))
        self.log(f"  annulus {ia+1} winner: eval {res.winner_index+1}, "
                 f"{'validated' if validated else 'search'} score {res.winner_score:.3f}")
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
               "IMGTYPE": "running equal-weight stitch (radprof-flattened)", "EDGES": ",".join(f"{z[0]:.1f}" for z in zones)
               + f",{zones[-1][1]:.1f}"}
        st = radprof(stitch_tiles(cleans, zones))
        files = {"clean": os.path.join(self.run_dir, "klip_stitched_running.fits")}
        _write_fits(files["clean"], st, hdr)
        snr = snr_map(st, self.fwhm)
        files["snr"] = os.path.join(self.run_dir, "klip_stitched_running_snr.fits")
        _write_fits(files["snr"], snr, dict(hdr, IMGTYPE="running stitched S/N map (Mawet)"))
        out = {"clean": st, "snr": snr, "zones": zones}
        if any(i is not None for i in injs):
            sti = radprof(stitch_tiles([i if i is not None else c for i, c in zip(injs, cleans)], zones))
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
            cube, ids = stitch_stacks(stacks, parts, zones, w)
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
                                           return_maps=True)
            path = os.path.join(d, "verify_report.txt")
            write_verify_table(path, rows, maps.get("limits"), header={"run": os.path.abspath(self.run_dir)})
            out.update(rows=rows, limits=maps.get("limits"), report=path)
        # per-eval injection limit curve of this annulus + the run-level accumulation
        rows_a = self._eval_injection_rows([ia])
        lim = limit_tables(rows_a, self.fwhm, self.pxscale) if rows_a else None
        if lim is not None:
            write_curve_table(os.path.join(d, "verify_curve.txt"), lim,
                              header=f"annulus {ia+1}: {len(rows_a)} injections from {self.cfg.per_annulus(self.cfg.n_iter, ia)} search evals")
            out["curve"] = lim
        done = [r.annulus for r in self.results if r is not None]
        rows_all = self._eval_injection_rows(done)
        lim_all = limit_tables(rows_all, self.fwhm, self.pxscale) if rows_all else None
        if lim_all is not None:
            write_curve_table(os.path.join(self.run_dir, "verify_curve.txt"), lim_all,
                              header=f"annuli {','.join(str(a+1) for a in done)}: {len(rows_all)} search-eval injections")
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
                z = np.load(cpath)
                return z["clean"], z["inj"]
            cfg = self.space.decode(np.asarray(x, float))
            if self.cfg.k_mode in SCAN_MODES:
                m = np.flatnonzero(np.all(np.abs(hist.X - np.asarray(x, float)[None, :]) < 1e-9, axis=1))
                if m.size:
                    cfg = self._apply_k_flag(cfg, hist.flags[int(m[-1])].get("k"))
            clean = self._reduce(cfg, None, tag=f"a{ia+1}_pv_clean", zone=(rlo, rhi))
            inj = self._reduce(cfg, sources, tag=f"a{ia+1}_pv_inj", zone=(rlo, rhi))
            try:
                np.savez(cpath, clean=clean.image, inj=inj.image)
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
                           angle_convention=self.reducer.angle_convention, log=self.log)
        if res is not None:
            res["zone"] = (rlo, rhi)
            self._pv_results[ia] = res
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
        return stitch_param_maps(maps, zones, out_dir=self.run_dir)

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
                                   verify_kwargs={"n_boot": int(self.cfg.verify_n_boot), "rng": rng}, **kw)
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
        self._register()
        self.hb.start()                # from here on the directory says whether this is alive
        try:
            out = self._run()
        except BaseException:
            self.hb.stop()             # keep the stage it died in: that is the diagnosis
            raise
        self.hb.stop("finished")
        return out

    def _run(self) -> List[AnnulusResult]:
        if not self._resumed:
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
                        fm_curves=[r.fm_curve for r in results])
        with open(os.path.join(self.run_dir, "final_results.json"), "w") as f:
            json.dump(out, f, indent=1, default=_json_default)
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

    def _final_stitch(self, results: Sequence[AnnulusResult]) -> Optional[Dict[str, Any]]:
        from .products import snr_map
        from .stitch import seam_pixels, stitch_stacks, stitch_tiles, write_params_table
        items = [(r, self._annulus_images(r.annulus)) for r in results]
        items = [(r, d) for r, d in items if d is not None and d.get("clean") is not None]
        if not items:
            return None
        cleans = [d["clean"] for _, d in items]
        zones = [tuple(d["zone"]) for _, d in items]
        edges = [z[0] for z in zones] + [zones[-1][1]]
        w = self._stitch_weights(cleans, zones)
        st = stitch_tiles(cleans, zones, w)
        # seam patching: pixels inside the searched span left NaN -> padded re-run of that annulus' winner
        seam = seam_pixels(st, edges[0], edges[-1])
        npatch = 0
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
                    zone = (max(zones[j][0] - 3.0, 0.0), min(zones[j][1] + 3.0, self.cfg.r_cap))
                    img_p = self._reduce(cfg, None, tag=f"a{r.annulus+1}_seam", zone=zone).image
                    if img_p is not None and img_p.shape == st.shape:
                        st[m] = img_p[m]
                        npatch += int(m.sum())
                except Exception as exc:
                    self.log(f"  seam patch of annulus {r.annulus+1} failed: {exc!r}")
            self.ia = ia_keep
            self.log(f"  seam patching: {int(seam.sum())} pixel(s), {npatch} filled")
        st_f = radprof(st)
        hist = ["per-annulus best (edges px | k bin nang filt angsep anglemax | SNR | partitions):"]
        for r, d in items:
            p = r.winner_config.get("params", {})
            ptxt = " ".join(f"{ab}{p[k]}" for k, ab in (("k_klip", "k"), ("bin", "b"), ("n_ang", "na"), ("filter", "f"),
                                                       ("angsep", "as"), ("anglemax", "am")) if p.get(k) is not None)
            hist.append(f"  a{r.annulus+1} [{r.inrad:5.1f}-{r.outrad:5.1f}] {ptxt} "
                        f"SNR{r.winner_score:6.2f} c{r.contrast:.2E} n={','.join(r.partitions) or 'all'}")
        hdr = {"NANN": len(items), "PHASE": "final", "IMGTYPE": "radially-optimized stitch (radprof-flattened)",
               "LEGACY": int(self.cfg.legacy_stitch), "NSEAM": int(seam.sum()), "NPATCH": npatch,
               "EDGES": ",".join(f"{e:.1f}" for e in edges), "WEIGHTS": ",".join(f"{v:.3f}" for v in w),
               "METRIC": getattr(self.objective.metric, "name", "metric"), "KMODE": self.cfg.k_mode,
               "NITER": str(self.cfg.n_iter), "NINIT": str(self.cfg.n_init), "NVALID": self.cfg.validation.n_valid,
               "NTOP": self.cfg.validation.n_top, "GAMMA": self.cfg.gamma, "NCAND": self.cfg.ncand,
               "SEED": self.cfg.seed, "RUNDIR": os.path.basename(os.path.abspath(self.run_dir)),
               "OPTWIDTH": int(self.cfg.opt_width)}
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
            sti = radprof(stitch_tiles([d["inj"] if d.get("inj") is not None else d["clean"] for _, d in items], zones, w))
            hi = dict(hdr, IMGTYPE="validated-winner injected stitch (radprof-flattened)", INJNSRC=len(allsrc))
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
            cube, ids = stitch_stacks(stacks, [d.get("parts") for _, d in items], zones, w)
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

    def _final_verify(self, stitched: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """``near2_verify`` on the stitched per-partition cubes (known sources as
        candidates, all validated injections as the calibration block)."""
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
        rows, maps = verify_candidates(stack, stitched.get("night_ids"), known, self.fwhm, self.pxscale, known=known,
                                       injected=None if (istack is None or not allsrc) else (istack, allsrc),
                                       n_boot=int(self.cfg.verify_n_boot), rng=rng,
                                       angle_convention=self.reducer.angle_convention, return_maps=True)
        path = os.path.join(self.run_dir, "stitched_verify_report.txt")
        write_verify_table(path, rows, maps.get("limits"), header={"run": os.path.abspath(self.run_dir)})
        out = {"rows": rows, "limits": maps.get("limits"), "report": path}
        self._emit("on_verify", -1, "verify", out)
        return out

    # ---------------------------------------------------------------- checkpoint
    def checkpoint(self, final_annulus: bool = False) -> None:
        self._last_final = bool(final_annulus)
        st = {
            "version": self.CKPT_VERSION, "klip_tpe_version": __version__,
            "config": self.cfg.to_dict(), "space": self.space.to_dict(),
            "ia": self.ia, "contrast": self.contrast, "k_default": getattr(self, "_k_default", None),
            "recal_done": self._recal_done,
            "history": None if self.history is None else self.history.to_dict(),
            "annulus_done": final_annulus,
            "results": [None if r is None else r.to_dict() for r in self.results],
            "rng_state": self.rng.bit_generator.state,
            "wall_s": time.time() - self.wall0 + self.wall_prev,
            "records_count": self.records_count,
            "hooks_done": {str(k): list(v) for k, v in self._hooks_done.items()},
        }
        _atomic_write(os.path.join(self.run_dir, "checkpoint.json"), json.dumps(st, default=_json_default))

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

    def _apply_checkpoint(self, st: Dict[str, Any], project=None) -> None:
        """Adopt a checkpoint's state -- and its configuration -- onto this runner.

        The search configuration and space come from the checkpoint rather than from
        whatever the caller built, by design: the recorded history was produced under those
        settings and would not be comparable with evaluations made under any others.  A
        caller that wants different settings wants a new run directory.
        """
        self.cfg = RunConfig.from_dict(st["config"])
        self.space = SearchSpace.from_dict(
            st["space"], project=project if project is not None else getattr(self.space, "project", None))
        self.rng.bit_generator.state = st["rng_state"]
        self.contrast = float(st["contrast"])
        self._k_default = st.get("k_default") or self.cfg.defaults.get("k_klip", 10)
        self._recal_done = int(st.get("recal_done", 0))
        self.records_count = int(st.get("records_count", 0))
        self.wall_prev = float(st.get("wall_s", 0.0))
        self.results = [None if d is None else AnnulusResult.from_dict(d) for d in st.get("results", [])]
        self._hooks_done = {int(k): list(v) for k, v in (st.get("hooks_done") or {}).items()}
        self.ia = int(st["ia"])
        self.history = None if st["history"] is None else History.from_dict(st["history"])
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

    def _auto_resume(self) -> None:
        """Continue a check-pointed run in this directory, unless told not to.

        Every run is resumable, however it was built.  ``Runner.resume`` covers the command
        line, but a script that constructs a Runner directly used to start a fresh search
        over an interrupted one -- which looks like a resume, writes into the same
        directory, and throws the history away.
        """
        if self._resumed or self.resume_mode == "never":
            return
        ck = os.path.join(self.run_dir, "checkpoint.json")
        if not os.path.exists(ck):
            return
        try:
            with open(ck) as f:
                st = json.load(f)
        except Exception as exc:                       # a truncated checkpoint is not fatal
            self.log(f"  checkpoint in {self.run_dir} unreadable ({exc!r}); starting fresh")
            return
        self._apply_checkpoint(st, project=getattr(self.space, "project", None))

    @classmethod
    def resume(cls, run_dir: str, reducer: Reducer, objective: Objective, sampler: PositionSampler,
               project=None, throughput_fn=None, log=print, callbacks: Sequence[RunCallback] = ()) -> "Runner":
        """Rebuild a runner from ``run_dir/checkpoint.json``.  The search configuration,
        space, contrast, RNG state and history come from the checkpoint -- no other
        keywords are accepted, by design.  Histories of already-completed annuli are
        reconstructed from ``results.jsonl`` on demand (:meth:`annulus_history`), so the
        final stitch / verification of a resumed multi-annulus run is complete."""
        with open(os.path.join(run_dir, "checkpoint.json")) as f:
            st = json.load(f)
        cfg = RunConfig.from_dict(st["config"])
        space = SearchSpace.from_dict(st["space"], project=project)
        r = cls(reducer, space, objective, sampler, cfg, run_dir, throughput_fn=throughput_fn, log=log,
                callbacks=callbacks, _from_checkpoint=True)
        r._apply_checkpoint(st, project=project)
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
        with open(os.path.join(run_dir, "checkpoint.json")) as f:
            st = json.load(f)
        ck_ia = int(st["ia"])
        ck_hist = None if st.get("history") is None else History.from_dict(st["history"])
        nann = max(r.cfg.nann, len(new_n_iter))
        targets = [int(v) for v in new_n_iter] + [0] * (nann - len(new_n_iter))
        n_iter_list = [r.cfg.per_annulus(r.cfg.n_iter, ia) for ia in range(r.cfg.nann)]
        forced = list(r.cfg.calibration.forced or [])
        forced += [0.0] * (r.cfg.nann - len(forced))
        r._extend_mode = True
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
        return r
