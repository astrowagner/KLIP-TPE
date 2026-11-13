"""Re-score of each annulus' candidate configurations on the stitch, the first piece of ``refine``.

A finished run, the base, gives the stitch: its annuli, tiles, pads and weights, and its winners for
every annulus but the one being scored.  The candidates for annulus k are configurations of the base
run and of other runs of the same setup: their winners, their validation's entrants and, when asked,
the best of their histories.  Each candidate is reduced on common draws.  A draw puts sources in
tile k and in tiles k - 1 and k + 1, each tile's at the contrast that makes the base winners read
S/N 5 to 8 on the stitch (:meth:`StitchScorer.calibrate`).

The annuli whose zones hold any of the sources reduced together are reduced with them: the base
winners for the neighbors and the candidate for annulus k, as the final stitch reduces every annulus
with every source (``RunConfig.stitch_injections = "all"``).  The other annuli take the base winners'
clean images, which those sources do not reach.  The injected stitch is built as the final stitch
builds it, with the base run's tiles, pads, weights and flattening, and each source is measured as
the stitched verify measures it (Path A of :func:`klip_tpe.verify.verify_injections`).  The seam
patches of the final stitch go into its clean image only, which the injected sources are not
measured on.

With ``tiles="apart"``, the default for a new directory, each tile's sources go into reductions and
a stitch of their own and are measured there, as their contrast was set.  ``"together"`` puts all of
a draw's sources in each reduction.

A composite base (``base_winners``) takes other configurations as the base winners of some annuli,
for the stitch, the neighbors' reductions and the paired differences.  ``draws_from`` takes the
draws, their rule and the contrasts of an earlier re-score of the same base run, and compares each
annulus' base with that re-score's, draw for draw.

The runs are only read.  :func:`rescore` writes ``rescore.txt`` and ``rescore.json`` to its own
directory, with a ``state/`` directory from which an interrupted re-score resumes.

Notes
-----
The re-score follows NEAR2's notes of 2026-11-06 to 2026-11-09.

Each tile's contrast is set with that tile's sources alone, and the inner tiles' are brighter: on
NEAR2's s7 tile 2's sources were 7 and 28 times as bright as tiles 3 and 4's.  Reduced together, the
bright sources' residuals and the KL modes they take reach the faint ones (NEAR2, 2026-11-10: s7's
winner read 3.47 and 1.77 in tiles 3 and 4 of annulus 3's draws, against the calibration's 5 to 8).
Hence ``tiles="apart"``.  Before, the re-score reduced each draw's sources together, as
``"together"`` still does.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shlex
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import Source

__all__ = ["Entrant", "Pick", "StitchScorer", "config_key", "run_entrants", "entrant_at", "read_picks",
           "resolve_picks", "parse_annuli", "rescore", "rescore_plan", "paired_summary", "PlacementError",
           "RescoreStop", "TILES_MODES"]

#: the S/N the base winners should read on the stitch, at the calibrated contrasts
TARGET = (5.0, 8.0)

#: how a draw's tiles are reduced: each tile's sources apart, or all of them together
TILES_MODES = ("apart", "together")

#: the clearance between the tiles' sources (FWHM) a draw falls back to when the sampler's own
#: finds no placement (NEAR2, 2026-11-10: s7's tiles are 4.7 px wide against a FWHM of 6.4 px)
RELAXED_FWHM = 1.5


class RescoreStop(RuntimeError):
    """Error that stops one annulus of the re-score, which goes on with the next.
    """


class PlacementError(RescoreStop):
    """Error raised when a draw's sources have no placement under the rule asked.
    """


@dataclass
class Entrant:
    """One candidate configuration for an annulus, with the run and the evaluation it comes from.

    Attributes
    ----------
    run : str
        Name of the run.
    kind : {"winner", "validation", "top", "history"}
        How it got there: the run's winner, an entrant of its validation, one of the best of its history
        or an evaluation of its history picked by index.
    eval_index : int or None
        Its evaluation, counted from 0.
    config : dict
        Its configuration as the run stored it (``Config.to_dict``).
    search_score : float or None, default None
        Its score in the search.
    validated_score : float or None, default None
        Its score in the validation.
    """
    run: str
    kind: str
    eval_index: Optional[int]
    config: Dict[str, Any]
    search_score: Optional[float] = None
    validated_score: Optional[float] = None

    @property
    def key(self) -> str:
        """The configuration's identity, from :func:`config_key`.
        """
        return config_key(self.config)

    @property
    def tag(self) -> str:
        """Short, stable name for its state file, from its run and its configuration.
        """
        return hashlib.sha1((self.run + "|" + self.key).encode()).hexdigest()[:12]


def config_key(d: Dict[str, Any]) -> str:
    """Return a configuration's identity, its parameters, per-partition blocks and partitions as JSON.

    Parameters
    ----------
    d : dict
        The configuration, as ``Config.to_dict`` stores it.

    Returns
    -------
    key : str
        The JSON of ``params``, ``per_partition`` and the sorted ``selected``, its keys sorted.
    """
    return json.dumps({"params": d.get("params", {}), "per_partition": d.get("per_partition", {}),
                       "selected": sorted(str(s) for s in d.get("selected", []))}, sort_keys=True, default=float)


def parse_annuli(spec: str, nann: int) -> List[int]:
    """Parse a list of annuli, such as ``"1-5"``, ``"2,4,6"`` or ``"all"`` (1-based), into 0-based indices.

    Parameters
    ----------
    spec : str
        The list, ranges and single annuli separated by commas.  Empty or "all" takes every annulus.
    nann : int
        Number of the run's annuli.

    Returns
    -------
    annuli : list of int
        The 0-based indices, sorted, each once.

    Raises
    ------
    ValueError
        If an annulus is not among the run's.
    """
    s = str(spec).strip().lower()
    if s in ("", "all"):
        return list(range(nann))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            out += list(range(a, b + 1))
        elif part:
            out.append(int(part))
    bad = [v for v in out if not 1 <= v <= nann]
    if bad:
        raise ValueError(f"annuli {bad} are not among the run's 1-{nann}")
    return sorted({v - 1 for v in out})


def _read_json(path: str) -> Any:
    """Read a JSON file.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    data : object
        The file's content.
    """
    with open(path) as f:
        return json.load(f)


def run_entrants(run_dir: str, annulus: int, top: int = 0) -> List[Entrant]:
    """Return the candidates a finished run offers for an annulus.

    They are the configurations its validation scored, the elected one marked ``"winner"``, and with
    ``top`` the ``top`` best of its history that are not among them.  They are read from
    ``final_results.json`` and ``results.jsonl``, and a configuration offered twice comes once.

    Parameters
    ----------
    run_dir : str
        Directory of the run.
    annulus : int
        The annulus, counted from 0.
    top : int, default 0
        Number of the best evaluations of the history added.

    Returns
    -------
    entrants : list of Entrant
        The candidates, the validation's first.

    Raises
    ------
    ValueError
        If ``final_results.json`` has no such annulus.
    """
    name = os.path.basename(os.path.abspath(run_dir).rstrip(os.sep))
    fr = _read_json(os.path.join(run_dir, "final_results.json"))
    ann = next((a for a in fr.get("annuli") or [] if int(a.get("annulus", -1)) == int(annulus)), None)
    if ann is None:
        raise ValueError(f"{name}: no annulus {annulus + 1} in final_results.json")
    out: List[Entrant] = []
    seen: Dict[str, Entrant] = {}

    def add(e: Entrant) -> None:
        if e.key not in seen:
            seen[e.key] = e
            out.append(e)

    for row in ann.get("validation_table") or []:
        if row.get("config"):
            add(Entrant(name, "validation", row.get("eval_index"), row["config"], row.get("search_score"),
                        row.get("validated_score")))
    win = ann.get("winner_config")
    if win:
        e = seen.get(config_key(win))
        if e is not None:
            e.kind = "winner"
        else:
            add(Entrant(name, "winner", ann.get("winner_index"), win, None, ann.get("winner_score")))
    if top > 0:
        rows = []
        path = os.path.join(run_dir, "results.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for ln in f:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if int(r.get("annulus", -1)) == int(annulus) and r.get("score") is not None and r.get("config"):
                        rows.append(r)
        rows.sort(key=lambda r: -float(r["score"]))
        n = 0
        for r in rows:
            if n >= top:
                break
            if config_key(r["config"]) not in seen:
                add(Entrant(name, "top", r.get("index"), r["config"], float(r["score"]), None))
                n += 1
    return out


def entrant_at(run_dir: str, annulus: int, eval_index: int) -> Entrant:
    """Return an evaluation of a finished run's annulus as a candidate.

    It is the validation's entrant or the winner when it is one, else its configuration in
    ``results.jsonl``, of kind ``"history"``.

    Parameters
    ----------
    run_dir : str
        Directory of the run.
    annulus : int
        The annulus, counted from 0.
    eval_index : int
        The evaluation, counted from 0 as ``rescore.json`` and ``final_results.json`` count.

    Returns
    -------
    entrant : Entrant
        The candidate.

    Raises
    ------
    ValueError
        If the annulus has no such evaluation.
    """
    name = os.path.basename(os.path.abspath(run_dir).rstrip(os.sep))
    for e in run_entrants(run_dir, annulus):
        if e.eval_index is not None and int(e.eval_index) == int(eval_index):
            return e
    path = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if (int(r.get("annulus", -1)) == int(annulus) and r.get("index") is not None
                        and int(r["index"]) == int(eval_index) and r.get("config")):
                    sc = r.get("score")
                    return Entrant(name, "history", int(eval_index), r["config"], None if sc is None else float(sc))
    raise ValueError(f"{name}: annulus {annulus + 1} has no evaluation {eval_index} (counted from 0, as "
                     f"rescore.json counts)")


@dataclass
class Pick:
    """Line of a ``--base-winners`` or ``--candidates`` file, an annulus with a run's evaluation.

    Attributes
    ----------
    annulus : int
        The annulus, counted from 0 here and from 1 in the file.
    run : str
        Directory of the run.
    eval_index : int
        The evaluation, counted from 0 as ``rescore.json`` counts.
    """
    annulus: int
    run: str
    eval_index: int


def _resolve_run(token: str, known: Sequence[str], base_dir: str) -> str:
    """Return the directory of a run named in a picks file.

    Parameters
    ----------
    token : str
        The run, as a directory, the name of a run in ``known`` or the name of a run beside the base
        run.
    known : sequence of str
        Directories of the runs known, the base run's first.
    base_dir : str
        Directory of the base run.

    Returns
    -------
    run_dir : str
        The run's absolute directory.

    Raises
    ------
    ValueError
        If the token names none of these.
    """
    t = str(token).strip().rstrip("/" + os.sep) or str(token)
    if os.path.isdir(t):
        return os.path.abspath(t)
    for rd in known:
        if os.path.basename(os.path.abspath(rd).rstrip(os.sep)) == t:
            return os.path.abspath(rd)
    sib = os.path.join(os.path.dirname(os.path.abspath(base_dir).rstrip(os.sep)), t)
    if os.path.isdir(sib):
        return os.path.abspath(sib)
    raise ValueError(f"run {token!r}: not a directory, the base run, a run of --entrants-from or a run beside the "
                     f"base run")


def read_picks(path: str, base_dir: str, known: Sequence[str] = (), nann: Optional[int] = None,
               one_per_annulus: bool = False) -> List[Pick]:
    """Read a ``--base-winners`` or ``--candidates`` file.

    Each line gives the annulus (1-based), the run and the evaluation (0-based, as ``rescore.json``
    counts), with ``#`` starting a comment.  A JSON list of ``{"annulus", "run", "eval"}`` is read too.
    The run is a directory, the name of the base run or of a run in ``known``, or the name of a run
    beside the base run (:func:`_resolve_run`).

    Parameters
    ----------
    path : str
        Path of the file.
    base_dir : str
        Directory of the base run.
    known : sequence of str, default ()
        Directories of other runs whose names the file may use, such as those of ``--entrants-from``.
    nann : int or None, default None
        Number of the run's annuli, to check the annuli against.  None checks only that each is at
        least 1.
    one_per_annulus : bool, default False
        Whether an annulus listed twice is an error.

    Returns
    -------
    picks : list of Pick
        The picks, in the file's order.

    Raises
    ------
    ValueError
        If a line or entry is malformed, an annulus is out of range, an evaluation is negative, an
        annulus repeats under ``one_per_annulus``, a run is not found or the file lists nothing.
    """
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    rows: List[Tuple[str, Any, Any, Any]] = []
    if isinstance(data, list):
        for i, d in enumerate(data):
            if not isinstance(d, dict) or not {"annulus", "run", "eval"} <= set(d):
                raise ValueError(f"{path}: entry {i + 1} is not {{\"annulus\", \"run\", \"eval\"}}")
            rows.append((f"entry {i + 1}", d["annulus"], d["run"], d["eval"]))
    else:
        for n, ln in enumerate(text.splitlines(), 1):
            parts = shlex.split(ln, comments=True)
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}, line {n}: give the annulus, the run and the evaluation, not {ln.strip()!r}")
            rows.append((f"line {n}", *parts))
    known = [base_dir] + list(known)
    out: List[Pick] = []
    seen: Dict[int, str] = {}
    for where, ann, run, ev in rows:
        try:
            a, e = int(ann), int(ev)
        except (TypeError, ValueError):
            raise ValueError(f"{path}, {where}: the annulus and the evaluation are whole numbers")
        if a < 1 or (nann is not None and a > int(nann)):
            raise ValueError(f"{path}, {where}: annulus {a} is not among the run's 1-{nann}")
        if e < 0:
            raise ValueError(f"{path}, {where}: evaluation {e}, where they count from 0 as rescore.json does")
        if one_per_annulus and a in seen:
            raise ValueError(f"{path}, {where}: annulus {a} again (first on {seen[a]})")
        seen.setdefault(a, where)
        out.append(Pick(a - 1, _resolve_run(str(run), known, base_dir), e))
    if not out:
        raise ValueError(f"{path} lists nothing")
    return out


def _setup(run_dir: str) -> Dict[str, Any]:
    """Return what must agree between the runs, their annuli, the search space's dimensions and the frames.

    Parameters
    ----------
    run_dir : str
        Directory of the run.

    Returns
    -------
    setup : dict
        ``zones`` (each annulus' ``(inrad, outrad)``, rounded), ``names`` (the dimensions' names) and
        ``frames`` (the data edits that say what the frames are, as JSON).

    Raises
    ------
    ValueError
        If the run has no readable checkpoint.
    """
    from .runner import _core_edits, load_checkpoint
    st, _ = load_checkpoint(run_dir, log=lambda s: None)
    if st is None:
        raise ValueError(f"{run_dir}: no readable checkpoint")
    cfg = st.get("config") or {}
    fr = _read_json(os.path.join(run_dir, "final_results.json"))
    zones = [(round(float(a["inrad"]), 6), round(float(a["outrad"]), 6)) for a in fr.get("annuli") or []]
    return {"zones": zones, "names": [p.get("name") for p in (st.get("space") or {}).get("params", [])],
            "frames": json.dumps(_core_edits(cfg.get("data_edits")), sort_keys=True, default=str)}


def _check_setup(rd: str, ref: Dict[str, Any]) -> None:
    """Check that a run has the base run's annuli, search space and frames.

    Parameters
    ----------
    rd : str
        Directory of the run.
    ref : dict
        The base run's setup, from :func:`_setup`.

    Raises
    ------
    ValueError
        If any of them differs.
    """
    s = _setup(rd)
    name = os.path.basename(rd)
    if s["zones"] != ref["zones"]:
        raise ValueError(f"{name}'s annuli {s['zones']} are not the base run's {ref['zones']}")
    if s["names"] != ref["names"]:
        raise ValueError(f"{name}'s search space is not the base run's")
    if s["frames"] != ref["frames"]:
        raise ValueError(f"{name} was made on other frames than the base run ({s['frames']} against "
                         f"{ref['frames']})")


def resolve_picks(base_dir: str, picks: Sequence[Pick], ref: Optional[Dict[str, Any]] = None
                  ) -> List[Tuple[Pick, Entrant]]:
    """Return each pick with its configuration, its run checked against the base run's setup.

    Parameters
    ----------
    base_dir : str
        Directory of the base run.
    picks : sequence of Pick
        The picks.
    ref : dict or None, default None
        The base run's setup (:func:`_setup`).  None reads it.

    Returns
    -------
    resolved : list of tuple
        ``(pick, entrant)`` per pick, the entrant from :func:`entrant_at`.

    Raises
    ------
    ValueError
        If a run's annuli, search space or frames are not the base run's, or a pick's evaluation is not
        found.
    """
    base_dir = os.path.abspath(base_dir)
    ref = _setup(base_dir) if ref is None else ref
    checked = {base_dir}
    out = []
    for p in picks:
        rd = os.path.abspath(p.run)
        if rd not in checked:
            _check_setup(rd, ref)
            checked.add(rd)
        out.append((p, entrant_at(rd, p.annulus, p.eval_index)))
    return out


def _nanmedian(v) -> float:
    """Return the median of the finite values, NaN without any.

    Parameters
    ----------
    v : array_like
        The values.

    Returns
    -------
    median : float
        The median.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _pct(v, q) -> float:
    """Return a percentile of the finite values, NaN without any.

    Parameters
    ----------
    v : array_like
        The values.
    q : float
        The percentile, from 0 to 100.

    Returns
    -------
    value : float
        The percentile.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def paired_summary(per_draw: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    """Summarize an entrant's statistic over the common draws and its paired difference from the base winner's.

    Parameters
    ----------
    per_draw : ndarray, shape (n,)
        The entrant's statistic on each of the ``n`` draws.
    base : ndarray, shape (n,)
        The base winner's statistic on the same draws.

    Returns
    -------
    summary : dict
        ``median``, ``p16`` and ``p84`` of the entrant's finite values, ``n`` (their number),
        ``n_paired`` (the draws finite for both), ``diff_mean`` with its standard error ``diff_se``,
        ``diff_median`` and ``p_better``, the fraction of paired draws where the entrant reads higher.
        The differences are NaN without paired draws, and ``diff_se`` with one.
    """
    a = np.asarray(per_draw, float)
    b = np.asarray(base, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    out = {"median": _nanmedian(a), "p16": _pct(a, 16), "p84": _pct(a, 84), "n": int(np.isfinite(a).sum()),
           "n_paired": int(d.size)}
    if d.size:
        out.update(diff_mean=float(np.mean(d)), diff_median=float(np.median(d)),
                   diff_se=float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan"),
                   p_better=float(np.mean(d > 0)))
    else:
        out.update(diff_mean=float("nan"), diff_median=float("nan"), diff_se=float("nan"), p_better=float("nan"))
    return out


class StitchScorer:
    """The base run's stitch with one annulus' images replaced, its sources measured as the stitched verify does.

    Parameters
    ----------
    runner : Runner
        The base run, resumed read-only (``Runner.resume(..., same_frames=False)``).
    overlap_pads : bool or None, default None
        Whether overlapping annuli take pads between them (``RunConfig.stitch_overlap_pads``), as
        ``rereduce`` sets it.  None keeps the run's.
    log : callable, default print
        Function that takes each line of the log.
    card : bool, default False
        Whether the injected reductions may go to the card workers, as the search's draws do.

    Attributes
    ----------
    results : dict of int to AnnulusResult
        The base run's result for each annulus.
    order : list of int
        The annuli, sorted.
    zones : list of tuple of 2 floats
        Each annulus' zone, ``(inrad, outrad)`` in pixels.
    tiles : list of tuple of 2 floats
        Each annulus' tile, in pixels.
    pads : list of tuple of 2 floats
        Each tile's ``(pad_in, pad_out)``, in pixels.
    itiles : list of tuple of 2 floats
        The tiles the stitched images are made over, the innermost and the outermost reaching the
        whole KLIP zones of their annuli.
    base_cfg : dict of int to Config
        Each annulus' base winner.
    base_from : dict of int to Entrant
        The annuli whose base winner is another configuration than the run's (:meth:`use_winners`).
    known : list of tuple
        The known companions of the run's sampler.
    clean : dict of int to ndarray
        Each annulus' clean image under its base winner (:meth:`prepare`).
    weights : ndarray or None
        The stitch weights (:meth:`prepare`).
    weights_from : str
        Where the weights came from.
    n_reductions : int
        Number of reductions made.

    Raises
    ------
    ValueError
        If the run has not finished, searched the widths of its annuli (``opt_width``) or scanned k.
    """

    def __init__(self, runner, overlap_pads: Optional[bool] = None, log: Callable[[str], None] = print,
                 card: bool = False):
        from .runner import SCAN_MODES
        self.r = runner
        self.card = bool(card)
        self.log = log
        res = [x for x in (runner.results or []) if x is not None]
        if len(res) < runner.cfg.nann:
            raise ValueError(f"{os.path.basename(runner.run_dir)} has not finished: {len(res)} of "
                             f"{runner.cfg.nann} annuli have a winner")
        if runner.cfg.opt_width:
            raise ValueError("the re-score takes runs of fixed annuli, not opt_width runs")
        if runner.cfg.k_mode in SCAN_MODES:
            raise ValueError("the re-score takes runs whose k is searched (k_mode search), not a k scan")
        self.results = {x.annulus: x for x in res}
        self.order = sorted(self.results)
        self.zones = [(float(self.results[a].inrad), float(self.results[a].outrad)) for a in self.order]
        pads_run = bool(getattr(runner.cfg, "stitch_overlap_pads", True))
        self.overlap_pads = pads_run if overlap_pads is None else bool(overlap_pads)
        keep = runner.cfg.stitch_overlap_pads
        try:
            runner.cfg.stitch_overlap_pads = self.overlap_pads
            self.tiles = [tuple(float(v) for v in t) for t in runner._tiles(self.zones)]
            self.pads = runner._tile_pads(self.tiles, annuli=self.zones)
        finally:
            runner.cfg.stitch_overlap_pads = keep
        self.itiles = runner._image_tiles(self.tiles)
        self.base_cfg = {a: runner._config_from_dict(self.results[a].winner_config) for a in self.order}
        #: the annuli whose base winner is another configuration than the run's (:meth:`use_winners`)
        self.base_from: Dict[int, Entrant] = {}
        self.known = [tuple(k) for k in (getattr(runner.sampler, "known", ()) or ())]
        self.clean: Dict[int, np.ndarray] = {}
        self.weights: Optional[np.ndarray] = None
        self.weights_from = ""
        self.n_reductions = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- geometry
    def pos(self, ia: int) -> int:
        """Return an annulus' position in the stitch's order.

        Parameters
        ----------
        ia : int
            The annulus, counted from 0.

        Returns
        -------
        pos : int
            Its index in ``order``.
        """
        return self.order.index(ia)

    def tile(self, ia: int) -> Tuple[float, float]:
        """Return an annulus' tile.

        Parameters
        ----------
        ia : int
            The annulus, counted from 0.

        Returns
        -------
        tile : tuple of 2 floats
            The tile's inner and outer radii, in pixels.
        """
        return self.tiles[self.pos(ia)]

    def _zone_pad(self) -> float:
        """Return the reducer's pad around a KLIP zone, the largest of a partitioned reducer's.

        Returns
        -------
        pad : float
            The pad, in pixels, 2 where the reducer gives none.
        """
        red = self.r.reducer
        zp = getattr(red, "zone_pad", None)
        if zp is None and getattr(red, "reducers", None):
            zp = max((float(getattr(s, "zone_pad", 0.0) or 0.0) for s in red.reducers.values()), default=None)
        return 2.0 if zp is None else float(zp)

    def reach(self, ia: int) -> Tuple[float, float]:
        """Return the radii an annulus' KLIP zone can cover under its base winner.

        They are its zone widened by the reducer's pad and the winner's largest ``zone_in`` and
        ``zone_out``, with 1 px to spare.

        Parameters
        ----------
        ia : int
            The annulus, counted from 0.

        Returns
        -------
        r_in : float
            Inner radius, in pixels, at least 0.
        r_out : float
            Outer radius, in pixels.
        """
        cfg = self.base_cfg[ia]
        blocks = [cfg.params] + list(cfg.per_partition.values())
        zi = max([float(b.get("zone_in", 0.0) or 0.0) for b in blocks] + [0.0])
        zo = max([float(b.get("zone_out", 0.0) or 0.0) for b in blocks] + [0.0])
        a_in, a_out = self.zones[self.pos(ia)]
        m = self._zone_pad() + 1.0
        return max(a_in - zi - m, 0.0), a_out + zo + m

    def affected(self, sources: Sequence[Source], k: int, force: bool = True) -> List[int]:
        """Return the annuli whose zones a draw's sources reach, within 2.5 FWHM of each.

        Parameters
        ----------
        sources : sequence of Source
            The draw's sources.
        k : int
            The annulus scored, counted from 0.
        force : bool, default True
            Whether ``k`` is among them whatever the sources reach.

        Returns
        -------
        annuli : list of int
            The annuli, in the stitch's order.
        """
        R = 2.5 * float(self.r.fwhm)
        rs = [float(s.rho) / float(self.r.pxscale) for s in sources]
        out = []
        for ia in self.order:
            lo, hi = self.reach(ia)
            if (force and ia == k) or any(lo - R <= r <= hi + R for r in rs):
                out.append(ia)
        return out

    def use_winners(self, winners: Dict[int, Entrant]) -> None:
        """Make a composite base, with other configurations as the base winners of some annuli.

        Each such winner serves for the clean stitch, the neighbors' reductions, the calibration and its
        zone's reach, and its clean image is reduced again.  Call it before :meth:`prepare`.

        Parameters
        ----------
        winners : dict of int to Entrant
            The base winner of each annulus replaced, by 0-based annulus.

        Raises
        ------
        ValueError
            If an annulus is not among the base run's.
        """
        for a, e in winners.items():
            if a not in self.base_cfg:
                raise ValueError(f"annulus {a + 1} is not among the base run's")
            self.base_cfg[a] = self.r._config_from_dict(e.config)
            self.base_from[a] = e
            self.clean.pop(a, None)

    # ---------------------------------------------------------------- reductions
    def reduce(self, ia: int, cfg, sources: Optional[Sequence[Source]], tag: str) -> np.ndarray:
        """Reduce a configuration in an annulus' zone, with injected sources and no KLIP-FM pass.

        Each call counts in ``n_reductions``.

        Parameters
        ----------
        ia : int
            The annulus, counted from 0.
        cfg : Config
            The configuration.
        sources : sequence of Source or None
            The sources injected, None for the clean image.
        tag : str
            Label of the reduction, for logs.

        Returns
        -------
        image : ndarray, shape (ny, nx)
            The reduced image.

        Raises
        ------
        ValueError
            If the reduction returns no image.
        """
        more = {"card": True} if (self.card and sources) else {}
        ev = self.r._reduce(cfg, list(sources) if sources else None, tag=tag, zone=self.r._zone(ia, cfg),
                            fm_sources=[], **more)
        with self._lock:
            self.n_reductions += 1
        if ev.image is None:
            raise ValueError(f"annulus {ia + 1}: the reduction returned no image")
        return np.asarray(ev.image, float)

    def run(self, calls: Sequence[Callable[[], Any]], window: int):
        """Run reductions, at most ``window`` in flight, and yield their results in order.

        Parameters
        ----------
        calls : sequence of callable
            The reductions, each a function of no arguments.
        window : int
            Number of reductions in flight at most (:meth:`klip_tpe.runner.Runner._run_ordered`).

        Yields
        ------
        i : int
            Index of the call.
        result : object
            Its result.

        Raises
        ------
        Exception
            The exception of a failed reduction.
        """
        for i, res in self.r._run_ordered(calls, window):
            if isinstance(res, Exception):
                raise res
            yield i, res

    def prepare(self, window: int) -> None:
        """Reduce the base winners' clean images again on this code, and set the stitch weights.

        The weights are the base run's (``final_results.json``), else worked out from those images as the
        run worked them out.  Only the annuli without a clean image are reduced.  It fills ``clean``,
        ``weights`` and ``weights_from``.

        Parameters
        ----------
        window : int
            Number of reductions in flight at most.
        """
        todo = [a for a in self.order if a not in self.clean]
        calls = [(lambda a=a: self.reduce(a, self.base_cfg[a], None, f"rescore_a{a + 1}_clean")) for a in todo]
        for i, img in self.run(calls, window):
            self.clean[todo[i]] = img
        if self.weights is None:
            try:
                wl = _read_json(os.path.join(self.r.run_dir, "final_results.json")).get("stitch_weights")
            except (OSError, ValueError):
                wl = None
            if wl is not None and len(wl) == len(self.order):
                self.weights, self.weights_from = np.asarray(wl, float), "final_results.json"
            else:
                self.weights = np.asarray(self.r._stitch_weights([self.clean[a] for a in self.order], self.zones),
                                          float)
                self.weights_from = "the base winners' images"

    # ---------------------------------------------------------------- the stitch and the measure
    def stitch(self, images: Dict[int, np.ndarray]) -> np.ndarray:
        """Build the injected stitch of some annuli's images, with the base winners' clean images for the rest.

        It is built as the final stitch builds it, with the base run's tiles, pads, weights and flattening.

        Parameters
        ----------
        images : dict of int to ndarray
            The images, by 0-based annulus.

        Returns
        -------
        stitch : ndarray, shape (ny, nx)
            The stitched image.
        """
        from .stitch import stitch_tiles
        imgs = [images.get(a, self.clean[a]) for a in self.order]
        return self.r._maybe_flatten(stitch_tiles(imgs, self.itiles, self.weights, pads=self.pads))

    def measure(self, img: np.ndarray, sources: Sequence[Source]) -> np.ndarray:
        """Measure each source's S/N on the stitch as the stitched verify's Path A does.

        Each ring of :func:`klip_tpe.verify.verify_injections` leaves out the other sources and the known
        companions.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            The stitched image.
        sources : sequence of Source
            The sources.

        Returns
        -------
        snr : ndarray, shape (n,)
            S/N of each of the ``n`` sources.
        """
        from .verify import verify_injections
        rows, _ = verify_injections(np.asarray(img, float)[None], [s.as_tuple() for s in sources], self.r.fwhm,
                                    self.r.pxscale, known=self.known, excl_fwhm=self.r._excl_fwhm(),
                                    angle_convention=self.r.reducer.angle_convention, combined=img, clean=None,
                                    sidelobes=self.r._verify_rule(), rules=self.r._ring_rules())
        return np.array([np.nan if r.get("snr") is None else float(r["snr"]) for r in rows], float)

    # ---------------------------------------------------------------- draws
    @property
    def clear_fwhm(self) -> float:
        """The run's sampler's distance between sources, in FWHM, the draws' clearance by default.
        """
        return float(getattr(self.r.sampler, "min_mutual_fwhm", 2.0))

    @property
    def fwhm_px(self) -> float:
        """The FWHM the run's sampler places sources by, in pixels.
        """
        return float(getattr(self.r.sampler, "fwhm_as", self.r.fwhm * self.r.pxscale)) / float(self.r.pxscale)

    def draw(self, tiles: Sequence[int], contrasts: Dict[int, float], rng: np.random.Generator,
             counts: Optional[Dict[int, int]] = None, clear_fwhm: Optional[float] = None, restarts: int = 20
             ) -> List[Tuple[int, Source]]:
        """Draw sources in some tiles, each at its tile's contrast and clear of the other tiles' sources.

        The run's sampler places each tile's sources in its radii, at least ``clear_fwhm`` FWHM from the
        sources of the other tiles and, where the sampler checks them, outside each other's ADI sidelobes.
        The tile with the most sources is placed first, since a pair 180 degrees apart, placed after the
        other tiles' sources, has to clear all of them at once.  A tile that finds no place in 300 tries
        starts the draw again.

        Parameters
        ----------
        tiles : sequence of int
            The tiles, by 0-based annulus.
        contrasts : dict of int to float
            Contrast of each tile's sources.
        rng : numpy.random.Generator
            Generator of the positions.
        counts : dict of int to int or None, default None
            Number of sources in each tile.  None, or a tile left out, takes the run's number.
        clear_fwhm : float or None, default None
            Smallest distance between two tiles' sources, in FWHM.  None takes the sampler's
            (:attr:`clear_fwhm`).
        restarts : int, default 20
            Number of starts before giving up.

        Returns
        -------
        draw : list of tuple
            ``(tile, source)`` per source, in the order of ``tiles``.

        Raises
        ------
        PlacementError
            If no start places every tile.
        """
        smp = self.r.sampler
        px = float(self.r.pxscale)
        fwhm_as = float(getattr(smp, "fwhm_as", self.r.fwhm * px))
        cf = self.clear_fwhm if clear_fwhm is None else float(clear_fwhm)
        dmin = cf * fwhm_as
        n_of = {j: int(self.r._nsrc(j)) if counts is None or j not in counts else int(counts[j]) for j in tiles}
        order = sorted(tiles, key=lambda j: -n_of[j])
        rank = {j: i for i, j in enumerate(tiles)}
        hard = None
        for _ in range(max(int(restarts), 1)):
            out: List[Tuple[int, Source]] = []
            for j in order:
                if n_of[j] <= 0:
                    continue
                t0, t1 = self.tile(j)
                for _t in range(300):
                    src = smp.sample(n_of[j], t0 * px, t1 * px, rng, float(contrasts[j]))
                    ok = all(_dist(s, o) >= dmin for s in src for _, o in out)
                    if ok and out and hasattr(smp, "_in_sidelobes"):
                        allr = [o.rho for _, o in out] + [s.rho for s in src]
                        allt = [o.theta for _, o in out] + [s.theta for s in src]
                        ok = not smp._in_sidelobes(allr, allt)
                    if ok:
                        break
                else:
                    hard = j
                    break
                out += [(j, s) for s in src]
            else:
                return sorted(out, key=lambda js: rank[js[0]])
        raise PlacementError(f"no placement in tile {hard + 1} clear of the other tiles' sources by {cf:g} FWHM "
                             f"({max(int(restarts), 1)} starts)")

    def score_draws(self, draws: Sequence[Sequence[Tuple[int, Source]]], k: int, window: int,
                    cfg_k=None, apart: bool = False) -> List[np.ndarray]:
        """Measure each draw's sources on the stitch, with every annulus they reach reduced with them.

        Annulus ``k`` is reduced under ``cfg_k`` and the others under their base winners.

        Parameters
        ----------
        draws : sequence of sequence of tuple
            The draws, ``(tile, source)`` per source (:meth:`draw`).
        k : int
            The annulus scored, counted from 0.
        window : int
            Number of reductions in flight at most.
        cfg_k : Config or None, default None
            Configuration of annulus ``k``.  None takes its base winner.
        apart : bool, default False
            Whether each tile's sources go into reductions and a stitch of their own.  Annulus ``k`` is then
            reduced only where a tile's sources reach it.

        Returns
        -------
        snr : list of ndarray
            Each draw's S/N, one per source in the draw's order.
        """
        cfg_k = self.base_cfg[k] if cfg_k is None else cfg_k
        subs = [_subdraws(d, apart) for d in draws]
        jobs, slots = [], []
        for t, sub in enumerate(subs):
            for g, (_, src, j) in enumerate(sub):
                for a in self.affected(src, k, force=not apart):
                    cfg = cfg_k if a == k else self.base_cfg[a]
                    tag = _tag(a, t, j)
                    jobs.append(lambda a=a, cfg=cfg, src=src, tag=tag: self.reduce(a, cfg, src, tag))
                    slots.append((t, g, a))
        imgs: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
        for i, img in self.run(jobs, window):
            t, g, a = slots[i]
            imgs.setdefault((t, g), {})[a] = img
        out = []
        for t, d in enumerate(draws):
            v = np.full(len(d), np.nan)
            for g, (idx, src, _) in enumerate(subs[t]):
                v[idx] = self.measure(self.stitch(imgs.get((t, g), {})), src)
            out.append(v)
        return out

    def calibrate(self, j: int, rng: np.random.Generator, n_cal: int = 8, target: Tuple[float, float] = TARGET,
                  max_iter: int = 4, window: int = 2, n_src: Optional[int] = None) -> Dict[str, Any]:
        """Find the tile contrast that puts the base winners' median S/N on the stitch inside ``target``.

        The S/N is measured on ``n_cal`` draws of tile ``j`` alone, at the same positions at every step.
        The first step takes the base run's contrast for annulus ``j`` (1e-4 without one), and each step
        after takes the contrast :meth:`_next_contrast` gives.

        Parameters
        ----------
        j : int
            The tile, by 0-based annulus.
        rng : numpy.random.Generator
            Generator of the positions.
        n_cal : int, default 8
            Number of draws.
        target : tuple of 2 floats, default TARGET
            Range of the median S/N sought.
        max_iter : int, default 4
            Number of steps at most.
        window : int, default 2
            Number of reductions in flight at most.
        n_src : int or None, default None
            Number of sources per draw.  None takes the run's.

        Returns
        -------
        calibration : dict
            ``contrast``, ``converged`` (whether a step read inside ``target``) and ``history``, each step's
            ``contrast``, ``median_snr`` and ``n``.  Without convergence the contrast is the next one, not
            measured.
        """
        c = float(self.results[j].contrast or 0.0)
        if not np.isfinite(c) or c <= 0:
            c = 1e-4
        hist = []
        places = [self.draw([j], {j: c}, rng, counts=None if n_src is None else {j: int(n_src)})
                  for _ in range(n_cal)]
        for it in range(max_iter):
            draws = [[(jj, Source(s.rho, s.theta, c)) for jj, s in d] for d in places]
            snr = self.score_draws(draws, j, window)
            m = _nanmedian(np.concatenate(snr) if snr else [])
            hist.append({"contrast": c, "median_snr": m, "n": int(sum(np.isfinite(v).sum() for v in snr))})
            self.log(f"  tile {j + 1}: contrast {c:.3e} -> median S/N {m:.2f} on the stitch "
                     f"({n_cal} draws, step {it + 1})")
            if target[0] <= m <= target[1]:
                return {"contrast": c, "converged": True, "history": hist}
            c = self._next_contrast(hist, target)
        self.log(f"  tile {j + 1}: not inside S/N {target[0]:g}-{target[1]:g} after {max_iter} steps, "
                 f"taking {c:.3e}")
        return {"contrast": c, "converged": False, "history": hist}


    @staticmethod
    def _next_contrast(hist: Sequence[Dict[str, Any]], target: Tuple[float, float]) -> float:
        """Return the next contrast to try in a calibration.

        Once a step has read below ``target`` and one above, it is the contrast between the nearest two,
        interpolated in log contrast against their S/N.  Before, it is the last contrast scaled by the
        target's middle over its S/N, or by 3 where that S/N is not finite or at most 0.5.

        Parameters
        ----------
        hist : sequence of dict
            The steps so far, each with ``contrast`` and ``median_snr``.
        target : tuple of 2 floats
            Range of the median S/N sought.

        Returns
        -------
        contrast : float
            The next contrast.
        """
        lo, hi = float(target[0]), float(target[1])
        mid = 0.5 * (lo + hi)
        pts = [(float(h["contrast"]), float(h["median_snr"])) for h in hist if np.isfinite(h["median_snr"])]
        below = [p for p in pts if p[1] < lo]
        above = [p for p in pts if p[1] > hi]
        if below and above:
            (cl, ml), (ch, mh) = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
            f = (mid - ml) / (mh - ml)
            return float(np.exp(np.log(cl) + f * (np.log(ch) - np.log(cl))))
        c, m = float(hist[-1]["contrast"]), float(hist[-1]["median_snr"])
        return c * mid / m if (np.isfinite(m) and m > 0.5) else c * 3.0


def _dist(a: Source, b: Source) -> float:
    """Return the distance between two sources.

    Parameters
    ----------
    a : Source
        The first source.
    b : Source
        The second source.

    Returns
    -------
    dist : float
        The distance, in arcsec.
    """
    pa, pb = np.deg2rad(a.theta + 90.0), np.deg2rad(b.theta + 90.0)
    return float(np.hypot(a.rho * np.cos(pa) - b.rho * np.cos(pb), a.rho * np.sin(pa) - b.rho * np.sin(pb)))


def _subdraws(d: Sequence[Tuple[int, Source]], apart: bool) -> List[Tuple[List[int], List[Source], Optional[int]]]:
    """Split a draw into the parts reduced and measured together.

    Parameters
    ----------
    d : sequence of tuple
        The draw, ``(tile, source)`` per source.
    apart : bool
        Whether each tile's sources form a part of their own.

    Returns
    -------
    parts : list of tuple
        ``(places, sources, tile)`` per part, ``places`` being the sources' indices in the draw.  Without
        ``apart`` the whole draw is one part, of tile None.
    """
    if not apart:
        return [(list(range(len(d))), [s for _, s in d], None)]
    parts: Dict[int, Tuple[List[int], List[Source], Optional[int]]] = {}
    for i, (j, s) in enumerate(d):
        p = parts.setdefault(int(j), ([], [], int(j)))
        p[0].append(i)
        p[1].append(s)
    return list(parts.values())


def _tag(a: int, t: int, j: Optional[int], cand: bool = False) -> str:
    """Return a reduction's tag, from the annulus, the draw and, apart, the tile.

    Parameters
    ----------
    a : int
        The annulus, counted from 0.
    t : int
        Index of the draw.
    j : int or None
        The tile, by 0-based annulus, None for a whole draw.
    cand : bool, default False
        Whether the reduction is a candidate's, marked ``_c``.

    Returns
    -------
    tag : str
        The tag, such as ``rescore_a3_c_d4_t2``.
    """
    return f"rescore_a{a + 1}{'_c' if cand else ''}_d{t}" + ("" if j is None else f"_t{j + 1}")


def _options_record(sd: str) -> Optional[Dict[str, Any]]:
    """Return what a re-score directory's ``state`` was started with, its ``options.json``.

    Parameters
    ----------
    sd : str
        The ``state`` directory.

    Returns
    -------
    options : dict or None
        The options, None for a directory not started.

    Notes
    -----
    A directory started before the record (2026-11-11) scored its tiles together on the base run's
    winners and its own draws, and gets those options.
    """
    p = os.path.join(sd, "options.json")
    if os.path.exists(p):
        return _read_json(p)
    if os.path.isdir(sd) and any(n.endswith(".json") for n in os.listdir(sd)):
        return {"tiles": "together", "base_winners": {}, "draws_from": None}
    return None


def _copy_draws(src_dir: str, sd: str, k: int, n: Optional[int]) -> int:
    """Copy an annulus' draws and their rule from an earlier re-score, unless the directory holds them already.

    It writes ``annKK_draws.json`` and, where the earlier re-score has it, ``annKK_placement.json`` into
    ``sd``.

    Parameters
    ----------
    src_dir : str
        Directory of the earlier re-score.
    sd : str
        This re-score's ``state`` directory.
    k : int
        The annulus, counted from 0.
    n : int or None
        Number of draws, the first ones.  None takes all.

    Returns
    -------
    n_draws : int
        Number of draws in ``sd``.

    Raises
    ------
    RescoreStop
        If the earlier re-score has no draws of the annulus, or fewer than ``n``.
    """
    dst = os.path.join(sd, f"ann{k + 1:02d}_draws.json")
    if os.path.exists(dst):
        return len(_read_json(dst))
    name = os.path.basename(src_dir)
    src = os.path.join(src_dir, "state", f"ann{k + 1:02d}_draws.json")
    if not os.path.exists(src):
        raise RescoreStop(f"{name} has no draws of annulus {k + 1}: re-score it there first, or leave it out")
    d = _read_json(src)
    n = len(d) if n is None else int(n)
    if len(d) < n:
        raise RescoreStop(f"{name} holds {len(d)} draws of annulus {k + 1}, not {n}")
    pp = os.path.join(src_dir, "state", f"ann{k + 1:02d}_placement.json")
    if os.path.exists(pp):
        _atomic_json(os.path.join(sd, f"ann{k + 1:02d}_placement.json"), _read_json(pp))
    _atomic_json(dst, d[:n])
    return n


def _per_draw(snr: Sequence[np.ndarray], tiles_of: Sequence[Sequence[int]]) -> Dict[Any, np.ndarray]:
    """Return each draw's median S/N, of all its sources and of each tile's.

    Parameters
    ----------
    snr : sequence of ndarray
        Each draw's S/N, per source.
    tiles_of : sequence of sequence of int
        Each draw's tile of each source.

    Returns
    -------
    medians : dict
        ``"all"`` and each tile (0-based) to an array of one median per draw.
    """
    out: Dict[Any, np.ndarray] = {"all": np.array([_nanmedian(v) for v in snr])}
    for j in sorted({int(j) for tj in tiles_of for j in tj}):
        out[j] = np.array([_nanmedian(np.asarray(v)[np.asarray(tj) == j]) for v, tj in zip(snr, tiles_of)])
    return out


def _entrant_brief(e: Dict[str, Any]) -> Dict[str, Any]:
    """Return an entrant's record cut to its run, kind and evaluation.

    Parameters
    ----------
    e : dict
        The entrant's record, as ``rescore.json`` stores it.

    Returns
    -------
    brief : dict
        ``run``, ``kind`` and ``eval_index``.
    """
    return {"run": e.get("run"), "kind": e.get("kind"), "eval_index": e.get("eval_index")}


def _against(src_dir: str, k: int, rep_k: Dict[str, Any], draws, tiles_of) -> Dict[str, Any]:
    """Compare this re-score's base winner of an annulus with an earlier re-score's, on the same draws.

    Parameters
    ----------
    src_dir : str
        Directory of the earlier re-score.
    k : int
        The annulus, counted from 0.
    rep_k : dict
        This re-score's report of the annulus, with its ``entrants``.
    draws : sequence of sequence of tuple
        The annulus' draws, ``(tile, source)`` per source.
    tiles_of : sequence of sequence of int
        Each draw's tile of each source.

    Returns
    -------
    comparison : dict
        ``dir`` and either ``missing``, why there is no comparison, or ``base`` (the earlier base
        winner), ``tiles`` and ``base_winners`` (the earlier re-score's options) and the paired
        differences of the per-draw medians, this base's from the earlier one's (:func:`paired_summary`):
        ``all`` for all of a draw's sources, ``tile`` for tile ``k`` and ``per_tile`` for each tile.
    """
    name = os.path.basename(src_dir)
    out: Dict[str, Any] = {"dir": src_dir}
    try:
        other = _read_json(os.path.join(src_dir, "rescore.json"))
    except (OSError, ValueError):
        return dict(out, missing=f"{name} has no rescore.json")
    a = (other.get("annuli") or {}).get(str(k + 1)) or {}
    tb = next((e for e in a.get("entrants") or [] if e.get("base")), None)
    if tb is None:
        return dict(out, missing=f"{name} has no scores of annulus {k + 1}")
    n = len(draws)
    mine = np.asarray([_draw_json(d) for d in draws], float)
    theirs = np.asarray((a.get("draws") or [])[:n], float)
    if theirs.shape != mine.shape or not np.allclose(theirs, mine, rtol=0, atol=1e-12):
        return dict(out, missing=f"{name}'s draws of annulus {k + 1} are not these")
    ob = next((e for e in rep_k["entrants"] if e["base"]), None)
    if ob is None:
        return dict(out, missing="this re-score has no base winner's scores")
    po = _per_draw([np.asarray(v, float) for v in ob["snr"]], tiles_of)
    pt = _per_draw([np.asarray(v, float) for v in tb["snr"][:n]], tiles_of)
    opts = _options_record(os.path.join(src_dir, "state")) or {}
    return dict(out, base=_entrant_brief(tb["entrant"]), tiles=opts.get("tiles", "together"),
                base_winners=opts.get("base_winners") or {}, all=paired_summary(po["all"], pt["all"]),
                tile=paired_summary(po[k], pt[k]) if k in po else None,
                per_tile={str(j + 1): paired_summary(po[j], pt[j]) for j in po if j != "all"})


def _draw_json(d: Sequence[Tuple[int, Source]]) -> List[List[float]]:
    """Return a draw as JSON lists, ``[tile, rho, theta, contrast]`` per source.

    Parameters
    ----------
    d : sequence of tuple
        The draw, ``(tile, source)`` per source.

    Returns
    -------
    rows : list of list
        One row per source.
    """
    return [[int(j), float(s.rho), float(s.theta), float(s.contrast)] for j, s in d]


def _draw_from_json(d) -> List[Tuple[int, Source]]:
    """Return a draw from its JSON lists (:func:`_draw_json`).

    Parameters
    ----------
    d : sequence of sequence
        ``[tile, rho, theta, contrast]`` per source.

    Returns
    -------
    draw : list of tuple
        ``(tile, source)`` per source.
    """
    return [(int(v[0]), Source(float(v[1]), float(v[2]), float(v[3]))) for v in d]


def _atomic_json(path: str, obj: Any) -> None:
    """Write an object as JSON, through a temporary file replaced into place.

    Parameters
    ----------
    path : str
        Path of the file.
    obj : object
        The object.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def _neighbors(scorer: StitchScorer, k: int) -> List[int]:
    """Return an annulus and its neighbors in the stitch, the tiles of its draws.

    Parameters
    ----------
    scorer : StitchScorer
        The scorer.
    k : int
        The annulus, counted from 0.

    Returns
    -------
    annuli : list of int
        Annulus ``k`` with the annuli before and after it in the stitch's order, where there are.
    """
    p = scorer.pos(k)
    return [scorer.order[q] for q in (p - 1, p, p + 1) if 0 <= q < len(scorer.order)]


def _placements(scorer: StitchScorer, k: int, sources_per_tile: Optional[int] = None,
                clear_fwhm: Optional[float] = None) -> List[Dict[str, Any]]:
    """Return the rules annulus ``k``'s draws are placed by, in the order they are tried.

    A rule asked for, by ``sources_per_tile`` or ``clear_fwhm``, is the one rule.  Otherwise the rules
    are the run's numbers at the sampler's distance, the same numbers at :data:`RELAXED_FWHM` where that
    is smaller, then one source in each neighbor's tile at the smaller distance, annulus ``k``'s own
    keeping the run's number, where that changes the numbers.

    Parameters
    ----------
    scorer : StitchScorer
        The scorer.
    k : int
        The annulus, counted from 0.
    sources_per_tile : int or None, default None
        Most sources in each tile.
    clear_fwhm : float or None, default None
        Smallest distance between two tiles' sources, in FWHM.

    Returns
    -------
    rules : list of dict
        ``counts`` (sources per tile), ``clear_fwhm`` and ``rule`` ("asked", "the run's" or "relaxed")
        per rule.
    """
    tiles = _neighbors(scorer, k)
    run_n = {j: int(scorer.r._nsrc(j)) for j in tiles}
    cf0 = scorer.clear_fwhm
    if sources_per_tile is not None or clear_fwhm is not None:
        n = {j: run_n[j] if sources_per_tile is None else min(run_n[j], int(sources_per_tile)) for j in tiles}
        return [{"counts": n, "clear_fwhm": cf0 if clear_fwhm is None else float(clear_fwhm), "rule": "asked"}]
    out = [{"counts": dict(run_n), "clear_fwhm": cf0, "rule": "the run's"}]
    if cf0 > RELAXED_FWHM:
        out.append({"counts": dict(run_n), "clear_fwhm": RELAXED_FWHM, "rule": "relaxed"})
    one = {j: run_n[j] if j == k else min(run_n[j], 1) for j in tiles}
    if one != run_n:
        out.append({"counts": one, "clear_fwhm": min(cf0, RELAXED_FWHM), "rule": "relaxed"})
    return out


def _rule_text(rule: Dict[str, Any], fwhm_px: float) -> str:
    """Describe a placement rule in words.

    An example is ``1 source in tile 1, 1 in tile 2 and 2 in tile 3, 2 FWHM (12.8 px) apart``.

    Parameters
    ----------
    rule : dict
        The rule, with ``counts`` and ``clear_fwhm``.
    fwhm_px : float
        FWHM, in pixels.

    Returns
    -------
    text : str
        The rule in words.
    """
    items = sorted((int(j), int(n)) for j, n in rule["counts"].items())
    parts = [f"{n} in tile {j + 1}" for j, n in items]
    parts[0] = parts[0].replace(" in ", " source in " if items[0][1] == 1 else " sources in ", 1)
    txt = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    cf = float(rule["clear_fwhm"])
    return f"{txt}, {cf:g} FWHM ({cf * fwhm_px:.1f} px) apart"


def _rule_json(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Return a placement rule as JSON, its tiles counted from 1.

    Parameters
    ----------
    rule : dict
        The rule, its tiles counted from 0.

    Returns
    -------
    record : dict
        ``counts``, ``clear_fwhm`` and ``rule``.
    """
    return {"counts": {str(int(j) + 1): int(n) for j, n in rule["counts"].items()},
            "clear_fwhm": float(rule["clear_fwhm"]), "rule": rule["rule"]}


def _rule_from_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a placement rule from its JSON (:func:`_rule_json`), its tiles counted from 0.

    Parameters
    ----------
    d : dict
        The record.

    Returns
    -------
    rule : dict
        ``counts``, ``clear_fwhm`` and ``rule``.
    """
    return {"counts": {int(j) - 1: int(n) for j, n in d["counts"].items()}, "clear_fwhm": float(d["clear_fwhm"]),
            "rule": d.get("rule", "")}


def _annulus_draws(scorer: StitchScorer, k: int, contrasts: Dict[int, float], n_draws: int, seed: int, sd: str,
                   rules: Sequence[Dict[str, Any]], log: Callable[[str], None]
                   ) -> Tuple[List[List[Tuple[int, Source]]], Dict[str, Any]]:
    """Return annulus ``k``'s draws and the rule they were placed by.

    They are the draws kept in ``sd`` from an earlier start, else those placed by the first of ``rules``
    that places every draw, each rule with its own random stream (the first the stream of the code
    before the rules).  New draws and their rule are written to ``annKK_draws.json`` and
    ``annKK_placement.json`` in ``sd``.

    Parameters
    ----------
    scorer : StitchScorer
        The scorer.
    k : int
        The annulus, counted from 0.
    contrasts : dict of int to float
        Contrast of each tile's sources.
    n_draws : int
        Number of draws.
    seed : int
        Seed of the random streams.
    sd : str
        The re-score's ``state`` directory.
    rules : sequence of dict
        The placement rules, in the order tried (:func:`_placements`).
    log : callable
        Function that takes each line of the log.

    Returns
    -------
    draws : list of list of tuple
        The draws, ``(tile, source)`` per source.
    rule : dict
        The rule they were placed by.

    Raises
    ------
    RescoreStop
        If the kept draws are not ``n_draws``, or were placed by another rule than the one asked.
    PlacementError
        If no rule places every draw.
    """
    dpath = os.path.join(sd, f"ann{k + 1:02d}_draws.json")
    ppath = os.path.join(sd, f"ann{k + 1:02d}_placement.json")
    fw = scorer.fwhm_px
    if os.path.exists(dpath):
        draws = [_draw_from_json(d) for d in _read_json(dpath)]
        if len(draws) != int(n_draws):
            raise RescoreStop(f"{dpath} holds {len(draws)} draws, not {n_draws}: give the re-score another "
                              f"directory, or the same number of draws")
        # draws kept before the rules were recorded were placed by the run's numbers and distance
        rule = _rule_from_json(_read_json(ppath)) if os.path.exists(ppath) else \
            {"counts": {j: int(scorer.r._nsrc(j)) for j in _neighbors(scorer, k)}, "clear_fwhm": scorer.clear_fwhm,
             "rule": "the run's"}
        if len(rules) == 1 and rules[0]["rule"] == "asked" and (
                rules[0]["counts"] != rule["counts"] or abs(rules[0]["clear_fwhm"] - rule["clear_fwhm"]) > 1e-9):
            raise RescoreStop(f"its draws in {sd} have {_rule_text(rule, fw)}, not {_rule_text(rules[0], fw)}: give "
                              f"the re-score another directory, or the same options")
        log(f"annulus {k + 1}: {len(draws)} draws kept from the last start, {_rule_text(rule, fw)} ({rule['rule']})")
        return draws, rule
    tiles = _neighbors(scorer, k)
    last = None
    for i, rule in enumerate(rules):
        rk = np.random.default_rng([int(seed), k] if i == 0 else [int(seed), k, i])
        try:
            draws = [scorer.draw(tiles, contrasts, rk, counts=rule["counts"], clear_fwhm=rule["clear_fwhm"])
                     for _ in range(int(n_draws))]
        except PlacementError as exc:
            last = exc
            log(f"annulus {k + 1}: no placement with {_rule_text(rule, fw)}: {exc}")
            continue
        _atomic_json(dpath, [_draw_json(d) for d in draws])
        _atomic_json(ppath, _rule_json(rule))
        log(f"annulus {k + 1}: {len(draws)} draws, {_rule_text(rule, fw)} ({rule['rule']})")
        return draws, rule
    tried = "the rule asked" if len(rules) == 1 else f"any of {len(rules)} rules"
    raise PlacementError(f"its draws find no placement under {tried} ({last}); --sources-per-tile and "
                         f"--clear-fwhm set the rule")


def rescore_plan(scorer: StitchScorer, entrants: Dict[int, List[Entrant]], n_draws: Union[int, Dict[int, int]],
                 n_cal: int = 8, max_iter: int = 4, apart: bool = False, calibrate: bool = True) -> Dict[str, Any]:
    """Count the reductions a re-score makes at most, without making any.

    They are the base winners' clean images, the calibration of each tile and, per annulus, the
    neighbors' and the entrants' reductions of its draws.

    Parameters
    ----------
    scorer : StitchScorer
        The scorer.
    entrants : dict of int to list of Entrant
        The entrants of each annulus scored, by 0-based annulus.
    n_draws : int or dict of int to int
        Number of draws, or per annulus.
    n_cal : int, default 8
        Number of draws of each calibration step.
    max_iter : int, default 4
        Number of calibration steps at most.
    apart : bool, default False
        Whether each tile's sources are reduced on their own.
    calibrate : bool, default True
        Whether the tiles are calibrated.

    Returns
    -------
    plan : dict
        ``base_clean``, ``calibration`` (per tile), ``annuli`` (per annulus its ``entrants``,
        ``affected``, ``draws``, ``own_per_draw``, ``neighbors`` and ``entrant`` counts), ``total`` and
        ``tiles`` ("apart" or "together").
    """
    R = 2.5 * float(scorer.r.fwhm)
    tiles_needed = sorted({j for k in entrants for j in _neighbors(scorer, k)})

    def reached(lo: float, hi: float) -> List[int]:
        return [a for a in scorer.order if scorer.reach(a)[0] <= hi and scorer.reach(a)[1] >= lo]

    def n_aff(tiles: Sequence[int], k: int) -> int:
        lo = min(scorer.tile(j)[0] for j in tiles) - R
        hi = max(scorer.tile(j)[1] for j in tiles) + R
        return len(set(reached(lo, hi)) | {k})

    cal = {j: max_iter * n_cal * n_aff([j], j) for j in tiles_needed} if calibrate else {}
    per = {}
    for k, ents in entrants.items():
        nd = int(n_draws[k] if isinstance(n_draws, dict) else n_draws)
        if apart:
            n_own = n_nb = 0
            for j in _neighbors(scorer, k):
                aff = reached(scorer.tile(j)[0] - R, scorer.tile(j)[1] + R)
                n_own += int(k in aff)
                n_nb += len(aff) - int(k in aff)
            per[k] = {"entrants": len(ents), "affected": n_aff(_neighbors(scorer, k), k), "draws": nd,
                      "own_per_draw": n_own, "neighbors": nd * n_nb, "entrant": len(ents) * nd * n_own}
        else:
            na = n_aff(_neighbors(scorer, k), k)
            per[k] = {"entrants": len(ents), "affected": na, "draws": nd, "own_per_draw": 1,
                      "neighbors": nd * (na - 1), "entrant": len(ents) * nd}
    total = len(scorer.order) + sum(cal.values()) + sum(v["neighbors"] + v["entrant"] for v in per.values())
    return {"base_clean": len(scorer.order), "calibration": cal, "annuli": per, "total": total,
            "tiles": "apart" if apart else "together"}


def _base_reduce_seconds(run_dir: str) -> Optional[float]:
    """Return the median time the base run's search took to reduce a trial.

    Parameters
    ----------
    run_dir : str
        Directory of the base run.

    Returns
    -------
    seconds : float or None
        The median ``reduce_s`` in ``results.jsonl``, the validation left out, None without any.
    """
    path = os.path.join(run_dir, "results.jsonl")
    vals = []
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                v = r.get("reduce_s")
                if v is not None and r.get("phase") not in ("validation",):
                    vals.append(float(v))
    return float(np.median(vals)) if vals else None


def rescore(runner, entrant_runs: Sequence[str], out_dir: str, *, annuli: Sequence[int],
            n_draws: Optional[int] = None, top: int = 0, seed: int = 1, target: Tuple[float, float] = TARGET,
            n_cal: int = 8, max_iter: int = 4, window: Optional[int] = None, overlap_pads: Optional[bool] = None,
            plan: bool = False, card: bool = False, sources_per_tile: Optional[int] = None,
            clear_fwhm: Optional[float] = None, base_winners: Optional[Sequence[Pick]] = None,
            candidates: Optional[Sequence[Pick]] = None, draws_from: Optional[str] = None,
            tiles: Optional[str] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Score the candidates of some annuli on the base run's stitch, the other annuli at the base winners.

    Each annulus' candidates are its base winner first, then the entrants of the base run and of
    ``entrant_runs`` (:func:`run_entrants`), or with ``candidates`` only the base run's winner and those
    picks.  Each tile's contrast is calibrated on the base winners' stitch
    (:meth:`StitchScorer.calibrate`), or taken from ``draws_from``.  An annulus whose draws find no
    place under any rule stops with a message, and the re-score goes on with the next.

    It writes ``rescore.json`` and ``rescore.txt`` to ``out_dir`` after each annulus, and keeps its
    draws, its calibration, its options and each entrant's S/N in ``out_dir/state``, from which a
    re-score started again on the same directory resumes.

    Parameters
    ----------
    runner : Runner
        The base run, resumed read-only.
    entrant_runs : sequence of str
        Directories of the other runs of the same setup whose candidates enter too.  The base run's
        always do.
    out_dir : str
        Directory of the re-score, beside the runs.
    annuli : sequence of int
        The annuli scored, counted from 0.
    n_draws : int or None, default None
        Number of common draws per annulus.  None takes 50, or all of ``draws_from``'s.
    top : int, default 0
        Number of the best evaluations of each run's history that enter too.
    seed : int, default 1
        Seed of the draws and the calibration.
    target : tuple of 2 floats, default TARGET
        Range of the base winners' median S/N the calibration seeks.
    n_cal : int, default 8
        Number of draws of each calibration step.
    max_iter : int, default 4
        Number of calibration steps at most.
    window : int or None, default None
        Number of reductions in flight at most.  None takes the runner's.
    overlap_pads : bool or None, default None
        Whether overlapping annuli take pads between them.  None keeps the base run's.
    plan : bool, default False
        Whether to count the reductions and return, making none.
    card : bool, default False
        Whether the injected reductions may go to the card workers.
    sources_per_tile : int or None, default None
        Most sources in each tile of a draw, which with ``clear_fwhm`` sets the one placement rule.
    clear_fwhm : float or None, default None
        Distance between the tiles' sources, in FWHM.  Without it and ``sources_per_tile``, an annulus
        whose draws find no place under the run's rule falls back as :func:`_placements` lists.
    base_winners : sequence of Pick or None, default None
        Picks (:func:`read_picks`) whose configurations replace the base run's winners of their annuli,
        which then enter as candidates.
    candidates : sequence of Pick or None, default None
        The only picks that enter, with each annulus' base winner.
    draws_from : str or None, default None
        Directory of an earlier re-score of the same base run, whose draws, their rule and contrasts
        this one takes, without calibration, and against whose base winner each annulus' base is
        compared, draw for draw.
    tiles : {"apart", "together"} or None, default None
        Whether each tile's sources are reduced and measured on their own, or all of a draw's at once.
        None takes "apart" for a new directory, the directory's own for one started before, and
        ``draws_from``'s with it.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    report : dict
        The report written to ``rescore.json``, with each annulus' ranked entrants and their paired
        differences from the base winner.  With ``plan``, ``plan`` (:func:`rescore_plan`), ``reduce_s``,
        ``tiles`` and ``entrants`` instead.

    Raises
    ------
    ValueError
        If an option is out of range or clashes with another, ``out_dir`` lies inside a run, the
        reducer's frames or a run's setup are not the base run's, an annulus has two base winners, or
        ``out_dir`` was started with other options.
    """
    if sources_per_tile is not None and int(sources_per_tile) < 1:
        raise ValueError("sources_per_tile must be at least 1")
    if clear_fwhm is not None and not float(clear_fwhm) > 0:
        raise ValueError("clear_fwhm must be positive")
    if n_draws is not None and int(n_draws) < 1:
        raise ValueError("n_draws must be at least 1")
    if tiles is not None and tiles not in TILES_MODES:
        raise ValueError(f"tiles {tiles!r}: one of {', '.join(TILES_MODES)}")
    t0 = time.time()
    base_dir = os.path.abspath(runner.run_dir)
    runs = [base_dir] + [os.path.abspath(p) for p in entrant_runs if os.path.abspath(p) != base_dir]
    out_dir = os.path.abspath(out_dir)
    for rd in runs:
        if out_dir == rd or out_dir.startswith(rd + os.sep):
            raise ValueError(f"{out_dir} is inside {rd}: the re-score writes beside the runs")
    if draws_from:
        draws_from = os.path.abspath(draws_from)
        if sources_per_tile is not None or clear_fwhm is not None:
            raise ValueError("the draws of --draws-from come placed: --sources-per-tile and --clear-fwhm do not go "
                             "with it")
        if draws_from == out_dir:
            raise ValueError("--draws-from is this re-score's own directory")
        if not os.path.isdir(os.path.join(draws_from, "state")):
            raise ValueError(f"{draws_from} holds no re-score (no state directory)")
        try:
            dbase = _read_json(os.path.join(draws_from, "rescore.json")).get("base")
        except (OSError, ValueError):
            dbase = None
        if dbase and os.path.abspath(dbase) != base_dir:
            raise ValueError(f"{os.path.basename(draws_from)} re-scored {os.path.basename(dbase)}, not "
                             f"{os.path.basename(base_dir)}: its draws lie in another run's tiles")
    scorer = StitchScorer(runner, overlap_pads=overlap_pads, log=log, card=card)
    from .runner import _core_edits, _describe_edits
    frames = _describe_edits(_core_edits(dict(getattr(runner.reducer, "data_edits", None) or {})))
    run_frames = _describe_edits(_core_edits(runner.cfg.data_edits))
    if frames != run_frames:
        raise ValueError(f"the reducer reduces {frames}, the base run was made on {run_frames}: the re-score "
                         f"scores on the runs' own frames")
    ref = _setup(base_dir)
    for rd in runs[1:]:
        _check_setup(rd, ref)
    # the composite, and the candidates of a file
    bw: Dict[int, Entrant] = {}
    for p, e in resolve_picks(base_dir, base_winners or [], ref=ref):
        if p.annulus in bw:
            raise ValueError(f"annulus {p.annulus + 1} has two base winners")
        bw[p.annulus] = e
    picked: Optional[Dict[int, List[Entrant]]] = None
    if candidates is not None:
        picked = {}
        for p, e in resolve_picks(base_dir, candidates, ref=ref):
            picked.setdefault(p.annulus, []).append(e)
    for rd in {os.path.abspath(p.run) for p in list(base_winners or []) + list(candidates or [])}:
        if out_dir == rd or out_dir.startswith(rd + os.sep):
            raise ValueError(f"{out_dir} is inside {rd}: the re-score writes beside the runs")
    if bw:
        scorer.use_winners(bw)
        for a, e in sorted(bw.items()):
            log(f"  base winner of annulus {a + 1}: {e.run} {e.kind} eval {e.eval_index} (as rescore.json counts; "
                f"search score {_f(e.search_score)}, validated {_f(e.validated_score)})")
    # what this directory scores under: a directory started before keeps its options
    sd = os.path.join(out_dir, "state")
    bw_rec = {str(a + 1): {"run": e.run, "kind": e.kind, "eval": e.eval_index, "key": e.key}
              for a, e in sorted(bw.items())}
    rec = _options_record(sd)
    if rec is not None:
        was = rec.get("tiles", "together")
        if tiles is not None and tiles != was:
            raise ValueError(f"{out_dir} was started with --tiles {was}: give the same, or another --out")
        rkeys = {a: (v.get("key") if isinstance(v, dict) else v) for a, v in (rec.get("base_winners") or {}).items()}
        if rkeys != {a: v["key"] for a, v in bw_rec.items()}:
            what = "other base winners" if rkeys else "the base run's winners"
            raise ValueError(f"{out_dir} was started with {what}: give the same --base-winners, or another --out")
        if (rec.get("draws_from") or None) != (draws_from or None):
            raise ValueError(f"{out_dir} was started with "
                             f"{('--draws-from ' + rec['draws_from']) if rec.get('draws_from') else 'its own draws'}: "
                             f"give the same, or another --out")
        mode = was
    elif tiles is not None:
        mode = tiles
    elif draws_from:
        mode = (_options_record(os.path.join(draws_from, "state")) or {}).get("tiles", "together")
    else:
        mode = "apart"
    apart = mode == "apart"
    # the candidates of each annulus, its base winner first
    ents: Dict[int, List[Entrant]] = {}
    for k in annuli:
        lst: List[Entrant] = []
        seen = set()

        def add(e: Entrant) -> None:
            if e.key not in seen:
                seen.add(e.key)
                lst.append(e)
        if k in bw:
            add(bw[k])
        own = run_entrants(base_dir, k, top=top if picked is None else 0)
        if picked is None:
            for rd in runs:
                for e in (own if rd == base_dir else run_entrants(rd, k, top=top)):
                    add(e)
        else:
            for e in own:
                if e.kind == "winner":
                    add(e)
            for e in picked.get(k, []):
                add(e)
        ents[k] = lst
    base_key = {k: (bw[k].key if k in bw else config_key(scorer.results[k].winner_config)) for k in annuli}
    # the draws per annulus
    nd: Dict[int, int] = {}
    for k in annuli:
        if draws_from:
            kept = os.path.join(sd, f"ann{k + 1:02d}_draws.json")
            src = os.path.join(draws_from, "state", f"ann{k + 1:02d}_draws.json")
            m = len(_read_json(kept)) if os.path.exists(kept) else (len(_read_json(src)) if os.path.exists(src) else 0)
            nd[k] = m if n_draws is None else int(n_draws)
        else:
            nd[k] = 50 if n_draws is None else int(n_draws)
    pl = rescore_plan(scorer, ents, nd, n_cal, max_iter, apart=apart, calibrate=not draws_from)
    sec = _base_reduce_seconds(base_dir)
    nds = sorted(set(nd.values()))
    log(f"re-score of {os.path.basename(base_dir)}: annuli {', '.join(str(k + 1) for k in annuli)}, candidates from "
        f"{len(runs)} run(s), {nds[0] if len(nds) == 1 else nds} common draws per annulus, each draw's tiles "
        f"{mode}, at most {pl['total']} reductions")
    if draws_from:
        log(f"  the draws, their rule and the contrasts from {draws_from}")
    for k in annuli:
        p = pl["annuli"][k]
        own_txt = "" if not apart else f", the candidate reduced {p['own_per_draw']} time(s) per draw"
        log(f"  annulus {k + 1}: {p['entrants']} candidates x {p['draws']} draws{own_txt}, {p['affected']} annuli "
            f"reached by a draw: {p['entrant'] + p['neighbors']} reductions")
    if sec is not None:
        per_trial = 1 + int(getattr(runner.cfg, "n_remeasure", 1) or 1)
        inflight = max(int(getattr(runner.cfg, "batch", 1) or 1), 1)
        hours = pl["total"] * sec / (per_trial * inflight) / 3600.0
        log(f"  the base run's search took a median {sec:.1f} s to reduce a trial of {per_trial} reductions, "
            f"{inflight} trial(s) in flight: at that pace about {hours:.1f} h")
    if plan:
        return {"plan": pl, "reduce_s": sec, "tiles": mode, "entrants": {k + 1: [asdict(e) for e in v]
                                                                        for k, v in ents.items()}}
    os.makedirs(sd, exist_ok=True)
    if rec is None:
        _atomic_json(os.path.join(sd, "options.json"), {
            "tiles": mode, "base_winners": bw_rec, "draws_from": draws_from,
            "started": _dt.datetime.now().isoformat(timespec="seconds")})
    runner._start_workers()
    w = int(window or runner._reduction_window())
    scorer.prepare(w)
    log(f"  stitch weights from {scorer.weights_from}, pads between overlapping annuli "
        f"{'on' if scorer.overlap_pads else 'off'}, {w} reductions in flight")
    # the contrast of each tile the draws use, on the base winners' stitch, or --draws-from's
    cpath = os.path.join(sd, "calibration.json")
    cal = {int(k): v for k, v in (_read_json(cpath).items() if os.path.exists(cpath) else [])}
    needed = sorted({j for k in annuli for j in _neighbors(scorer, k)})
    if draws_from:
        dpath = os.path.join(draws_from, "state", "calibration.json")
        dcal = {int(k): v for k, v in (_read_json(dpath).items() if os.path.exists(dpath) else [])}
        new = {j: dict(dcal[j], taken_from=draws_from) for j in needed if j not in cal and j in dcal}
        if new:
            cal.update(new)
            _atomic_json(cpath, {str(a): v for a, v in cal.items()})
    else:
        for j in needed:
            if j not in cal:
                cal[j] = scorer.calibrate(j, np.random.default_rng([int(seed), 1000 + j]), n_cal=n_cal, target=target,
                                          max_iter=max_iter, window=w,
                                          n_src=None if sources_per_tile is None
                                          else min(int(scorer.r._nsrc(j)), int(sources_per_tile)))
                _atomic_json(cpath, {str(a): v for a, v in cal.items()})
    contrasts = {j: float(v["contrast"]) for j, v in cal.items()}
    report: Dict[str, Any] = {"base": base_dir, "runs": runs, "annuli": {}, "seed": int(seed),
                              "n_draws": nds[0] if len(nds) == 1 else nds, "target": list(target),
                              "calibration": {str(j + 1): v for j, v in cal.items()},
                              "tiles_px": [list(t) for t in scorer.tiles], "weights": [float(v) for v in scorer.weights],
                              "weights_from": scorer.weights_from, "overlap_pads": scorer.overlap_pads,
                              "tiles": mode, "base_winners": {str(a + 1): asdict(e) for a, e in sorted(bw.items())},
                              "draws_from": draws_from,
                              "candidates": None if picked is None else
                              {str(a + 1): [_entrant_brief(asdict(e)) for e in v] for a, v in sorted(picked.items())}}
    stopped: Dict[int, str] = {}
    for k in annuli:
        try:
            if draws_from:
                _copy_draws(draws_from, sd, k, nd[k])
            draws, rule = _annulus_draws(scorer, k, contrasts, nd[k], int(seed), sd,
                                         _placements(scorer, k, sources_per_tile, clear_fwhm), log)
        except RescoreStop as exc:
            stopped[k] = str(exc)
            log(f"annulus {k + 1}: stopped, {exc}")
            report["annuli"][str(k + 1)] = {"tile_px": list(scorer.tile(k)),
                                            "draw_tiles": [j + 1 for j in _neighbors(scorer, k)],
                                            "stopped": str(exc), "position": scorer.pos(k)}
            _write_reports(out_dir, report, log=None)
            continue
        tiles_of = [[j for j, _ in d] for d in draws]
        # each draw as the parts reduced together: the whole draw, or each tile's sources apart
        subs = [_subdraws(d, apart) for d in draws]
        parts = [(t, g) for t in range(len(draws)) for g in range(len(subs[t]))]
        reach = {(t, g): scorer.affected(subs[t][g][1], k, force=not apart) for t, g in parts}
        # the neighbors' reductions of each part, shared by every candidate
        nb_jobs, nb_slots = [], []
        for t, g in parts:
            src, j = subs[t][g][1], subs[t][g][2]
            for a in reach[(t, g)]:
                if a != k:
                    nb_jobs.append(lambda a=a, src=src, tag=_tag(a, t, j): scorer.reduce(a, scorer.base_cfg[a], src,
                                                                                         tag))
                    nb_slots.append((t, g, a))
        todo = [e for e in ents[k] if not os.path.exists(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"))]
        nb: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {p: {} for p in parts}
        # a part whose sources do not reach annulus k reads the same for every candidate
        fixed: Dict[Tuple[int, int], np.ndarray] = {}
        if todo:
            log(f"annulus {k + 1}: the neighbors' reductions of {len(draws)} draws ({len(nb_jobs)})")
            for i, img in scorer.run(nb_jobs, w):
                t, g, a = nb_slots[i]
                nb[(t, g)][a] = img
            for p in parts:
                if k not in reach[p]:
                    fixed[p] = scorer.measure(scorer.stitch(nb[p]), subs[p[0]][p[1]][1])
            log(f"annulus {k + 1}: {len(todo)} of {len(ents[k])} candidates to score")
        own = [p for p in parts if k in reach[p]]
        n_own = {t: 0 for t in range(len(draws))}
        for t, _ in own:
            n_own[t] += 1
        jobs, slots = [], []
        for e in todo:
            cfg = runner._config_from_dict(e.config)
            for t, g in own:
                src, j = subs[t][g][1], subs[t][g][2]
                jobs.append(lambda cfg=cfg, src=src, tag=_tag(k, t, j, cand=True): scorer.reduce(k, cfg, src, tag))
                slots.append((e, t, g))
        acc: Dict[str, Dict[str, Any]] = {}
        t_ann = time.time()
        n_here = 0

        def started(e: Entrant) -> Dict[str, Any]:
            rows = []
            for t, d in enumerate(draws):
                v = np.full(len(d), np.nan)
                for g, (idx, _, _) in enumerate(subs[t]):
                    if (t, g) in fixed:
                        v[idx] = fixed[(t, g)]
                rows.append(v)
            return {"rows": rows, "left": dict(n_own)}

        def finished(e: Entrant, row: Dict[str, Any]) -> None:
            nonlocal n_here
            _atomic_json(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"),
                         {"entrant": asdict(e), "snr": [[float(x) for x in v] for v in row["rows"]]})
            done = sum(1 for x in ents[k] if os.path.exists(os.path.join(sd, f"ann{k + 1:02d}_{x.tag}.json")))
            el = time.time() - t_ann
            n_here += 1
            left = (len(todo) - n_here) * el / n_here
            log(f"  annulus {k + 1}: {e.run} {e.kind} eval {e.eval_index}: median S/N "
                f"{_nanmedian([_nanmedian(v) for v in row['rows']]):.2f} ({done}/{len(ents[k])}, {el / 60:.1f} min, "
                f"about {left / 60:.0f} min left in this annulus)")

        if not own:
            for e in todo:
                finished(e, started(e))
        for i, img in scorer.run(jobs, w):
            e, t, g = slots[i]
            row = acc.get(e.tag)
            if row is None:
                row = acc[e.tag] = started(e)
            idx, src, _ = subs[t][g]
            row["rows"][t][idx] = scorer.measure(scorer.stitch({**nb[(t, g)], k: img}), src)
            row["left"][t] -= 1
            if not any(row["left"].values()):
                finished(e, row)
                del acc[e.tag]
        rep_k = _annulus_report(scorer, k, ents[k], draws, tiles_of, sd, base_key[k])
        rep_k["placement"] = dict(_rule_json(rule), text=_rule_text(rule, scorer.fwhm_px))
        rep_k["n_draws"] = len(draws)
        if k in bw:
            rep_k["base_winner"] = _entrant_brief(asdict(bw[k]))
        if draws_from:
            rep_k["against"] = _against(draws_from, k, rep_k, draws, tiles_of)
        report["annuli"][str(k + 1)] = rep_k
        _write_reports(out_dir, report, log=None)
    if stopped:
        ks = [str(k + 1) for k in stopped]
        log(f"stopped: {'annulus' if len(ks) == 1 else 'annuli'} {', '.join(ks)}, for the reasons logged above")
    report["reductions"] = scorer.n_reductions
    report["wall_s"] = time.time() - t0
    report["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    _write_reports(out_dir, report, log=log)
    return report


def _annulus_report(scorer: StitchScorer, k: int, ents: Sequence[Entrant], draws, tiles_of, sd: str,
                    base_key: str) -> Dict[str, Any]:
    """Rank an annulus' entrants by their statistics over the common draws.

    Each entrant's statistic is, per draw, the median of its sources' S/N, for all of the draw's
    sources, for tile ``k``'s alone and for each tile's, with the paired differences from the base
    winner (:func:`paired_summary`).  The entrants rank by their median over all the sources, then by
    their mean difference.

    Parameters
    ----------
    scorer : StitchScorer
        The scorer.
    k : int
        The annulus, counted from 0.
    ents : sequence of Entrant
        The entrants, each with its S/N in ``sd``.
    draws : sequence of sequence of tuple
        The draws, ``(tile, source)`` per source.
    tiles_of : sequence of sequence of int
        Each draw's tile of each source.
    sd : str
        The re-score's ``state`` directory.
    base_key : str
        The base winner's :func:`config_key`.

    Returns
    -------
    report : dict
        ``tile_px``, ``draw_tiles`` (1-based), ``draws``, ``entrants`` (ranked, each with ``entrant``,
        ``base``, ``snr``, ``all``, ``tile``, ``tiles`` and ``rank``), ``base_rank``, ``position`` and
        ``base_tiles`` (the base winner's median S/N per tile).
    """
    rows = []
    for e in ents:
        rec = _read_json(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"))
        snr = [np.asarray(v, float) for v in rec["snr"]]
        rows.append({"entrant": asdict(e), "base": e.key == base_key, "snr": [[float(x) for x in v] for v in snr],
                     "_per": _per_draw(snr, tiles_of)})
    base = next((r for r in rows if r["base"]), None)
    none = np.full(len(draws), np.nan)
    for r in rows:
        per = r["_per"]
        bper = base["_per"] if base else {}
        r["all"] = paired_summary(per["all"], bper.get("all", none))
        r["tile"] = paired_summary(per[k] if k in per else none, bper.get(k, none))
        r["tiles"] = {str(j + 1): paired_summary(per[j], bper.get(j, none)) for j in per if j != "all"}
    rows.sort(key=lambda r: (-(r["all"]["median"] if np.isfinite(r["all"]["median"]) else -np.inf),
                             -(r["all"]["diff_mean"] if np.isfinite(r["all"]["diff_mean"]) else -np.inf)))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r.pop("_per")
    p = scorer.pos(k)
    return {"tile_px": list(scorer.tile(k)), "draw_tiles": [j + 1 for j in _neighbors(scorer, k)],
            "draws": [_draw_json(d) for d in draws], "entrants": rows,
            "base_rank": next((r["rank"] for r in rows if r["base"]), None), "position": p,
            "base_tiles": None if base is None else {j: v["median"] for j, v in base["tiles"].items()}}


def _f(v, fmt="{:.2f}") -> str:
    """Format a number, "-" where it is None or not finite.

    Parameters
    ----------
    v : float or None
        The value.
    fmt : str, default "{:.2f}"
        The format.

    Returns
    -------
    text : str
        The value formatted.
    """
    return fmt.format(v) if v is not None and np.isfinite(v) else "-"


def _ev1(v) -> str:
    """Return an evaluation as the text reports count it, from 1.

    Parameters
    ----------
    v : int or None
        The evaluation, counted from 0.

    Returns
    -------
    text : str
        The evaluation counted from 1, "-" for None.
    """
    return "-" if v is None else str(int(v) + 1)


def _diff_text(st: Optional[Dict[str, Any]]) -> str:
    """Format a paired difference as "mean +/- standard error".

    Parameters
    ----------
    st : dict or None
        The summary, from :func:`paired_summary`.

    Returns
    -------
    text : str
        The difference, "-" without a summary.
    """
    return "-" if not st else f"{_f(st.get('diff_mean'), '{:+.2f}')} +/- {_f(st.get('diff_se'))}"


def _write_reports(out_dir: str, report: Dict[str, Any], log: Optional[Callable[[str], None]]) -> None:
    """Write the re-score's ``rescore.json`` and ``rescore.txt`` to its directory.

    Each is written through a temporary file replaced into place.

    Parameters
    ----------
    out_dir : str
        Directory of the re-score.
    report : dict
        The report.
    log : callable or None
        Function that takes the line saying where the reports went.  None logs nothing.
    """
    _atomic_json(os.path.join(out_dir, "rescore.json"), report)
    mode = report.get("tiles", "together")
    bw = report.get("base_winners") or {}
    L = [f"# klip_tpe re-score -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         f"# base run: {report['base']}",
         f"# candidates from: {', '.join(os.path.basename(r) for r in report['runs'])}"
         + ("" if report.get("candidates") is None else " (only those of --candidates, with each base winner)"),
         f"# {report['n_draws']} common draws per annulus, sources in the annulus' tile and its neighbors', each "
         f"tile's at the contrast that gives the base winners a median S/N of {report['target'][0]:g}-"
         f"{report['target'][1]:g} on the stitch" + (f", all as {report['draws_from']} drew them"
                                                      if report.get("draws_from") else ""),
         ("# tiles apart: each tile's sources reduced and stitched on their own and measured there, at the contrast "
          "set with them alone" if mode == "apart" else
          "# tiles together: all of a draw's sources in each reduction, as the re-score did before 2026-11-11"),
         "# each source measured on the injected stitch as the stitched verify measures it (Path A), the other "
         "annuli at the base winners"]
    if bw:
        L.append("# base winners: " + ", ".join(
            f"annulus {a} {e['run']} {e['kind']} eval {_ev1(e.get('eval_index'))}" for a, e in bw.items())
            + "; the other annuli the base run's")
    L += ["# per draw: the median S/N of the draw's sources (all) and of the annulus' own tile's (tile).  diff: the "
          "paired difference from the base winner (*) over the draws, mean +/- standard error, and P: the share of "
          "draws where the candidate reads higher",
          "# eval: counted from 1, as results.txt counts (rescore.json and the --base-winners and --candidates files "
          "count from 0)", "#"]
    L.append("# tile contrasts: " + ", ".join(f"{j} {float(v['contrast']):.3e}" + ("" if v.get("converged") else " (?)")
                                             for j, v in report["calibration"].items()))
    for ann, a in report["annuli"].items():
        t = a["tile_px"]
        tiles = '-'.join(str(v) for v in (a['draw_tiles'][0], a['draw_tiles'][-1]))
        if a.get("stopped"):
            L += ["", f"== annulus {ann}  tile {t[0]:.2f}-{t[1]:.2f} px  draws in tiles {tiles}  stopped ==",
                  f"  {a['stopped']}"]
            continue
        L += ["", f"== annulus {ann}  tile {t[0]:.2f}-{t[1]:.2f} px  draws in tiles {tiles}  "
                  f"(the base winner ranks {a['base_rank']} of {len(a['entrants'])}) =="]
        if a.get("placement"):
            L.append(f"  draws: {a['placement']['text']} ({a['placement']['rule']})")
        L.append(f"  {'rank':>4}  {'run':<24} {'kind':<10} {'eval':>6}  {'S/N all (16-84%)':>20}  "
                 f"{'diff all':>16}  {'P':>4}  {'S/N tile':>8}  {'diff tile':>16}")
        for r in a["entrants"]:
            e, s, st = r["entrant"], r["all"], r["tile"]
            star = "*" if r["base"] else " "
            rng = f"{_f(s['median'])} ({_f(s['p16'], '{:.1f}')}-{_f(s['p84'], '{:.1f}')})"
            dif = "0" if r["base"] else _diff_text(s)
            dift = "0" if r["base"] else _diff_text(st)
            ev = _ev1(e.get("eval_index"))
            L.append(f"  {r['rank']:>3}{star}  {e['run']:<24} {e['kind']:<10} {ev:>6}  {rng:>20}  {dif:>16}  "
                     f"{_f(s['p_better'], '{:.2f}') if not r['base'] else '-':>4}  {_f(st['median']):>8}  {dift:>16}")
        if a.get("base_tiles"):
            L.append("  the base winner per tile, median S/N: "
                     + ", ".join(f"tile {j} {_f(v)}" for j, v in sorted(a["base_tiles"].items(), key=lambda x: int(x[0]))))
        ag = a.get("against")
        if ag:
            name = os.path.basename(ag["dir"])
            if ag.get("missing"):
                L.append(f"  against {name}: {ag['missing']}")
            else:
                b = ag["base"]
                ev = _ev1(b.get("eval_index"))
                there = f"tiles {ag['tiles']} there" + (", a composite base" if ag.get("base_winners") else "")
                per = ", ".join(f"tile {j} {_diff_text(v)}"
                                for j, v in sorted((ag.get("per_tile") or {}).items(), key=lambda x: int(x[0])))
                L.append(f"  against {name}'s base ({b['run']} {b['kind']} eval {ev}, {there}): all "
                         f"{_diff_text(ag['all'])}, {per}")
    if "reductions" in report:
        L += ["", f"# {report['reductions']} reductions in {report['wall_s'] / 3600:.2f} h"]
    with open(os.path.join(out_dir, "rescore.txt") + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(os.path.join(out_dir, "rescore.txt") + ".tmp", os.path.join(out_dir, "rescore.txt"))
    if log is not None:
        log(f"  wrote {os.path.join(out_dir, 'rescore.txt')} and rescore.json")
