"""The re-score: candidates for an annulus scored on the stitch, the first piece of ``refine``
(NEAR2's notes of 2026-11-06 to 2026-11-09).

A finished run, the base, gives the stitch: its annuli, tiles, pads and weights, and its winners
for every annulus but the one being scored.  The candidates for annulus k are configurations of
the base run and of other runs of the same setup: their winners, their validation's entrants and,
when asked, the best of their histories.  Each candidate is reduced on common draws.  A draw puts
sources in tile k and in tiles k - 1 and k + 1, each tile's at the contrast that makes the base
winners read S/N 5 to 8 on the stitch (:meth:`StitchScorer.calibrate`).

The annuli whose zones hold any of a draw's sources are reduced with all of them: the base
winners for the neighbors and the candidate for annulus k, as the final stitch reduces every
annulus with every source (``RunConfig.stitch_injections = "all"``).  The other annuli take the
base winners' clean images, which those sources do not reach.  The injected stitch is built as
the final stitch builds it, with the base run's tiles, pads, weights and flattening, and each
source is measured as the stitched verify measures it (Path A of
:func:`klip_tpe.verify.verify_injections`).  The seam patches of the final stitch go into its
clean image only, which the injected sources are not measured on.

The runs are only read.  :func:`rescore` writes ``rescore.txt`` and ``rescore.json`` to its own
directory, with a ``state/`` directory from which an interrupted re-score resumes.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source

__all__ = ["Entrant", "StitchScorer", "config_key", "run_entrants", "parse_annuli", "rescore", "rescore_plan",
           "paired_summary", "PlacementError", "RescoreStop"]

#: the S/N the base winners should read on the stitch, at the calibrated contrasts
TARGET = (5.0, 8.0)

#: the clearance between the tiles' sources (FWHM) a draw falls back to when the sampler's own
#: finds no placement (NEAR2, 2026-11-10: s7's tiles are 4.7 px wide against a FWHM of 6.4 px)
RELAXED_FWHM = 1.5


class RescoreStop(RuntimeError):
    """An annulus of the re-score stops here; the re-score goes on with the next one."""


class PlacementError(RescoreStop):
    """No placement of a draw's sources under the rule asked."""


@dataclass
class Entrant:
    """One candidate configuration for an annulus: the run it comes from, how it got there
    (``"winner"``, ``"validation"`` or ``"top"``, the best of the history), its evaluation and
    its configuration as the run stored it (``Config.to_dict``)."""
    run: str
    kind: str
    eval_index: Optional[int]
    config: Dict[str, Any]
    search_score: Optional[float] = None
    validated_score: Optional[float] = None

    @property
    def key(self) -> str:
        return config_key(self.config)

    @property
    def tag(self) -> str:
        """A short, stable name for its state file."""
        return hashlib.sha1((self.run + "|" + self.key).encode()).hexdigest()[:12]


def config_key(d: Dict[str, Any]) -> str:
    """A configuration's identity: its parameters, its per-partition blocks and its partitions."""
    return json.dumps({"params": d.get("params", {}), "per_partition": d.get("per_partition", {}),
                       "selected": sorted(str(s) for s in d.get("selected", []))}, sort_keys=True, default=float)


def parse_annuli(spec: str, nann: int) -> List[int]:
    """``"1-5"``, ``"2,4,6"`` or ``"all"`` (1-based) as 0-based annulus indices."""
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
    with open(path) as f:
        return json.load(f)


def run_entrants(run_dir: str, annulus: int, top: int = 0) -> List[Entrant]:
    """The candidates a finished run offers for annulus ``annulus`` (0-based): the configurations
    its validation scored, the elected one marked ``"winner"``, and with ``top`` the ``top``
    best of its history that are not among them.  Read from ``final_results.json`` and
    ``results.jsonl``."""
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


def _setup(run_dir: str) -> Dict[str, Any]:
    """What must agree between the runs: the annuli, the space's dimensions and the frames."""
    from .runner import _core_edits, load_checkpoint
    st, _ = load_checkpoint(run_dir, log=lambda s: None)
    if st is None:
        raise ValueError(f"{run_dir}: no readable checkpoint")
    cfg = st.get("config") or {}
    fr = _read_json(os.path.join(run_dir, "final_results.json"))
    zones = [(round(float(a["inrad"]), 6), round(float(a["outrad"]), 6)) for a in fr.get("annuli") or []]
    return {"zones": zones, "names": [p.get("name") for p in (st.get("space") or {}).get("params", [])],
            "frames": json.dumps(_core_edits(cfg.get("data_edits")), sort_keys=True, default=str)}


def _nanmedian(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _pct(v, q) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def paired_summary(per_draw: np.ndarray, base: np.ndarray) -> Dict[str, Any]:
    """An entrant's statistic over the common draws, and its paired difference from the base
    winner's: the mean difference with its standard error, the median difference and the
    fraction of draws where the entrant reads higher."""
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
    """The base run's stitch with one annulus' images replaced, and sources measured on it as the
    stitched verify measures them.  ``runner`` is the base run resumed read-only
    (``Runner.resume(..., same_frames=False)``), ``overlap_pads`` overrides its pads between
    overlapping annuli (``RunConfig.stitch_overlap_pads``), as ``rereduce`` does.  ``card``: the
    injected reductions may go to the card workers, as the search's draws do."""

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
        self.known = [tuple(k) for k in (getattr(runner.sampler, "known", ()) or ())]
        self.clean: Dict[int, np.ndarray] = {}
        self.weights: Optional[np.ndarray] = None
        self.weights_from = ""
        self.n_reductions = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- geometry
    def pos(self, ia: int) -> int:
        return self.order.index(ia)

    def tile(self, ia: int) -> Tuple[float, float]:
        return self.tiles[self.pos(ia)]

    def _zone_pad(self) -> float:
        red = self.r.reducer
        zp = getattr(red, "zone_pad", None)
        if zp is None and getattr(red, "reducers", None):
            zp = max((float(getattr(s, "zone_pad", 0.0) or 0.0) for s in red.reducers.values()), default=None)
        return 2.0 if zp is None else float(zp)

    def reach(self, ia: int) -> Tuple[float, float]:
        """Radii annulus ``ia``'s KLIP zone can cover under its base winner (px): its zone with the
        reducer's pad and the winner's ``zone_in`` / ``zone_out``, and 1 px to spare."""
        cfg = self.base_cfg[ia]
        blocks = [cfg.params] + list(cfg.per_partition.values())
        zi = max([float(b.get("zone_in", 0.0) or 0.0) for b in blocks] + [0.0])
        zo = max([float(b.get("zone_out", 0.0) or 0.0) for b in blocks] + [0.0])
        a_in, a_out = self.zones[self.pos(ia)]
        m = self._zone_pad() + 1.0
        return max(a_in - zi - m, 0.0), a_out + zo + m

    def affected(self, sources: Sequence[Source], k: int) -> List[int]:
        """The annuli whose zones a draw's sources reach (within 2.5 FWHM of each), and ``k``."""
        R = 2.5 * float(self.r.fwhm)
        rs = [float(s.rho) / float(self.r.pxscale) for s in sources]
        out = []
        for ia in self.order:
            lo, hi = self.reach(ia)
            if ia == k or any(lo - R <= r <= hi + R for r in rs):
                out.append(ia)
        return out

    # ---------------------------------------------------------------- reductions
    def reduce(self, ia: int, cfg, sources: Optional[Sequence[Source]], tag: str) -> np.ndarray:
        """``cfg`` reduced in annulus ``ia``'s zone, with ``sources`` (no KLIP-FM pass)."""
        more = {"card": True} if (self.card and sources) else {}
        ev = self.r._reduce(cfg, list(sources) if sources else None, tag=tag, zone=self.r._zone(ia, cfg),
                            fm_sources=[], **more)
        with self._lock:
            self.n_reductions += 1
        if ev.image is None:
            raise ValueError(f"annulus {ia + 1}: the reduction returned no image")
        return np.asarray(ev.image, float)

    def run(self, calls: Sequence[Callable[[], Any]], window: int):
        """The reductions in ``calls``, at most ``window`` in flight, in order
        (:meth:`klip_tpe.runner.Runner._run_ordered`).  A failed reduction raises."""
        for i, res in self.r._run_ordered(calls, window):
            if isinstance(res, Exception):
                raise res
            yield i, res

    def prepare(self, window: int) -> None:
        """The base winners' clean images, reduced again on this code, and the stitch weights:
        the base run's (``final_results.json``), else worked out from those images as the run did."""
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
        """The injected stitch of ``images`` (annulus -> image), the base winners' clean images for
        the rest, as the final stitch builds it."""
        from .stitch import stitch_tiles
        imgs = [images.get(a, self.clean[a]) for a in self.order]
        return self.r._maybe_flatten(stitch_tiles(imgs, self.itiles, self.weights, pads=self.pads))

    def measure(self, img: np.ndarray, sources: Sequence[Source]) -> np.ndarray:
        """Each source's S/N as the stitched verify measures it (Path A), its ring without the other
        sources and the known companions."""
        from .verify import verify_injections
        rows, _ = verify_injections(np.asarray(img, float)[None], [s.as_tuple() for s in sources], self.r.fwhm,
                                    self.r.pxscale, known=self.known, excl_fwhm=self.r._excl_fwhm(),
                                    angle_convention=self.r.reducer.angle_convention, combined=img, clean=None,
                                    sidelobes=self.r._verify_rule(), rules=self.r._ring_rules())
        return np.array([np.nan if r.get("snr") is None else float(r["snr"]) for r in rows], float)

    # ---------------------------------------------------------------- draws
    @property
    def clear_fwhm(self) -> float:
        """The run's sampler's distance between sources (FWHM), the draws' clearance by default."""
        return float(getattr(self.r.sampler, "min_mutual_fwhm", 2.0))

    @property
    def fwhm_px(self) -> float:
        return float(getattr(self.r.sampler, "fwhm_as", self.r.fwhm * self.r.pxscale)) / float(self.r.pxscale)

    def draw(self, tiles: Sequence[int], contrasts: Dict[int, float], rng: np.random.Generator,
             counts: Optional[Dict[int, int]] = None, clear_fwhm: Optional[float] = None, restarts: int = 20
             ) -> List[Tuple[int, Source]]:
        """One draw: ``counts[j]`` sources in each of ``tiles`` (annuli; default the run's number),
        placed by the run's sampler in the tile's radii, each at its tile's contrast, at least
        ``clear_fwhm`` FWHM (default the sampler's distance) from the sources of the other tiles.

        The tile with the most sources is placed first: a pair 180 deg apart, placed after the
        other tiles' sources, has to clear all of them at once.  A tile that finds no place in 300
        tries starts the draw again, ``restarts`` times, before :class:`PlacementError`.  The
        sources come back in the order of ``tiles``."""
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
                    cfg_k=None) -> List[np.ndarray]:
        """Each draw's sources measured on the stitch with every annulus they reach reduced with
        them: annulus ``k`` under ``cfg_k`` (default its base winner), the others under their base
        winners."""
        cfg_k = self.base_cfg[k] if cfg_k is None else cfg_k
        jobs, slots = [], []
        for t, d in enumerate(draws):
            src = [s for _, s in d]
            for a in self.affected(src, k):
                cfg = cfg_k if a == k else self.base_cfg[a]
                jobs.append(lambda a=a, cfg=cfg, src=src, t=t: self.reduce(a, cfg, src, f"rescore_a{a + 1}_d{t}"))
                slots.append((t, a))
        imgs: Dict[int, Dict[int, np.ndarray]] = {}
        for i, img in self.run(jobs, window):
            t, a = slots[i]
            imgs.setdefault(t, {})[a] = img
        return [self.measure(self.stitch(imgs[t]), [s for _, s in d]) for t, d in enumerate(draws)]

    def calibrate(self, j: int, rng: np.random.Generator, n_cal: int = 8, target: Tuple[float, float] = TARGET,
                  max_iter: int = 4, window: int = 2, n_src: Optional[int] = None) -> Dict[str, Any]:
        """The contrast of tile ``j``'s sources at which the base winners read a median S/N inside
        ``target`` on the stitch, on ``n_cal`` draws of tile ``j`` alone, the same positions at
        every step: from the base run's contrast for annulus ``j``, each step as
        :meth:`_next_contrast` takes it.  ``n_src``: the sources per draw (default the run's)."""
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
        """The next contrast to try.  Once one step read below ``target`` and one above, the
        contrast between the nearest two, interpolated in log contrast against their S/N.  Before,
        the last contrast scaled by the target's middle over its S/N (times 3 when it read
        nothing)."""
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
    pa, pb = np.deg2rad(a.theta + 90.0), np.deg2rad(b.theta + 90.0)
    return float(np.hypot(a.rho * np.cos(pa) - b.rho * np.cos(pb), a.rho * np.sin(pa) - b.rho * np.sin(pb)))


def _draw_json(d: Sequence[Tuple[int, Source]]) -> List[List[float]]:
    return [[int(j), float(s.rho), float(s.theta), float(s.contrast)] for j, s in d]


def _draw_from_json(d) -> List[Tuple[int, Source]]:
    return [(int(v[0]), Source(float(v[1]), float(v[2]), float(v[3]))) for v in d]


def _atomic_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def _neighbors(scorer: StitchScorer, k: int) -> List[int]:
    p = scorer.pos(k)
    return [scorer.order[q] for q in (p - 1, p, p + 1) if 0 <= q < len(scorer.order)]


def _placements(scorer: StitchScorer, k: int, sources_per_tile: Optional[int] = None,
                clear_fwhm: Optional[float] = None) -> List[Dict[str, Any]]:
    """The rules annulus ``k``'s draws are placed by, in the order they are tried: the sources in
    each tile (``counts``) and their clearance (``clear_fwhm``).  Asked for, ``sources_per_tile``
    (at most that many in each tile) and ``clear_fwhm`` are the one rule.  Otherwise the run's
    numbers at the sampler's distance, then at :data:`RELAXED_FWHM`, then one source in each
    neighbor's tile, annulus ``k``'s own keeping the run's number."""
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
    """``1 source in tile 1, 1 in tile 2 and 2 in tile 3, 2 FWHM (12.8 px) apart``"""
    items = sorted((int(j), int(n)) for j, n in rule["counts"].items())
    parts = [f"{n} in tile {j + 1}" for j, n in items]
    parts[0] = parts[0].replace(" in ", " source in " if items[0][1] == 1 else " sources in ", 1)
    txt = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    cf = float(rule["clear_fwhm"])
    return f"{txt}, {cf:g} FWHM ({cf * fwhm_px:.1f} px) apart"


def _rule_json(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {"counts": {str(int(j) + 1): int(n) for j, n in rule["counts"].items()},
            "clear_fwhm": float(rule["clear_fwhm"]), "rule": rule["rule"]}


def _rule_from_json(d: Dict[str, Any]) -> Dict[str, Any]:
    return {"counts": {int(j) - 1: int(n) for j, n in d["counts"].items()}, "clear_fwhm": float(d["clear_fwhm"]),
            "rule": d.get("rule", "")}


def _annulus_draws(scorer: StitchScorer, k: int, contrasts: Dict[int, float], n_draws: int, seed: int, sd: str,
                   rules: Sequence[Dict[str, Any]], log: Callable[[str], None]
                   ) -> Tuple[List[List[Tuple[int, Source]]], Dict[str, Any]]:
    """Annulus ``k``'s draws and the rule they were placed by: the ones kept in ``sd`` from an
    earlier start, else placed by the first of ``rules`` that places every draw, each rule with
    its own random stream (the first the stream of the code before the rules).  Raises
    :class:`RescoreStop` when the kept draws do not fit the options or no rule places them."""
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


def rescore_plan(scorer: StitchScorer, entrants: Dict[int, List[Entrant]], n_draws: int, n_cal: int = 8,
                 max_iter: int = 4) -> Dict[str, Any]:
    """How many reductions a re-score makes, at most: the base winners' clean images, the
    calibration of each tile and, per annulus, the neighbors' and the entrants' reductions of
    its draws.  No reduction is made."""
    R = 2.5 * float(scorer.r.fwhm)
    tiles_needed = sorted({j for k in entrants for j in _neighbors(scorer, k)})

    def n_aff(tiles: Sequence[int], k: int) -> int:
        lo = min(scorer.tile(j)[0] for j in tiles) - R
        hi = max(scorer.tile(j)[1] for j in tiles) + R
        return sum(1 for a in scorer.order if a == k or (scorer.reach(a)[0] <= hi and scorer.reach(a)[1] >= lo))

    cal = {j: max_iter * n_cal * n_aff([j], j) for j in tiles_needed}
    per = {}
    for k, ents in entrants.items():
        na = n_aff(_neighbors(scorer, k), k)
        per[k] = {"entrants": len(ents), "affected": na, "neighbors": n_draws * (na - 1), "entrant": len(ents) * n_draws}
    total = len(scorer.order) + sum(cal.values()) + sum(v["neighbors"] + v["entrant"] for v in per.values())
    return {"base_clean": len(scorer.order), "calibration": cal, "annuli": per, "total": total}


def _base_reduce_seconds(run_dir: str) -> Optional[float]:
    """The median time the base run's search took to reduce a trial (``reduce_s``)."""
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


def rescore(runner, entrant_runs: Sequence[str], out_dir: str, *, annuli: Sequence[int], n_draws: int = 50,
            top: int = 0, seed: int = 1, target: Tuple[float, float] = TARGET, n_cal: int = 8, max_iter: int = 4,
            window: Optional[int] = None, overlap_pads: Optional[bool] = None, plan: bool = False,
            card: bool = False, sources_per_tile: Optional[int] = None, clear_fwhm: Optional[float] = None,
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Score the candidates of ``annuli`` (0-based) on the base run's stitch, the other annuli at
    the base run's winners.  ``runner`` is the base run resumed read-only, ``entrant_runs`` the
    other runs of the same setup whose candidates enter too (the base run's always do).

    Writes ``rescore.json`` and ``rescore.txt`` to ``out_dir`` after each annulus, and keeps its
    draws, its calibration and each entrant's S/N in ``out_dir/state``, from which a re-score
    started again on the same directory resumes.  ``plan``: count the reductions and return,
    making none.  ``card``: the injected reductions may go to the card workers.

    ``sources_per_tile`` (at most that many sources in each tile of a draw) and ``clear_fwhm``
    (the distance between the tiles' sources, FWHM) set the rule the draws are placed by.  Without
    them an annulus whose draws find no place under the run's rule falls back as
    :func:`_placements` lists.  An annulus whose draws find no place under any rule stops with a
    message, and the re-score goes on with the next."""
    if sources_per_tile is not None and int(sources_per_tile) < 1:
        raise ValueError("sources_per_tile must be at least 1")
    if clear_fwhm is not None and not float(clear_fwhm) > 0:
        raise ValueError("clear_fwhm must be positive")
    t0 = time.time()
    base_dir = os.path.abspath(runner.run_dir)
    runs = [base_dir] + [os.path.abspath(p) for p in entrant_runs if os.path.abspath(p) != base_dir]
    out_dir = os.path.abspath(out_dir)
    for rd in runs:
        if out_dir == rd or out_dir.startswith(rd + os.sep):
            raise ValueError(f"{out_dir} is inside {rd}: the re-score writes beside the runs")
    scorer = StitchScorer(runner, overlap_pads=overlap_pads, log=log, card=card)
    from .runner import _core_edits, _describe_edits
    frames = _describe_edits(_core_edits(dict(getattr(runner.reducer, "data_edits", None) or {})))
    run_frames = _describe_edits(_core_edits(runner.cfg.data_edits))
    if frames != run_frames:
        raise ValueError(f"the reducer reduces {frames}, the base run was made on {run_frames}: the re-score "
                         f"scores on the runs' own frames")
    ref = _setup(base_dir)
    for rd in runs[1:]:
        s = _setup(rd)
        name = os.path.basename(rd)
        if s["zones"] != ref["zones"]:
            raise ValueError(f"{name}'s annuli {s['zones']} are not the base run's {ref['zones']}")
        if s["names"] != ref["names"]:
            raise ValueError(f"{name}'s search space is not the base run's")
        if s["frames"] != ref["frames"]:
            raise ValueError(f"{name} was made on other frames than the base run ({s['frames']} against "
                             f"{ref['frames']})")
    ents: Dict[int, List[Entrant]] = {}
    for k in annuli:
        lst: List[Entrant] = []
        seen = set()
        for rd in runs:
            for e in run_entrants(rd, k, top=top):
                if e.key not in seen:
                    seen.add(e.key)
                    lst.append(e)
        ents[k] = lst
    base_key = {k: config_key(scorer.results[k].winner_config) for k in annuli}
    pl = rescore_plan(scorer, ents, n_draws, n_cal, max_iter)
    sec = _base_reduce_seconds(base_dir)
    log(f"re-score of {os.path.basename(base_dir)}: annuli {', '.join(str(k + 1) for k in annuli)}, candidates from "
        f"{len(runs)} run(s), {n_draws} common draws per annulus, at most {pl['total']} reductions")
    for k in annuli:
        p = pl["annuli"][k]
        log(f"  annulus {k + 1}: {p['entrants']} candidates x {n_draws} draws, {p['affected']} annuli reached by a "
            f"draw: {p['entrant'] + p['neighbors']} reductions")
    if sec is not None:
        per_trial = 1 + int(getattr(runner.cfg, "n_remeasure", 1) or 1)
        inflight = max(int(getattr(runner.cfg, "batch", 1) or 1), 1)
        hours = pl["total"] * sec / (per_trial * inflight) / 3600.0
        log(f"  the base run's search took a median {sec:.1f} s to reduce a trial of {per_trial} reductions, "
            f"{inflight} trial(s) in flight: at that pace about {hours:.1f} h")
    if plan:
        return {"plan": pl, "reduce_s": sec, "entrants": {k + 1: [asdict(e) for e in v] for k, v in ents.items()}}
    sd = os.path.join(out_dir, "state")
    os.makedirs(sd, exist_ok=True)
    runner._start_workers()
    w = int(window or runner._reduction_window())
    scorer.prepare(w)
    log(f"  stitch weights from {scorer.weights_from}, pads between overlapping annuli "
        f"{'on' if scorer.overlap_pads else 'off'}, {w} reductions in flight")
    # the contrast of each tile the draws use, on the base winners' stitch
    cpath = os.path.join(sd, "calibration.json")
    cal = {int(k): v for k, v in (_read_json(cpath).items() if os.path.exists(cpath) else [])}
    for j in sorted({j for k in annuli for j in _neighbors(scorer, k)}):
        if j not in cal:
            cal[j] = scorer.calibrate(j, np.random.default_rng([int(seed), 1000 + j]), n_cal=n_cal, target=target,
                                      max_iter=max_iter, window=w,
                                      n_src=None if sources_per_tile is None
                                      else min(int(scorer.r._nsrc(j)), int(sources_per_tile)))
            _atomic_json(cpath, {str(a): v for a, v in cal.items()})
    contrasts = {j: float(v["contrast"]) for j, v in cal.items()}
    report: Dict[str, Any] = {"base": base_dir, "runs": runs, "annuli": {}, "seed": int(seed), "n_draws": int(n_draws),
                              "target": list(target), "calibration": {str(j + 1): v for j, v in cal.items()},
                              "tiles_px": [list(t) for t in scorer.tiles], "weights": [float(v) for v in scorer.weights],
                              "weights_from": scorer.weights_from, "overlap_pads": scorer.overlap_pads}
    stopped: Dict[int, str] = {}
    for k in annuli:
        try:
            draws, rule = _annulus_draws(scorer, k, contrasts, int(n_draws), int(seed), sd,
                                         _placements(scorer, k, sources_per_tile, clear_fwhm), log)
        except RescoreStop as exc:
            stopped[k] = str(exc)
            log(f"annulus {k + 1}: stopped, {exc}")
            report["annuli"][str(k + 1)] = {"tile_px": list(scorer.tile(k)),
                                            "draw_tiles": [j + 1 for j in _neighbors(scorer, k)],
                                            "stopped": str(exc), "position": scorer.pos(k)}
            _write_reports(out_dir, report, log=None)
            continue
        srcs = [[s for _, s in d] for d in draws]
        tiles_of = [[j for j, _ in d] for d in draws]
        # the neighbors' reductions of each draw, shared by every candidate
        nb_jobs, nb_slots = [], []
        for t, src in enumerate(srcs):
            for a in scorer.affected(src, k):
                if a != k:
                    nb_jobs.append(lambda a=a, src=src, t=t: scorer.reduce(a, scorer.base_cfg[a], src,
                                                                           f"rescore_a{a + 1}_d{t}"))
                    nb_slots.append((t, a))
        todo = [e for e in ents[k] if not os.path.exists(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"))]
        nb: Dict[int, Dict[int, np.ndarray]] = {t: {} for t in range(len(draws))}
        if todo:
            log(f"annulus {k + 1}: the neighbors' reductions of {len(draws)} draws ({len(nb_jobs)})")
            for i, img in scorer.run(nb_jobs, w):
                t, a = nb_slots[i]
                nb[t][a] = img
            log(f"annulus {k + 1}: {len(todo)} of {len(ents[k])} candidates to score")
        jobs, slots = [], []
        for e in todo:
            cfg = runner._config_from_dict(e.config)
            for t, src in enumerate(srcs):
                jobs.append(lambda cfg=cfg, src=src, t=t: scorer.reduce(k, cfg, src, f"rescore_a{k + 1}_c_d{t}"))
                slots.append((e, t))
        acc: Dict[str, List[Optional[np.ndarray]]] = {}
        t_ann = time.time()
        n_here = 0
        for i, img in scorer.run(jobs, w):
            e, t = slots[i]
            row = acc.setdefault(e.tag, [None] * len(draws))
            row[t] = scorer.measure(scorer.stitch({**nb[t], k: img}), srcs[t])
            if all(v is not None for v in row):
                _atomic_json(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"),
                             {"entrant": asdict(e), "snr": [[float(x) for x in v] for v in row]})
                done = sum(1 for x in ents[k] if os.path.exists(os.path.join(sd, f"ann{k + 1:02d}_{x.tag}.json")))
                el = time.time() - t_ann
                n_here += 1
                left = (len(todo) - n_here) * el / n_here
                log(f"  annulus {k + 1}: {e.run} {e.kind} eval {e.eval_index}: median S/N "
                    f"{_nanmedian([_nanmedian(v) for v in row]):.2f} ({done}/{len(ents[k])}, {el / 60:.1f} min, "
                    f"about {left / 60:.0f} min left in this annulus)")
                del acc[e.tag]
        report["annuli"][str(k + 1)] = _annulus_report(scorer, k, ents[k], draws, tiles_of, sd, base_key[k])
        report["annuli"][str(k + 1)]["placement"] = dict(_rule_json(rule), text=_rule_text(rule, scorer.fwhm_px))
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
    """Each entrant's statistics over the common draws: the median over each draw's sources of
    their S/N, for all the draw's sources and for tile ``k``'s alone, and the paired differences
    from the base winner."""
    rows = []
    for e in ents:
        rec = _read_json(os.path.join(sd, f"ann{k + 1:02d}_{e.tag}.json"))
        snr = [np.asarray(v, float) for v in rec["snr"]]
        per_all = np.array([_nanmedian(v) for v in snr])
        per_k = np.array([_nanmedian(v[np.asarray(tj) == k]) for v, tj in zip(snr, tiles_of)])
        rows.append({"entrant": asdict(e), "base": e.key == base_key, "snr": [[float(x) for x in v] for v in snr],
                     "_all": per_all, "_k": per_k})
    base = next((r for r in rows if r["base"]), None)
    for r in rows:
        r["all"] = paired_summary(r["_all"], base["_all"] if base else np.full(len(draws), np.nan))
        r["tile"] = paired_summary(r["_k"], base["_k"] if base else np.full(len(draws), np.nan))
    rows.sort(key=lambda r: (-(r["all"]["median"] if np.isfinite(r["all"]["median"]) else -np.inf),
                             -(r["all"]["diff_mean"] if np.isfinite(r["all"]["diff_mean"]) else -np.inf)))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r.pop("_all")
        r.pop("_k")
    p = scorer.pos(k)
    return {"tile_px": list(scorer.tile(k)), "draw_tiles": [j + 1 for j in _neighbors(scorer, k)],
            "draws": [_draw_json(d) for d in draws], "entrants": rows,
            "base_rank": next((r["rank"] for r in rows if r["base"]), None), "position": p}


def _f(v, fmt="{:.2f}") -> str:
    return fmt.format(v) if v is not None and np.isfinite(v) else "-"


def _write_reports(out_dir: str, report: Dict[str, Any], log: Optional[Callable[[str], None]]) -> None:
    _atomic_json(os.path.join(out_dir, "rescore.json"), report)
    L = [f"# klip_tpe re-score -- {_dt.datetime.now().isoformat(timespec='seconds')}",
         f"# base run: {report['base']}",
         f"# candidates from: {', '.join(os.path.basename(r) for r in report['runs'])}",
         f"# {report['n_draws']} common draws per annulus, sources in the annulus' tile and its neighbors', each "
         f"tile's at the contrast that gives the base winners a median S/N of {report['target'][0]:g}-"
         f"{report['target'][1]:g} on the stitch",
         "# each source measured on the injected stitch as the stitched verify measures it (Path A), the other "
         "annuli at the base run's winners",
         "# per draw: the median S/N of the draw's sources (all) and of the annulus' own tile's (tile).  diff: the "
         "paired difference from the base winner (*) over the draws, mean +/- standard error, and P: the share of "
         "draws where the candidate reads higher", "#"]
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
            dif = "0" if r["base"] else f"{_f(s['diff_mean'], '{:+.2f}')} +/- {_f(s['diff_se'])}"
            dift = "0" if r["base"] else f"{_f(st['diff_mean'], '{:+.2f}')} +/- {_f(st['diff_se'])}"
            ev = "-" if e.get("eval_index") is None else str(int(e["eval_index"]) + 1)
            L.append(f"  {r['rank']:>3}{star}  {e['run']:<24} {e['kind']:<10} {ev:>6}  {rng:>20}  {dif:>16}  "
                     f"{_f(s['p_better'], '{:.2f}') if not r['base'] else '-':>4}  {_f(st['median']):>8}  {dift:>16}")
    if "reductions" in report:
        L += ["", f"# {report['reductions']} reductions in {report['wall_s'] / 3600:.2f} h"]
    with open(os.path.join(out_dir, "rescore.txt") + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(os.path.join(out_dir, "rescore.txt") + ".tmp", os.path.join(out_dir, "rescore.txt"))
    if log is not None:
        log(f"  wrote {os.path.join(out_dir, 'rescore.txt')} and rescore.json")
