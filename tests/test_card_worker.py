"""The card workers in the pool (PartitionedReducer.card_share, parallel.CardSpec).

A share of the search's injected draws goes to card worker processes with their own queue,
picked by a hash of the job's tag, so a run and its resumes send the same jobs there.  Each trial
sends the card the lowest-hash share of its jobs (the "trial" rule, since 2026-11-15), or each job
goes by its own slot (the "job" rule, which a run started before keeps).  Here a stand-in card
runs the CPU reducer in those processes: its images are the CPU's to the bit, so a run with it
must give the CPU-only run's scores, and every draw's partitions on the card must be the ones the
rule picked.  When the card cannot start, fails jobs, or a worker dies, its jobs go back to the
CPU workers and the run goes on without it.  The card the workers use (``--gpu-device``) and the
rule stay with the run in ``card.json``, and each annulus' line gives the card's busy fraction.
"""
from __future__ import annotations

import json
import math
import os
import types

import numpy as np
import pytest

from klip_tpe import RunConfig, Runner, ValidationConfig, cli
from klip_tpe.parallel import card_picks, card_slot, describe_card
from klip_tpe.reducer import PartitionedReducer

from conftest import build_synthetic_run

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="the card workers are forked")

SHARE = 50.0


class StandIn:
    """A card reducer that runs the CPU reducer (the same bits)."""

    fail_warm = False
    fail_every_job = False
    die_on_job = 0

    def __init__(self, r):
        self.r, self.n = r, 0

    @classmethod
    def from_cpu(cls, r):
        return cls(r)

    def warm(self):
        if self.fail_warm:
            raise RuntimeError("no card here")
        return {"name": "stand-in card", "driver": "0.0"}

    def supports(self, req):
        return req.stage == "reduce" and not req.fm_sources and not req.k_scan and not req.extras

    def reduce(self, req):
        self.n += 1
        if self.die_on_job and self.n >= self.die_on_job:
            os._exit(3)
        if self.fail_every_job:
            raise RuntimeError("card job failed")
        return self.r.reduce(req)


class NoWarm(StandIn):
    fail_warm = True


class Broken(StandIn):
    fail_every_job = True


class Dies(StandIn):
    die_on_job = 2


class OutOfMemoryError(Exception):
    """CuPy's name for it, which the worker recognizes without importing CuPy."""


class Seen(StandIn):
    """A stand-in card that reports the card its worker was given."""

    def warm(self):
        return {"name": "stand-in card", "driver": "0.0", "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}


class Hungry(StandIn):
    """A card whose workers are each held to a share of its memory, and on which the jobs in
    the lowest fifth of the card's slots need more than that share."""

    def limit_memory(self, workers, fraction=0.9):
        return int(fraction * 96e9 / workers)

    def after_job(self):
        pass

    def reduce(self, req):
        if card_slot(req.tag) < SHARE * 0.4:
            raise OutOfMemoryError("Out of memory allocating 22,155,776,512 bytes (allocated so far: 17 GB)")
        return self.r.reduce(req)


def _run(tmp_path, name, factory=None, share=SHARE, rule="trial"):
    red, space, obj, samp, _ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    if factory is not None:
        pr.card_share, pr.card_workers, pr.card_factory, pr.card_rule = share, 2, factory, rule
    cfg = RunConfig(ann_edges=[8, 30], n_iter=[10], n_init=4, seed=11, defaults={"k_klip": 5},
                    n_remeasure=2, batch=2, validation=ValidationConfig(n_top=1, n_valid=1))
    logs = []
    run_dir = str(tmp_path / name)
    try:
        Runner(pr, space, obj, samp, cfg, run_dir, log=logs.append).run()
    finally:
        pr.close()
    recs = [json.loads(ln) for ln in open(os.path.join(run_dir, "results.jsonl")) if ln.strip()]
    return recs, logs, run_dir


def _scores(recs):
    return [(r["annulus"], r["index"], r["score"], r["raw_score"]) for r in recs]


def _trial_jobs(r):
    """A record's draws' jobs, ``{(draw, partition): tag}``, as the runner tagged them."""
    draws = r["meta"].get("draw_card") or [[]] * int(r["meta"].get("draw_n") or 1)
    ev = r["phase"] if r["phase"] in ("seed", "reseed") else f"e{r['index'] + 1}"
    return draws, {(j, pid): f"a{r['annulus'] + 1}_{ev}_d{j + 1}_inj_{pid}"
                   for j in range(len(draws)) for pid in r["meta"]["selected"]}


def _picked(tags, rule):
    return card_picks(tags, SHARE) if rule == "trial" else {t for t in tags if card_slot(t) < SHARE}


@pytest.fixture(scope="module")
def cpu_only(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("card"), "cpu")


@pytest.mark.slow
@pytest.mark.parametrize("rule", ["trial", "job"])
def test_the_card_takes_the_share_its_rule_picks_and_changes_no_score(tmp_path, cpu_only, rule):
    recs, logs, run_dir = _run(tmp_path, "card", StandIn.from_cpu, rule=rule)
    assert _scores(recs) == _scores(cpu_only[0]), "the stand-in card gives the CPU's bits"
    on, off = 0, 0
    for r in recs:
        draws, jobs = _trial_jobs(r)
        picked = _picked(list(jobs.values()), rule)
        for (j, pid), tag in jobs.items():
            want = tag in picked
            assert (pid in draws[j]) == want, (tag, draws[j])
            on, off = on + want, off + (not want)
        if rule == "trial" and jobs:            # no trial sends the card more than its share, rounded up
            assert len(picked) <= math.ceil(SHARE * len(jobs) / 100.0)
    assert on and off
    assert any("card: 2 workers on stand-in card (driver 0.0), 50 % of the search's injected draws" in ln
               for ln in logs), logs
    setup = json.load(open(os.path.join(run_dir, "run_setup.json")))
    assert setup["reducer"]["card"] == {"share": SHARE, "workers": 2, "rule": rule}
    card = json.load(open(os.path.join(run_dir, "final_results.json")))["card"]
    assert card["up"] and card["jobs"] == on and 0 < card["busy_fraction"] < 1 and card["rule"] == rule
    assert card["marked"] == on and card["refused"] == {}
    # the annulus' lines: the search's marked jobs, all reduced on the card, and how busy it was
    assert any(ln == f"  annulus 1: the card reduced {on} of the {on} jobs marked for it" for ln in logs), \
        [ln for ln in logs if "the card reduced" in ln]
    assert any(ln.startswith("  annulus 1: the card's 2 workers were busy ") and "% of the search's" in ln
               for ln in logs), [ln for ln in logs if "annulus 1: the card" in ln]
    (ann,) = card["annuli"]
    assert ann["annulus"] == 1 and ann["jobs"] == ann["marked"] == on and 0 < ann["busy_fraction"] < 1.2
    assert ann["search_s"] > 0 and not ann["since_resume"]
    done = json.load(open(os.path.join(run_dir, "annulus01", "annulus_done.json")))
    assert done["card"] == ann, "a resumed run reads the finished annuli' numbers from here"
    summary = open(os.path.join(run_dir, "bench_summary.txt")).read().splitlines()
    assert any(ln.startswith("# card: share 50 %") for ln in summary)
    assert any(ln.startswith(f"# card annulus 1: {on} jobs of {on} marked, busy ") for ln in summary), summary


@pytest.mark.slow
@pytest.mark.parametrize("kind,why", [(NoWarm, "it did not start (RuntimeError: no card here)"),
                                      (Broken, "3 jobs failed on it"),
                                      (Dies, "died")])
def test_without_the_card_the_run_goes_on_on_the_cpu(tmp_path, cpu_only, kind, why):
    recs, logs, run_dir = _run(tmp_path, kind.__name__, kind.from_cpu)
    assert _scores(recs) == _scores(cpu_only[0])
    assert any(ln.startswith("  card:") and why in ln and "goes on without it" in ln for ln in logs), logs
    card = json.load(open(os.path.join(run_dir, "final_results.json")))["card"]
    assert not card["up"] and why in card["why_down"]
    if kind is NoWarm:
        assert not any(r["meta"].get("card") or r["meta"].get("draw_card") for r in recs)


@pytest.mark.slow
def test_a_job_past_a_workers_memory_goes_to_the_cpu_and_the_card_stays(tmp_path, cpu_only):
    """NEAR2's second s6 test run lost the card after three jobs ran out of memory in annulus 2.
    A job past its worker's share now goes to the CPU workers, named in the log, and the card
    stays in the run: the same jobs come back each time, and the scores are the CPU's."""
    recs, logs, run_dir = _run(tmp_path, "hungry", Hungry.from_cpu)
    assert _scores(recs) == _scores(cpu_only[0])
    assert any("each worker held to 43.2 GB of its memory" in ln for ln in logs), logs
    named = [ln for ln in logs if "needed more than its worker's share of the card's memory" in ln]
    assert named and all("_inj_" in ln and "OutOfMemoryError" in ln for ln in named), named
    assert not any("goes on without it" in ln for ln in logs)
    card = json.load(open(os.path.join(run_dir, "final_results.json")))["card"]
    assert card["up"] and card["memory"] > 0 and card["jobs"] > 0
    assert any("needed more than a worker's share of its memory" in ln and "the card reduced" in ln for ln in logs), \
        [ln for ln in logs if "the card reduced" in ln]
    for r in recs:                     # every job past the share was reduced on the CPU
        draws, jobs = _trial_jobs(r)
        picked = _picked(list(jobs.values()), "trial")
        for (j, pid), tag in jobs.items():
            assert (pid in draws[j]) == (tag in picked and card_slot(tag) >= SHARE * 0.4), tag


def test_the_card_reducer_runs_in_the_card_workers():
    """klip_tpe.gpu_reducer's reducer on its NumPy backend, in the card workers: built in the
    parent, warmed after the fork, every partition job of a marked draw on the card, and its
    images the CPU's to 1e-3 of the noise with the same NaN footprint."""
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    from klip_tpe.injection import GaussianPSF
    from klip_tpe.metrics import Source
    from klip_tpe.reducer import Dataset, KLIPReducer
    from klip_tpe.space import Config

    def numpy_card(r):
        g = GPUKLIPReducer.from_cpu(r)
        g.backend_name = "numpy"
        return g

    def night(seed, n=40, ny=64):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:ny, 0:ny]
        rr = np.hypot(xx - (ny - 1) / 2.0, yy - (ny - 1) / 2.0)
        halo = (3000.0 * np.exp(-rr / 6.0) + 40.0 * np.exp(-rr / 25.0)).astype(np.float32)
        speck = rng.normal(size=(ny, ny)).astype(np.float32) * 6.0
        cube = (halo[None] + speck[None] * np.linspace(0.8, 1.2, n)[:, None, None]
                + rng.normal(size=(n, ny, ny)).astype(np.float32) * 2.0)
        model = GaussianPSF(4.0, star_flux=2.0e5)
        return KLIPReducer(Dataset(cube, np.linspace(-60.0, 70.0, n), texp=float(n)), pxscale=0.05, lam_m=4e-6,
                           diam_m=8.0, injection_model=model, fallback_model=model, fwhm_px=4.0, outrad_cap=28.0,
                           defaults={"fast": False, "n_min_ref": 5, "comb_type": "nwadi"})

    def parts():
        return {f"n{i}": night(i) for i in (1, 2)}
    params = {"inrad": 6.0, "outrad": 18.0, "k_klip": 5, "bin": 2}
    cfg = Config(params=dict(params), per_partition={p: dict(params) for p in ("n1", "n2")},
                 selected=["n1", "n2"], x=np.zeros(0))
    srcs = [Source(0.6, 30.0, 0.05)]
    cpu = PartitionedReducer(parts(), max_workers=1).reduce_config(cfg, srcs, tag="a1_e1_d1_inj")
    pr = PartitionedReducer(parts(), max_workers=4, pool="processes")
    pr.card_share, pr.card_workers, pr.card_factory = 100.0, 1, numpy_card
    try:
        ev = pr.reduce_config(cfg, srcs, tag="a1_e1_d1_inj", card=True)
        clean = pr.reduce_config(cfg, None, tag="a1_e1_d1_clean")          # not marked: the CPU's
        summary = pr.card_summary()
    finally:
        pr.close()
    for pid in ("n1", "n2"):
        assert ev.meta[pid]["card"] is True and ev.meta[pid]["device"] == "gpu"
        assert not clean.meta[pid].get("card") and "device" not in clean.meta[pid]
    a, b = ev.stack, cpu.stack
    assert np.array_equal(np.isnan(a), np.isnan(b))
    noise = np.nanstd(b, axis=(1, 2))
    assert np.all(np.nanmax(np.abs(a - b), axis=(1, 2)) < 1e-3 * noise)
    assert summary["up"] and summary["jobs"] == 2 and summary["info"]["name"] == "numpy"


def test_no_share_no_card_workers(tmp_path):
    red, *_ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    pr.card_share, pr.card_factory = 0.0, StandIn.from_cpu
    try:
        assert pr.start_workers() == "processes"
        assert pr._procs.card is None and pr._procs.card_procs == [] and pr.card_summary() is None
    finally:
        pr.close()


def test_the_slot_is_the_tag_s_and_spreads():
    assert card_slot("a1_e12_d3_inj_n1") == card_slot("a1_e12_d3_inj_n1")
    slots = [card_slot(f"a1_e{i}_d1_inj_n2") for i in range(1, 2001)]
    assert min(slots) >= 0 and max(slots) <= 99
    assert abs(np.mean(np.array(slots) < 30) - 0.30) < 0.04


def test_a_run_keeps_its_share(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_auto_card_share", lambda: 30.0)
    run = tmp_path / "run_x"
    run.mkdir()
    base = ["resume", "--root", str(tmp_path), "--run-dir", str(run)]

    def settle(extra):
        a = cli._build_parser().parse_args(base + extra)
        a._argv = base + extra
        cli._card_for_run(a)
        return a.gpu_share, a.gpu_workers

    rec = run / "card.json"
    assert settle([]) == (30.0, 3) and rec.exists(), "auto, settled and recorded at the start"
    monkeypatch.setattr(cli, "_auto_card_share", lambda: 0.0)
    assert settle([]) == (30.0, 3), "a resume keeps the share, wherever it runs"
    assert settle(["--gpu-share", "40", "--gpu-workers", "4"]) == (40.0, 4)
    assert settle([]) == (40.0, 4)
    d = json.loads(rec.read_text())
    assert [h["gpu_share"] for h in d["history"]] == [30.0, 40.0]
    with pytest.raises(SystemExit):
        settle(["--gpu-share", "140"])
    # no card on this machine and none asked: nothing recorded
    run2 = tmp_path / "run_y"
    run2.mkdir()
    a = cli._build_parser().parse_args(["resume", "--root", str(tmp_path), "--run-dir", str(run2)])
    a._argv = ["resume", "--root", str(tmp_path), "--run-dir", str(run2)]
    cli._card_for_run(a)
    assert a.gpu_share == 0.0 and not (run2 / "card.json").exists()
    # the settled values reach the reducer; a command that did not settle them gets no card
    red, *_ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers)
    a.gpu_share, a.gpu_workers = 25.0, 2
    cli._set_card(pr, a)
    assert (pr.card_share, pr.card_workers) == (25.0, 2)
    b = cli._build_parser().parse_args(["resume", "--root", str(tmp_path), "--run-dir", str(run2)])
    cli._set_card(pr, b)
    assert pr.card_share == 0.0


def test_a_trial_sends_the_card_its_share_lowest_hash_first():
    """The trial rule: the card's long-run share is the share asked, no trial sends it more than
    the share of its jobs rounded up, and the picks depend on the tags alone."""
    total, jobs, most = 0, 0, 0
    for e in range(1, 3001):
        tags = [f"a1_e{e}_d{j}_inj_n{p}" for j in (1, 2, 3) for p in range(1, 7)]
        got = card_picks(tags, 12.0)
        assert got == card_picks(list(reversed(tags)), 12.0) and got <= set(tags)
        ranked = sorted(tags, key=lambda t: (int.from_bytes(__import__("hashlib").sha1(t.encode()).digest()[:4],
                                                            "big"), t))
        assert got == set(ranked[:len(got)]), "the lowest hashes"
        total, jobs, most = total + len(got), jobs + len(tags), max(most, len(got))
    assert abs(total / jobs - 0.12) < 0.005 and most == math.ceil(0.12 * 18)
    # a trial of few jobs still sends the card its share over many trials
    few = sum(len(card_picks([f"a2_e{e}_d{j}_inj_n1" for j in (1, 2, 3)], 12.0)) for e in range(1, 6001))
    assert abs(few / 18000 - 0.12) < 0.01
    assert len(card_picks(["a", "b", "c"], 100.0)) == 3 and card_picks(["a", "b"], 0.0) == set()
    assert card_picks([], 50.0) == set()


def test_the_card_workers_use_the_card_named(monkeypatch):
    """``card_device`` reaches the card workers' CUDA_VISIBLE_DEVICES and no one else's, and the
    pool reports the card's warm-up to ``card_on_ready`` once, its line going to the log."""
    from klip_tpe.metrics import Source
    from klip_tpe.space import Config
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    red, *_ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    seen = []
    pr.card_share, pr.card_workers, pr.card_factory = 100.0, 1, Seen.from_cpu
    pr.card_device, pr.card_on_ready = "GPU-1234", (lambda info: seen.append(info) or "  card: seen it")
    params = {"inrad": 8.0, "outrad": 30.0, "k_klip": 5, "bin": 12}
    parts = list(red.reducers)
    cfg = Config(params=dict(params), per_partition={p: dict(params) for p in parts}, selected=parts,
                 x=np.zeros(0))
    try:
        ev = pr.reduce_config(cfg, [Source(0.6, 30.0, 1e-3)], tag="a1_e1_d1_inj", card=True,
                              card_group=["a1_e1_d1_inj"])
        summary, notes = pr.card_summary(), pr.take_card_notes()
    finally:
        pr.close()
    assert all(ev.meta[str(p)].get("card") for p in parts), "share 100: every job on the card"
    assert summary["info"]["cvd"] == "GPU-1234" and summary["device"] == "GPU-1234"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0", "the parent's environment is left alone"
    assert len(seen) == 1 and seen[0]["name"] == "stand-in card" and "  card: seen it" in notes
    assert "on stand-in card (CUDA_VISIBLE_DEVICES=GPU-1234) (driver 0.0)" in describe_card(summary)


def test_a_run_keeps_its_rule_and_its_card(tmp_path, monkeypatch):
    """A run started before the trial rule keeps picking job by job until a resume names
    ``--gpu-share``.  The card ``CUDA_VISIBLE_DEVICES`` named is recorded and kept at a resume
    without it, and over another one in the environment, until ``--gpu-device`` names another."""
    monkeypatch.setattr(cli, "_auto_card_share", lambda: 12.0)
    run = tmp_path / "run_old"
    run.mkdir()
    rec = run / "card.json"
    rec.write_text(json.dumps({"gpu_share": 30.0, "gpu_workers": 3, "history": []}))
    base = ["resume", "--root", str(tmp_path), "--run-dir", str(run)]

    def settle(extra, env=None, b=base):
        if env is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
        a = cli._build_parser().parse_args(b + extra)
        a._argv = b + extra
        cli._card_for_run(a)
        return a

    a = settle([])
    assert (a.gpu_share, a.gpu_rule, a.gpu_device) == (30.0, "job", None)
    assert "rule" not in json.loads(rec.read_text()), "nothing changed, nothing written"
    a = settle([], env="1")
    assert (a.gpu_device, a.gpu_rule) == ("1", "job")
    d = json.loads(rec.read_text())
    assert d["gpu_device"] == "1" and d["rule"] == "job"
    assert settle([]).gpu_device == "1", "a resume without CUDA_VISIBLE_DEVICES keeps the card"
    assert settle([], env="0").gpu_device == "1", "the run's card over the environment's"
    a = settle(["--gpu-share", "30"])
    assert (a.gpu_share, a.gpu_rule, a.gpu_device) == (30.0, "trial", "1"), "naming the share takes the trial rule"
    a = settle(["--gpu-share", "12"])
    assert (a.gpu_share, a.gpu_rule, a.gpu_device) == (12.0, "trial", "1")
    a = settle(["--gpu-device", "GPU-abc"])
    assert a.gpu_device == "GPU-abc" and a.gpu_moved and json.loads(rec.read_text())["gpu_device"] == "GPU-abc"
    with pytest.raises(SystemExit):
        settle(["--gpu-device", "0,1"])
    hist = json.loads(rec.read_text())["history"]
    assert [h["gpu_device"] for h in hist] == ["1", "1", "1", "GPU-abc"] and hist[-1]["previous"]["gpu_device"] == "1"
    assert [h["rule"] for h in hist] == ["job", "trial", "trial", "trial"]
    # a new run takes the trial rule
    run2 = tmp_path / "run_new"
    run2.mkdir()
    b2 = ["resume", "--root", str(tmp_path), "--run-dir", str(run2)]
    a2 = settle([], b=b2)
    assert (a2.gpu_share, a2.gpu_rule) == (12.0, "trial")
    assert json.loads((run2 / "card.json").read_text())["rule"] == "trial"
    # the settled values reach the reducer, which records the card's name as it comes up
    red, *_ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers)
    cli._set_card(pr, a)
    assert (pr.card_share, pr.card_rule, pr.card_device) == (12.0, "trial", "GPU-abc")
    assert callable(pr.card_on_ready)
    assert pr.card_on_ready({"name": "Card A"}) is None, "the first card is recorded quietly"
    assert json.loads(rec.read_text())["gpu_name"] == "Card A"
    p = str(rec)
    line = cli._card_came_up(p, False, {"name": "Card B"})
    assert "this is Card B, but the run has been reducing on Card A" in line
    assert json.loads(rec.read_text())["gpu_name"] == "Card A", "a warning, not a new record"
    assert cli._card_came_up(p, True, {"name": "Card B"}) == "  card: now on Card B (it was Card A), recorded in card.json"
    assert cli._card_came_up(p, False, {"name": "Card B"}) is None
    settle(["--gpu-workers", "4"])
    assert json.loads(rec.read_text())["gpu_name"] == "Card B", "a rewrite keeps the card's name"
    assert cli._card_came_up(p, False, {"name": "numpy"}) is None


def test_auto_is_12_where_a_card_and_cupy_are(monkeypatch):
    import importlib.util
    from klip_tpe import parallel, reducer
    find = importlib.util.find_spec
    monkeypatch.setattr(parallel, "card_present", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "cupy" else find(name, *a))
    monkeypatch.setattr(reducer, "_gpu_card_factory", lambda: object())
    assert cli._auto_card_share() == cli.AUTO_CARD_SHARE == 12.0
    monkeypatch.setattr(parallel, "card_present", lambda: False)
    assert cli._auto_card_share() == 0.0


def test_a_resumed_runs_summary_reads_the_finished_annuli(tmp_path):
    """The run's summary takes an annulus finished before a resume from its annulus_done.json."""
    r = Runner.__new__(Runner)
    r.run_dir, r._card_annuli = str(tmp_path), {1: {"annulus": 2, "jobs": 5, "marked": 5, "busy_fraction": 0.4,
                                                    "search_s": 60.0}}
    os.makedirs(tmp_path / "annulus01")
    rec = {"annulus": 1, "jobs": 7, "marked": 8, "busy_fraction": 0.5, "search_s": 90.0}
    (tmp_path / "annulus01" / "annulus_done.json").write_text(json.dumps({"annulus": 1, "card": rec}))
    got = r._card_per_annulus([types.SimpleNamespace(annulus=0), types.SimpleNamespace(annulus=1),
                               types.SimpleNamespace(annulus=2)])
    assert got == [rec, r._card_annuli[1]]
