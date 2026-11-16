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
from klip_tpe.parallel import card_assign, card_balance, card_picks, card_slot, describe_card
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


def _run(tmp_path, name, factory=None, share=SHARE, rule="trial", devices=None, adapt=0):
    red, space, obj, samp, _ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    if factory is not None:
        pr.card_share, pr.card_workers, pr.card_factory, pr.card_rule = share, 2, factory, rule
        pr.card_device, pr.card_adapt_every = devices, adapt
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


def _cards_of(r):
    """Each draw's partitions on each card, ``{(draw, partition): card}``, from a record."""
    out = {}
    for j, got in enumerate(r["meta"].get("draw_card") or []):
        for pid in got:
            out[(j, pid)] = int((r["meta"].get("draw_card_index") or [{}] * (j + 1))[j].get(pid, 0))
    return out


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
    """``auto`` is recorded as such, with its re-balance, and a resume takes it up again: adaptive
    where a card is, none where none is.  A number is kept wherever the run resumes."""
    monkeypatch.setattr(cli, "_auto_card_share", lambda: 30.0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run = tmp_path / "run_x"
    run.mkdir()
    base = ["resume", "--root", str(tmp_path), "--run-dir", str(run)]

    def settle(extra):
        a = cli._build_parser().parse_args(base + extra)
        a._argv = base + extra
        cli._card_for_run(a)
        return a.gpu_share, a.gpu_workers, a.gpu_adapt

    rec = run / "card.json"
    assert settle([]) == (30.0, 3, 25) and rec.exists(), "auto, settled and recorded at the start"
    d = json.loads(rec.read_text())
    assert d["gpu_share"] == "auto" and d["adapt_every"] == 25 and d["rule"] == "trial"
    assert settle([]) == (30.0, 3, 25)
    monkeypatch.setattr(cli, "_auto_card_share", lambda: 0.0)
    assert settle([]) == (0.0, 3, 25), "auto on a machine without a card: none"
    assert settle(["--gpu-share", "40", "--gpu-workers", "4"]) == (40.0, 4, 0), "a number stays as set"
    assert settle([]) == (40.0, 4, 0), "and is kept wherever the run resumes"
    assert settle(["--gpu-share", "auto", "--gpu-adapt-every", "10"]) == (0.0, 4, 10)
    d = json.loads(rec.read_text())
    assert [h["gpu_share"] for h in d["history"]] == ["auto", 40.0, "auto"] and d["adapt_every"] == 10
    with pytest.raises(SystemExit):
        settle(["--gpu-share", "140"])
    with pytest.raises(SystemExit):
        settle(["--gpu-share", "12,8"]), "two shares for one card"
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
    a.gpu_share, a.gpu_workers, a.gpu_adapt = 25.0, 2, 25
    cli._set_card(pr, a)
    assert (pr.card_share, pr.card_workers, pr.card_adapt_every) == (25.0, 2, 25)
    b = cli._build_parser().parse_args(["resume", "--root", str(tmp_path), "--run-dir", str(run2)])
    cli._set_card(pr, b)
    assert pr.card_share == 0.0 and pr.card_adapt_every == 0


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
        settle(["--gpu-device", "0,0"])
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


# ------------------------------------------------------------------ several cards, the automatic share

def test_cards_take_their_shares_in_hash_order():
    """Two cards: each its share over many trials, at most its share rounded up in a trial, no job
    twice, the first card's jobs those one card would take, and the second card's the next."""
    on = [0, 0]
    jobs = 0
    for e in range(1, 3001):
        tags = [f"a1_e{e}_d{j}_inj_n{p}" for j in (1, 2, 3) for p in range(1, 7)]
        got = card_assign(tags, [12.0, 8.0])
        first = {t for t, k in got.items() if k == 0}
        assert first == card_picks(tags, 12.0), "card 0 picks as one card would"
        ranked = sorted(tags, key=lambda t: (int.from_bytes(__import__("hashlib").sha1(t.encode()).digest()[:4],
                                                            "big"), t))
        assert set(got) == set(ranked[:len(got)]) and all(got[t] == 0 for t in ranked[:len(first)])
        for k, share in enumerate((12.0, 8.0)):
            n = sum(1 for v in got.values() if v == k)
            assert n <= math.ceil(share * len(tags) / 100.0)
            on[k] += n
        jobs += len(tags)
    assert abs(on[0] / jobs - 0.12) < 0.005 and abs(on[1] / jobs - 0.08) < 0.005
    assert card_assign(["a", "b"], [0.0, 100.0]) == {"a": 1, "b": 1}
    assert card_assign([], [50.0, 50.0]) == {}


def test_the_balance_moves_the_share_toward_both_finishing_together():
    """card_balance: the share at which each pool, at its measured rate, finishes its part of the
    stretch's jobs at the same time, approached by half the ratio a step, at most twice a step."""
    # CPU: 48 workers, 960 jobs at 50 s; card: 3 workers, 60 jobs at 2.5 s; 765 of the 1020 eligible
    cpu, card = (960, 960 * 50.0, 48), (60, 60 * 2.5, 3)
    new, m = card_balance(cpu, [card], 765, [12.0])
    assert m["cpu_rate"] == pytest.approx(0.96) and m["card_rates"][0] == pytest.approx(1.2)
    assert m["balanced"][0] == pytest.approx(100 * 1020 * 1.2 / (0.96 + 1.2) / 765)
    assert new == [17.0], "at most a factor 2 a step, half of it in the power: 12 x 2 ** 0.5"
    # near the balance it stays; below the floor it stops; together at most the cap
    near = round(m["balanced"][0], 1)
    assert card_balance(cpu, [card], 765, [near])[0] == [near]
    slow = (60, 60 * 400.0, 3)
    assert card_balance(cpu, [slow], 765, [2.5])[0] == [2.0]
    fast = (600, 600 * 0.05, 3)
    two, _ = card_balance(cpu, [fast, fast], 765, [80.0, 80.0])
    assert sum(two) <= 95.0 + 1e-9 and two[0] == two[1]
    # too few jobs, or a card out of the run, keep the shares
    assert card_balance((5, 250.0, 48), [card], 765, [12.0])[0] is None
    assert card_balance(cpu, [None], 765, [12.0])[0] is None
    assert card_balance(cpu, [card, None], 765, [12.0, 9.0])[0][1] == 9.0


def test_two_cards_each_take_their_jobs(monkeypatch):
    """Two stand-in cards, "A" and "B": each card's workers get their own CUDA_VISIBLE_DEVICES, each
    job goes to the card card_assign gives it, both come up in the log, and on_ready hears from each."""
    from klip_tpe.metrics import Source
    from klip_tpe.space import Config
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    red, *_ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    heard = []
    pr.card_share, pr.card_workers, pr.card_factory = [40.0, 35.0], 1, Seen.from_cpu
    pr.card_device, pr.card_on_ready = ["A", "B"], (lambda info: heard.append(info["card_index"]))
    params = {"inrad": 8.0, "outrad": 30.0, "k_klip": 5, "bin": 12}
    parts = list(red.reducers)
    cfg = Config(params=dict(params), per_partition={p: dict(params) for p in parts}, selected=parts,
                 x=np.zeros(0))
    tags = [f"a1_e7_d{j}_inj" for j in (1, 2, 3)]
    try:
        evs = [pr.reduce_config(cfg, [Source(0.6, 30.0, 1e-3)], tag=t, card=True, card_group=tags) for t in tags]
        summary, notes = pr.card_summary(), pr.take_card_notes()
    finally:
        pr.close()
    want = card_assign([f"{t}_{p}" for t in tags for p in parts], [40.0, 35.0])
    for t, ev in zip(tags, evs):
        for p in parts:
            m = ev.meta[str(p)]
            k = want.get(f"{t}_{p}")
            assert bool(m.get("card")) == (k is not None) and (k is None or m["card_index"] == k), (t, p, m)
    assert [c["info"]["cvd"] for c in summary["cards"]] == ["A", "B"] and summary["devices"] == ["A", "B"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert sorted(heard) == [0, 1]
    assert any(ln.startswith("  card 0: 1 worker on stand-in card (CUDA_VISIBLE_DEVICES=A)") for ln in notes), notes
    assert any(ln.startswith("  card 1: 1 worker on stand-in card (CUDA_VISIBLE_DEVICES=B)") for ln in notes), notes
    assert summary["cpu"]["jobs"] + summary["jobs"] == len(tags) * len(parts) and summary["cpu"]["busy_s"] > 0
    assert summary["eligible"] == len(tags) * len(parts) and summary["shares"] == [40.0, 35.0]
    assert "card 0: " in describe_card(summary) and " and card 1: " in describe_card(summary)


@pytest.mark.slow
def test_the_automatic_share_moves_and_every_trial_keeps_its_own(tmp_path, cpu_only, monkeypatch):
    """The automatic share on two stand-in cards: the scores are the CPU's, each trial records the
    shares it went out with, and its jobs are on the cards those shares give, so a resume that
    takes them from the checkpoint picks the same.  Each change is logged and kept."""
    import klip_tpe.parallel as P
    real = P.card_balance

    def stepped(cpu, cards, eligible, shares, **kw):    # the real measurement, a step every time
        new, m = real(cpu, cards, eligible, shares, **kw)
        return (None if new is None else [max(2.0, round(s - 7.5, 1)) for s in shares]), m
    monkeypatch.setattr(P, "card_balance", stepped)
    recs, logs, run_dir = _run(tmp_path, "auto", StandIn.from_cpu, share=[40.0, 30.0], devices=["A", "B"], adapt=3)
    assert _scores(recs) == _scores(cpu_only[0])
    seen = set()
    for r in recs:
        shares = r["meta"].get("card_shares")
        assert shares and len(shares) == 2, r["meta"]
        seen.add(tuple(shares))
        draws, jobs = _trial_jobs(r)
        want = card_assign(list(jobs.values()), shares)
        got = _cards_of(r)
        for (j, pid), tag in jobs.items():
            assert got.get((j, pid)) == want.get(tag), (tag, shares, got)
    assert len(seen) > 1, "the share moved"
    lines = [json.loads(ln) for ln in open(os.path.join(run_dir, "card_shares.jsonl"))]
    assert lines and all(e["shares"] == [max(2.0, round(s - 7.5, 1)) for s in e["previous"]] for e in lines)
    assert any(ln.startswith("  card: share 40 / 30 % -> 32.5 / 22.5 % over the last") for ln in logs), \
        [ln for ln in logs if "card: share" in ln]
    st = json.load(open(os.path.join(run_dir, "checkpoint.json")))
    assert st["card_shares"] == lines[-1]["shares"]
    card = json.load(open(os.path.join(run_dir, "final_results.json")))["card"]
    assert card["annuli"][0]["shares_end"] == lines[-1]["shares"] and len(card["cards"]) == 2
    assert len(card["annuli"][0]["cards_busy_fraction"]) == 2
    assert any("card 0's 2 workers were busy" in ln and "card 1's 2 workers" in ln for ln in logs)


def test_the_checkpoint_keeps_the_automatic_share(tmp_path):
    """The automatic share and each trial in flight's shares go into the checkpoint and come back."""
    red, space, obj, samp, cfg = build_synthetic_run()
    pr = PartitionedReducer(red.reducers)
    pr.card_share, pr.card_adapt_every = 12.0, 25
    r = Runner(pr, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    assert r._card_shares_now() == [12.0]
    r._card_shares = [17.0]
    r.ia, r.history = 0, None
    x = space.default_vector()
    r._inflight = [{"phase": "tpe", "x": x, "tag": "a1_e1", "draws": [[]], "rng_after": r.rng.bit_generator.state,
                    "card_shares": [17.0]}]
    st = json.loads(r._serialize_checkpoint(r._checkpoint_snapshot(False)))
    assert st["card_shares"] == [17.0] and st["pending"][0]["card_shares"] == [17.0]
    r2 = Runner(pr, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    r2._apply_checkpoint(st, same_frames=False)
    assert r2._card_shares_now() == [17.0]
    r2.history = __import__("klip_tpe").optimizers.History(space.ndim)
    r2._resume_pending, r2.ia = st["pending"], 0
    assert r2._take_resume_pending()[0]["card_shares"] == [17.0]
    pr.card_adapt_every = 0
    assert r2._card_shares_now() is None, "a fixed share: the trials carry none"
