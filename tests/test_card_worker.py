"""The card workers in the pool (PartitionedReducer.card_share, parallel.CardSpec).

A share of the search's injected draws goes to card worker processes with their own queue,
picked by a hash of the job's tag, so a run and its resumes send the same jobs there.  Here a
stand-in card runs the CPU reducer in those processes: its images are the CPU's to the bit, so
a run with it must give the CPU-only run's scores, and every draw's partitions on the card must
be the ones the hash picked.  When the card cannot start, fails jobs, or a worker dies, its jobs
go back to the CPU workers and the run goes on without it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import RunConfig, Runner, ValidationConfig, cli
from klip_tpe.parallel import card_slot
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


def _run(tmp_path, name, factory=None, share=SHARE):
    red, space, obj, samp, _ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=4, pool="processes")
    if factory is not None:
        pr.card_share, pr.card_workers, pr.card_factory = share, 2, factory
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


@pytest.fixture(scope="module")
def cpu_only(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("card"), "cpu")


@pytest.mark.slow
def test_the_card_takes_the_share_its_hash_picks_and_changes_no_score(tmp_path, cpu_only):
    recs, logs, run_dir = _run(tmp_path, "card", StandIn.from_cpu)
    assert _scores(recs) == _scores(cpu_only[0]), "the stand-in card gives the CPU's bits"
    on, off = 0, 0
    for r in recs:
        draws = r["meta"].get("draw_card") or [[]] * int(r["meta"].get("draw_n") or 1)
        for j, got in enumerate(draws):
            for pid in r["meta"]["selected"]:
                ev = r["phase"] if r["phase"] in ("seed", "reseed") else f"e{r['index'] + 1}"
                tag = f"a{r['annulus'] + 1}_{ev}_d{j + 1}_inj_{pid}"
                want = card_slot(tag) < SHARE
                assert (pid in got) == want, (tag, got)
                on, off = on + want, off + (not want)
    assert on and off
    assert any("card: 2 workers on stand-in card (driver 0.0), 50 % of the search's injected draws" in ln
               for ln in logs), logs
    setup = json.load(open(os.path.join(run_dir, "run_setup.json")))
    assert setup["reducer"]["card"] == {"share": SHARE, "workers": 2}
    card = json.load(open(os.path.join(run_dir, "final_results.json")))["card"]
    assert card["up"] and card["jobs"] == on and 0 < card["busy_fraction"] < 1
    assert card["marked"] == on and card["refused"] == {}
    # the annulus' line: the search's marked jobs, all reduced on the card
    assert any(ln == f"  annulus 1: the card reduced {on} of the {on} jobs marked for it" for ln in logs), \
        [ln for ln in logs if "the card reduced" in ln]
    assert any(ln.startswith("# card: share 50 %") for ln in open(os.path.join(run_dir, "bench_summary.txt")))


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
