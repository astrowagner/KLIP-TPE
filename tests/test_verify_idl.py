"""A run keeps the verify it was made with (``RunConfig.verify_snr``, Kevin Wagner,
2026-11-01).  New runs measure as the search does (``"search"``).  Runs saved before then, and
``--from-idl-setup`` runs, keep the IDL verify's S/N, the candidate search and param_verify as
they measured, and no pixel mask outside the search (``"idl"``, :data:`metrics.VERIFY_IDL`).

The numbers below are the ones d104527, the last code before the change, gave on these inputs.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from klip_tpe import candidates as C
from klip_tpe import cli
from klip_tpe import display as D
from klip_tpe import param_verify as PV
from klip_tpe import verify as V
from klip_tpe.metrics import VERIFY_IDL, Source, source_xy, star_center
from klip_tpe.runner import RunConfig, Runner

import test_verify_stack as T
from conftest import build_synthetic_run

IDL = dict(VERIFY_IDL)
KEEP_SPIKES = {"declip_nsig": None}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def test_new_runs_measure_as_the_search_and_saved_runs_as_before():
    assert RunConfig().verify_snr == "search"
    assert RunConfig.from_dict(RunConfig().to_dict()).verify_snr == "search"
    d = RunConfig().to_dict()
    d.pop("verify_snr")
    assert RunConfig.from_dict(d).verify_snr == "idl", "a run saved before 2026-11-01 keeps its verify"


def test_a_run_that_mirrors_an_idl_setup_verifies_as_the_idl(tmp_path):
    from klip_tpe.idl_compare import matched_run_config
    setup = tmp_path / "run_setup.txt"
    setup.write_text("# KLIP-TPE run setup\nann_edges (px)     : 8.00, 30.00\nn_iter / n_init    : 6 / 3\n"
                     "n_valid / n_top    : 2 / 1\ngamma/ncand/explore: 0.250 / 48 / 0.15\n"
                     "p_local / n_elite  : 0.15 / 5\nseed / bench_tag   : 4242 / none\n")
    assert matched_run_config(str(setup), seed=1)[0].verify_snr == "idl"
    assert matched_run_config(str(setup), seed=1, verify_snr="search")[0].verify_snr == "search"


def test_the_command_line_picks_the_verify():
    p = cli._build_parser()
    assert "verify_snr" not in cli._protocol_config(p.parse_args(["near", "--root", "/r"]))
    assert cli._protocol_config(p.parse_args(["near", "--root", "/r", "--verify-snr", "idl"]))["verify_snr"] == "idl"
    a = p.parse_args(["rereduce", "--root", "/r", "--run-dir", "x", "--verify-snr", "search"])
    assert a.verify_snr == "search"
    with pytest.raises(SystemExit):
        p.parse_args(["near", "--root", "/r", "--verify-snr", "both"])


def test_the_idl_verify_gives_the_numbers_of_before():
    stack = T._planted_stack(6, 0.8, 40.0, 8.0)
    s = V.snr_fap(np.nanmean(stack, axis=0), 0.8, 40.0, T.PX, T.FWHM, rules=IDL)
    assert (s["x"], s["y"], s["nap"]) == (37, 60, 22)
    assert s["snr"] == pytest.approx(7.340025853543736, rel=1e-9)
    assert s["fap"] == pytest.approx(2.14645868238118e-07, rel=1e-6)
    inj = T._planted_stack(6, 1.0, 200.0, 8.0, seed=11)
    rows = V.verify_candidates(stack, list(range(6)), [(0.8, 40.0), (0.7, 120.0)], T.FWHM, T.PX, n_boot=50,
                               rng=np.random.default_rng(3), known=[(1.1, 300.0)],
                               injected=(inj, [(1.0, 200.0, 1e-4), (0.9, 20.0, 2e-4)]), rules=IDL)
    c, j = rows[0], rows[2]
    assert c["kind"] == "candidate" and j["kind"] == "injection"
    assert c["snr"] == pytest.approx(7.289857708546292, rel=1e-9)
    assert c["fap"] == pytest.approx(3.2391918054859343e-07, rel=1e-6)
    assert c["stim"] == pytest.approx(2.9123892618266853, rel=1e-9) and c["stim_fap"] == 0.0
    assert c["boot_sig"] == pytest.approx(6.928776695446888, rel=1e-9)
    assert j["snr"] == pytest.approx(7.150890040570416, rel=1e-9)
    assert j["stim"] == pytest.approx(3.9122931950671123, rel=1e-9)
    # the solved contrasts, on their Gaussian maps of the clean image and the difference
    sol = V.solved_contrasts(np.nanmean(T._noise_stack(4, 7), axis=0), np.nanmean(inj, axis=0), 1.0, 200.0, 1e-4,
                             T.PX, T.FWHM, rules=IDL)
    assert sol["c_det"] == pytest.approx(6.895081874800849e-05, rel=1e-6)
    assert sol["snr1"] == pytest.approx(7.40139071557372, rel=1e-9)


def test_the_idl_candidate_search_gives_the_numbers_of_before():
    stack = T._planted_stack(6, 0.8, 40.0, 8.0)
    rows = C.find_candidates(None, stack, T.FWHM, T.PX, night_ids=list(range(6)), known=[(1.1, 300.0)], rules=IDL)
    top = rows[0]
    assert (top["x"], top["y"], top["n_nights"]) == (38, 59, 4)
    assert top["snr"] == pytest.approx(7.289857708546292, rel=1e-9)
    want = [0.6739082534844881, 4.459982414634926, 1.999593861920271, 3.50743603829111, 2.736410411704548,
            5.158567488678676]
    assert top["per_night"] == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("rules", [KEEP_SPIKES, IDL], ids=["search", "idl"])
def test_each_night_of_the_candidate_search_is_measured_on_its_own_image(rules):
    """dd433e7 cached each night's map under the address of a view made anew for every peak, and
    an address freed before came back: every night took one night's S/N."""
    stack = T._planted_stack(6, 0.8, 40.0, 8.0)
    rows = C.find_candidates(None, stack, T.FWHM, T.PX, night_ids=list(range(6)), rules=rules)
    for r in rows[:3]:
        own = [V.snr_fap(stack[j], r["rho"], r["theta"], T.PX, T.FWHM, rules=rules)["snr"] for j in range(6)]
        assert r["per_night"] == pytest.approx(own, rel=1e-12)
        assert len(set(np.round(r["per_night"], 9))) == 6


def test_the_idl_param_verify_gives_the_numbers_of_before():
    cl = np.stack([T._noise_stack(1, s)[0] for s in range(4)])
    src = [Source(0.8, 40.0, 1e-4), Source(0.8, 220.0, 1e-4)]
    cx, cy = star_center(cl.shape[-2:])
    yy, xx = np.mgrid[0:cl.shape[-2], 0:cl.shape[-1]]
    add = np.zeros(cl.shape[-2:])
    for s_ in src:
        x, y = source_xy(s_.rho, s_.theta, T.PX, cx, cy)
        add += 3.0 * np.exp(-0.5 * ((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (T.FWHM / 2.3548) ** 2)
    pv = PV.param_verify_cubes(cl, cl + add[None], [1.0] * 4, src, 1e-4, T.PX, T.FWHM, 10.0, 30.0,
                               write_fits=False, rules=IDL, known=[(1.1, 300.0)])
    assert pv["Kc"] == pytest.approx(18263.092374245316, rel=1e-9)


def test_the_runner_keeps_a_saved_run_on_its_verify(tmp_path):
    for how, rules, masked in (("idl", IDL, False), ("search", None, True)):
        red, space, obj, samp, cfg = build_synthetic_run(verify_snr=how)
        size = red.reducers["n1"].size
        mask = np.zeros((size, size), bool)
        mask[10:20, 10:20] = True
        obj.metric.pixel_mask = mask
        run = Runner(red, space, obj, samp, cfg, str(tmp_path / how), log=lambda s: None)
        got = run._ring_rules()
        if rules is not None:
            assert got == rules
        else:
            assert got["pixel_mask"] is mask and "verify" not in got
        assert (run._pixel_mask() is mask) is masked


def test_the_display_reads_the_verify_a_run_was_made_with():
    metric = {"name": "mawet_peak", "excl_fwhm": 1.5, "declip_nsig": None, "band_fwhm": 0.6}
    old = {"config": {"seed": 1}, "objective": {"metric": metric}}
    new = {"config": {"seed": 1, "verify_snr": "search"}, "objective": {"metric": metric}}
    assert D._run_rules(old) == IDL and D._run_verify_idl(old)
    got = D._run_rules(new)
    assert not D._run_verify_idl(new) and "declip_nsig" in got and got["declip_nsig"] is None
    assert json.loads(json.dumps(IDL)) == IDL
