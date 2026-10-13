"""``--n-sources`` on the command line.

From the NEAR2 note of 2026-10-13: the next NEAR2 run wants two injected sources in every
annulus instead of the per-annulus rule.  ``RunConfig.n_sources`` already did that, but
``klip-tpe near`` had no option for it.  What must hold:

* ``--n-sources`` reaches ``RunConfig.n_sources`` from ``near`` and ``generic``: one value as
  a count, several as a list per annulus, none as the rule;
* a count below 0 is refused;
* the run setup records it, and each annulus' first log line gives the count used and where
  it came from.
"""
from __future__ import annotations

import json

import pytest

from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, cli
from klip_tpe.positions import n_sources_rule

from conftest import build_synthetic_run


def _args(*extra, cmd="near"):
    base = ["near", "--root", "data"] if cmd == "near" else ["generic", "--cube", "c.fits"]
    return cli._build_parser().parse_args(base + list(extra))


def test_the_option_reaches_the_run_config():
    assert "n_sources" not in cli._protocol_config(_args())           # the rule
    assert cli._protocol_config(_args("--n-sources", "2"))["n_sources"] == 2
    assert cli._protocol_config(_args("--n-sources", "2", "3", "0"))["n_sources"] == [2, 3, 0]
    assert cli._protocol_config(_args("--n-sources", "4", cmd="generic"))["n_sources"] == 4
    with pytest.raises(SystemExit, match="0 or more"):
        cli._protocol_config(_args("--n-sources", "2", "-1"))


def test_the_run_records_it_and_says_what_it_used(tmp_path):
    ns = cli._protocol_config(_args("--n-sources", "2"))["n_sources"]
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=4, n_init=2, n_sources=ns, param_verify=False,
        calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    logs = []
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append).run()
    banner = [l for l in logs if l.startswith("=== annulus 1/1")]
    assert banner and "2 sources (n_sources 2)" in banner[0], banner
    assert json.load(open(tmp_path / "run_setup.json"))["config"]["n_sources"] == 2
    assert [l for l in open(tmp_path / "run_setup.txt") if l.split(":")[0].strip() == "n_sources"] \
        == [f"{'n_sources':<19s}: 2\n"]
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert recs and all(len(r["sources"]) == 2 for r in recs)


def test_an_annulus_left_to_the_rule_says_so(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30, 45], n_sources=[2, 0])
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "a"), log=lambda s: None)
    assert r._nsrc_asked(0) == 2 and r._nsrc_asked(1) is None and r._nsrc_asked(5) is None
    assert r._nsrc(1) == n_sources_rule(1, 45 * red.pxscale, None)
    assert Runner(red, space, obj, samp, RunConfig(ann_edges=[8, 30]), str(tmp_path / "b"),
                  log=lambda s: None)._nsrc_asked(0) is None
