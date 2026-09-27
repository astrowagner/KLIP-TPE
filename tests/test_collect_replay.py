"""collect.py re-scores each run's validated winner beside its seeded default, and the paper
quotes the ratio.  It has to rebuild the problem the run solved, or it measures something
else under the run's name.  Three ways it did not, all found on 2026-09-26 by replaying each
run's committed validation trial (the winner at ``winner_sources``) and comparing with the
score the run committed:

* the beta Pic objective lacked the disk (forbidden injection PAs, masked noise pixels);
* the metric's radial-profile flattening followed today's default (off since 2026-09-22),
  while A2, B2, C and the E2/F2/G2 benchmarks were searched and validated with it on;
* B2's winner vector was decoded position by position in a space that no longer carries its
  pinned ``bin_gN`` dimensions (38 entries into 34), which read a different configuration.

With the three fixed, A2, B2, C and D replay their committed trials to four decimals.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(os.path.dirname(HERE), "paper_runs"),
              os.path.join(os.path.dirname(os.path.dirname(HERE)), "paper_runs")]
PAPER_RUNS = next((d for d in CANDIDATES if os.path.exists(os.path.join(d, "collect.py"))), None)
pytestmark = pytest.mark.skipif(PAPER_RUNS is None, reason="paper_runs/ not next to the package")


def _text(name):
    with open(os.path.join(PAPER_RUNS, name)) as f:
        return f.read()


@pytest.fixture(scope="module")
def collect():
    if PAPER_RUNS not in sys.path:
        sys.path.insert(0, PAPER_RUNS)
    spec = importlib.util.spec_from_file_location("paper_collect", os.path.join(PAPER_RUNS, "collect.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_collect"] = mod
    spec.loader.exec_module(mod)
    return mod


def _space_like_b2(pin_bin):
    """Two groups, (bin, n_ang, k_klip) each, plus the drop slot -- with bin either searched
    over [1, 1] (the space B2 ran in) or pinned into ``fixed`` (the space make_space builds now)."""
    from klip_tpe import Param
    from klip_tpe.space import SearchSpace
    block = [Param("n_ang", 1, 8, "int", default=2), Param("k_klip", 1, 12, "int", default=5)]
    fixed = {"spat_mean": False}
    if pin_bin:
        fixed["bin"] = 1
    else:
        block = [Param("bin", 1, 1, "int", default=1)] + block
    sp = SearchSpace(fixed=fixed)
    sp.replicate(block, ["g1", "g2"], name_fmt="{base}_{pid}")
    sp.with_selection("two_slot", partitions=["g1", "g2"], max_drop=1)
    return sp


def _recorded(space):
    """The ``space.params`` records a run writes into run_setup.json."""
    return [{"name": p.name, "base": getattr(p, "base", None) or p.name} for p in space.params]


def test_a_winner_from_a_space_with_pinned_dims_decodes_to_the_same_configuration(collect):
    old, new = _space_like_b2(pin_bin=False), _space_like_b2(pin_bin=True)
    assert old.ndim == new.ndim + 2
    x_old = np.array([1, 6, 10, 1, 3, 11, 0], float)        # bin, n_ang, k per group; no drop
    want = old.decode(x_old)
    x_new = collect.run_vector(x_old, _recorded(old), new)
    got = new.decode(x_new)
    assert [str(s) for s in got.selected] == [str(s) for s in want.selected] == ["g1", "g2"]
    for g in ("g1", "g2"):
        for k in ("bin", "n_ang", "k_klip"):
            assert got.per_partition[g][k] == want.per_partition[g][k], (g, k)
    # and the old way -- the recorded vector read position by position -- does not
    bad = new.decode(x_old[:new.ndim])
    assert any(bad.per_partition[g][k] != want.per_partition[g][k]
               for g in ("g1", "g2") for k in ("n_ang", "k_klip"))


def test_run_vector_refuses_what_it_cannot_rebuild(collect):
    old, new = _space_like_b2(pin_bin=False), _space_like_b2(pin_bin=True)
    with pytest.raises(ValueError, match="entries"):
        collect.run_vector([1, 2, 3], _recorded(old), new)
    # a run dimension held at a value this space does not pin
    with pytest.raises(ValueError, match="neither searched nor pinned"):
        collect.run_vector([2, 6, 10, 1, 3, 11, 0], _recorded(old), new)
    # a dimension this space searches that the run never had
    with pytest.raises(ValueError, match="did not search"):
        collect.run_vector([6, 10, 3, 11, 0], _recorded(new)[:-1] + [{"name": "other"}], new)


def test_run_flatten_follows_the_runs_own_setup(collect, monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "OUT", str(tmp_path))
    monkeypatch.setitem(collect.TARGETS, "T", ("T_run", "t", (0.5, 0.0)))
    d = tmp_path / "T_run"
    d.mkdir()
    for metric, want in (({"flatten": True}, True), ({"flatten": False}, False),
                         ({}, True)):            # absent: a run from before the key was recorded
        (d / "run_setup.json").write_text(json.dumps({"objective": {"metric": metric}}))
        assert collect.run_flatten("T") is want, metric
    (d / "run_setup.json").unlink()
    assert collect.run_flatten("T") is False     # no setup file: today's default


def test_collect_rebuilds_the_objective_the_run_searched():
    col = _text("collect.py")
    assert "R.bp_disk(red) if which in (\"A\", \"A2\", \"B\", \"B2\")" in col
    assert "forbidden_pa=fpa, pixel_mask=pmask" in col and "flatten=run_flatten(which)" in col
    assert "runner._nsrc_cache[ia] = n_run" in col, "the run's own source count"
    assert "run_vector(a[\"winner_x\"], run_params, space)" in col
    assert col.count("planet_snr(") >= 4 and col.count("obj.metric)") >= 3, \
        "the companion is measured with the run's metric (its noise ring has the disk masked)"


def test_f5_shows_the_default_it_captions():
    """f5 captions its panels with the companion S/N collect measured, on the images it shows.
    The default panel once reduced ``space.default_vector()`` unprojected at the space's own k,
    cached under a name without the parameters; then ``default_params`` through
    ``red.reduce(params)``, which gives every partition the channel-averaged parameters (HD
    95086's K1 and K2 differ in angsep).  Both panels now come from collect.clean_image -- the
    run's own reduction path -- keyed on the search vector, and the winner is not read from
    best_clean.fits, which a run that subtracts the companion writes without it."""
    src = _text("figs.py")
    assert "def default_image(which, a):" in src and "def winner_image(which, a):" in src
    assert 'a["default_x"]' in src and 'a["winner_x_space"]' in src and "C.clean_image(" in src
    assert "hexdigest()" in src and "space.decode(space.default_vector())" not in src
    gal = src[src.index("def fig_gallery"):src.index("def fig_trace")]
    assert "best_clean.fits" not in gal and "companion_snr_of(a, " in gal


def test_collect_measures_the_companion_on_the_runs_own_reduction_path():
    """Default and winner go through Runner._reduce (the zone, each partition's own block),
    with the companion present; the paired re-measurement subtracts it when the run did; the
    paper's companion S/N is the forward-model one (klip_tpe.companion)."""
    col = _text("collect.py")
    assert "red.reduce(ReductionRequest" not in col.split('"""', 2)[2]      # outside the docstring
    assert 'runner0._reduce(cfg0, None, tag="collect_default")' in col
    assert 'runner0._reduce(cfgw, None, tag="collect_winner")' in col
    assert "subtract_known=list(sub)" in col and "replace(cfg, subtract_known=[])" in col
    assert "def companion_fm(" in col and 'rec["companion_fm_default"]' in col and 'rec["companion_fm_optimized"]' in col
    assert '"D2": (R.ENGINE_DIRS["D2"]["pyklip"]' in col and '"D2K": (R.ENGINE_DIRS["D2"]["klip"]' in col


def test_companion_tests_rebuild_each_runs_objective():
    """The paper's companion tests (beta Pic b at its own brightness, HD 95086 b's noise ring,
    HIP 65426 b in F1140C and in F444W) re-score configurations the way collect.py does: the
    run's own objective, the winner mapped by parameter name, the raw validation metric."""
    src = _text("companion_tests.py")
    assert src.count("C.build(which)") == 3 and src.count("C.run_vector(") == 3
    assert "raw_only=True" in src and 'LA._rebuild("miri"' in src
    assert '"companion_fm_default"' in src and 'fm["contrast"]' in src, \
        "beta Pic b's brightness is its negative-companion fit, not flux_scale x published"
    assert "Source(rho_b, pa, cc)]" in src, "one source per reduction"


def test_miri_figure_uses_the_ablation_rebuild():
    """f13 must show the configurations Table 3 scores: rebuilt by library_ablation, which
    refuses a setup that does not reproduce the run."""
    src = _text("miri_fig.py")
    assert 'LA._rebuild("miri"' in src and "LA._x_from_params(" in src and "LA.CARTER" in src


def test_miri_figure_shows_the_zone_inside_the_annulus():
    """f13 runs from the inner working angle out.  The default and the winners were scored
    over the annulus only; inside it they are reduced once more with the same parameters and
    the annulus' bins, for display, and joined at the annulus' inner edge."""
    src = _text("miri_fig.py")
    assert 'zone=(iwa, float(fr["inrad"]))' in src and "with _binned_as(red," in src
    assert 'out = (rr < r["iwa"]) | (rr > r["outrad"])' in src


def test_image_figures_use_inferno_and_viridis():
    """Two color maps for every image panel: inferno for intensity, viridis for S/N and the
    parameter-verification maps.  Until 2026-09-26 they were white-to-black and red/white/blue."""
    for name in ("figs.py", "miri_fig.py"):
        src = _text(name)
        for cm in ("RdBu", "Greys", '"Reds"', "Blues", "bwr", "coolwarm", "seismic", "TwoSlopeNorm"):
            assert cm not in src, (name, cm)
    assert 'cmap="viridis" if snr else "inferno"' in _text("figs.py")
