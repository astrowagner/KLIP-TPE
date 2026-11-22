"""The README panel's script: RX J0534 by default, through run_rxj0534.py, titled by name.

The run takes about three days on four nights of LMIRCam data (three annuli), so a wrong argument
found at the end costs the run.  These tests check the command line it hands run_rxj0534.py, and that the
display's title replaces the run directory's name.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
readme_panel = pytest.importorskip("readme_panel")


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_the_default_is_rxj0534_through_run_rxj0534(monkeypatch, tmp_path):
    seen = {}
    fake = types.ModuleType("run_rxj0534")
    fake.main = lambda argv: (seen.__setitem__("argv", list(argv)), 0)[1]
    monkeypatch.setitem(sys.modules, "run_rxj0534", fake)
    assert readme_panel.main(["--run-dir", str(tmp_path / "readme")]) == 0
    argv = seen["argv"]
    assert _opt(argv, "--partitions") == "4", "the README panel is the four-night search"
    assert _opt(argv, "--n-iter") == "3000"
    assert _opt(argv, "--title") == "RX J0534"
    assert _opt(argv, "--n-sources") == "2" and "--even-pa" in argv, "two at a time, 180 deg apart"
    assert _opt(argv, "--contrast-grid") == "grid"
    assert _opt(argv, "--cg-step") == "0.5", "half-FWHM steps: five separations in the +/- 2 FWHM annulus"
    assert _opt(argv, "--cg-pa") == "12"
    assert _opt(argv, "--extra-annuli") == "2", "three annuli: the companion's and two outside it"
    assert _opt(argv, "--out") == str(tmp_path / "readme")
    assert argv[-1] == "--show" and "--panels" not in argv
    assert "--seed" not in argv and "--n-init" not in argv, "run_rxj0534.py's own seed and warm-up"


def test_no_show_still_writes_the_panels(monkeypatch):
    a = readme_panel.parse(["--no-show", "--n-iter", "5000", "--cg-step", "1", "--seed", "4"])
    argv = readme_panel.rxj0534_argv(a)
    assert "--panels" in argv and "--show" not in argv
    assert _opt(argv, "--n-iter") == "5000"
    assert _opt(argv, "--cg-step") == "1"
    assert _opt(argv, "--seed") == "4"
    assert a.run_dir == os.path.expanduser("~/klip_tpe_runs/rxj0534/readme_3ann"), \
        "a new directory: the single-annulus run in .../readme would otherwise be resumed"


def test_the_companions_annulus_alone_is_still_one_option_away():
    argv = readme_panel.rxj0534_argv(readme_panel.parse(["--extra-annuli", "0"]))
    assert "--extra-annuli" not in argv


def test_the_three_annuli_touch_and_the_crop_holds_them():
    """The geometry the README run gets, without loading data: the first annulus is the
    companion's +/- 2 FWHM, the next two are as wide and touch it, and the default crop holds
    the outermost with the single-annulus run's margin."""
    rx = pytest.importorskip("run_rxj0534")
    r_px = rx.PLANET[0] / rx.PXSCALE
    e = rx.annulus_edges(r_px, rx.FWHM_PX, extra=2)
    assert len(e) == 4
    assert 0.5 * (e[0] + e[1]) == pytest.approx(r_px), "the first annulus is centered on the companion"
    widths = [e[k + 1] - e[k] for k in range(3)]
    assert widths == pytest.approx([4 * rx.FWHM_PX] * 3)
    assert [round(v * rx.PXSCALE, 3) for v in e] == [0.198, 0.582, 0.966, 1.349]
    assert rx.crop_for(e) == 152 and rx.crop_for(e) * 2 + 1 <= 400, "inside the 400-px frames"
    assert rx.crop_for(rx.annulus_edges(r_px, rx.FWHM_PX)) == 80, "one annulus keeps the old crop"
    assert rx.annulus_edges(r_px, rx.FWHM_PX, ann=(20, 40), extra=1) == [20.0, 40.0, 60.0]


def test_seam_annuli_reach_run_rxj0534_with_a_directory_of_their_own():
    """--seam-annuli goes through to run_rxj0534.py, off unless asked.  Its default directory is not
    the touching annuli's, whose run would otherwise be resumed with its own settings."""
    a = readme_panel.parse(["--extra-annuli", "1", "--seam-annuli"])
    argv = readme_panel.rxj0534_argv(a)
    assert "--seam-annuli" in argv and _opt(argv, "--extra-annuli") == "1"
    assert a.run_dir == os.path.expanduser("~/klip_tpe_runs/rxj0534/readme_seam")
    assert "--seam-annuli" not in readme_panel.rxj0534_argv(readme_panel.parse([]))
    given = readme_panel.parse(["--extra-annuli", "1", "--seam-annuli", "--run-dir", "/tmp/seam_run"])
    assert given.run_dir == "/tmp/seam_run"
    with pytest.raises(SystemExit):
        readme_panel.parse(["--target", "hd95086", "--seam-annuli"])


def test_a_seam_annulus_keeps_every_tile_edge_a_fwhm_inside_the_zones():
    """The geometry --seam-annuli gives, without loading data: one annulus per boundary, from the
    center of the annulus inside to the center of the one outside.  The runner numbers the three by
    center, and the stitch's tile edges lie a quarter of the 4-FWHM width, 1 FWHM, inside both zones
    that meet there, so no KLIP zone edge reaches the stitched images."""
    rx = pytest.importorskip("run_rxj0534")
    from klip_tpe import RunConfig

    r_px = rx.PLANET[0] / rx.PXSCALE
    e = rx.annulus_edges(r_px, rx.FWHM_PX, extra=1)
    seams = rx.seam_annuli(e)
    assert seams == [pytest.approx([r_px, r_px + 4 * rx.FWHM_PX])]
    assert [round(v * rx.PXSCALE, 3) for v in seams[0]] == [0.39, 0.774]
    cfg = RunConfig(ann_edges=e, ann_extra=seams)
    zones, tiles = cfg.zones, cfg.tiles
    assert zones == [pytest.approx((e[0], e[1])), pytest.approx(tuple(seams[0])), pytest.approx((e[1], e[2]))]
    for k in range(len(tiles) - 1):
        edge = tiles[k][1]
        assert edge == pytest.approx(tiles[k + 1][0])
        assert zones[k][1] - edge == pytest.approx(rx.FWHM_PX), "1 FWHM inside the inner zone"
        assert edge - zones[k + 1][0] == pytest.approx(rx.FWHM_PX), "1 FWHM inside the outer zone"
    assert tiles[0][0] == pytest.approx(e[0]) and tiles[-1][1] == pytest.approx(e[-1]), "the same span"
    assert tiles[0][0] <= r_px <= tiles[0][1], "the companion stays in the first annulus' tile"
    assert rx.seam_annuli(rx.annulus_edges(r_px, rx.FWHM_PX)) == [], "one annulus has no boundary"
    assert len(rx.seam_annuli(rx.annulus_edges(r_px, rx.FWHM_PX, extra=2))) == 2


def test_hd95086_keeps_its_own_settings():
    a = readme_panel.parse(["--target", "hd95086"])
    assert (a.seed, a.n_init, a.cg_step, a.title) == (3, 200, 1.0, "HD 95086")
    assert a.run_dir == os.path.expanduser("~/klip_tpe_runs/hd95086")
    assert tuple(a.angsep) == (0.1, 2.0) and a.contrast == 3e-6


def test_the_title_replaces_the_run_directory_name(monkeypatch, tmp_path):
    """LiveDisplay(title=...) names the run on every panel and book.  Without it the name is
    the run directory's, as before."""
    import klip_tpe.display as D

    monkeypatch.setattr(D, "annulus_from_runner", lambda runner, recs: types.SimpleNamespace(run_name="readme"))
    monkeypatch.setattr(D, "annulus_from_run", lambda run_dir, ia: types.SimpleNamespace(run_name="readme"))
    fresh = types.SimpleNamespace(history=None, ia=0)
    resumed = types.SimpleNamespace(history=[1, 2], ia=0)              # resumed mid-annulus: read from disk
    for runner in (fresh, resumed):
        titled = D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0, title="RX J0534")
        assert titled._data(runner).run_name == "RX J0534"
        plain = D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0)
        assert plain._data(runner).run_name == "readme"
    assert D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0, title="  ").title is None
