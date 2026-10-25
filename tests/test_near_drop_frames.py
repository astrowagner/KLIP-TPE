"""``--drop-frames``: frames left out of a NEAR night at load, and the run's record of them.

What must hold:

* ``near.load_night(drop_frames=[i, ...])`` leaves those frames out of the cube, the angles and
  the frame tags, by their index in the night's cube counted from 0, and scales ``texp`` by the
  frames kept.  Without a list the night loads as before, to the bit.
* An index past the cube, and a list that would leave no frame, are refused.  An index that
  ``--max-frames`` leaves out anyway drops nothing.
* ``near.make_reducer(drop_frames={"n4": [551]})`` drops them before the sources are
  subtracted, refuses a night it does not load, and records the list in its ``data_edits``.
* A run records the list, a resume with the same frames goes on and one with other frames is
  refused.  On the command line a resume takes the list from the checkpoint when it is not
  typed and refuses a typed list that differs, and ``rereduce`` takes the typed list (``none``
  for none) or else the run's.
"""
from __future__ import annotations

import argparse
import inspect
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import cli
from klip_tpe.instruments import near
from klip_tpe.metrics import Source
from klip_tpe.runner import Runner, _core_edits, _describe_edits, _dropped_text, load_checkpoint

from test_near_data_edits import EDIT, _psf, _red, _run

NF, FULL = 16, 152
TEXP = 800.0


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A two-night NEAR tree with frame tags and exposure times.  IDL save files cannot be
    written here, so ``_readsav`` reads an .npz beside the name."""
    rng = np.random.default_rng(7)
    data = {}
    for seq in (1, 2):
        d = tmp_path / f"n{seq}"
        d.mkdir()
        cube = (100.0 + 5.0 * rng.standard_normal((NF, FULL, FULL))).astype(np.float32)
        fits.writeto(d / "AB_cube_skysub_cen_clean.fits", cube)
        angles = np.linspace(-30.0, 30.0, NF) + seq
        np.savez(d / "AB_parang_clean.npz", angles=angles)
        tags = {"corrs": 0.99 - 0.001 * np.arange(NF), "noises": 1.0 + 0.01 * np.arange(NF),
                "coronoise": 2.0 + 0.02 * np.arange(NF)}
        np.savez(d / "AB_frametags.npz", **tags)
        np.savez(d / "texp.npz", texp=np.array([TEXP * seq]))
        for name in ("AB_frametags.sav", "texp.sav"):          # the loader looks for these first
            (d / name).write_bytes(b"")
        fits.writeto(d / "PSF_ACenB.fits", _psf())
        data[seq] = (cube, angles, tags)

    def readsav(path):
        z = np.load(os.path.splitext(path)[0] + ".npz")
        return {k: z[k] for k in z.files}
    monkeypatch.setattr(near, "_readsav", readsav)
    return str(tmp_path), data


def _crop(cube):
    return cube[:, 1:151, 1:151]


# ---------------------------------------------------------------- at load
def test_dropped_frames_leave_the_cube_the_angles_and_the_tags(root):
    r, data = root
    cube, angles, tags = data[1]
    ds = near.load_night(r, 1, drop_frames=[3, 9])
    keep = [i for i in range(NF) if i not in (3, 9)]
    assert ds.nframes == NF - 2
    assert np.array_equal(ds.cube, _crop(cube)[keep])
    assert np.array_equal(ds.angles, angles[keep])
    for k, v in tags.items():
        assert np.array_equal(ds.tags[k], v[keep])
    assert ds.texp == pytest.approx(TEXP * (NF - 2) / NF)
    assert ds.meta["dropped_frames"] == [3, 9] and list(ds.meta["frame_index"]) == keep
    assert ds.meta["nframes_total"] == NF


def test_without_a_list_the_night_loads_as_before(root):
    r, data = root
    cube, angles, tags = data[1]
    for drop in (None, [], ()):
        ds = near.load_night(r, 1, drop_frames=drop)
        assert np.array_equal(ds.cube.view(np.uint32), _crop(cube).view(np.uint32))
        assert np.array_equal(ds.angles, angles) and ds.texp == TEXP
        assert ds.meta["dropped_frames"] == [] and list(ds.meta["frame_index"]) == list(range(NF))
    # --max-frames, as before: the first frames, texp scaled
    ds = near.load_night(r, 1, frames=slice(0, 10))
    assert np.array_equal(ds.cube, _crop(cube)[:10]) and ds.texp == TEXP * 10 / NF


def test_past_the_cube_or_every_frame_is_refused_and_past_max_frames_drops_nothing(root):
    r, data = root
    cube = data[1][0]
    with pytest.raises(near.DropFramesError, match="frame 16 is not in the night's cube, which holds frames 0-15"):
        near.load_night(r, 1, drop_frames=[16])
    with pytest.raises(near.DropFramesError, match="would leave no frame"):
        near.load_night(r, 1, frames=slice(0, 2), drop_frames=[0, 1])
    a = near.load_night(r, 1, frames=slice(0, 10))
    b = near.load_night(r, 1, frames=slice(0, 10), drop_frames=[12])
    assert np.array_equal(a.cube, b.cube) and a.texp == b.texp and b.meta["dropped_frames"] == []
    c = near.load_night(r, 1, frames=slice(0, 10), drop_frames=[4, 12])
    assert np.array_equal(c.cube, _crop(cube)[[0, 1, 2, 3, 5, 6, 7, 8, 9]])
    assert c.texp == pytest.approx(TEXP * 9 / NF) and c.meta["dropped_frames"] == [4]


# ---------------------------------------------------------------- the reducer
def test_make_reducer_drops_the_frames_records_them_and_refuses_other_nights(root):
    r, data = root
    logs = []
    red = near.make_reducer(r, [1, 2], use_library=False, max_workers=1, log=logs.append,
                            drop_frames={"n1": [3], 2: [0, 15, 0]})
    assert red.data_edits == {"dropped_frames": {"n1": [3], "n2": [0, 15]}}
    assert red.reducers["n1"].data.nframes == NF - 1 and red.reducers["n2"].data.nframes == NF - 2
    assert red.reducers["n1"].data_edits == {"dropped_frames": {"n1": [3]}}
    assert red.reducers["n2"].data_edits == {"dropped_frames": {"n2": [0, 15]}}
    d = red.describe()
    assert d["data_edits"]["dropped_frames"] == {"n1": [3], "n2": [0, 15]}
    assert d["partitions"]["n1"]["nframes"] == NF - 1
    assert any("n1: frame 3 left out at load (drop_frames), 15 of 16 kept" in s for s in logs)
    assert any("n2: frames 0 15 left out at load (drop_frames), 14 of 16 kept" in s for s in logs)
    assert np.array_equal(red.reducers["n1"].data.angles, np.delete(data[1][1], 3))
    # nothing asked: nothing recorded, the frames as before
    red0 = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, drop_frames={})
    assert not getattr(red0, "data_edits", None) and red0.reducers["n1"].data.nframes == NF
    with pytest.raises(near.DropFramesError, match="n3 is not among the nights loaded \\(n1 n2\\)"):
        near.make_reducer(r, [1, 2], use_library=False, max_workers=1, log=lambda s: None, drop_frames={"n3": [1]})
    ds = near.load_night(r, 1)
    with pytest.raises(near.DropFramesError, match="loaded already"):
        near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, datasets={1: ds},
                          drop_frames={"n1": [3]})
    for bad, match in (({"x": [1]}, "is not a night"), ({"n1": [-1]}, "count from 0"),
                       ({"n1": [1.5]}, "whole number")):
        with pytest.raises(near.DropFramesError, match=match):
            near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, drop_frames=bad)


def test_the_frames_go_before_the_sources_are_subtracted(root):
    r, _ = root
    line = [Source(0.30 + 0.12 * i, 13.0 if i % 2 else 193.0, 1.0 + 0.2 * i) for i in range(4)]
    a = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, subtract_sources=line)
    b = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, subtract_sources=line,
                          drop_frames={"n1": [5]})
    ca, cb = a.reducers["n1"].data.cube, b.reducers["n1"].data.cube
    assert np.array_equal(np.delete(ca, 5, axis=0).view(np.uint32), cb.view(np.uint32))
    assert set(b.data_edits) == {"dropped_frames", "subtracted_sources"}


# ---------------------------------------------------------------- the record
def test_the_record_describes_and_compares_the_dropped_frames():
    assert _core_edits({"dropped_frames": {"n4": [552, 551, 551], "n6": []}}) == {"dropped_frames": {"n4": [551, 552]}}
    assert _core_edits({"dropped_frames": {}}) == {}
    assert _dropped_text({"n10": [1], "n4": [551]}) == "n4 frame 551, n10 frame 1 dropped"
    assert _dropped_text({"n4": [550, 551]}) == "n4 frames 550 551 dropped"
    assert _dropped_text({}) == "no frames dropped"
    assert _describe_edits(_core_edits({"cube_name": "c.fits", "dropped_frames": {"n4": [551]}})) == (
        "cube c.fits, n4 frame 551 dropped")


def test_a_run_records_the_frames_and_a_resume_must_drop_the_same(tmp_path):
    def red(drop):
        x = _red()
        if drop:
            x.data_edits = {"dropped_frames": drop}
        return x
    r, _, obj, samp = _run(red({"n1": [3]}), tmp_path / "run")
    r.run()
    st, _ = load_checkpoint(str(tmp_path / "run"))
    assert st["config"]["data_edits"]["dropped_frames"] == {"n1": [3]}
    Runner.resume(str(tmp_path / "run"), red({"n1": [3]}), obj, samp, log=lambda s: None)
    for other in (red(None), red({"n1": [4]}), red({"n1": [3, 4]})):
        with pytest.raises(ValueError, match="cannot change the frames"):
            Runner.resume(str(tmp_path / "run"), other, obj, samp, log=lambda s: None)
    # a run made on all its frames refuses a reducer that drops some
    r2, *_ = _run(red(None), tmp_path / "run2")
    r2.run()
    with pytest.raises(ValueError, match="n1 frame 3 dropped"):
        Runner.resume(str(tmp_path / "run2"), red({"n1": [3]}), obj, samp, log=lambda s: None)


# ---------------------------------------------------------------- the command line
def test_the_option_parses():
    assert cli._parse_drop_frames(["n4:551", "4:552,553", "N6:10,10"]) == {"n4": [551, 552, 553], "n6": [10]}
    assert cli._parse_drop_frames(["none"]) == {} and cli._parse_drop_frames(None) == {}
    for bad, match in (("n4", "NIGHT:INDEX"), ("x:1", "NIGHT:INDEX"), ("n4:a", "whole numbers"),
                       ("n4:", "no frame index"), ("n4:-1", "count from 0")):
        with pytest.raises(SystemExit, match=match):
            cli._parse_drop_frames([bad])
    with pytest.raises(SystemExit, match="cannot go with a list"):
        cli._parse_drop_frames(["none", "n4:551"])
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--drop-frames", "n4:551", "--drop-frames", "6:1,2"])
    assert a.drop_frames == ["n4:551", "6:1,2"] and cli._drop_frames(a) == {"n4": [551], "n6": [1, 2]}
    assert cli._build_parser().parse_args(["near", "--root", "/r"]).drop_frames is None
    assert "drop_frames" in cli._explicit_dests(["resume", "--drop-frames", "n4:551"])
    assert '"drop_frames"' in inspect.getsource(cli._fill_from_registry)
    assert cli._drop_argv({"n6": [1, 2], "n4": [551]}) == ["n4:551", "n6:1,2"]
    for sub in ("near", "generic", "resume", "extend", "fit-chain", "rereduce"):
        argv = [sub, "--drop-frames", "n4:551"] + (["--n-iter", "5"] if sub == "extend" else []) + (
            ["--run-dir", "x"] if sub in ("extend",) else [])
        assert cli._build_parser().parse_args(argv).drop_frames == ["n4:551"]


def test_other_instruments_refuse_it():
    cli._no_drop_frames(argparse.Namespace(drop_frames=None), "--instrument generic")
    cli._no_drop_frames(argparse.Namespace(drop_frames=["none"]), "--instrument generic")
    with pytest.raises(SystemExit, match="--instrument ales does not take it"):
        cli._no_drop_frames(argparse.Namespace(drop_frames=["n4:551"]), "--instrument ales")


def _ns(run_dir, argv, drop=None):
    return argparse.Namespace(run_dir=str(run_dir), cube_name=None, subtract_sources=None, subtract_line=None,
                              inject_sources=None, drop_frames=drop, _argv=argv)


def test_resume_takes_the_list_from_the_checkpoint_and_refuses_another(tmp_path, capsys):
    red = _red(EDIT)
    red.data_edits = dict(red.data_edits, dropped_frames={"n1": [3, 7]})
    r, *_ = _run(red, tmp_path / "run")
    r.run()
    base = ["resume", "--run-dir", str(tmp_path / "run")]
    # not typed: the checkpoint's list, over anything the run record restored
    a = _ns(tmp_path / "run", base, drop=["n1:9"])
    cli._data_edits_for_run(a)
    assert a.drop_frames == ["n1:3,7"] and cli._drop_frames(a) == {"n1": [3, 7]}
    assert "--drop-frames n1:3,7, as run was started (its checkpoint)" in capsys.readouterr().out
    # typed and the same, in any spelling: fine
    for same in (["n1:3,7"], ["1:7", "n1:3"]):
        a = _ns(tmp_path / "run", base + [x for v in same for x in ("--drop-frames", v)], drop=same)
        cli._data_edits_for_run(a)
        assert a.drop_frames == same
    # typed and different: refused
    for other in (["n1:3"], ["none"], ["n1:3,7", "n2:1"]):
        a = _ns(tmp_path / "run", base + [x for v in other for x in ("--drop-frames", v)], drop=other)
        with pytest.raises(SystemExit, match="cannot change the frames"):
            cli._data_edits_for_run(a)
    # a run made on all its frames: nothing restored, and a typed list refused
    r2, *_ = _run(_red(), tmp_path / "run2")
    r2.run()
    base2 = ["resume", "--run-dir", str(tmp_path / "run2")]
    a = _ns(tmp_path / "run2", base2)
    cli._data_edits_for_run(a)
    assert a.drop_frames is None
    a = _ns(tmp_path / "run2", base2 + ["--drop-frames", "none"], drop=["none"])
    cli._data_edits_for_run(a)
    a = _ns(tmp_path / "run2", base2 + ["--drop-frames", "n1:3"], drop=["n1:3"])
    with pytest.raises(SystemExit, match="was made with no frames dropped, and this command asks for n1 frame 3"):
        cli._data_edits_for_run(a)


def test_rereduce_takes_the_typed_list_or_the_runs(tmp_path, capsys):
    red = _red()
    red.data_edits = {"dropped_frames": {"n1": [3]}}
    r, *_ = _run(red, tmp_path / "run")
    r.run()
    base = ["rereduce", "--run-dir", str(tmp_path / "run")]
    a = _ns(tmp_path / "run", base, drop=["n1:9"])            # what the run record may have put there
    cli._data_edits_for_rereduce(a)
    assert a.drop_frames == ["n1:3"]
    assert "--drop-frames n1:3, as run was made (its checkpoint)" in capsys.readouterr().out
    a = _ns(tmp_path / "run", base + ["--drop-frames", "n1:4,5"], drop=["n1:4,5"])
    cli._data_edits_for_rereduce(a)                            # nothing is refused: the run is only read
    assert cli._drop_frames(a) == {"n1": [4, 5]}
    a = _ns(tmp_path / "run", base + ["--drop-frames", "none"], drop=["none"])
    cli._data_edits_for_rereduce(a)
    assert cli._drop_frames(a) == {}
    # a run made on all its frames, nothing typed: nothing dropped
    r2, *_ = _run(_red(), tmp_path / "run2")
    r2.run()
    a = _ns(tmp_path / "run2", ["rereduce", "--run-dir", str(tmp_path / "run2")], drop=["n1:9"])
    cli._data_edits_for_rereduce(a)
    assert a.drop_frames is None


@pytest.mark.slow
def test_near_and_resume_on_the_command_line_keep_the_dropped_frames(root, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))       # keep the run record here
    r, _ = root
    run = tmp_path / "run"
    common = ["--root", r, "--nights", "1", "--no-library", "--workers", "1", "--no-display"]
    cli.main(["near", *common, "--run-dir", str(run), "--ann-edges", "12", "30", "--n-iter", "2", "--n-init", "2",
              "--n-top", "1", "--n-valid", "1", "--use-contrast", "0.5", "--seed", "1", "--no-fm-curve",
              "--no-param-verify", "--no-setup-files", "--drop-frames", "n1:3,7"])
    st, _ = load_checkpoint(str(run))
    assert st["config"]["data_edits"]["dropped_frames"] == {"n1": [3, 7]}
    out = capsys.readouterr().out
    assert "loaded n1: 14 frames" in out and "n1: frames 3 7 left out at load" in out
    # a resume that does not name them drops the same frames (the run is finished, so it only
    # reloads and re-checks), and one that names others is refused before anything is loaded
    cli.main(["resume", *common, "--run-dir", str(run)])
    out = capsys.readouterr().out
    assert "--drop-frames n1:3,7, as run was started" in out and "loaded n1: 14 frames" in out
    with pytest.raises(SystemExit, match="cannot change the frames"):
        cli.main(["resume", *common, "--run-dir", str(run), "--drop-frames", "n1:3"])
    assert "loaded n1" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--drop-frames: n1: frame 99 is not in the night's cube"):
        cli.main(["near", *common, "--run-dir", str(tmp_path / "run2"), "--ann-edges", "12", "30", "--n-iter", "2",
                  "--drop-frames", "n1:99"])
