"""The ALES release layout: nights x wavelength groups as partitions.

What must hold for a data set in the layout of ``datasets.fetch("lbti_ales_mwc758")``:

* ``ales.load_release`` makes one partition per night and group, ``<label>_<group>``, the
  nights labelled ``n1``, ``n2``, ... in date order, each partition with its night's angles
  (plus the 180 deg of the header's convention, once), its night's template and wavelength,
  and the night and date in its provenance.
* Nights can be picked by date or by label, a night loaded alone keeps its label, and the
  groups can be picked the same way on every night.
* Nights whose group grids differ are refused, since ``g07`` would not mean one band.
* An SDI pool across nights is refused, since it would lend another night's frames.
* The space is one shared block plus a flag per partition.
* The data set entry has every role, and ``--ales-dataset`` / ``--ales-nights`` reach the
  loader through the command line.
* A tiny search on two synthetic nights runs end to end (slow).
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, cli, datasets
from klip_tpe.instruments import ales

QUIET = (lambda s: None)
N = 21
NG = 3
NIGHTS = {"20191113": 12, "20191114": 16}
LO, HI = 2.90, 4.15


def _release(tmp_path, nights=NIGHTS, grid_hi=None):
    """A synthetic release: per night a meta file and NG group cubes of a star with noise."""
    rng = np.random.default_rng(3)
    edges = np.exp(np.linspace(np.log(LO), np.log(HI), NG + 1))
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    files = {}
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for k, (date, nf) in enumerate(nights.items()):
        pa = np.linspace(-30.0 - 5 * k, 30.0 + 5 * k, nf)
        waves = np.linspace(2.85 + 0.01 * k, 4.2, 12)
        rows = []
        tpl = np.zeros((NG, N, N), np.float32)
        for g in range(NG):
            ch = np.flatnonzero((waves >= edges[g]) & (waves < edges[g + 1]))
            lam = float(waves[ch].mean())
            sg = 1.03 * (lam * 1e-6 / 8.4) * 206265.0 / 0.0345 / 2.3548
            star = 2000.0 * np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2) / sg ** 2)
            tpl[g] = star
            cube = (star[None] + 20.0 * rng.standard_normal((nf, N, N))).astype(np.float32)
            fn = d / f"syn_{date}_g{g:02d}.fits"
            fits.writeto(fn, cube, overwrite=True)
            files[f"g{g:02d}_{date}"] = str(fn)
            rows.append((f"g{g:02d}", edges[g], edges[g + 1], lam, ch.size, int(ch[0]), int(ch[-1])))
        h = fits.Header()
        h["PIXSCALE"] = 0.0345
        h["ANGCONV"] = "LBT_PARA + 180"
        h["EXPTIME"] = 2.4588
        h["GRIDLO"] = LO
        h["GRIDHI"] = HI if (grid_hi is None or k == 0) else grid_hi
        h["NGROUP"] = NG
        cols = [fits.Column("NAME", "A4", array=[r[0] for r in rows]),
                fits.Column("LAM_LO", "D", array=[r[1] for r in rows]),
                fits.Column("LAM_HI", "D", array=[r[2] for r in rows]),
                fits.Column("LAM_MEAN", "D", array=[r[3] for r in rows]),
                fits.Column("NCHAN", "J", array=[r[4] for r in rows]),
                fits.Column("CHAN0", "J", array=[r[5] for r in rows]),
                fits.Column("CHAN1", "J", array=[r[6] for r in rows])]
        fr = [fits.Column("UT", "A16", array=[f"08:{i:02d}:00" for i in range(nf)]),
              fits.Column("LBT_PARA", "D", array=pa)]
        hl = fits.HDUList([fits.PrimaryHDU(tpl, h), fits.ImageHDU(pa, name="ANGLES"),
                           fits.BinTableHDU.from_columns(fr, name="FRAMES"),
                           fits.BinTableHDU.from_columns(cols, name="GROUPS"), fits.ImageHDU(waves, name="WAVES")])
        fn = d / f"syn_{date}_meta.fits"
        hl.writeto(fn, overwrite=True)
        files[f"meta_{date}"] = str(fn)
    return files


def test_the_release_loads_as_nights_by_groups(tmp_path):
    files = _release(tmp_path)
    assert ales.release_nights(files) == ["20191113", "20191114"]
    ds, lam, fw, table = ales.load_release(files, log=QUIET)
    names = [f"n{k}_g{g:02d}" for k in (1, 2) for g in range(NG)]
    assert list(ds) == names and [r["name"] for r in table] == names
    for r in table:
        d = ds[r["name"]]
        night = "20191113" if r["night"] == "n1" else "20191114"
        pa = fits.getdata(files[f"meta_{night}"], "ANGLES")
        np.testing.assert_allclose(d.angles, pa + 180.0)
        assert d.cube.shape == (NIGHTS[night], N, N)
        assert lam[r["name"]] == pytest.approx(r["lam_mean"] * 1e-6)
        assert r["date"] == night and r["group"] == r["name"][3:]
        pv = d.meta["provenance"]
        assert (pv["night"], pv["date"], pv["pxscale"], pv["angle_offset_deg"]) == (r["night"], night, 0.0345, 180.0)
        assert d.meta["psf_center"] == ((N - 1) / 2.0, (N - 1) / 2.0)
        assert d.meta["psf"].shape == (N, N)
        assert d.texp == pytest.approx(2.4588 * NIGHTS[night])
    assert table[0]["lam_mean"] != table[NG]["lam_mean"], "each night keeps its own channels' mean"
    assert table[0]["lam_lo"] == table[NG]["lam_lo"], "but the same group edges"


def test_nights_and_groups_can_be_picked(tmp_path):
    files = _release(tmp_path)
    assert list(ales.load_release(files, nights=["n2"], log=QUIET)[0]) == [f"n2_g{g:02d}" for g in range(NG)]
    assert list(ales.load_release(files, nights=["20191113"], log=QUIET)[0]) == [f"n1_g{g:02d}" for g in range(NG)]
    assert list(ales.load_release(files, only=["g01"], log=QUIET)[0]) == ["n1_g01", "n2_g01"]
    assert list(ales.load_release(files, labels={"20191113": "nov13"}, only=["g00"], log=QUIET)[0]) == \
        ["nov13_g00", "n2_g00"]
    with pytest.raises(ValueError, match="not in this data set"):
        ales.load_release(files, nights=["20191115"], log=QUIET)
    with pytest.raises(ValueError, match="none of the groups"):
        ales.load_release(files, only=["g09"], log=QUIET)


def test_the_group_grids_must_match(tmp_path):
    files = _release(tmp_path, grid_hi=4.2)
    with pytest.raises(ValueError, match="group grid"):
        ales.load_release(files, log=QUIET)


def test_sdi_across_nights_is_refused(tmp_path):
    files = _release(tmp_path)
    ds, lam, fw, table = ales.load_release(files, log=QUIET)
    with pytest.raises(ValueError, match="nights"):
        ales.make_reducer(ds, lam, fw, max_workers=1, table=table, sdi=True, log=QUIET)
    ds, lam, fw, table = ales.load_release(files, nights=["n1"], log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, sdi=True, log=QUIET)
    assert red.provenance["reference_library"]["sdi"]["n_groups"] == NG


def test_one_block_and_a_flag_per_partition(tmp_path):
    files = _release(tmp_path)
    ds, lam, fw, table = ales.load_release(files, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    space = ales.make_space(red, k_klip_max=6, bin_range=(1, 3))
    assert space.selection == "binary"
    assert [space.params[i].name for i in space.selection_dims] == [f"sel_{n}" for n in red.partitions()]
    block = [p.name for p in space.params if p.role == "reduction"]
    assert "k_klip" in block and not any(n.startswith("k_klip_") for n in block), "one shared block"
    assert red.representative in red.partitions()
    assert [g["name"] for g in red.provenance["groups"]] == red.partitions()


def test_the_data_set_entry_and_the_command_line(tmp_path, monkeypatch):
    entry = datasets.DATASETS["lbti_ales_mwc758"]
    assert len(entry) == 42 and datasets._SOURCE["lbti_ales_mwc758"] == "kt"
    assert sorted(k for k in entry if k.startswith("meta_")) == ["meta_20191113", "meta_20191114"]
    files = _release(tmp_path)
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "data"))
    monkeypatch.setitem(datasets.DATASETS, "syn_ales", {k: os.path.basename(v) for k, v in files.items()})
    monkeypatch.setitem(datasets._SOURCE, "syn_ales", "kt")

    def no_download(url, filename):
        raise AssertionError(f"downloaded {url}: the files were there")

    monkeypatch.setattr(urllib.request, "urlretrieve", no_download)
    base = ["near", "--instrument", "ales", "--ales-dataset", "syn_ales", "--workers", "1", "--known", "0.3", "240"]
    a = cli._build_parser().parse_args(base + ["--ales-nights", "n2"])
    red, obj, samp, guard, thr = cli._near_objects(a, log=QUIET)
    assert red.partitions() == [f"n2_g{g:02d}" for g in range(NG)]
    assert obj.metric.known == [(0.3, 240.0)]
    a = cli._build_parser().parse_args(base)
    red, obj, samp, guard, thr = cli._near_objects(a, log=QUIET)
    assert len(red.partitions()) == 2 * NG and red.pxscale == 0.0345
    with pytest.raises(SystemExit, match="reference star"):
        cli._near_objects(cli._build_parser().parse_args(base + ["--ales-ref", "HIP1"]), log=QUIET)
    with pytest.raises(SystemExit, match="unknown data set"):
        cli._near_objects(cli._build_parser().parse_args(
            ["near", "--instrument", "ales", "--ales-dataset", "nope", "--workers", "1"]), log=QUIET)
    with pytest.raises(SystemExit, match="--ales-dataset"):
        cli._near_objects(cli._build_parser().parse_args(["near", "--instrument", "ales", "--workers", "1"]),
                          log=QUIET)
    assert {"ales_dataset", "ales_nights"} <= cli._explicit_dests(base + ["--ales-nights", "n1"])


@pytest.mark.slow
def test_a_tiny_search_on_two_synthetic_nights(tmp_path):
    files = _release(tmp_path)
    ds, lam, fw, table = ales.load_release(files, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    space = ales.make_space(red, defaults={"bin": 1, "k_klip": 3, "filter": 0}, k_klip_max=6, bin_range=(1, 2))
    space.project = ales.make_guard(red, max_excluded=2, n_min_ref=3, ref_frac=0.5, k_max=6)
    obj, samp = ales.default_config(red)
    cfg = RunConfig(ann_edges=[4, 9], n_iter=5, n_init=2, seed=2, fm_curve=False, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[2e-2]))
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET)
    res = r.run()
    assert len(res) == 1 and res[0].validated
    setup = json.load(open(tmp_path / "run" / "run_setup.json"))
    groups = setup["reducer"]["provenance"]["groups"]
    assert [(g["night"], g["date"]) for g in groups[:1]] == [("n1", "20191113")]
    recs = [json.loads(l) for l in open(tmp_path / "run" / "results.jsonl")]
    assert recs and all(len(rec["meta"]["selected"]) >= 2 * NG - 2 for rec in recs), "at most two partitions off"
