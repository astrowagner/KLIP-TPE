"""The per-tile measure on the stitch with the tile put in (NEAR2 note, 2026-11-04).

An annulus' image is NaN outside its zone, and the matched filter and the noise ring take zeros
there.  Where the next annulus in has residuals several times the zone's, the image alone
leaves noise out of the ring and the radial band.  Inside about 1.3" of NEAR2's runs it read
1.3 to 6 times the stitch's S/N.  What must hold:

* ``put_tile_in`` takes the annulus' finite pixels on its tile and the stitch's everywhere else,
  and ``put_tile_in_stack`` does so for each partition by its id, leaving a partition the
  stitch lacks as the annulus has it.
* With residuals in the next annulus in ten times the zone's, a source just inside the tile
  reads well above the stitch's S/N on the annulus image alone.  On the stitch with the tile put
  in it reads the stitch's S/N, to the bit, where the tile takes no blend.
* Where the stitch blends a tile's pixels, the per-tile measure reads the annulus' own pixels:
  it is the measure of the stitch with the tile put in by hand, not the stitch's.
* The reports name the image, and the stitched report's closing lines say the two S/N differ
  only where the stitch blends or patches a tile.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

from klip_tpe import verify as V
from klip_tpe.metrics import source_xy, star_center
from klip_tpe.stitch import put_tile_in, put_tile_in_stack, tile_index

PX, FW, N = 0.05, 4.0, 81
TILES = [(6.0, 16.0), (16.0, 30.0)]
ZONES = [(6.0, 18.0), (14.0, 30.0)]          # each zone padded 2 px where the tiles meet
CX, CY = star_center((N, N))
YY, XX = np.mgrid[0:N, 0:N]
RR = np.hypot(XX - CX, YY - CY)
IDX = tile_index((N, N), TILES)
SRC = [(17 * PX, pa, 1e-3) for pa in (30.0, 120.0, 210.0, 300.0)]   # 1 px inside the second tile
CAND = [(23 * PX, 75.0)]


def _blob(s, amp):
    x, y = source_xy([s[0]], [s[1]], PX, CX, CY)
    sg = FW / 2.3548
    return amp * np.exp(-0.5 * ((XX - x[0]) ** 2 + (YY - y[0]) ** 2) / sg ** 2)


def _scene(seed=0, inner=10.0, amp=6.0):
    """Two annuli: residuals ``inner`` times as large in the first, a companion and four
    sources in the second tile, every source in both injected images."""
    rng = np.random.default_rng(seed)
    R = []
    for a in (inner, 1.0):
        g = ndimage.gaussian_filter(rng.standard_normal((N, N)), FW / 2.3548)
        R.append(a * g / g.std())
    comp = _blob(CAND[0], 8.0)
    srcs = sum(_blob(s, amp) for s in SRC)
    clean = [np.where((RR >= z0) & (RR < z1), R[k] + comp, np.nan) for k, (z0, z1) in enumerate(ZONES)]
    inj = [c + srcs for c in clean]
    st = np.where(IDX == 0, R[0], np.where(IDX == 1, R[1], np.nan)) + comp
    annuli = [{"annulus": k, "tile": TILES[k], "clean": clean[k], "inj": inj[k], "inj_sources": list(SRC)}
              for k in range(2)]
    return annuli, st, st + srcs, R


def _snr(rows):
    return np.array([r["snr"] for r in rows if r["kind"] == "injection"])


def test_a_tile_goes_into_the_stitch_and_each_partition_by_its_id():
    base, own = np.full((5, 5), 1.0), np.full((5, 5), 2.0)
    own[0, 0] = np.nan
    tile = np.zeros((5, 5), bool)
    tile[:2] = True
    out = put_tile_in(base, own, tile)
    assert np.all(out[2:] == 1.0) and np.all(out[:2][np.isfinite(own[:2])] == 2.0)
    assert out[0, 0] == 1.0, "a pixel the annulus does not have keeps the stitch's"
    assert put_tile_in(None, own, tile) is not None and np.array_equal(put_tile_in(None, own, tile), own,
                                                                       equal_nan=True)
    assert put_tile_in(base, None, tile) is None
    assert np.array_equal(put_tile_in(np.ones((4, 4)), own, tile), own, equal_nan=True), "another shape: own alone"
    cube = np.stack([np.full((5, 5), v) for v in (10.0, 20.0, 30.0)])
    stack = np.stack([np.full((5, 5), v) for v in (3.0, 1.0, 4.0)])
    got = put_tile_in_stack(cube, ["n1", "n2", "n3"], stack, ["n3", "n1", "n4"], tile)
    assert got.shape == (3, 5, 5)
    assert np.all(got[0][2:] == 30.0) and np.all(got[0][:2] == 3.0), "n3 into the stitch's n3"
    assert np.all(got[1][2:] == 10.0) and np.all(got[1][:2] == 1.0), "n1 into the stitch's n1"
    assert np.all(got[2] == 4.0), "n4, which the stitch lacks, as the annulus has it"
    assert np.array_equal(put_tile_in_stack(cube, None, stack, ["n1", "n2", "n3"], tile), stack)


def test_the_image_alone_reads_above_the_stitch_and_the_stitch_with_the_tile_does_not():
    annuli, st, st_inj, _ = _scene()
    alone, _ = V.verify_tiles(annuli, [], SRC, FW, PX, n_boot=0)
    rows, maps = V.verify_tiles(annuli, [], SRC, FW, PX, n_boot=0,
                                stitch={"clean": st, "inj": st_inj, "inj_sources": SRC})
    ref, _ = V.verify_injections(st_inj[None], SRC, FW, PX, combined=st_inj, clean=st)
    on_st = np.array([r["snr"] for r in ref])
    assert np.all(on_st > 0) and np.all(np.isfinite(_snr(alone)))
    assert np.median(_snr(alone) / on_st) > 1.5, (_snr(alone), on_st)
    assert np.array_equal(_snr(rows), on_st), "the tile takes no blend: the stitch's S/N"
    inj = [r for r in rows if r["kind"] == "injection"]
    assert all(r["annulus"] == 2 and r["in_stitch"] for r in inj) and maps["limits"]["n_inj"] == len(SRC)
    # residuals like the zone's: the two agree within about 15 % in the median (0.85 here, the
    # stitch's seam joining two fields the filter averages)
    ratio = []
    for seed in range(4):
        a1, s1, si1, _ = _scene(seed=seed, inner=1.0)
        alone1, _ = V.verify_tiles(a1, [], SRC, FW, PX, n_boot=0)
        ref1, _ = V.verify_injections(si1[None], SRC, FW, PX, combined=si1, clean=s1)
        ratio += list(_snr(alone1) / np.array([r["snr"] for r in ref1]))
    assert 0.75 < np.median(ratio) < 1.1, np.median(ratio)


def test_where_the_stitch_blends_a_tile_the_tile_measure_reads_the_annulus_own_pixels():
    annuli, st, st_inj, R = _scene(seed=2, inner=3.0)
    band = (IDX == 1) & (RR < 18.0)                 # a blend over the tile's first 2 px, as a pad
    blend = st.copy()
    blend[band] = 0.5 * (R[0][band] + R[1][band])
    srcs = st_inj - st
    blend_inj = blend + srcs
    rows, _ = V.verify_tiles(annuli, [], SRC, FW, PX, n_boot=0,
                             stitch={"clean": blend, "inj": blend_inj, "inj_sources": SRC})
    hand_c, hand_i = (put_tile_in(b, a, IDX == 1) for b, a in ((blend, annuli[1]["clean"]),
                                                               (blend_inj, annuli[1]["inj"])))
    by_hand, _ = V.verify_injections(hand_i[None], SRC, FW, PX, combined=hand_i, clean=hand_c)
    on_blend, _ = V.verify_injections(blend_inj[None], SRC, FW, PX, combined=blend_inj, clean=blend)
    assert np.array_equal(_snr(rows), np.array([r["snr"] for r in by_hand]))
    assert np.max(np.abs(_snr(rows) / np.array([r["snr"] for r in on_blend]) - 1.0)) > 0.01


def test_the_reports_name_the_image():
    annuli, st, st_inj, _ = _scene(seed=3)
    rows, maps = V.verify_tiles(annuli, CAND, SRC, FW, PX, known=CAND, n_boot=0,
                                stitch={"clean": st, "inj": st_inj, "inj_sources": SRC})
    text = V.write_verify_table(os.devnull, rows, maps["limits"])
    assert "measured on the stitch with annulus 2's tile put in, tile 16.00-30.00 px" in text
    assert "the annulus whose tile, put into the stitch, measured it" in text
    assert "PATH A = the stitched image with the tile of the annulus that holds the position put in" in text
    lines = V.tile_comparison_lines([dict(r, snr=1.0) for r in rows], rows)
    assert all(ln.startswith("#") for ln in lines) and len(lines) == 6 + len(rows)
    assert any("differ only where the stitch blends or patches that tile" in ln for ln in lines)
