"""The contrast grid's aperture S/N sums exact apertures, as the IDL contrast generators do (the audit of 2026-11-25).

The whole pixels whose centers lie inside an aperture of radius FWHM/2 number 3 to 6 at a 2.3 px FWHM
and 10 to 14 at 4 px, by sub-pixel position, so a constant level in the image moved the S/N.
``ContrastGridConfig.aperture = "whole"`` keeps the old sums for runs saved before.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe import RunConfig
from klip_tpe.contrast_grid import ContrastGridConfig
from klip_tpe.instruments.generic import aperture_sum
from klip_tpe.metrics import _aperture_sum, aperture_mawet_snr, source_xy, star_center


def test_the_sum_is_the_exact_aperture_and_skips_nan():
    rng = np.random.default_rng(4)
    img = rng.normal(size=(41, 43))
    for x, y, r in ((20.0, 20.0, 1.15), (17.3, 22.8, 2.0), (25.5, 12.5, 3.2), (1.2, 39.7, 2.0)):
        assert _aperture_sum(img, x, y, r, exact=True) == pytest.approx(aperture_sum(img, x, y, r), rel=1e-12,
                                                                        abs=1e-12)
    flat = np.ones((41, 43))
    for x, y, r in ((20.0, 20.0, 1.15), (17.3, 22.8, 2.0), (25.37, 12.61, 3.2)):
        assert _aperture_sum(flat, x, y, r, exact=True) == pytest.approx(np.pi * r * r, rel=1e-12)
    holed = flat.copy()
    holed[20, 20] = np.nan
    assert _aperture_sum(holed, 20.0, 20.0, 2.0, exact=True) == pytest.approx(np.pi * 4.0 - 1.0)
    assert np.isnan(_aperture_sum(np.full((9, 9), np.nan), 4.0, 4.0, 2.0, exact=True))
    assert np.isnan(_aperture_sum(flat, -30.0, 5.0, 2.0, exact=True))
    # the whole pixels, as before: their count moves with the sub-pixel position
    counts = {_aperture_sum(flat, 20.0 + dx, 20.0 + dy, 1.15) for dx in np.linspace(0, 0.9, 10)
              for dy in np.linspace(0, 0.9, 10)}
    assert {4.0, 5.0, 6.0} <= counts                       # where the exact area is 4.15 px^2


@pytest.mark.parametrize("fwhm", [2.3, 4.0])
def test_a_constant_level_leaves_the_aperture_snr_as_it_was(fwhm):
    rng = np.random.default_rng(8)
    n, px = 141, 0.05
    img = rng.normal(size=(n, n))
    cx, cy = star_center((n, n))
    rho, pa = 30.0 * px, 77.0
    (x,), (y,) = source_xy([rho], [pa], px, cx, cy, "pa")
    yy, xx = np.mgrid[0:n, 0:n]
    img += 8.0 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (fwhm / 2.3548) ** 2)
    s0 = aperture_mawet_snr(img, [rho], [pa], px, fwhm)[0]
    for level in (3.0, -12.0):
        assert aperture_mawet_snr(img + level, [rho], [pa], px, fwhm)[0] == pytest.approx(s0, rel=1e-9)
    assert aperture_mawet_snr(7.0 * img, [rho], [pa], px, fwhm)[0] == pytest.approx(s0, rel=1e-9)
    # with whole pixels a level of 3 times the pixel rms moves it
    w0 = aperture_mawet_snr(img, [rho], [pa], px, fwhm, exact=False)[0]
    assert aperture_mawet_snr(img + 3.0, [rho], [pa], px, fwhm, exact=False)[0] != pytest.approx(w0, rel=1e-3)


def test_a_run_sums_exact_apertures_and_an_old_run_whole_pixels():
    from klip_tpe import cli
    assert ContrastGridConfig().aperture == "exact"
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    assert RunConfig.from_dict(d).contrast_grid.aperture == "exact"
    d["contrast_grid"].pop("aperture")
    assert RunConfig.from_dict(d).contrast_grid.aperture == "whole"
    d.pop("contrast_grid")
    old = RunConfig.from_dict(d).contrast_grid
    assert old.mode == "off" and old.aperture == "whole"
    p = cli._build_parser()
    assert cli._contrast_grid_config(p.parse_args(["near"])).aperture == "exact"
    assert cli._contrast_grid_config(p.parse_args(["near", "--cg-aperture", "whole"])).aperture == "whole"
