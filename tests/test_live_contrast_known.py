"""The live contrast preview leaves a known companion out of its noise rings, as the run's own
curve does.  Without that, HD 95086 b put a spike in tutorial 02's preview at its separation.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.display import _live_curve_info, annulus_from_records
from klip_tpe.metrics import star_center

PX, FWHM = 0.05, 4.0
RHO, PA = 1.5, 145.0                       # the companion: 30 px out


def _annulus(known):
    space = {"params": [{"name": "k_klip", "base": "k_klip", "lo": 1, "hi": 30, "kind": "int"}]}
    cfg = {"ann_edges": [12, 48], "n_iter": 2, "n_init": 1}
    recs = [{"annulus": 0, "index": i, "phase": "warmup", "x": [5], "config": {"params": {"k_klip": 5}},
             "sources": [(0.9, 30.0, 1e-4), (1.2, 150.0, 1e-4), (2.0, 270.0, 1e-4)], "score": 5.0,
             "raw_score": 5.0, "per_source": [6.0, 5.0, 4.0], "contrast": 1e-4, "wall_s": 1.0, "meta": {}}
            for i in range(2)]
    return annulus_from_records(recs, space, cfg, 0, PX, FWHM, known=known)


def test_the_live_preview_leaves_a_known_companion_out_of_its_noise_rings():
    img = np.random.default_rng(3).standard_normal((121, 121))
    cx, cy = star_center(img.shape)
    th, rr = np.deg2rad(PA), RHO / PX
    x, y = cx - rr * np.sin(th), cy + rr * np.cos(th)          # products.noise_profile's convention
    yy, xx = np.mgrid[0:121, 0:121]
    img += 300.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (FWHM / 2.355) ** 2))
    masked, why = _live_curve_info(img, _annulus([(RHO, PA)]), 1)
    assert why is None
    open_, _ = _live_curve_info(img, _annulus([]), 1)
    r = np.asarray(masked["r_as"], float)
    at, far = int(np.argmin(np.abs(r - RHO))), int(np.argmin(np.abs(r - 1.0)))
    assert open_["curve"][at] > 3 * masked["curve"][at], "the companion no longer raises its own ring"
    assert masked["curve"][far] == pytest.approx(open_["curve"][far]), "rings away from it are unchanged"
