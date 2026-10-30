"""The finished annulus's frame: its contrast panel shows validation's curve and the KLIP-FM
cross-check, and not the best evaluation's own preview.

The preview is that evaluation's few injections.  Validation's curve replaces it, and the IDL
panel colors the two alike, so leaving the preview in drew two solid curves that disagree
for no reason a reader could see.
"""
from __future__ import annotations

import types

from klip_tpe.display import LiveDisplay


def test_the_finished_annulus_frame_drops_the_best_evals_preview(tmp_path):
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0)
    cc = {"r_as": [0.3, 0.5], "curve": [1e-4, 5e-5]}
    disp._best_curve = cc
    res = types.SimpleNamespace(annulus=0, contrast_curve=cc, fm_curve=cc)
    runner = types.SimpleNamespace(results=[res], best_images={})
    ad = types.SimpleNamespace(annulus=0)
    running = [c[0] for c in disp._curves(runner, ad, None)]
    done = [c[0] for c in disp._curves(runner, ad, None, done=True)]
    assert "best eval (preview)" in running, "while the search runs, the best eval's preview is the curve to watch"
    assert "best eval (preview)" not in done
    assert any(lab.startswith("validated winner") for lab in done)
    assert any(lab.startswith("KLIP-FM cross-check") for lab in done)
