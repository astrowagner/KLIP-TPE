"""The calibration k-scan's argmax, and the two cases where it is not a measurement.

``k_default`` is the seed of the whole run: the configuration the contrast is calibrated at,
the one ``collect.py`` re-scores as the anchor, and the starting point the search is judged
against.  It is chosen by ``argmax`` of a curve of S/N against the number of KL modes, and
that curve is measured -- so the argmax can be an artefact of the measurement rather than a
statement about the data.  Two ways, both seen on real MIRI data:

* the argmax sits at an **edge** of the scanned range, so it is not an interior optimum;
* it beats the incumbent by less than the **scatter between the scan's own draws**.

These tests are fast (the curve is dictated by a stub, no reductions happen), so unlike the
rest of ``test_runner.py`` they run in the quick suite.
"""
import pytest

class _ScanStub:
    """Just enough of a Runner for ``_scan_k``: the curve is dictated per draw."""

    def __init__(self, draws):
        import types
        from klip_tpe.runner import CalibrationConfig, Runner
        self._draws, self._i, self.msgs = list(draws), 0, []
        self.cfg = types.SimpleNamespace(
            calibration=CalibrationConfig(n_remeasure=len(draws)))
        self.sampler = types.SimpleNamespace(sample=lambda *a, **k: [object()])
        self.objective = types.SimpleNamespace(
            metric=types.SimpleNamespace(needs_clean=False),
            score_raw=lambda img, src, clean: types.SimpleNamespace(score=float(img)))
        self.rng = None
        self._scan_k = types.MethodType(Runner._scan_k, self)

    def _k_scan_max(self):
        return len(self._draws[0])

    def _with_k(self, cfg, k):
        return cfg

    def _reduce(self, cfg, src, k_scan=False, tag=""):
        import numpy as np, types
        c = self._draws[self._i % len(self._draws)]
        self._i += 1
        return types.SimpleNamespace(image=np.asarray(c, float))

    def log(self, s):
        self.msgs.append(s)

    def scan(self, k_now=6):
        info = {}
        return self._scan_k(None, 0.0, 1.0, 1, 3e-4, info, k_now), info, self.msgs


def test_a_kscan_argmax_at_the_edge_of_the_range_is_called_an_edge_hit():
    """Measured on HIP 65426 F1140C, both ends, and neither was a measurement.

    With a background-mismatched RDI library the curve was flat to 1.5% over k = 4..20 with
    k = 1 on top by 1.6% -- the leading KL mode was the sky pedestal, so removing it was the
    only subtraction that helped.  With the background fixed the argmax moved to k = 20, the
    *other* end, winning by 0.1%.  ``locate_boundaries`` already refuses to accept an
    extremum at the edge of its own window; this is the same rule for this scan.
    """
    v2 = [6.151, 5.785, 5.985, 6.008, 6.054, 6.051, 5.980, 5.976, 5.972, 5.997,
          6.011, 6.003, 5.998, 6.007, 5.994, 5.978, 5.967, 6.048, 6.056, 6.013]
    _, info, msgs = _ScanStub([v2, v2, v2]).scan(k_now=6)
    assert info["kscan_edge"] == "bottom"
    assert any("at the bottom of the scanned range" in m for m in msgs)
    assert any("first component is not the star" in m for m in msgs)

    v3 = [5.370, 5.411, 5.713, 6.471, 6.474, 6.525, 6.482, 6.470, 6.480, 6.457,
          6.486, 6.546, 6.575, 6.554, 6.645, 6.652, 6.612, 6.640, 6.652, 6.657]
    _, info, msgs = _ScanStub([v3, v3, v3]).scan(k_now=6)
    assert info["kscan_edge"] == "top"
    assert any("raise k_klip_max" in m for m in msgs), \
        "an argmax at the top must say the SEARCH cap is what is limiting, not just the scan"


def test_an_unresolved_kscan_keeps_the_incumbent_instead_of_moving_on_noise():
    """k_default seeds the whole run and anchors what collect.py reports against, so it must
    not move on a win smaller than the scatter between the scan's own draws."""
    # draws that disagree at EVERY k, which is what a real curve does: the per-k spread is
    # ~0.4 and k = 3 beats the incumbent k = 1 by 0.05.  That is the v3 situation (0.1%).
    a = [6.00, 5.2, 6.05, 5.1, 4.9]
    b = [6.10, 4.8, 6.15, 5.5, 5.3]
    k, info, msgs = _ScanStub([a, b]).scan(k_now=1)
    assert k is None, "moved the seed on a difference inside the draw-to-draw spread"
    assert info["kscan_unresolved"] is True
    assert any("is not resolved" in m for m in msgs)
    # a clear win, with the draws agreeing, is still taken
    c = [5.00, 5.00, 9.00, 5.00, 5.00]
    d = [5.05, 5.05, 9.05, 5.05, 5.05]
    k, info, _ = _ScanStub([c, d]).scan(k_now=1)
    assert k == 3 and not info.get("kscan_unresolved")


def test_the_kscan_records_the_draw_spread_it_used_to_discard():
    a, b = [1.0, 2.0, 3.0], [1.3, 2.3, 3.3]        # 0.3 apart at every k
    _, info, _ = _ScanStub([a, b]).scan(k_now=1)
    assert info["kscan_draw_spread"] == pytest.approx(0.3, abs=0.01)
    # and with a single draw there is nothing to estimate it from, so the rule stands down
    _, info1, _ = _ScanStub([a]).scan(k_now=1)
    assert info1["kscan_draw_spread"] is None and not info1.get("kscan_unresolved")
