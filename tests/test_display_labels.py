"""Cheap checks on the live panel's labels and placeholders (quick suite; the end-to-end
display tests live in ``test_display.py`` and are marked slow)."""
import numpy as np

from klip_tpe.display import StepImages, render_step


def test_library_counts_get_distinct_labels():
    """``nkeep_altroll`` and ``nkeep_psfref`` used to abbreviate to the same ``nkeep_`` (a
    bare six-character cut), so the importance panel showed two bars both labelled
    ``nkeep_`` and the config tables could not say which pool a number belonged to."""
    from klip_tpe.display import _abbrev, _abbrev_unique
    assert _abbrev("nkeep_altroll") == "nkalt" and _abbrev("nkeep_psfref") == "nkref"
    assert _abbrev("nkeep_starA") == "nkstar"                       # any other named group
    names = ["bin", "n_ang", "filter", "k_klip", "nkeep_altroll", "nkeep_psfref"]
    lab = _abbrev_unique(names)
    assert lab == ["bin", "nang", "filt", "k", "nkalt", "nkref"]
    # whatever a future space calls its parameters, two must never share a label
    lab = _abbrev_unique(["abcdefX", "abcdefY", "bin"])
    assert len(set(lab)) == 3 and lab[2] == "bin"


def test_importance_panel_labels_are_distinct():
    """The live importance cell, fed the MIRI v6 space, labels every bar differently."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klip_tpe.display import panel_importance_live
    rng = np.random.default_rng(3)
    names = ["bin", "n_ang", "filter", "k_klip", "nkeep_altroll", "nkeep_psfref"]
    X = rng.uniform(0, 1, (40, len(names)))
    y = X @ np.linspace(0.2, 1.2, len(names)) + 0.1 * rng.standard_normal(40)
    fig, ax = plt.subplots()
    panel_importance_live(ax, None, cs={"y": y, "X": X, "names": names})
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    assert len(labels) == len(names) and len(set(labels)) == len(names)
    assert {"nkalt", "nkref"} <= set(labels)


def test_fm_cell_says_why_it_is_empty(tmp_path):
    """On a backend without KLIP-FM the cell used to read "(after 1st best)" for the whole
    run -- still there on the finished panel.  It now names the reason, and keeps the old
    text only where a model really is on its way."""
    from klip_tpe.display import _fm_unavailable_note, annulus_from_records
    v6 = {"name": "spaceklip", "supports_fm": False,
          "partitions": {"sci": {"name": "spaceklip_sci", "supports_fm": False, "backend": "pyklip"}}}
    note = _fm_unavailable_note(v6)
    assert note and "pyklip" in note and "KLIP-FM" in note
    assert _fm_unavailable_note({"name": "klip", "supports_fm": True}) is None
    assert _fm_unavailable_note(None) is None

    space = {"params": [{"name": "bin", "lo": 1, "hi": 20, "kind": "int"}], "partitions": []}
    cfg = {"ann_edges": [5, 20], "n_iter": 2, "n_init": 1, "gamma": 0.3, "search_mode": "tpe", "seed_default": True}
    recs = [{"annulus": 0, "index": i, "phase": "seed" if i == 0 else "tpe", "x": [1 + i],
             "config": {"params": {"bin": 1 + i}, "per_partition": {}, "selected": [], "x": [1 + i]},
             "sources": [(0.5, 30.0, 1e-4)], "score": 3.0 + i, "raw_score": 3.0 + i, "per_source": [3.0 + i],
             "raw_per_source": [], "clean_per_source": None, "partition_snr": {}, "k_used": None,
             "contrast": 1e-4, "wall_s": 0.1, "meta": {"failed": False, "selected": []}} for i in range(2)]
    ad = annulus_from_records(recs, space, cfg, 0, 0.05, 4.0, "m", "fm")
    img = np.random.default_rng(0).standard_normal((48, 48))
    texts = {}
    for key, fm in (("none", note), ("pending", None)):
        fig = render_step(ad, 1, StepImages(cur_inj=img, cur_clean=img, best_inj=img, best_clean=img,
                                            best_index=1, fm_unavailable=fm), None, dpi=50)
        texts[key] = [t.get_text() for a in fig.axes for t in a.texts]
    assert any("no KLIP-FM" in t for t in texts["none"])
    assert not any("(after 1st best)" == t for t in texts["none"] if "KLIP" in t)
    assert any(t == "(after 1st best)" for t in texts["pending"])
