"""``klip-tpe resume`` must keep working however many times it is resumed.

The failure this file exists for: a resume that got its data root *from* the registry then
recorded its own sparse command line -- ``klip-tpe resume --show`` -- as the newest entry
for that run.  Since the newest entry wins, the next resume inherited a record with no
``--root`` in it and stopped with "--root is required for --instrument near / nomic", on a
run that had been resuming fine all day.  The record degraded a little with every resume
until it was useless, and the only way out was to retype everything the feature exists to
remember.

Two things keep it fixed: a resume looks back through the history until it finds a record
that actually carries what the reducer needs, and it re-records a *complete* command line
so the history stops degrading in the first place.
"""
import json
import os

import pytest

from klip_tpe import cli, registry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "root" / "comb" / "opt" / "run_20260912_160024"
    d.mkdir(parents=True)
    with open(d / "results.jsonl", "w") as f:
        f.write(json.dumps({"annulus": 0, "index": 0}) + "\n")
    return str(d)


FULL = ["klip-tpe", "near", "--root", "/data/NEAR2_py", "--nights", "1", "2", "3",
        "--run-dir", "RUN", "--workers", "8", "--show"]


def _ns(**kw):
    import argparse
    base = dict(root=None, run_dir="last", nights=None, instrument="near", workers=None,
                obj=None, name=None, binned=False, frames=None, crop_half=None,
                parang_sign=None, pre_bin=None, cube=None, angles=None, psf=None,
                pxscale=None, lam=None, diam=None, ref_cube=None, names=None,
                star_flux=None, pool=None, known=None, no_library=False, fast=False,
                n_min_ref=None, truenorth=None, fwhm_px=None, metric=None, backend=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_bare_resume_recovers_the_root(reg, capsys):
    """The baseline: one full record, then ``klip-tpe resume`` with nothing typed."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    a = _ns()
    cli._fill_from_registry(a, ["resume"])
    assert a.root == ["/data/NEAR2_py"], a.root
    assert a.nights == [1, 2, 3]


def test_a_resume_after_a_resume_still_finds_the_root(reg, capsys):
    """The reported failure, exactly: the newest record is the sparse one a previous
    ``klip-tpe resume --show`` wrote, and it has no root in it."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    registry.register(reg, argv=["klip-tpe", "resume", "--show"])          # what broke it
    a = _ns()
    cli._fill_from_registry(a, ["resume", "--show"])
    assert a.root == ["/data/NEAR2_py"], (
        "the sparse resume record shadowed the one that carries the root")
    assert a.nights == [1, 2, 3], "the night list was lost with it"


def test_it_survives_many_sparse_resumes(reg):
    """Five bare resumes in a row must not walk the record off the end."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    for _ in range(5):
        registry.register(reg, argv=["klip-tpe", "resume"])
    a = _ns()
    cli._fill_from_registry(a, ["resume"])
    assert a.root == ["/data/NEAR2_py"]


def test_the_newest_complete_record_wins_over_an_older_one(reg):
    """Looking back must not reach past a *newer* record that does carry a root -- a run
    restarted against moved data must resume against the new location."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    moved = [t if t != "RUN" else reg for t in FULL]
    moved[moved.index("/data/NEAR2_py")] = "/data2/NEAR2_py"
    registry.register(reg, argv=moved)
    registry.register(reg, argv=["klip-tpe", "resume"])
    a = _ns()
    cli._fill_from_registry(a, ["resume"])
    assert a.root == ["/data2/NEAR2_py"], "resumed against the stale root"


def test_what_the_user_typed_still_wins(reg):
    """Looking back through the history must never override an explicit argument."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    registry.register(reg, argv=["klip-tpe", "resume"])
    a = _ns(root=["/somewhere/else"])
    cli._fill_from_registry(a, ["resume", "--root", "/somewhere/else"])
    assert a.root == ["/somewhere/else"]


def test_a_resume_records_a_command_line_that_works_on_its_own(reg):
    """The other half of the fix: the entry a resume writes must be complete, so the
    history stops degrading.  Without this the test above is only papering over it."""
    registry.register(reg, argv=[t if t != "RUN" else reg for t in FULL])
    a = _ns()
    cli._fill_from_registry(a, ["resume", "--show"])
    rec_argv = getattr(a, "_record_argv", None)
    assert rec_argv, "a resume left nothing better than its own sparse argv to record"
    assert "--root" in rec_argv and "/data/NEAR2_py" in rec_argv
    assert "--nights" in rec_argv

    # and registering it leaves the next resume able to work from that entry alone
    registry.register(reg, argv=rec_argv)
    b = _ns()
    cli._fill_from_registry(b, ["resume"])
    assert b.root == ["/data/NEAR2_py"] and b.nights == [1, 2, 3]


def test_no_usable_record_says_so_instead_of_failing_deep_inside(reg):
    """With nothing on record, the message must name what is missing at the point the user
    can act on it -- not surface later as an instrument error."""
    registry.register(reg, argv=["klip-tpe", "resume"])
    a = _ns()
    cli._fill_from_registry(a, ["resume"])
    assert a.root is None                       # nothing to restore, and no exception
