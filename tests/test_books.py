"""The annulus books drawn by a process of their own.

From the NEAR2 parallelism note of 2026-10-13: the minutes between annuli were the display's
books, drawn on the run's main thread while every worker waited (26 min after annulus 1 of
run_20261008_093817).  What must hold:

* the books from the separate process are the same files, byte for byte, as the books the
  display drew itself;
* the display goes on without waiting for them, reports them once they are written, and
  waits for them at the end of the run;
* a process that dies leaves the books to the run, which draws them from the snapshot;
* a Ctrl-C of the run does not stop the books process;
* ``KLIP_TPE_BOOKS=inline`` draws them on the main thread, as before.
"""
from __future__ import annotations

import os
import sys

import pytest

from klip_tpe import CalibrationConfig, Runner, ValidationConfig, books
from klip_tpe.display import LiveDisplay, annulus_from_run

from conftest import build_synthetic_run


@pytest.fixture(scope="module")
def ad(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("books_run"))
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=8, n_init=4, calibration=CalibrationConfig(recal_budget=0),
        validation=ValidationConfig(n_top=1, n_valid=1))
    Runner(red, space, obj, samp, cfg, d, log=lambda s: None).run()
    return annulus_from_run(d, 0)


#: two quick books, for the tests about the process rather than the pictures
FEW = tuple(b for b in books.BOOKS if b[0] in ("edf", "partition map"))


def _files(d):
    return {name: open(os.path.join(d, name), "rb").read() for _, _, name in books.BOOKS}


def test_the_books_from_the_separate_process_are_the_same_files(ad, tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")        # the PDFs' creation date, in both
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    assert all(s["ok"] for s in books.draw(ad, str(here)))
    statuses, note = books.start(ad, str(there), 0).finish()
    assert all(s["ok"] for s in statuses) and [s["book"] for s in statuses] == [b[0] for b in books.BOOKS]
    assert note.startswith("annulus 1 books: 10 of 10 written") and "separate process" in note
    a, b = _files(here), _files(there)
    assert all(len(v) > 1000 for v in a.values())
    assert [k for k in a if a[k] != b[k]] == []


def test_the_display_goes_on_and_reports_them_when_written(ad, tmp_path):
    logs = []
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, log=logs.append)
    d = str(tmp_path / "annulus01")
    os.makedirs(d)
    disp._books_start(None, ad, d, 0)
    assert len(disp._books) == 1, "handed to a process, not drawn here"
    disp._collect_books(None, wait=True)                       # the end of the run waits
    assert disp._books == []
    assert all(os.path.getsize(os.path.join(d, name)) > 1000 for _, _, name in books.BOOKS)
    assert [l for l in logs if "annulus 1 books: 10 of 10 written" in l], logs
    assert not [l for l in logs if "failed" in l], logs


def test_a_books_process_that_dies_leaves_them_to_the_run(ad, tmp_path, monkeypatch):
    monkeypatch.setattr(books, "_command", lambda job_dir: [sys.executable, "-c", "import sys; sys.exit(3)"])
    job = books.start(ad, str(tmp_path), 0, books=FEW)
    statuses, note = job.finish()
    assert "ended with code 3" in note and "drawn here" in note
    assert all(s["ok"] for s in statuses) and len(statuses) == 2
    assert all(os.path.getsize(tmp_path / name) > 1000 for _, _, name in FEW)
    assert not os.path.exists(job.job_dir), "the snapshot is removed once the books are drawn"


def test_ctrl_c_does_not_stop_the_books(ad, tmp_path, monkeypatch):
    """A SIGINT 0.2 s in, while it loads matplotlib and draws: it finishes and reports."""
    script = ("import os, signal, sys, threading; from klip_tpe.books import main; "
              "threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start(); "
              "sys.exit(main([sys.argv[1]]))")
    monkeypatch.setattr(books, "_command", lambda job_dir: [sys.executable, "-c", script, job_dir])
    statuses, note = books.start(ad, str(tmp_path), 0, books=FEW).finish()
    assert all(s["ok"] for s in statuses)
    assert "by a separate process" in note, note


def test_inline_draws_them_on_the_main_thread(ad, tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_BOOKS", "inline")
    assert not books.in_background()
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, log=lambda s: None)
    disp._books_start(None, ad, str(tmp_path), 0)
    assert disp._books == []
    assert all(os.path.getsize(tmp_path / name) > 1000 for _, _, name in books.BOOKS)
