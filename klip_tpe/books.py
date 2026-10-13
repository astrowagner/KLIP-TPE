"""The annulus books, drawn by a process of their own.

At the end of an annulus the live display writes ten books from the annulus' records
(corner, landscapes, parhist, edf, partition map, kbook, importance, paracoord, rank,
slice).  Drawn on the run's main thread they held up the next annulus, with every worker
idle: 3.8 min at 5,000 evaluations in the sandbox, and 26 min after annulus 1 (10,000
evaluations) of ``run_20261008_093817`` on Mac-Pro-5.

:func:`start` pickles the same :class:`~klip_tpe.display.AnnulusData` the display built and
starts ``python -m klip_tpe.books`` on it.  That process calls the same functions in the
same order on the same data, so the files are the same, and the run goes on with the next
annulus meanwhile.  It runs at a lower priority than the workers, and a Ctrl-C of the run
does not stop it: it finishes the books and exits.  The run waits for it only at its end.
``KLIP_TPE_BOOKS=inline`` draws the books on the main thread, as before.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["BOOKS", "draw", "start", "BooksJob", "in_background"]

#: (label, renderer in :mod:`klip_tpe.display`, file in the annulus directory), in the order
#: the display has always drawn them
BOOKS: Tuple[Tuple[str, str, str], ...] = (
    ("corner", "render_corner", "corner.pdf"),
    ("landscapes", "render_landscapes", "landscapes.pdf"),
    ("parhist", "render_parhist_book", "parhist.pdf"),
    ("edf", "render_edf", "edf.png"),
    ("partition map", "render_nightmap", "partition_map.png"),
    ("kbook", "render_kbook", "kbook.pdf"),
    ("importance", "render_importance", "importance.pdf"),
    ("paracoord", "render_paracoord", "paracoord.pdf"),
    ("rank", "render_rank", "rank.pdf"),
    ("slice", "render_slice", "slice.pdf"),
)

_JOB = "annulus.pkl"
_RESULT = "result.json"
_LOG = "books.log"


def in_background() -> bool:
    """False when ``KLIP_TPE_BOOKS`` asks for the books on the main thread."""
    return os.environ.get("KLIP_TPE_BOOKS", "").strip().lower() not in ("inline", "main", "0", "off", "no")


def draw(ad, out_dir: str, books: Sequence[Tuple[str, str, str]] = BOOKS) -> List[Dict[str, Any]]:
    """Draw the books of ``ad`` into ``out_dir`` in order, each on its own: one that fails
    does not stop the rest.  Returns ``{"book", "ok", "s"[, "error"]}`` per book."""
    from . import display
    out = []
    for label, fn, name in books:
        t0 = time.perf_counter()
        try:
            getattr(display, fn)(ad, os.path.join(out_dir, name))
            out.append({"book": label, "ok": True, "s": time.perf_counter() - t0})
        except Exception as exc:
            out.append({"book": label, "ok": False, "s": time.perf_counter() - t0,
                        "error": f"{exc!r}\n" + traceback.format_exc(limit=3)})
    return out


def _command(job_dir: str) -> List[str]:
    return [sys.executable, "-m", "klip_tpe.books", job_dir]


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"                   # files only, never a GUI toolkit
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, "1")
    # The klip_tpe this run imported: a checkout put on sys.path by hand is passed on.  An
    # installed one is found anyway, and its site-packages ahead of the standard library
    # could shadow a module there.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        import site
        sites = {os.path.abspath(p) for p in list(site.getsitepackages()) + [site.getusersitepackages()]}
    except Exception:
        sites = set()
    if root not in sites:
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class BooksJob:
    """One annulus' books being drawn by a background process."""

    def __init__(self, ia: int, out_dir: str, job_dir: str, proc: subprocess.Popen):
        self.ia, self.out_dir, self.job_dir, self.proc = int(ia), out_dir, job_dir, proc
        self.t0 = time.time()

    def done(self) -> bool:
        return self.proc.poll() is not None

    def finish(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Wait for the process, then return its per-book results and a note for the log
        (or None).  A process that ended without reporting -- killed, out of memory, an
        interpreter that would not start -- leaves the pickle behind, and the books are
        drawn here from it instead."""
        code = self.proc.wait()
        note = None
        try:
            with open(os.path.join(self.job_dir, _RESULT)) as f:
                res = json.load(f)
            statuses = list(res["books"])
            secs = float(res.get("s", time.time() - self.t0))
            note = (f"annulus {self.ia + 1} books: {sum(s['ok'] for s in statuses)} of {len(statuses)} "
                    f"written in {secs:.0f} s by a separate process")
        except Exception:
            tail = ""
            try:
                with open(os.path.join(self.job_dir, _LOG), errors="replace") as f:
                    tail = f.read()[-600:].strip()
            except Exception:
                pass
            with open(os.path.join(self.job_dir, _JOB), "rb") as f:
                job = pickle.load(f)
            statuses = draw(job["ad"], self.out_dir, job.get("books", BOOKS))
            note = (f"annulus {self.ia + 1} books: the separate process ended with code {code} before "
                    f"finishing, so they were drawn here" + (f" ({tail})" if tail else ""))
        shutil.rmtree(self.job_dir, ignore_errors=True)
        return statuses, note


def start(ad, out_dir: str, ia: int, books: Sequence[Tuple[str, str, str]] = BOOKS) -> BooksJob:
    """Start drawing the books of ``ad`` into ``out_dir`` in a separate process.  Raises when
    no process can be started, and the caller then draws them itself (:func:`draw`)."""
    job_dir = tempfile.mkdtemp(prefix="klip_books_")
    try:
        with open(os.path.join(job_dir, _JOB), "wb") as f:
            pickle.dump({"ad": ad, "out_dir": os.path.abspath(out_dir), "books": tuple(books)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(job_dir, _LOG), "wb") as log:
            # started in the job directory, so nothing in the run's working directory can
            # stand in for a module
            proc = subprocess.Popen(_command(job_dir), env=_env(), cwd=job_dir, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return BooksJob(ia, out_dir, job_dir, proc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m klip_tpe.books JOB_DIR``: draw the pickled books and report."""
    job_dir = (sys.argv[1:] if argv is None else list(argv))[0]
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)      # a Ctrl-C of the run: finish the books
    except Exception:
        pass
    try:
        os.nice(10)                                       # behind the workers
    except Exception:
        pass
    t0 = time.time()
    pkl = os.path.join(job_dir, _JOB)
    with open(pkl, "rb") as f:
        job = pickle.load(f)
    statuses = draw(job["ad"], job["out_dir"], job.get("books", BOOKS))
    tmp = os.path.join(job_dir, _RESULT + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"books": statuses, "s": time.time() - t0}, f)
    os.replace(tmp, os.path.join(job_dir, _RESULT))
    os.remove(pkl)                                        # reported: the snapshot is not needed again
    return 0


if __name__ == "__main__":
    sys.exit(main())
