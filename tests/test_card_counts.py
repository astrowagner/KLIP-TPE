"""What the card took and what it turned away, per annulus (NEAR2 s6 note, section 6).

The pool counts the jobs marked for the card and in its share, and of those the ones the card's
reducer turns away, by the reason ``GPUKLIPReducer.refusal`` gives.  After each annulus' search
the runner logs how many the card reduced and how many it turned away.
"""
from __future__ import annotations

import types

from klip_tpe.reducer import ReductionRequest
from klip_tpe.runner import Runner

from test_gpu_reducer import A_IN, A_OUT, SOURCES, synthetic_night  # noqa: E402


def test_the_card_reducer_says_why_it_turns_a_job_away():
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    g = GPUKLIPReducer.from_cpu(synthetic_night("refusal", 40, seed=4))
    p = dict(bin=5, n_ang=2, filter=7, angsep=0.3, anglemax=120.0, k_klip=6, inrad=A_IN, outrad=A_OUT)
    assert g.refusal(ReductionRequest(p, SOURCES, False, "plain")) is None and g.supports(
        ReductionRequest(p, SOURCES, False, "plain"))
    cases = {"KLIP-FM": ReductionRequest(p, SOURCES, False, "fm", fm_sources=SOURCES),
             "k-scan": ReductionRequest(p, SOURCES, True, "ks"),
             "extras": ReductionRequest(p, SOURCES, False, "x", extras=True),
             "fast": ReductionRequest(dict(p, fast=True), SOURCES, False, "f"),
             "do_destripe": ReductionRequest(dict(p, do_destripe=True), SOURCES, False, "d"),
             "comb_type weighted": ReductionRequest(dict(p, comb_type="weighted"), SOURCES, False, "c")}
    for why, req in cases.items():
        assert g.refusal(req) == why and not g.supports(req), why


def test_the_annulus_line_counts_the_jobs_since_the_search_started():
    logs = []
    counts = {"marked": 10, "jobs": 7, "refused": {"extras": 1}}
    red = types.SimpleNamespace(card_summary=lambda: dict(counts))
    stub = types.SimpleNamespace(reducer=red, log=logs.append)
    stub._card_counts = lambda: Runner._card_counts(stub)
    stub._card_mark = Runner._card_counts(stub)
    counts.update(marked=40, jobs=32, refused={"extras": 1, "KLIP-FM": 3, "fast": 2})
    Runner._card_note(stub, 1)
    assert logs[-1] == ("  annulus 2: the card reduced 25 of the 30 jobs marked for it, and turned away 5 "
                        "(KLIP-FM 3, fast 2)"), logs
    counts.update(marked=50, jobs=40, refused={"extras": 1, "KLIP-FM": 3, "fast": 2})
    Runner._card_note(stub, 2)
    assert logs[-1] == ("  annulus 3: the card reduced 8 of the 10 jobs marked for it; 2 went back to the "
                        "CPU workers"), logs
    logs.clear()
    Runner._card_note(types.SimpleNamespace(reducer=types.SimpleNamespace(), log=logs.append,
                                            _card_counts=lambda: None), 0)
    assert logs == [], "no card workers, no line"


def test_the_card_reducers_memory_hooks_do_nothing_on_the_host():
    """``limit_memory`` holds a CuPy pool and ``after_job`` empties it.  With the NumPy backend
    (the tests) there is no pool: no limit, and nothing to free."""
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    from klip_tpe.parallel import CardMemoryError, _is_out_of_memory
    g = GPUKLIPReducer.from_cpu(synthetic_night("memory", 30, seed=5))
    g.backend_name = "numpy"
    assert g.limit_memory(3) is None
    g.after_job()
    g.warm()
    g.after_job()

    class OutOfMemoryError(Exception):
        pass
    assert _is_out_of_memory(OutOfMemoryError("Out of memory allocating 22,155,776,512 bytes"))
    assert _is_out_of_memory(MemoryError()) and not _is_out_of_memory(ValueError("no memory of it"))
    e = CardMemoryError("a2_e11_d1_inj_n6", "OutOfMemoryError: Out of memory")
    import pickle
    e2 = pickle.loads(pickle.dumps(e))
    assert e2.tag == "a2_e11_d1_inj_n6" and str(e2) == "a2_e11_d1_inj_n6: OutOfMemoryError: Out of memory"
