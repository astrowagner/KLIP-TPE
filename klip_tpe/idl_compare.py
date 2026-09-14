"""Live comparison against a running (or finished) IDL ``optimize_near_2_tpe`` run.

Two complementary checks, both incremental so they can be re-run as the IDL log grows:

1. **Replay** (:class:`LiveComparison`): every configuration the IDL run has evaluated is
   re-scored by the Python objective on the same nights, same annulus, same contrast and
   the same injection geometry (exact positions when the IDL step-setup file exists,
   otherwise a fresh azimuth anchor at the same radii).  Output: a per-evaluation table
   ``idl_score, py_score, py_raw`` and agreement statistics (Pearson / Spearman, median
   ratio, running-best curves, agreement of the top-10 sets).  This isolates reducer +
   metric differences from sampler differences.

2. **Matched run** (:func:`matched_run_config`): a ``RunConfig`` built from the IDL
   ``run_setup.txt`` (annulus, budget, contrast, validation, TPE settings) so a Python
   search can be run side by side; compare best-so-far curves and validated winners with
   :func:`compare_runs`.

CLI::

    klip-tpe compare --idl-run <IDL run dir> --root <NEAR root> \\
                     --out runs/compare_2026... [--max-new 40] [--workers 4]
    klip-tpe near --from-idl-setup <IDL run dir>/run_setup.txt --run-dir runs/py_match ...
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .idl_replay import IDLLog, idl_row_to_vector, read_idl_log, read_run_setup
from .metrics import Source

__all__ = ["read_step_setup", "LiveComparison", "matched_run_config", "compare_runs", "agreement_stats"]


def read_step_setup(path: str) -> Dict[str, Any]:
    """Parse an IDL ``evalNNNN_setup.txt`` / ``calibNNNN_setup.txt`` (parameters, nights,
    median S/N and the injected ``(rho, PA, contrast)`` list)."""
    out: Dict[str, Any] = {"sources": []}
    in_src = False
    with open(path) as f:
        for line in f:
            if line.startswith("# injected sources"):
                in_src = True
                continue
            if line.startswith("#"):
                if in_src and "PER-PARTITION" in line:
                    in_src = False
                continue
            if in_src:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        out["sources"].append(Source(float(parts[0]), float(parts[1]), float(parts[2])))
                    except ValueError:
                        in_src = False
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    for k in ("step", "annulus"):
        if k in out:
            try:
                out[k] = int(float(out[k]))
            except ValueError:
                pass
    if "median_SNR" in out:
        try:
            out["median_SNR"] = float(out["median_SNR"])
        except ValueError:
            pass
    return out


def _idl_step_setups(idl_run_dir: str, annulus: int = 1) -> Dict[int, str]:
    d = os.path.join(idl_run_dir, f"annulus{annulus:02d}")
    files = glob.glob(os.path.join(d, "eval*_setup.txt")) + glob.glob(os.path.join(idl_run_dir, "opt_steps_tpe", "eval*_setup.txt"))
    out = {}
    for f in files:
        m = re.search(r"eval(\d+)_setup", os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def agreement_stats(idl: np.ndarray, py: np.ndarray) -> Dict[str, Any]:
    from scipy import stats
    a, b = np.asarray(idl, float), np.asarray(py, float)
    g = np.isfinite(a) & np.isfinite(b)
    out: Dict[str, Any] = {"n": int(g.sum())}
    if g.sum() < 3:
        return out
    a, b = a[g], b[g]
    rho, p = stats.spearmanr(a, b)
    r, _ = stats.pearsonr(a, b)
    out.update(spearman=float(rho), spearman_p=float(p), pearson=float(r),
               mean_idl=float(a.mean()), mean_py=float(b.mean()),
               median_ratio=float(np.median(b / np.where(a != 0, a, np.nan))),
               rms_diff=float(np.sqrt(np.mean((b - a) ** 2))))
    # regression PY = s * IDL + o  (slope < 1 with o ~ 0: Python systematically lower)
    s, o = np.polyfit(a, b, 1)
    out.update(slope=float(s), offset=float(o))
    # top-set overlap: do the two scorers agree on which configurations are best?
    for k in (5, 10, 20):
        if len(a) >= 2 * k:
            ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
            out[f"top{k}_overlap"] = len(ta & tb) / k
    # running best positions
    out["argmax_idl"], out["argmax_py"] = int(np.argmax(a)), int(np.argmax(b))
    out["best_idl"], out["best_py"] = float(a.max()), float(b.max())
    out["py_at_idl_best"], out["idl_at_py_best"] = float(b[np.argmax(a)]), float(a[np.argmax(b)])
    return out


class LiveComparison:
    """Incrementally replay a (running) IDL run through the Python objective.

    Parameters
    ----------
    idl_run_dir : the IDL ``run_YYYYMMDD_HHMMSS`` directory (needs ``optimize_tpe_results.txt``
        and ``run_setup.txt``; ``annulus01/evalNNNN_setup.txt`` files are used when present).
    runner : a :class:`~klip_tpe.runner.Runner` built on the same nights / space (its
        ``space`` must use the IDL parameter names, e.g. from ``near.make_space``).  The
        runner is only used for :meth:`Runner.evaluate` (no search state is touched).
    out_dir : where ``compare.jsonl`` / ``compare.txt`` / ``compare.png`` go.
    """

    def __init__(self, idl_run_dir: str, runner, out_dir: str, annulus: int = 1, n_repeat: int = 1,
                 log: Callable[[str], None] = print):
        self.idl_run_dir = idl_run_dir
        self.runner = runner
        self.out_dir = out_dir
        self.annulus = int(annulus)
        self.n_repeat = int(n_repeat)
        self.log = log
        os.makedirs(out_dir, exist_ok=True)
        self.setup = read_run_setup(os.path.join(idl_run_dir, "run_setup.txt"))
        self.jsonl = os.path.join(out_dir, "compare.jsonl")
        self.rows: List[Dict[str, Any]] = []
        if os.path.exists(self.jsonl):
            with open(self.jsonl) as f:
                self.rows = [json.loads(l) for l in f if l.strip()]
        self._done = {int(r["iter"]) for r in self.rows}
        # the IDL contrast for this annulus
        uc = self.setup.get("use_contrast", "")
        m = re.search(r"([-\d.eE+]+)", uc)
        self.contrast = float(m.group(1)) if m else float(runner.contrast or runner.cfg.contrast0)
        runner.ia = self.annulus - 1
        runner.contrast = self.contrast

    # ------------------------------------------------------------------ core
    def pending(self) -> List[int]:
        log = read_idl_log(os.path.join(self.idl_run_dir, "optimize_tpe_results.txt"))
        idx = [i for i in range(len(log.iter)) if int(log.annulus[i]) == self.annulus and int(log.iter[i]) not in self._done]
        self._log = log
        return idx

    def sync(self, max_new: Optional[int] = None, stride: int = 1, offset: int = 0) -> int:
        """Replay every IDL evaluation not yet compared (oldest first; ``stride``/``offset``
        pick every n-th iteration, e.g. ``stride=25, offset=1`` = iters 1, 26, 51 ...).
        Returns how many were added.  Safe to call repeatedly while the IDL run is writing."""
        idx = self.pending()
        if stride > 1:
            idx = [i for i in idx if (int(self._log.iter[i]) - offset) % stride == 0]
        if max_new is not None:
            idx = idx[:int(max_new)]
        setups = _idl_step_setups(self.idl_run_dir, self.annulus)
        log = self._log
        for n, i in enumerate(idx):
            it = int(log.iter[i])
            x = self.runner.space.sanitize(idl_row_to_vector(log, i, self.runner.space))
            src = None
            exact = False
            if it in setups:
                st = read_step_setup(setups[it])
                if st["sources"]:
                    src = st["sources"]
                    exact = True
            t0 = time.time()
            py, raw = [], []
            for _ in range(self.n_repeat):
                rec, inj, clean = self.runner.evaluate(x, "compare", contrast=self.contrast, sources=src,
                                                       tag=f"cmp{it}")
                py.append(np.nan if rec.score is None else float(rec.score))
                raw.append(np.nan if rec.raw_score is None else float(rec.raw_score))
                if src is None:
                    src_used = rec.sources
                else:
                    src_used = [s.as_tuple() for s in src]
            row = {"iter": it, "annulus": self.annulus,
                   "idl_score": None if not np.isfinite(log.score[i]) else float(log.score[i]),
                   "py_score": float(np.nanmean(py)) if np.isfinite(py).any() else None,
                   "py_raw": float(np.nanmean(raw)) if np.isfinite(raw).any() else None,
                   "py_scores": [None if not np.isfinite(v) else v for v in py],
                   "exact_positions": exact, "sources": [list(s) for s in src_used],
                   "k_idl": float(log.k[i]), "wall_s": time.time() - t0,
                   "config": log.config_dict(i)}
            self.rows.append(row)
            self._done.add(it)
            with open(self.jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.log(f"compare {n+1}/{len(idx)}  IDL iter {it}: IDL {row['idl_score']}  PY {row['py_score']:.3f} "
                     f"(raw {row['py_raw']:.3f}){'  [exact positions]' if exact else ''}  {row['wall_s']:.0f}s")
            if (n + 1) % 5 == 0 or n + 1 == len(idx):
                self.report()                       # keep the report current (chunks may be killed mid-way)
        return len(idx)

    # --------------------------------------------------------------- reports
    def stats(self) -> Dict[str, Any]:
        a = np.array([r["idl_score"] if r["idl_score"] is not None else np.nan for r in self.rows])
        b = np.array([r["py_score"] if r["py_score"] is not None else np.nan for r in self.rows])
        s = agreement_stats(a, b)
        s["n_exact"] = int(sum(bool(r.get("exact_positions")) for r in self.rows))
        s["idl_run"] = os.path.basename(os.path.normpath(self.idl_run_dir))
        s["contrast"] = self.contrast
        return s

    def report(self) -> str:
        s = self.stats()
        lines = [f"# klip-tpe vs IDL replay -- {s['idl_run']}  annulus {self.annulus}  contrast {self.contrast:.2e}",
                 f"# {time.strftime('%Y-%m-%d %H:%M:%S')}   n={s.get('n', 0)}  (exact positions for {s['n_exact']})"]
        if "pearson" in s:
            lines += [f"# pearson {s['pearson']:.3f}  spearman {s['spearman']:.3f} (p={s['spearman_p']:.1e})  "
                      f"PY = {s['slope']:.3f} IDL + {s['offset']:.3f}   median PY/IDL {s['median_ratio']:.3f}  rms diff {s['rms_diff']:.3f}",
                      f"# mean IDL {s['mean_idl']:.3f}  mean PY {s['mean_py']:.3f}   best IDL {s['best_idl']:.3f} @row {s['argmax_idl']}"
                      f" (PY there {s['py_at_idl_best']:.3f})   best PY {s['best_py']:.3f} @row {s['argmax_py']} (IDL there {s['idl_at_py_best']:.3f})",
                      "# top-set overlap: " + "  ".join(f"top{k} {s[f'top{k}_overlap']:.2f}" for k in (5, 10, 20) if f"top{k}_overlap" in s)]
        lines.append("# iter   idl_score   py_score   py_raw   exact  k_idl")
        for r in self.rows:
            lines.append(f"{r['iter']:6d}  {_f(r['idl_score'])}  {_f(r['py_score'])}  {_f(r['py_raw'])}   "
                         f"{'Y' if r.get('exact_positions') else 'n'}   {r['k_idl']:.0f}")
        txt = "\n".join(lines) + "\n"
        with open(os.path.join(self.out_dir, "compare.txt"), "w") as f:
            f.write(txt)
        with open(os.path.join(self.out_dir, "compare_stats.json"), "w") as f:
            json.dump(s, f, indent=1)
        try:
            self.plot()
        except Exception as exc:                          # matplotlib optional
            self.log(f"  (plot skipped: {exc!r})")
        return txt

    def plot(self, path: Optional[str] = None) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        a = np.array([r["idl_score"] if r["idl_score"] is not None else np.nan for r in self.rows])
        b = np.array([r["py_score"] if r["py_score"] is not None else np.nan for r in self.rows])
        it = np.array([r["iter"] for r in self.rows])
        ex = np.array([bool(r.get("exact_positions")) for r in self.rows])
        s = self.stats()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        g = np.isfinite(a) & np.isfinite(b)
        ax[0].scatter(a[g & ~ex], b[g & ~ex], s=14, alpha=0.6, label="fresh azimuth")
        if ex.any():
            ax[0].scatter(a[g & ex], b[g & ex], s=18, alpha=0.9, marker="s", label="exact positions")
        lim = [np.nanmin(np.r_[a[g], b[g]]) - 0.3, np.nanmax(np.r_[a[g], b[g]]) + 0.3] if g.any() else [0, 1]
        ax[0].plot(lim, lim, "k:", lw=1)
        if "slope" in s:
            xx = np.linspace(lim[0], lim[1], 10)
            ax[0].plot(xx, s["slope"] * xx + s["offset"], "r-", lw=1, label=f"PY = {s['slope']:.2f} IDL + {s['offset']:.2f}")
            ax[0].set_title(f"per-evaluation score  (r={s['pearson']:.2f}, rho={s['spearman']:.2f}, n={s['n']})")
        ax[0].set_xlabel("IDL search score"); ax[0].set_ylabel("Python search score"); ax[0].legend(fontsize=8)
        order = np.argsort(it)
        ax[1].plot(it[order], np.fmax.accumulate(np.nan_to_num(a[order], nan=-np.inf)), label="IDL running best")
        ax[1].plot(it[order], np.fmax.accumulate(np.nan_to_num(b[order], nan=-np.inf)), label="Python running best (same configs)")
        ax[1].scatter(it, a, s=6, alpha=0.4); ax[1].scatter(it, b, s=6, alpha=0.4)
        ax[1].set_xlabel("IDL evaluation"); ax[1].set_ylabel("score"); ax[1].legend(fontsize=8)
        ax[1].set_title("running best of the same configuration sequence")
        d = b - a
        ax[2].hist(d[g], bins=25, color="grey")
        ax[2].axvline(0, color="k", lw=1)
        ax[2].set_xlabel("Python - IDL"); ax[2].set_title(f"difference (median ratio {s.get('median_ratio', float('nan')):.2f})")
        fig.suptitle(f"klip-tpe replay of {s['idl_run']} (annulus {self.annulus}, contrast {self.contrast:.1e})")
        fig.tight_layout()
        path = path or os.path.join(self.out_dir, "compare.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path


# ----------------------------------------------------------------------------
# matched Python run
# ----------------------------------------------------------------------------
def matched_run_config(run_setup_path: str, seed: Optional[int] = None, **overrides):
    """A :class:`~klip_tpe.runner.RunConfig` mirroring an IDL ``run_setup.txt``: annulus
    edges, ``n_iter``/``n_init``, forced contrast, validation ``n_valid``/``n_top``, TPE
    ``gamma``/``ncand``/``explore``/``pbest``/``p_local``, density mode.  Returns
    ``(config, info)`` where ``info`` lists the nights and the IDL seed."""
    from .runner import CalibrationConfig, RunConfig, ValidationConfig
    st = read_run_setup(run_setup_path)

    def nums(key, cast=float):
        v = st.get(key, "")
        out = []
        for tok in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", v):
            try:
                out.append(cast(float(tok)))
            except ValueError:
                pass
        return out
    edges = nums("ann_edges (px)")
    # "10000,2500,2500 / 500,500,500" -> per-annulus lists (a scalar stays a scalar)
    it_part, _, in_part = st.get("n_iter / n_init", "").partition("/")
    n_iter = [int(float(v)) for v in re.findall(r"\d+(?:\.\d*)?", it_part)] or [100]
    n_init = [int(float(v)) for v in re.findall(r"\d+(?:\.\d*)?", in_part.split("(")[0])] or [20]
    n_iter = n_iter[0] if len(n_iter) == 1 else n_iter
    n_init = n_init[0] if len(n_init) == 1 else n_init
    n_valid, n_top = nums("n_valid / n_top", int)[:2]
    gamma, ncand, explore = nums("gamma/ncand/explore")[:3]
    pbest = nums("seed_best_frac")[0] if nums("seed_best_frac") else 0.0
    p_local, n_elite = nums("p_local / n_elite")[:2]
    tpe_mv = int(nums("tpe_mv (density)", int)[0]) if nums("tpe_mv (density)", int) else 0
    uc = nums("use_contrast")
    cmax = nums("cmax_cal (ceiling)")
    nights = [int(v) for v in re.findall(r"\d+", st.get("nights (seq_values)", ""))]
    idl_seed = nums("seed / bench_tag", int)
    kw = dict(ann_edges=edges, n_iter=n_iter, n_init=n_init, seed=seed,
              blocks={0: "univariate", 1: "partitions", 2: "full"}.get(tpe_mv, "univariate"),
              pbest=float(pbest), p_local=float(p_local), n_elite=int(n_elite),
              gamma=float(gamma), ncand=int(ncand), explore_frac=float(explore),
              validation=ValidationConfig(n_top=int(n_top), n_valid=int(n_valid)),
              calibration=CalibrationConfig(forced=[v if v > 0 else 0.0 for v in uc[:max(len(edges) - 1, 1)]] if uc and any(v > 0 for v in uc) else None,
                                            ceiling=cmax[0] if cmax else None),
              bench_tag=f"match_{os.path.basename(os.path.dirname(os.path.abspath(run_setup_path)))}")
    # keep only fields RunConfig knows (older/newer setup files may differ)
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    kw = {k: v for k, v in kw.items() if k in fields}
    kw.update(overrides)
    info = {"nights": nights, "idl_seed": idl_seed[0] if idl_seed else None, "setup": st}
    return RunConfig(**kw), info


def compare_runs(idl_run_dir: str, py_run_dir: str, annulus: int = 1) -> Dict[str, Any]:
    """Side-by-side of an IDL run and a Python run on the same data: best-so-far curves,
    final (validated) winners and their parameters."""
    log = read_idl_log(os.path.join(idl_run_dir, "optimize_tpe_results.txt"))
    sel = [i for i in range(len(log.iter)) if int(log.annulus[i]) == annulus]
    idl_scores = np.array([log.score[i] for i in sel], float)
    py_rows = []
    p = os.path.join(py_run_dir, "results.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for l in f:
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("annulus") == annulus - 1:
                    py_rows.append(r)
    py_scores = np.array([np.nan if r.get("score") is None else r["score"] for r in py_rows], float)
    out = {"idl_n": int(len(idl_scores)), "py_n": int(len(py_scores)),
           "idl_best_curve": np.fmax.accumulate(np.nan_to_num(idl_scores, nan=-np.inf)).tolist(),
           "py_best_curve": np.fmax.accumulate(np.nan_to_num(py_scores, nan=-np.inf)).tolist() if len(py_scores) else []}
    # winners
    fs = os.path.join(idl_run_dir, f"annulus{annulus:02d}", "final_setup.txt")
    if os.path.exists(fs):
        out["idl_final"] = read_step_setup(fs)
    w = os.path.join(py_run_dir, f"annulus{annulus:02d}", "winner.json")
    if os.path.exists(w):
        out["py_winner"] = json.load(open(w))
    return out


def _f(v) -> str:
    return "    nan " if v is None or not np.isfinite(v) else f"{v:8.3f}"
