# Contributing

Issues and pull requests are welcome at https://github.com/astrowagner/klip-tpe.

* **Setup**: `pip install -e ".[all]"`, then `python -m pytest -q` (≈3 min, synthetic data
  only; no downloads).  New behaviour comes with a test in `tests/`.
* **Adapters** for other instruments go in `klip_tpe/instruments/` (see `generic.py` for
  the minimum: `load_*` → `Dataset`, `make_reducer`, `make_space`, `make_guard`,
  `default_config`); other PSF-subtraction engines in `klip_tpe/backends/` (subclass
  `KLIPReducer`, override `_subtract`, register in `reducer.reducer_class`).
* **Conventions** worth knowing before touching the numerics: angles rotate frames
  counter-clockwise to North-up; the star sits at `((nx-1)/2, (ny-1)/2)`; the clean term of
  the objective is clamped at ≥ 0; every proposal goes through the feasibility projection;
  only validated scores are reported.
* **Docs**: `docs/CLI.md` is generated (`python scripts/gen_cli_doc.py`); tutorials are
  authored as `tutorials/NN_name.py` and rebuilt with `python tutorials/_build_notebooks.py`.
* Keep `CHANGELOG.md` current.
