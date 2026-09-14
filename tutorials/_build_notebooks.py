"""Author the tutorial notebooks from Python (cells separated by ``# %%`` / ``# %% [markdown]``
markers) so they stay diff-able; ``python tutorials/_build_notebooks.py`` rewrites the
``.ipynb`` files, ``--execute`` also runs them (nbconvert) and stores the outputs."""
import argparse
import glob
import os
import re
import subprocess
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))


def script_to_notebook(path: str) -> nbformat.NotebookNode:
    src = open(path).read()
    parts = re.split(r"^# %%(.*)$", src, flags=re.M)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = []
    # parts: [preamble, marker1, body1, marker2, body2, ...]
    for marker, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if not body.strip():
            continue
        if "[markdown]" in marker:
            text = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                             for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    nb.cells = cells
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    for src in sorted(glob.glob(os.path.join(HERE, "[0-9][0-9]_*.py"))):
        name = os.path.splitext(os.path.basename(src))[0]
        if a.only and not any(o in name for o in a.only):
            continue
        out = os.path.join(HERE, name + ".ipynb")
        nbformat.write(script_to_notebook(src), out)
        print("wrote", out)
        if a.execute:
            cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                   f"--ExecutePreprocessor.timeout={a.timeout}", out]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
