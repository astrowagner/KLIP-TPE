#!/usr/bin/env python
"""Write docs/CLI.md from the argparse help of every klip-tpe sub-command."""
import contextlib
import io
import os
import sys

from klip_tpe import cli

COMMANDS = ["generic", "near", "resume", "extend", "plots", "replay", "compare", "testbed"]
INTRO = """# Command-line reference

Generated from `klip-tpe <command> --help` (`python scripts/gen_cli_doc.py`).  `near` and
`generic` share the same search / protocol arguments; `--instrument near | nomic | generic`
selects the data adapter (`klip-tpe generic` is `near --instrument generic`).  `resume` and
`extend` take the same data arguments as the run they continue.

"""

def main():
    out = [INTRO]
    for c in COMMANDS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
            cli.main([c, "--help"])
        out.append(f"## `klip-tpe {c}`\n\n```\n{buf.getvalue().strip()}\n```\n")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "CLI.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("wrote", os.path.normpath(path))

if __name__ == "__main__":
    main()
