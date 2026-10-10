"""Which klip-tpe is running: the version, the git commit of a source checkout, and where the
package is imported from.

A run logs this line at its start and ``klip-tpe --version`` prints it, so a log shows
whether the code that ran is the code in the repository.  A copy installed with ``pip
install .`` (not ``-e``) stays as it was installed when the repository is updated, and is
reported as an installed copy.
"""
from __future__ import annotations

import os
from typing import Optional

from . import __version__

__all__ = ["git_commit", "describe"]


def _package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def git_commit(package_dir: Optional[str] = None) -> Optional[str]:
    """The short commit of the git checkout holding the package, read from ``.git`` without
    running git.  None for an installed copy, or when it cannot be read."""
    root = os.path.dirname(package_dir or _package_dir())
    git = os.path.join(root, ".git")
    try:
        if os.path.isfile(git):                           # a worktree: "gitdir: <path>"
            with open(git) as f:
                line = f.read().strip()
            if not line.startswith("gitdir:"):
                return None
            git = os.path.normpath(os.path.join(root, line.split(":", 1)[1].strip()))
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:7] or None                       # a detached HEAD holds the commit itself
        ref = head.split(":", 1)[1].strip()
        common = git
        cd = os.path.join(git, "commondir")               # a worktree keeps refs in the main .git
        if os.path.isfile(cd):
            with open(cd) as f:
                common = os.path.normpath(os.path.join(git, f.read().strip()))
        for base in (git, common):
            p = os.path.join(base, ref)
            if os.path.isfile(p):
                with open(p) as f:
                    return f.read().strip()[:7] or None
        for base in (git, common):
            p = os.path.join(base, "packed-refs")
            if os.path.isfile(p):
                with open(p) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            return parts[0][:7]
    except OSError:
        return None
    return None


def describe() -> str:
    """``klip-tpe 0.1.0 (git d2fbb74) from /path/to/klip_tpe``, or ``(installed copy)`` in
    place of the commit when the package is not in a git checkout."""
    c = git_commit()
    return f"klip-tpe {__version__} ({'git ' + c if c else 'installed copy'}) from {_package_dir()}"
