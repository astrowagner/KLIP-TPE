"""A run logs which klip-tpe it is: version, git commit of a checkout, and the package path."""
from __future__ import annotations

import os

from klip_tpe import __version__, version


def _pkg(tmp_path):
    pkg = tmp_path / "repo" / "klip_tpe"
    pkg.mkdir(parents=True)
    return pkg


def test_the_commit_of_a_checkout_is_read_without_git(tmp_path):
    pkg = _pkg(tmp_path)
    git = tmp_path / "repo" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("d2fbb7412345678901234567890123456789abcd\n")
    assert version.git_commit(str(pkg)) == "d2fbb74"


def test_packed_refs_a_detached_head_and_a_worktree(tmp_path):
    pkg = _pkg(tmp_path)
    git = tmp_path / "repo" / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs with: peeled\n"
                                     "a461c67aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa refs/heads/main\n")
    assert version.git_commit(str(pkg)) == "a461c67"
    (git / "HEAD").write_text("36a2101bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
    assert version.git_commit(str(pkg)) == "36a2101"
    # a worktree: .git is a file naming the real git dir, refs live in the common dir
    wt = tmp_path / "wt" / "klip_tpe"
    wt.mkdir(parents=True)
    gd = git / "worktrees" / "wt"
    gd.mkdir(parents=True)
    (tmp_path / "wt" / ".git").write_text(f"gitdir: {gd}\n")
    (gd / "HEAD").write_text("ref: refs/heads/main\n")
    (gd / "commondir").write_text("../..\n")
    assert version.git_commit(str(wt)) == "a461c67"


def test_an_installed_copy_says_so(tmp_path, monkeypatch):
    pkg = _pkg(tmp_path)
    assert version.git_commit(str(pkg)) is None
    monkeypatch.setattr(version, "_package_dir", lambda: str(pkg))
    line = version.describe()
    assert line.startswith(f"klip-tpe {__version__} (installed copy) from ") and line.endswith(str(pkg))


def test_this_checkout_describes_itself():
    line = version.describe()
    here = os.path.dirname(version.__file__)
    assert line.startswith(f"klip-tpe {__version__} (") and line.endswith(here)


def test_the_cli_prints_it(capsys):
    import pytest
    from klip_tpe import cli
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == version.describe()
