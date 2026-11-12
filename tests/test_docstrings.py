"""Check the package's documentation rule on every module brought up to it.

The rule is Kevin Wagner's of 2026-11-12, modeled on VIP's docstrings after Vivek Vijayakumar's
review.  Every function, method, class and script gets a numpydoc docstring, clear and direct for
a researcher who knows the field but not the code.  It opens with a one-line summary.  The
field's terms (PSF, FWHM, lambda/D, ADI, RDI, KL modes, field rotation) stand without a
definition, and every variable that is not a parameter is defined.  Each parameter, field,
option and output gets an entry whose type line carries the type, the shape and the default,
then one sentence with units, and a second or third only where a reader needs it.  The code's
history, such as the IDL predecessor it succeeds, goes under Notes.

What this test checks, on each module in ``DONE``:

* Every function, method and class has a docstring.  Functions nested inside other functions
  are local helpers and are left out.
* A function's docstring has a numpydoc ``Parameters`` section with an entry for each of its
  parameters (``self`` and ``cls`` aside), and each entry has 1 to 3 sentences.  A class's
  ``__init__`` may leave its parameters to the class docstring.
* A function that returns a value has a ``Returns`` section, a generator a ``Yields`` section,
  with 1 to 3 sentences per entry.  A property only needs its summary.
* A dataclass documents every field in an ``Attributes`` (or ``Parameters``) section.

``python tests/test_docstrings.py klip_tpe/MODULE.py`` lists what a module still lacks.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the modules brought up to the rule, checked here.  The list grows module by module.
DONE = [
    "klip_tpe/injection.py",
    "klip_tpe/klip.py",
    "klip_tpe/metrics.py",
    "tests/test_docstrings.py",
    "tests/test_klip_annular_doc.py",
]

_SECTION = re.compile(r"^\s*-{3,}\s*$")


def sections(doc: str) -> Dict[str, List[str]]:
    """Split a numpydoc docstring into its sections.

    Parameters
    ----------
    doc : str
        Docstring, already cleaned of its common indentation (``ast.get_docstring``).

    Returns
    -------
    out : dict of str to list of str
        Lines under each section title, such as ``"Parameters"``, the dashes left out.  Text
        before the first section is under ``""``.
    """
    out: Dict[str, List[str]] = {"": []}
    cur = ""
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and _SECTION.match(lines[i + 1]):
            cur = lines[i].strip()
            out[cur] = []
            i += 2
            continue
        out[cur].append(lines[i])
        i += 1
    return out


def entries(lines: List[str]) -> Dict[str, str]:
    """Read the entries of a numpydoc section, each name with its description.

    Parameters
    ----------
    lines : list of str
        Lines of the section, as :func:`sections` gives them.

    Returns
    -------
    out : dict of str to str
        Description of each entry, its lines joined by spaces.  ``a, b : float`` gives entries
        ``a`` and ``b``, and ``*args`` gives ``args``.
    """
    body = [ln for ln in lines]
    ind = min((len(ln) - len(ln.lstrip()) for ln in body if ln.strip()), default=0)
    out: Dict[str, str] = {}
    names: List[str] = []
    for ln in body:
        if not ln.strip():
            continue
        lead = len(ln) - len(ln.lstrip())
        if lead == ind:
            head = ln.strip().split(" : ", 1)[0].strip()
            names = [n.strip().lstrip("*") for n in head.split(",") if n.strip()]
            for n in names:
                out[n] = ""
        else:
            for n in names:
                out[n] = (out[n] + " " + ln.strip()).strip()
    return out


def n_sentences(text: str) -> int:
    """Count the sentences of a description, split after ``.``, ``!`` or ``?`` and a space.

    Parameters
    ----------
    text : str
        The description.

    Returns
    -------
    n : int
        Number of sentences, 0 for an empty description.
    """
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _params(fn: ast.FunctionDef) -> List[str]:
    """List a function's parameter names, ``self`` and ``cls`` left out.

    Parameters
    ----------
    fn : ast.FunctionDef
        The function's node.

    Returns
    -------
    names : list of str
        Positional-only, ordinary, keyword-only, ``*args`` and ``**kwargs`` names, in that order.
    """
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def _own_nodes(fn: ast.AST):
    """Walk the nodes of a function's body, its nested functions, classes and lambdas left out.

    Parameters
    ----------
    fn : ast.AST
        The function's node.

    Yields
    ------
    node : ast.AST
        Each node of the body that belongs to the function itself.
    """
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _returns_value(fn: ast.AST) -> Tuple[bool, bool]:
    """Tell whether a function returns a value and whether it is a generator.

    Parameters
    ----------
    fn : ast.AST
        The function's node.

    Returns
    -------
    ret : bool
        True when a ``return`` gives a value other than None.
    gen : bool
        True when the body has a ``yield`` or ``yield from``.
    """
    ret = gen = False
    for node in _own_nodes(fn):
        if isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            ret = True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            gen = True
    return ret, gen


def _decorators(fn: ast.AST) -> List[str]:
    """List the names of a function's or class's decorators.

    Parameters
    ----------
    fn : ast.AST
        The function's or class's node.

    Returns
    -------
    out : list of str
        Each decorator's last name, such as ``"property"`` for ``@property`` and ``"setter"`` for
        ``@x.setter``.
    """
    out = []
    for d in getattr(fn, "decorator_list", []):
        if isinstance(d, ast.Name):
            out.append(d.id)
        elif isinstance(d, ast.Attribute):
            out.append(d.attr)
        elif isinstance(d, ast.Call):
            f = d.func
            out.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


def _check_entries(where: str, sec: Dict[str, List[str]], title: str, need: List[str], problems: List[str]) -> None:
    """Check that a section has an entry of 1 to 3 sentences for each name needed.

    Parameters
    ----------
    where : str
        Location of the docstring, ``module:line name``, that starts each problem.
    sec : dict of str to list of str
        The docstring's sections, as :func:`sections` gives them.
    title : str
        Title of the section to check, such as ``"Parameters"``.
    need : list of str
        Names that must have an entry.
    problems : list of str
        List the problems found are appended to.
    """
    got = entries(sec.get(title, []))
    for n in need:
        if n not in got:
            problems.append(f"{where}: no {title} entry for {n!r}")
        else:
            k = n_sentences(got[n])
            if not 1 <= k <= 3:
                problems.append(f"{where}: {title} entry {n!r} has {k} sentences")


def _check_function(where: str, fn: ast.AST, problems: List[str], class_doc: Optional[str] = None) -> None:
    """Check a function's or method's docstring against the rule.

    Parameters
    ----------
    where : str
        Location of the function, ``module:line name``, that starts each problem.
    fn : ast.AST
        The function's node.
    problems : list of str
        List the problems found are appended to.
    class_doc : str or None, default None
        Docstring of the class a method belongs to, where an ``__init__`` without its own
        docstring may document its parameters.  None for a module-level function.
    """
    doc = ast.get_docstring(fn)
    params = _params(fn)
    if not doc:
        if fn.name == "__init__" and class_doc is not None:
            _check_entries(where, sections(class_doc), "Parameters", params, problems)
            return
        problems.append(f"{where}: no docstring")
        return
    sec = sections(doc)
    if not "".join(sec.get("", [])).strip():
        problems.append(f"{where}: no summary before the sections")
    decos = _decorators(fn)
    if "property" in decos or "setter" in decos or "cached_property" in decos:
        return
    if params:
        _check_entries(where, sec, "Parameters", params, problems)
    ret, gen = _returns_value(fn)
    if gen:
        if "Yields" not in sec:
            problems.append(f"{where}: a generator without a Yields section")
        else:
            _check_entries(where, sec, "Yields", list(entries(sec["Yields"])), problems)
    elif ret:
        if "Returns" not in sec:
            problems.append(f"{where}: returns a value but has no Returns section")
        else:
            got = entries(sec["Returns"])
            if not got:
                problems.append(f"{where}: an empty Returns section")
            _check_entries(where, sec, "Returns", list(got), problems)


def _is_dataclass(node: ast.ClassDef) -> bool:
    """Tell whether a class is decorated as a dataclass.

    Parameters
    ----------
    node : ast.ClassDef
        The class's node.

    Returns
    -------
    is_dc : bool
        True when one of its decorators is named ``dataclass``.
    """
    return "dataclass" in _decorators(node)


def _fields(node: ast.ClassDef) -> List[str]:
    """List a dataclass's fields, its ``ClassVar`` annotations left out.

    Parameters
    ----------
    node : ast.ClassDef
        The class's node.

    Returns
    -------
    out : list of str
        Names of the annotated class attributes, in order.
    """
    out = []
    for st in node.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            ann = ast.unparse(st.annotation) if hasattr(ast, "unparse") else ""
            if "ClassVar" not in ann:
                out.append(st.target.id)
    return out


def check_module(path: str) -> List[str]:
    """List everything a module lacks under the rule.

    Parameters
    ----------
    path : str
        Path of the module relative to the repository's top directory, such as
        ``"klip_tpe/klip.py"``.

    Returns
    -------
    problems : list of str
        One line per problem, ``module:line name: what is missing``, empty when the module follows
        the rule.
    """
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    problems: List[str] = []

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        where = f"{path}:{node.lineno} {prefix}{node.name}"
        cdoc = ast.get_docstring(node)
        if not cdoc:
            problems.append(f"{where}: no docstring")
        elif _is_dataclass(node):
            sec = sections(cdoc)
            title = "Attributes" if "Attributes" in sec else "Parameters"
            _check_entries(where, sec, title, _fields(node), problems)
        for st in node.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_function(f"{path}:{st.lineno} {prefix}{node.name}.{st.name}", st, problems, class_doc=cdoc)
            elif isinstance(st, ast.ClassDef):
                visit_class(st, f"{prefix}{node.name}.")

    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(f"{path}:{st.lineno} {st.name}", st, problems)
        elif isinstance(st, ast.ClassDef):
            visit_class(st, "")
    return problems


@pytest.mark.parametrize("path", DONE)
def test_a_finished_module_follows_the_documentation_rule(path):
    """Check each module in ``DONE`` against the rule.

    Parameters
    ----------
    path : str
        Path of the module, from ``DONE``.
    """
    problems = check_module(path)
    assert not problems, f"{len(problems)} problem(s):\n" + "\n".join(problems[:60])


def test_the_checker_finds_what_is_missing(tmp_path):
    """Check that the checker reports what a faulty module lacks.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory from pytest, where the faulty module is written.
    """
    bad = tmp_path / "bad.py"
    bad.write_text('''
def f(a, b=2):
    """Does a thing.

    Parameters
    ----------
    a : int
        The first.  It counts.  It matters.  It is long.
    """
    return a + b

class C:
    def m(self):
        return 1
''')
    old = globals()["ROOT"]
    try:
        globals()["ROOT"] = str(tmp_path)
        got = "\n".join(check_module("bad.py"))
    finally:
        globals()["ROOT"] = old
    assert "no Parameters entry for 'b'" in got and "'a' has 4 sentences" in got
    assert "returns a value but has no Returns section" in got
    assert "C: no docstring" in got and "C.m: no docstring" in got


if __name__ == "__main__":
    for p in sys.argv[1:]:
        found = check_module(p)
        print(f"{p}: {len(found)} problem(s)")
        for line in found:
            print("  " + line)
