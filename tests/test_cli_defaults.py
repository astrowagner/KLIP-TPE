"""The command line must start the same search the Python API does.

Until 2026-10-08 ``--blocks`` defaulted to ``partitions`` while ``RunConfig.blocks`` and the
``TPE`` class had moved to ``univariate`` (IDL addendum 2), so a ``klip-tpe near`` or
``klip-tpe generic`` call without the flag silently ran block-multivariate densities.
"""
import inspect

import pytest

from klip_tpe import cli
from klip_tpe.optimizers import TPE
from klip_tpe.runner import RunConfig

ARGV = {
    "near": ["near", "--root", "/data/NEAR2_py"],
    "generic": ["generic", "--cube", "cube.fits", "--angles", "angles.fits"],
}


@pytest.mark.parametrize("cmd", sorted(ARGV))
def test_cli_blocks_default_is_the_package_default(cmd):
    a = cli._build_parser().parse_args(ARGV[cmd])
    assert a.blocks == RunConfig.__dataclass_fields__["blocks"].default == "univariate"
    assert a.blocks == inspect.signature(TPE.__init__).parameters["blocks"].default


@pytest.mark.parametrize("cmd", sorted(ARGV))
def test_cli_blocks_still_selectable(cmd):
    a = cli._build_parser().parse_args(ARGV[cmd] + ["--blocks", "partitions"])
    assert a.blocks == "partitions"
