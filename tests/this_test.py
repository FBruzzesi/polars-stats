from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_import_prints_the_zen(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delitem(sys.modules, "polars_stats.this", raising=False)

    zen = importlib.import_module("polars_stats.this").ZEN

    assert zen.startswith("The Zen of polars-stats\n\n")
    assert capsys.readouterr().out == f"{zen}\n"
