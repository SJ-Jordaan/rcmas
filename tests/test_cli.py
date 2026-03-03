"""Tests for CLI argument parsing and --demands integration."""

from __future__ import annotations

import pytest

from rcmas.cli import _parse_args, _parse_demands


class TestParseDemands:
    """Unit tests for _parse_demands helper."""

    def test_basic(self):
        assert _parse_demands("3,3,5,5", 4) == (3, 3, 5, 5)

    def test_single_agent(self):
        assert _parse_demands("7", 1) == (7,)

    def test_spaces_stripped(self):
        assert _parse_demands(" 2 , 4 , 6 ", 3) == (2, 4, 6)

    def test_wrong_count_raises(self):
        with pytest.raises(SystemExit, match="exactly 3"):
            _parse_demands("1,2", 3)

    def test_non_integer_raises(self):
        with pytest.raises(SystemExit, match="must be integers"):
            _parse_demands("1,abc", 2)


class TestCliDemandsArg:
    """Tests for --demands in full argument parsing."""

    def test_ibis_with_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args([
            "ibis", "--grid", str(grid), "--agents", "2", "--horizon", "1",
            "--demands", "1,1",
        ])
        assert args.demands == (1, 1)

    def test_ibis_without_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args([
            "ibis", "--grid", str(grid), "--agents", "2", "--horizon", "1",
        ])
        assert args.demands is None

    def test_co_with_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args([
            "co", "--grid", str(grid), "--agents", "2", "--horizon", "1",
            "--demands", "2,2",
        ])
        assert args.demands == (2, 2)

    def test_cegar_with_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args([
            "cegar", "--grid", str(grid), "--agents", "2", "--horizon", "1",
            "--demands", "1,1",
        ])
        assert args.demands == (1, 1)

    def test_demands_count_mismatch(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        with pytest.raises(SystemExit, match="exactly 2"):
            _parse_args([
                "ibis", "--grid", str(grid), "--agents", "2", "--horizon", "1",
                "--demands", "1,1,1",
            ])
