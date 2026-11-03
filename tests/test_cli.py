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
        args = _parse_args(
            [
                "ibis",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "1",
                "--demands",
                "1,1",
            ]
        )
        assert args.demands == (1, 1)

    def test_ibis_without_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args(
            [
                "ibis",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "1",
            ]
        )
        assert args.demands is None

    def test_co_with_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args(
            [
                "co",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "1",
                "--demands",
                "2,2",
            ]
        )
        assert args.demands == (2, 2)

    def test_cegar_with_demands(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args(
            [
                "cegar",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--horizon",
                "1",
                "--demands",
                "1,1",
            ]
        )
        assert args.demands == (1, 1)

    def test_demands_count_mismatch(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        with pytest.raises(SystemExit, match="exactly 2"):
            _parse_args(
                [
                    "ibis",
                    "--grid",
                    str(grid),
                    "--agents",
                    "2",
                    "--horizon",
                    "1",
                    "--demands",
                    "1,1,1",
                ]
            )


class TestCliDefaults:
    """Defaults follow Chapter 7: horizon m = floor(|T|/|Agt|), N = 30."""

    def test_horizon_defaults_to_none_and_bound_to_30(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        for cmd in ("ibis", "sibis", "qibis", "cegar"):
            args = _parse_args([cmd, "--grid", str(grid), "--agents", "2"])
            assert args.horizon is None
            assert args.max_iters == 30

    def test_cegar_defaults(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args(["cegar", "--grid", str(grid), "--agents", "2"])
        assert args.partition == "orbit"
        assert args.synthesiser == "sibis"

    def test_qibis_seed_and_top_k(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        args = _parse_args(
            [
                "qibis",
                "--grid",
                str(grid),
                "--agents",
                "2",
                "--seed",
                "7",
                "--top-k",
                "1",
            ]
        )
        assert (args.seed, args.top_k) == (7, 1)

    def test_negative_values_rejected(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        for extra in (["--horizon", "-1"], ["--max-iters", "-1"]):
            with pytest.raises(SystemExit):
                _parse_args(["ibis", "--grid", str(grid), "--agents", "2", *extra])


class TestCliRuns:
    """End-to-end runs of main() on tiny grids."""

    GRID = "...\n...\n...\n"

    def _grid(self, tmp_path):
        grid = tmp_path / "grid.txt"
        grid.write_text(self.GRID)
        return str(grid)

    @pytest.mark.parametrize("cmd", ["ibis", "sibis", "qibis", "cegar"])
    def test_default_horizon_returns_certified_equilibrium(self, tmp_path, capsys, cmd):
        from rcmas.certify import certify
        from rcmas.cli import main
        from rcmas.model import Territory

        grid = self._grid(tmp_path)
        rc = main([cmd, "--grid", grid, "--agents", "2"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "outcome=ne" in out
        # The play is printed in thesis numbering: sectors 1..|T|.
        rows = [ln for ln in out.splitlines() if ln.strip().startswith("agent ")]
        assert len(rows) == 2
        claims = [[int(x) - 1 for x in ln.split(":")[1].split(",")] for ln in rows]
        assert all(len(c) == 4 for c in claims)  # m = floor(9/2) = 4 rounds
        actions = [[claims[a][t] for a in range(2)] for t in range(4)]
        territory = Territory.from_ascii(self.GRID.splitlines())
        assert certify(territory, 2, None, None, actions).accepted

    def test_wrong_horizon_is_rejected(self, tmp_path):
        from rcmas.cli import main

        with pytest.raises(SystemExit, match="horizon must be m"):
            main(
                [
                    "sibis",
                    "--grid",
                    self._grid(tmp_path),
                    "--agents",
                    "2",
                    "--horizon",
                    "3",
                ]
            )

    def test_no_equilibrium_gives_exit_code_2(self, tmp_path, capsys):
        """Demands above m: no satisfying strategy exists (outcome uns)."""
        from rcmas.cli import main

        rc = main(
            [
                "sibis",
                "--grid",
                self._grid(tmp_path),
                "--agents",
                "2",
                "--demands",
                "5,5",
            ]
        )
        assert rc == 2
        assert "outcome=uns" in capsys.readouterr().out

    def test_co_uses_default_horizon(self, tmp_path, capsys):
        from rcmas.cli import main

        rc = main(["co", "--grid", self._grid(tmp_path), "--agents", "3", "--render"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "horizon=3" in out and "welfare=9" in out
