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

    @pytest.mark.parametrize(
        "extra", [[], ["--symmetry"], ["--render", "--dump-model"]]
    )
    def test_co_with_more_agents_than_sectors(self, tmp_path, capsys, extra):
        """|Agt| > |T|: m = 0, and s_0 is a satisfying, optimal play."""
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(["co", "--grid", str(grid), "--agents", "5", *extra])
        out = capsys.readouterr().out
        assert rc == 0
        assert "sat=True" in out and "horizon=0" in out
        assert "rewards=(0, 0, 0, 0, 0) welfare=0" in out

    def test_co_horizon_zero_with_positive_demand_is_unsat(self, tmp_path, capsys):
        from rcmas.cli import main

        grid = tmp_path / "grid.txt"
        grid.write_text("..\n..\n")
        rc = main(
            ["co", "--grid", str(grid), "--agents", "5", "--demands", "0,0,1,0,0"]
        )
        assert rc == 2
        assert "sat=False reason=unsat horizon=0" in capsys.readouterr().out

    def test_co_uses_default_horizon(self, tmp_path, capsys):
        from rcmas.cli import main

        rc = main(["co", "--grid", self._grid(tmp_path), "--agents", "3", "--render"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "horizon=3" in out and "welfare=9" in out


class TestCliProtocol:
    """``--max-iters none``, ``--initial`` and the CEGAR-NE bounds (Chapter 7 protocol)."""

    def _grid(self, tmp_path, text="...\n...\n"):
        grid = tmp_path / "grid.txt"
        grid.write_text(text)
        return str(grid)

    def test_parse_bounds_and_initial(self, tmp_path):
        g = self._grid(tmp_path)
        for cmd in ("ibis", "sibis", "qibis", "cegar"):
            base = [cmd, "--grid", g, "--agents", "2"]
            assert _parse_args(base).initial == "satisfying"
            args = _parse_args(
                base + ["--max-iters", "none", "--initial", "collective"]
            )
            assert args.max_iters is None and args.initial == "collective"
            assert _parse_args(base + ["--max-iters", "None"]).max_iters is None
            assert _parse_args(base + ["--max-iters", "0"]).max_iters == 0
            for bad in (["--max-iters", "x"], ["--initial", "best"]):
                with pytest.raises(SystemExit):
                    _parse_args(base + bad)
        args = _parse_args(["cegar", "--grid", g, "--agents", "2"])
        assert args.inner_max_iters is ... and args.discrete_max_iters is ...
        args = _parse_args(
            [
                "cegar",
                "--grid",
                g,
                "--agents",
                "2",
                "--inner-max-iters",
                "30",
                "--discrete-max-iters",
                "none",
            ]
        )
        assert args.inner_max_iters == 30 and args.discrete_max_iters is None

    def test_parse_inner_initial(self, tmp_path):
        """--inner-initial: the coarse initial query of CEGAR-NE (default: --initial)."""
        g = self._grid(tmp_path)
        base = ["cegar", "--grid", g, "--agents", "2"]
        assert _parse_args(base).inner_initial is ...
        args = _parse_args(
            base + ["--initial", "collective", "--inner-initial", "satisfying"]
        )
        assert args.initial == "collective" and args.inner_initial == "satisfying"
        with pytest.raises(SystemExit):
            _parse_args(base + ["--inner-initial", "best"])
        with pytest.raises(SystemExit):  # a CEGAR-NE option only
            _parse_args(
                ["sibis", "--grid", g, "--agents", "2", "--inner-initial", "satisfying"]
            )
        assert _parse_args(["sibis", "--grid", g, "--agents", "2"]).inner_initial is ...

    @pytest.mark.parametrize(
        "extra, shown",
        [
            ([], "initial=collective inner_initial=collective"),
            (
                ["--inner-initial", "satisfying"],
                "initial=collective inner_initial=satisfying",
            ),
        ],
    )
    def test_cegar_reports_both_initial_queries(self, tmp_path, capsys, extra, shown):
        from rcmas.cli import main

        g = self._grid(tmp_path, "....\n....\n")
        code = main(
            ["cegar", "--grid", g, "--agents", "2", "--initial", "collective", *extra]
        )
        out = capsys.readouterr().out
        assert code == 0 and "outcome=ne" in out and shown in out

    @pytest.mark.parametrize("cmd", ["ibis", "sibis", "qibis", "cegar"])
    def test_runs_with_the_thesis_protocol(self, tmp_path, capsys, cmd):
        from rcmas.cli import main

        g = self._grid(tmp_path)
        extra = (
            ["--inner-max-iters", "30", "--discrete-max-iters", "none"]
            if cmd == "cegar"
            else []
        )
        code = main(
            [
                cmd,
                "--grid",
                g,
                "--agents",
                "2",
                "--max-iters",
                "none",
                "--initial",
                "collective",
                *extra,
            ]
        )
        out = capsys.readouterr().out
        assert code == 0 and "outcome=ne" in out and "initial=collective" in out
        if cmd != "cegar":  # CEGAR-NE reports the lift of an abstract initial play
            assert "initial_rewards=(3, 3)" in out and "returned_initial=True" in out
