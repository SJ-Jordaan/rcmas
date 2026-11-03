"""Tests for the experiment pipeline of thesis Chapter 7 (``experiments/``).

The runner tests start real child processes and run the synthesisers on tiny
grids.  The asset and number tests use the examined log in
``results/examined/``: run on it, the scripts must reproduce the numbers of
the examined thesis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("z3")

from experiments import provenance  # noqa: E402
from experiments.config import Protocol, load_config, load_protocol  # noqa: E402
from experiments.results import load  # noqa: E402
from experiments.runner import (  # noqa: E402
    LogInUseError,
    MixedLogError,
    family_of,
    read_log,
    run_batch,
)
from experiments.scenarios import GRIDS_DIR, Scenario  # noqa: E402
from experiments.thesis_assets import (  # noqa: E402
    ASSETS,
    cell,
    check_thesis_inputs,
    render,
    write_tree,
)
from experiments.thesis_numbers import (  # noqa: E402
    Analysis,
    build_report,
    facts,
    find_reruns,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMINED_900 = ROOT / "results" / "examined" / "thesis-eval-900.jsonl"
EXAMINED_300 = ROOT / "results" / "examined" / "thesis-eval-300.jsonl"


def _sc(
    name: str, grid: str, n: int, alg: str, sym: bool = False, part: str = "orbit"
) -> Scenario:
    from rcmas.model import Territory

    path = os.path.join(GRIDS_DIR, grid)
    with open(path, encoding="utf-8") as f:
        S = len(Territory.from_ascii(f))
    return Scenario(
        suite="t",
        name=name,
        grid_path=path,
        num_agents=n,
        horizon=S // n,
        algorithm=alg,
        symmetry=sym,
        partition=part,
    )


# ---------------------------------------------------------------------------
# Configuration (thesis Section 7.2)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_thesis_protocol(self) -> None:
        cfg = ROOT / "experiments" / "thesis-eval.toml"
        proto = load_protocol(cfg)
        assert proto == Protocol(
            time_limit_s=900, runs=1, max_iters=30, demand=0, seed=0
        )
        scenarios = load_config(cfg)
        instances = {(s.grid_path, s.num_agents) for s in scenarios}
        grids = {s.grid_path for s in scenarios}
        assert len(instances) == 70
        assert len(grids) == 27
        assert len(scenarios) == 560
        configs = {(s.algorithm, s.symmetry) for s in scenarios}
        assert configs == {
            (a, sym)
            for a in ("ibis", "sibis", "qibis", "cegar-sibis")
            for sym in (False, True)
        }
        assert all(
            s.partition == "orbit" for s in scenarios if s.algorithm == "cegar-sibis"
        )
        # n in 2..6 dividing |T|, horizon m = |T|/n
        from rcmas.model import Territory

        for s in scenarios:
            with open(s.grid_path, encoding="utf-8") as f:
                S = len(Territory.from_ascii(f))
            assert 2 <= s.num_agents <= 6 and S % s.num_agents == 0
            assert s.horizon == S // s.num_agents
            assert 6 <= S <= 30

    def test_instances_table_and_smoke(self, tmp_path: Path) -> None:
        smoke = load_config(ROOT / "experiments" / "smoke.toml")
        assert len({(s.grid_path, s.num_agents) for s in smoke}) == 4
        assert len(smoke) == 32
        assert load_protocol(ROOT / "experiments" / "smoke.toml").time_limit_s == 30
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            '[[instances]]\ngrid = "symmetric/3x3.txt"\nagents = [2, 3]\n'
            '[parameters]\nalgorithms = ["sibis"]\nsymmetry = [false]\n'
        )
        sc = load_config(cfg)
        assert [(s.num_agents, s.horizon) for s in sc] == [(2, 4), (3, 3)]
        assert load_protocol(cfg) == Protocol()

    def test_unknown_protocol_key(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.toml"
        cfg.write_text("[protocol]\ntime_limit = 5\n")
        with pytest.raises(ValueError, match="unknown"):
            load_protocol(cfg)

    def test_reads_configs_without_tomllib(self) -> None:
        """Python 3.10 has no tomllib; the config reader must use tomli there.

        A fresh interpreter in which ``import tomllib`` fails, as on 3.10,
        loads the thesis configuration through the ``tomli`` module (stood in
        for by the real parser under that name).
        """
        import subprocess
        import sys

        code = (
            "import sys\n"
            "try:\n"
            "    import tomllib as parser\n"
            "except ModuleNotFoundError:\n"
            "    import tomli as parser\n"
            "sys.modules['tomli'] = parser\n"
            "sys.modules['tomllib'] = None\n"
            "from experiments.config import load_config, load_protocol\n"
            "import experiments.config as c\n"
            "assert c.tomllib is parser\n"
            "cfg = 'experiments/thesis-eval.toml'\n"
            "print(len(load_config(cfg)), load_protocol(cfg).time_limit_s)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["560", "900"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_collect(self, tmp_path: Path) -> None:
        cfg = ROOT / "experiments" / "smoke.toml"
        rec = provenance.collect(config_path=cfg, argv=["run.py", "--x", "a b"])
        assert rec["config_path"] == "experiments/smoke.toml"
        assert rec["config_sha256"] == provenance.sha256_file(cfg)
        assert rec["command"] == "python -m experiments.run --x 'a b'"
        assert rec["python"]["version"].count(".") == 2
        assert rec["z3_version"]
        assert rec["machine"]["logical_cores"] >= 1
        tool = rec["tool"]
        assert tool["version"]
        if tool["source"] == "git":
            assert len(tool["commit"]) == 40 and isinstance(tool["dirty"], bool)

    def test_enclosing_repository_is_not_the_tool(self, tmp_path: Path) -> None:
        import subprocess

        outer = tmp_path / "outer"
        tool = outer / "tool"
        tool.mkdir(parents=True)
        (outer / "f").write_text("x")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(outer)]
        try:
            subprocess.run(git + ["init", "-q"], check=True)
            subprocess.run(git + ["add", "f"], check=True)
            subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git not available")
        assert provenance.git_info(tool)["commit"] is None
        (tool / ".git-commit").write_text("a" * 40 + "\n")
        info = provenance.git_info(tool)
        assert info["commit"] == "a" * 40 and info["source"] == ".git-commit"
        assert provenance.git_info(outer)["source"] == "git"

    def test_workers_and_paths(self) -> None:
        assert provenance.physical_cores() >= 1
        assert provenance.default_workers() == max(1, provenance.physical_cores() - 1)
        assert provenance.relative_to_tool(
            os.path.join(GRIDS_DIR, "symmetric", "3x2.txt")
        ) == ("grids/symmetric/3x2.txt")
        assert family_of("grids/local_symmetry/L-4x4-2.txt") == "local_symmetry"
        assert (
            family_of("/Users/x/rcmas/experiments/../grids/asymmetric/a.txt")
            == "asymmetric"
        )


# ---------------------------------------------------------------------------
# Runner: one fresh process per run, wall-clock limit, certification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    out = tmp_path_factory.mktemp("sweep") / "log.jsonl"
    scenarios = [
        _sc("a_sibis", "symmetric/3x2.txt", 2, "sibis"),
        _sc("a_cegar", "symmetric/3x2.txt", 3, "cegar-sibis", sym=True),
        _sc("a_qibis", "disconnected/2x2-2x2.txt", 2, "qibis"),
        _sc("big_ibis", "symmetric/6x5.txt", 2, "ibis"),
        _sc("bad_alg_scenario", "symmetric/3x2.txt", 2, "ibis"),
    ]
    # An invalid horizon makes the tool raise inside the child.
    bad = scenarios[-1]
    scenarios[-1] = Scenario(
        **{**{f: getattr(bad, f) for f in bad.__dataclass_fields__}, "horizon": 2}
    )
    records = run_batch(
        scenarios,
        protocol=Protocol(time_limit_s=2),
        output_path=out,
        workers=3,
        progress=False,
        config_path=ROOT / "experiments" / "smoke.toml",
    )
    return out, records


class TestRunner:
    def test_log_layout(self, sweep: tuple[Path, list[dict]]) -> None:
        out, records = sweep
        lines = [json.loads(x) for x in out.read_text().splitlines()]
        assert lines[0]["kind"] == "session"
        assert lines[-1]["kind"] == "session_end" and lines[-1]["runs_written"] == 5
        header = lines[0]
        for key in (
            "tool",
            "python",
            "z3_version",
            "machine",
            "config_sha256",
            "command",
            "protocol",
        ):
            assert key in header
        assert header["protocol"]["time_limit_s"] == 2
        assert header["env"]["PYTHONHASHSEED"] == "0"
        runs = [x for x in lines if x["kind"] == "run"]
        assert len(runs) == 5 and len(records) == 5
        for r in runs:
            assert r["grid_path"].startswith("grids/")
            assert r["session"] == header["session"]
            assert r["tool_commit"] == header["tool"]["commit"]
            assert r["z3_version"] == header["z3_version"]

    def test_equilibria_are_certified(self, sweep: tuple[Path, list[dict]]) -> None:
        _, records = sweep
        by = {r["name"]: r for r in records}
        for name in ("a_sibis", "a_cegar", "a_qibis"):
            r = by[name]
            assert r["outcome"] == "ne" and r["found_ne"]
            assert r["certificate"]["status"] == "accepted"
            assert r["solved"]
            assert r["play"] and len(r["play"]) == r["horizon"]
            assert all(len(p) == r["num_agents"] for p in r["play"])
            assert r["welfare"] == sum(r["rewards"])
            assert r["certificate"]["best_responses"] == r["rewards"] or all(
                b <= x for b, x in zip(r["certificate"]["best_responses"], r["rewards"])
            )
            assert 0 <= r["time_s"] < 2
        cg = by["a_cegar"]
        assert cg["cegar_final_blocks"] >= 1 and cg["cegar_at_discrete"] in (
            True,
            False,
        )
        assert cg["orbit_blocks"] == cg["cegar_initial_blocks"]
        assert by["a_sibis"]["geo_group_order"] == 4  # the 3x2 rectangle has V4

    def test_timeout_and_error(self, sweep: tuple[Path, list[dict]]) -> None:
        _, records = sweep
        by = {r["name"]: r for r in records}
        to = by["big_ibis"]
        assert to["outcome"] == "t/o" and to["time_s"] == 2.0 and not to["solved"]
        assert to["elapsed_s"] is not None and to["elapsed_s"] < 2 + 5 + 3
        err = by["bad_alg_scenario"]
        assert err["outcome"] == "error" and "horizon" in err["reason"]
        assert not err["solved"] and err["certificate"] is None

    def test_resume_skips_done_runs_and_repairs_tail(
        self, sweep: tuple[Path, list[dict]], tmp_path: Path
    ) -> None:
        out, _ = sweep
        copy = tmp_path / "log.jsonl"
        text = out.read_text()
        copy.write_text(text + '{"kind": "run", "trunc')  # a cut-off last line
        sessions, runs, valid = read_log(copy)
        assert len(sessions) == 1 and len(runs) == 5 and valid == len(text.encode())
        scenarios = [
            _sc("a_sibis", "symmetric/3x2.txt", 2, "sibis"),
            _sc("new", "symmetric/3x2.txt", 3, "ibis"),
        ]
        recs = run_batch(
            scenarios,
            protocol=Protocol(time_limit_s=2),
            output_path=copy,
            workers=1,
            resume=True,
            progress=False,
            config_path=ROOT / "experiments" / "smoke.toml",
        )
        names = [r["name"] for r in recs]
        assert names.count("a_sibis") == 1 and "new" in names
        sessions, runs, valid = read_log(copy)
        assert len(sessions) == 2 and len(runs) == 6 and valid == copy.stat().st_size

    def test_resume_refuses_mixed_logs(
        self, sweep: tuple[Path, list[dict]], tmp_path: Path
    ) -> None:
        out, _ = sweep
        lines = out.read_text().splitlines()
        header = json.loads(lines[0])
        header["tool"]["commit"] = "0" * 40
        copy = tmp_path / "mixed.jsonl"
        copy.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
        sc = [_sc("a_sibis", "symmetric/3x2.txt", 2, "sibis")]
        cfg = ROOT / "experiments" / "smoke.toml"
        with pytest.raises(MixedLogError, match="tool_commit"):
            run_batch(
                sc,
                protocol=Protocol(time_limit_s=2),
                output_path=copy,
                resume=True,
                progress=False,
                config_path=cfg,
            )
        with pytest.raises(MixedLogError, match="protocol"):
            run_batch(
                sc,
                protocol=Protocol(time_limit_s=3),
                output_path=out,
                resume=True,
                progress=False,
                config_path=cfg,
            )
        with pytest.raises(FileExistsError):
            run_batch(
                sc,
                protocol=Protocol(time_limit_s=2),
                output_path=out,
                resume=False,
                progress=False,
                config_path=cfg,
            )

    @pytest.mark.skipif(os.name != "posix", reason="flock")
    def test_second_session_on_a_log_is_refused(
        self, sweep: tuple[Path, list[dict]], tmp_path: Path
    ) -> None:
        """While one process holds the log, another session must not start."""
        import subprocess
        import sys

        out, _ = sweep
        copy = tmp_path / "log.jsonl"
        copy.write_text("\n".join(out.read_text().splitlines()[:-3]) + "\n")
        before = copy.read_bytes()
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl, sys, time\n"
                "f = open(sys.argv[1], 'a')\n"
                "fcntl.flock(f, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "time.sleep(60)\n",
                str(copy),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        sc = [_sc("new", "symmetric/3x2.txt", 3, "ibis")]
        kw: dict[str, Any] = dict(
            protocol=Protocol(time_limit_s=2),
            output_path=copy,
            resume=True,
            progress=False,
            config_path=ROOT / "experiments" / "smoke.toml",
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"
            with pytest.raises(LogInUseError, match="another session"):
                run_batch(sc, **kw)
            assert copy.read_bytes() == before  # not read, repaired or appended
        finally:
            holder.kill()
            holder.wait()
        recs = run_batch(sc, **kw)  # the lock ends with the process
        assert "new" in [r["name"] for r in recs]

    def test_no_empty_log_is_left_behind(self, tmp_path: Path) -> None:
        log = tmp_path / "sub" / "never.jsonl"
        assert run_batch([], output_path=log, progress=False) == []
        assert not log.exists()

    def test_unknown_algorithm_rejected_up_front(self) -> None:
        with pytest.raises(ValueError, match="unknown algorithm"):
            run_batch([_sc("x", "symmetric/3x2.txt", 2, "nope")], progress=False)

    def test_results_loader_and_certify_cli(
        self, sweep: tuple[Path, list[dict]], tmp_path: Path
    ) -> None:
        out, _ = sweep
        d = load(out, run_number=None)
        assert len(d.runs) == 5
        assert sum(r.solved for r in d.runs) == 3
        assert d.time_limit_s == 2
        from rcmas.certify import run_logs

        certs = tmp_path / "certs.jsonl"
        assert run_logs([out], output=certs, stream=open(os.devnull, "w")) == 0
        d2 = load(out, certificates=certs, run_number=None)
        assert d2.recertified == 3 and sum(r.solved for r in d2.runs) == 3
        # A rejection in the re-certification file overrides the runner's certificate.
        rows = [json.loads(x) for x in certs.read_text().splitlines()]
        rows[0]["status"] = "rejected"
        certs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        d3 = load(out, certificates=certs, run_number=None)
        assert sum(r.solved for r in d3.runs) == 2
        assert any("REJECTED" in p for p in d3.problems)

    def test_analyze_reads_new_records(self, sweep: tuple[Path, list[dict]]) -> None:
        from experiments.analyze import load_results

        out, _ = sweep
        rows = load_results(str(out))
        assert len(rows) == 5
        assert sum(r.found_ne for r in rows) == 3
        assert {r.status for r in rows} == {"ok", "timeout", "error"}


# ---------------------------------------------------------------------------
# Stopping a sweep: no run may outlive its session
# ---------------------------------------------------------------------------

LONG_CONFIG = (
    '[suite]\nname = "long"\n[protocol]\ntime_limit_s = 120\n'
    '[[instances]]\ngrid = "symmetric/6x5.txt"\nagents = [2, 3]\n'
    '[parameters]\nalgorithms = ["ibis"]\nsymmetry = [false]\n'
)


def _process_table() -> dict[int, tuple[int, str]]:
    """``pid -> (ppid, state)`` for every process (``ps`` on Linux and macOS)."""
    import subprocess

    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,stat="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            table[int(parts[0])] = (int(parts[1]), parts[2])
    return table


def _live_descendants(root: int) -> set[int]:
    table = _process_table()
    found: set[int] = set()
    frontier = [root]
    while frontier:
        p = frontier.pop()
        for pid, (ppid, _) in table.items():
            if ppid == p and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return {p for p in found if not table[p][1].startswith("Z")}


def _alive(pids: set[int]) -> set[int]:
    table = _process_table()
    return {p for p in pids if p in table and not table[p][1].startswith("Z")}


def _wait_for(predicate: Any, timeout_s: float) -> bool:
    import time

    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.2)
    return bool(predicate())


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
class TestStopping:
    """Stopping the runner or reproduce.sh stops every run process.

    Each test starts the sweep in its own session, waits until two runs are
    being solved, stops it in one of the ways an author might, and checks
    that no process of the sweep is left after a few seconds.
    """

    def _start(self, cmd: list[str], tmp_path: Path) -> Any:
        import signal
        import subprocess

        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=open(tmp_path / "stdout", "w"),
            stderr=open(tmp_path / "stderr", "w"),
            start_new_session=True,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        return proc

    def _runs_started(self, proc: Any, count: int) -> set[int]:
        import time

        def spawned() -> bool:
            return len(self._workers(proc.pid)) >= count

        assert _wait_for(spawned, 60), "the runs did not start"
        time.sleep(3)  # past the imports, inside Z3
        return _live_descendants(proc.pid)

    @staticmethod
    def _workers(root: int) -> set[int]:
        import subprocess

        pids = _live_descendants(root)
        out = set()
        for p in pids:
            cmd = subprocess.run(
                ["ps", "-o", "command=", "-p", str(p)], capture_output=True, text=True
            ).stdout
            if "multiprocessing.spawn" in cmd:
                out.add(p)
        return out

    def _cleanup(self, proc: Any, pids: set[int]) -> None:
        import signal

        for p in _alive(pids | {proc.pid}):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=10)

    def _runner_cmd(self, tmp_path: Path) -> tuple[list[str], Path]:
        import sys

        cfg = tmp_path / "long.toml"
        cfg.write_text(LONG_CONFIG)
        log = tmp_path / "log.jsonl"
        cmd = [sys.executable, "-m", "experiments.run", "--config", str(cfg)]
        cmd += ["--output", str(log), "--workers", "2", "--resume", "--no-progress"]
        return cmd, log

    def test_sigterm_to_runner_kills_runs_and_ends_session(
        self, tmp_path: Path
    ) -> None:
        import signal

        cmd, log = self._runner_cmd(tmp_path)
        proc = self._start(cmd, tmp_path)
        pids: set[int] = set()
        try:
            pids = self._runs_started(proc, 2)
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=30) == 128 + signal.SIGTERM
            assert _wait_for(lambda: not _alive(pids), 15), _alive(pids)
            last = json.loads(log.read_text().splitlines()[-1])
            assert last["kind"] == "session_end"
            assert last["interrupted"] is True and last["signal"] == "SIGTERM"
            assert "Stopped by SIGTERM" in (tmp_path / "stderr").read_text()
        finally:
            self._cleanup(proc, pids)

    def test_runs_end_when_the_runner_is_killed(self, tmp_path: Path) -> None:
        """SIGKILL leaves no chance for cleanup; the runs notice it themselves."""
        import signal

        cmd, _ = self._runner_cmd(tmp_path)
        proc = self._start(cmd, tmp_path)
        pids: set[int] = set()
        try:
            pids = self._runs_started(proc, 2)
            workers = self._workers(proc.pid)
            assert len(workers) == 2
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
            assert _wait_for(lambda: not _alive(pids), 15), _alive(pids)
        finally:
            self._cleanup(proc, pids)

    def test_sigterm_to_reproduce_sh_stops_the_sweep(self, tmp_path: Path) -> None:
        import signal

        if not (ROOT / ".venv" / "bin" / "python").exists():
            pytest.skip("reproduce.sh needs the .venv of setup.sh")
        results = tmp_path / "results"
        cmd = ["bash", str(ROOT / "reproduce.sh"), "--smoke", "--only", "sweep"]
        cmd += ["--workers", "2", "--results-dir", str(results)]
        proc = self._start(cmd, tmp_path)
        pids: set[int] = set()
        try:
            pids = self._runs_started(proc, 2)
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=30) == 143
            assert _wait_for(lambda: not _alive(pids), 15), _alive(pids)
            log = results / "smoke-30.jsonl"
            last = json.loads(log.read_text().splitlines()[-1])
            assert last["kind"] == "session_end" and last["interrupted"] is True
        finally:
            self._cleanup(proc, pids)


# ---------------------------------------------------------------------------
# Thesis assets and numbers, checked against the examined thesis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def examined() -> Analysis:
    d = load(EXAMINED_900, require_certificate=False)
    rep = load(EXAMINED_300, require_certificate=False)
    return Analysis(d, rep)


class TestExaminedNumbers:
    """The scripts reproduce the published numbers from the examined log."""

    def test_counts(self, examined: Analysis) -> None:
        f = facts(examined)
        assert f["instances"] == 70 and f["runs"] == 560
        off = [f[f"solved[{x}]"] for x in ("S-IBIS", "CEGAR", "IBIS", "Q-IBIS")]
        on = [f[f"solved[{x}+sym]"] for x in ("S-IBIS", "CEGAR", "IBIS", "Q-IBIS")]
        assert off == [57, 51, 45, 39]
        assert on == [58, 53, 49, 37]
        assert round(f["median[S-IBIS]"], 1) == 8.1

    def test_symmetry_and_time(self, examined: Analysis) -> None:
        f = facts(examined)
        assert f["speedup_pairs"] == 182 and round(f["median_speedup"], 2) == 1.09
        assert (f["twice_as_fast"], f["twice_as_slow"]) == (35, 15)
        assert (f["only_with_symmetry"], f["only_without_symmetry"]) == (15, 10)
        assert (f["solved_runs"], f["solved_over_300s"], f["solved_at_bound"]) == (
            389,
            32,
            205,
        )
        assert f["qibis_nc_bound"] == 17 and f["max_iterations_ibis_sibis"] == 4
        assert (f["all_solved_up_to"], f["first_unsolved_size"]) == (22, 24)
        assert f["variation_pairs"] > 300

    def test_report_reproduces_sentences(
        self, examined: Analysis, tmp_path: Path
    ) -> None:
        thesis = tmp_path / "thesis"
        (thesis / "chapters" / "experiment").mkdir(parents=True)
        (thesis / "chapters" / "experiment" / "main.tex").write_text(
            "% \\rerun{in a comment}\n"
            "Without symmetry reduction, \\gls{sibis} solves \\rerun{57} of the 70 instances, "
            "CEGAR-NE \\rerun{51}, \\gls{ibis} \\rerun{45} and \\gls{qibis} \\rerun{39}.\n"
            "\\rerun{Over the 182 (instance, algorithm) pairs solved both with and without symmetry "
            "reduction, the median speed-up is 1.09, within the measured variation "
            "(Figure~\\ref{fig:exp:scatter}(a)).}\n"
            "\\rerun{In 35 pairs the run with symmetry reduction is at least twice as fast, and in 15 "
            "pairs it is at least twice as slow.}\n"
            "\\rerun{Something {nested}\n over two lines}\n"
        )
        occ = find_reruns(thesis)
        assert [o.text for o in occ][:4] == ["57", "51", "45", "39"]
        assert occ[0].count == 4 and occ[3].index == 4
        assert occ[-1].text == "Something {nested} over two lines"
        report, entries = build_report(examined, occ, [], {"results": "examined"})
        for e in entries[:6]:
            assert e["rule"] is not None and e["proposed"] == e["current"], e
        assert entries[-1]["rule"] is None
        assert "No rule matches" in report

    def test_full_thesis_markers_have_rules(self, examined: Analysis) -> None:
        thesis = ROOT.parent.parent / "thesis"
        if not (thesis / "chapters" / "experiment" / "main.tex").is_file():
            pytest.skip("no thesis checkout next to the tool")
        occ = find_reruns(thesis)
        assert len(occ) >= 50
        _, entries = build_report(examined, occ, [], {})
        assert all(e["rule"] is not None for e in entries)


class TestAssets:
    def test_examined_summary_table(self) -> None:
        d = load(EXAMINED_900, require_certificate=False)
        assets = render(d)
        tab = assets["tab-summary"]
        assert "S-IBIS & off & 57/70 & 13 & 8.1 &" in tab
        assert " & on & 37/70 & 33 & 18.5 & 5x5-shaped ($|T|=24$, $n=6$)" in tab
        full = assets["tab-full"]
        assert (
            "3x2 & 6 & 2 & 3 & 0.2 & 0.2 & 0.1 & 0.2 & 1.0 & 0.7 & 0.2 & 0.1 \\\\"
            in full
        )
        assert "\\emph{t/o} marks a timeout, \\emph{n/c} a run" in full
        assert "with a limit of 900 seconds per run" in full
        assert "(0,300) (62,300)" in assets["fig-cactus"]

    def test_uncertified_equilibria_are_not_solved(self) -> None:
        d = load(EXAMINED_900)  # certificates required: the examined log has none
        assert sum(r.solved for r in d.runs) == 0
        assert any("without a certificate" in p for p in d.problems)
        r = next(r for r in d.runs if r.found_ne)
        assert cell(r) == "unc"
        assert cell(None) == "--"

    def test_write_tree_matches_thesis_inputs(self, tmp_path: Path) -> None:
        d = load(EXAMINED_900, require_certificate=False)
        paths = write_tree(render(d), tmp_path)
        rel = sorted(p.relative_to(tmp_path).as_posix() for p in paths)
        assert rel == sorted(f"{ASSETS[n]}/{n}.tex" for n in ASSETS)
        thesis = tmp_path / "thesis"
        for sub, names in (
            (
                "chapters/experiment",
                ["tab-summary", "fig-cactus", "fig-scatter-sym", "fig-scatter-cegar"],
            ),
            ("appendices/experiment", ["tab-full"]),
        ):
            (thesis / sub).mkdir(parents=True)
            body = "".join(f"\\rerun{{\\input{{{sub}/{n}}}}}\n" for n in names)
            (thesis / sub / "main.tex").write_text(
                body + "% \\input{chapters/experiment/old}\n"
            )
        assert check_thesis_inputs(thesis) == (set(), set())
        (thesis / "chapters/experiment/main.tex").write_text(
            "\\input{chapters/experiment/fig-new}\n"
        )
        missing, extra = check_thesis_inputs(thesis)
        assert missing == {"chapters/experiment/fig-new"}
        assert "chapters/experiment/tab-summary" in extra
