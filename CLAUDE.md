# rcmas — Code Context

## Overview

Python 3.10+, Z3 SMT solver. Nash equilibrium synthesis for region-control multi-agent systems.
This directory is a git submodule (`github.com/SJ-Jordaan/rcmas`). Commits here must be pushed separately.
The code follows the PhD thesis (`thesis/chapters/*`, Chapters 3-7); module docstrings cite thesis
definitions by number (e.g. "Definition 4.4.2"). Section 7.1 of the thesis describes this implementation;
keep the two in agreement.

## Setup

```bash
./setup.sh               # .venv from uv.lock (or requirements-lock.txt with pip) + tests; --no-tests skips them
source .venv/bin/activate
pytest -q
```

z3-solver is pinned to 5.3.3.0 (`pyproject.toml`, `uv.lock`, `requirements-lock.txt`); Chapter 7's
results depend on the solver version.

## Numbering

- Sectors `0..|T|-1` in the enumeration of Remark 3.1.1: **bottom row first**, left to right.
  Grid files list rows top to bottom, so the last line of a file holds sectors 0, 1, ...
  `Coord(x, y)` keeps the grid-file position (`y` = line number from the top).
- Agents `0..n-1`. Thesis sector `i` = tool `i-1`; thesis agent `a` = tool `a-1`. The CLI prints plays
  in thesis numbering.
- `UNOWNED = -1` (the thesis uses 0 for an uncontrolled sector).
- Blocks of a partition are numbered by their smallest sector (Section 6.1).

## Module map

| Module | Thesis | Role |
|--------|--------|------|
| `model.py` | Ch3 (Defs 3.1.1-3.9.2) | Territory (bottom-row-first order, `neighbor_map`), State, evolve, components, rewards (`rewards_from_owner`, weighted for abstract games, Def 6.1.2(4)) |
| `strategy.py` | Secs 3.8, 4.4 (Defs 4.2.5, 4.4.2) | `Play` (validated, hashable), `is_live`, `free_sector`, `default_action`, `CanonicalCompletion`; legacy Policy helpers |
| `smt_variables.py` | Secs 4.2, 4.3, 4.7 | Z3 variables: `owner`, `action`, witness `member/root/prefix/level`, `reward`, `unocc`; `neighbors`/`weights` for abstract games |
| `smt_constraints.py` | Secs 4.2-4.7, Ch5 | `[Init]`, `[Evol]`, `[Prot]`, `[Dom]`, `[Wit]`, `[Rew]`, `[phi_sat]`, implied bound `reward_a <= R_max`, opponent constraint `[Fix^a_pi]` (Def 4.7.1), `[Cand]`; symmetry constraints (`symmetry_plan_constraint`; the local constraint restricts agent 1's first action only) |
| `smt_objectives.py` | Secs 4.6, 4.7 | `collective_objective` (Def 4.6.1), `individual_objective` |
| `smt_solve.py` | Ch4, Sec 7.1 | Cached formula F (`build_base_encoding`, no symmetry inside), `find_feasible_from_base`, BR query `best_response_from_base`, `check_improvement_from_base`, S-IBIS binary search, `symmetry_constraints` (initial queries only); Z3 unknown -> reason `unknown` |
| `ibis.py` | Algs 1, 2 (Secs 4.7, 4.8) | IBIS / S-IBIS on a single play; `run_ibis_loop`; outcomes `ne`/`n/c`/`unk`/`uns` |
| `qlearning.py` | Secs 2.4, 4.9 | Q-IBIS learner (index-based TrainQ, shaped reward, epsilon-greedy, TopK) + self-play trainer for `rcmas train` |
| `qibis.py` | Alg 3 (Sec 4.9) | Q-IBIS: restricted queries plus a certification round; `QibisConfig` holds the RL hyperparameters and seed |
| `symmetry.py` | Ch5 (Defs 5.1.1-5.2.3, Thm 5.3.2) | `H_geo` (geometric subgroup, not Aut(T)), orbits/Reps, local `BP(H_R)`, component copy classes and group H, `SymmetryBreaking` plan (`symmetry_breaking_plan`, `.admits`) |
| `abstraction.py` | Defs 6.1.1-6.3.3, Sec 6.5 | `Partition`, `orbit_partition` (orbits of the group the Ch5 constraints use), `AbstractRCMAS` (block weights/adjacency, `rewards_of`), weighted-graph symmetry, `lift_play`, deviation set, `refine_partition`, `bfs_bisect` |
| `cegar.py` | Alg 4 (Ch6) | CEGAR-NE: SynthesiseNE, lift, satisfaction check, `verify_ne` (one query per agent, no symmetry), refine |
| `certify.py` | Sec 7.5 | Independent pure-Python certifier (own replay, phi_sat, rewards and canonical completion, memoised BR search); `rcmas certify --results` |
| `cli.py` | Sec 7.1 | CLI: co, ibis, sibis, qibis, cegar, train, certify |
| `experiments/runner.py` | Sec 7.2 | One fresh spawned process per run, wall-clock limit enforced by the parent, inline certification, provenance header, resumable JSONL |
| `experiments/thesis_assets.py` | Table 7.1, Figs 7.2-7.3, Table A.1 | Generates the `.tex` files the thesis inputs |
| `experiments/thesis_numbers.py` | Ch7 | Report with new values for every `\rerun{...}` marker of the thesis |

## Key types

- `Territory` — frozen, ordered sectors parsed from ASCII (`Territory.from_ascii()`), bottom row first
- `State` — frozen control map at a timestep (`owner_by_index`)
- `Play` — states and action profiles of a run (Definition 4.2.5); `CanonicalCompletion(play)` — Definition 4.4.2
- `BaseEncoding` — cached formula F (`variables`, `constraints`, `max_reward`, `rewards_of`)
- `SmtSolution` — query result (`is_sat`, `reason` sat|unsat|unknown, `play`, `payoff_by_agent` = true rewards, `encoded_reward_by_agent` = model values)
- `IbisResult` / `QibisResult` — `outcome` (ne|n/c|unk|uns), `reason`, `play`, `payoff_by_agent`, `iterations` (index i of the last iteration, N at the bound), `trace`, `completion()`
- `CegarResult` — `outcome` (ne|fail|unk|t/o|n/c), `at_discrete`, `iterations` (Alg 4 loop), `final_partition_size`, `inner_iterations`, `trace`
- `Partition`, `AbstractRCMAS` — abstraction (Chapter 6)
- `SymmetryInfo`, `ComponentInfo`, `SymmetryBreaking` — symmetry (Chapter 5)
- `CertResult` — certifier verdict (`accepted`, `reason`, `rewards`, `best_responses`)

## Code conventions

- **Formatter**: black (via pre-commit)
- **Import order**: isort with black profile
- **Linter**: flake8
- **Type checker**: mypy (--ignore-missing-imports)
- **Data classes**: `@dataclass(frozen=True, slots=True)` for immutable value objects
- **Type hints**: comprehensive on all function signatures. Z3 types typed as `Any`.
- **Z3 imports**: lazily imported inside functions, not at module level
- **Docstrings**: module-level docstrings reference thesis definitions. Function docstrings are prose.
- **Constants**: `UNOWNED = -1` (model.py)

## Testing

- **Framework**: pytest >= 8.0
- **Run**: `pytest -q` from repo root (with venv activated); the full suite takes about 3 minutes
- **Test files mirror source**: `tests/test_<module>.py` for each `rcmas/<module>.py`
- **Oracles**: `tests/brute_force.py` (exhaustive deviations against the canonical completion,
  `best_deviation`, `is_equilibrium`); `tests/test_symmetry.py` exports symmetry oracle helpers
- **Shared fixtures** in `conftest.py`: territory_1x1, territory_2x1, territory_2x2, territory_3x3, territory_L_shape, territory_disconnected, state_2x2, state_3x3
- **Pattern**: use small grids (1x1 to 3x3) for fast Z3 solves; every returned equilibrium in a test is checked by brute force
- **Z3 state**: Z3's models depend on process-wide state, so a test must not assume which of several models Z3 returns
- **Z3 guard**: `pytest.importorskip("z3")` in SMT test files
- `tests/test_experiments.py` spawns child processes (the runner) and reads `results/examined/`

## CLI

Entry point: `rcmas = "rcmas.cli:main"`. All subcommands except `certify` require `--grid` and `--agents`.
`--horizon` defaults to `m = floor(|T|/|Agt|)`, the only value the synthesisers accept; `--max-iters`
(the iteration bound N) defaults to 30. Exit code 0 only for outcome `ne` (or a sat `co`).

| Command | Description | Key flags |
|---------|-------------|-----------|
| `rcmas co` | Collectively optimal play, smt(F, f) | `--symmetry`, `--render`, `--dump-model`, `--timeout-ms` |
| `rcmas ibis` | IBIS (Algorithm 1) | `--symmetry`, `--max-iters`, `--timeout-ms`, `--progress`, `--timing`, `--render` |
| `rcmas sibis` | S-IBIS (Algorithm 2) | Same as ibis |
| `rcmas qibis` | Q-IBIS (Algorithm 3) | Same as ibis, plus `--seed`, `--top-k`; other RL hyperparameters live in `QibisConfig` |
| `rcmas cegar` | CEGAR-NE (Algorithm 4) | `--partition` (default orbit), `--synthesiser` (default sibis), `--max-iters` = inner N, `--seed`, `--symmetry` |
| `rcmas train` | Q-learning self-play | Hard-coded: 2000 episodes |
| `rcmas certify` | Re-check logged equilibria without SMT | `--results LOG.jsonl`, `--grid-root`, `--timeout`, `--output`, `--verbose` |

## Reproducing Chapter 7

```bash
./setup.sh --no-tests && ./reproduce.sh [--workers N]   # sweep, certify, encoding, repeat, assets, numbers
./reproduce.sh --smoke                                   # ~2 min end-to-end check, outputs in results/smoke/
```

Benchmark: `experiments/thesis-eval.toml`. Logs of the examined thesis: `results/examined/` (no plays, so
not certifiable; read with `--no-certificate-required`); older pre-revision logs in `results/examined/legacy/`.

## Grid format

ASCII text: `.` = accessible sector, anything else = inaccessible. Benchmark files in `grids/symmetric/`,
`grids/local_symmetry/`, `grids/disconnected/`, `grids/asymmetric/`.

## What's NOT implemented yet

- **Epsilon-NE**: no epsilon parameter in NE checking
- **No-op/pass action**: agents must claim every round

## Dependencies

Runtime: `z3-solver==5.3.3.0`, `tomli` on Python < 3.11. Dev: `pytest>=8.0`. Pre-commit: black, isort, flake8, mypy.
