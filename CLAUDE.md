# rcmas — Code Context

## Overview

Python 3.10+, Z3 SMT solver. Nash equilibrium synthesis for region-control multi-agent systems.
This directory is a git submodule (`github.com/SJ-Jordaan/rcmas`). Commits here must be pushed separately.

## Setup

```bash
./setup.sh    # or: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest -q     # 295 tests
```

## Module map

| Module | Paper Defs | Role |
|--------|-----------|------|
| `model.py` | Def 1-8 | Territory, State, evolve, adjacency, connected components, scores |
| `strategy.py` | Def 9-14 | Policy types (StateKey, Policy), reward, freeze/hash, fallback action |
| `smt_variables.py` | Def 15 | Z3 variable arrays: owner, action, adj, cr, size, payoff |
| `smt_constraints.py` | Def 16-23 | One function per constraint + structural constraints (fixed policy, candidates, symmetry breaking with demand-class lex-leader) |
| `smt_objectives.py` | Def 24-25 | Qualitative (sum) and quantitative (individual) objectives |
| `smt_solve.py` | — | Solver assembly: compose constraints, call Z3, extract solution |
| `ibis.py` | Alg 1 | Iterative best-response NE search |
| `qlearning.py` | Sec 5.2 | Tabular Q-learning, self-play trainer, reward shaping |
| `qibis.py` | Alg 2 | Q-learning-guided IBIS with RL bootstrap |
| `symmetry.py` | EUMAS Sec 3-4 | Automorphism detection, orbits, canonical states, demand classes (`demand_classes`) |
| `abstraction.py` | EUMAS Def 8-12 | Partition, AbstractRCMAS, lifting, deviation set, refinement |
| `cegar.py` | EUMAS Alg 1 | CEGAR-NE loop: abstract → synthesise → lift → verify → refine |
| `cli.py` | — | CLI entry point: co, ibis, qibis, cegar, train |

## Key types

- `Coord(x, y)` — frozen dataclass, grid coordinate
- `Territory` — frozen, ordered sectors parsed from ASCII (`Territory.from_ascii()`)
- `State` — frozen, ownership map at a timestep. `UNOWNED = -1` (paper uses 0)
- `Policy = dict[StateKey, Coord | None]` — state-to-action mapping
- `SmtVariables` — container for all Z3 variable arrays
- `SmtSolution` — solver result (is_sat, final_state, actions_by_round, debug fields)
- `SymmetryInfo` — automorphisms, orbits, representatives
- `Partition` — blocks + membership mapping for territory abstraction
- `AbstractRCMAS` — abstract game: synthetic territory, weights, abstract adjacency
- `LiftedStrategy` — concrete strategy obtained by lifting an abstract solution
- `CegarResult` — result of the CEGAR-NE loop (sat, reason, payoff, iterations)

## Code conventions

- **Formatter**: black (via pre-commit)
- **Import order**: isort with black profile
- **Linter**: flake8
- **Type checker**: mypy (--ignore-missing-imports)
- **Data classes**: `@dataclass(frozen=True, slots=True)` for immutable value objects
- **Type hints**: comprehensive on all function signatures. Z3 types typed as `Any`.
- **Z3 imports**: lazily imported inside functions, not at module level
- **Docstrings**: module-level docstrings reference paper definitions. Function docstrings are prose.
- **Constants**: `UNOWNED = -1` (model.py)

## Testing

- **Framework**: pytest >= 8.0
- **Run**: `pytest -q` from repo root (with venv activated)
- **Test files mirror source**: `tests/test_<module>.py` for each `rcmas/<module>.py`
- **Shared fixtures** in `conftest.py`: territory_1x1, territory_2x1, territory_2x2, territory_3x3, territory_L_shape, territory_disconnected, state_2x2, state_3x3
- **Pattern**: use small grids (1x1 to 3x3) for fast Z3 solves
- **Z3 guard**: `pytest.importorskip("z3")` in SMT test files
- **Error testing**: `pytest.raises(ValueError)`, `pytest.raises(CollisionError)`
- **Oracle tests**: brute-force enumeration on tiny instances to verify SMT correctness

## CLI

Entry point: `rcmas = "rcmas.cli:main"`. All subcommands require `--grid`, `--agents`, `--horizon`.

| Command | Description | Key flags |
|---------|-------------|-----------|
| `rcmas co` | Collectively optimal strategy | `--symmetry`, `--render`, `--dump-model`, `--timeout-ms` |
| `rcmas ibis` | IBIS NE synthesis | `--symmetry`, `--max-iters`, `--progress`, `--timing` |
| `rcmas qibis` | Q-IBIS NE synthesis | Same as ibis (RL hyperparams not exposed on CLI) |
| `rcmas cegar` | CEGAR-NE abstraction refinement | `--partition orbit\|discrete`, `--symmetry`, `--max-iters`, `--progress`, `--timing` |
| `rcmas train` | Q-learning self-play | Hard-coded: 2000 episodes |

## Grid format

ASCII text: `.` = playable sector, anything else = wall. Files in `grids/symmetric/` and `grids/asymmetric/`.

## What's NOT implemented yet

- **Epsilon-NE**: no epsilon parameter in NE checking
- **No-op/pass action**: agents must claim every round (limits scenarios where A*H > S)

## Dependencies

Runtime: `z3-solver`. Dev: `pytest>=8.0`. Pre-commit: black, isort, flake8, mypy.
