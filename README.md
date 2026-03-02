# RCMAS — Region Control Multi-Agent System

A tool for synthesising Nash equilibrium strategies in territory-control games via SMT solving and Q-learning.

## Quick start

```bash
git clone <repo-url> && cd rcmas
./setup.sh
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8
```

## Manual setup

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Usage

The `rcmas` CLI has four subcommands. All require `--grid`, `--agents`, and `--horizon`.

### Collective optimality (`co`)

Find a collectively optimal strategy (maximise total payoff) via SMT (Def 24).

```bash
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8

# Show per-timestep ASCII trace
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8 --render

# Print model variables
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8 --dump-model

# Set a Z3 timeout (milliseconds)
rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8 --timeout-ms 10000
```

### Nash equilibrium via IBIS (`ibis`)

Iterative best-response search for a Nash equilibrium (Algorithm 1).

```bash
rcmas ibis --grid grids/asymmetric/interesting.txt --agents 2 --horizon 8 \
      --max-iters 25 --progress --timing --timeout-ms 10000

# Enable symmetry-breaking constraints (reduces search space on symmetric grids)
rcmas ibis --grid grids/symmetric/4x4.txt --agents 2 --horizon 8 --symmetry
```

### Q-learning-guided IBIS (`qibis`)

Best-response iterations guided by Q-learning action proposals (Algorithm 2).

```bash
rcmas qibis --grid grids/asymmetric/interesting.txt --agents 2 --horizon 8 \
      --max-iters 25 --progress --timing --timeout-ms 10000

# With symmetry breaking
rcmas qibis --grid grids/symmetric/4x4.txt --agents 2 --horizon 8 --symmetry
```

### Q-learning self-play (`train`)

Train agents in self-play (default 2000 episodes), then evaluate (Sec 5.2).

```bash
rcmas train --grid grids/symmetric/6x6.txt --agents 2 --horizon 8
```

## Grid format

Grid files are plain text. Each character maps to a coordinate:

- `.` is a sector (playable cell)
- Any other character (including `#` and space) is a wall

Rows do not need to be the same length. Example (an asymmetric territory):

```text
..#.
#...
.#.#
```

Pre-made grids are in `grids/symmetric/` and `grids/asymmetric/`.

## Paper-to-code mapping

Every module in `rcmas/` maps to a section of the paper. The table below gives the correspondence.

| Paper Reference | Module | Key Functions / Classes |
|---|---|---|
| Def 1--3: Territory, Sector | `model.py` | `Territory`, `Coord` |
| Def 4--5: State, Evolve | `model.py` | `State`, `evolve()` |
| Def 6: Adjacency | `model.py` | `neighbors4()`, `adjacent_owned()` |
| Def 7--8: Components, Scores | `model.py` | `connected_components()`, `largest_region_size()`, `scores()` |
| Def 9--10: Strategy types | `strategy.py` | `Policy`, `ActionCandidates`, `StateKey` |
| Def 11--12: Freeze / hash | `strategy.py` | `freeze_policy()`, `freeze_profile()` |
| Def 13: Policy from solution | `strategy.py` | `policy_from_solution()` |
| Def 14: Reward | `strategy.py` | `reward()` |
| Def 15: SMT variables | `smt_variables.py` | `SmtVariables`, `create_variables()` |
| Def 16: Init constraint | `smt_constraints.py` | `init_constraint()` |
| Def 17: Evolution constraint | `smt_constraints.py` | `evolution_constraint()` |
| Def 18: Protocol constraint | `smt_constraints.py` | `protocol_constraint()` |
| Def 19: Collision constraint | `smt_constraints.py` | `collision_constraint()` |
| Def 20: Adjacency constraint | `smt_constraints.py` | `adjacency_constraint()` |
| Def 21: Cohesive region | `smt_constraints.py` | `cohesive_region_constraint()` |
| Def 22: Size constraint | `smt_constraints.py` | `size_constraint()` |
| Def 23: Reward constraint | `smt_constraints.py` | `reward_constraint()` |
| Def 24: Qualitative objective | `smt_objectives.py` | `qualitative_objective()` |
| Def 25: Quantitative objective | `smt_objectives.py` | `quantitative_objective()` |
| Alg 1: IBIS | `ibis.py` | `solve_ibis()` |
| Sec 5.2: Q-learning | `qlearning.py` | `QTable`, `train_self_play()` |
| Alg 2: Q-IBIS | `qibis.py` | `solve_qibis()` |
| Symmetry reduction | `symmetry.py` | `territory_automorphisms()`, `symmetry_info()`, `SymmetryInfo` |

## Project structure

```
rcmas/
    model.py              # Def 1-8:   Territory, State, evolve, adjacency, scores
    strategy.py           # Def 9-14:  Policy types, reward, freeze/hash
    smt_variables.py      # Def 15:    Z3 variable creation
    smt_constraints.py    # Def 16-23: One function per SMT constraint
    smt_objectives.py     # Def 24-25: Qualitative and quantitative objectives
    smt_solve.py          # Solver assembly: compose constraints, call Z3, extract solution
    ibis.py               # Alg 1:     IBIS (iterative best-response via SMT)
    qlearning.py          # Sec 5.2:   Tabular Q-learning, self-play training
    qibis.py              # Alg 2:     Q-IBIS (Q-learning-guided IBIS)
    symmetry.py           # Symmetry detection and reduction (automorphisms, orbits)
    cli.py                # Command-line interface
tests/                    # 186 tests covering every module
grids/                    # ASCII territory files
pyproject.toml            # Single project configuration
setup.sh                  # One-command setup script
```

## Running tests

```bash
source .venv/bin/activate
pytest
```

## Associated publications

- S. Jordaan and N. Timm, "Nash Equilibrium Strategy Synthesis for Region-Control Multi-Agent Systems via Q-Learning-Guided SMT Solving," in *NASA Formal Methods (NFM)*, 2026.
