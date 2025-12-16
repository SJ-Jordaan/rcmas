# rcmas (monorepo)

This repository contains the code for the RCMAS paper experiments.

## Layout

- `grids/`: shared ASCII territories (`.` = sector)
- `packages/rcmas-core`: shared core engine (deterministic, I/O-free)
- `packages/rcmas-testbeds`: experiment runners (SMT-CO, SMT-NE, Hybrid, Q-learning)
- `packages/qlearning`: Q-learning implementation reused by the testbeds
- `legacy/rcmas-legacy/`: previous rcmas codebase (archived)

## Setup (first time)

Prereqs:

- Python $\ge$ 3.10

Create a venv and install the editable workspace packages:

```bash
cd /Users/stevenjordaan/Desktop/Development/rcmas

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip

# install all workspace packages editable (+ pytest via root dev extra)
python -m pip install -e packages/rcmas-core -e packages/qlearning -e packages/rcmas-testbeds -e ".[dev]"
```

Sanity check:

```bash
rcmas --help
```

## Grids (`grids/`)

Grid files are plain text. The engine parses them as follows:

- Each line is a row ($y=0$ is the first line).
- Each character is a column ($x=0$ is the first character).
- A `.` character means “this coordinate is a sector”.
- Any other character (including spaces) is ignored and treated as empty.
- The territory can be any shape; rows do not need to be rectangular.

Example (a non-rectangular territory):

```text
.. ..
 ....
..  .
```

The repo includes a few starter grids:

- `grids/4x4.txt`
- `grids/6x6.txt`
- `grids/interesting.txt`

## Run testbeds

The top-level `rcmas` CLI delegates to `rcmas-testbeds`. Common required flags:

- `--testbed {smt-co,qlearning,smt-ne,hybrid}`
- `--grid PATH`
- `--agents N`

Other flags:

- `--max-rounds N` (default: 10)
- `--timeout-ms MS` (SMT: 0 means “unset”)
- `--render` (SMT modes: print an ASCII trace)
- `--dump-model` (SMT-CO: print debug/model-derived variables)
- `--max-iters N`, `--progress`, `--timing` (SMT-NE/Hybrid)

### Collective optimality (SMT-CO)

```bash
rcmas --testbed smt-co --grid grids/4x4.txt --agents 2 --max-rounds 8

# show per-timestep ASCII trace
rcmas --testbed smt-co --grid grids/4x4.txt --agents 2 --max-rounds 8 --render

# dump key solution/model variables
rcmas --testbed smt-co --grid grids/4x4.txt --agents 2 --max-rounds 8 --dump-model

# apply a Z3 timeout per solve
rcmas --testbed smt-co --grid grids/4x4.txt --agents 2 --max-rounds 8 --timeout-ms 10000
```

### Pure Q-learning (train then play)

This trains in self-play (default: 2000 episodes) and then evaluates via the deterministic engine.

```bash
rcmas --testbed qlearning --grid grids/6x6.txt --agents 2 --max-rounds 8
```

### Nash equilibrium via SMT (SMT-NE)

Iterative best-response search for a (naive) Nash equilibrium.

```bash
rcmas --testbed smt-ne --grid grids/interesting.txt --agents 2 --max-rounds 8 --max-iters 25 --progress --timing --timeout-ms 10000
```

### Hybrid (SMT + Q-learning proposals)

Best-response iterations like SMT-NE, but uses RL to propose a small action set for visited states.

```bash
rcmas --testbed hybrid --grid grids/interesting.txt --agents 2 --max-rounds 8 --max-iters 25 --progress --timing --timeout-ms 10000
```

## Tests

```bash
pytest -q
```
