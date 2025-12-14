# rcmas (monorepo)

This repository contains the code for the RCMAS paper experiments.

## Layout

- `grids/`: shared ASCII territories (`.` = sector)
- `packages/rcmas-core`: shared core engine (deterministic, I/O-free)
- `packages/rcmas-testbeds`: experiment runners (SMT-CO + Q-learning; more coming)
- `packages/qlearning`: Q-learning implementation reused by the testbeds
- `legacy/rcmas-legacy/`: previous rcmas codebase (archived)

## Setup

```bash
cd /home/zandolor/Projects/rcmas

# using your existing workspace venv
/home/zandolor/Projects/.venv/bin/python -m pip install -e packages/rcmas-core -e packages/qlearning -e packages/rcmas-testbeds -e .
```

## Run testbeds

### Collective optimality (SMT)

```bash
rcmas --testbed smt-co --grid grids/example.txt --agents 2 --max-rounds 8

# dump key model variables
rcmas --testbed smt-co --grid grids/example.txt --agents 2 --max-rounds 8 --dump-model

# show per-timestep ASCII trace
rcmas --testbed smt-co --grid grids/example.txt --agents 2 --max-rounds 8 --render
```

### Pure Q-learning (train then play)

```bash
rcmas --testbed qlearning --grid grids/example.txt --agents 2 --max-rounds 8
```

### Nash equilibrium (SMT) / Hybrid (SMT+Q-learning)

These are scaffolded but not implemented yet:

```bash
rcmas --testbed smt-ne --grid grids/example.txt --agents 2 --max-rounds 8
rcmas --testbed hybrid --grid grids/example.txt --agents 2 --max-rounds 8
```
