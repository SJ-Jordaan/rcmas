# rcmas-testbeds

CLI scaffolding for four experiment modes:
- `smt-co` (collective optimality via SMT)
- `qlearning` (pure Q-learning)
- `smt-ne` (naive NE discovery via SMT) — stub
- `hybrid` (SMT + Q-learning) — stub

## Run

```bash
pip install -e '.[dev]'
rcmas-testbeds --help

# SMT-CO with model inspection
rcmas-testbeds --testbed smt-co --grid ../../grids/example.txt --agents 2 --max-rounds 8 --dump-model

# SMT-CO with per-timestep ASCII trace
rcmas-testbeds --testbed smt-co --grid ../../grids/example.txt --agents 2 --max-rounds 8 --render

# Q-learning baseline (trains then plays)
rcmas-testbeds --testbed qlearning --grid ../../grids/example.txt --agents 2 --max-rounds 8
```
