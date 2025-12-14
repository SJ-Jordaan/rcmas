# qlearning

Territory capture game simulation + Q-learning trainer.

## Quick start

```bash
python -m qlearning.cli smoke
```

## Territory files

Provide an ASCII file where `.` means an acquirable sector; any other character is ignored.

Example (`grids/plus.txt` in the monorepo root):

```
..#..
..#..
#####
..#..
..#..
```

## Simulate

```bash
python -m qlearning.cli simulate --grid ../../grids/example.txt --agents random greedy
```

## Train (tabular Q-learning)

```bash
python -m qlearning.cli train --grid ../../grids/example.txt --num-agents 2 --episodes 2000 --out-dir q_tables
```

## Dev

```bash
python -m pip install -e ".[dev]"
pytest
```
