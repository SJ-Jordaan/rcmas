# Copilot instructions

## Big picture
- Core rules live in `src/qlearning/engine/` and are I/O-free.
- The simulation is concurrent per round: each agent proposes a sector; collisions end the game in **defeat**.
- Score is per-agent `GameState.largest_region_size()` (4-neighborhood connectivity).

## Key modules
- Territory shape + stable indexing: `qlearning.engine.territory.Territory` (`from_ascii`, `index_of`).
- Immutable state + rules: `qlearning.engine.state.GameState` (`step`, `scores`).
- Runner that calls agents: `qlearning.engine.game.GameEngine`.
- Agents (policies) implement `select_action(state, agent_id)`; see `src/qlearning/agents/`.
- Tabular RL:
  - State/action encoding: `qlearning.rl.encoding.TabularEncoding` (base64 state key; action key = sector index).
  - Storage: `qlearning.rl.qtable.QTable` (JSON).
  - Training: `qlearning.rl.trainer.SelfPlayTrainer` (independent Q-learning self-play).

## Conventions (important)
- Treat `GameState` as immutable: always use `state.step(...)` and keep the returned `next_state`.
- Only coordinates in `Territory.sectors` are acquirable; CLI uses `.` in ASCII grids.
- Keep engine free of printing/logging; all CLI/IO goes in `src/qlearning/cli.py`.
- Q-tables are saved per grid under `q_tables/<territory_id>/agent_<n>.json`.

## Typical workflows
- Install dev deps:
  - `python -m pip install -e ".[dev]"`
- Run tests:
  - `pytest`
- Run a game:
  - `python -m qlearning.cli simulate --grid grids/example.txt --agents random greedy`
- Train:
  - `python -m qlearning.cli train --grid grids/example.txt --num-agents 2 --episodes 2000 --out-dir q_tables`

## When changing rules
- Update `GameState.step()` first, then adjust any agent assumptions (`src/qlearning/agents/`) and trainer reward shaping (`src/qlearning/rl/trainer.py`).
- Keep encodings stable for a fixed territory; if you change state representation, bump output format or directory naming.
