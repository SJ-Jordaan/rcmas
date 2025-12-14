# rcmas (monorepo)

This repository contains the code for the RCMAS paper experiments.

## Packages

- `packages/rcmas-core`: shared core game engine (deterministic, I/O-free)
- `packages/rcmas-testbeds`: experiment runners (SMT-CO + Q-learning; more coming)
- `packages/qlearning`: standalone Q-learning project reused by testbeds

## Legacy

The previous `rcmas` codebase was moved to `legacy/rcmas-legacy/`.
