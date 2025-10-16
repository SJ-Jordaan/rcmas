# Region Control Multi-Agent Systems (RCMAS) Simulator

A robust, extensible framework for simulating and analyzing competitive region control scenarios with multiple agents. This project implements the formal RCMAS model for strategy synthesis and Q-Learning-assisted formal reasoning.

## Overview

This framework provides:
- **Formal mathematical model implementation** based on the RCMAS definition
- **Flexible agent strategy framework** supporting various learning algorithms
- **Comprehensive simulation engine** with collision detection and mission objectives
- **Extensible architecture** using design patterns (Strategy, Observer, Factory, Builder)
- **Visualization and analysis tools** for research insights

## Project Structure

```
rcmas/
├── src/
│   ├── domain/          # Core domain models (Agent, Territory, State, Action)
│   ├── strategies/      # Agent strategy implementations
│   ├── simulation/      # Simulation engine and evolution logic
│   ├── analysis/        # Cohesive region computation and metrics
│   ├── visualization/   # Rendering and plotting tools
│   └── utils/           # Common utilities and helpers
├── tests/               # Comprehensive test suite
├── examples/            # Example scenarios and use cases
├── docs/                # Extended documentation
└── paper.tex            # Associated research paper

```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from src.domain.territory import Territory
from src.domain.agent import Agent
from src.simulation.simulator import RCMASSimulator
from src.strategies.random_strategy import RandomStrategy

# Create a 5x5 territory
territory = Territory(width=5, height=5)

# Create agents with objectives
agents = [
    Agent(id="a1", objective=3, strategy=RandomStrategy()),
    Agent(id="a2", objective=3, strategy=RandomStrategy())
]

# Run simulation
simulator = RCMASSimulator(territory, agents)
result = simulator.run()

print(f"Winner: {result.winner}")
print(f"Final regions: {result.region_sizes}")
```

## Key Design Patterns

### Strategy Pattern
Agent behaviors are encapsulated as strategies, allowing easy experimentation with different approaches (random, Q-learning, minimax, etc.)

### Observer Pattern
Simulation events are broadcast to observers for visualization, logging, and analysis

### Builder Pattern
Complex simulation scenarios can be constructed incrementally with a fluent interface

### Factory Pattern
Strategies and agents can be created through factories for consistency and extensibility

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints throughout
- Document all public APIs with docstrings
- Maintain >80% test coverage

### Adding New Strategies
1. Inherit from `AbstractStrategy`
2. Implement `select_action(state, agent, available_actions)` method
3. Add tests in `tests/strategies/`
4. Document in `docs/strategies.md`

### Running Tests
```bash
pytest tests/ -v --cov=src
```

## Research Applications

This framework supports:
- **Strategy synthesis** for competitive multi-agent scenarios
- **Q-Learning integration** with formal verification
- **Game-theoretic analysis** of equilibrium strategies
- **Performance benchmarking** across different approaches

## Citation

```bibtex
@article{jordaan2025rcmas,
  title={Strategy Synthesis for Competitive Region Control via Q-Learning-Assisted Formal Reasoning},
  author={Jordaan, Steven and Timm, Nils},
  journal={...},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

- Steven Jordaan: u18074848@tuks.co.za
- Nils Timm: ntimm@cs.up.ac.za
