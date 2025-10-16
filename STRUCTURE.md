# RCMAS Project Structure

```
rcmas/
│
├── 📄 Core Documentation
│   ├── README.md                    # Quick start and overview
│   ├── PROJECT_SUMMARY.md           # Comprehensive project summary
│   ├── ARCHITECTURE.md              # Design and architecture details
│   ├── DEVELOPMENT.md               # Developer guidelines
│   ├── CONTRIBUTING.md              # Contribution guide
│   └── LICENSE                      # MIT License
│
├── ⚙️ Configuration Files
│   ├── setup.py                     # Package setup and dependencies
│   ├── requirements.txt             # Python dependencies
│   ├── pyproject.toml              # Tool configuration (pytest, black, mypy)
│   ├── .gitignore                  # Git ignore patterns
│   └── quickstart.sh               # Quick setup script (executable)
│
├── 📦 src/ - Core Framework
│   ├── __init__.py                 # Main package exports
│   │
│   ├── domain/                     # Domain Models Layer
│   │   ├── __init__.py            # Domain exports
│   │   ├── territory.py           # Territory, Sector, TerritoryBuilder
│   │   ├── agent.py               # Agent, DummyAgent, AgentFactory
│   │   ├── state.py               # State, FailureState
│   │   └── action.py              # Action, ActionProfile, ActionProfileBuilder
│   │
│   ├── simulation/                 # Simulation Engine Layer
│   │   ├── __init__.py            # Simulation exports
│   │   ├── simulator.py           # RCMASSimulator, SimulationResult
│   │   ├── protocol.py            # ActionAvailabilityProtocol
│   │   └── evolution.py           # Evolution (state transitions)
│   │
│   ├── strategies/                 # Strategy Pattern Layer
│   │   ├── __init__.py            # Strategy exports
│   │   ├── abstract_strategy.py   # AbstractStrategy (base class)
│   │   ├── random_strategy.py     # RandomStrategy
│   │   ├── greedy_strategy.py     # GreedyExpansionStrategy
│   │   └── qlearning_strategy.py  # QLearningStrategy
│   │
│   ├── analysis/                   # Analysis Tools Layer
│   │   ├── __init__.py            # Analysis exports
│   │   └── regions.py             # CohesiveRegionAnalyzer, RegionMetrics
│   │
│   └── visualization/              # Visualization Layer
│       ├── __init__.py            # Visualization exports
│       ├── terminal_viz.py        # TerminalVisualizer
│       └── matplotlib_viz.py      # MatplotlibVisualizer
│
├── 🧪 tests/ - Test Suite
│   ├── __init__.py                # Test package
│   ├── test_domain.py             # Domain model tests
│   ├── test_simulation.py         # Simulation engine tests
│   └── test_analysis.py           # Analysis tools tests
│
├── 📚 examples/ - Example Scenarios
│   ├── __init__.py                # Examples package
│   ├── basic_simulation.py        # Basic 5x5 simulation
│   └── complex_territory.py       # Territory with obstacles
│
├── 📖 docs/ - Documentation (Future)
│   └── (Sphinx documentation to be added)
│
└── 📝 paper.tex - Research Paper
    └── Associated research paper
```

## File Statistics

### Source Code
- **Domain Models**: 5 files, ~800 LOC
- **Simulation Engine**: 4 files, ~500 LOC
- **Strategies**: 5 files, ~600 LOC
- **Analysis Tools**: 2 files, ~400 LOC
- **Visualization**: 3 files, ~500 LOC
- **Total Source**: ~2,800 LOC

### Tests
- **Test Files**: 3 files, ~700 LOC
- **Coverage Target**: >80%

### Documentation
- **User Docs**: 5 markdown files, ~2,000 lines
- **Code Docstrings**: Comprehensive coverage

### Examples
- **Example Scripts**: 2 files, ~200 LOC

## Key Components by Purpose

### 🎯 For Researchers
- `examples/` - Start here for usage examples
- `src/strategies/` - Add your custom strategies here
- `src/analysis/` - Metrics and analysis tools
- `ARCHITECTURE.md` - Understanding the design

### 🔧 For Developers
- `src/domain/` - Core mathematical model implementation
- `src/simulation/` - Simulation loop and evolution
- `tests/` - Comprehensive test suite
- `DEVELOPMENT.md` - Development guidelines

### 📊 For Visualizations
- `src/visualization/terminal_viz.py` - Quick terminal output
- `src/visualization/matplotlib_viz.py` - Publication figures

### ⚡ Quick Actions
- `./quickstart.sh` - Complete setup in one command
- `pytest` - Run all tests
- `python examples/basic_simulation.py` - See it in action

## Import Paths

```python
# Domain models
from src.domain import Agent, Territory, Sector, State, Action

# Simulation
from src.simulation import RCMASSimulator, SimulationResult

# Strategies
from src.strategies import RandomStrategy, GreedyExpansionStrategy

# Analysis
from src.analysis import CohesiveRegionAnalyzer, RegionMetrics

# Visualization
from src.visualization import TerminalVisualizer, MatplotlibVisualizer
```

## Design Pattern Locations

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Strategy** | `src/strategies/abstract_strategy.py` | Agent behaviors |
| **Builder** | `src/domain/territory.py`, `src/domain/action.py` | Object construction |
| **Factory** | `src/domain/agent.py` | Agent creation |
| **Template Method** | `src/strategies/abstract_strategy.py` | Learning hooks |
| **Singleton** | `src/domain/agent.py` | Dummy agent |

## Extensibility Points

1. **New Strategies** → Add to `src/strategies/`
2. **New Analysis Tools** → Add to `src/analysis/`
3. **New Visualizations** → Add to `src/visualization/`
4. **Custom Examples** → Add to `examples/`
5. **Integration Tests** → Add to `tests/`

---

**Total Project Size**: ~40 files, ~6,000+ lines including docs and tests
**Estimated Development Time**: 20-30 hours for complete framework
**Maintenance Level**: Research-grade, production-ready foundation
