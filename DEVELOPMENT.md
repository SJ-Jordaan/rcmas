# RCMAS Development Guide

## Getting Started

### Installation

```bash
# Clone repository
git clone <repository-url>
cd rcmas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Project Structure

```
rcmas/
├── src/                    # Source code
│   ├── domain/            # Core domain models
│   ├── strategies/        # Agent strategies
│   ├── simulation/        # Simulation engine
│   ├── analysis/          # Analysis utilities
│   └── visualization/     # Visualization tools
├── tests/                 # Unit and integration tests
├── examples/              # Example simulations
├── docs/                  # Documentation
└── paper.tex              # Research paper
```

## Development Workflow

### Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for all public APIs
- Docstrings for all public classes and methods

Format code with Black:
```bash
black src/ tests/ examples/
```

Check with flake8:
```bash
flake8 src/ tests/
```

Type check with mypy:
```bash
mypy src/
```

### Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_domain.py -v
```

### Adding New Features

#### Adding a New Strategy

1. Create new file in `src/strategies/`
2. Inherit from `AbstractStrategy`
3. Implement `select_action()` method
4. Add optional hooks: `on_simulation_start()`, `on_round_complete()`, `on_simulation_end()`
5. Add tests in `tests/test_strategies.py`
6. Update `src/strategies/__init__.py`

Example:
```python
from src.strategies.abstract_strategy import AbstractStrategy

class MyStrategy(AbstractStrategy):
    def select_action(self, state, agent, available_actions):
        # Your logic here
        return chosen_action
```

#### Adding Analysis Tools

1. Create new file in `src/analysis/`
2. Implement analysis functions/classes
3. Use `CohesiveRegionAnalyzer` for region computations
4. Add tests
5. Update `src/analysis/__init__.py`

#### Adding Visualization Tools

1. Create new visualizer in `src/visualization/`
2. Follow existing patterns (TerminalVisualizer, MatplotlibVisualizer)
3. Support state and trajectory visualization
4. Add tests
5. Update `src/visualization/__init__.py`

## Design Patterns Used

### Strategy Pattern
Agent behaviors are encapsulated as strategies, allowing runtime selection and easy experimentation.

### Builder Pattern
Complex objects (Territory, ActionProfile) can be constructed incrementally with fluent interface.

### Factory Pattern
AgentFactory provides consistent agent creation with auto-generated IDs.

### Observer Pattern (Future)
Simulation events can be observed for real-time visualization and logging.

## Best Practices

### Immutability
- Domain value objects (Sector, Action) are immutable (frozen dataclasses)
- States can be mutated but should be copied before modification
- Use `.copy()` methods when needed

### Type Hints
Always use type hints:
```python
def get_sectors_for_agent(self, agent: Agent) -> Set[Sector]:
    ...
```

### Documentation
All public APIs should have docstrings:
```python
def select_action(self, state: State, agent: Agent, available_actions: Set[Action]) -> Action:
    """
    Select an action for the agent.
    
    Args:
        state: Current state
        agent: Agent making decision
        available_actions: Available actions
        
    Returns:
        Chosen action
        
    Raises:
        ValueError: If no actions available
    """
```

### Error Handling
- Validate inputs early
- Raise descriptive exceptions
- Use custom exception types when appropriate

## Testing Guidelines

### Unit Tests
- Test each component in isolation
- Use fixtures for common setup
- Mock external dependencies
- Aim for >80% code coverage

### Integration Tests
- Test component interactions
- Verify end-to-end workflows
- Use realistic scenarios

### Test Organization
```python
class TestClassName:
    """Tests for ClassName."""
    
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...
```

## Performance Considerations

### State Hashing
States are hashable for use in Q-learning and caching. Be aware that hash computation can be expensive for large territories.

### Region Computation
CohesiveRegionAnalyzer caches results. Create new analyzer instance when state changes.

### Large Territories
For territories > 100x100, consider:
- Limiting max_rounds appropriately
- Using more efficient state representations
- Implementing parallel simulation for multiple runs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run full test suite
6. Submit pull request

## Troubleshooting

### Import Errors
Make sure you installed in development mode:
```bash
pip install -e .
```

### Test Failures
Run tests with verbose output:
```bash
pytest -vv
```

### Visualization Issues
For matplotlib issues, ensure backend is set:
```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive
```

## Resources

- [RCMAS Paper](paper.tex)
- [API Documentation](docs/)
- [Examples](examples/)
- [Issue Tracker](https://github.com/yourusername/rcmas/issues)
