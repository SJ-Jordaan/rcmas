# RCMAS Architecture Documentation

## Overview

The RCMAS (Region Control Multi-Agent Systems) framework implements a formal mathematical model for competitive multi-agent region control scenarios. This document describes the architectural design and key patterns used.

## Core Principles

1. **Separation of Concerns**: Domain logic, simulation, strategies, and visualization are cleanly separated
2. **Extensibility**: Easy to add new strategies, analysis tools, and visualizations
3. **Type Safety**: Comprehensive type hints throughout
4. **Testability**: All components designed for isolated testing
5. **Immutability**: Value objects are immutable where appropriate

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│              Visualization Layer                     │
│  (Terminal, Matplotlib, Future: Web Interface)       │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Analysis Layer                          │
│  (Region Computation, Metrics, Statistics)           │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Simulation Layer                        │
│  (Simulator, Evolution, Protocol)                    │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Strategy Layer                          │
│  (Random, Greedy, Q-Learning, Future: DRL)          │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              Domain Layer                            │
│  (Agent, Territory, State, Action)                   │
└─────────────────────────────────────────────────────┘
```

## Domain Model

### Core Entities

#### Territory & Sector
- **Sector**: Immutable value object representing a position (i, j)
- **Territory**: Collection of accessible sectors, supports obstacles
- **TerritoryBuilder**: Builder pattern for complex territory construction

#### Agent
- Represents autonomous agent with objective
- Can have associated strategy
- Uses value equality based on id
- **DummyAgent**: Special singleton for unoccupied sectors

#### State
- Maps sectors to agents (state function s: T → Agt⁺)
- Provides queries for occupancy, regions, etc.
- **FailureState**: Special state for mission failure

#### Action & ActionProfile
- **Action**: Immutable (agent, sector) pair
- **ActionProfile**: Maps each agent to their chosen action
- **ActionProfileBuilder**: Fluent interface for construction

### Relationships

```
Territory ──contains──> Sector
State ──maps──> (Sector → Agent)
Agent ──uses──> Strategy
ActionProfile ──contains──> (Agent → Action)
Action ──references──> (Agent, Sector)
```

## Design Patterns

### 1. Strategy Pattern

**Purpose**: Encapsulate agent decision-making algorithms

**Implementation**:
```python
class AbstractStrategy(ABC):
    @abstractmethod
    def select_action(self, state, agent, available_actions) -> Action:
        pass
```

**Benefits**:
- Easy to add new strategies
- Can swap strategies at runtime
- Strategies can be tested independently

**Examples**:
- RandomStrategy
- GreedyExpansionStrategy
- QLearningStrategy
- Future: DeepQLearningStrategy, MinimaxStrategy, etc.

### 2. Builder Pattern

**Purpose**: Construct complex objects incrementally

**Implementation**:
```python
territory = (TerritoryBuilder(10, 10)
            .add_obstacle(5, 5)
            .add_obstacle_region(1, 1, 2, 2)
            .build())
```

**Benefits**:
- Fluent, readable API
- Validation before construction
- Separates construction from representation

**Used in**:
- TerritoryBuilder
- ActionProfileBuilder

### 3. Factory Pattern

**Purpose**: Centralized object creation

**Implementation**:
```python
agent = AgentFactory.create_agent(objective=5, strategy=strategy)
agents = AgentFactory.create_agents(count=3, objective=5)
```

**Benefits**:
- Consistent ID generation
- Encapsulates creation logic
- Easy to extend

### 4. Template Method Pattern

**Purpose**: Define algorithm structure with customizable steps

**Implementation**:
```python
class AbstractStrategy:
    def select_action(self, ...):  # Must implement
        pass
    
    def on_simulation_start(self, ...):  # Optional hook
        pass
    
    def on_round_complete(self, ...):  # Optional hook
        pass
```

**Benefits**:
- Provides hooks for learning strategies
- Maintains consistent interface
- Separates invariant from variant behavior

### 5. Singleton Pattern

**Purpose**: Ensure single instance exists

**Implementation**:
```python
DUMMY_AGENT = DummyAgent()  # Global singleton
```

**Benefits**:
- Memory efficient
- Semantic clarity (dummy agent concept)
- Easy equality checks

## Simulation Flow

```
┌──────────────┐
│ Initialize   │
│ State (s₀)   │
└──────┬───────┘
       │
       ↓
┌──────────────────────────┐
│ For each round:          │
│                          │
│ 1. Get available actions │
│    (Protocol)            │
│                          │
│ 2. Agents select actions │
│    (Strategies)          │
│                          │
│ 3. Form ActionProfile    │
│                          │
│ 4. Apply Evolution       │
│    (State transition)    │
│                          │
│ 5. Check termination     │
└──────┬───────────────────┘
       │
       ↓
   ┌───┴────┐
   │Terminal?│
   └───┬────┘
       │No
       └─────> Continue loop
       │Yes
       ↓
┌──────────────┐
│ Compute      │
│ Results      │
└──────────────┘
```

## Extensibility Points

### Adding New Strategies

1. Inherit from `AbstractStrategy`
2. Implement `select_action()`
3. Optionally override hooks for learning
4. Register in `src/strategies/__init__.py`

### Adding New Analysis Tools

1. Create new module in `src/analysis/`
2. Use `CohesiveRegionAnalyzer` for region queries
3. Export functions/classes
4. Add to `src/analysis/__init__.py`

### Adding New Visualizations

1. Create new visualizer class
2. Implement state and trajectory rendering
3. Support customization options
4. Add to `src/visualization/__init__.py`

### Future Extensions

#### 1. Gymnasium Environment
Wrap RCMAS as a Gymnasium environment for RL training:
```python
class RCMASEnv(gym.Env):
    def step(self, action): ...
    def reset(self): ...
    def render(self, mode='human'): ...
```

#### 2. Multi-Objective Optimization
- Extend objectives beyond region size
- Add sector rewards/values
- Implement Pareto-optimal strategies

#### 3. Partial Observability
- Limit agent knowledge of state
- Implement belief states
- Add communication mechanisms

#### 4. Dynamic Territories
- Sectors appear/disappear over time
- Moving obstacles
- Time-varying rewards

#### 5. Advanced Strategies
- Monte Carlo Tree Search (MCTS)
- Deep Reinforcement Learning (PPO, A3C)
- Game-theoretic equilibria
- Distributed decision-making

## Performance Considerations

### Time Complexity

- **State Creation**: O(w × r) for territory of width w, height r
- **Available Actions**: O(w × r) in worst case (all unoccupied)
- **Cohesive Region Computation**: O(w × r) using BFS
- **Evolution**: O(n) for n agents

### Space Complexity

- **State Storage**: O(w × r)
- **Q-Table**: O(|S| × |A|) - can be large!
- **Trajectory**: O(t × w × r) for t rounds

### Optimization Strategies

1. **Caching**: CohesiveRegionAnalyzer caches results
2. **Early Termination**: Detect terminal states quickly
3. **Sparse Representations**: Use sets/dicts for large territories
4. **Parallel Execution**: Run multiple simulations in parallel

## Testing Strategy

### Unit Tests
- Test each class in isolation
- Mock dependencies
- Cover edge cases

### Integration Tests
- Test component interactions
- Verify simulation correctness
- Test with realistic scenarios

### Property-Based Tests (Future)
- Use hypothesis for property testing
- Verify invariants hold
- Fuzz test with random inputs

## Code Quality

### Static Analysis
- **Type Checking**: mypy for type safety
- **Linting**: flake8 for style
- **Formatting**: black for consistency

### Coverage
- Target: >80% code coverage
- Critical paths must be tested
- Integration tests for workflows

## Documentation

### Code Documentation
- Docstrings for all public APIs
- Type hints throughout
- Examples in docstrings

### User Documentation
- README: Quick start
- DEVELOPMENT.md: Developer guide
- Examples: Practical demonstrations
- This doc: Architecture reference

## Dependencies

### Core
- numpy: Numerical operations
- networkx: Graph algorithms (future use)

### Visualization
- matplotlib: Publication-quality figures
- seaborn: Statistical visualizations

### RL Integration
- gymnasium: RL environment interface
- stable-baselines3: DRL algorithms

### Development
- pytest: Testing framework
- black: Code formatting
- mypy: Type checking
- flake8: Linting

## Future Roadmap

### Phase 1: Core Functionality (Current)
- ✅ Domain model implementation
- ✅ Basic strategies
- ✅ Simulation engine
- ✅ Visualization tools
- ✅ Testing infrastructure

### Phase 2: Advanced Strategies
- Deep reinforcement learning integration
- Game-theoretic analysis
- Formal verification hooks
- Strategy benchmarking suite

### Phase 3: Extensions
- Partial observability support
- Communication between agents
- Dynamic environments
- Multi-objective optimization

### Phase 4: Tools & Analysis
- Web-based visualization
- Interactive scenario builder
- Performance profiling tools
- Statistical analysis suite

### Phase 5: Research Integration
- Integration with formal verification tools
- Automated strategy synthesis
- Theoretical guarantees
- Publication-ready results generation

## References

- [Original Paper](paper.tex): Formal RCMAS definition
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Builder Pattern](https://refactoring.guru/design-patterns/builder)
- [Gymnasium](https://gymnasium.farama.org/): RL environment interface
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): DRL library
