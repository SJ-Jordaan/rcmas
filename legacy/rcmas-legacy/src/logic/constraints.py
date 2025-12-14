"""Constraint generation for the RCMAS system."""
from itertools import product, combinations
from z3 import Implies, And, Or, If, Sum, Bool
from logic.base import BaseConstraint
from core.state import PipelineContext
from utils.coords import sector_to_coords
# Q learning algorithm per agent
#  combative learning
# state     | action  | q-value
# (0,0,0,0) | 0       | 0         |
# (0,0,0,0) | 1       | 0.343451  |
# (0,0,0,0) | 2       | 0.67575   |
# (0,0,0,0) | 3       | -3.44545  |
# ...
# What is the definition of a Q learning state in this instance?

# Formal Strategy
# state     | a1 | a2 |
# (0,0,0,0) | 2  | 1  |
# (1,0,2,0) | 0  | 2  |
#   ...
# (1,0,0,2) | 2  | 1  |

# If every region is unoccupied -> timestep 0

# The number of occupied regions / number of agents = timestep

class InitialStateConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    state = ctx.z3_vars['state']
    inaccessible = ctx.config.grid.inaccessible_sectors

    inaccessibility_constraints = [
      state[s][t] == -1
      for s in inaccessible
      for t in range(ctx.num_timesteps + 1)
    ]

    empty_constraints = [
      state[s][0] == 0
      for s in range(ctx.num_sectors)
      if s not in inaccessible
    ]

    return inaccessibility_constraints + empty_constraints

# --- DYNAMICS CONSTRAINTS ---

class ActionAvailabilityConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    state = ctx.z3_vars['state']
    action = ctx.z3_vars['action']

    T = range(ctx.num_timesteps)
    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)

    # 1. Bounds
    lower = [action[a][t] >= 0 for t, a in product(T, A)]
    upper = [action[a][t] < ctx.num_sectors for t, a in product(T, A)]

    # 2. Availability (Cannot move to occupied sector)
    occupied = [
      Implies(state[s][t] != 0, action[a][t] != s)
      for t, s, a in product(T, S, A)
    ]

    return lower + upper + occupied

class EvolutionConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    state = ctx.z3_vars['state']
    action = ctx.z3_vars['action']

    T = range(ctx.num_timesteps)
    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)

    constraints = []

    for t, s in product(T, S):
      # Case 1: Successful move
      for a in A:
        agent_id = a + 1
        unique_action = And(
          action[a][t] == s,
          And([action[other_a][t] != s for other_a in A if other_a != a])
        )
        constraints.append(Implies(unique_action, state[s][t + 1] == agent_id))

      # Case 2: No action
      no_action = And([action[a][t] != s for a in A])
      constraints.append(Implies(no_action, state[s][t + 1] == state[s][t]))

      # Case 3: Conflict (2+ agents try to move to same sector)
      conflict = Or([
        And(action[a1][t] == s, action[a2][t] == s)
        for a1, a2 in combinations(A, 2)
      ])
      constraints.append(Implies(conflict, state[s][t + 1] == state[s][t]))

    return constraints

# --- TOPOLOGY CONSTRAINTS ---

class AdjacencyConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    # Retrieve specialized variables created for topology
    # (We assume these are created in variables.py and stored in ctx.z3_vars)
    state = ctx.z3_vars['state']
    adj_vars = ctx.z3_vars['adj']  # Helper boolean variables

    constraints = []
    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)
    width = ctx.config.grid.width
    final_t = ctx.num_timesteps

    # Logic to link physical adjacency + ownership to the 'adj' variable
    for i, j in combinations(S, 2):
      for a in A:
        # Helper to access the correct adj variable key
        low, high = (i, j) if i < j else (j, i)
        adj_var = adj_vars[(low, high, a)]

        xi, yi = sector_to_coords(i, width)
        xj, yj = sector_to_coords(j, width)

        is_physically_adj = Or(
          And(xi == xj, abs(yi - yj) == 1),
          And(yi == yj, abs(xi - xj) == 1)
        )

        is_owned = And(
          state[i][final_t] == (a + 1),
          state[j][final_t] == (a + 1)
        )

        constraints.append(adj_var == And(is_physically_adj, is_owned))

    return constraints

class ExhaustiveTransitivityConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    constraints = []

    # Unpack necessary variables
    adj_vars = ctx.z3_vars['adj']  # Defined in variables phase
    cr_vars = ctx.z3_vars['cr']  # Defined in variables phase

    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)
    width = ctx.config.grid.width

    # 1. Pre-calculate physical neighbors to optimize the loop
    neighbors = {s: [] for s in S}
    for i in S:
      xi, yi = sector_to_coords(i, width)
      for j in S:
        if i == j: continue
        xj, yj = sector_to_coords(j, width)
        if abs(xi - xj) + abs(yi - yj) == 1:
          neighbors[i].append(j)

    # Helper to safely access the symmetric adj_vars dictionary
    def safe_adj(u, v, agent):
      low, high = (u, v) if u < v else (v, u)
      return adj_vars.get((low, high, agent))

    # Dictionary to store intermediate reachability variables
    # Format: reach[(step, agent, i, j)] = Z3_Bool
    reach = {}

    # 2. Base Case (Step 0): Direct connections
    for a in A:
      for i in S:
        for j in S:
          r_var = Bool(f"reach_step_0_{a}_{i}_{j}")
          reach[(0, a, i, j)] = r_var

          if i == j:
            constraints.append(r_var == True)
          elif j in neighbors[i]:
            # Linked if physically adjacent AND valid in adj_vars
            constraints.append(r_var == safe_adj(i, j, a))
          else:
            constraints.append(r_var == False)

    # 3. Iterative Expansion (Step 1 to NUM_SECTORS)
    # Transitive closure: path of length k exists if path of length k-1 exists
    # AND we extend by one neighbor.
    max_diameter = ctx.num_sectors

    for k in range(1, max_diameter):
      for a in A:
        for i in S:
          for j in S:
            current_r_var = Bool(f"reach_step_{k}_{a}_{i}_{j}")
            reach[(k, a, i, j)] = current_r_var

            # Option 1: Path already existed at step k-1
            path_options = [reach[(k - 1, a, i, j)]]

            # Option 2: Extend from a neighbor 'm'
            for m in neighbors[i]:
              # Exists path i->m (adj) AND path m->j (at step k-1)
              extension = And(safe_adj(i, m, a), reach[(k - 1, a, m, j)])
              path_options.append(extension)

            constraints.append(current_r_var == Or(path_options))

    # 4. Bind Final Reachability to 'cr' (Cohesive Relation) variables
    # The solver aims to make 'cr' True/False, and this constraint forces 'cr'
    # to match the calculated reachability.
    final_step = max_diameter - 1

    # We iterate over the cr_vars keys which are (i, j, a)
    for (i, j, a), cr_bool in cr_vars.items():
      constraints.append(cr_bool == reach[(final_step, a, i, j)])

    return constraints

class TransitivityConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    # cr(i,j,a) ↔ ad(i,j,a) ∨ ∃k∈(i,j): (ad(i,k,a) ∧ cr(k,j,a)) ∨ (cr(i,k,a) ∧ ad(k,j,a))
    constraints = []

    adj_vars = ctx.z3_vars['adj']
    cr_vars = ctx.z3_vars['cr']

    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)

    def get_adj(i: int, j: int, a: int):
      low, high = (i, j) if i < j else (j, i)
      return adj_vars[(low, high, a)]

    def get_cr(i: int, j: int, a: int):
      low, high = (i, j) if i < j else (j, i)
      return cr_vars[(low, high, a)]

    for a in A:
      for i in S:
        for j in S:
          if i >= j:
            continue

          chain_terms = []
          for k in range(i + 1, j):
            chain_terms.append(And(get_adj(i, k, a), get_cr(k, j, a)))
            chain_terms.append(And(get_cr(i, k, a), get_adj(k, j, a)))

          body = [get_adj(i, j, a)] + chain_terms
          constraints.append(get_cr(i, j, a) == Or(body))

    return constraints

# --- OBJECTIVE CONSTRAINTS ---

class PayoffConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    # Link 'size' vars to 'cr' (connected relation) vars
    # And define 'payoff' as the max size
    constraints = []
    size_vars = ctx.z3_vars['size']
    cr_vars = ctx.z3_vars['cr']
    payoff_vars = ctx.z3_vars['payoff']
    state = ctx.z3_vars['state']

    S = range(ctx.num_sectors)
    A = range(ctx.config.agents.count)
    final_t = ctx.num_timesteps

    # 1. Define Size
    for i, a in product(S, A):
      # Size = 1 (self) + sum of all connected sectors j (where i < j)
      connections = Sum([
        If(cr_vars.get((i, j, a), False), 1, 0) for j in S if i < j
      ])

      # Note: This logic might need adjustment based on your exact transitive closure implementation
      # For simplicity, we assume cr(i,j) exists.

      constraints.append(
        If(state[i][final_t] == (a + 1),
           size_vars[i][a] == 1 + connections,
           size_vars[i][a] == 0
           )
      )

    # 2. Define Payoff (Max Size)
    for a in A:
      p_var = payoff_vars[a]
      agent_sizes = [size_vars[s][a] for s in S]

      # Payoff is an upper bound
      for s_var in agent_sizes:
        constraints.append(p_var >= s_var)

      # Tightness (must equal at least one)
      constraints.append(Or([p_var == s_var for s_var in agent_sizes]))

    return constraints

class FullBoardConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    state = ctx.z3_vars['state']

    # Logic adapted from your snippet
    accessible_sectors = [
      s for s in range(ctx.num_sectors)
      if s not in ctx.config.grid.inaccessible_sectors
    ]

    # "At the final timestep, occupied state must not be 0"
    return [
      state[s][ctx.num_timesteps] != 0
      for s in accessible_sectors
    ]

class BlockSolutionConstraint(BaseConstraint):
  def build(self, ctx: PipelineContext) -> list:
    """
    Generates a constraint to block the LAST found model in the context.
    """
    if not ctx.found_models:
      return []

    # Get the most recent model to block
    model = ctx.found_models[-1]

    state = ctx.z3_vars['state']
    action = ctx.z3_vars['action']

    # Flatten state variables
    all_state_vars = [
      state[s][t]
      for s in range(ctx.num_sectors)
      for t in range(ctx.num_timesteps + 1)
    ]

    # Flatten action variables
    all_action_vars = [
      action[a][t]
      for a in range(ctx.config.agents.count)
      for t in range(ctx.num_timesteps)
    ]

    all_vars = all_state_vars + all_action_vars

    # Create the blocking clause: "At least one variable must be different"
    block = [
      var != model.eval(var, model_completion=True)
      for var in all_vars
    ]

    # Return as a list containing one Or() constraint
    return [Or(block)]
