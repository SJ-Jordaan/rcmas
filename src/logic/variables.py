"""Z3 variable creation for the RCMAS system."""
from itertools import combinations, product

from z3 import Int, Bool, Or, And, Implies, Sum, If, Not

from utils import sector_to_coords

def encode_state(num_sectors, num_timesteps):
  S = range(num_sectors)
  T = range(num_timesteps + 1)

  return [
    [Int(f"sector_{s}_{t}") for t in T]
    for s in S
  ]

def encode_action(num_timesteps, num_agents):
  T = range(num_timesteps)
  A = range(num_agents)

  return [
    [Int(f"action_{a}_{t}") for t in T]
    for a in A
  ]

def adj(s1, s2, a):
  return Bool(f"adj_{s1}_{s2}_{a}")

def cr(s1, s2, a):
  return Bool(f"cohesive_relation_{s1}_{s2}_{a}")

def size(s, a):
  return Int(f"size_{s}_{a}")

def encode_variables(num_sectors, num_agents, num_timesteps):
  return encode_state(num_sectors, num_timesteps), encode_action(num_timesteps, num_agents)

def encode_initial_state(state, inaccessible_sectors, num_timesteps, num_sectors):
  inaccessible_constraints = [
    state[s][t] == -1
    for s in inaccessible_sectors
    for t in range(num_timesteps + 1)
  ]

  initial_empty_constraints = [
    state[s][0] == 0
    for s in range(num_sectors) if s not in inaccessible_sectors
  ]

  return inaccessible_constraints + initial_empty_constraints

def encode_adjacency(state, num_sectors, num_agents, num_timesteps, grid_width):
  T = range(num_sectors)
  Agt = range(num_agents)

  constraints = []

  for i, j in combinations(T, 2):
    for a in Agt:
      xi, yi = sector_to_coords(i, grid_width)
      xj, yj = sector_to_coords(j, grid_width)

      are_physically_adjacent = Or(
        And(xi == xj, abs(yi - yj) == 1),
        And(yi == yj, abs(xi - xj) == 1)
      )

      both_owned_by_a = And(
        state[i][num_timesteps] == (a + 1),
        state[j][num_timesteps] == (a + 1)
      )

      constraints.append(
        adj(i, j, a) == And(are_physically_adjacent, both_owned_by_a)
      )

  return constraints


def encode_transitivity(num_sectors, num_agents, grid_width):
  S = range(num_sectors)
  Agt = range(num_agents)

  constraints = []

  # 1. Pre-calculate physical neighbors to optimize solver loop
  #    This prevents checking every sector against every other sector.
  neighbors = {s: [] for s in S}
  for i in S:
    xi, yi = sector_to_coords(i, grid_width)
    for j in S:
      if i == j: continue
      xj, yj = sector_to_coords(j, grid_width)
      # Check physical adjacency (Manhattan distance of 1)
      if abs(xi - xj) + abs(yi - yj) == 1:
        neighbors[i].append(j)

  # Helper to ensure we access the exact variable created in encode_adjacency
  # encode_adjacency uses combinations, so variables are likely stored as min_max
  def safe_adj(u, v, agent):
    low, high = (u, v) if u < v else (v, u)
    return adj(low, high, agent)

  # Dictionary to store reachability variables: reach[(step, agent, i, j)]
  reach = {}

  # 2. Define Base Case (Step 0)
  # At step 0, a sector reaches itself. It reaches neighbors if they are adj.
  # We treat this as the "Direct Connection" layer.
  for a in Agt:
    for i in S:
      for j in S:
        # Define internal variable for this step
        r_var = Bool(f"reach_step_0_{a}_{i}_{j}")
        reach[(0, a, i, j)] = r_var

        if i == j:
          constraints.append(r_var == True)
        elif j in neighbors[i]:
          # Direct connection exists if physically adjacent AND owned by agent
          constraints.append(r_var == safe_adj(i, j, a))
        else:
          constraints.append(r_var == False)

  # 3. Iterative Expansion (Step 1 to NUM_SECTORS)
  # We unroll the graph traversal. Max diameter is NUM_SECTORS.
  for k in range(1, num_sectors):
    for a in Agt:
      for i in S:
        for j in S:
          current_r_var = Bool(f"reach_step_{k}_{a}_{i}_{j}")
          reach[(k, a, i, j)] = current_r_var

          # Optimization: Only check paths through valid physical neighbors
          # Path exists if:
          # 1. Already existed in previous step (k-1)
          # 2. OR exists a neighbor 'm' such that i->m (adj) and m->j (reach at k-1)

          path_options = [reach[(k - 1, a, i, j)]]  # Option 1: Persistence

          for m in neighbors[i]:
            # Option 2: Extension
            # i is connected to m (adj), and m was already connected to j
            path_options.append(
              And(safe_adj(i, m, a), reach[(k - 1, a, m, j)])
            )

          constraints.append(current_r_var == Or(path_options))

  # 4. Bind the Final State to the Interface Variable `cr`
  # Your interface expects cr(i, j, a). We map the final reachability step to this.
  final_step = num_sectors - 1
  for (i, j), a in product(combinations(S, 2), Agt):
    constraints.append(
      cr(i, j, a) == reach[(final_step, a, i, j)]
    )

  return constraints

def encode_size(state, num_sectors, num_agents, num_timesteps):
  T = range(num_sectors)
  Agt = range(num_agents)

  constraints = []

  for i, a in product(T, Agt):
    constraints.append(
      If(
        state[i][num_timesteps] == (a + 1),
        size(i, a) == 1 + Sum(
          cr(i, j, a)
          for j in T if i < j
        ),
        size(i, a) == 0
      )
    )

  return constraints

def encode_payoff(num_sectors, num_agents):
  """
  Creates payoff variables and constraints.
  Payoff_a = Max(size_{0,a}, size_{1,a}, ... size_{N,a})
  """
  constraints = []
  payoff_vars = []

  for a in range(num_agents):
    # Create the variable
    p_var = Int(f"payoff_{a}")
    payoff_vars.append(p_var)

    # Get all size variables for this agent
    agent_size_vars = [size(s, a) for s in range(num_sectors)]

    # Constraint 1: Payoff must be an upper bound (>= all sizes)
    for s_var in agent_size_vars:
      constraints.append(p_var >= s_var)

    # Constraint 2: Tightness (Payoff must equal at least one size)
    # This forces p_var to be exactly the maximum, not just an arbitrary upper bound.
    constraints.append(Or([p_var == s_var for s_var in agent_size_vars]))

  return payoff_vars, constraints

