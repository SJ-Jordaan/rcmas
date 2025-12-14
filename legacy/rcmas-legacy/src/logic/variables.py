"""Z3 variable creation for the RCMAS system."""
from z3 import Int, Bool
from itertools import combinations

def create_base_variables(num_sectors, num_timesteps, num_agents):
  # State: sector s at time t
  state = [[Int(f"sector_{s}_{t}")
            for t in range(num_timesteps + 1)]
           for s in range(num_sectors)]

  # Action: agent a at time t
  action = [[Int(f"action_{a}_{t}")
             for t in range(num_timesteps)]
            for a in range(num_agents)]

  return state, action


def create_topology_variables(num_sectors, num_agents):
  # Adjacency helper (i, j, agent)
  adj = {}
  cr = {}  # Cohesive Relation (final transitive connectivity)

  for i, j in combinations(range(num_sectors), 2):
    for a in range(num_agents):
      adj[(i, j, a)] = Bool(f"adj_{i}_{j}_{a}")
      cr[(i, j, a)] = Bool(f"cohesive_relation_{i}_{j}_{a}")

  return adj, cr


def create_objective_variables(num_sectors, num_agents):
  # Size: sector s, agent a
  size = [[Int(f"size_{s}_{a}") for a in range(num_agents)]
          for s in range(num_sectors)]

  # Payoff: agent a
  payoff = [Int(f"payoff_{a}") for a in range(num_agents)]

  return size, payoff
