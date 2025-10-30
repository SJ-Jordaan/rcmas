from z3 import Int, Solver, Implies, And, Or, sat

grid_height = 8
grid_width = 8
num_agents = 4

num_sectors = grid_height * grid_width
num_timesteps = int(num_sectors / num_agents)

sectors = [Int(f"sector_{id}_{t}") for t in range(num_timesteps) for id in range(num_sectors)]
action = [Int(f"action_{agt}_{t}") for t in range(num_timesteps) for agt in range(1, num_agents + 1)]
initial_state = [Int(f"sector_{id}_0") == 0 for id in range(num_sectors)]

# sector_id_t != 0 -> act_agt_t != id
action_availability = [
    *[Implies(Int(f"sector_{id}_{t}") != 0, Int(f"action_{agt}_{t}") != id)
    for t in range(num_timesteps)
    for id in range(num_sectors)
    for agt in range(1, num_agents + 1)],
    *[0 <= Int(f"action_{agt}_{t}") for t in range(num_timesteps) for agt in range(1, num_agents + 1)],
    *[Int(f"action_{agt}_{t}") < num_sectors for t in range(num_timesteps) for agt in range(1, num_agents + 1)]
]

all_unique_pairs = [(agt1, agt2) for agt1 in range(1, num_agents + 1) for agt2 in range(agt1 + 1, num_agents + 1)]


evolution = [
    *[Implies(
        And(
            Int(f"action_{agt}_{t}") == id,
            *[Int(f"action_{other_agent}_{t}") != id for other_agent in range(1, num_agents + 1) if other_agent != agt]
        ),
        Int(f"sector_{id}_{t+1}") == agt
    )
    for t in range(num_timesteps)
    for id in range(num_sectors)
    for agt in range(1, num_agents + 1)],
    *[Implies(
        And(*[Int(f"action_{agt}_{t}") != id for agt in range(1, num_agents + 1)]),
        Int(f"sector_{id}_{t}") == Int(f"sector_{id}_{t+1}")
    )
    for t in range(num_timesteps)
    for id in range(num_sectors)],
    *[Implies(
        And(
            Int(f"action_{agt1}_{t}") == id,
            Int(f"action_{agt2}_{t}") == id,
        ),
        Int(f"sector_{id}_{t}") == Int(f"sector_{id}_{t+1}")
    )
        for t in range(num_timesteps)
        for id in range(num_sectors)
        for (agt1, agt2) in all_unique_pairs],
]


solver = Solver()
solver.add(initial_state)
solver.add(action_availability)
solver.add(evolution)
res = solver.check()

# Function to display the model as a grid
def display_grid(model, grid_height, grid_width, num_timesteps, num_agents):
    for t in range(num_timesteps + 1):  # Include t=0 (initial state)
        print(f"Timestep {t}:")
        for row in range(grid_height):
            row_values = []
            for col in range(grid_width):
                sector_id = row * grid_width + col
                sector_var = Int(f"sector_{sector_id}_{t}")
                # Evaluate the variable in the model
                try:
                    value = model.eval(sector_var, model_completion=True)
                    row_values.append(str(value))
                except Exception:
                    row_values.append("?")
            print(" ".join(row_values))
       
        # Show actions after this timestep (if not the last timestep)
        if t < num_timesteps:
            print(f"Actions at t={t}:")
            for agt in range(1, num_agents + 1):
                action_var = Int(f"action_{agt}_{t}")
                try:
                    action_value = model.eval(action_var, model_completion=True)
                    action_row = int(action_value.as_long()) // grid_width
                    action_col = int(action_value.as_long()) % grid_width
                    print(f"  Agent {agt} -> sector {action_value} (row {action_row}, col {action_col})")
                except Exception:
                    print(f"  Agent {agt} -> ?")
        print()

def get_grid_string(model, grid_height, grid_width, num_timesteps, num_agents):
    """Returns the grid display as a string instead of printing it."""
    output = []
    for t in range(num_timesteps + 1):  # Include t=0 (initial state)
        output.append(f"Timestep {t}:")
        for row in range(grid_height):
            row_values = []
            for col in range(grid_width):
                sector_id = row * grid_width + col
                sector_var = Int(f"sector_{sector_id}_{t}")
                try:
                    value = model.eval(sector_var, model_completion=True)
                    row_values.append(str(value))
                except Exception:
                    row_values.append("?")
            output.append(" ".join(row_values))
       
        if t < num_timesteps:
            output.append(f"Actions at t={t}:")
            for agt in range(1, num_agents + 1):
                action_var = Int(f"action_{agt}_{t}")
                try:
                    action_value = model.eval(action_var, model_completion=True)
                    action_row = int(action_value.as_long()) // grid_width
                    action_col = int(action_value.as_long()) % grid_width
                    output.append(f"  Agent {agt} -> sector {action_value} (row {action_row}, col {action_col})")
                except Exception:
                    output.append(f"  Agent {agt} -> ?")
        output.append("")
    return "\n".join(output)

# Check if the constraints are satisfiable and find all models
model_count = 0
with open("models_output.txt", "w") as f:
    if solver.check() == sat:
        model_count += 1
        model = solver.model()
       
        # Write to file
        f.write(f"{'='*60}\n")
        f.write(f"Model {model_count}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Raw model:\n{model}\n\n")
        f.write(get_grid_string(model, grid_height, grid_width, num_timesteps, num_agents))
        f.write("\n\n")
       
        # Print to console
        print(f"Model {model_count} found and written to file")
        display_grid(model, grid_height, grid_width, num_timesteps, num_agents)
       
        # Block this solution to find different ones
        # Create OR constraint: at least one variable must be different
        block = []
        # Add all sector variables
        for t in range(num_timesteps + 1):
            for id in range(num_sectors):
                sector_var = Int(f"sector_{id}_{t}")
                block.append(sector_var != model.eval(sector_var, model_completion=True))
        # Add all action variables
        for t in range(num_timesteps):
            for agt in range(1, num_agents + 1):
                action_var = Int(f"action_{agt}_{t}")
                block.append(action_var != model.eval(action_var, model_completion=True))
       
        solver.add(Or(block))
   
    if model_count == 0:
        print("No solution exists.")
        f.write("No solution exists.\n")
    else:
        print(f"\nTotal models found: {model_count}")
        f.write(f"Total models found: {model_count}\n")