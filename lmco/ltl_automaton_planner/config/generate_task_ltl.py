import yaml

# Define all possible points
all_points = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
# all_points = ['b', 'c', 'd', 'e', 'f', 'g', 'h','i', 'j', 'k', 'l']

# Define groups of start points that share the same unload point
grouped_tasks = {
    'e': ['b', 'c', 'd'],  # These points unload at 'e'
    'h': ['f', 'g'],       # These points unload at 'h'
    'l': ['i', 'j', 'k'],        # These points unload at 'l'
    # 'l': ['j'],
    'o': ['m', 'n']
}

def generate_formula(start, end, all_points):
    """
    Generate the formula based on the given start and end points.
    The format is "<>(start) && [](start -> X(loaded && (<disallowed conditions>) U (end && unloaded)))"
    where disallowed conditions apply to all points except the start and end.
    """
    # Generate the list of disallowed points (all points except start and end)
    disallowed = [p for p in all_points if p not in [start, end]]
    # Create the disallowed condition string, e.g., "(!c && !d && ...)"
    disallowed_str = " && ".join(f"(!{p})" for p in disallowed)
    # Construct the complete formula string
    formula = f"<>({start}) && []({start} -> X(loaded && ({disallowed_str} ) U ({end} && unloaded)))"
    return formula

# Construct the new YAML data
yaml_data = {'tasks': {}}
task_counter = 1  # Task numbering

for unload_point, start_points in grouped_tasks.items():
    for start_point in start_points:
        task_name = f'task{task_counter}'
        formula = generate_formula(start_point, unload_point, all_points)
        yaml_data['tasks'][task_name] = {
            'hard_task': formula,
            'soft_task': ""
        }
        task_counter += 1  # Increment task number

# Overwrite the YAML file with the new data
yaml_file = "/home/nanli/ros2_ws/src/lmco/ltl_automaton_planner/config/task_ltl.yaml"
with open(yaml_file, "w", encoding="utf-8") as f:
    yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)  # Preserve order!

print(f"YAML file '{yaml_file}' has been overwritten successfully!")
