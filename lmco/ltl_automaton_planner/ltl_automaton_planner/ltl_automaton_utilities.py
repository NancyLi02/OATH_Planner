import os
import yaml
import re
from networkx.classes.digraph import DiGraph
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np

# Global lists for dynamic obstacles
BLOCK_POLYGONS = [
    Polygon([(5, 3.9), (6, 3.9), (6, 4.1), (5, 4.1)]),  
    Polygon([(4, 14.9), (5, 14.9), (5, 15.1), (4, 15.1)])  
]

BUMP_POLYGONS = [
    Polygon([(4.1, 1.1), (4.1, 2.0), (2.5, 2.0), (2.5, 1.1)]),
    Polygon([(17.5, 15), (20, 15), (20, 13), (17.5, 13)]),
    Polygon([(17.5, 5), (20, 5), (20, 3), (17.5, 3)])
]

def add_block_polygon(coords):
    """Adds a new block polygon from a list of vertex coordinates."""
    if len(coords) >= 3:
        BLOCK_POLYGONS.append(Polygon(coords))
        print(f"Added new block: {coords}")

def add_bump_polygon(coords):
    """Adds a new bump polygon from a list of vertex coordinates."""
    if len(coords) >= 3:
        BUMP_POLYGONS.append(Polygon(coords))
        print(f"Added new bump: {coords}")

def load_lines_from_yaml():
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner')
    )
    file_path = os.path.join(parent_dir, 'config', 'wall.yaml')

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    line_coords = yaml_data.get('lines', [])
    return [LineString(coords) for coords in line_coords]

def update_graph_with_obstacle(nodes, actions, new_obstacle_polygon):
    """
    Locally updates the graph by removing nodes and actions that fall within a new obstacle,
    and removing edges that intersect with the new obstacle.
    """
    nodes_to_remove = set()
    actions_to_remove = set()

    # Step 1: Identify nodes inside the new obstacle
    for node_id, node_data in nodes.items():
        pose = node_data['attr']['pose']
        if new_obstacle_polygon.contains(Point(pose)):
            nodes_to_remove.add(node_id)
    
    # Step 2: Identify edges that intersect the new obstacle
    for action_name in list(actions.keys()):
        if not action_name.startswith("from_"):
            continue
        
        try:
            from_idx, to_idx = (str(n) for n in extract_numbers(action_name))
        except (TypeError, ValueError):
            continue

        if from_idx not in nodes_to_remove and to_idx not in nodes_to_remove:
            if from_idx in nodes and to_idx in nodes:
                from_pose = nodes[from_idx]['attr']['pose']
                to_pose = nodes[to_idx]['attr']['pose']
                
                connection = LineString([from_pose, to_pose])
                
                if connection.intersects(new_obstacle_polygon):
                    actions_to_remove.add(action_name)
                    reverse_action_name = f"from_{to_idx}_to_{from_idx}"
                    actions_to_remove.add(reverse_action_name)

    if not nodes_to_remove and not actions_to_remove:
        print("New obstacle does not conflict with any existing nodes or edges.")
        return

    # Step 3: Consolidate all actions connected to nodes that are being removed
    for node_id in nodes_to_remove:
        if node_id in nodes:
            for action in nodes[node_id]['connected_to'].values():
                actions_to_remove.add(action)

    # Step 4: Execute removals
    if nodes_to_remove:
        print(f"Removing {len(nodes_to_remove)} nodes inside the new obstacle.")
        for node_id in nodes_to_remove:
            if node_id in nodes:
                del nodes[node_id]

    if actions_to_remove:
        print(f"Removing {len(actions_to_remove)} actions due to conflicts.")
        for action_name in actions_to_remove:
            if action_name in actions:
                del actions[action_name]
        
        for node_id in list(nodes.keys()): # Iterate over a copy of the keys
            if node_id not in nodes: # Check if node still exists
                continue
            connections_to_pop = []
            for connected_node_id, action_name in nodes[node_id]['connected_to'].items():
                if action_name in actions_to_remove or connected_node_id in nodes_to_remove:
                    connections_to_pop.append(connected_node_id)
            
            for conn_id in connections_to_pop:
                if conn_id in nodes[node_id]['connected_to']:
                    del nodes[node_id]['connected_to'][conn_id]

# Import TS and action attributes from file
def import_ts_from_file(transition_system_textfile):
    try:
        with open(transition_system_textfile, 'r') as file:
            return yaml.safe_load(file)
    except:
        raise ValueError("cannot load transition system from textfile")

def halton_sequence(size, base=2):
    sequence = []
    for i in range(1, size+1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i = i // base
        sequence.append(r)
    return np.array(sequence)

def point_to_lines_distance(point, lines):
    return min(line.distance(point) for line in lines)

# Probability density function based on distance
def density_probability(d, d_min, d_opt, sigma, floor):
    if d < d_min:
        return 0
    return floor + (1 - floor) * np.exp(-((d - d_opt) ** 2) / (2 * sigma ** 2))

# Rejection sampling algorithm
def rejection_sampling(n_samples, lines, area_size, d_min=0.3, d_opt=0.4, sigma=0.5, floor=0.2):
    np.random.seed(42)
    random_value = np.random.rand()
    # print(f'random_value is {random_value}')
    samples = []
    multiplier = 10
    while len(samples) < n_samples:
        halton_x = halton_sequence(n_samples * multiplier, 2) * area_size
        halton_y = halton_sequence(n_samples * multiplier, 3) * area_size
        for x, y in zip(halton_x, halton_y):
            if len(samples) >= n_samples:
                break
            p = Point(x, y)
            d = point_to_lines_distance(p, lines)
            if d < d_min:
                continue
            prob = density_probability(d, d_min, d_opt, sigma, floor)
            if np.random.rand() < prob:
                samples.append(p)
        multiplier += 5
    return samples[:n_samples]


def build_graph_halton(x_length=20, y_length=20, n_points=700, new_task_points=None):
    points_with_label = load_points_with_label(new_task_points)

    x_length = 20
    n_points = 1000
    x = halton_sequence(n_points, 2) * 20
    y = halton_sequence(n_points, 3) * 20
    points = np.vstack((x, y)).T

    # Filter points (pseudo-code)
    obstacles = []  # List of Shapely polygons
    lines = load_lines_from_yaml()
    
    for line in lines:
        buffered = line.buffer(distance=0.1, cap_style=3)
        obstacles.append(buffered)
        
    # valid_points = [Point(p) for p in points if not any(poly.contains(Point(p)) for poly in obstacles)]
    valid_points = rejection_sampling(n_points, lines, x_length)
    # print(valid_points)
    nodes = dict()
    actions = dict()
    index = 0

    for p in valid_points:
        cell_key = f'{index}'
        position = [p.x, p.y]
        nodes[cell_key] = {
            'attr': {
                'pose': tuple(position),
                'labels': [],
            },
            'connected_to': {f'{index}':'stay'}
        }
        index = index + 1
        
    for key, value in points_with_label.items():
        cell_key = f'{index}'
        nodes[cell_key] = {
            'attr': {
                'pose': key,
                'labels': [points_with_label[key]],
            },
            'connected_to': {f'{index}':'stay'}
        }
        valid_points.append(Point(key))
        # print(f"Point {key} assigned index: {index}")
        index = index + 1

    # print(valid_points)

    tri = Delaunay([(p.x, p.y) for p in valid_points])
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            # print("a: ", a)
            if a < b:  # Avoid duplicates
                line = LineString([valid_points[a], valid_points[b]])
                if not any(line.intersects(obstacle) for obstacle in obstacles):
                    edges.add(line)
                    nodes[f'{a}']["connected_to"][f'{b}'] = f'from_{a}_to_{b}'
                    nodes[f'{b}']["connected_to"][f'{a}'] = f'from_{b}_to_{a}'
                    actions[f'from_{a}_to_{b}'] = {
                        'guard': '1',
                        'type': 'move',
                        'weight':math.dist([valid_points[a].x, valid_points[a].y], 
                                        [valid_points[b].x, valid_points[b].y])
                    }
                    actions[f'from_{b}_to_{a}'] = {
                        'guard': '1',
                        'type': 'move',
                        'weight':math.dist([valid_points[a].x, valid_points[a].y], 
                                        [valid_points[b].x, valid_points[b].y])
                    }
                    
    return nodes, actions

def check_in_block(action, nodes):
    from_pose_index = extract_numbers(str(action))[0]
    to_pose_index = extract_numbers(str(action))[1]
    from_pose = nodes[f'{from_pose_index}']['attr']['pose']
    to_pose = nodes[f'{to_pose_index}']['attr']['pose']
    
    # blocks = []  # List of Shapely polygons
    blocks = BLOCK_POLYGONS
        
    A = Point(from_pose)
    B = Point(to_pose)

    if any(block.contains(A) or block.contains(B) for block in blocks):
        return True
    
    connection = LineString([A, B])

    return any(connection.intersects(block) for block in blocks)
        
def check_in_bump(action, nodes, agent_name):
    # Only 'robot2' and 'robot3' can possibly trigger a bump check; others always return False.
    if agent_name not in ['robot2', 'robot3']:
        return False

    to_pose_index = extract_numbers(str(action))[1]
    to_pose = nodes[f'{to_pose_index}']['attr']['pose']

    # Define bump polygon coordinates
    bumps = BUMP_POLYGONS
    
    # Return True if the to_pose is contained within any of the bump polygons, otherwise return False.
    if any(poly.contains(Point(to_pose)) for poly in bumps):
        return True
    return False


def state_models_from_ts(TS_dict, initial_states_dict=None, new_task_points=None):
    state_models = []

    # Only rebuild the graph from scratch if new task points are being added.
    # Otherwise, use the graph that has been updated in-place in the calling node.
    if new_task_points:
        # Note: The parameters for build_graph_halton might need to be configurable
        # if they differ from the initial setup. For now, they are hardcoded.
        nodes, actions = build_graph_halton(20, 20, 1000, new_task_points)
        TS_dict['state_models']['2d_pose_region']['nodes'] = nodes
        TS_dict['actions'].update(actions)
    
    # If initial states are given as argument
    if initial_states_dict:
        # Check that dimensions are conform
        if (len(initial_states_dict.keys()) != len(TS_dict['state_dim'])):
            raise ValueError("initial states don't match TS state models: "+len(initial_states)+" initial states and "+len(TS_dict['state_dim'])+" state models")

    # For every state model define in file, using state_dim to ensure order (dict are not ordered)
    for model_dim in TS_dict['state_dim']:
        state_model_dict = TS_dict['state_models'][model_dim]
        state_model = DiGraph(initial=set(), ts_state_format=[str(model_dim)])
        #------------------
        # Create all nodes
        #------------------
        for node in state_model_dict['nodes']:
            # state_model.add_node(tuple([node]), label=set([str(node)]))
            state_model.add_node(tuple([node]), label=set(state_model_dict['nodes'][node]['attr']['labels']))
        # If no initial states in arguments, use initial state from TS dict
        if not initial_states_dict:
            state_model.graph['initial']=set([tuple([state_model_dict['initial']])])
        else:
            state_model.graph['initial']=set([tuple([initial_states_dict[model_dim]])])
        #----------------------------------
        # Connect previously created nodes
        #----------------------------------
        # Go through all nodes
        for node in state_model_dict['nodes']:
            # Go through all connected node
            for connected_node in state_model_dict['nodes'][node]['connected_to']:
                # Add edge between node and connected node
                # Get associated action from "connected_to" tag of state node
                act = TS_dict['state_models'][model_dim]["nodes"][node]['connected_to'][connected_node]
                # Use action to retrieve weight and guard from action dictionnary
                act_guard = TS_dict['actions'][act]['guard']
                act_weight = TS_dict['actions'][act]['weight']
                state_model.add_edge(tuple([node]), tuple([connected_node]), action = act, guard = act_guard, weight = act_weight)
        #-------------------------
        # Add state model to list
        #-------------------------
        state_models.append(state_model)

    return state_models

def handle_ts_state_msg(ts_state_msg):
    # Extract TS state from request message
    # If only 1-dimensional state, state is directly a string
    #   in this case create tuple from the state from message array without the function tuple() 
    #   to avoid conversion from string to tuple of char
    if not (len(ts_state_msg.states) == len(ts_state_msg.state_dimension_names)):
        raise ValueError("Received TS states don't match TS state models: "+str(len(ts_state_msg.states))+" initial states and "+str(len(ts_state_msg.state_dimension_names))+" state models")
    elif len(ts_state_msg.states) > 1:
        ts_state = tuple(ts_state_msg.states)
        return ts_state
    elif len(ts_state_msg.states) == 1:
        ts_state = (ts_state_msg.states[0],)
        return ts_state
    else:
        raise ValueError("received empty TS state")

    #TODO Add check for message malformed (not corresponding fields)


def extract_numbers(input_string):
    # Define a regular expression pattern to find numbers
    pattern = re.compile(r'\d+')

    # Find all matches in the input string
    matches = pattern.findall(input_string)

    # Extract the first and second numbers
    if len(matches) >= 2:
        first_number = int(matches[0])
        second_number = int(matches[1])
        return (first_number, second_number)
    else:
        # Handle the case where there are not enough numbers
        return None


def read_yaml_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_ = os.path.dirname(script_dir)
    parent_dir = os.path.abspath(os.path.join(script_dir_, '..', 'log/'))
    file_path = parent_dir +'/'+ file_name
    try:
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data if data is not None else list()
            except yaml.YAMLError as e:
                print(f"Error reading YAML file: {e}")
                return None
    except IOError as e:
        return list()

def write_to_yaml(data, file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_ = os.path.dirname(script_dir)
    parent_dir = os.path.abspath(os.path.join(script_dir_, '..', 'log/')) 
    file_path = parent_dir +'/'+ file_name
    try:
        with open(file_path, 'w') as file:
            try:
                yaml.dump(data, file)
            except yaml.YAMLError as e:
                print(f"Error writing to YAML file: {e}")
    except IOError as e:
        pass
            
def delete_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_ = os.path.dirname(script_dir)
    parent_dir = os.path.abspath(os.path.join(script_dir_, '..', 'log/'))
    # print(parent_dir)
    file_path = parent_dir +'/'+ file_name
    # print(file_path)
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def load_points_with_label(new_task_points=None):
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner')
    )
    yaml_path = os.path.join(parent_dir, 'config', 'Task_Points.yaml')
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    points_with_label = {}

    for robot, pos in data.get('robot_positions', {}).items():
        point = tuple(float(x.strip()) for x in pos.split(','))
        points_with_label[point] = ''


    for k, v in data.get('task_points', {}).items():
        point = tuple(float(x.strip()) for x in k.split(','))
        points_with_label[point] = v


    for k, v in data.get('delivery_points', {}).items():
        point = tuple(float(x.strip()) for x in k.split(','))
        points_with_label[point] = v


    if new_task_points:
        for point, label in new_task_points:
            points_with_label[point] = label
            print(f"New task point: {point}, label: {label}")

    return points_with_label