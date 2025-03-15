import os
import yaml
import re
from networkx.classes.digraph import DiGraph
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np

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
    print(f'random_value is {random_value}')
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


def build_graph_halton(x_length=20, y_length=20, n_points=700):
    points_with_label = {(1, 19): 'a',
                         (11, 19): '' ,
                         (9, 11): '' ,
                         (11, 9): '' ,
                         (1, 6.5): 'b', 
                         (5.5, 9.5): 'c',
                         (8, 6.5): 'd',
                         (6.5, 3): 'e', # unload
                         (1, 13.5): 'f',
                         (9, 16.5): 'g',
                         (5, 16): 'h', # unload
                         (11, 13.5): 'i',
                         (11, 16.5): 'j',
                         (18, 16.5): 'k',
                         (16, 13): 'l', # unload
                         (11, 4): 'm',
                         (19, 6.5): 'n',
                         (16, 5.5): 'o'} # unload

    x_length = 20
    n_points = 1000
    x = halton_sequence(n_points, 2) * 20
    y = halton_sequence(n_points, 3) * 20
    points = np.vstack((x, y)).T

    # Filter points (pseudo-code)
    obstacles = []  # List of Shapely polygons
    lines = [LineString([(0, 3), (2, 3), (2, 4)]),
            LineString([(0, 5), (2, 5)]),
            LineString([(0, 7), (2, 7), (2, 6)]),
            LineString([(4, 9), (6, 9), (6, 10)]),
            LineString([(6, 7), (4, 7), (4, 5)]),
            LineString([(5, 5), (7, 5), (7, 7)]),
            LineString([(8, 5), (8, 7), (10, 7)]),
            LineString([(4, 2), (4, 4), (5, 4)]),
            LineString([(6, 4), (7, 4), (7, 2), (5, 2)]),

            LineString([(10, 3), (12, 3), (12, 4)]),
            LineString([(10, 5), (12, 5)]),
            LineString([(10, 7), (12, 7), (12, 6)]),
            LineString([(14, 9), (16, 9), (16, 10)]),
            LineString([(16, 7), (14, 7), (14, 5)]),
            LineString([(15, 5), (17, 5), (17, 7)]),
            LineString([(18, 5), (18, 7), (20, 7)]),
            LineString([(14, 2), (14, 4), (15, 4)]),
            LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),

            LineString([(0, 13), (2, 13), (2, 14)]),
            LineString([(0, 15), (2, 15)]),
            LineString([(0, 17), (2, 17), (2, 16)]),
            LineString([(4, 19), (6, 19), (6, 20)]),
            LineString([(6, 17), (4, 17), (4, 15)]),
            LineString([(5, 15), (7, 15), (7, 17)]),
            LineString([(8, 15), (8, 17), (10, 17)]),
            LineString([(4, 12), (4, 14), (5, 14)]),
            LineString([(6, 14), (7, 14), (7, 12), (5, 12)]),
            
            LineString([(10, 13), (12, 13), (12, 14)]),
            LineString([(10, 15), (12, 15)]),
            LineString([(10, 17), (12, 17), (12, 16)]),
            LineString([(14, 19), (16, 19), (16, 20)]),
            LineString([(16, 17), (14, 17), (14, 15)]),
            LineString([(15, 15), (17, 15), (17, 17)]),
            LineString([(18, 15), (18, 17), (20, 17)]),
            LineString([(14, 12), (14, 14), (15, 14)]),
            LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
            
            LineString([(0, 10), (6, 10)]),
            LineString([(10, 0), (10, 7)]),
            LineString([(14, 10), (20, 10)]),
            LineString([(10, 13), (10, 20)])]
    
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
    
    blocks = []  # List of Shapely polygons
    lines = [LineString([(5, 4), (6, 4)]),
             LineString([(4, 15), (5, 15)])]
    for line in lines:
        buffered = line.buffer(distance=0.1, cap_style=3)
        blocks.append(buffered)
        
    A = Point(from_pose)
    B = Point(to_pose)
    connection = LineString([A, B])
    if not any(connection.intersects(block) for block in blocks):
        return False
    return True
        
def check_in_bump(action, nodes):
    #from_pose_index = extract_numbers(str(action))[0]
    to_pose_index = extract_numbers(str(action))[1]
    #from_pose = nodes[f'{from_pose_index}']['attr']['pose']
    to_pose = nodes[f'{to_pose_index}']['attr']['pose']
    
    bumps = []
    coords = [[(4.1, 1.1), (4.1, 2.0), (6.0, 2.0), (6.0, 1.1)], \
              [(1.1, 3.1), (1.1, 5.0), (2.0, 5.0), (2, 4), (3.0, 4.0), (3.0, 3.0)], \
              [(6, 6), (6, 7), (7, 7), (7, 6)], \
                [(7, 1), (7, 3), (8, 3), (8, 1)]]
    for coord in coords:
        polygon = Polygon(coord)
        bumps.append(polygon)
    
    if not any(poly.contains(Point(to_pose)) for poly in bumps):
        return False
    return True

def state_models_from_ts(TS_dict, initial_states_dict=None):
    state_models = []

    nodes, actions = build_graph_halton(20, 20, 200)
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
    print(parent_dir)
    file_path = parent_dir +'/'+ file_name
    print(file_path)
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass