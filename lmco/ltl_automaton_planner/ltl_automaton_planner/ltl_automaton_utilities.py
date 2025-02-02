import os
import yaml
import re
from networkx.classes.digraph import DiGraph

# Import TS and action attributes from file
def import_ts_from_file(transition_system_textfile):
    try:
        with open(transition_system_textfile, 'r') as file:
            return yaml.safe_load(file)
    except:
        raise ValueError("cannot load transition system from textfile")

def state_models_from_ts(TS_dict, initial_states_dict=None):
    state_models = []

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
