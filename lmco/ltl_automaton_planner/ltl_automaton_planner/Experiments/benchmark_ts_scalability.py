#!/usr/bin/env python
"""
Benchmark script to test the relationship between the number of sampling points
in the transition system and the time to build product automaton / find optimal run.

This script tests with different numbers of sampling points: 500, 1000, 2000, 3000, 5000
Each configuration is run 5 times to get statistically meaningful results.
Results are saved to a JSON file for later visualization.

This script uses the same approach as planner_cluster_node.py
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString
import math
import yaml

# Add the parent directory to path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to ltl_automaton_planner
# Insert at the beginning to ensure we use source code, not installed package
sys.path.insert(0, parent_dir)

# Use the same imports as planner_cluster_node.py
from ltl_automaton_planner.ltl_tools.product import ProdAut
from ltl_automaton_planner.ltl_tools.buchi import mission_to_buchi
from ltl_automaton_planner.ltl_tools.ts import TSModel
from ltl_automaton_planner.ltl_tools.ltl_planner import LTLPlanner

# Import from ltl_automaton_utilities - same as planner_cluster_node.py
# Import directly from source file to avoid using installed package
import importlib.util
ltl_utils_path = os.path.join(parent_dir, 'ltl_automaton_utilities.py')
spec = importlib.util.spec_from_file_location("ltl_automaton_utilities", ltl_utils_path)
ltl_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ltl_utils)

# Import the functions we need
state_models_from_ts = ltl_utils.state_models_from_ts
import_ts_from_file = ltl_utils.import_ts_from_file
halton_sequence = ltl_utils.halton_sequence
rejection_sampling = ltl_utils.rejection_sampling


def get_config_dir():
    """Get the config directory path - works when running directly from source"""
    # From Experiments/ directory, go up to ltl_automaton_planner package root, then to config
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))


def load_lines_from_yaml_local():
    """Load wall lines from wall.yaml - local version with correct path"""
    file_path = os.path.join(get_config_dir(), 'wall.yaml')
    
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    line_coords = yaml_data.get('lines', [])
    return [LineString(coords) for coords in line_coords]


def load_points_with_label_local(new_task_points=None):
    """Load task points from Task_Points.yaml - local version with correct path
    
    Returns:
        points_with_label: dict of (x,y) -> label
        robot_positions: dict of robot_name -> index (to be filled after building graph)
    """
    yaml_path = os.path.join(get_config_dir(), 'Task_Points.yaml')
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    points_with_label = {}
    robot_position_order = []  # Track order of robot positions

    for robot, pos in data.get('robot_positions', {}).items():
        point = tuple(float(x.strip()) for x in pos.split(','))
        points_with_label[point] = ''
        robot_position_order.append((robot, point))

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

    return points_with_label, robot_position_order


def build_graph_halton_variable(n_points, x_length=20, y_length=20, new_task_points=None, initial_robot='robot2', random_seed=None):
    """
    Build graph with Halton sequence sampling - MODIFIED to actually use n_points parameter.
    This is a copy of build_graph_halton from ltl_automaton_utilities.py, 
    but fixed to use the n_points parameter instead of hardcoding 1000.
    
    Args:
        n_points: Number of sampling points to generate (actually used!)
        x_length: X dimension of the area
        y_length: Y dimension of the area
        new_task_points: Optional new task points to add
        initial_robot: Which robot position to use as initial state (default: 'robot2')
        random_seed: Optional random seed for sampling. If None, uses current random state.
    
    Returns:
        nodes: Dictionary of nodes
        actions: Dictionary of actions
        initial_node_index: Index of the specified robot position
    """
    # Use local versions of these functions with correct paths
    points_with_label, robot_position_order = load_points_with_label_local(new_task_points)

    # Load obstacles from wall.yaml
    lines = load_lines_from_yaml_local()
    obstacles = []
    for line in lines:
        buffered = line.buffer(distance=0.1, cap_style=3)
        obstacles.append(buffered)
        
    # Generate valid points using rejection sampling - use the actual n_points parameter!
    valid_points = rejection_sampling(n_points, lines, x_length, random_seed=random_seed)
    
    nodes = dict()
    actions = dict()
    index = 0

    # Add sampling points (indices 0 to n_points-1)
    for p in valid_points:
        cell_key = f'{index}'
        position = [p.x, p.y]
        nodes[cell_key] = {
            'attr': {
                'pose': tuple(position),
                'labels': [],
            },
            'connected_to': {f'{index}': 'stay'}
        }
        index = index + 1
    
    # Record the starting index for labeled points (robot positions, task points, etc.)
    first_labeled_index = index
    
    # Build a map from position to robot name for tracking
    robot_pos_to_name = {pos: name for name, pos in robot_position_order}
    robot_indices = {}
        
    # Add task points with labels (robot positions, task points, delivery points)
    # IMPORTANT: Must use [points_with_label[key]] exactly like the original build_graph_halton
    # to ensure labels are consistent (even empty string labels must be in a list)
    for key, value in points_with_label.items():
        cell_key = f'{index}'
        nodes[cell_key] = {
            'attr': {
                'pose': key,
                'labels': [points_with_label[key]],  # Same as original - always a list with the value
            },
            'connected_to': {f'{index}': 'stay'}
        }
        valid_points.append(Point(key))
        
        # Track robot position indices as we create nodes
        if key in robot_pos_to_name:
            robot_indices[robot_pos_to_name[key]] = index
        
        index = index + 1
    
    # Get the initial node index for the specified robot
    initial_node_index = robot_indices.get(initial_robot, first_labeled_index)
    print(f"    Robot positions: {robot_indices}")
    print(f"    Using {initial_robot} at index {initial_node_index} as initial position")

    # Delaunay triangulation for edges
    tri = Delaunay([(p.x, p.y) for p in valid_points])
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1) % 3]
            if a < b:  # Avoid duplicates
                line = LineString([valid_points[a], valid_points[b]])
                if not any(line.intersects(obstacle) for obstacle in obstacles):
                    edges.add(line)
                    nodes[f'{a}']["connected_to"][f'{b}'] = f'from_{a}_to_{b}'
                    nodes[f'{b}']["connected_to"][f'{a}'] = f'from_{b}_to_{a}'
                    actions[f'from_{a}_to_{b}'] = {
                        'guard': '1',
                        'type': 'move',
                        'weight': math.dist([valid_points[a].x, valid_points[a].y], 
                                           [valid_points[b].x, valid_points[b].y])
                    }
                    actions[f'from_{b}_to_{a}'] = {
                        'guard': '1',
                        'type': 'move',
                        'weight': math.dist([valid_points[a].x, valid_points[a].y], 
                                           [valid_points[b].x, valid_points[b].y])
                    }
                    
    return nodes, actions, initial_node_index


def get_transition_system_path():
    """Get path to isaac_known.yaml transition system file"""
    return os.path.join(get_config_dir(), 'isaac_known.yaml')


def run_benchmark(n_points, ltl_formula, initial_beta=1000, gamma=10, algo_type='dstar', grid_size=40, random_seed=None):
    """
    Run a single benchmark with specified number of sampling points.
    Uses the same approach as planner_cluster_node.py
    
    Args:
        n_points: Number of sampling points
        ltl_formula: LTL formula to use
        initial_beta: Initial beta parameter
        gamma: Gamma parameter
        algo_type: Algorithm type
        grid_size: Grid size
        random_seed: Optional random seed for sampling. If None, uses current random state.
    
    Returns:
        dict with timing results
    """
    results = {
        'n_points': n_points,
        'actual_nodes': 0,
        'actual_edges': 0,
        'initial_node': 0,
        'graph_build_time': 0,
        'state_model_build_time': 0,
        'buchi_build_time': 0,
        'product_build_time': 0,
        'optimal_run_time': 0,
        'total_time': 0,
        'success': False
    }
    
    total_start = time.time()
    
    try:
        # ============================================================
        # Step 1: Build the graph with Halton sampling
        # Same as: self.nodes, self.actions = build_graph_halton(20, 20, 1000)
        # But with variable n_points
        # ============================================================
        print(f"  Building graph with {n_points} sampling points...")
        graph_start = time.time()
        nodes, actions, initial_node_index = build_graph_halton_variable(n_points, initial_robot='robot2', random_seed=random_seed)
        graph_end = time.time()
        results['graph_build_time'] = graph_end - graph_start
        results['actual_nodes'] = len(nodes)
        results['actual_edges'] = len([a for a in actions if a.startswith('from_')])
        results['initial_node'] = initial_node_index  # robot2's position index
        print(f"    Actual nodes: {len(nodes)}, initial node (robot2): {initial_node_index}")
        
        # ============================================================
        # Step 2: Load transition system and update with new graph
        # Same as in planner_cluster_node.py init_params()
        # ============================================================
        transition_system_textfile = get_transition_system_path()
        transition_system = import_ts_from_file(transition_system_textfile)
        transition_system['state_models']['2d_pose_region']['nodes'] = nodes
        transition_system['actions'].update(actions)
        
        # ============================================================
        # Step 3: Set initial state
        # Same as: self.initial_state_ts_dict = {'2d_pose_region': f'{self.init_state}', 'Drone_state': 'unloaded'}
        # Use initial_node_index which is robot2's position (like init_state=1001 in launch file for 1000 points)
        # ============================================================
        initial_state_ts_dict = {
            '2d_pose_region': f'{initial_node_index}',
            'Drone_state': 'unloaded'
        }
        print(f"  Initial state: {initial_state_ts_dict}")
        
        # ============================================================
        # Step 4: Build state models
        # Same as: state_models = state_models_from_ts(self.transition_system, initial_state_ts_dict, self.new_task_points)
        # Note: In the original node, new_task_points is usually an empty list [], not None
        # ============================================================
        print("  Building state models...")
        state_model_start = time.time()
        state_models = state_models_from_ts(transition_system, initial_state_ts_dict, [])
        state_model_end = time.time()
        results['state_model_build_time'] = state_model_end - state_model_start
        
        # ============================================================
        # Step 5: Build Buchi automaton
        # Same as: buchi = mission_to_buchi(hard_task, soft_task)
        # ============================================================
        print("  Building Buchi automaton...")
        buchi_start = time.time()
        hard_task = ltl_formula
        soft_task = ''
        buchi = mission_to_buchi(hard_task, soft_task)
        buchi_end = time.time()
        results['buchi_build_time'] = buchi_end - buchi_start
        
        # ============================================================
        # Step 6: Build TS model, Product automaton
        # Same as in build_and_run_automaton():
        #   self.robot_model = TSModel(state_models)
        #   self.product_automaton = ProdAut(self.robot_model, buchi, self.initial_beta)
        #   self.product_automaton.graph['ts'].build_full()
        #   self.product_automaton.build_full_relaxed()
        # ============================================================
        print("  Building product automaton...")
        product_start = time.time()
        robot_model = TSModel(state_models)
        print("    Step 1: finish building robot model")
        
        product_automaton = ProdAut(robot_model, buchi, initial_beta)
        print("    Step 2: finish building product automaton structure")
        
        product_automaton.graph['ts'].build_full()
        print("    Step 3: finish building TS full graph")
        
        product_automaton.build_full_relaxed()
        print("    Step 4: finish building product automaton full relaxed")
        product_end = time.time()
        results['product_build_time'] = product_end - product_start
        
        # ============================================================
        # Step 7: Find optimal run
        # Same as:
        #   self.ltl_planner = LTLPlanner(self.robot_model, hard_task, soft_task, self.initial_beta, self.gamma)
        #   self.ltl_planner.optimal(self.product_automaton, algo=self.algo_type, N=self.grid_size)
        # ============================================================
        print("  Finding optimal run...")
        optimal_start = time.time()
        ltl_planner = LTLPlanner(robot_model, hard_task, soft_task, initial_beta, gamma)
        print("    Step 5: finish building ltl planner")
        
        ltl_planner.optimal(product_automaton, algo=algo_type, N=grid_size)
        print("    Step 6: finish running ltl planner")
        optimal_end = time.time()
        results['optimal_run_time'] = optimal_end - optimal_start
        
        # Check if we got a valid run
        if ltl_planner.run is not None:
            results['success'] = True
            # Initialize storage of set of possible runs in product (same as node)
            ltl_planner.curr_ts_state = list(ltl_planner.product.graph['ts'].graph['initial'])[0]
            ltl_planner.posb_runs = set([(n,) for n in ltl_planner.product.graph['initial']])
            print("    Step 7: finish initializing ltl planner - SUCCESS!")
            print(f"    Prefix plan length: {len(ltl_planner.run.pre_plan)}")
            print(f"    Suffix plan length: {len(ltl_planner.run.suf_plan)}")
        else:
            print("    WARNING: No valid plan found!")
            results['success'] = False
        
    except Exception as e:
        import traceback
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        results['error'] = str(e)
    
    total_end = time.time()
    results['total_time'] = total_end - total_start
    
    return results


def main():
    # Configuration
    sampling_points_list = [500, 1000, 2000, 3000, 5000]
    num_runs = 5
    
    # Fixed LTL formula (same as provided by user)
    ltl_formula = "<>((cd && loaded1 && !loaded2 && !loaded3) && (<>((bd && loaded2) && (<>((ed && loaded3) && (<>((d && unloaded))))))))"
    
    # Parameters (same as in planner_cluster_node.py)
    # NOTE: Launch file uses 'brute-force' as default, NOT 'dstar'!
    # The dstar algorithm's heuristic function expects node names like 'c1_r2' with two numbers,
    # but Halton-based graphs use simple numeric indices like '890', causing extract_numbers() to return None.
    initial_beta = 1000
    gamma = 10
    algo_type = 'brute-force'  # Changed from 'dstar' to match launch file default
    grid_size = 40
    
    # Results storage
    all_results = {
        'config': {
            'ltl_formula': ltl_formula,
            'sampling_points_list': sampling_points_list,
            'num_runs': num_runs,
            'initial_beta': initial_beta,
            'gamma': gamma,
            'algo_type': algo_type,
            'grid_size': grid_size,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        },
        'results': {}
    }
    
    print("=" * 60)
    print("Product Automaton Scalability Benchmark")
    print("=" * 60)
    print(f"LTL Formula: {ltl_formula}")
    print(f"Sampling points to test: {sampling_points_list}")
    print(f"Number of runs per configuration: {num_runs}")
    print(f"Algorithm: {algo_type}")
    print("=" * 60)
    
    for n_points in sampling_points_list:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points} sampling points")
        print("=" * 60)
        
        all_results['results'][n_points] = []
        
        for run_idx in range(num_runs):
            print(f"\nRun {run_idx + 1}/{num_runs}:")
            # Use different random seed for each run to ensure different sampling points
            # Combine n_points and run_idx to create unique seeds for each configuration and run
            run_seed = n_points * 10000 + run_idx * 1000 + int(time.time() * 1000) % 1000
            result = run_benchmark(
                n_points=n_points,
                ltl_formula=ltl_formula,
                initial_beta=initial_beta,
                gamma=gamma,
                algo_type=algo_type,
                grid_size=grid_size,
                random_seed=run_seed
            )
            all_results['results'][n_points].append(result)
            
            print(f"  Graph build time: {result['graph_build_time']:.3f}s")
            print(f"  State model build time: {result['state_model_build_time']:.3f}s")
            print(f"  Buchi build time: {result['buchi_build_time']:.3f}s")
            print(f"  Product automaton build time: {result['product_build_time']:.3f}s")
            print(f"  Optimal run time: {result['optimal_run_time']:.3f}s")
            print(f"  Total time: {result['total_time']:.3f}s")
            print(f"  Success: {result['success']}")
    
    # Save results to JSON
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = all_results['config']['timestamp']
    output_file = os.path.join(output_dir, f'ts_scalability_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Points':<10} {'Avg Total (s)':<15} {'Avg Product (s)':<18} {'Avg Optimal (s)':<18} {'Success Rate':<15}")
    print("-" * 75)
    
    for n_points in sampling_points_list:
        results = all_results['results'][n_points]
        avg_total = np.mean([r['total_time'] for r in results])
        avg_product = np.mean([r['product_build_time'] for r in results])
        avg_optimal = np.mean([r['optimal_run_time'] for r in results])
        success_rate = sum([1 for r in results if r['success']]) / len(results) * 100
        print(f"{n_points:<10} {avg_total:<15.3f} {avg_product:<18.3f} {avg_optimal:<18.3f} {success_rate:<15.1f}%")
    
    return all_results


if __name__ == '__main__':
    main()
