import os
import re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point, Polygon

def plot_initial_map():
    """
    Generates and displays a plot of the initial map layout, including walls,
    Halton points, task locations, and initial robot positions.
    This script combines visualization elements from both multi_source_cluster.py
    and showmove_node.py to provide a static overview of the environment setup.
    """
    # --- Parameters & File Paths ---
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, '..', 'config')
    task_points_yaml = os.path.join(config_dir, 'Task_Points.yaml')
    wall_yaml = os.path.join(config_dir, 'wall.yaml')
    halton_points_csv = os.path.join(current_dir, 'all_points_in_Halton.csv')
    output_figure = os.path.join(current_dir, 'initial_map_40x40.png')
    wall_thick = 0.2 

    # As defined in showmove_node.py
    GREEN = (107/255, 142/255, 35/255)       # For unload task points
    CYAN = (0/255, 255/255, 255/255)         # For unfinished load task points
    RED = (255/255, 0/255, 0/255)            # For special unfinished tasks
    BLACK = (0, 0, 0)
    # For robots (unloaded state)
    PINK_UNLOADED = (255/255, 182/255, 193/255)    # Special robot
    BLUE_UNLOADED = (135/255, 206/255, 250/255)   # Normal robot


    with open(task_points_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)

    task_points = {}
    if 'task_points' in yaml_data:
        for k, v in yaml_data['task_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                task_points[(x, y)] = v

    delivery_points = {}
    if 'delivery_points' in yaml_data:
        for k, v in yaml_data['delivery_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                delivery_points[(x, y)] = v

    robot_positions = {}
    if 'robot_positions' in yaml_data:
        for robot, coord_str in yaml_data['robot_positions'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                robot_positions[robot] = (x, y)

    special_robot_ids = set(yaml_data.get('special_robot', []))
    special_labels = set(yaml_data.get('special_labels', []))

    with open(wall_yaml, 'r') as f:
        wall_data = yaml.safe_load(f)
    wall_lines = [LineString(coords) for coords in wall_data.get('lines', [])]
    obstacles = [line.buffer(wall_thick, cap_style=3) for line in wall_lines]

    halton_df = pd.read_csv(halton_points_csv)
    halton_coords = np.array([(row['x'], row['y']) for _, row in halton_df.iterrows()])
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Draw Known Walls (dark gray)
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='dimgray', alpha=1.0, zorder=2)


    # for poly in BLOCK_POLYGONS:
    #     x, y = poly.exterior.xy
    #     ax.fill(x, y, color=RED, alpha=0.8, zorder=2)

    # Draw bushes for BUMP_POLYGONS
    # np.random.seed(42)  # For reproducible random bushes
    # for poly in BUMP_POLYGONS:
    #     min_x, min_y, max_x, max_y = poly.bounds
    #     # More circles for a denser bush
    #     num_circles = 35
    #     for _ in range(num_circles):
    #         rand_x = np.random.uniform(min_x, max_x)
    #         rand_y = np.random.uniform(min_y, max_y)
    #         if poly.contains(Point(rand_x, rand_y)):
    #             rand_radius = np.random.uniform(0.1, 0.4)
    #             # Shades of green
    #             color = (np.random.uniform(0.2, 0.4), np.random.uniform(0.5, 0.8), np.random.uniform(0.2, 0.4))
    #             bush_circle = Circle((rand_x, rand_y), rand_radius, color=color, alpha=0.6, zorder=3, ec=None)
    #             ax.add_patch(bush_circle)

    if len(halton_coords) > 0:
        tri = Delaunay(halton_coords)
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                p1_coord, p2_coord = halton_coords[a], halton_coords[b]
                edge = LineString([p1_coord, p2_coord])
                # Check if edge intersects any wall
                if not any(edge.intersects(obs) for obs in obstacles):
                    ax.plot([p1_coord[0], p2_coord[0]], [p1_coord[1], p2_coord[1]], color='lightgray', linewidth=0.5, zorder=1)
    

    task_size = 0.5  # World units
    
    # Loading tasks
    for pt, label in task_points.items():
        x, y = pt
        color = RED if label in special_labels else CYAN
        
        if label in special_labels:
            # Triangle for special tasks
            vertices = [(x, y + task_size/2), (x - task_size/2, y - task_size/2), (x + task_size/2, y - task_size/2)]
            shape = MplPolygon(vertices, color=color, zorder=3)
        else:
            # Rectangle for normal tasks
            shape = Rectangle((x - task_size/2, y - task_size/2), task_size, task_size, color=color, zorder=3)
        ax.add_patch(shape)
        ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black', weight='bold', zorder=5)

    # Delivery points
    for pt, label in delivery_points.items():
        x, y = pt
        shape = Rectangle((x - task_size/2, y - task_size/2), task_size, task_size, color=GREEN, zorder=3)
        ax.add_patch(shape)
        ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black', weight='bold', zorder=5)
        
    for robot_id, pt in robot_positions.items():
        x, y = pt
        is_special = robot_id in special_robot_ids
        radius = 0.5 if is_special else 1/3
        color = PINK_UNLOADED if is_special else BLUE_UNLOADED
        
        circle = Circle((x, y), radius, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, robot_id, fontsize=10, ha='center', va='center', color=BLACK, weight='bold', zorder=5)

    # --- Plot Configuration ---
    ax.add_patch(Rectangle((0, 0), 40, 40, linewidth=2, edgecolor='black', facecolor='none', zorder=0))
    # ax.set_title("Multi-Agent Task Planner - 40x40", fontsize=16)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    print(f"40x40 map plot saved to {output_figure}")
    plt.show()

if __name__ == '__main__':
    plot_initial_map() 