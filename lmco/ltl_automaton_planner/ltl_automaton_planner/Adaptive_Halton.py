import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import yaml
import csv
import os
import re

# Global parameter definitions
d_min = 0.3       # Minimum allowed distance to avoid points being too close to obstacles
d_opt = 0.4       # Optimal distance (highest sampling probability)
sigma = 0.5       # Controls the width of the probability distribution
floor_prob = 0.2  # Minimum sampling probability in open areas
wall_thick = 0.1  # Thickness of the walls

# Halton sequence generation function
def halton_sequence(size, base=2):
    sequence = []
    for i in range(1, size+1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        sequence.append(r)
    return np.array(sequence)

# === 从YAML读取机器人、pickup、delivery点 ===

task_points_yaml = '/home/nanli/ros2_ws/src/lmco/ltl_automaton_planner/config/Task_Points.yaml'
with open(task_points_yaml, 'r') as f:
    yaml_data = yaml.safe_load(f)

points_with_label = {}
# 机器人初始点
if 'robot_positions' in yaml_data:
    for robot, coord_str in yaml_data['robot_positions'].items():
        match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            points_with_label[(x, y)] = robot
# pickup点
for k, v in yaml_data['task_points'].items():
    match = re.match(r"([\d\.]+),([\d\.]+)", k)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        points_with_label[(x, y)] = v
# delivery点
if 'delivery_points' in yaml_data:
    for k, v in yaml_data['delivery_points'].items():
        match = re.match(r"([\d\.]+),([\d\.]+)", k)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            points_with_label[(x, y)] = v

x_length, y_length = 20, 20

# ---------- Walls ----------
def load_lines_from_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    line_coords = data.get('lines', [])
    return [LineString(coords) for coords in line_coords]

filepath = "lmco/ltl_automaton_planner/config/wall.yaml"
lines = load_lines_from_yaml(filepath)

def point_to_lines_distance(point, lines):
    return min(line.distance(point) for line in lines)

# Probability density function based on distance
def density_probability(d, d_min, d_opt, sigma, floor):
    if d < d_min:
        return 0
    return floor + (1 - floor) * np.exp(-((d - d_opt) ** 2) / (2 * sigma ** 2))

# Rejection sampling algorithm - modified to match utilities behavior
def rejection_sampling(n_samples, lines, area_size, d_min, d_opt, sigma, floor):
    # Set random seed inside the function to match utilities behavior
    np.random.seed(42)
    # Add the random_value call to match utilities exactly
    random_value = np.random.rand()
    
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

# Generate valid sampling points
valid_points = rejection_sampling(1000, lines, x_length, d_min, d_opt, sigma, floor_prob)

# Add labeled points to the valid points
for key in points_with_label.keys():
    valid_points.append(Point(key))

# Delaunay triangulation
tri = Delaunay([(p.x, p.y) for p in valid_points])
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i+1)%3]
        if a < b:
            line_edge = LineString([valid_points[a], valid_points[b]])
            if not any(line_edge.intersects(obs) for obs in [line.buffer(wall_thick, cap_style=3) for line in lines]):
                edges.add(line_edge)

# Visualization
obstacles = [line.buffer(wall_thick, cap_style=3) for line in lines]

fig, ax = plt.subplots(figsize=(8, 8))

for buffered in obstacles:
    x_buffered, y_buffered = buffered.exterior.xy
    ax.fill(x_buffered, y_buffered, alpha=0.6, color='red')

for edge in edges:
    ax.plot(*edge.xy, color='lightblue', linewidth=0.8)

ax.scatter([p.x for p in valid_points], [p.y for p in valid_points], s=5, color='blue')

ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)
ax.set_aspect('equal')
# plt.title('Adaptive Halton Sequence Map')
plt.grid(True)
plt.show()

# ---------- Save all point coordinates to CSV ----------
output_filename = 'all_points_in_Halton.csv'
with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'label'])

    for p in valid_points:
        coord = (round(p.x, 4), round(p.y, 4))
        label = points_with_label.get(coord, '')
        writer.writerow([p.x, p.y, label])

print(f"All points saved to '{output_filename}'")
