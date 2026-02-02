import os
import yaml
import numpy as np
import heapq
import math
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import Delaunay


# ------------------------ Parameters ------------------------
d_min = 0.8
d_opt = 2
sigma = 0.5
floor_prob = 0.2
wall_thick = 0.1
x_length, y_length = 18, 18

# ------------------------ Wall Loading ------------------------
current_dir = os.path.dirname(__file__)
wall_path = os.path.join(current_dir, '..', 'config', 'wall.yaml')
wall_path = os.path.abspath(wall_path)
with open(wall_path, 'r') as f:
    wall_data = yaml.safe_load(f)
lines = [LineString(coords) for coords in wall_data.get('lines', [])]
obstacles = [line.buffer(wall_thick, cap_style=3) for line in lines]

# ------------------------ Load Halton Points from CSV ------------------------
def load_halton_points_from_csv(csv_path, task_points_yaml_path):
    """Load Halton points from the pre-generated CSV file and filter task_points + delivery_points"""
    df = pd.read_csv(csv_path)
    points = []
    points_with_label = {}
    
    # Load task_points and delivery_points from YAML
    with open(task_points_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Collect labels from both task_points and delivery_points
    valid_labels = set()
    if 'task_points' in yaml_data:
        for coord_str, label in yaml_data['task_points'].items():
            valid_labels.add(label)
    if 'delivery_points' in yaml_data:
        for coord_str, label in yaml_data['delivery_points'].items():
            valid_labels.add(label)
    
    for _, row in df.iterrows():
        x, y, label = row['x'], row['y'], row['label']
        point = Point(x, y)
        points.append(point)
        
        # Store labels that are in task_points or delivery_points
        if pd.notna(label) and str(label).strip() and str(label) in valid_labels:
            points_with_label[(x, y)] = str(label)
    
    return points, points_with_label

# Load points from the pre-generated CSV file
csv_path = os.path.join(current_dir, 'all_points_in_Halton.csv')
csv_path = os.path.abspath(csv_path)
task_points_yaml_path = os.path.join(current_dir, '..', 'config', 'Task_Points.yaml')
task_points_yaml_path = os.path.abspath(task_points_yaml_path)
valid_points, points_with_label = load_halton_points_from_csv(csv_path, task_points_yaml_path)

print(f"Loaded {len(valid_points)} points from CSV")
print(f"Found {len(points_with_label)} labeled points (task_points + delivery_points)")

point_coords = [(p.x, p.y) for p in valid_points]
point_index = {pt: i for i, pt in enumerate(point_coords)}

# ------------------------ Delaunay Graph ------------------------
tri = Delaunay(point_coords)
graph = {i: [] for i in range(len(point_coords))}
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i+1)%3]
        p1, p2 = valid_points[a], valid_points[b]
        edge = LineString([p1, p2])
        if not any(edge.intersects(obs) for obs in obstacles):
            dist = p1.distance(p2)
            graph[a].append((b, dist))
            graph[b].append((a, dist))

# ------------------------ Dijkstra with Path Recovery ------------------------
def dijkstra_sparse_with_prev(graph, start_idx):
    dist = {i: math.inf for i in graph}
    prev = {}
    dist[start_idx] = 0
    heap = [(0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, cost in graph[u]:
            alt = d + cost
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    return dist, prev

def reconstruct_path(prev, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path

# ------------------------ Label Index Mapping ------------------------
label_indices = {label: point_index[(x, y)] for (x, y), label in points_with_label.items()}

# ------------------------ Visualize Specific Path ------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# 设置字体大小
plt.rcParams.update({'font.size': 18})

# Draw walls
for obs in obstacles:
    x, y = obs.exterior.xy
    ax.fill(x, y, color='red', alpha=0.4)

# Draw all valid graph edges (optional for context)
for u in graph:
    for v, _ in graph[u]:
        x1, y1 = point_coords[u]
        x2, y2 = point_coords[v]
        ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.4)

# Draw all shortest paths between labeled points (task_points + delivery_points)
labels = list(label_indices.keys())
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        label1, label2 = labels[i], labels[j]
        idx1, idx2 = label_indices[label1], label_indices[label2]
        _, prev_map = dijkstra_sparse_with_prev(graph, idx1)
        try:
            path_indices = reconstruct_path(prev_map, idx1, idx2)
        except KeyError:
            continue  # no path exists
        path_coords = [point_coords[k] for k in path_indices]
        path_x = [x for x, y in path_coords]
        path_y = [y for x, y in path_coords]
        ax.plot(path_x, path_y, color='blue', linewidth=1.0, alpha=0.5)

# Draw labeled points (task_points + delivery_points)
for (x, y), label in points_with_label.items():
    ax.plot(x, y, 'bo')
    # ax.text(x + 0.2, y + 0.2, label, fontsize=16)

ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)
ax.set_aspect('equal')
ax.tick_params(axis='both', which='major', labelsize=18)
# ax.set_title("Multi-Source Dijkstra: Shortest Paths Between All Task Points", fontsize=18)
plt.grid(False)
plt.show()

# -------- Save one-way pairwise distances to CSV --------
output_csv = os.path.join(current_dir, 'multi_source_dijkstra_distances.csv')
output_csv = os.path.abspath(output_csv)
with open(output_csv, 'w') as f:
    f.write("from,to,distance\n")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            idx1, idx2 = label_indices[label1], label_indices[label2]
            dist_map, _ = dijkstra_sparse_with_prev(graph, idx1)
            distance = dist_map.get(idx2, math.inf)
            if distance < math.inf:
                f.write(f"{label1},{label2},{round(distance, 3)}\n")

print(f"Distance matrix saved to '{output_csv}'")

