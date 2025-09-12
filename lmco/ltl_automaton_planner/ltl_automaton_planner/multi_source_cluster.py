# Re-run the plotting code after environment reset

import os
import math
import yaml
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering
from matplotlib.patches import Rectangle
import re

# ----------------- Parameters -----------------
current_dir = os.path.dirname(__file__)
input_csv = os.path.join(current_dir, 'multi_source_dijkstra_distances.csv')
output_csv = os.path.join(current_dir, 'clustered_points.csv')
wall_yaml = os.path.join(current_dir, '..', 'config', 'wall.yaml')
output_figure = os.path.join(current_dir, 'dijkstra_clustering_plot.png')
halton_points_csv = os.path.join(current_dir, 'all_points_in_Halton.csv')
wall_thick = 0.1
n_clusters = 4

task_points_yaml = os.path.join(current_dir, '..', 'config', 'Task_Points.yaml')
with open(task_points_yaml, 'r') as f:
    yaml_data = yaml.safe_load(f)

points_with_label = {}

if 'robot_positions' in yaml_data:
    for robot, coord_str in yaml_data['robot_positions'].items():
        match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            points_with_label[(x, y)] = robot

task_points_label_to_coord = {}
for k, v in yaml_data['task_points'].items():
    match = re.match(r"([\d\.]+),([\d\.]+)", k)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        points_with_label[(x, y)] = v
        task_points_label_to_coord[v] = (x, y)
# delivery点
if 'delivery_points' in yaml_data:
    for k, v in yaml_data['delivery_points'].items():
        match = re.match(r"([\d\.]+),([\d\.]+)", k)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            points_with_label[(x, y)] = v


label_to_coord = task_points_label_to_coord

df = pd.read_csv(input_csv)
labels = [label for label in pd.unique(df[['from', 'to']].values.ravel('K')) if label in label_to_coord]
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
n = len(labels)

dist_matrix = np.full((n, n), np.inf)
np.fill_diagonal(dist_matrix, 0.0)
for _, row in df.iterrows():
    if row['from'] not in label_to_index or row['to'] not in label_to_index:
        continue
    i = label_to_index[row['from']]
    j = label_to_index[row['to']]
    dist_matrix[i][j] = row['distance']
    dist_matrix[j][i] = row['distance']

clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
cluster_labels = clustering.fit_predict(dist_matrix)

cluster_df = pd.DataFrame({
    'point': [index_to_label[i] for i in range(n)],
    'cluster': cluster_labels,
    'x': [label_to_coord[index_to_label[i]][0] for i in range(n)],
    'y': [label_to_coord[index_to_label[i]][1] for i in range(n)],
})
cluster_df.to_csv(output_csv, index=False)

# ----------------- Load Walls -----------------
with open(wall_yaml, 'r') as f:
    wall_data = yaml.safe_load(f)
wall_lines = [LineString(coords) for coords in wall_data.get('lines', [])]
obstacles = [line.buffer(wall_thick, cap_style=3) for line in wall_lines]

# ----------------- Load Halton Points -----------------
halton_df = pd.read_csv(halton_points_csv)
halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
halton_points = [Point(x, y) for x, y in halton_coords]

tri = Delaunay(halton_coords)
graph = {i: [] for i in range(len(halton_points))}
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i + 1) % 3]
        p1, p2 = halton_points[a], halton_points[b]
        edge = LineString([p1, p2])
        if not any(edge.intersects(obs) for obs in obstacles):
            dist = p1.distance(p2)
            graph[a].append((b, dist))
            graph[b].append((a, dist))

# ----------------- Dijkstra -----------------
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

# ----------------- Plotting -----------------
fig, ax = plt.subplots(figsize=(10, 8))

# Draw walls
for obs in obstacles:
    x, y = obs.exterior.xy
    ax.fill(x, y, color='red', alpha=0.4)

# Draw Delaunay edges
for u in graph:
    for v, _ in graph[u]:
        x1, y1 = halton_points[u].x, halton_points[u].y
        x2, y2 = halton_points[v].x, halton_points[v].y
        ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.5, alpha=0.5)

# Map task labels to halton point indices
coord_to_index = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(halton_points)}
label_to_idx = {}
for label, (x, y) in label_to_coord.items():
    rounded = (round(x, 4), round(y, 4))
    if rounded in coord_to_index:
        label_to_idx[label] = coord_to_index[rounded]

# Draw clustered task points and shortest paths
colors = plt.cm.tab10(np.arange(n_clusters))
for cluster_id in range(n_clusters):
    cluster_points = cluster_df[cluster_df['cluster'] == cluster_id]['point'].values
    cluster_coords = [label_to_coord[label] for label in cluster_points]
    xs, ys = zip(*cluster_coords)
    ax.scatter(xs, ys, color=colors[cluster_id], s=40)

    for i in range(len(cluster_points)):
        for j in range(i + 1, len(cluster_points)):
            label1, label2 = cluster_points[i], cluster_points[j]
            if label1 not in label_to_idx or label2 not in label_to_idx:
                continue
            idx1, idx2 = label_to_idx[label1], label_to_idx[label2]
            _, prev = dijkstra_sparse_with_prev(graph, idx1)
            try:
                path = reconstruct_path(prev, idx1, idx2)
                path_coords = [(halton_points[k].x, halton_points[k].y) for k in path]
                path_xs, path_ys = zip(*path_coords)
                ax.plot(path_xs, path_ys, color=colors[cluster_id], linewidth=2.5, alpha=1.0)
            except:
                continue

# Draw boundary
ax.add_patch(Rectangle((0, 0), 20, 20, linewidth=2, edgecolor='black', facecolor='none'))

# ax.set_title("Clustered Dijkstra Paths on Halton Map")
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_aspect('equal')
ax.grid(True)
plt.tight_layout()
plt.show()
