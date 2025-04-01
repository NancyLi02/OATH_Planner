# Combine full pipeline into a single code block

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import cKDTree
import networkx as nx

# ---------- Parameters ----------
WALL_WIDTH = 0.3
GRID_SPACING = 0.1
MAP_BOUNDS = (0, 0, 20, 20)
cost_boost_factor = 6.0
num_clusters = 4

# ---------- Task Points ----------
points_with_label = {
    (1, 4): '', (6, 6): '',
    (1, 6.5): 'b', (5.5, 9.5): 'c', (9, 6.5): 'd',
    (1, 13.5): 'f', (9, 16.5): 'g', (1, 16): '', (6, 13): '',
    (11, 13.5): 'i', (11, 16.5): 'j', (19, 16.5): 'k',
    (19, 19): '', (15, 19.5): '', (11, 4): 'm',
    (19, 6.5): 'n', (16, 3): '', (19, 1): ''
}
task_points = list(points_with_label.keys())
n_tasks = len(task_points)

# ---------- Walls ----------
lines = [
    LineString([(0, 3), (2, 3), (2, 4)]), 
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
    LineString([(10, 13), (10, 20)])
]

# ---------- Costmap generation ----------
G = nx.Graph()
for i, line1 in enumerate(lines):
    for j, line2 in enumerate(lines):
        if i >= j: continue
        if line1.touches(line2):
            G.add_edge(i, j)
G.add_nodes_from(range(len(lines)))
components = list(nx.connected_components(G))

sample_points = []
sample_values = []

for comp in components:
    group_lines = [lines[i] for i in comp]
    group_geom = unary_union(group_lines)
    group_center = group_geom.centroid
    group_length = sum(line.length for line in group_lines)
    base_cost = (group_length / 5) ** 1.5
    dist_to_edge = min(group_center.x - MAP_BOUNDS[0], MAP_BOUNDS[2] - group_center.x,
                       group_center.y - MAP_BOUNDS[1], MAP_BOUNDS[3] - group_center.y)
    boundary_penalty = np.exp(-dist_to_edge / 2)
    max_possible_cost = base_cost + 2.0

    for line in group_lines:
        poly = line.buffer(WALL_WIDTH / 2, cap_style=2, join_style=2)
        minx, miny, maxx, maxy = poly.bounds
        x_vals = np.arange(minx, maxx, GRID_SPACING)
        y_vals = np.arange(miny, maxy, GRID_SPACING)

        for x in x_vals:
            for y in y_vals:
                pt = Point(x, y)
                if poly.contains(pt):
                    dist_to_center = pt.distance(group_center)
                    min_cost = 0.3
                    decay_radius = np.clip(group_length / 3, 0.2, 0.5)
                    center_weight = np.exp(-dist_to_center / decay_radius)
                    local_cost = min_cost + (base_cost - min_cost) * (center_weight ** 0.5)
                    edge_dist = min(x - MAP_BOUNDS[0], MAP_BOUNDS[2] - x,
                                    y - MAP_BOUNDS[1], MAP_BOUNDS[3] - y)
                    edge_weight = 1 - np.clip(edge_dist / 2.0, 0, 1)
                    local_cost = local_cost * (1 - edge_weight) + max_possible_cost * edge_weight
                    sample_points.append((x, y))
                    sample_values.append(local_cost)

# ---------- Distance Matrix ----------
boosted_sample_values = [v * cost_boost_factor for v in sample_values]
boosted_cost_tree = cKDTree(sample_points)
distance_matrix = np.zeros((n_tasks, n_tasks))

for i in range(n_tasks):
    for j in range(i + 1, n_tasks):
        p1, p2 = task_points[i], task_points[j]
        line = LineString([p1, p2])
        num_samples = int(line.length / 0.01)
        sampled_points = [line.interpolate(t, normalized=True) for t in np.linspace(0, 1, num_samples)]
        cost_along_path = sum(boosted_sample_values[boosted_cost_tree.query((pt.x, pt.y))[1]] for pt in sampled_points)
        manhattan = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        avg_cost = cost_along_path / num_samples + manhattan
        distance_matrix[i, j] = distance_matrix[j, i] = avg_cost

# ---------- Clustering ----------
model = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
labels = model.fit_predict(distance_matrix)

# ---------- Plot ----------
cmap = plt.get_cmap("tab10")
# === Plot 1: Clustering result ===
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_facecolor("#fcf8e8")

for i in range(num_clusters):
    indices = np.where(labels == i)[0]
    cluster_coords = np.array([task_points[idx] for idx in indices])
    color = cmap(i % 10)
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=100, c=[color], marker='o',
                edgecolors='white', linewidths=1)
    center = cluster_coords.mean(axis=0)
    radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
    circle = patches.Circle(center, radius, facecolor=color, edgecolor='black', alpha=0.4)
    ax.add_patch(circle)

for line in lines:
    poly = line.buffer(WALL_WIDTH / 2, cap_style=2, join_style=2)
    x, y = poly.exterior.xy
    plt.fill(x, y, color='gray', alpha=0.8, zorder=0)

x_min, y_min, x_max, y_max = MAP_BOUNDS
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], color='black', linewidth=2)

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title(f'Cost-Aware Clustering ({num_clusters} Clusters)', fontsize=14)
plt.xticks([]); plt.yticks([])
plt.axis('equal')
plt.tight_layout()
plt.show()

# === Plot 2: CostMap Heatmap ===
plt.figure(figsize=(10, 8))
sample_points_np = np.array(sample_points)
sample_values_np = np.array(sample_values)

sc = plt.scatter(sample_points_np[:, 0], sample_points_np[:, 1], c=sample_values_np,
                 cmap='viridis_r', s=10, marker='s')
plt.colorbar(sc, label='Cost')

plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         color='black', linewidth=2)

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title("Wall CostMap Heatmap", fontsize=14)
plt.xticks([]); plt.yticks([])
plt.axis('equal')
plt.tight_layout()
plt.show()