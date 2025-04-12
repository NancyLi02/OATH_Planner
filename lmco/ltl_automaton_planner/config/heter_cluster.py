import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import cKDTree
import networkx as nx
from collections import defaultdict

# ---------- Parameters ----------
WALL_WIDTH = 0.3
GRID_SPACING = 0.1
MAP_BOUNDS = (0, 0, 20, 20)
cost_boost_factor = 6
# Define the number of clusters for each task type
num_clusters_other = 4
num_clusters_special = 2

# ---------- Task Points ----------
points_with_label = {
    (1.0, 4): 'bb', (6, 6): 'cb', (1, 6.5): 'db', (5.5, 9.5): 'eb', (9, 6.5): 'fb',
    (1.0, 13.5): 'bc', (9, 16.5): 'cc', (1, 16): 'dc', (6, 13): 'ec',
    (11.0, 13.5): 'bd', (11, 16.5): 'cd', (19, 16.5): 'dd', (19, 19): 'ed', (15, 19.5): 'fd',
    (11.0, 4.0): 'be', (19, 6.5): 'ce', (16, 3): 'de', (19, 1): 'ee'
}

# Group tasks based on labels (special tasks and other tasks)
special_labels = {'db', 'eb', 'bb', 'cd', 'fd'}
special_task_points = []
other_task_points = []
for pt, label in points_with_label.items():
    if label in special_labels:
        special_task_points.append(pt)
    else:
        other_task_points.append(pt)

# ---------- Walls ----------
lines = [
    LineString([(0, 3), (2, 3), (2, 4)]), LineString([(0, 5), (2, 5)]),
    LineString([(0, 7), (2, 7), (2, 6)]), LineString([(4, 9), (6, 9), (6, 10)]),
    LineString([(6, 7), (4, 7), (4, 5)]), LineString([(5, 5), (7, 5), (7, 7)]),
    LineString([(8, 5), (8, 7), (10, 7)]), LineString([(4, 2), (4, 4), (5, 4)]),
    LineString([(6, 4), (7, 4), (7, 2), (5, 2)]), LineString([(10, 3), (12, 3), (12, 4)]),
    LineString([(10, 5), (12, 5)]), LineString([(10, 7), (12, 7), (12, 6)]),
    LineString([(14, 9), (16, 9), (16, 10)]), LineString([(16, 7), (14, 7), (14, 5)]),
    LineString([(15, 5), (17, 5), (17, 7)]), LineString([(18, 5), (18, 7), (20, 7)]),
    LineString([(14, 2), (14, 4), (15, 4)]), LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),
    LineString([(0, 13), (2, 13), (2, 14)]), LineString([(0, 15), (2, 15)]),
    LineString([(0, 17), (2, 17), (2, 16)]), LineString([(4, 19), (6, 19), (6, 20)]),
    LineString([(6, 17), (4, 17), (4, 15)]), LineString([(5, 15), (7, 15), (7, 17)]),
    LineString([(8, 15), (8, 17), (10, 17)]), LineString([(4, 12), (4, 14), (5, 14)]),
    LineString([(6, 14), (7, 14), (7, 12), (5, 12)]), LineString([(10, 13), (12, 13), (12, 14)]),
    LineString([(10, 15), (12, 15)]), LineString([(10, 17), (12, 17), (12, 16)]),
    LineString([(14, 19), (16, 19), (16, 20)]), LineString([(16, 17), (14, 17), (14, 15)]),
    LineString([(15, 15), (17, 15), (17, 17)]), LineString([(18, 15), (18, 17), (20, 17)]),
    LineString([(14, 12), (14, 14), (15, 14)]), LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
    LineString([(0, 10), (6, 10)]), LineString([(10, 0), (10, 7)]),
    LineString([(14, 10), (20, 10)]), LineString([(10, 13), (10, 20)])
]

# ---------- Costmap Generation ----------
G = nx.Graph()
for i, line1 in enumerate(lines):
    for j, line2 in enumerate(lines):
        if i >= j:
            continue
        if line1.touches(line2):
            G.add_edge(i, j)
G.add_nodes_from(range(len(lines)))
components = list(nx.connected_components(G))

sample_points = []
sample_values = []
map_box = box(*MAP_BOUNDS)
map_boundary = map_box.boundary

for comp in components:
    group_lines = [lines[i] for i in comp]
    group_geom = unary_union(group_lines)
    group_length = sum(line.length for line in group_lines)
    base_cost = (group_length / 6) ** 2
    max_possible_cost = base_cost + 3.0
    min_cost = 0.3
    decay_radius = 1
    touches_boundary = group_geom.touches(map_boundary)

    point_freq = defaultdict(int)
    for line in group_lines:
        coords = list(line.coords)
        point_freq[coords[0]] += 1
        point_freq[coords[-1]] += 1

    endpoints = [Point(p) for p, freq in point_freq.items() if freq == 1]
    junctions = [Point(p) for p, freq in point_freq.items() if freq > 1]

    for line in group_lines:
        poly = line.buffer(WALL_WIDTH / 2, cap_style=2, join_style=2)
        minx, miny, maxx, maxy = poly.bounds
        x_vals = np.arange(minx, maxx, GRID_SPACING)
        y_vals = np.arange(miny, maxy, GRID_SPACING)

        for x in x_vals:
            for y in y_vals:
                pt = Point(x, y)
                if poly.contains(pt):
                    dist_to_junction = min(pt.distance(j) for j in junctions) if junctions else 0
                    decay_weight = np.exp(-dist_to_junction / decay_radius)
                    local_cost = min_cost + (base_cost - min_cost) * decay_weight

                    if len(group_lines) == 1 and len(junctions) > 0:
                        line_center = LineString(group_lines[0].coords).centroid
                        dist_to_center = pt.distance(line_center)
                        center_penalty_weight = np.exp(-dist_to_center / (decay_radius * 0.6))

                        max_penalty = 1.0
                        scale = 3.0 
                        penalty_strength = max_penalty * (1 - np.exp(-group_length / scale))

                        local_cost += penalty_strength * center_penalty_weight

                    if touches_boundary:
                        edge_dist = min(x - MAP_BOUNDS[0], MAP_BOUNDS[2] - x,
                                        y - MAP_BOUNDS[1], MAP_BOUNDS[3] - y)
                        edge_weight = 1 - np.clip(edge_dist / 5.0, 0, 1)

                        length_factor = np.clip(group_length / 10, 0.2, 1.0)
                        effective_weight = edge_weight * length_factor

                        local_cost = local_cost * (1 - effective_weight) + max_possible_cost * effective_weight

                    sample_points.append((x, y))
                    sample_values.append(local_cost)

# ---------- Build Costmap Tree ----------
boosted_sample_values = [v * cost_boost_factor for v in sample_values]
boosted_cost_tree = cKDTree(sample_points)

# Define a function to compute the cost distance between two points
def compute_distance(p1, p2):
    line = LineString([p1, p2])
    num_samples = int(line.length / 0.1)
    num_samples = max(num_samples, 1)
    sampled_points = [line.interpolate(t, normalized=True) for t in np.linspace(0, 1, num_samples)]
    cost_along_path = sum(boosted_sample_values[boosted_cost_tree.query((pt.x, pt.y))[1]] for pt in sampled_points)
    manhattan = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    avg_cost = cost_along_path / num_samples + manhattan
    return avg_cost

# ---------- Distance Matrix & Clustering for Other Tasks ----------
n_other = len(other_task_points)
distance_matrix_other = np.zeros((n_other, n_other))
for i in range(n_other):
    for j in range(i + 1, n_other):
        d = compute_distance(other_task_points[i], other_task_points[j])
        distance_matrix_other[i, j] = d
        distance_matrix_other[j, i] = d

model_other = AgglomerativeClustering(n_clusters=num_clusters_other, metric='precomputed', linkage='average')
labels_other = model_other.fit_predict(distance_matrix_other)

# ---------- Distance Matrix & Clustering for Special Tasks ----------
n_special = len(special_task_points)
distance_matrix_special = np.zeros((n_special, n_special))
for i in range(n_special):
    for j in range(i + 1, n_special):
        d = compute_distance(special_task_points[i], special_task_points[j])
        distance_matrix_special[i, j] = d
        distance_matrix_special[j, i] = d

model_special = AgglomerativeClustering(n_clusters=num_clusters_special, metric='precomputed', linkage='average')
labels_special = model_special.fit_predict(distance_matrix_special)

# ---------- Plot Clustering Result ----------
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_facecolor("#fcf8e8")

# Plot Other Tasks (blue circles)
for cluster_label in range(num_clusters_other):
    indices = np.where(labels_other == cluster_label)[0]
    if len(indices) == 0:
        continue
    cluster_coords = np.array([other_task_points[idx] for idx in indices])
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=100, c='blue', marker='o',
                edgecolors='white', linewidths=1, label='A tasks' if cluster_label == 0 else "")
    center = cluster_coords.mean(axis=0)
    radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
    circle = patches.Circle(center, radius, facecolor='lightblue', edgecolor='black', alpha=0.4)
    ax.add_patch(circle)

# Plot Special Tasks (red triangles)
for cluster_label in range(num_clusters_special):
    indices = np.where(labels_special == cluster_label)[0]
    if len(indices) == 0:
        continue
    cluster_coords = np.array([special_task_points[idx] for idx in indices])
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=100, c='red', marker='^',
                edgecolors='white', linewidths=1, label='B tasks' if cluster_label == 0 else "")
    center = cluster_coords.mean(axis=0)
    radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
    circle = patches.Circle(center, radius, facecolor='lightcoral', edgecolor='black', alpha=0.4)
    ax.add_patch(circle)

# Plot Walls
for line in lines:
    poly = line.buffer(WALL_WIDTH / 2, cap_style=2, join_style=2)
    x, y = poly.exterior.xy
    plt.fill(x, y, color='gray', alpha=0.8, zorder=0)

x_min, y_min, x_max, y_max = MAP_BOUNDS
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], color='black', linewidth=2)

plt.xlim(0, 20)
plt.ylim(0, 20)
# plt.title('Obstacles-Aware Clustering\n(A tasks: 4 clusters, B tasks: 2 clusters)', fontsize=14)
plt.xticks([]); plt.yticks([])
plt.axis('equal')
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# ---------- CostMap Heatmap ----------
# sample_points_np = np.array(sample_points)
# sample_values_np = np.array(sample_values)

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(sample_points_np[:, 0], sample_points_np[:, 1], c=sample_values_np,
#                  cmap='viridis_r', s=10, marker='s')
# plt.colorbar(sc, label='Cost')

# plt.plot([MAP_BOUNDS[0], MAP_BOUNDS[2], MAP_BOUNDS[2], MAP_BOUNDS[0], MAP_BOUNDS[0]],
#          [MAP_BOUNDS[1], MAP_BOUNDS[1], MAP_BOUNDS[3], MAP_BOUNDS[3], MAP_BOUNDS[1]],
#          color='black', linewidth=2)

# plt.xlim(0, 20)
# plt.ylim(0, 20)
# plt.title("Wall CostMap Heatmap", fontsize=14)
# plt.xticks([]); plt.yticks([])
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
