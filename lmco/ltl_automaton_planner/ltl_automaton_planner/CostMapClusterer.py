import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import cKDTree
import networkx as nx
from collections import defaultdict


class CostMapClusterer:
    def __init__(self, wall_yaml_path, points_with_label, map_bounds=(0, 0, 20, 20), wall_width=0.3, grid_spacing=0.1, cost_boost_factor=6.0, num_clusters=4):
        self.wall_yaml_path = wall_yaml_path
        self.points_with_label = points_with_label
        self.map_bounds = map_bounds
        self.wall_width = wall_width
        self.grid_spacing = grid_spacing
        self.cost_boost_factor = cost_boost_factor
        self.num_clusters = num_clusters

        self.task_points = list(self.points_with_label.keys())
        self.n_tasks = len(self.task_points)

        self.lines = self._load_lines_from_yaml()
        self.sample_points = []
        self.sample_values = []
        self.distance_matrix = None
        self.labels = None

    def _load_lines_from_yaml(self):
        with open(self.wall_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return [LineString(coords) for coords in data.get('lines', [])]

    def generate_costmap(self):
        G = nx.Graph()
        for i, line1 in enumerate(self.lines):
            for j, line2 in enumerate(self.lines):
                if i >= j:
                    continue
                if line1.touches(line2):
                    G.add_edge(i, j)
        G.add_nodes_from(range(len(self.lines)))
        components = list(nx.connected_components(G))

        map_box = box(*self.map_bounds)
        map_boundary = map_box.boundary

        for comp in components:
            group_lines = [self.lines[i] for i in comp]
            group_geom = unary_union(group_lines)
            group_length = sum(line.length for line in group_lines)

            C_b = (group_length / 8) ** 1.8
            C_max = C_b + 3.0
            C_min = 0.3
            r = 1.0
            touches_boundary = group_geom.touches(map_boundary)

            point_freq = defaultdict(int)
            for line in group_lines:
                coords = list(line.coords)
                point_freq[coords[0]] += 1
                point_freq[coords[-1]] += 1

            endpoints = [Point(p) for p, freq in point_freq.items() if freq == 1]
            junctions = [Point(p) for p, freq in point_freq.items() if freq > 1]

            for line in group_lines:
                poly = line.buffer(self.wall_width / 2, cap_style=2, join_style=2)
                minx, miny, maxx, maxy = poly.bounds
                x_vals = np.arange(minx, maxx, self.grid_spacing)
                y_vals = np.arange(miny, maxy, self.grid_spacing)

                for x in x_vals:
                    for y in y_vals:
                        pt = Point(x, y)
                        if poly.contains(pt):
                            d_j = min(pt.distance(j) for j in junctions) if junctions else 0
                            w_j = np.exp(-d_j / r)
                            C_pre = C_min + (C_b - C_min) * w_j

                            if len(group_lines) == 1 and len(junctions) > 0:
                                line_center = LineString(group_lines[0].coords).centroid
                                dist_to_center = pt.distance(line_center)
                                center_penalty_weight = np.exp(-dist_to_center / (r * 0.6))

                                max_penalty = 1.0
                                scale = 3.0 
                                penalty_strength = max_penalty * (1 - np.exp(-group_length / scale))

                                C_pre += penalty_strength * center_penalty_weight

                            if touches_boundary:
                                d_e = min(x - self.map_bounds[0], self.map_bounds[2] - x,
                                          y - self.map_bounds[1], self.map_bounds[3] - y)
                                w_e = 1 - np.clip(d_e / 5.0, 0, 1)
                                lambda_factor = np.clip(group_length / 10, 0.2, 1.0)
                                C_final = (1 - w_e * lambda_factor) * C_pre + C_max * (w_e * lambda_factor)
                            else:
                                C_final = C_pre

                            self.sample_points.append((x, y))
                            self.sample_values.append(C_final)

    def compute_distance_matrix(self):
        boosted_sample_values = [v * self.cost_boost_factor for v in self.sample_values]
        cost_tree = cKDTree(self.sample_points)
        n = self.n_tasks
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.task_points[i], self.task_points[j]
                line = LineString([p1, p2])
                num_samples = int(line.length / 0.1)
                sampled_points = [line.interpolate(t, normalized=True) for t in np.linspace(0, 1, num_samples)]
                cost_along_path = sum(boosted_sample_values[cost_tree.query((pt.x, pt.y))[1]] for pt in sampled_points)
                manhattan = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                avg_cost = cost_along_path / num_samples + manhattan
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = avg_cost

    def cluster(self):
        if self.distance_matrix is None:
            if not self.sample_points:
                self.generate_costmap()
            self.compute_distance_matrix()

        model = AgglomerativeClustering(n_clusters=self.num_clusters, metric='precomputed', linkage='average')
        self.labels = model.fit_predict(self.distance_matrix)

        cluster_centers = []
        cluster_points = []

        for i in range(self.num_clusters):
            indices = np.where(self.labels == i)[0]
            cluster_coords = np.array([self.task_points[idx] for idx in indices])
            center = cluster_coords.mean(axis=0)
            cluster_centers.append(tuple(center))
            cluster_points.append([tuple(map(float, p)) for p in cluster_coords])

        self.plot_clusters()
        return cluster_centers, cluster_points

    def plot_clusters(self):
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_facecolor("#fcf8e8")

        for i in range(self.num_clusters):
            indices = np.where(self.labels == i)[0]
            cluster_coords = np.array([self.task_points[idx] for idx in indices])
            color = cmap(i % 10)
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=100, c=[color], marker='o',
                        edgecolors='white', linewidths=1)
            center = cluster_coords.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
            circle = patches.Circle(center, radius, facecolor=color, edgecolor='black', alpha=0.4)
            ax.add_patch(circle)

        for line in self.lines:
            poly = line.buffer(self.wall_width / 2, cap_style=2, join_style=2)
            x, y = poly.exterior.xy
            plt.fill(x, y, color='gray', alpha=0.8, zorder=0)

        x_min, y_min, x_max, y_max = self.map_bounds
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min], color='black', linewidth=2)

        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.title(f'Cost-Aware Clustering ({self.num_clusters} Clusters)', fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_costmap(self):
        plt.figure(figsize=(10, 8))
        sample_points_np = np.array(self.sample_points)
        sample_values_np = np.array(self.sample_values)

        sc = plt.scatter(sample_points_np[:, 0], sample_points_np[:, 1], c=sample_values_np,
                         cmap='viridis_r', s=10, marker='s')
        plt.colorbar(sc, label='Cost')

        x_min, y_min, x_max, y_max = self.map_bounds
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min], color='black', linewidth=2)

        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.title("Wall CostMap Heatmap", fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
