import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import math
import heapq

class CostMapClusterer:
    def __init__(self, points_with_label, wall_yaml_path,
                 map_bounds=(0, 0, 20, 20),
                 num_clusters=4,
                 halton_points_csv=None, wall_thick=0.1,
                 precomputed_distances_csv=None):
        self.points_with_label = points_with_label
        self.wall_yaml_path = wall_yaml_path
        self.map_bounds = map_bounds
        self.num_clusters = num_clusters
        self.halton_points_csv = halton_points_csv
        self.wall_thick = wall_thick
        self.precomputed_distances_csv = precomputed_distances_csv
        
        # Multi-source clustering related variables
        self.halton_points = []
        self.halton_coords = []
        self.graph = {}
        self.obstacles = []
        self.lines = []
        self.label_to_idx = {}
        self.task_points = list(points_with_label.values())
        self.task_coords = [tuple(map(float, pt)) for pt in points_with_label.keys()]
        self.labels = None
        self.distance_matrix = None
        
        # Always load walls and Halton points for visualization
        self._load_walls_and_halton_points()
        
        # If precomputed distance matrix is provided, use it directly
        if self.precomputed_distances_csv:
            self._load_precomputed_distances()
        else:
            self._setup_multi_source_clustering()

    def _load_walls_and_halton_points(self):
        """Load walls and Halton points for visualization"""
        # Load walls
        with open(self.wall_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.lines = [LineString(coords) for coords in data.get('lines', [])]
        self.obstacles = [line.buffer(self.wall_thick, cap_style=3) for line in self.lines]
        
        # Load Halton points if available
        if self.halton_points_csv:
            halton_df = pd.read_csv(self.halton_points_csv)
            self.halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
            self.halton_points = [Point(x, y) for x, y in self.halton_coords]
            
            # Build Delaunay triangulation graph for visualization
            tri = Delaunay(self.halton_coords)
            self.graph = {i: [] for i in range(len(self.halton_points))}
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = simplex[i], simplex[(i + 1) % 3]
                    p1, p2 = self.halton_points[a], self.halton_points[b]
                    edge = LineString([p1, p2])
                    if not any(edge.intersects(obs) for obs in self.obstacles):
                        dist = p1.distance(p2)
                        self.graph[a].append((b, dist))
                        self.graph[b].append((a, dist))
            
            # Map task labels to Halton point indices
            coord_to_index = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(self.halton_points)}
            for label, coord in self.points_with_label.items():
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    x, y = coord
                    try:
                        x = float(x)
                        y = float(y)
                        rounded = (round(x, 4), round(y, 4))
                        if rounded in coord_to_index:
                            self.label_to_idx[label] = coord_to_index[rounded]
                    except (ValueError, TypeError):
                        continue

    def _load_precomputed_distances(self):
        """Load precomputed distance matrix"""
        df = pd.read_csv(self.precomputed_distances_csv)
        labels = pd.unique(df[['from', 'to']].values.ravel('K'))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        n = len(labels)
        
        dist_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(dist_matrix, 0.0)
        
        for _, row in df.iterrows():
            i = label_to_index[row['from']]
            j = label_to_index[row['to']]
            dist_matrix[i][j] = row['distance']
            dist_matrix[j][i] = row['distance']
        
        # Reorder distance matrix to match our task point order
        self.distance_matrix = self._reorder_distance_matrix(dist_matrix, label_to_index)

    def _reorder_distance_matrix(self, dist_matrix, label_to_index):
        """Reorder distance matrix to match task point order"""
        n = len(self.task_points)
        reordered_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(reordered_matrix, 0.0)
        
        for i, label1 in enumerate(self.task_points):
            for j, label2 in enumerate(self.task_points):
                if label1 in label_to_index and label2 in label_to_index:
                    idx1 = label_to_index[label1]
                    idx2 = label_to_index[label2]
                    reordered_matrix[i, j] = dist_matrix[idx1, idx2]
        
        return reordered_matrix

    def _setup_multi_source_clustering(self):
        """Setup multi-source clustering with Halton point sampling and Delaunay triangulation"""
        if self.halton_points_csv is None:
            raise ValueError("halton_points_csv must be provided for multi-source clustering")
        
        # Load Halton points
        halton_df = pd.read_csv(self.halton_points_csv)
        self.halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
        self.halton_points = [Point(x, y) for x, y in self.halton_coords]
        
        # Load walls and create obstacles
        with open(self.wall_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.lines = [LineString(coords) for coords in data.get('lines', [])]
        self.obstacles = [line.buffer(self.wall_thick, cap_style=3) for line in self.lines]
        
        # Build Delaunay triangulation graph
        tri = Delaunay(self.halton_coords)
        self.graph = {i: [] for i in range(len(self.halton_points))}
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                p1, p2 = self.halton_points[a], self.halton_points[b]
                edge = LineString([p1, p2])
                if not any(edge.intersects(obs) for obs in self.obstacles):
                    dist = p1.distance(p2)
                    self.graph[a].append((b, dist))
                    self.graph[b].append((a, dist))
        
        # Map task labels to Halton point indices
        coord_to_index = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(self.halton_points)}
        for label, coord in self.points_with_label.items():
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                x, y = coord
                try:
                    x = float(x)
                    y = float(y)
                    rounded = (round(x, 4), round(y, 4))
                    if rounded in coord_to_index:
                        self.label_to_idx[label] = coord_to_index[rounded]
                except (ValueError, TypeError):
                    continue

    def compute_distance_matrix(self):
        """Compute distance matrix - returns precomputed matrix if available"""
        # If precomputed distance matrix exists, return it directly
        if self.distance_matrix is not None:
            return self.distance_matrix
            
        # Otherwise, compute using Halton points and graph
        n = len(self.task_coords)
        dist_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(dist_matrix, 0.0)
        point_to_idx = {}
        for i, label in enumerate(self.task_points):
            if label in self.label_to_idx:
                point_to_idx[i] = self.label_to_idx[label]
        for i in range(n):
            for j in range(i + 1, n):
                if i in point_to_idx and j in point_to_idx:
                    start_idx = point_to_idx[i]
                    goal_idx = point_to_idx[j]
                    dist_dict, _ = self._dijkstra_sparse_with_prev(self.graph, start_idx)
                    if goal_idx in dist_dict and dist_dict[goal_idx] != math.inf:
                        dist_matrix[i, j] = dist_dict[goal_idx]
                        dist_matrix[j, i] = dist_dict[goal_idx]
                    else:
                        manhattan_dist = abs(self.task_coords[i][0] - self.task_coords[j][0]) + abs(self.task_coords[i][1] - self.task_coords[j][1])
                        dist_matrix[i, j] = manhattan_dist
                        dist_matrix[j, i] = manhattan_dist
                else:
                    manhattan_dist = abs(self.task_coords[i][0] - self.task_coords[j][0]) + abs(self.task_coords[i][1] - self.task_coords[j][1])
                    dist_matrix[i, j] = manhattan_dist
                    dist_matrix[j, i] = manhattan_dist
        self.distance_matrix = dist_matrix
        return dist_matrix

    def _dijkstra_sparse_with_prev(self, graph, start_idx):
        """Dijkstra algorithm with path recovery (kept for compatibility)"""
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

    def cluster(self):
        """Perform clustering on all task points"""
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        model = AgglomerativeClustering(n_clusters=self.num_clusters, metric='precomputed', linkage='average')
        self.labels = model.fit_predict(self.distance_matrix)
        cluster_centers = []
        cluster_points = []
        for cluster_label in range(self.num_clusters):
            indices = np.where(self.labels == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.task_coords[idx] for idx in indices])
            center = cluster_coords.mean(axis=0)
            cluster_centers.append(tuple(center))
            cluster_points.append([tuple(map(float, p)) for p in cluster_coords])
        return cluster_centers, cluster_points

    def plot_clusters(self):
        """Plot clustering results with walls and Halton map"""
        if self.labels is None:
            self.cluster()
        
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.set_facecolor("#fcf8e8")
        
        # Draw walls
        if self.lines:
            for line in self.lines:
                poly = line.buffer(self.wall_thick, cap_style=3)
                x, y = poly.exterior.xy
                plt.fill(x, y, color='red', alpha=0.4, zorder=1)
        
        # Draw Halton points and Delaunay edges
        if self.halton_points:
            # Draw Delaunay edges
            for u in self.graph:
                for v, _ in self.graph[u]:
                    x1, y1 = self.halton_points[u].x, self.halton_points[u].y
                    x2, y2 = self.halton_points[v].x, self.halton_points[v].y
                    ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.3, alpha=0.4, zorder=0)
            
            # Draw Halton points (small dots)
            halton_x = [p.x for p in self.halton_points]
            halton_y = [p.y for p in self.halton_points]
            plt.scatter(halton_x, halton_y, color='lightblue', s=2, alpha=0.3, zorder=0, label='Halton Points')
        
        # Draw clustered task points
        colors = plt.cm.tab10(np.arange(self.num_clusters))
        for cluster_label in range(self.num_clusters):
            indices = np.where(self.labels == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.task_coords[idx] for idx in indices])
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                        s=120, color=colors[cluster_label], marker='o',
                        edgecolors='white', linewidths=2,
                        label=f'Cluster {cluster_label+1}', zorder=3)
            
            # Draw cluster boundaries
            center = cluster_coords.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
            circle = plt.Circle(center, radius, facecolor=colors[cluster_label],
                                edgecolor='black', alpha=0.2, zorder=2)
            ax.add_patch(circle)
        
        # Draw map boundary
        x_min, y_min, x_max, y_max = self.map_bounds
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 color='black', linewidth=2, zorder=4)
        
        plt.legend(fontsize=12, loc='upper right')
        plt.xlim(self.map_bounds[0], self.map_bounds[2])
        plt.ylim(self.map_bounds[1], self.map_bounds[3])
        plt.xticks([]); plt.yticks([])
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
