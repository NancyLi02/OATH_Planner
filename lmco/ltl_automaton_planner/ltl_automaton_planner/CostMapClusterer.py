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
    def __init__(self, points_with_label, wall_yaml_path,
                 special_labels=None, map_bounds=(0, 0, 20, 20),
                 wall_width=0.3, grid_spacing=0.1,
                 cost_boost_factor=6, num_clusters_other=4, num_clusters_special=2):

        self.points_with_label = points_with_label
        self.wall_yaml_path = wall_yaml_path
        self.special_labels = special_labels if special_labels is not None else {'db', 'eb', 'bb', 'cd', 'fd'}
        self.map_bounds = map_bounds
        self.wall_width = wall_width
        self.grid_spacing = grid_spacing
        self.cost_boost_factor = cost_boost_factor
        self.num_clusters_other = num_clusters_other
        self.num_clusters_special = num_clusters_special

        # 通过 YAML 文件加载墙体数据
        self.lines = self._load_lines_from_yaml()

        # 根据标签将任务点分为特殊任务和其他任务
        self.special_task_points = []
        self.other_task_points = []
        for pt, label in self.points_with_label.items():
            if label in self.special_labels:
                self.special_task_points.append(pt)
            else:
                self.other_task_points.append(pt)

        # 用于存储 costmap 生成的采样点和对应成本
        self.sample_points = []
        self.sample_values = []

        # 距离矩阵和聚类结果
        self.distance_matrix_other = None
        self.distance_matrix_special = None
        self.labels_other = None
        self.labels_special = None

    def _load_lines_from_yaml(self):
        """
        从 YAML 文件中加载墙体数据，并返回由 LineString 组成的列表。
        YAML 文件格式示例：
          lines:
            - [[x1, y1], [x2, y2], [x3, y3]]
            - [[x4, y4], [x5, y5]]
        """
        with open(self.wall_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return [LineString(coords) for coords in data.get('lines', [])]

    def generate_costmap(self):
        """
        根据墙体生成 costmap：
          - 先通过墙体相接构建图的连通分量，
          - 对每个连通分量内的墙体，通过缓冲区采样计算局部成本，
          - 考虑与地图边界的接触对成本的影响。
        """
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

            # endpoints 和 junctions 在当前计算中未做进一步使用，可根据需求扩展
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
                                edge_dist = min(x - self.map_bounds[0], self.map_bounds[2] - x,
                                                y - self.map_bounds[1], self.map_bounds[3] - y)
                                edge_weight = 1 - np.clip(edge_dist / 5.0, 0, 1)
                                length_factor = np.clip(group_length / 10, 0.2, 1.0)
                                effective_weight = edge_weight * length_factor
                                local_cost = local_cost * (1 - effective_weight) + max_possible_cost * effective_weight

                            self.sample_points.append((x, y))
                            self.sample_values.append(local_cost)

    def _compute_distance(self, p1, p2, boosted_sample_values, cost_tree):
        """
        计算点 p1 和 p2 之间的成本距离：
          - 沿直线 p1 到 p2 均匀采样，
          - 累加每个采样点的成本（从 KDTree 中查询对应成本索引），
          - 加上曼哈顿距离，最终返回平均成本。
        """
        line = LineString([p1, p2])
        num_samples = int(line.length / 0.1)
        num_samples = max(num_samples, 1)
        sampled_points = [line.interpolate(t, normalized=True) for t in np.linspace(0, 1, num_samples)]
        cost_along_path = sum(boosted_sample_values[cost_tree.query((pt.x, pt.y))[1]] for pt in sampled_points)
        manhattan = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        avg_cost = cost_along_path / num_samples + manhattan
        return avg_cost

    def compute_distance_matrices(self):
        """
        根据生成的 costmap 对其他任务和特殊任务分别计算距离矩阵，
        需要先生成 costmap，如果未生成则调用 generate_costmap()。
        """
        if not self.sample_points:
            self.generate_costmap()
        boosted_sample_values = [v * self.cost_boost_factor for v in self.sample_values]
        cost_tree = cKDTree(self.sample_points)
        
        # 其他任务距离矩阵
        n_other = len(self.other_task_points)
        self.distance_matrix_other = np.zeros((n_other, n_other))
        for i in range(n_other):
            for j in range(i + 1, n_other):
                d = self._compute_distance(self.other_task_points[i],
                                           self.other_task_points[j],
                                           boosted_sample_values, cost_tree)
                self.distance_matrix_other[i, j] = d
                self.distance_matrix_other[j, i] = d
        
        # 特殊任务距离矩阵
        n_special = len(self.special_task_points)
        self.distance_matrix_special = np.zeros((n_special, n_special))
        for i in range(n_special):
            for j in range(i + 1, n_special):
                d = self._compute_distance(self.special_task_points[i],
                                           self.special_task_points[j],
                                           boosted_sample_values, cost_tree)
                self.distance_matrix_special[i, j] = d
                self.distance_matrix_special[j, i] = d

    def cluster(self):
        """
        分别对其他任务和特殊任务进行聚类：
          - 使用预先计算好的距离矩阵，
          - 使用 AgglomerativeClustering（链接方式：average，距离度量：precomputed）完成聚类，
          - 并调用绘图函数展示结果。
        返回值为：
          ((其他任务聚类中心, 其他任务聚类结果点列表), (特殊任务聚类中心, 特殊任务聚类结果点列表))
        """
        if self.distance_matrix_other is None or self.distance_matrix_special is None:
            self.compute_distance_matrices()

        model_other = AgglomerativeClustering(n_clusters=self.num_clusters_other,
                                              metric='precomputed', linkage='average')
        self.labels_other = model_other.fit_predict(self.distance_matrix_other)
        model_special = AgglomerativeClustering(n_clusters=self.num_clusters_special,
                                                metric='precomputed', linkage='average')
        self.labels_special = model_special.fit_predict(self.distance_matrix_special)

        # 整理其他任务聚类结果
        cluster_centers_other = []
        cluster_points_other = []
        for cluster_label in range(self.num_clusters_other):
            indices = np.where(self.labels_other == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.other_task_points[idx] for idx in indices])
            center = cluster_coords.mean(axis=0)
            cluster_centers_other.append(tuple(center))
            cluster_points_other.append([tuple(map(float, p)) for p in cluster_coords])

        # 整理特殊任务聚类结果
        cluster_centers_special = []
        cluster_points_special = []
        for cluster_label in range(self.num_clusters_special):
            indices = np.where(self.labels_special == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.special_task_points[idx] for idx in indices])
            center = cluster_coords.mean(axis=0)
            cluster_centers_special.append(tuple(center))
            cluster_points_special.append([tuple(map(float, p)) for p in cluster_coords])

        # self.plot_clusters()
        return (cluster_centers_other, cluster_points_other), (cluster_centers_special, cluster_points_special)

    def plot_clusters(self):
        """
        绘制聚类结果：
          - 其他任务以蓝色圆点及包围圆绘制，
          - 特殊任务以红色三角形及包围圆绘制，
          - 同时展示墙体及地图边界。
        """
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_facecolor("#fcf8e8")

        # 绘制其他任务（蓝色）
        for cluster_label in range(self.num_clusters_other):
            indices = np.where(self.labels_other == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.other_task_points[idx] for idx in indices])
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                        s=100, c='blue', marker='o',
                        edgecolors='white', linewidths=1,
                        label='A tasks' if cluster_label == 0 else "")
            center = cluster_coords.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
            circle = patches.Circle(center, radius, facecolor='lightblue',
                                    edgecolor='black', alpha=0.4)
            ax.add_patch(circle)

        # 绘制特殊任务（红色）
        for cluster_label in range(self.num_clusters_special):
            indices = np.where(self.labels_special == cluster_label)[0]
            if len(indices) == 0:
                continue
            cluster_coords = np.array([self.special_task_points[idx] for idx in indices])
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                        s=100, c='red', marker='^',
                        edgecolors='white', linewidths=1,
                        label='B tasks' if cluster_label == 0 else "")
            center = cluster_coords.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
            circle = patches.Circle(center, radius, facecolor='lightcoral',
                                    edgecolor='black', alpha=0.4)
            ax.add_patch(circle)

        # 绘制墙体
        for line in self.lines:
            poly = line.buffer(self.wall_width / 2, cap_style=2, join_style=2)
            x, y = poly.exterior.xy
            plt.fill(x, y, color='gray', alpha=0.8, zorder=0)

        x_min, y_min, x_max, y_max = self.map_bounds
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 color='black', linewidth=2)
        plt.xlim(self.map_bounds[0], self.map_bounds[2])
        plt.ylim(self.map_bounds[1], self.map_bounds[3])
        plt.xticks([]); plt.yticks([])
        plt.axis('equal')
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_costmap(self):
        """
        绘制生成的 costmap 热力图。
        """
        plt.figure(figsize=(10, 8))
        sample_points_np = np.array(self.sample_points)
        sample_values_np = np.array(self.sample_values)

        sc = plt.scatter(sample_points_np[:, 0], sample_points_np[:, 1],
                         c=sample_values_np, cmap='viridis_r', s=10, marker='s')
        plt.colorbar(sc, label='Cost')
        x_min, y_min, x_max, y_max = self.map_bounds
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 color='black', linewidth=2)
        plt.xlim(self.map_bounds[0], self.map_bounds[2])
        plt.ylim(self.map_bounds[1], self.map_bounds[3])
        plt.title("Wall CostMap Heatmap", fontsize=14)
        plt.xticks([]); plt.yticks([])
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
