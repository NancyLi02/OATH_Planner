#!/usr/bin/env python3
"""
任务分配可视化工具
从taskassign_cluster_node.py中提取的聚类、拍卖和任务选择逻辑
用于可视化最终的任务分配结果和机器人路径
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from shapely.geometry import LineString, Point
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering
import heapq
import math
from MILP import ClusterTaskPlanner

# -------------------- CWA拍卖算法 --------------------
class CWA:
    def __init__(self):
        pass

    def select_cluster(self, scores_list, y, robot_index, robot_types, cluster_types):
        """
        基于每个机器人的分数和集群类型执行拍卖:
          - 对于普通机器人，特殊集群的分数总是0
          - 如果普通机器人选择特殊集群，返回-1
          - 分数越低优先级越高（更好）
          - 分数为0意味着集群不能被选择
          - 机器人只能对分数>0且<当前最高出价(y值)的集群出价
          - 当y值为0时，意味着没有之前的出价，所以任何正分数都有效
        """
        robot_scores = scores_list[robot_index]
        valid_cluster = [(1 if score > 0 and (y_value == 0 or score < y_value) else 0) 
                        for score, y_value in zip(robot_scores, y)]
        
        if sum(valid_cluster) > 0:
            valid_indices = [i for i, valid in enumerate(valid_cluster) if valid == 1]
            # 如果机器人是普通类型，过滤掉特殊集群
            if robot_types[robot_index] == 'normal':
                valid_indices = [i for i in valid_indices if cluster_types[i] != 'special']
            if not valid_indices:
                return -1
            best_cluster_index = min(valid_indices, key=lambda idx: robot_scores[idx])
            best_cluster_score = robot_scores[best_cluster_index]
            y[best_cluster_index] = best_cluster_score
            return best_cluster_index
        else:
            return -1

    def conflict_resolve(self, cluster_index, assigned_clusters, x):
        for robot, assigned_cluster in assigned_clusters:
            if assigned_cluster == cluster_index:
                x[robot][cluster_index] = 0
                assigned_clusters.remove((robot, assigned_cluster))
                break
        return x, assigned_clusters

    def initial_cluster_assignment(self, scores_list, robot_types, cluster_types):
        """
        继续拍卖过程直到每个机器人都被分配了一个集群
        """
        cluster_count = len(scores_list[0])
        num_robots = len(scores_list)
        x = [[0] * cluster_count for _ in range(num_robots)]
        y = [0] * cluster_count
        assigned_clusters = []
        unassigned_robots = []

        # 继续直到每个机器人都有集群分配
        while any(sum(row) == 0 for row in x):
            for robot_index in range(num_robots):
                if sum(x[robot_index]) == 0:
                    cluster_index = self.select_cluster(scores_list, y, robot_index, robot_types, cluster_types)
                    if cluster_index != -1:
                        x, assigned_clusters = self.conflict_resolve(cluster_index, assigned_clusters, x)
                        x[robot_index][cluster_index] = 1
                        assigned_clusters.append((robot_index, cluster_index))
                    else:
                        unassigned_robots.append(robot_index)

        return assigned_clusters, unassigned_robots


# -------------------- 简化的聚类器 --------------------
class SimpleCostMapClusterer:
    def __init__(self, points_with_label, wall_yaml_path, num_clusters=4, 
                 halton_points_csv=None, wall_thick=0.1, precomputed_distances_csv=None):
        self.points_with_label = points_with_label
        self.wall_yaml_path = wall_yaml_path
        self.num_clusters = num_clusters
        self.halton_points_csv = halton_points_csv
        self.wall_thick = wall_thick
        self.precomputed_distances_csv = precomputed_distances_csv
        
        # 任务点数据
        self.task_points = list(points_with_label.values())
        self.task_coords = [tuple(map(float, pt)) for pt in points_with_label.keys()]
        self.distance_matrix = None
        
        # 加载预计算的距离
        if self.precomputed_distances_csv:
            self._load_precomputed_distances()
        
        # 加载墙体数据用于可视化
        self._load_walls()

    def _load_walls(self):
        """加载墙体数据"""
        with open(self.wall_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.lines = [LineString(coords) for coords in data.get('lines', [])]
        self.obstacles = [line.buffer(self.wall_thick, cap_style=3) for line in self.lines]

    def _load_precomputed_distances(self):
        """加载预计算的距离矩阵"""
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
        
        # 重新排序距离矩阵以匹配我们的任务点顺序
        self.distance_matrix = self._reorder_distance_matrix(dist_matrix, label_to_index)

    def _reorder_distance_matrix(self, dist_matrix, label_to_index):
        """重新排序距离矩阵以匹配任务点顺序"""
        n = len(self.task_points)
        reordered_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(reordered_matrix, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                label1 = self.task_points[i]
                label2 = self.task_points[j]
                
                dist = np.inf
                # 使用预计算的距离（如果可用）
                if label1 in label_to_index and label2 in label_to_index:
                    idx1 = label_to_index[label1]
                    idx2 = label_to_index[label2]
                    dist = dist_matrix[idx1, idx2]
                
                # 回退到曼哈顿距离
                if dist == np.inf:
                    coord1 = self.task_coords[i]
                    coord2 = self.task_coords[j]
                    dist = abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

                reordered_matrix[i, j] = dist
                reordered_matrix[j, i] = dist
            
        return reordered_matrix

    def cluster(self, task_points=None, task_coords=None):
        """执行聚类"""
        if self.distance_matrix is None:
            # 如果没有预计算的距离，使用欧几里得距离
            n = len(self.task_coords)
            self.distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        coord1 = self.task_coords[i]
                        coord2 = self.task_coords[j]
                        self.distance_matrix[i, j] = np.linalg.norm(np.array(coord1) - np.array(coord2))

        if task_points is not None:
            # 获取全局label顺序
            all_labels = self.task_points
            # 找到当前未分配点在全局label中的索引
            indices = [all_labels.index(lbl) for lbl in task_points]
            # 提取子矩阵
            dist_matrix = self.distance_matrix[np.ix_(indices, indices)]
            use_task_coords = task_coords
            use_task_points = task_points
        else:
            dist_matrix = self.distance_matrix
            use_task_coords = self.task_coords
            use_task_points = self.task_points

        n_points = len(use_task_points)
        n_clusters = min(self.num_clusters, n_points)
        if n_points == 0:
            return [], []
        if n_points == 1:
            return [use_task_coords[0]], [[use_task_coords[0]]]
        
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = model.fit_predict(dist_matrix)
        
        cluster_centers = []
        cluster_points = []
        for cluster_label in range(n_clusters):
            idxs = np.where(labels == cluster_label)[0]
            if len(idxs) == 0:
                continue
            cluster_coords = np.array([use_task_coords[i] for i in idxs])
            center = cluster_coords.mean(axis=0)
            cluster_centers.append(tuple(center))
            cluster_points.append([tuple(map(float, p)) for p in cluster_coords])
        
        return cluster_centers, cluster_points


# -------------------- 主要的任务分配可视化器 --------------------
class TaskAssignmentVisualizer:
    def __init__(self, task_points_yaml, wall_yaml, distances_csv, halton_csv):
        self.task_points_yaml = task_points_yaml
        self.wall_yaml = wall_yaml
        self.distances_csv = distances_csv
        self.halton_csv = halton_csv
        
        # 加载数据
        self._load_task_data()
        
        # 初始化组件
        self.clusterer = SimpleCostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=self.wall_yaml,
            num_clusters=4,
            halton_points_csv=self.halton_csv,
            wall_thick=0.1,
            precomputed_distances_csv=self.distances_csv
        )
        
        self.cwa_algorithm = CWA()
        
        # 存储分配结果
        self.robot_cluster_map = {}
        self.robot_task_sequence = {}
        self.robot_route_labels = {}
        
        # 加载Halton图用于Dijkstra路径计算
        self.halton_points, self.halton_graph, self.coord_to_index = self._load_halton_graph()

    def _load_task_data(self):
        """从YAML文件加载任务数据"""
        with open(self.task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # 解析机器人位置
        robot_positions_dict = yaml_data.get('robot_positions', {})
        self.robot_names = list(robot_positions_dict.keys())
        self.robot_poses = []
        for coord_str in robot_positions_dict.values():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                self.robot_poses.append((x, y))
        
        # 定义机器人类型
        special_robot_list = yaml_data.get('special_robot', [])
        self.robot_types = ['special' if name in special_robot_list else 'normal' 
                           for name in self.robot_names]
        
        # 解析任务点
        points_with_label = {}
        for k, v in yaml_data['task_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
        self.points_with_label = points_with_label
        
        # 特殊标签
        self.special_labels = set(yaml_data.get('special_labels', []))
        
        # 任务到配送的映射
        task_to_delivery = {}
        for group, label_list in yaml_data['task_to_delivery'].items():
            for label in label_list:
                task_to_delivery[label] = group
        self.task_to_delivery = task_to_delivery
        
        # 配送点
        delivery_points = {}
        for k, v in yaml_data.get('delivery_points', {}).items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                delivery_points[v] = (x, y)
        self.delivery_points = delivery_points

    def _load_halton_graph(self):
        """加载Halton点和图结构用于Dijkstra路径计算"""
        try:
            # 加载Halton点
            halton_df = pd.read_csv(self.halton_csv)
            halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
            halton_points = [Point(x, y) for x, y in halton_coords]
            
            # 创建坐标到索引的映射
            coord_to_index = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(halton_points)}
            
            # 加载墙体数据
            with open(self.wall_yaml, 'r') as f:
                wall_data = yaml.safe_load(f)
            lines = [LineString(coords) for coords in wall_data.get('lines', [])]
            obstacles = [line.buffer(0.1, cap_style=3) for line in lines]
            
            # 构建Delaunay三角剖分图
            tri = Delaunay(halton_coords)
            graph = {i: [] for i in range(len(halton_points))}
            
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = simplex[i], simplex[(i + 1) % 3]
                    p1, p2 = halton_points[a], halton_points[b]
                    edge = LineString([p1, p2])
                    
                    # 检查边是否与障碍物相交
                    if not any(edge.intersects(obs) for obs in obstacles):
                        dist = p1.distance(p2)
                        graph[a].append((b, dist))
                        graph[b].append((a, dist))
            
            print(f"Loaded Halton graph with {len(halton_points)} points and {sum(len(neighbors) for neighbors in graph.values())//2} edges")
            return halton_points, graph, coord_to_index
            
        except Exception as e:
            print(f"Warning: Could not load Halton graph: {e}")
            return [], {}, {}

    def dijkstra_path(self, start_idx, goal_idx):
        """计算两点间的Dijkstra最短路径"""
        if not self.halton_graph or start_idx not in self.halton_graph or goal_idx not in self.halton_graph:
            return []
            
        dist = {i: math.inf for i in self.halton_graph}
        prev = {}
        dist[start_idx] = 0
        heap = [(0, start_idx)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == goal_idx:
                break
                
            for v, cost in self.halton_graph[u]:
                alt = d + cost
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
        
        # 重构路径
        if goal_idx not in prev and goal_idx != start_idx:
            return []
            
        path = [goal_idx]
        while path[-1] != start_idx:
            if path[-1] not in prev:
                return []
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def run_clustering(self):
        """执行聚类"""
        print("执行任务聚类...")
        self.cluster_centers, self.cluster_points = self.clusterer.cluster()
        
        # 计算集群的γ值和类型
        self.cluster_gamma = []
        self.cluster_types = []
        for i, (center, points) in enumerate(zip(self.cluster_centers, self.cluster_points)):
            normal_count = 0
            special_count = 0
            for coord in points:
                for point_coord, label in self.points_with_label.items():
                    if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                        if label in self.special_labels:
                            special_count += 1
                        else:
                            normal_count += 1
                        break
            
            gamma_param = (special_count + 0.1) / (normal_count + 0.1)
            self.cluster_gamma.append(gamma_param)
            
            # 确定集群类型
            if normal_count > 0 and special_count > 0:
                cluster_type = 'hybrid'
            elif special_count > 0:
                cluster_type = 'special'
            else:
                cluster_type = 'normal'
            
            self.cluster_types.append(cluster_type)
        
        print(f"聚类完成: {len(self.cluster_centers)} 个集群")
        for i, (center, ctype) in enumerate(zip(self.cluster_centers, self.cluster_types)):
            print(f"  集群 {i}: 中心 {center}, 类型 {ctype}, γ={self.cluster_gamma[i]:.3f}")

    def run_auction(self):
        """执行拍卖算法进行集群分配"""
        print("\n执行CWA拍卖算法...")
        
        # 计算每个机器人对每个集群的分数
        scores = []
        for i, robot_pose in enumerate(self.robot_poses):
            robot_name = self.robot_names[i]
            robot_scores = []
            
            for j, (center, t_type) in enumerate(zip(self.cluster_centers, self.cluster_types)):
                # 普通机器人不能处理特殊集群
                if self.robot_types[i] == 'normal' and t_type == 'special':
                    robot_scores.append(0.0)
                    continue
                
                dist = np.linalg.norm(np.array(robot_pose) - np.array(center))
                base_score = dist
                
                # 应用γ调整
                gamma_param = self.cluster_gamma[j]
                if self.robot_types[i] == 'normal':
                    score = base_score * gamma_param
                else:  # special robot
                    if gamma_param > 0:
                        score = base_score / gamma_param
                    else:
                        score = base_score
                
                robot_scores.append(score)
            scores.append(robot_scores)
        
        # 执行拍卖
        assigned, unassigned = self.cwa_algorithm.initial_cluster_assignment(
            scores, self.robot_types, self.cluster_types)
        
        # 映射结果
        self.robot_cluster_map = {name: [] for name in self.robot_names}
        for robot_index, cluster_index in assigned:
            robot_name = self.robot_names[robot_index]
            if cluster_index != -1:
                self.robot_cluster_map[robot_name] = self.cluster_points[cluster_index]
                print(f"  {robot_name} -> 集群 {cluster_index}")
            else:
                print(f"  {robot_name} -> 无分配")

    def run_task_planning(self):
        """为每个机器人执行集群内任务规划"""
        print("\n执行集群内任务规划...")
        
        planner = ClusterTaskPlanner(self.distances_csv)
        
        self.robot_task_sequence = {}
        self.robot_route_labels = {}
        
        for robot_name in self.robot_names:
            task_points = self.robot_cluster_map.get(robot_name, [])
            if not task_points:
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue
            
            # 获取机器人信息
            robot_index = self.robot_names.index(robot_name)
            robot_start = self.robot_poses[robot_index]
            
            # 获取任务标签
            task_labels = []
            for coord in task_points:
                for point_coord, label in self.points_with_label.items():
                    if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                        task_labels.append(label)
                        break
            
            # 分离pickup和delivery点
            pickup_points = []
            pickup_labels = []
            delivery_points = []
            delivery_labels_unique = []
            
            # 获取唯一的delivery标签
            unique_deliveries = set()
            for label in task_labels:
                if label in self.task_to_delivery:
                    delivery_label = self.task_to_delivery[label]
                    unique_deliveries.add(delivery_label)
            
            # 构建pickup和delivery数据
            for i, (point, task_label) in enumerate(zip(task_points, task_labels)):
                pickup_points.append(point)
                pickup_labels.append(task_label)
            
            for delivery_label in unique_deliveries:
                if delivery_label in self.delivery_points:
                    delivery_points.append(self.delivery_points[delivery_label])
                    delivery_labels_unique.append(delivery_label)
            
            # 使用真正的MILP进行任务规划
            try:
                robot_capacity = 3  # 默认容量

                if not pickup_labels:
                    print(f"  {robot_name}: 在其集群中没有可执行的任务")
                    self.robot_route_labels[robot_name] = []
                    continue

                chosen_pickups, chosen_deliveries, route_labels, total_cost = \
                    planner.plan_cluster_tasks(
                        robot_start=robot_start,
                        pickup_points=pickup_points,
                        pickup_labels=pickup_labels,
                        delivery_points=delivery_points,
                        delivery_labels=delivery_labels_unique,
                        pickup_to_delivery=self.task_to_delivery,
                        robot_capacity=min(robot_capacity, len(pickup_labels))
                    )
                
                self.robot_route_labels[robot_name] = route_labels
                print(f"  {robot_name}: 规划完成: {len(chosen_pickups)} 个任务, 成本: {total_cost:.2f}")

            except Exception as e:
                print(f"  {robot_name}: MILP任务规划失败: {e}")
                self.robot_route_labels[robot_name] = []

    def get_dijkstra_path(self, start_label, end_label):
        """获取两个标签之间的真正Dijkstra路径"""
        start_coord = None
        end_coord = None
        
        # 查找起始坐标
        for coord, label in self.points_with_label.items():
            if label == start_label:
                start_coord = coord
            elif label == end_label:
                end_coord = coord
        
        # 检查delivery点
        if start_coord is None and start_label in self.delivery_points:
            start_coord = self.delivery_points[start_label]
        if end_coord is None and end_label in self.delivery_points:
            end_coord = self.delivery_points[end_label]
        
        # 如果找不到坐标，返回空路径
        if start_coord is None or end_coord is None:
            return []
        
        # 找到最近的Halton点索引
        start_rounded = (round(start_coord[0], 4), round(start_coord[1], 4))
        end_rounded = (round(end_coord[0], 4), round(end_coord[1], 4))
        
        start_idx = self.coord_to_index.get(start_rounded)
        end_idx = self.coord_to_index.get(end_rounded)
        
        if start_idx is not None and end_idx is not None:
            # 使用Dijkstra计算路径
            path_indices = self.dijkstra_path(start_idx, end_idx)
            if path_indices:
                # 转换为坐标
                path_coords = [(self.halton_points[idx].x, self.halton_points[idx].y) for idx in path_indices]
                return path_coords
        
        # 如果Dijkstra失败，返回直线路径
        return [start_coord, end_coord]

    def get_dijkstra_path_from_coords(self, start_coord, end_coord):
        """从坐标获取Dijkstra路径"""
        # 找到最近的Halton点索引
        start_rounded = (round(start_coord[0], 4), round(start_coord[1], 4))
        end_rounded = (round(end_coord[0], 4), round(end_coord[1], 4))
        
        start_idx = self.coord_to_index.get(start_rounded)
        end_idx = self.coord_to_index.get(end_rounded)
        
        if start_idx is not None and end_idx is not None:
            # 使用Dijkstra计算路径
            path_indices = self.dijkstra_path(start_idx, end_idx)
            if path_indices:
                # 转换为坐标
                path_coords = [(self.halton_points[idx].x, self.halton_points[idx].y) for idx in path_indices]
                return path_coords
        
        # 如果Dijkstra失败，返回直线路径
        return [start_coord, end_coord]

    def visualize_assignment(self):
        """可视化任务分配结果"""
        print("\nGenerating visualization...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor("#fcf8e8")
        
        # 绘制墙体
        if hasattr(self.clusterer, 'lines') and self.clusterer.lines:
            for line in self.clusterer.lines:
                poly = line.buffer(self.clusterer.wall_thick, cap_style=3)
                x, y = poly.exterior.xy
                ax.fill(x, y, color='red', alpha=0.4, zorder=1)
        
        # 定义颜色和标记
        robot_colors = ['blue', 'green', 'orange', 'purple']
        type_colors = {'normal': '#0080FF', 'special': '#FF0000'}  # 蓝色和红色
        type_markers = {'normal': 'o', 'special': 's'}  # 圆形和方形

        # 绘制任务点（根据类型区分）
        for coord, label in self.points_with_label.items():
            task_type = 'special' if label in self.special_labels else 'normal'
            ax.scatter(coord[0], coord[1], s=120, c=type_colors[task_type], marker=type_markers[task_type],
                       edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)

        # 绘制机器人和它们的路径
        for i, robot_name in enumerate(self.robot_names):
            robot_pos = self.robot_poses[i]
            robot_color = robot_colors[i % len(robot_colors)]
            
            # 绘制机器人起始位置
            ax.scatter(robot_pos[0], robot_pos[1], s=200, color=robot_color,
                      marker='s', edgecolors='black', linewidths=2,
                      label=f'{robot_name} ({self.robot_types[i]})', zorder=5)
            
            # 绘制机器人的任务路径
            route_labels = self.robot_route_labels.get(robot_name, [])
            if route_labels:
                # 构建路径坐标
                path_coords = [robot_pos]
                
                for label in route_labels:
                    # 查找标签对应的坐标
                    coord = None
                    for point_coord, point_label in self.points_with_label.items():
                        if point_label == label:
                            coord = point_coord
                            break
                    
                    # 如果是delivery点，查找delivery坐标
                    if coord is None and label in self.delivery_points:
                        coord = self.delivery_points[label]
                    
                    if coord:
                        path_coords.append(coord)
                
                # 绘制Dijkstra路径段
                for j in range(len(path_coords) - 1):
                    start_pos = path_coords[j]
                    end_pos = path_coords[j + 1]
                    
                    # 使用Dijkstra计算真正的路径
                    dijkstra_path = self.get_dijkstra_path_from_coords(start_pos, end_pos)
                    
                    if len(dijkstra_path) > 1:
                        # 绘制Dijkstra路径
                        path_xs, path_ys = zip(*dijkstra_path)
                        ax.plot(path_xs, path_ys, color=robot_color, linewidth=3, alpha=0.8, zorder=4)
                        
                        # 在路径的最后一段添加箭头
                        last_start = dijkstra_path[-2]
                        last_end = dijkstra_path[-1]
                        ax.annotate('', xy=last_end, xytext=last_start,
                                  arrowprops=dict(arrowstyle='->', color=robot_color,
                                                lw=2, alpha=0.8), zorder=4)
                    else:
                        # 回退到直线
                        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                               color=robot_color, linewidth=3, alpha=0.8, zorder=4, linestyle='--')
                        ax.annotate('', xy=end_pos, xytext=start_pos,
                                  arrowprops=dict(arrowstyle='->', color=robot_color,
                                                lw=2, alpha=0.8), zorder=4)
        
        # 绘制delivery点
        for delivery_label, coord in self.delivery_points.items():
            ax.scatter(coord[0], coord[1], s=150, color='red', marker='D',
                      edgecolors='black', linewidths=2, alpha=0.8, zorder=5)
            ax.annotate(delivery_label, (coord[0], coord[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        # 添加任务点标签
        for coord, label in self.points_with_label.items():
            ax.annotate(label, coord, xytext=(3, 3), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        # 设置图表属性
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('two_type_gamma_task_assignment', fontsize=16, fontweight='bold')
        
        # --- 创建新的图例 ---
        legend_handles = []
        # 机器人图例
        for i, robot_name in enumerate(self.robot_names):
            legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                                             markerfacecolor=robot_colors[i % len(robot_colors)], 
                                             markersize=10, label=f'{robot_name} ({self.robot_types[i]})'))
        # 任务类型图例
        legend_handles.append(plt.Line2D([0], [0], marker=type_markers['normal'], color='w', 
                                         markerfacecolor=type_colors['normal'], markersize=10, 
                                         label='Normal Task', linestyle='None'))
        legend_handles.append(plt.Line2D([0], [0], marker=type_markers['special'], color='w', 
                                         markerfacecolor=type_colors['special'], markersize=10, 
                                         label='Special Task', linestyle='None'))
        # Delivery点图例
        legend_handles.append(plt.Line2D([0], [0], marker='D', color='w',
                                            markerfacecolor='red', markersize=10,
                                            label='Delivery Point', linestyle='None'))
        
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # 保存图片
        output_file = '/home/nanli/ros2_ws/src/lmco/ltl_automaton_planner/ltl_automaton_planner/two_type_gamma_task_assignment.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Overview visualization saved to: {output_file}")
        
        plt.show()

    def print_assignment_summary(self):
        """打印分配摘要"""
        print("\n" + "="*60)
        print("任务分配摘要")
        print("="*60)
        
        for robot_name in self.robot_names:
            robot_index = self.robot_names.index(robot_name)
            robot_type = self.robot_types[robot_index]
            robot_pos = self.robot_poses[robot_index]
            route_labels = self.robot_route_labels.get(robot_name, [])
            
            print(f"\n{robot_name} ({robot_type}机器人):")
            print(f"  起始位置: {robot_pos}")
            print(f"  分配的任务路径: {route_labels}")
            
            # 分离pickup和delivery任务
            pickup_tasks = [label for label in route_labels 
                           if label in [lbl for lbl in self.points_with_label.values()]]
            delivery_tasks = [label for label in route_labels 
                             if label in self.delivery_points.keys()]
            
            print(f"  Pickup任务: {pickup_tasks}")
            print(f"  Delivery任务: {delivery_tasks}")

    def run_complete_assignment(self):
        """运行完整的任务分配流程"""
        print("Starting task assignment process...")
        
        # 步骤1: 聚类
        self.run_clustering()
        
        # 步骤2: 拍卖
        self.run_auction()
        
        # 步骤3: 任务规划
        self.run_task_planning()
        
        # 步骤4: 打印摘要
        self.print_assignment_summary()
        
        # 步骤5: 可视化
        self.visualize_assignment()


def main():
    """主函数"""
    # 文件路径
    base_path = "/home/nanli/ros2_ws/src/lmco/ltl_automaton_planner"
    task_points_yaml = os.path.join(base_path, "config", "Task_Points.yaml")
    wall_yaml = os.path.join(base_path, "config", "wall.yaml")
    distances_csv = os.path.join(base_path, "ltl_automaton_planner", "multi_source_dijkstra_distances.csv")
    halton_csv = os.path.join(base_path, "ltl_automaton_planner", "all_points_in_Halton.csv")
    
    # 检查文件是否存在
    for file_path in [task_points_yaml, wall_yaml, distances_csv, halton_csv]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return
    
    # 创建可视化器并运行
    visualizer = TaskAssignmentVisualizer(
        task_points_yaml=task_points_yaml,
        wall_yaml=wall_yaml,
        distances_csv=distances_csv,
        halton_csv=halton_csv
    )
    
    visualizer.run_complete_assignment()


if __name__ == "__main__":
    main()
