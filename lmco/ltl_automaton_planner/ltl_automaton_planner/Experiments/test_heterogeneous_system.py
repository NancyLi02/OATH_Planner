#!/usr/bin/env python3
"""
完整的异构多机器人任务分配测试系统
实现多类型任务的聚类、评分、拍卖和路径规划
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, LineString
from scipy.spatial import Delaunay

from heterogenous_experiment import Robot
from MILP import ClusterTaskPlanner

class CWA:
    def __init__(self):
        pass

    def select_cluster(self, scores_list, y, robot_index, robot_types, cluster_types):

        robot_scores = scores_list[robot_index]
        valid_cluster = [(1 if score > 0 and (y_value == 0 or score < y_value) else 0) 
                        for score, y_value in zip(robot_scores, y)]
        
        if sum(valid_cluster) > 0:
            valid_indices = [i for i, valid in enumerate(valid_cluster) if valid == 1]

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
        cluster_count = len(scores_list[0])
        num_robots = len(scores_list)
        x = [[0] * cluster_count for _ in range(num_robots)]
        y = [0] * cluster_count
        assigned_clusters = []
        unassigned_robots = []

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


class HeterogeneousTaskSystem:
    
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        self.task_points_yaml = os.path.join(self.current_dir, '..', 'config', 'Task_Points_multitype.yaml')
        self.distances_csv = os.path.join(self.current_dir, 'multi_source_dijkstra_distances.csv')
        self.wall_yaml = os.path.join(self.current_dir, '..', 'config', 'wall.yaml')
        self.halton_csv = os.path.join(self.current_dir, 'all_points_in_Halton.csv')
        
        required_files = [
            (self.task_points_yaml, "Task_Points_multitype.yaml"),
            (self.distances_csv, "multi_source_dijkstra_distances.csv"),
            (self.wall_yaml, "wall.yaml"),
            (self.halton_csv, "all_points_in_Halton.csv")
        ]
        
        for file_path, file_name in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必需文件不存在: {file_name} at {file_path}")
        
        self.task_data = {}
        self.robots_data = {}
        self.distance_matrix = None
        self.task_labels = []
        self.task_positions = []
        self.task_types = []
        self.robot_positions = []
        self.robot_capabilities = []
        self.num_task_types = 0
        self.type_labels_map = {}
        self.delivery_points_data = {}
        
        self._load_task_data()
        self._process_task_types_from_yaml()
        self._load_distance_matrix()
        self._prepare_robot_data()
        
    def _load_task_data(self):
        with open(self.task_points_yaml, 'r') as f:
            self.task_data = yaml.safe_load(f)
        
        task_points = self.task_data.get('task_points', {})
        for coord_str, label in task_points.items():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                self.task_labels.append(label)
                self.task_positions.append((x, y))
        
        

        if len(self.task_labels) == 0:
            raise ValueError("没有加载到任何任务点")
        if len(self.task_positions) != len(self.task_labels):
            raise ValueError("任务标签和位置数量不匹配")
        

        delivery_points = self.task_data.get('delivery_points', {})
        for coord_str, label in delivery_points.items():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                self.delivery_points_data[label] = (x, y)
        print(f"加载了 {len(self.delivery_points_data)} 个交付点")
        
    def _process_task_types_from_yaml(self):
        """从YAML数据中动态处理任务类型信息"""
        max_type_id = 0
        type_labels_map = {}
        for key, value in self.task_data.items():
            match = re.match(r"type(\d+)_labels", key)
            if match:
                type_id = int(match.group(1))
                type_labels_map[type_id] = value
                if type_id > max_type_id:
                    max_type_id = type_id
        
        if max_type_id == 0:
            raise ValueError("在YAML文件中没有找到任务类型定义 (例如 'type1_labels').")
        
        self.num_task_types = max_type_id
        self.type_labels_map = type_labels_map
        print(f"动态识别到 {self.num_task_types} 种任务类型。")

    def _load_distance_matrix(self):
        """加载距离矩阵"""
        df = pd.read_csv(self.distances_csv)
        
        # 从距离矩阵CSV中获取所有标签
        all_labels_from_csv = list(set(df['from'].tolist() + df['to'].tolist()))
        
        # 只保留既在YAML中又在距离矩阵CSV中的标签
        valid_labels = [label for label in self.task_labels if label in all_labels_from_csv]
        
        # 打印被过滤掉的任务
        filtered_out = [label for label in self.task_labels if label not in all_labels_from_csv]
        if filtered_out:
            print(f"警告：以下任务在距离矩阵中没有数据，将被过滤: {filtered_out}")
        
        label_to_idx = {label: i for i, label in enumerate(valid_labels)}
        n = len(valid_labels)
        
        # 初始化距离矩阵
        self.distance_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(self.distance_matrix, 0.0)
        
        # 填充距离矩阵
        for _, row in df.iterrows():
            if row['from'] in label_to_idx and row['to'] in label_to_idx:
                i = label_to_idx[row['from']]
                j = label_to_idx[row['to']]
                self.distance_matrix[i, j] = row['distance']
                self.distance_matrix[j, i] = row['distance']
        
        # 更新任务标签和位置以匹配距离矩阵
        original_task_labels = self.task_labels.copy()
        original_task_positions = self.task_positions.copy()
        
        self.task_labels = valid_labels
        
        # 重新构建位置列表，只包含有效的任务
        label_to_pos = dict(zip(original_task_labels, original_task_positions))
        self.task_positions = [label_to_pos[label] for label in valid_labels]
        
        print(f"构建了 {n}x{n} 的距离矩阵，包含 {len(valid_labels)} 个任务")
        
    def _prepare_robot_data(self):
        """准备机器人数据"""
        # 解析机器人位置
        robot_positions_dict = self.task_data.get('robot_positions', {})
        robot_names = list(robot_positions_dict.keys())
        
        for name, coord_str in robot_positions_dict.items():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                self.robot_positions.append((x, y))
        
        # 解析机器人能力
        robot_capabilities_dict = self.task_data.get('robot_capabilities', {})
        for name in robot_names:
            if name in robot_capabilities_dict:
                self.robot_capabilities.append(robot_capabilities_dict[name])
            else:
                # 默认能力：能完成所有类型的任务
                self.robot_capabilities.append([1] * self.num_task_types)
        
        print(f"加载了 {len(robot_names)} 个机器人")

    def _assign_task_types(self):
        """为每个任务分配类型"""
        # 类型信息已在初始化时从YAML加载
        type_labels = self.type_labels_map
        
        self.task_types = []
        for label in self.task_labels:
            # 找到任务属于哪个类型（可能属于多个类型，取第一个）
            assigned_type = 1  # 默认类型1
            for type_id, labels in type_labels.items():
                if label in labels:
                    assigned_type = type_id
                    break
            self.task_types.append(assigned_type)
        
        print(f"分配了任务类型: {set(self.task_types)}")
        
    def perform_clustering(self, n_clusters=4):
        """执行聚类分析"""
        from sklearn.cluster import AgglomerativeClustering
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed', 
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(self.distance_matrix)
        
        print(f"完成聚类，生成 {n_clusters} 个集群")
        return cluster_labels
        
    def calculate_scores(self, cluster_labels):
        """使用新方法计算机器人-集群分数矩阵"""
        # 准备机器人对象
        robots = []
        for i, (pos, cap) in enumerate(zip(self.robot_positions, self.robot_capabilities)):
            robot = Robot(
                name=f"robot{i+1}",
                is_hybrid=True,  # 这里可以根据需要调整
                U=np.array(cap),
                cap=3  # 每个机器人的容量
            )
            robots.append(robot)
        
        # 转换任务类型为numpy数组
        task_types_array = np.array(self.task_types)
        
        # 手动计算集群中心点（Centroids）
        cluster_centers = {}
        unique_clusters = np.unique(cluster_labels)
        for k in unique_clusters:
            cluster_indices = np.where(cluster_labels == k)[0]
            if len(cluster_indices) > 0:
                # 获取集群中所有任务点的坐标
                cluster_positions = np.array([self.task_positions[i] for i in cluster_indices])
                # 计算几何中心 (Centroid)
                cluster_centers[int(k)] = np.mean(cluster_positions, axis=0)
        
        # 手动计算多类型组成
        num_types = self.num_task_types # 使用动态获取的任务类型数量
        psi = {}
        theta = 1e-3
        
        for k in unique_clusters:
            cluster_indices = np.where(cluster_labels == k)[0]
            cluster_task_types = task_types_array[cluster_indices]
            
            # 计算每种类型的数量
            counts = np.bincount(cluster_task_types, minlength=num_types + 1)[1:]  # 忽略类型0
            vec = counts.astype(float) + theta
            psi[int(k)] = vec / np.sum(vec)
        
        # 计算机器人能力的归一化
        zetas = []
        for robot in robots:
            U = robot.U
            s = np.sum(np.abs(U))
            if s <= 0:
                zeta = np.zeros_like(U, dtype=float)
            else:
                zeta = U.astype(float) / s
            zetas.append(zeta)
        zetas = np.stack(zetas, axis=0)
        
        # 计算分数矩阵
        R = len(robots)
        K = len(unique_clusters)
        S = np.zeros((R, K), dtype=float)
        
        for r_idx, robot in enumerate(robots):
            robot_pos = self.robot_positions[r_idx]
            for j, k in enumerate(unique_clusters):
                # 获取集群中心点（centroid）
                centroid_pos = cluster_centers[k]
                
                # 计算机器人到集群中心的距离
                delta = np.linalg.norm(np.array(robot_pos) - np.array(centroid_pos))
                
                # 计算能力匹配度
                gamma_rk = float(np.dot(psi[k], zetas[r_idx]))
                
                # 计算分数（距离/能力匹配度）
                if gamma_rk > 0:
                    s_rk = delta / gamma_rk
                else:
                    s_rk = np.inf
                
                S[r_idx, j] = s_rk
        
        print(f"计算了 {S.shape[0]} 个机器人对 {S.shape[1]} 个集群的分数矩阵")
        return S, cluster_centers
        
    def auction_assignment(self, scores_matrix):
        """使用拍卖算法分配集群"""
        cwa = CWA()
        
        # 准备数据
        scores_list = scores_matrix.tolist()
        robot_types = ['normal'] * len(self.robot_positions)  # 假设都是普通机器人
        cluster_types = ['normal'] * scores_matrix.shape[1]  # 假设都是普通集群
        
        # 执行拍卖
        assigned_clusters, unassigned_robots = cwa.initial_cluster_assignment(
            scores_list, robot_types, cluster_types
        )
        
        print(f"拍卖完成，分配了 {len(assigned_clusters)} 个机器人到集群")
        print(f"未分配的机器人: {unassigned_robots}")
        
        return assigned_clusters, unassigned_robots
        
    def plan_tasks_milp(self, cluster_labels, assigned_clusters):
        """为每个机器人在其分配的集群内规划任务"""
        planner = ClusterTaskPlanner(self.distances_csv)
        
        robot_plans = {}
        
        for robot_idx, cluster_idx in assigned_clusters:
            # 获取该集群中的任务
            cluster_tasks = [i for i, label in enumerate(cluster_labels) if label == cluster_idx]
            cluster_task_labels = [self.task_labels[i] for i in cluster_tasks]
            cluster_task_positions = [self.task_positions[i] for i in cluster_tasks]
            
            # 过滤机器人能完成的任务
            robot_capability = self.robot_capabilities[robot_idx]
            valid_tasks = []
            valid_labels = []
            valid_positions = []
            
            for i, task_idx in enumerate(cluster_tasks):
                task_type = self.task_types[task_idx]
                # 确保task_type在有效范围内
                if 1 <= task_type <= len(robot_capability) and robot_capability[task_type - 1] == 1:
                    valid_tasks.append(task_idx)
                    valid_labels.append(cluster_task_labels[i])
                    valid_positions.append(cluster_task_positions[i])
            
            if len(valid_tasks) == 0:
                print(f"机器人 {robot_idx+1} 在集群 {cluster_idx} 中没有可执行的任务")
                continue
                
            # 准备MILP输入
            robot_start = self.robot_positions[robot_idx]
            pickup_points = valid_positions
            pickup_labels = valid_labels
            
            # 简化：假设每个任务都有对应的交付点
            # --- START: 采用 task_assignment_visualizer.py 中更健壮的逻辑 ---

            # 从 task_data 中获取 task_to_delivery 映射
            task_to_delivery_raw = self.task_data.get('task_to_delivery', {})
            
            # 创建一个 pickup -> delivery 的反向映射以便于查找
            pickup_to_delivery_map = {}
            for delivery_label, pickup_list in task_to_delivery_raw.items():
                for pickup_label in pickup_list:
                    pickup_to_delivery_map[pickup_label] = delivery_label
            
            # 找出此集群中所有任务对应的唯一交付点
            unique_delivery_labels = set()
            for label in pickup_labels:
                if label in pickup_to_delivery_map:
                    unique_delivery_labels.add(pickup_to_delivery_map[label])
            
            # 构建交付点坐标和标签列表
            delivery_points = []
            delivery_labels = []
            for label in unique_delivery_labels:
                if label in self.delivery_points_data:
                    delivery_points.append(self.delivery_points_data[label])
                    delivery_labels.append(label)
            
            # --- END: 新逻辑 ---
            
            # 执行MILP规划
            try:
                # 使用机器人的实际容量，如果没有设置则默认为3
                robot_capacity = getattr(self, 'robot_capacities', [3] * len(self.robot_positions))[robot_idx]
                if robot_capacity <= 0:
                    robot_capacity = 3  # 默认容量
                    
                chosen_pickup_labels, chosen_delivery_labels, route_labels, total_cost = planner.plan_cluster_tasks(
                    robot_start=robot_start,
                    pickup_points=pickup_points,
                    pickup_labels=pickup_labels,
                    delivery_points=delivery_points,
                    delivery_labels=delivery_labels,
                    pickup_to_delivery=pickup_to_delivery_map,
                    robot_capacity=min(robot_capacity, len(pickup_labels))
                )
                
                robot_plans[robot_idx] = {
                    'cluster': cluster_idx,
                    'pickup_tasks': chosen_pickup_labels,
                    'delivery_tasks': chosen_delivery_labels,
                    'route': route_labels,
                    'cost': total_cost
                }
                
                print(f"机器人 {robot_idx+1} 规划完成: {len(chosen_pickup_labels)} 个任务, 成本: {total_cost:.2f}")
                
            except Exception as e:
                print(f"机器人 {robot_idx+1} MILP规划失败: {e}")
                
        return robot_plans
        
    def load_halton_graph(self):
        """加载Halton图用于路径规划"""
        # 加载Halton点
        halton_df = pd.read_csv(self.halton_csv)
        halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
        halton_points = [Point(x, y) for x, y in halton_coords]
        
        # 加载墙壁
        with open(self.wall_yaml, 'r') as f:
            wall_data = yaml.safe_load(f)
        wall_lines = [LineString(coords) for coords in wall_data.get('lines', [])]
        obstacles = [line.buffer(0.1, cap_style=3) for line in wall_lines]
        
        # 构建Delaunay三角网
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
        
        return halton_points, graph, obstacles
        
    def dijkstra_path(self, graph, start_idx, goal_idx):
        """Dijkstra最短路径算法"""
        dist = {i: math.inf for i in graph}
        prev = {}
        dist[start_idx] = 0
        heap = [(0, start_idx)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == goal_idx:
                break
                
            for v, cost in graph[u]:
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
        
    def visualize_results(self, cluster_labels, robot_plans):
        """可视化结果"""
        halton_points, graph, obstacles = self.load_halton_graph()
        
        # 创建坐标到Halton点索引的映射
        coord_to_index = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(halton_points)}
        
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.rcParams.update({'font.size': 14})
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制墙壁
        for obs in obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, color='red', alpha=0.4)
        
        # 绘制Delaunay边
        for u in graph:
            for v, _ in graph[u]:
                x1, y1 = halton_points[u].x, halton_points[u].y
                x2, y2 = halton_points[v].x, halton_points[v].y
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5, alpha=0.2)
        
        # 定义高区分度的颜色和标记
        base_type_colors = ['#FF0000', '#00FF00', '#0080FF', '#FF8000', '#8000FF', '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#42D4F4', '#FABEBE', '#469990', '#E6BEFF', '#9A6324']
        base_type_markers = ['s', 'o', '^', 'D', 'v', 'p', 'P', '*', 'X', 'd']
        
        type_colors = {i + 1: base_type_colors[i % len(base_type_colors)] for i in range(self.num_task_types)}
        type_markers = {i + 1: base_type_markers[i % len(base_type_markers)] for i in range(self.num_task_types)}
        robot_colors = ['#FF1493', '#00FFFF', '#32CD32', '#FFD700']  # 深粉、青色、酸橙绿、金色
        
        # 创建任务类型图例的句柄
        type_legend_handles = []
        
        # 绘制任务点（按类型着色和标记）
        plotted_types = set()
        for i, (pos, task_type, label) in enumerate(zip(self.task_positions, self.task_types, self.task_labels)):
            color = type_colors.get(task_type, 'gray')
            marker = type_markers.get(task_type, 's')
            
            scatter = ax.scatter(pos[0], pos[1], c=color, s=300, marker=marker, 
                               edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # 为每种类型创建图例句柄（只创建一次）
            if task_type not in plotted_types:
                type_legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                                    markerfacecolor=color, markersize=10, 
                                                    markeredgecolor='black', markeredgewidth=1.5,
                                                    label=f'Task Type {task_type}', linestyle='None'))
                plotted_types.add(task_type)
            
            # ax.annotate(label, (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points', 
            #            fontsize=12, fontweight='bold')
        
        # 绘制交付点
        for label, pos in self.delivery_points_data.items():
            ax.scatter(pos[0], pos[1], s=200, c='grey', marker='D', 
                       edgecolor='black', linewidth=1.5, alpha=0.9, zorder=5)
            ax.annotate(label, (pos[0], pos[1]), xytext=(8, 8),
                       textcoords='offset points', fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'))
        
        # 绘制机器人起始位置
        robot_legend_handles = []
        for i, pos in enumerate(self.robot_positions):
            color = robot_colors[i % len(robot_colors)]
            scatter = ax.scatter(pos[0], pos[1], c=color, s=250, marker='*', 
                               edgecolor='black', linewidth=2, alpha=0.9)
            robot_legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', 
                                                 markerfacecolor=color, markersize=15, 
                                                 markeredgecolor='black', markeredgewidth=2,
                                                 label=f'Robot {i+1}', linestyle='None'))
            
            # 添加机器人标签，字体大一些
            ax.annotate(f'Robot {i+1}', (pos[0], pos[1]), xytext=(10, 10), 
                       textcoords='offset points', fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'))
        
        # 绘制机器人路径
        for robot_idx, plan in robot_plans.items():
            if 'route' not in plan or not plan['route']:
                continue
                
            robot_color = robot_colors[robot_idx % len(robot_colors)]
            robot_pos = self.robot_positions[robot_idx]
            
            # 构建完整路径：起始位置 -> 任务点 (pickup和delivery)
            full_route_coords = [robot_pos]
            
            # 从plan['route']中获取完整路径
            if 'route' in plan:
                # 创建一个从标签到坐标的快速查找字典
                label_to_pos = {label: pos for label, pos in zip(self.task_labels, self.task_positions)}
                label_to_pos.update(self.delivery_points_data)

                for label in plan['route']:
                    if label in label_to_pos:
                        full_route_coords.append(label_to_pos[label])
            
            # 绘制路径段
            for i in range(len(full_route_coords) - 1):
                start_pos = full_route_coords[i]
                end_pos = full_route_coords[i + 1]
                
                # 找到最近的Halton点
                start_rounded = (round(start_pos[0], 4), round(start_pos[1], 4))
                end_rounded = (round(end_pos[0], 4), round(end_pos[1], 4))
                
                start_idx = coord_to_index.get(start_rounded)
                end_idx = coord_to_index.get(end_rounded)
                
                if start_idx is not None and end_idx is not None:
                    # 使用Dijkstra找路径
                    path_indices = self.dijkstra_path(graph, start_idx, end_idx)
                    if path_indices:
                        path_coords = [(halton_points[idx].x, halton_points[idx].y) for idx in path_indices]
                        path_xs, path_ys = zip(*path_coords)
                        ax.plot(path_xs, path_ys, color=robot_color, linewidth=3, alpha=0.8)
                else:
                    # 直线连接
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           color=robot_color, linewidth=3, alpha=0.8, linestyle='--')
        
        # 设置图形属性
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_aspect('equal')
        
        # 隐藏坐标轴刻度和标签
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # 为交付点创建图例句柄
        delivery_handle = plt.Line2D([0], [0], marker='D', color='w', 
                                     markerfacecolor='red', markersize=12, 
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label='Delivery Point', linestyle='None')
        type_legend_handles.append(delivery_handle)
        
        # 创建两个分离的图例
        # 任务类型图例
        type_legend = ax.legend(handles=type_legend_handles, 
                               title='Task Types', 
                               loc='upper left', 
                               bbox_to_anchor=(1.02, 1.0),
                               frameon=True, 
                               fancybox=True, 
                               shadow=True)
        type_legend.get_title().set_fontweight('bold')
        
        # 机器人图例
        robot_legend = ax.legend(handles=robot_legend_handles, 
                                title='Robots', 
                                loc='upper left', 
                                bbox_to_anchor=(1.02, 0.6),
                                frameon=True, 
                                fancybox=True, 
                                shadow=True)
        robot_legend.get_title().set_fontweight('bold')
        
        # 添加任务类型图例到图中
        ax.add_artist(type_legend)
        
        # 不显示标题
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.current_dir, 'multiple_type_gamma_task_assignment.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {output_path}")
        
        plt.show()
        
        # 打印详细结果
        print("\n=== 任务分配结果 ===")
        for robot_idx, plan in robot_plans.items():
            print(f"\n机器人 {robot_idx+1}:")
            print(f"  分配集群: {plan['cluster']}")
            print(f"  拾取任务: {plan['pickup_tasks']}")
            print(f"  交付任务: {plan['delivery_tasks']}")
            print(f"  执行路径: {plan['route']}")
            print(f"  总成本: {plan['cost']:.2f}")


def main():
    """主函数"""
    print("开始异构多机器人任务分配测试...")
    
    # 创建系统实例
    system = HeterogeneousTaskSystem()
    
    # 1. 分配任务类型
    system._assign_task_types()
    
    # 2. 执行聚类
    cluster_labels = system.perform_clustering(n_clusters=4)
    
    # 打印聚类结果分析
    print("\n=== 聚类结果分析 ===")
    unique_clusters = np.unique(cluster_labels)
    for k in unique_clusters:
        cluster_indices = np.where(cluster_labels == k)[0]
        cluster_tasks = [system.task_labels[i] for i in cluster_indices]
        cluster_types = [system.task_types[i] for i in cluster_indices]
        print(f"集群 {k}: 任务 {cluster_tasks}")
        print(f"        任务类型分布: {dict(zip(*np.unique(cluster_types, return_counts=True)))}")
    
    # 3. 计算分数矩阵
    scores_matrix, cluster_centers = system.calculate_scores(cluster_labels)
    
    # 打印分数矩阵
    print("\n=== 机器人-集群分数矩阵 ===")
    print("分数越低表示机器人越适合该集群")
    print("行: 机器人, 列: 集群")
    print(scores_matrix)
    
    # 打印机器人能力
    print("\n=== 机器人能力分析 ===")
    for i, cap in enumerate(system.robot_capabilities):
        print(f"机器人 {i+1}: 能力向量 {cap} (类型1-{system.num_task_types}: {cap})")
    
    # 4. 拍卖分配
    assigned_clusters, unassigned_robots = system.auction_assignment(scores_matrix)
    
    # 5. MILP任务规划
    robot_plans = system.plan_tasks_milp(cluster_labels, assigned_clusters)
    
    # 6. 可视化结果
    system.visualize_results(cluster_labels, robot_plans)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
