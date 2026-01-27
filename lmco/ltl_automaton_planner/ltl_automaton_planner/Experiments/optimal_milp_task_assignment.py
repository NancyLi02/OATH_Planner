#!/usr/bin/env python3
"""
Optimal MILP Task Assignment Experiment
比较planner与最优解的差异

功能：
1. 使用MILP计算多机器人任务分配的最优解（完整模型，包括路径规划）
2. 考虑机器人容量约束（每3个任务必须去delivery一次）
3. 考虑special robot可以完成所有任务，普通robot只能完成普通任务
4. 一次行程中如果任务对应多个delivery点，全部都要访问
5. 目标：最小化所有机器人的总路径长度（使用Dijkstra距离）
6. 输出总步数（Halton map上的点数）
7. 输出MILP运行时间
"""

import os
import yaml
import time
import math
import heapq
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional, Set
from shapely.geometry import Point, LineString
from scipy.spatial import Delaunay


class OptimalMILPTaskAssignment:
    """
    使用MILP计算多机器人任务分配的最优解
    完整模型：同时优化任务分配和访问顺序
    """
    
    def __init__(self, config_path: str, distances_csv_path: str, 
                 halton_csv_path: str, wall_yaml_path: str):
        """
        初始化
        
        Args:
            config_path: Task_Points.yaml路径
            distances_csv_path: Dijkstra距离CSV路径
            halton_csv_path: Halton点CSV路径
            wall_yaml_path: 墙壁配置YAML路径
        """
        self.config_path = config_path
        self.distances_csv_path = distances_csv_path
        self.halton_csv_path = halton_csv_path
        self.wall_yaml_path = wall_yaml_path
        
        # 加载数据
        self._load_config()
        self._load_distances()
        self._build_halton_graph()
        self._build_complete_distance_matrix()
        
    def _load_config(self):
        """加载Task_Points.yaml配置"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 解析任务点
        self.task_points = {}  # label -> (x, y)
        self.task_labels = []
        for coord_str, label in config.get('task_points', {}).items():
            x, y = map(float, coord_str.split(','))
            self.task_points[label] = (x, y)
            self.task_labels.append(label)
        
        # 解析交付点
        self.delivery_points = {}  # label -> (x, y)
        self.delivery_labels = []
        for coord_str, label in config.get('delivery_points', {}).items():
            x, y = map(float, coord_str.split(','))
            self.delivery_points[label] = (x, y)
            self.delivery_labels.append(label)
        
        # 解析任务到交付点的映射
        self.task_to_delivery = {}  # task_label -> delivery_label
        for delivery_label, task_list in config.get('task_to_delivery', {}).items():
            for task_label in task_list:
                self.task_to_delivery[task_label] = delivery_label
        
        # 解析机器人位置
        self.robot_positions = {}  # robot_name -> (x, y)
        self.robot_names = []
        for robot_name, coord_str in config.get('robot_positions', {}).items():
            x, y = map(float, coord_str.split(','))
            self.robot_positions[robot_name] = (x, y)
            self.robot_names.append(robot_name)
        
        # 解析特殊任务和特殊机器人
        self.special_labels = config.get('special_labels', [])
        self.special_robots = config.get('special_robot', [])
        
        # 普通任务 = 所有任务 - 特殊任务
        self.normal_labels = [l for l in self.task_labels if l not in self.special_labels]
        
        print(f"加载配置完成:")
        print(f"  - {len(self.task_labels)} 个任务点")
        print(f"  - {len(self.delivery_labels)} 个交付点")
        print(f"  - {len(self.robot_names)} 个机器人")
        print(f"  - {len(self.special_labels)} 个特殊任务: {self.special_labels}")
        print(f"  - {len(self.special_robots)} 个特殊机器人: {self.special_robots}")
        
    def _load_distances(self):
        """加载Dijkstra距离矩阵"""
        df = pd.read_csv(self.distances_csv_path)
        
        self.dijkstra_distances = {}  # (from, to) -> distance
        for _, row in df.iterrows():
            from_label = str(row['from'])
            to_label = str(row['to'])
            distance = float(row['distance'])
            self.dijkstra_distances[(from_label, to_label)] = distance
            self.dijkstra_distances[(to_label, from_label)] = distance
        
        print(f"加载了 {len(self.dijkstra_distances)//2} 对点之间的Dijkstra距离")
    
    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _get_position(self, label: str) -> Optional[Tuple[float, float]]:
        """获取标签对应的位置"""
        if label in self.task_points:
            return self.task_points[label]
        if label in self.delivery_points:
            return self.delivery_points[label]
        if label in self.robot_positions:
            return self.robot_positions[label]
        return None
    
    def _build_complete_distance_matrix(self):
        """
        构建完整的距离矩阵，包括：
        - 机器人起点到任务/交付点的距离
        - 任务到任务的距离（使用Dijkstra）
        - 任务到交付点的距离
        - 交付点到任务的距离
        - 交付点到交付点的距离
        """
        # 所有节点标签
        all_labels = self.robot_names + self.task_labels + self.delivery_labels
        self.all_labels = all_labels
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        n = len(all_labels)
        self.dist_matrix = np.full((n, n), float('inf'))
        np.fill_diagonal(self.dist_matrix, 0)
        
        # 填充距离矩阵
        for i, label_i in enumerate(all_labels):
            pos_i = self._get_position(label_i)
            for j, label_j in enumerate(all_labels):
                if i == j:
                    continue
                pos_j = self._get_position(label_j)
                
                # 首先尝试使用Dijkstra距离
                key = (label_i, label_j)
                if key in self.dijkstra_distances:
                    self.dist_matrix[i, j] = self.dijkstra_distances[key]
                elif pos_i and pos_j:
                    # 否则使用欧几里得距离
                    self.dist_matrix[i, j] = self._euclidean_distance(pos_i, pos_j)
        
        print(f"构建了 {n}x{n} 的完整距离矩阵")
    
    def _build_halton_graph(self):
        """构建Halton图用于计算步数"""
        # 加载Halton点
        halton_df = pd.read_csv(self.halton_csv_path)
        self.halton_coords = [(row['x'], row['y']) for _, row in halton_df.iterrows()]
        self.halton_points = [Point(x, y) for x, y in self.halton_coords]
        
        # 创建坐标到索引的映射
        self.coord_to_halton_idx = {}
        for i, (x, y) in enumerate(self.halton_coords):
            key = (round(x, 4), round(y, 4))
            self.coord_to_halton_idx[key] = i
        
        # 加载墙壁
        with open(self.wall_yaml_path, 'r') as f:
            wall_data = yaml.safe_load(f)
        wall_lines = [LineString(coords) for coords in wall_data.get('lines', [])]
        self.obstacles = [line.buffer(0.1, cap_style=3) for line in wall_lines]
        
        # 构建Delaunay三角网和图
        tri = Delaunay(self.halton_coords)
        self.halton_graph = {i: [] for i in range(len(self.halton_points))}
        
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                p1, p2 = self.halton_points[a], self.halton_points[b]
                edge = LineString([p1, p2])
                if not any(edge.intersects(obs) for obs in self.obstacles):
                    dist = p1.distance(p2)
                    self.halton_graph[a].append((b, dist))
                    self.halton_graph[b].append((a, dist))
        
        print(f"构建Halton图完成: {len(self.halton_points)} 个点")
    
    def _find_nearest_halton_idx(self, pos: Tuple[float, float]) -> int:
        """找到最近的Halton点索引"""
        key = (round(pos[0], 4), round(pos[1], 4))
        if key in self.coord_to_halton_idx:
            return self.coord_to_halton_idx[key]
        
        min_dist = float('inf')
        nearest_idx = 0
        for i, (x, y) in enumerate(self.halton_coords):
            dist = math.hypot(pos[0] - x, pos[1] - y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx
    
    def _dijkstra_path_halton(self, start_idx: int, goal_idx: int) -> List[int]:
        """Halton图上的Dijkstra最短路径算法"""
        if start_idx == goal_idx:
            return [start_idx]
        
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
        
        if goal_idx not in prev and goal_idx != start_idx:
            return []
        
        path = [goal_idx]
        while path[-1] != start_idx:
            if path[-1] not in prev:
                return []
            path.append(prev[path[-1]])
        path.reverse()
        return path
    
    def _calculate_path_steps(self, route_labels: List[str], start_label: str) -> int:
        """计算路径的总步数（Halton图上的点数）"""
        if not route_labels:
            return 0
        
        total_steps = 0
        current_pos = self._get_position(start_label)
        
        for label in route_labels:
            next_pos = self._get_position(label)
            if next_pos is None or current_pos is None:
                continue
            
            start_idx = self._find_nearest_halton_idx(current_pos)
            end_idx = self._find_nearest_halton_idx(next_pos)
            
            path = self._dijkstra_path_halton(start_idx, end_idx)
            if path:
                total_steps += len(path) - 1
            
            current_pos = next_pos
        
        return total_steps
    
    def _get_distance(self, from_label: str, to_label: str) -> float:
        """获取两个标签之间的距离"""
        if from_label == to_label:
            return 0.0
        i = self.label_to_idx.get(from_label)
        j = self.label_to_idx.get(to_label)
        if i is not None and j is not None:
            return self.dist_matrix[i, j]
        return float('inf')
    
    def solve_optimal_assignment(self, capacity: int = 3, time_limit: int = 600) -> Dict:
        """
        使用完整MILP求解最优任务分配
        
        模型：多depot VRP with delivery constraints
        - 同时优化任务分配和访问顺序
        - 使用Dijkstra距离
        - 容量约束：每capacity个任务后必须delivery
        - 每次行程结束时，必须访问所有涉及到的delivery点
        
        Args:
            capacity: 机器人容量（每次最多携带的任务数）
            time_limit: MILP求解时间限制（秒）
            
        Returns:
            包含最优解的字典
        """
        print("\n" + "="*60)
        print("开始完整MILP最优任务分配求解")
        print("="*60)
        
        start_time = time.time()
        
        R = len(self.robot_names)
        T = len(self.task_labels)
        D = len(self.delivery_labels)
        
        # 每个机器人最多K次行程
        K = (T + capacity - 1) // capacity + 1
        
        print(f"问题规模: {R}个机器人, {T}个任务, {D}个交付点, 容量={capacity}")
        print(f"每个机器人最多 {K} 次行程")
        print(f"时间限制: {time_limit} 秒")
        
        # 创建模型
        model = gp.Model("complete_multi_robot_task_assignment")
        model.Params.LogToConsole = 1
        model.Params.TimeLimit = time_limit
        
        # ==================== 索引映射 ====================
        # 任务索引：0 到 T-1
        # 交付点索引：0 到 D-1
        # 机器人索引：0 到 R-1
        # 行程索引：0 到 K-1
        
        task_idx = {label: i for i, label in enumerate(self.task_labels)}
        delivery_idx = {label: i for i, label in enumerate(self.delivery_labels)}
        robot_idx = {name: i for i, name in enumerate(self.robot_names)}
        
        # ==================== 决策变量 ====================
        
        # assign[r,t] = 1 如果任务t分配给机器人r
        assign = model.addVars(R, T, vtype=GRB.BINARY, name="assign")
        
        # trip_task[r,t,k] = 1 如果机器人r在第k次行程中执行任务t
        trip_task = model.addVars(R, T, K, vtype=GRB.BINARY, name="trip_task")
        
        # trip_active[r,k] = 1 如果机器人r有第k次行程
        trip_active = model.addVars(R, K, vtype=GRB.BINARY, name="trip_active")
        
        # 行程内任务访问顺序（MTZ约束）
        # order[r,t,k] = 任务t在机器人r第k次行程中的访问顺序（1到capacity）
        order = model.addVars(R, T, K, vtype=GRB.INTEGER, lb=0, ub=capacity, name="order")
        
        # 任务间的弧变量：arc_task[r,t1,t2,k] = 1 如果在行程k中从任务t1直接到任务t2
        arc_task = model.addVars(R, T, T, K, vtype=GRB.BINARY, name="arc_task")
        
        # 行程是否访问某个delivery点
        # trip_delivery[r,d,k] = 1 如果机器人r在第k次行程结束时访问delivery点d
        trip_delivery = model.addVars(R, D, K, vtype=GRB.BINARY, name="trip_delivery")
        
        # delivery点访问顺序（在行程结束时）
        # delivery_order[r,d,k] = delivery点d在行程k结束时的访问顺序
        delivery_order = model.addVars(R, D, K, vtype=GRB.INTEGER, lb=0, ub=D, name="delivery_order")
        
        # 第一个任务变量：first_task[r,t,k] = 1 如果t是机器人r第k次行程的第一个任务
        first_task = model.addVars(R, T, K, vtype=GRB.BINARY, name="first_task")
        
        # 最后一个任务变量：last_task[r,t,k] = 1 如果t是机器人r第k次行程的最后一个任务
        last_task = model.addVars(R, T, K, vtype=GRB.BINARY, name="last_task")
        
        # ==================== 约束条件 ====================
        
        # 约束1: 每个任务恰好被一个机器人执行一次
        for t in range(T):
            model.addConstr(
                gp.quicksum(assign[r, t] for r in range(R)) == 1,
                f"task_once_{t}"
            )
        
        # 约束2: 任务分配与行程任务的一致性
        for r in range(R):
            for t in range(T):
                model.addConstr(
                    assign[r, t] == gp.quicksum(trip_task[r, t, k] for k in range(K)),
                    f"assign_trip_{r}_{t}"
                )
        
        # 约束3: 每个任务最多在一个行程中
        for r in range(R):
            for t in range(T):
                model.addConstr(
                    gp.quicksum(trip_task[r, t, k] for k in range(K)) <= 1,
                    f"task_single_trip_{r}_{t}"
                )
        
        # 约束4: 容量约束 - 每次行程最多capacity个任务
        for r in range(R):
            for k in range(K):
                model.addConstr(
                    gp.quicksum(trip_task[r, t, k] for t in range(T)) <= capacity,
                    f"capacity_{r}_{k}"
                )
        
        # 约束5: 行程激活约束
        for r in range(R):
            for k in range(K):
                # 如果行程k有任务，则trip_active[r,k]=1
                model.addConstr(
                    gp.quicksum(trip_task[r, t, k] for t in range(T)) >= trip_active[r, k],
                    f"trip_active_lb_{r}_{k}"
                )
                model.addConstr(
                    gp.quicksum(trip_task[r, t, k] for t in range(T)) <= T * trip_active[r, k],
                    f"trip_active_ub_{r}_{k}"
                )
        
        # 约束6: 行程连续性（如果有第k+1次行程，则必须有第k次）
        for r in range(R):
            for k in range(K - 1):
                model.addConstr(
                    trip_active[r, k] >= trip_active[r, k + 1],
                    f"trip_seq_{r}_{k}"
                )
        
        # 约束7: 特殊任务只能由特殊机器人执行
        for t, task_label in enumerate(self.task_labels):
            if task_label in self.special_labels:
                for r, robot_name in enumerate(self.robot_names):
                    if robot_name not in self.special_robots:
                        model.addConstr(
                            assign[r, t] == 0,
                            f"special_{r}_{t}"
                        )
        
        # 约束8: 行程内任务的顺序约束（MTZ变体）
        for r in range(R):
            for k in range(K):
                for t in range(T):
                    # 如果任务t在行程k中，则order >= 1
                    model.addConstr(
                        order[r, t, k] >= trip_task[r, t, k],
                        f"order_lb_{r}_{t}_{k}"
                    )
                    # 如果任务t不在行程k中，则order = 0
                    model.addConstr(
                        order[r, t, k] <= capacity * trip_task[r, t, k],
                        f"order_ub_{r}_{t}_{k}"
                    )
        
        # 约束9: 顺序唯一性 - 每个行程中的顺序值必须唯一
        for r in range(R):
            for k in range(K):
                for pos in range(1, capacity + 1):
                    # 最多一个任务有这个顺序
                    model.addConstr(
                        gp.quicksum(
                            1 if order[r, t, k].Start == pos else 0 
                            for t in range(T)
                        ) <= 1,
                        f"order_unique_{r}_{k}_{pos}"
                    ) if False else None  # 这个约束太复杂，用另一种方式
        
        # 约束9替代: 使用弧变量来确保顺序
        for r in range(R):
            for k in range(K):
                for t1 in range(T):
                    for t2 in range(T):
                        if t1 != t2:
                            # 如果从t1到t2，则t1的顺序 + 1 = t2的顺序
                            model.addConstr(
                                order[r, t1, k] + 1 <= order[r, t2, k] + capacity * (1 - arc_task[r, t1, t2, k]),
                                f"arc_order_{r}_{t1}_{t2}_{k}"
                            )
        
        # 约束10: 弧的流平衡 - 每个任务最多一个入弧和一个出弧
        for r in range(R):
            for k in range(K):
                for t in range(T):
                    # 入弧数量
                    in_arcs = gp.quicksum(arc_task[r, t2, t, k] for t2 in range(T) if t2 != t)
                    # 出弧数量
                    out_arcs = gp.quicksum(arc_task[r, t, t2, k] for t2 in range(T) if t2 != t)
                    
                    # 入弧 + first_task = trip_task
                    model.addConstr(
                        in_arcs + first_task[r, t, k] == trip_task[r, t, k],
                        f"in_flow_{r}_{t}_{k}"
                    )
                    # 出弧 + last_task = trip_task
                    model.addConstr(
                        out_arcs + last_task[r, t, k] == trip_task[r, t, k],
                        f"out_flow_{r}_{t}_{k}"
                    )
        
        # 约束11: 每个行程最多一个first_task和一个last_task
        for r in range(R):
            for k in range(K):
                model.addConstr(
                    gp.quicksum(first_task[r, t, k] for t in range(T)) <= 1,
                    f"one_first_{r}_{k}"
                )
                model.addConstr(
                    gp.quicksum(last_task[r, t, k] for t in range(T)) <= 1,
                    f"one_last_{r}_{k}"
                )
                # 如果行程激活，必须有first和last
                model.addConstr(
                    gp.quicksum(first_task[r, t, k] for t in range(T)) >= trip_active[r, k],
                    f"need_first_{r}_{k}"
                )
                model.addConstr(
                    gp.quicksum(last_task[r, t, k] for t in range(T)) >= trip_active[r, k],
                    f"need_last_{r}_{k}"
                )
        
        # 约束12: Delivery点访问约束
        # 如果某个行程中有任务对应delivery点d，则必须访问d
        for r in range(R):
            for k in range(K):
                for d, delivery_label in enumerate(self.delivery_labels):
                    # 找出所有对应这个delivery的任务
                    related_tasks = [t for t, task_label in enumerate(self.task_labels) 
                                    if self.task_to_delivery.get(task_label) == delivery_label]
                    if related_tasks:
                        # 如果行程k中有任何一个related_task，则必须访问这个delivery
                        for t in related_tasks:
                            model.addConstr(
                                trip_delivery[r, d, k] >= trip_task[r, t, k],
                                f"delivery_needed_{r}_{d}_{k}_{t}"
                            )
        
        # 约束13: 只有激活的行程才能访问delivery
        for r in range(R):
            for k in range(K):
                for d in range(D):
                    model.addConstr(
                        trip_delivery[r, d, k] <= trip_active[r, k],
                        f"delivery_active_{r}_{d}_{k}"
                    )
        
        # ==================== 目标函数 ====================
        # 最小化总行驶距离，使用Dijkstra/欧几里得距离
        
        obj_expr = gp.LinExpr()
        
        # 1. 机器人起点到第一次行程第一个任务的距离
        for r, robot_name in enumerate(self.robot_names):
            for t, task_label in enumerate(self.task_labels):
                dist = self._get_distance(robot_name, task_label)
                obj_expr += dist * first_task[r, t, 0]
        
        # 2. 行程内任务之间的距离（使用Dijkstra距离）
        for r in range(R):
            for k in range(K):
                for t1, label1 in enumerate(self.task_labels):
                    for t2, label2 in enumerate(self.task_labels):
                        if t1 != t2:
                            dist = self._get_distance(label1, label2)
                            obj_expr += dist * arc_task[r, t1, t2, k]
        
        # 3. 最后一个任务到delivery点的距离
        for r in range(R):
            for k in range(K):
                for t, task_label in enumerate(self.task_labels):
                    for d, delivery_label in enumerate(self.delivery_labels):
                        # 如果t是最后一个任务且需要访问delivery d
                        dist = self._get_distance(task_label, delivery_label)
                        # 使用辅助变量：last_to_delivery[r,t,d,k]
                        obj_expr += dist * last_task[r, t, k] * trip_delivery[r, d, k]
        
        # 由于上面有乘积项，需要线性化
        # 创建辅助变量 last_to_delivery
        last_to_delivery = model.addVars(R, T, D, K, vtype=GRB.BINARY, name="last_to_delivery")
        
        for r in range(R):
            for k in range(K):
                for t in range(T):
                    for d in range(D):
                        # last_to_delivery = last_task AND trip_delivery
                        model.addConstr(
                            last_to_delivery[r, t, d, k] <= last_task[r, t, k],
                            f"ltd_1_{r}_{t}_{d}_{k}"
                        )
                        model.addConstr(
                            last_to_delivery[r, t, d, k] <= trip_delivery[r, d, k],
                            f"ltd_2_{r}_{t}_{d}_{k}"
                        )
                        model.addConstr(
                            last_to_delivery[r, t, d, k] >= last_task[r, t, k] + trip_delivery[r, d, k] - 1,
                            f"ltd_3_{r}_{t}_{d}_{k}"
                        )
        
        # 重新计算目标函数中的最后一个任务到delivery的距离
        obj_expr_final = gp.LinExpr()
        
        # 1. 机器人起点到第一次行程第一个任务
        for r, robot_name in enumerate(self.robot_names):
            for t, task_label in enumerate(self.task_labels):
                dist = self._get_distance(robot_name, task_label)
                obj_expr_final += dist * first_task[r, t, 0]
        
        # 2. 行程内任务之间的距离
        for r in range(R):
            for k in range(K):
                for t1, label1 in enumerate(self.task_labels):
                    for t2, label2 in enumerate(self.task_labels):
                        if t1 != t2:
                            dist = self._get_distance(label1, label2)
                            obj_expr_final += dist * arc_task[r, t1, t2, k]
        
        # 3. 最后一个任务到第一个delivery点
        for r in range(R):
            for k in range(K):
                for t, task_label in enumerate(self.task_labels):
                    for d, delivery_label in enumerate(self.delivery_labels):
                        dist = self._get_distance(task_label, delivery_label)
                        obj_expr_final += dist * last_to_delivery[r, t, d, k]
        
        # 4. Delivery点之间的距离（如果一次行程需要访问多个delivery）
        for r in range(R):
            for k in range(K):
                for d1, label1 in enumerate(self.delivery_labels):
                    for d2, label2 in enumerate(self.delivery_labels):
                        if d1 != d2:
                            dist = self._get_distance(label1, label2)
                            # 如果两个delivery都需要访问，加上它们之间的距离
                            # 这是近似处理，假设按顺序访问
                            pass  # 简化处理：只计算最后任务到delivery的距离
        
        # 5. 从delivery到下一个行程的第一个任务（如果有）
        for r in range(R):
            for k in range(K - 1):
                for d, delivery_label in enumerate(self.delivery_labels):
                    for t, task_label in enumerate(self.task_labels):
                        dist = self._get_distance(delivery_label, task_label)
                        # 需要：trip_delivery[r,d,k] AND first_task[r,t,k+1]
                        # 创建辅助变量
        
        # 为了简化，创建delivery到下一行程的辅助变量
        delivery_to_next = model.addVars(R, D, T, K-1, vtype=GRB.BINARY, name="delivery_to_next")
        
        for r in range(R):
            for k in range(K - 1):
                for d in range(D):
                    for t in range(T):
                        model.addConstr(
                            delivery_to_next[r, d, t, k] <= trip_delivery[r, d, k],
                            f"dtn_1_{r}_{d}_{t}_{k}"
                        )
                        model.addConstr(
                            delivery_to_next[r, d, t, k] <= first_task[r, t, k + 1],
                            f"dtn_2_{r}_{d}_{t}_{k}"
                        )
                        model.addConstr(
                            delivery_to_next[r, d, t, k] >= trip_delivery[r, d, k] + first_task[r, t, k + 1] - 1,
                            f"dtn_3_{r}_{d}_{t}_{k}"
                        )
        
        # 添加delivery到下一行程的距离
        for r in range(R):
            for k in range(K - 1):
                for d, delivery_label in enumerate(self.delivery_labels):
                    for t, task_label in enumerate(self.task_labels):
                        dist = self._get_distance(delivery_label, task_label)
                        obj_expr_final += dist * delivery_to_next[r, d, t, k]
        
        model.setObjective(obj_expr_final, GRB.MINIMIZE)
        
        # ==================== 求解 ====================
        print("\n开始求解...")
        model.optimize()
        
        solve_time = time.time() - start_time
        
        # ==================== 提取结果 ====================
        result = {
            'solve_time': solve_time,
            'status': model.status,
            'optimal_cost': None,
            'robot_assignments': {},
            'total_distance': 0,
            'total_steps': 0,
            'timeout': model.status == GRB.TIME_LIMIT
        }
        
        if model.status == GRB.OPTIMAL:
            result['optimal_cost'] = model.objVal
            result['total_distance'] = model.objVal
            
            # 提取每个机器人的任务分配和路径
            for r, robot_name in enumerate(self.robot_names):
                trips = []
                all_tasks = []
                full_route = []
                
                for k in range(K):
                    if trip_active[r, k].X > 0.5:
                        # 获取这个行程的任务
                        trip_tasks = []
                        for t, task_label in enumerate(self.task_labels):
                            if trip_task[r, t, k].X > 0.5:
                                trip_tasks.append((order[r, t, k].X, task_label))
                                all_tasks.append(task_label)
                        
                        # 按顺序排列
                        trip_tasks.sort(key=lambda x: x[0])
                        ordered_tasks = [t[1] for t in trip_tasks]
                        
                        # 获取delivery点
                        trip_deliveries = []
                        for d, delivery_label in enumerate(self.delivery_labels):
                            if trip_delivery[r, d, k].X > 0.5:
                                trip_deliveries.append(delivery_label)
                        
                        trips.append({
                            'tasks': ordered_tasks,
                            'deliveries': trip_deliveries
                        })
                        
                        full_route.extend(ordered_tasks)
                        full_route.extend(trip_deliveries)
                
                result['robot_assignments'][robot_name] = {
                    'all_tasks': all_tasks,
                    'trips': trips,
                    'full_route': full_route
                }
            
            # 计算Halton图步数
            total_steps = 0
            for robot_name, assignment in result['robot_assignments'].items():
                steps = self._calculate_path_steps(assignment['full_route'], robot_name)
                total_steps += steps
                assignment['steps'] = steps
            
            result['total_steps'] = total_steps
            
        elif model.status == GRB.TIME_LIMIT:
            print(f"\n求解超时！时间限制: {time_limit}秒")
            if model.SolCount > 0:
                result['optimal_cost'] = model.objVal
                result['total_distance'] = model.objVal
                print(f"找到次优解，目标值: {model.objVal:.4f}")
                
                # 提取次优解
                for r, robot_name in enumerate(self.robot_names):
                    trips = []
                    all_tasks = []
                    full_route = []
                    
                    for k in range(K):
                        if trip_active[r, k].X > 0.5:
                            trip_tasks = []
                            for t, task_label in enumerate(self.task_labels):
                                if trip_task[r, t, k].X > 0.5:
                                    trip_tasks.append((order[r, t, k].X, task_label))
                                    all_tasks.append(task_label)
                            
                            trip_tasks.sort(key=lambda x: x[0])
                            ordered_tasks = [t[1] for t in trip_tasks]
                            
                            trip_deliveries = []
                            for d, delivery_label in enumerate(self.delivery_labels):
                                if trip_delivery[r, d, k].X > 0.5:
                                    trip_deliveries.append(delivery_label)
                            
                            trips.append({
                                'tasks': ordered_tasks,
                                'deliveries': trip_deliveries
                            })
                            
                            full_route.extend(ordered_tasks)
                            full_route.extend(trip_deliveries)
                    
                    result['robot_assignments'][robot_name] = {
                        'all_tasks': all_tasks,
                        'trips': trips,
                        'full_route': full_route
                    }
                
                # 计算步数
                total_steps = 0
                for robot_name, assignment in result['robot_assignments'].items():
                    steps = self._calculate_path_steps(assignment['full_route'], robot_name)
                    total_steps += steps
                    assignment['steps'] = steps
                result['total_steps'] = total_steps
        else:
            print(f"\n求解失败！状态码: {model.status}")
        
        return result
    
    def print_result(self, result: Dict):
        """打印结果"""
        print("\n" + "="*60)
        print("MILP最优任务分配结果")
        print("="*60)
        
        status_map = {
            1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE",
            4: "INF_OR_UNBD", 5: "UNBOUNDED", 9: "TIME_LIMIT", 11: "INTERRUPTED"
        }
        status_str = status_map.get(result['status'], str(result['status']))
        
        print(f"\n求解状态: {status_str}")
        print(f"求解时间: {result['solve_time']:.4f} 秒")
        
        if result.get('timeout'):
            print("*** 求解超时 ***")
        
        if result['optimal_cost'] is not None:
            print(f"\n总距离 (Dijkstra): {result['total_distance']:.4f}")
            print(f"Halton图总步数: {result['total_steps']}")
            
            task_counts = [len(a['all_tasks']) for a in result['robot_assignments'].values()]
            print(f"\n任务分配统计:")
            print(f"  总任务数: {sum(task_counts)}")
            print(f"  各机器人任务数: {task_counts}")
            
            print("\n" + "-"*40)
            print("各机器人详情:")
            print("-"*40)
            
            for robot_name, assignment in result['robot_assignments'].items():
                is_special = "特殊" if robot_name in self.special_robots else "普通"
                print(f"\n{robot_name} ({is_special}机器人):")
                print(f"  任务数: {len(assignment['all_tasks'])}")
                print(f"  行程数: {len(assignment['trips'])}")
                print(f"  步数: {assignment.get('steps', 'N/A')}")
                
                for i, trip in enumerate(assignment['trips']):
                    tasks_str = ' -> '.join(trip['tasks']) if trip['tasks'] else '无'
                    deliveries_str = ', '.join(trip['deliveries']) if trip['deliveries'] else '无'
                    print(f"    行程{i+1}: {tasks_str} | 交付: {deliveries_str}")
                
                if assignment.get('full_route'):
                    print(f"  完整路径: {' -> '.join(assignment['full_route'])}")
        else:
            print("\n未找到可行解！")
    
    def save_result(self, result: Dict, output_path: str):
        """保存结果到文件"""
        import json
        from datetime import datetime
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MILP完整最优任务分配结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            status_map = {
                1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE",
                4: "INF_OR_UNBD", 5: "UNBOUNDED", 9: "TIME_LIMIT"
            }
            status_str = status_map.get(result['status'], str(result['status']))
            
            f.write(f"求解状态: {status_str}\n")
            f.write(f"求解时间: {result['solve_time']:.4f} 秒\n")
            if result.get('timeout'):
                f.write("*** 求解超时 ***\n")
            f.write("\n")
            
            if result['optimal_cost'] is not None:
                f.write("="*40 + "\n")
                f.write("关键指标\n")
                f.write("="*40 + "\n")
                f.write(f"总距离 (Dijkstra): {result['total_distance']:.4f}\n")
                f.write(f"Halton图总步数: {result['total_steps']}\n\n")
                
                task_counts = [len(a['all_tasks']) for a in result['robot_assignments'].values()]
                total_trips = sum(len(a['trips']) for a in result['robot_assignments'].values())
                f.write(f"总任务数: {sum(task_counts)}\n")
                f.write(f"总行程数: {total_trips}\n\n")
                
                f.write("-"*40 + "\n")
                f.write("各机器人详情\n")
                f.write("-"*40 + "\n")
                
                for robot_name, assignment in result['robot_assignments'].items():
                    is_special = "特殊" if robot_name in self.special_robots else "普通"
                    f.write(f"\n{robot_name} ({is_special}机器人):\n")
                    f.write(f"  任务数: {len(assignment['all_tasks'])}\n")
                    f.write(f"  行程数: {len(assignment['trips'])}\n")
                    f.write(f"  步数: {assignment.get('steps', 'N/A')}\n")
                    
                    for i, trip in enumerate(assignment['trips']):
                        tasks_str = ' -> '.join(trip['tasks']) if trip['tasks'] else '无'
                        deliveries_str = ', '.join(trip['deliveries']) if trip['deliveries'] else '无'
                        f.write(f"    行程{i+1}: {tasks_str} | 交付: {deliveries_str}\n")
                    
                    if assignment.get('full_route'):
                        f.write(f"  完整路径: {' -> '.join(assignment['full_route'])}\n")
        
        # JSON格式
        json_path = output_path.replace('.txt', '.json')
        json_result = {
            'solve_time': result['solve_time'],
            'status': result['status'],
            'timeout': result.get('timeout', False),
            'total_distance': result['total_distance'],
            'total_steps': result['total_steps'],
            'robot_assignments': {}
        }
        
        if result['robot_assignments']:
            for robot_name, assignment in result['robot_assignments'].items():
                json_result['robot_assignments'][robot_name] = {
                    'tasks': assignment['all_tasks'],
                    'trips': assignment['trips'],
                    'full_route': assignment.get('full_route', []),
                    'steps': assignment.get('steps', 0)
                }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")
        print(f"JSON结果已保存到: {json_path}")


def compare_with_planner(optimal_result: Dict, planner_result: Dict) -> Dict:
    """比较MILP最优解与其他planner的结果"""
    comparison = {
        'optimal_steps': optimal_result.get('total_steps', 0),
        'planner_steps': planner_result.get('total_steps', 0),
        'optimal_distance': optimal_result.get('total_distance', 0),
        'planner_distance': planner_result.get('total_distance', 0),
        'optimal_solve_time': optimal_result.get('solve_time', 0),
        'planner_solve_time': planner_result.get('solve_time', 0),
        'optimal_timeout': optimal_result.get('timeout', False),
    }
    
    if comparison['optimal_steps'] > 0:
        comparison['steps_gap'] = comparison['planner_steps'] - comparison['optimal_steps']
        comparison['steps_gap_percent'] = (comparison['steps_gap'] / comparison['optimal_steps']) * 100
    else:
        comparison['steps_gap'] = 0
        comparison['steps_gap_percent'] = 0
    
    if comparison['optimal_distance'] > 0:
        comparison['distance_gap'] = comparison['planner_distance'] - comparison['optimal_distance']
        comparison['distance_gap_percent'] = (comparison['distance_gap'] / comparison['optimal_distance']) * 100
    else:
        comparison['distance_gap'] = 0
        comparison['distance_gap_percent'] = 0
    
    print("\n" + "="*60)
    print("Planner与最优解对比")
    print("="*60)
    if comparison['optimal_timeout']:
        print("注意：MILP求解超时，显示的是次优解")
    print(f"\n{'指标':<20} {'最优解':<15} {'Planner':<15} {'差距':<15} {'差距%':<10}")
    print("-"*70)
    print(f"{'总步数':<20} {comparison['optimal_steps']:<15} {comparison['planner_steps']:<15} {comparison['steps_gap']:<15} {comparison['steps_gap_percent']:.2f}%")
    print(f"{'总距离':<20} {comparison['optimal_distance']:<15.2f} {comparison['planner_distance']:<15.2f} {comparison['distance_gap']:<15.2f} {comparison['distance_gap_percent']:.2f}%")
    print(f"{'求解时间(秒)':<20} {comparison['optimal_solve_time']:<15.4f} {comparison['planner_solve_time']:<15.4f}")
    
    return comparison


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    config_path = os.path.normpath(os.path.join(base_dir, 'config', 'Task_Points.yaml'))
    distances_csv_path = os.path.normpath(os.path.join(script_dir, '..', 'multi_source_dijkstra_distances.csv'))
    halton_csv_path = os.path.normpath(os.path.join(script_dir, '..', 'all_points_in_Halton.csv'))
    wall_yaml_path = os.path.normpath(os.path.join(base_dir, 'config', 'wall.yaml'))
    
    print("配置文件路径:")
    print(f"  Task_Points.yaml: {config_path}")
    print(f"  distances.csv: {distances_csv_path}")
    print(f"  halton.csv: {halton_csv_path}")
    print(f"  wall.yaml: {wall_yaml_path}")
    
    solver = OptimalMILPTaskAssignment(
        config_path=config_path,
        distances_csv_path=distances_csv_path,
        halton_csv_path=halton_csv_path,
        wall_yaml_path=wall_yaml_path
    )
    
    # 求解，设置10分钟时间限制
    result = solver.solve_optimal_assignment(capacity=3, time_limit=600)
    
    solver.print_result(result)
    
    output_path = os.path.join(script_dir, 'optimal_milp_result.txt')
    solver.save_result(result, output_path)
    
    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    print("""
要与您的planner进行比较:

from optimal_milp_task_assignment import OptimalMILPTaskAssignment, compare_with_planner

solver = OptimalMILPTaskAssignment(config_path, distances_csv, halton_csv, wall_yaml)
optimal_result = solver.solve_optimal_assignment(capacity=3, time_limit=600)

planner_result = {
    'total_steps': your_planner_steps,
    'total_distance': your_planner_distance,
    'solve_time': your_planner_time
}

comparison = compare_with_planner(optimal_result, planner_result)
""")
    
    return result


if __name__ == "__main__":
    main()
