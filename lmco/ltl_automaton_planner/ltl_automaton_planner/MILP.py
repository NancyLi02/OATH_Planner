import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

class ClusterTaskPlanner:
    """
    MILP-based task planner for computing optimal task sequence within a cluster.
    """
    def __init__(self, dijkstra_distances_csv_path: str):
        self.dijkstra_distances_csv_path = dijkstra_distances_csv_path
        self.dijkstra_distances = self._load_dijkstra_distances()

    def _load_dijkstra_distances(self) -> Dict[Tuple[str, str], float]:
        try:
            df = pd.read_csv(self.dijkstra_distances_csv_path)
            distances = {}
            for _, row in df.iterrows():
                from_label = str(row['from'])
                to_label = str(row['to'])
                distance = float(row['distance'])
                
                distances[(from_label, to_label)] = distance
                distances[(to_label, from_label)] = distance
            return distances
        except Exception as e:
            print(f"Warning: Could not load Dijkstra distances from {self.dijkstra_distances_csv_path}: {e}")
            return {}


    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def plan_cluster_tasks(
        self,
        robot_start: Tuple[float, float],
        pickup_points: List[Tuple[float, float]],
        pickup_labels: List[str],
        delivery_points: List[Tuple[float, float]],
        delivery_labels: List[str],
        pickup_to_delivery: Dict[str, str],
        robot_capacity: int
    ) -> Tuple[List[str], List[str], List[str], float]:
        """
        Plan the optimal task sequence for the robot.

        Args:
            robot_start: Robot's starting (x, y).
            pickup_points: List of (x, y) of all pickup tasks.
            pickup_labels: Task labels, must be unique for pickup.
            delivery_points: List of (x, y) of all delivery points, label must be unique.
            delivery_labels: List of unique labels for delivery points.
            pickup_to_delivery: Dict mapping pickup label -> delivery label.
            robot_capacity: Maximum number of pickups allowed.

        Returns:
            (chosen_pickup_labels, chosen_delivery_labels, route_labels, total_cost)
        """
        n_pickups = len(pickup_points)
        n_deliveries = len(delivery_points)

        # Map: pickup index <-> label, delivery index <-> label
        pickup_label_to_idx = {label: i for i, label in enumerate(pickup_labels)}
        delivery_label_to_idx = {label: 1 + n_pickups + i for i, label in enumerate(delivery_labels)}
        delivery_idx_to_label = {1 + n_pickups + i: label for i, label in enumerate(delivery_labels)}

        # Build the list of all nodes: [robot_start] + pickup nodes + delivery nodes
        nodes = ['start'] + pickup_labels + delivery_labels
        node_coords = [robot_start] + pickup_points + delivery_points

        N = len(nodes)

        # Build distance matrix (robot-task uses Euclidean, task-task uses Dijkstra or fallback Euclidean)
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    d[i, j] = 1e6  # No self loop
                    continue
                if i == 0:  # start to others
                    d[i, j] = self._euclidean_distance(node_coords[i], node_coords[j])
                elif j == 0:  # others to start (should not happen, but keep for symmetry)
                    d[i, j] = self._euclidean_distance(node_coords[i], node_coords[j])
                else:
                    from_label = nodes[i]
                    to_label = nodes[j]
                    key = (from_label, to_label)
                    if key in self.dijkstra_distances:
                        d[i, j] = self.dijkstra_distances[key]
                    else:
                        d[i, j] = self._euclidean_distance(node_coords[i], node_coords[j])

        # Build model
        m = gp.Model("cluster_task_planning")
        m.Params.LogToConsole = 0

        # x[i,j]: whether to go from node i to node j
        x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

        # y[i]: whether to select pickup i (for 1..n_pickups)
        y = m.addVars(n_pickups, vtype=GRB.BINARY, name="y")

        # z[j]: whether to select delivery j (for 0..n_deliveries-1, referencing delivery_labels)
        z = m.addVars(n_deliveries, vtype=GRB.BINARY, name="z")

        # order[i]: position of node i in the tour (for subtour elimination)
        order = m.addVars(N, vtype=GRB.INTEGER, lb=0, ub=N-1, name="order")

        # Ensure robot leaves start node
        m.addConstr(gp.quicksum(x[0, j] for j in range(1, N)) == 1, "leave_start")

        # Each visited node must be entered and exited exactly once (if selected)
        for i in range(1, N):
            m.addConstr(gp.quicksum(x[j, i] for j in range(N) if j != i) == 
                        gp.quicksum(x[i, j] for j in range(N) if j != i), f"flow_{i}")

        # Only selected pickups can be visited (in/out flow == y[i])
        for i in range(n_pickups):
            node_idx = i + 1
            m.addConstr(gp.quicksum(x[j, node_idx] for j in range(N) if j != node_idx) == y[i], f"pickup_in_{i}")
            m.addConstr(gp.quicksum(x[node_idx, j] for j in range(N) if j != node_idx) == y[i], f"pickup_out_{i}")


        m.addConstr(y.sum() <= robot_capacity, "capacity")

        m.addConstr(y.sum() >= 1, "min_pickup")

        for j, delivery_label in enumerate(delivery_labels):
            related_pickup_idx = [i for i, pl in enumerate(pickup_labels) if pickup_to_delivery[pl] == delivery_label]
            if related_pickup_idx:
                m.addConstr(z[j] <= gp.quicksum(y[i] for i in related_pickup_idx), f"delivery_active_ub_{delivery_label}")
                for i in related_pickup_idx:
                    m.addConstr(z[j] >= y[i], f"delivery_active_lb_{delivery_label}_{i}")
            else:
                m.addConstr(z[j] == 0, f"delivery_active_zero_{delivery_label}")



        for j, delivery_label in enumerate(delivery_labels):
            delivery_idx = delivery_label_to_idx[delivery_label]
            m.addConstr(gp.quicksum(x[i, delivery_idx] for i in range(N) if i != delivery_idx) == z[j], f"delivery_visit_{delivery_label}")
            m.addConstr(gp.quicksum(x[delivery_idx, k] for k in range(N) if k != delivery_idx) == z[j], f"delivery_leave_{delivery_label}")

        # Pickup必须在其Delivery之前被访问
        for i, pickup_label in enumerate(pickup_labels):
            delivery_label = pickup_to_delivery[pickup_label]
            pickup_idx = i + 1
            delivery_idx = delivery_label_to_idx[delivery_label]
            m.addConstr(order[pickup_idx] + 1 <= order[delivery_idx] + N * (1 - y[i]), f"pickup_before_delivery_{pickup_label}")

        # Subtour elimination (MTZ)
        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    m.addConstr(order[i] + 1 <= order[j] + N * (1 - x[i, j]), f"subtour_{i}_{j}")

        # Objective: minimize total cost
        m.setObjective(gp.quicksum(d[i, j] * x[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)

        m.optimize()

        # Get results
        if m.status == GRB.OPTIMAL:
            # Selected pickups
            chosen_pickup_labels = [pickup_labels[i] for i in range(n_pickups) if y[i].X > 0.5]
            # Related deliveries
            chosen_delivery_labels = [delivery_labels[j] for j in range(n_deliveries) if z[j].X > 0.5]
            # Route reconstruction
            # Route reconstruction
            succ = {}
            for i in range(N):
                for j in range(N):
                    if x[i, j].X > 0.5:
                        succ[i] = j

            route_labels = []
            visited = set()
            node = 0
            while True:
                next_node = succ.get(node)
                if next_node is None or next_node == 0 or next_node in visited:
                    break
                route_labels.append(nodes[next_node])
                visited.add(next_node)
                node = next_node

            # 补全遗漏的 delivery
            chosen_delivery_set = set([delivery_labels[j] for j in range(n_deliveries) if z[j].X > 0.5])
            route_delivery_set = set([label for label in route_labels if label in delivery_labels])

            if chosen_delivery_set != route_delivery_set:
                for label in chosen_delivery_set - route_delivery_set:
                    route_labels.append(label)

            return chosen_pickup_labels, chosen_delivery_labels, route_labels, m.objVal
        else:
            return [], [], [], float("inf")
 