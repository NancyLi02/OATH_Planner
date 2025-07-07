import os
import numpy as np
import rclpy
from rclpy.node import Node
from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_planner.MILP import ClusterTaskPlanner
from ltl_automaton_msgs.msg import ClusterTaskassign, RobotID, ClusterRequest, AgentFailTask, TaskFail
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import yaml
import re

# -------------------- CWA Algorithm --------------------
class CWA:
    def __init__(self):
        pass

    def select_cluster(self, scores_list, y, robot_index, robot_types, cluster_types):
        """
        Perform auction based on each robot's scores and cluster types:
          - For normal robots, the score for special clusters is always 0.
          - If a normal robot selects a special cluster, return -1.
          - Lower scores have higher priority (better).
          - Score of 0 means the cluster cannot be selected.
          - Robot can only bid on clusters where its score is > 0 and < current highest bid (y value).
          - When y value is 0, it means no previous bid, so any positive score is valid.
        """
        robot_scores = scores_list[robot_index]
        valid_cluster = [(1 if score > 0 and (y_value == 0 or score < y_value) else 0) for score, y_value in zip(robot_scores, y)]
        
        if sum(valid_cluster) > 0:
            valid_indices = [i for i, valid in enumerate(valid_cluster) if valid == 1]
            # If the robot is normal, filter out special clusters.
            if robot_types[robot_index] == 'normal':
                valid_indices = [i for i in valid_indices if cluster_types[i] != 'special']
            if not valid_indices:
                return -1
            best_cluster_index = min(valid_indices, key=lambda idx: robot_scores[idx])
            best_cluster_score = robot_scores[best_cluster_index]
            y[best_cluster_index] = best_cluster_score
            # print(f"Robot {robot_index + 1}: Selected cluster index {best_cluster_index} with score {best_cluster_score:.2f}")
            return best_cluster_index
        else:
            return -1

    def conflict_resolve(self, cluster_index, assigned_clusters, x):
        for robot, assigned_cluster in assigned_clusters:
            if assigned_cluster == cluster_index:
                # print(f"Conflict detected for cluster {cluster_index}. Removing previous assignment for Robot {robot + 1}.")
                x[robot][cluster_index] = 0
                assigned_clusters.remove((robot, assigned_cluster))
                break
        return x, assigned_clusters

    def initial_cluster_assignment(self, scores_list, robot_types, cluster_types):
        """
        Continue the auction process until every robot has been assigned a cluster.
        """
        cluster_count = len(scores_list[0])
        num_robots = len(scores_list)
        x = [[0] * cluster_count for _ in range(num_robots)]
        y = [0] * cluster_count
        assigned_clusters = []
        unassigned_robots = []

        # Continue until each robot has a cluster assignment.
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

        # print("\nFinal y list:", y)
        # print("x list (cluster assignment status for each robot):")
        # for robot_index, cluster_assignments in enumerate(x):
        #     print(f"Robot {robot_index + 1}: {cluster_assignments}")
        # print(f"\nUnassigned Robots: {unassigned_robots}")

        return assigned_clusters, unassigned_robots


# -------------------- Task Assign Node --------------------
class TaskAssignNode(Node):
    def __init__(self):
        super().__init__('taskassign_node')

        # ----- Static robot poses and names -----
        self.robot_count = 4
        # Unified robot names: "robot1", "robot2", etc.
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        self.robot_poses = [
            (1, 19),   # robot1 (normal robot)
            (11, 19),  # robot2 (special robot)
            (9, 11),   # robot3 (special robot)
            (11, 9),   # robot4 (normal robot)
            # (1, 39),   # robot5 (normal robot)
            # (11, 39),  # robot6 (special robot)
            # (9, 31),   # robot7 (special robot)
            # (11, 29),  # robot8 (normal robot)
            # (21, 19),  # robot9 (normal robot)
            # (31, 19),  # robot10 (normal robot)
            # (29, 11),  # robot11 (normal robot)
            # (31, 9),   # robot12 (normal robot)
            # (21, 39),  # robot13 (normal robot)
            # (31, 39),  # robot14 (normal robot)
            # (29, 31),  # robot15 (normal robot)
            # (31, 29)   # robot16 (normal robot)
        ]

        # Define robot types: only robot2 and robot3 are 'special'; others are 'normal'
        self.robot_types = []
        for i in range(self.robot_count):
            if i in [2, 3]:
                self.robot_types.append('special')
            else:
                self.robot_types.append('normal')

        # Publishers for each robot namespace
        self.task_pubs = {}
        for idx, robot in enumerate(self.robot_names):
            topic = f"/{robot}/ClusterTaskassign"
            self.task_pubs[robot] = self.create_publisher(ClusterTaskassign, topic, 10)

        # Subscribers for new cluster request topics
        self.new_cluster_request_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/cluster_request"
            sub = self.create_subscription(ClusterRequest, topic, self.assign_new_cluster, 10)
            self.new_cluster_request_subs.append(sub)

        self.robot_fail_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/agent_fail_task"
            sub = self.create_subscription(AgentFailTask, topic, self.agent_fail_callback, 10)
            self.robot_fail_subs.append(sub)

        self.task_fail_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/task_failure_cluster"
            sub = self.create_subscription(TaskFail, topic, self.task_fail_callback, 10)
            self.task_fail_subs.append(sub)

        self.broke_agents = []

        # ----- Load wall and task info -----
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner'))
        wall_path = os.path.join(parent_dir, 'config', 'wall.yaml')
        halton_points_csv = "/home/nanli/ros2_ws/src/all_points_in_Halton.csv"
        precomputed_distances_csv = "/home/nanli/ros2_ws/src/multi_source_dijkstra_distances.csv"

        # === 从YAML读取任务点、映射和特殊标签 ===
        task_points_yaml = os.path.join(parent_dir, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # 解析任务点
        points_with_label = {}
        for k, v in yaml_data['task_points'].items():
            # k是"x,y"字符串，转为tuple(float, float)
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
        self.points_with_label = points_with_label
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        # 解析special_labels
        self.special_labels = set(yaml_data.get('special_labels', []))

        # 解析task_to_delivery
        # 先反向映射：label->group
        task_to_delivery = {}
        for group, label_list in yaml_data['task_to_delivery'].items():
            for label in label_list:
                task_to_delivery[label] = group
        self.task_to_delivery = task_to_delivery

        # Load wall data from YAML file and perform task clustering using new CostMapClusterer interface
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=wall_path,
            num_clusters=4,
            halton_points_csv=halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=precomputed_distances_csv
        )
        
        # Get clustering results - now returns (centers, points) instead of separate normal/special
        self.cluster_centers, self.cluster_points = self.clusterer.cluster()
        
        # 统一顺序，直接生成gamma和类型
        self.cluster_gamma = []  # Store γ values for each cluster
        self.cluster_types = []  # Store type: 'normal', 'special', or 'hybrid' for each cluster
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
            
            # Determine cluster type based on task composition
            if normal_count > 0 and special_count > 0:
                cluster_type = 'hybrid'
            elif special_count > 0:
                cluster_type = 'special'
            else:
                cluster_type = 'normal'
            
            self.cluster_types.append(cluster_type)

        
        # 有效性标记
        self.valid_cluster = [1] * len(self.cluster_centers)
        self.robot_cluster_indices = {}
        self.failed_task_owners = {}
        
        # CWA auction algorithm object.
        self.cwa_algorithm = CWA()
        
        # MILP cluster task planner
        self.cluster_planner = ClusterTaskPlanner(
            dijkstra_distances_csv_path=precomputed_distances_csv
        )

        self.assigned_points_global = set()

        self.finish_callback()

    def task_fail_callback(self, msg):  # Called when a task failure is reported
        self.solo_failure_task = msg.task_label
        self.corresponding_robot = msg.agent_id

        # Map label to point
        label_to_pt = {lbl:pt for pt,lbl in self.points_with_label.items()}
        pt = label_to_pt.get(self.solo_failure_task)
        if pt is None:
            self.get_logger().error(f"Failed task label {self.solo_failure_task} not found.")
            return
        
        # Determine cluster list and index for new single-task cluster
        if self.solo_failure_task in self.special_labels:
            self.clusters_special.append([pt])
            self.centers_special.append(tuple(pt))
            new_idx = len(self.clusters_other) + len(self.clusters_special) - 1
        else:
            self.clusters_other.append([pt])
            self.centers_other.append(tuple(pt))
            new_idx = len(self.clusters_other) - 1

        # Mark new cluster valid at its index
        self.valid_cluster.insert(new_idx, 1)
        
        # Record ownership so this robot gets zero score on this cluster
        self.failed_task_owners[new_idx] = self.corresponding_robot
        self.get_logger().info(f"Created failed-task cluster {new_idx} for robot {self.corresponding_robot}.")

    def agent_fail_callback(self, msg):
        self.get_logger().info(f'Task assignment node receive agent fail msg from {msg.robot_id}....')
        self.fail_task = msg.task_id  # int32
        self.fail_agent = msg.robot_id  # int32

        # Find which cluster contains this failed task
        all_clusters = self.clusters_other + self.clusters_special
        fail_cluster_idx = None
        for idx, cluster in enumerate(all_clusters):
            # Map cluster points to task indices
            indices = []
            for pt in cluster:
                label = self.points_with_label.get(tuple(map(float, pt)))
                if label is not None:
                    indices.append(self.label_to_index[label])
            if self.fail_task in indices:
                fail_cluster_idx = idx
                break

        if fail_cluster_idx is not None:
            # Mark this cluster as valid again
            self.valid_cluster[fail_cluster_idx] = 1

        # Trim this robot's sequence: keep only from the failed task onward
        robot_name = self.robot_names[self.fail_agent - 1]
        seq = self.robot_task_sequence.get(robot_name, [])
        if self.fail_task in seq:
            pos = seq.index(self.fail_task)
            onward_seq = seq[pos:]  # tasks from failure onward
            self.robot_task_sequence[robot_name] = onward_seq

            # Also trim the cluster itself so future assignments see only onward tasks
            # Build inverse mapping: label -> point
            label_to_point = {lbl: pt for pt, lbl in self.points_with_label.items()}
            # Convert indices back to points
            new_cluster_pts = [label_to_point[next(lbl for lbl, idx in self.label_to_index.items() if idx == task_id)]
                               for task_id in onward_seq]

            # Update clusters and centers for this index
            if fail_cluster_idx < len(self.clusters_other):
                # normal clusters
                self.clusters_other[fail_cluster_idx] = new_cluster_pts
                # recompute center
                self.centers_other[fail_cluster_idx] = tuple(np.mean(new_cluster_pts, axis=0))
            else:
                # special clusters
                sci = fail_cluster_idx - len(self.clusters_other)
                self.clusters_special[sci] = new_cluster_pts
                self.centers_special[sci] = tuple(np.mean(new_cluster_pts, axis=0))

        # Record this agent as broken
        if robot_name not in self.broke_agents:
            self.broke_agents.append(robot_name)

    def assign_new_cluster(self, msg):
        self.get_logger().info(f"Received new cluster assign request from Robot{msg.robot_id}.")
        # 检查是否还有未分配任务
        all_points = list(self.points_with_label.keys())
        unassigned_points = [pt for pt in all_points if pt not in self.assigned_points_global]
        if not unassigned_points:
            self.get_logger().info("No unassigned tasks left, skipping assignment.")
            return
        robot_id = msg.robot_id  # robot_id is an integer (1,2,3,4)
        robot_index = robot_id - 1
        robot_name = self.robot_names[robot_index]
        
        # Initialize robot_cluster_indices if it does not exist.
        if not hasattr(self, 'robot_cluster_indices'):
            self.robot_cluster_indices = {}

        # Update the robot's position using the position information from the message.
        # msg.position is already in float64[] format.
        new_position = tuple(msg.position)
        self.robot_poses[robot_index] = new_position
        self.get_logger().info(f"Updated position for {robot_name} to {new_position}.")

        # Prepare clustering information.
        all_clusters = self.cluster_points
        cluster_types = self.cluster_types
        self.get_logger().info(f"all_clusters: {all_clusters}")
        self.get_logger().info(f"cluster_types: {cluster_types}")
        # 判断是否所有cluster都是normal
        all_normal = all(t == 'normal' for t in cluster_types)
        # Compute the score for the requested robot only with γ adjustment.
        scores = []

         # If this robot has broken down, give zero score for all clusters
        if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
            scores = [0.0] * len(all_clusters)
        else:
            for j, (points, t_type) in enumerate(zip(all_clusters, cluster_types)):
                center = self.cluster_centers[j]
                self.get_logger().info(f"center: {center}")
                # If this is a failed-task cluster owned by this robot, score = 0
                if self.failed_task_owners.get(j) == robot_id:
                    scores.append(0.0)
                    continue
                # Normal robots cannot handle special clusters
                if self.robot_types[robot_index] == 'normal' and t_type == 'special':
                    scores.append(0.0)
                    continue
                dist = np.linalg.norm(np.array(new_position) - np.array(center))
                base_score = dist
                # 如果全是normal类型，直接用base_score
                if all_normal:
                    score = base_score
                else:
                    gamma_param = self.cluster_gamma[j]
                    if self.robot_types[robot_index] == 'normal':
                        score = base_score * gamma_param
                    else:  # special robot
                        if gamma_param > 0:
                            score = base_score / gamma_param
                        else:
                            score = base_score
                scores.append(score)
                self.get_logger().info(f"score: {score}")

        # Select the best candidate index for this robot.
        best_index = -1
        best_score = float('inf')  # Changed to inf since lower scores are better
        for j, sc in enumerate(scores):
            if sc > 0 and sc < best_score:  # Only consider valid scores (non-zero) and lower is better
                best_score = sc
                best_index = j

        if best_index == -1 or best_score == float('inf'):
            self.robot_cluster_map[robot_name] = []
            self.robot_cluster_indices[robot_name] = -1
            self.get_logger().info(f"No valid cluster found for {robot_name}. Assignment is empty.")
        else:
            self.robot_cluster_map[robot_name] = all_clusters[best_index]
            self.robot_cluster_indices[robot_name] = best_index
            self.get_logger().info(f"Assigned cluster {best_index} with score {best_score:.2f} to {robot_name}.")
            if not hasattr(self, 'cluster_task_labels'):
                self.cluster_task_labels = {}
            if not hasattr(self, 'cluster_delivery_labels'):
                self.cluster_delivery_labels = {}
            task_labels = []
            delivery_labels = []
            for coord in all_clusters[best_index]:
                for point_coord, label in self.points_with_label.items():
                    if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                        task_labels.append(label)
                        if label in self.task_to_delivery:
                            delivery_labels.append(self.task_to_delivery[label])
                        else:
                            delivery_labels.append(None)
                        break
            self.cluster_task_labels[best_index] = task_labels
            self.cluster_delivery_labels[best_index] = delivery_labels
            self.get_logger().info(f"Cluster {best_index} task labels: {task_labels}")
            self.get_logger().info(f"Cluster {best_index} delivery labels: {delivery_labels}")

        for pt in self.robot_cluster_map[robot_name]:
            self.assigned_points_global.add(tuple(pt))
        # Record this robot as the one that was just re-assigned.
        self.last_assigned_robot = robot_name
        self.get_logger().info(f"robot_cluster_map: {self.robot_cluster_map}")
        self.generate_task_sequences(robot_names=[robot_name])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()

        

    def finish_callback(self):
        # self.get_logger().info(f"Received finish_building_auto from {msg.robot_id}")
        # self.finished_robots.add(msg.robot_id)
        # # When all robots have finished building, start task assignment.
        # if all(name in self.finished_robots for name in self.robot_names):
        self.get_logger().info("Starting task assignment...")
        self.init_cluster_assign()
        self.generate_task_sequences()
        self.publish_task_assignments()

    def init_cluster_assign(self):
        """Step 1: Perform cluster-based auction assignment"""
        # Combine all cluster centers and types
        self.all_clusters = self.cluster_centers
        # Compute each robot's score for every cluster center with γ adjustment.
        scores = []
        for i, robot_pose in enumerate(self.robot_poses):
            robot_name = self.robot_names[i]
            if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
                scores.append([0.0] * len(self.all_clusters))
                continue
            robot_scores = []
            for j, (center, t_type) in enumerate(zip(self.cluster_centers, self.cluster_types)):
                # Normal robots cannot handle special clusters
                if self.robot_types[i] == 'normal' and t_type == 'special':
                    robot_scores.append(0.0)
                    continue
                
                dist = np.linalg.norm(np.array(robot_pose) - np.array(center))
                base_score = dist
                
                # Get γ value for this cluster
                gamma_param = self.cluster_gamma[j]
                
                # 统一公式：normal robot用乘法，special robot用除法
                if self.robot_types[i] == 'normal':
                    score = base_score * gamma_param
                else:  # special robot
                    if gamma_param > 0:
                        score = base_score / gamma_param
                    else:
                        score = base_score  # 当γ=0时使用base_score
                
                robot_scores.append(score)
            scores.append(robot_scores)

        self.get_logger().info(f"scores: {scores}")

        # Call the auction algorithm for initial cluster assignment.
        assigned, _ = self.cwa_algorithm.initial_cluster_assignment(scores, self.robot_types, self.cluster_types)
        self.get_logger().info(f"assigned: {assigned}")
        # Map the auction results with the clustering data:
        self.robot_cluster_map = {name: [] for name in self.robot_names}
        self.robot_cluster_indices = {}  # 初始化robot_cluster_indices
        for robot_index, cluster_index in assigned:
            robot_name = self.robot_names[robot_index]
            if cluster_index == -1:
                # If a normal robot wins a special cluster (or no valid cluster), mark assignment as empty.
                self.robot_cluster_map[robot_name] = []
                self.robot_cluster_indices[robot_name] = -1
            else:
                self.robot_cluster_map[robot_name] = self.cluster_points[cluster_index]
                self.robot_cluster_indices[robot_name] = cluster_index
                # Update valid_cluster: mark the assigned cluster as used by setting its valid_cluster flag to 0.
                self.valid_cluster[cluster_index] = 0

        # Add task labels and delivery labels for each cluster
        self.cluster_task_labels = {}  # cluster_index -> [task_labels]
        self.cluster_delivery_labels = {}  # cluster_index -> [delivery_labels]
        
        for cluster_index, points in enumerate(self.cluster_points):
            task_labels = []
            delivery_labels = []
            
            for coord in points:
                # Find the corresponding task label for this coordinate
                for point_coord, label in self.points_with_label.items():
                    if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                        task_labels.append(label)
                        # Get the corresponding delivery label
                        if label in self.task_to_delivery:
                            delivery_labels.append(self.task_to_delivery[label])
                        else:
                            delivery_labels.append(None)  # No delivery for this task
                        break
            
            self.cluster_task_labels[cluster_index] = task_labels
            self.cluster_delivery_labels[cluster_index] = delivery_labels

        self.get_logger().info("\n=== Final Cluster Assignment ===")
        for robot, clusters in self.robot_cluster_map.items():
            self.get_logger().info(f"{robot}: {clusters}")
        
        self.get_logger().info("\n=== Cluster Task and Delivery Information ===")
        for cluster_index in range(len(self.cluster_points)):
            self.get_logger().info(f"Cluster {cluster_index}:")
            self.get_logger().info(f"  Task labels: {self.cluster_task_labels.get(cluster_index, [])}")
            self.get_logger().info(f"  Delivery labels: {self.cluster_delivery_labels.get(cluster_index, [])}")


    def generate_task_sequences(self, robot_names=None):
        """Step 2: Use MILP to optimize task sequence within each cluster"""
        self.robot_task_sequence = getattr(self, 'robot_task_sequence', {})
        self.robot_route_labels = getattr(self, 'robot_route_labels', {})

        if robot_names is None:
            robot_names = self.robot_names

        for robot_name in robot_names:
            task_points = self.robot_cluster_map.get(robot_name, [])
            if not task_points:
                self.robot_task_sequence[robot_name] = []
                continue
            
            # Get robot information
            robot_index = self.robot_names.index(robot_name)
            robot_start = self.robot_poses[robot_index]
            
            # Get cluster index for this robot
            cluster_index = self.robot_cluster_indices.get(robot_name, -1)
            if cluster_index == -1:
                self.robot_task_sequence[robot_name] = []
                continue
            
            # Get task labels and delivery labels for this cluster
            task_labels = self.cluster_task_labels.get(cluster_index, [])
            delivery_labels = self.cluster_delivery_labels.get(cluster_index, [])
            
            if not task_labels:
                self.robot_task_sequence[robot_name] = []
                continue
            
            # Separate pickup and delivery points/labels
            pickup_points = []
            pickup_labels = []
            delivery_points = []
            delivery_labels_unique = []
            
            # Get unique delivery labels
            unique_deliveries = set()
            for delivery in delivery_labels:
                if delivery is not None:
                    unique_deliveries.add(delivery)
            
            # Find delivery points for unique delivery labels
            delivery_point_map = {}
            for delivery_label in unique_deliveries:
                # Find the delivery point coordinates (you may need to add this to your data)
                # For now, we'll use a placeholder - you should add actual delivery point coordinates
                delivery_point_map[delivery_label] = self._get_delivery_point(delivery_label)
            
            # Separate pickup tasks and their corresponding delivery points
            for i, (point, task_label) in enumerate(zip(task_points, task_labels)):
                pickup_points.append(point)
                pickup_labels.append(task_label)
                
                # Get corresponding delivery
                if task_label in self.task_to_delivery:
                    delivery_label = self.task_to_delivery[task_label]
                    if delivery_label in delivery_point_map:
                        delivery_points.append(delivery_point_map[delivery_label])
                        delivery_labels_unique.append(delivery_label)
                    else:
                        # If no delivery point found, skip this pickup
                        pickup_points.pop()
                        pickup_labels.pop()
                else:
                    # If no delivery mapping, skip this pickup
                    pickup_points.pop()
                    pickup_labels.pop()
            
            # Remove duplicates from delivery points and labels
            unique_delivery_data = {}
            for i, (point, label) in enumerate(zip(delivery_points, delivery_labels_unique)):
                if label not in unique_delivery_data:
                    unique_delivery_data[label] = point
            
            delivery_points = list(unique_delivery_data.values())
            delivery_labels_unique = list(unique_delivery_data.keys())
            
            # Create pickup_to_delivery mapping
            pickup_to_delivery = {}
            for pickup_label in pickup_labels:
                if pickup_label in self.task_to_delivery:
                    pickup_to_delivery[pickup_label] = self.task_to_delivery[pickup_label]
            
            self.get_logger().info(f"\n=== MILP Planning for {robot_name} ===")
            self.get_logger().info(f"Robot start: {robot_start}")
            self.get_logger().info(f"Pickup points: {pickup_points}")
            self.get_logger().info(f"Pickup labels: {pickup_labels}")
            self.get_logger().info(f"Delivery points: {delivery_points}")
            self.get_logger().info(f"Delivery labels: {delivery_labels_unique}")
            self.get_logger().info(f"Pickup to delivery mapping: {pickup_to_delivery}")
            
            # Use MILP to plan optimal task sequence within cluster
            try:
                chosen_pickups, chosen_deliveries, route_labels, total_cost = self.cluster_planner.plan_cluster_tasks(
                    robot_start=robot_start,
                    pickup_points=pickup_points,
                    pickup_labels=pickup_labels,
                    delivery_points=delivery_points,
                    delivery_labels=delivery_labels_unique,
                    pickup_to_delivery=pickup_to_delivery,
                    robot_capacity=3  # Default capacity
                )
                
                # Check if MILP failed (chosen_pickups is empty)
                if not chosen_pickups:
                    self.get_logger().warn(f"MILP failed for {robot_name}, using nearest neighbor fallback")
                    self.robot_task_sequence[robot_name], self.robot_route_labels[robot_name] = self._nearest_neighbor_fallback(
                        robot_start, task_points, task_labels
                    )
                else:
                    # Convert route labels to task indices
                    task_indices = []
                    for label in route_labels:
                        if label in self.label_to_index:
                            task_indices.append(self.label_to_index[label])
                    
                    self.robot_task_sequence[robot_name] = task_indices
                    self.robot_route_labels[robot_name] = route_labels
                    
                    self.get_logger().info(f"Chosen pickups: {chosen_pickups}")
                    self.get_logger().info(f"Chosen deliveries: {chosen_deliveries}")
                    self.get_logger().info(f"Optimal route: {route_labels}")
                    self.get_logger().info(f"Task indices: {task_indices}")
                    self.get_logger().info(f"Total cost: {total_cost:.3f}")
                
            except Exception as e:
                self.get_logger().error(f"Error in MILP planning for {robot_name}: {e}")
                # Fallback to nearest neighbor
                self.robot_task_sequence[robot_name], self.robot_route_labels[robot_name] = self._nearest_neighbor_fallback(
                    robot_start, task_points, task_labels
                )

        self.get_logger().info("\n=== Final Task Execution Sequences (by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            self.get_logger().info(f"{robot}: {seq}")

        for tasks in self.robot_task_sequence.values():
            for idx in tasks:
                # idx 是任务的 label index，需要反查 label 和坐标
                for pt, label in self.points_with_label.items():
                    if self.label_to_index[label] == idx:
                        self.assigned_points_global.add(pt)
                        break

        self.get_logger().info(f"assigned_points_global: {self.assigned_points_global}")

    def _get_delivery_point(self, delivery_label):
        """Get delivery point coordinates for a given delivery label"""
        # You need to add actual delivery point coordinates here
        # For now, using placeholder coordinates based on delivery label
        delivery_points = {
            'b': (10, 10),  # Delivery point for group b
            'c': (15, 15),  # Delivery point for group c
            'd': (20, 20),  # Delivery point for group d
            'e': (25, 25),  # Delivery point for group e
        }
        return delivery_points.get(delivery_label, (0, 0))

    def _nearest_neighbor_fallback(self, robot_start, task_points, task_labels):
        """Fallback method using nearest neighbor when MILP fails, with delivery points grouped after pickups."""
        current_pos = robot_start
        remaining = list(range(len(task_points)))
        pickup_sequence = []

        # 1. 先对pickup点做最近邻排序
        while remaining:
            nearest_idx = min(remaining, key=lambda i: np.linalg.norm(np.array(current_pos) - np.array(task_points[i])))
            pickup_sequence.append(nearest_idx)
            current_pos = task_points[nearest_idx]
            remaining.remove(nearest_idx)

        # 2. 先所有pickup，再所有delivery
        pickup_labels_ordered = [task_labels[idx] for idx in pickup_sequence]
        delivery_labels_ordered = [self.task_to_delivery[label] for label in pickup_labels_ordered if label in self.task_to_delivery]
        full_labels = pickup_labels_ordered + delivery_labels_ordered

        # 3. 转为task indices
        task_indices = [self.label_to_index[label] for label in full_labels if label in self.label_to_index]
        self.get_logger().info(f"task_indices: {task_indices}")
        return task_indices, full_labels

    def publish_task_assignments(self):
        for idx, robot in enumerate(self.robot_names):
            msg = ClusterTaskassign()
            msg.robot_id = idx + 1
            msg.task_sequence = self.robot_task_sequence.get(robot, [])
            msg.route_labels = self.robot_route_labels.get(robot, [])
            self.task_pubs[robot].publish(msg)
            self.get_logger().info(f"Published task sequence to {robot}: {msg.task_sequence}")
            self.get_logger().info(f"Published route labels to {robot}: {msg.route_labels}")
        # 每次发布后，重新聚类未分配的任务
        self.update_clusters_after_assignment()

    def publish_task_reassignments(self):
        # Publish task assignment only for the newly assigned robot.
        if hasattr(self, "last_assigned_robot"):
            robot = self.last_assigned_robot
            idx = self.robot_names.index(robot)
            msg = ClusterTaskassign()
            msg.robot_id = idx + 1
            msg.task_sequence = self.robot_task_sequence.get(robot, [])
            msg.route_labels = self.robot_route_labels.get(robot, [])
            self.task_pubs[robot].publish(msg)
            self.get_logger().info(f"Published task sequence to {robot}: {msg.task_sequence}")
            self.get_logger().info(f"Published route labels to {robot}: {msg.route_labels}")
            # 每次发布后，重新聚类未分配的任务
            self.update_clusters_after_assignment()
        else:
            self.get_logger().warn("No new assignment available to publish.")

    def update_clusters_after_assignment(self):
        # 获取所有未分配的任务点
        all_points = list(self.points_with_label.keys())
        # 未分配的点
        unassigned_points = [pt for pt in all_points if pt not in self.assigned_points_global]
        unassigned_labels = [self.points_with_label[pt] for pt in unassigned_points]
        self.get_logger().info(f"unassigned_points: {unassigned_points}")
        # 重新聚类
        if unassigned_points:
            # 只聚类未分配的点，传入label和坐标
            self.cluster_centers, self.cluster_points = self.clusterer.cluster(
                task_points=unassigned_labels,
                task_coords=unassigned_points
            )
            # 重新计算cluster类型和gamma
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
                if normal_count > 0 and special_count > 0:
                    cluster_type = 'hybrid'
                elif special_count > 0:
                    cluster_type = 'special'
                else:
                    cluster_type = 'normal'
                self.cluster_types.append(cluster_type)
            self.valid_cluster = [1] * len(self.cluster_centers)
        else:
            self.cluster_centers = []
            self.cluster_points = []
            self.cluster_gamma = []
            self.cluster_types = []
            self.valid_cluster = []

def main(args=None):
    rclpy.init(args=args)
    try:
        task_assign_node = TaskAssignNode()
        rclpy.spin(task_assign_node)
    except KeyboardInterrupt:
        task_assign_node.get_logger().info("KeyboardInterrupt detected, shutting down...")
    finally:
        task_assign_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
