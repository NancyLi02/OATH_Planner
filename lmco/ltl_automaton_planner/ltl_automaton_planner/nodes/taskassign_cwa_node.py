import os
import numpy as np
import rclpy
from rclpy.node import Node
from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_planner.MILP import ClusterTaskPlanner
from ltl_automaton_msgs.msg import ClusterTaskassign, RobotID, ClusterRequest, AgentFailTask, TaskFail
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

# -------------------- CBAA Algorithm --------------------
class CWA:
    def __init__(self):
        pass

    def select_cluster(self, scores_list, y, robot_index, robot_types):
        """
        Perform auction based on each robot's scores and task types:
          - For normal robots, the score for special tasks is always 0.
          - If a normal robot selects a special task, return -1.
        """
        robot_scores = scores_list[robot_index]
        valid_task = [(1 if score > y_value else 0) for score, y_value in zip(robot_scores, y)]
        
        if sum(valid_task) > 0:
            valid_indices = [i for i, valid in enumerate(valid_task) if valid == 1]
            # If the robot is normal, filter out special tasks.
            if robot_types[robot_index] == 'normal':
                valid_indices = [i for i in valid_indices if task_types[i] != 'special']
            if not valid_indices:
                return -1
            best_task_index = max(valid_indices, key=lambda idx: robot_scores[idx])
            best_task_score = robot_scores[best_task_index]
            y[best_task_index] = best_task_score
            print(f"Robot {robot_index + 1}: Selected task index {best_task_index} with score {best_task_score:.2f}")
            return best_task_index
        else:
            return -1

    def conflict_resolve(self, cluster_index, assigned_clusters, x):
        for robot, assigned_cluster in assigned_clusters:
            if assigned_cluster == cluster_index:
                print(f"Conflict detected for cluster {cluster_index}. Removing previous assignment for Robot {robot + 1}.")
                x[robot][cluster_index] = 0
                assigned_clusters.remove((robot, assigned_cluster))
                break
        return x, assigned_clusters

    def cluster_assignment(self, scores_list, robot_types):
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
                    cluster_index = self.select_cluster(scores_list, y, robot_index, robot_types)
                    if cluster_index != -1:
                        x, assigned_clusters = self.conflict_resolve(cluster_index, assigned_clusters, x)
                        x[robot_index][cluster_index] = 1
                        assigned_clusters.append((robot_index, cluster_index))
                    else:
                        unassigned_robots.append(robot_index)

        print("\nFinal y list:", y)
        print("x list (cluster assignment status for each robot):")
        for robot_index, cluster_assignments in enumerate(x):
            print(f"Robot {robot_index + 1}: {cluster_assignments}")
        print(f"\nUnassigned Robots: {unassigned_robots}")

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
            (11, 9)    # robot4 (normal robot)
        ]

        # Define robot types: only robot2 and robot3 are 'special'; others are 'normal'
        self.robot_types = []
        for i in range(self.robot_count):
            if i in [1, 2]:
                self.robot_types.append('special')
            else:
                self.robot_types.append('normal')

        # Publishers for each robot namespace
        self.task_pubs = {}
        for idx, robot in enumerate(self.robot_names):
            topic = f"/{robot}/ClusterTaskassign"
            self.task_pubs[robot] = self.create_publisher(ClusterTaskassign, topic, 10)

        # Subscribers for finish_building_auto topic from each robot
        self.finished_robots = set()
        self.finish_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/finish_building_auto"
            sub = self.create_subscription(RobotID, topic, self.finish_callback, 10)
            self.finish_subs.append(sub)

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

        points_with_label = {
            (1, 4): 'bb', 
            (6, 6): 'cb',
            (1, 6.5): 'db', 
            (5.5, 9.5): 'eb',
            (9, 6.5): 'fb',

            (1, 13.5): 'bc',
            (9, 16.5): 'cc',
            (1, 16): 'dc', 
            (6, 13): 'ec',

            (11, 13.5): 'bd',
            (11, 16.5): 'cd',
            (19, 16.5): 'dd',
            (19, 19): 'ed', 
            (15, 19.5): 'fd',

            (11, 4): 'be',
            (19, 6.5): 'ce',
            (16, 3): 'de', 
            (19, 1): 'ee'
        }

        self.points_with_label = {tuple(map(float, k)): v for k, v in points_with_label.items()}
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        # Define special task labels.
        special_labels = {'db', 'eb', 'bb', 'cd', 'fd'}
        self.special_labels = special_labels

        # Load wall data from YAML file and perform task clustering using new CostMapClusterer interface
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=wall_path,
            num_clusters=20,
            halton_points_csv=halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=precomputed_distances_csv
        )
        
        # Get clustering results - now returns (centers, points) instead of separate normal/special
        self.cluster_centers, self.cluster_points = self.clusterer.cluster()
        
        # 统一顺序，直接生成gamma和类型
        self.cluster_gamma = []  # Store γ values for each cluster
        self.cluster_types = []  # Store type: 'normal' or 'special' for each cluster
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
            self.cluster_types.append('special' if special_count > 0 else 'normal')
        print(f"Cluster γ values: {self.cluster_gamma}")
        print(f"Cluster types: {self.cluster_types}")
        for idx, (center, points) in enumerate(zip(self.cluster_centers, self.cluster_points)):
            print(f"Cluster {idx}: center={center}, points={points}")
        
        # 有效性标记
        self.valid_cluster = [1] * len(self.cluster_centers)
        self.robot_cluster_indices = {}
        self.failed_task_owners = {}
        
        # CBAA auction algorithm object.
        self.cbaa_algorithm = CBAA()
        
        # MILP cluster task planner
        self.cluster_planner = ClusterTaskPlanner()

    def task_fail_callback(self, msg):  # Called when a task failure is reported
        pass

    def agent_fail_callback(self, msg):
        pass

    def assign_new_cluster(self, msg):
        self.get_logger().info(f"Received new cluster assign request from Robot{msg.robot_id}.")
        robot_id = msg.robot_id  # robot_id is an integer (1,2,3,4)
        robot_index = robot_id - 1
        robot_name = self.robot_names[robot_index]
        
        # Initialize robot_cluster_indices if it does not exist.
        if not hasattr(self, 'robot_cluster_indices'):
            self.robot_cluster_indices = {}

        # Update valid_cluster: set the previous cluster assignment for this robot to 0.
        if self.robot_cluster_indices.get(robot_name, -1) != -1:
            finished_cluster_index = self.robot_cluster_indices[robot_name]
            self.get_logger().info(f"Updating valid_cluster: setting cluster {finished_cluster_index} to 0 for {robot_name}.")
            self.valid_cluster[finished_cluster_index] = 0

        # Update the robot's position using the position information from the message.
        # msg.position is already in float64[] format.
        new_position = msg.position
        self.robot_poses[robot_index] = new_position
        self.get_logger().info(f"Updated position for {robot_name} to {new_position}.")

        # Prepare clustering information.
        all_clusters = self.cluster_points
        task_types = self.cluster_types
        # Compute the score for the requested robot only with γ adjustment.
        scores = []

         # If this robot has broken down, give zero score for all clusters
        if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
            scores = [0.0] * len(all_clusters)
        else:
            for j, (points, t_type) in enumerate(zip(all_clusters, task_types)):
                center = self.cluster_centers[j]
                # If this is a failed-task cluster owned by this robot, score = 0
                if self.failed_task_owners.get(j) == robot_id:
                    scores.append(0.0)
                    continue
                # If the cluster is invalid (already assigned), set the score to 0.
                if self.valid_cluster[j] == 0:
                    scores.append(0.0)
                    continue
                
                dist = np.linalg.norm(np.array(new_position) - np.array(center))
                base_score = dist
                
                # Get γ value for this cluster
                gamma_param = self.cluster_gamma[j]
                
                # 统一公式：normal robot用乘法，special robot用除法
                if self.robot_types[robot_index] == 'normal':
                    score = base_score * gamma_param
                else:  # special robot
                    if gamma_param > 0:
                        score = base_score / gamma_param
                    else:
                        score = base_score  # 当γ=0时使用base_score
                
                scores.append(score)

        # Select the best candidate index for this robot.
        best_index = -1
        best_score = 0.0
        for j, sc in enumerate(scores):
            if sc > best_score:
                best_score = sc
                best_index = j

        if best_index == -1 or best_score == 0.0:
            self.robot_cluster_map[robot_name] = []
            self.robot_cluster_indices[robot_name] = -1
            self.get_logger().info(f"No valid cluster found for {robot_name}. Assignment is empty.")
        else:
            self.robot_cluster_map[robot_name] = all_clusters[best_index]
            self.robot_cluster_indices[robot_name] = best_index
            # Mark the selected cluster as used by updating its valid_cluster flag to 0.
            self.valid_cluster[best_index] = 0
            self.get_logger().info(f"Assigned cluster {best_index} with score {best_score:.2f} to {robot_name}.")

        # Record this robot as the one that was just re-assigned.
        self.last_assigned_robot = robot_name

        # Generate task sequences and publish re-assignment only for the updated robot.
        self.generate_task_sequences()
        self.publish_task_reassignments()

    def finish_callback(self, msg):
        self.get_logger().info(f"Received finish_building_auto from {msg.robot_id}")
        self.finished_robots.add(msg.robot_id)
        # When all robots have finished building, start task assignment.
        if all(name in self.finished_robots for name in self.robot_names):
            self.get_logger().info("All robots finished building, starting task assignment...")
            self.init_task_assign()
            self.generate_task_sequences()
            self.publish_task_assignments()

    def init_task_assign(self):
        """Step 1: Perform cluster-based auction assignment"""
        # Combine all cluster centers and types
        self.all_clusters = self.cluster_centers
        self.task_types = self.cluster_types
        # Compute each robot's score for every task center with γ adjustment.
        scores = []
        for i, robot_pose in enumerate(self.robot_poses):
            robot_name = self.robot_names[i]
            if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
                scores.append([0.0] * len(self.all_clusters))
                continue
            robot_scores = []
            for j, (center, t_type) in enumerate(zip(self.cluster_centers, self.cluster_types)):
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

        # Call the auction algorithm for initial task assignment.
        assigned, _ = self.cbaa_algorithm.initial_task_assignment(scores, self.robot_types, self.task_types)

        # Map the auction results with the clustering data:
        self.robot_cluster_map = {name: [] for name in self.robot_names}
        for robot_index, task_index in assigned:
            robot_name = self.robot_names[robot_index]
            if task_index == -1:
                # If a normal robot wins a special task (or no valid task), mark assignment as empty.
                self.robot_cluster_map[robot_name] = []
            else:
                self.robot_cluster_map[robot_name] = self.cluster_points[task_index]
                # Update valid_cluster: mark the assigned cluster as used by setting its valid_cluster flag to 0.
                self.valid_cluster[task_index] = 0

        print("\n=== Final Cluster Assignment ===")
        for robot, tasks in self.robot_cluster_map.items():
            print(f"{robot}: {tasks}")

    def generate_task_sequences(self):
        """Step 2: Use MILP to optimize task sequence within each cluster"""
        self.robot_task_sequence = {}
        
        for robot_name, task_points in self.robot_cluster_map.items():
            if not task_points:
                self.robot_task_sequence[robot_name] = []
                continue
            
            # Get robot information
            robot_index = self.robot_names.index(robot_name)
            robot_start = self.robot_poses[robot_index]
            robot_type = self.robot_types[robot_index]
            
            # Prepare task information
            task_labels = []
            task_types = []
            for point in task_points:
                label = self.points_with_label.get(tuple(map(float, point)))
                task_labels.append(label)
                task_types.append('special' if label in self.special_labels else 'normal')
            
            print(f"\n=== MILP Planning for {robot_name} ===")
            print(f"Robot type: {robot_type}")
            print(f"Task points: {task_points}")
            print(f"Task labels: {task_labels}")
            print(f"Task types: {task_types}")
            
            # Use MILP to plan optimal task sequence within cluster
            try:
                task_sequence, cost, info = self.cluster_planner.plan_cluster_tasks(
                    robot_start=robot_start,
                    task_points=task_points,
                    task_labels=task_labels,
                    task_types=task_types,
                    robot_type=robot_type,
                    max_tasks=len(task_points)
                )
                
                # Convert to task indices
                task_indices = [self.label_to_index[task_labels[i]] for i in task_sequence]
                self.robot_task_sequence[robot_name] = task_indices
                
                print(f"Optimal task sequence: {task_sequence}")
                print(f"Task indices: {task_indices}")
                print(f"Total cost: {cost:.3f}")
                print(f"Model status: {info['model_status']}")
                
            except Exception as e:
                print(f"Error in MILP planning for {robot_name}: {e}")
                # Fallback to nearest neighbor
                self.robot_task_sequence[robot_name] = self._nearest_neighbor_fallback(
                    robot_start, task_points, task_labels
                )

        print("\n=== Final Task Execution Sequences (by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            print(f"{robot}: {seq}")

    def _nearest_neighbor_fallback(self, robot_start, task_points, task_labels):
        """Fallback method using nearest neighbor when MILP fails"""
        current_pos = robot_start
        remaining = list(range(len(task_points)))
        sequence = []
        
        while remaining:
            # Find nearest task
            nearest_idx = min(remaining, 
                            key=lambda i: np.linalg.norm(np.array(current_pos) - np.array(task_points[i])))
            sequence.append(nearest_idx)
            current_pos = task_points[nearest_idx]
            remaining.remove(nearest_idx)
        
        # Convert to task indices
        task_indices = [self.label_to_index[task_labels[i]] for i in sequence]
        return task_indices

    def publish_task_assignments(self):
        for idx, robot in enumerate(self.robot_names):
            msg = ClusterTaskassign()
            msg.robot_id = idx + 1
            msg.task_sequence = self.robot_task_sequence.get(robot, [])
            self.task_pubs[robot].publish(msg)
            print(f"Published task sequence to {robot}: {msg.task_sequence}")
    
    def publish_task_reassignments(self):
        # Publish task assignment only for the newly assigned robot.
        if hasattr(self, "last_assigned_robot"):
            robot = self.last_assigned_robot
            idx = self.robot_names.index(robot)
            msg = ClusterTaskassign()
            msg.robot_id = idx + 1
            msg.task_sequence = self.robot_task_sequence.get(robot, [])
            self.task_pubs[robot].publish(msg)
            print(f"Published task sequence to {robot}: {msg.task_sequence}")
        else:
            self.get_logger().warn("No new assignment available to publish.")

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
