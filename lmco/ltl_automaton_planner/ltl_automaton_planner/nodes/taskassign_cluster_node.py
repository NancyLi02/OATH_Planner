import os
import numpy as np
import time
import json
import rclpy
from rclpy.node import Node
from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_planner.MILP import ClusterTaskPlanner
from ltl_automaton_msgs.msg import ClusterTaskassign, RobotID, ClusterRequest, AgentFailTask, TaskFail, AddTask, NoTask, ChangeTaskPriority
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import yaml
import re
from ament_index_python.packages import get_package_share_directory
import array

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
        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        robot_positions_dict = yaml_data.get('robot_positions', {})
        self.robot_names = list(robot_positions_dict.keys())
        self.robot_poses = []
        for coord_str in robot_positions_dict.values():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                self.robot_poses.append((x, y))
        self.robot_count = len(self.robot_names)
        # Define robot types: special/normal from yaml
        special_robot_list = yaml_data.get('special_robot', [])
        self.robot_types = ['special' if name in special_robot_list else 'normal' for name in self.robot_names]

        # Publishers for each robot namespace
        self.task_pubs = {}
        self.no_task_pubs = {}
        for idx, robot in enumerate(self.robot_names):
            topic = f"/{robot}/ClusterTaskassign"
            self.task_pubs[robot] = self.create_publisher(ClusterTaskassign, topic, 10)
            # NoTask publisher with namespace
            no_task_topic = f"/{robot}/no_task"
            self.no_task_pubs[robot] = self.create_publisher(NoTask, no_task_topic, 10)
        
        # Publisher for immediate robot state update (to showmove_node)
        self.robot_state_update_pub = self.create_publisher(String, '/robot_state_update', 10)

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

        self.llm_command_parser_subs = self.create_subscription(
            AddTask,
            '/add_task',
            self.add_task_callback,
            10
        )
        self.get_logger().info("Successfully created subscription to /add_task topic")

        self.change_task_priority_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/change_task_priority"
            sub = self.create_subscription(ChangeTaskPriority, topic, self.change_task_priority_callback, 10)
            self.change_task_priority_subs.append(sub)
        
        self.global_change_task_priority_sub = self.create_subscription(
            ChangeTaskPriority,
            '/change_task_priority',
            self.change_task_priority_callback,
            10
        )
        self.get_logger().info("Successfully created subscriptions to change_task_priority topics")

        # Timing data publisher
        self.timing_pub = self.create_publisher(String, '/timing_data', 10)

        self.broke_agents = []
        
        self.task_priorities = {}  # task_label -> 'high'/'low'/'normal'
        self.high_priority_pending_tasks = set()  # Tasks with high priority waiting to be assigned
        
        # Track robots in no_task state with their positions
        self.no_task_robots = {}  # robot_name -> (x, y) position
        
        # Publisher for global completion (all robots finished)
        self.all_robots_finished_pub = self.create_publisher(String, '/all_robots_finished', 10)
        self.all_finished_published = False  # Flag to avoid duplicate publishing
        
        # Task assignment timing tracking
        self.total_task_assignment_time = 0.0  # Cumulative time spent on task assignment
        self.task_assignment_count = 0  # Number of task assignment operations

        # ----- Load wall and task info -----
        package_share = get_package_share_directory('ltl_automaton_planner')
        self.wall_path = os.path.join(package_share, 'config', 'wall.yaml')
        self.halton_points_csv = os.path.join(package_share, 'ltl_automaton_planner', 'all_points_in_Halton.csv')
        self.precomputed_distances_csv = os.path.join(package_share, 'ltl_automaton_planner', 'multi_source_dijkstra_distances.csv')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        points_with_label = {}
        for k, v in yaml_data['task_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
        self.points_with_label = points_with_label
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        self.special_labels = set(yaml_data.get('special_labels', []))

        task_to_delivery = {}
        for group, label_list in yaml_data['task_to_delivery'].items():
            for label in label_list:
                task_to_delivery[label] = group
        self.task_to_delivery = task_to_delivery

        # Load wall data from YAML file and perform task clustering using new CostMapClusterer interface
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=self.wall_path,
            num_clusters=4,
            halton_points_csv=self.halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=self.precomputed_distances_csv
        )
        
        # Get clustering results - now returns (centers, points) instead of separate normal/special
        self.cluster_centers, self.cluster_points = self.clusterer.cluster()
        
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

    
        self.valid_cluster = [1] * len(self.cluster_centers)
        self.robot_cluster_indices = {}
        self.failed_task_owners = {}
        
        # CWA auction algorithm object.
        self.cwa_algorithm = CWA()
        
        # MILP cluster task planner
        self.cluster_planner = ClusterTaskPlanner(
            dijkstra_distances_csv_path=self.precomputed_distances_csv
        )

        self.assigned_points_global = set()

        self.get_logger().info("TaskAssignNode initialization completed successfully")
        self.finish_callback()

    def add_task_callback(self, msg):
        # Record response start time (monotonic for accurate duration)
        response_start_mono = time.monotonic()
        response_start_time = time.time()
        
        self.get_logger().info("=== ADD_TASK_CALLBACK TRIGGERED ===")
        self.get_logger().info(f"Received add task command from LLM, new {msg.task_type} task appears at {msg.location}, assignning new task......")
        self.add_task(msg.location, msg.task_type, msg.task_label, msg.delivery_point)
        
        # Record response end time and publish timing (monotonic for accurate duration)
        response_end_mono = time.monotonic()
        response_duration = response_end_mono - response_start_mono
        self.get_logger().info(f'[TIMING] add_task response completed. Duration: {response_duration:.4f}s')
        
        timing_msg = String()
        timing_msg.data = json.dumps({
            'type': 'system_response',
            'command': 'add_task',
            'task_label': msg.task_label,
            'task_type': msg.task_type,
            'location': list(msg.location),
            'start_time': response_start_time,
            'end_time': time.time(),
            'duration': response_duration,
            'timestamp': time.time()
        })
        self.timing_pub.publish(timing_msg)

    def add_task(self, location, task_type, task_label, delivery_point):
        self.get_logger().info(f"Adding task {task_label} at {location} with delivery point {delivery_point}.")

        if isinstance(location, array.array):
            location = list(location)
        if not (isinstance(location, (list, tuple)) and len(location) == 2):
            self.get_logger().error(f"Invalid location format: {location}")
            return

        if not isinstance(task_label, str) or not task_label:
            self.get_logger().error(f"Invalid task_label: {task_label}")
            return
        if not isinstance(delivery_point, str) or not delivery_point:
            self.get_logger().error(f"Invalid delivery_point: {delivery_point}")
            return
        if not isinstance(task_type, str) or not task_type:
            self.get_logger().error(f"Invalid task_type: {task_type}")
            return

        location_tuple = (float(location[0]), float(location[1]))
        if location_tuple in self.points_with_label:
            self.get_logger().warn(f"Task at {location_tuple} already exists, overwriting label to {task_label}")
        self.points_with_label[location_tuple] = task_label

        # 4. Update special_labels
        if task_type == "special":
            self.special_labels.add(task_label)
        else:
            self.special_labels.discard(task_label)  # Ensure normal tasks are not in special_labels

        # 5. Update task_to_delivery
        self.task_to_delivery[task_label] = delivery_point

        # 6. Update label_to_index
        # Regenerate label_to_index to ensure unique and increasing indices
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        # 7. Re-initialize the clusterer with the updated task list before reclustering
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=self.wall_path,
            num_clusters=4,
            halton_points_csv=self.halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=self.precomputed_distances_csv
        )

        # 8. Immediately run update_clusters_after_assignment
        self.update_clusters_after_assignment()

        self.get_logger().info(f"Task {task_label} added successfully. Current points_with_label: {self.points_with_label}")
        self.get_logger().info(f"Current special_labels: {self.special_labels}")
        self.get_logger().info(f"Current task_to_delivery: {self.task_to_delivery}")
        
        # Check if there are no_task robots that can take this new task
        if self.no_task_robots:
            self.get_logger().info(f"Found {len(self.no_task_robots)} no_task robots. Triggering auction for new task...")
            self.auction_new_task_to_idle_robots(location_tuple, task_label, task_type)

    def publish_robot_state_update(self, robot_name, new_state):
        """
        Immediately publish robot state update to showmove_node.
        This is called right after task assignment, before planner builds automaton.
        """
        state_msg = String()
        state_msg.data = json.dumps({
            'type': 'robot_state_update',
            'robot_id': robot_name,
            'new_state': new_state,  # 'waiting' or 'active'
            'timestamp': time.time()
        })
        self.robot_state_update_pub.publish(state_msg)
        self.get_logger().info(f"📤 Immediately published state update: {robot_name} -> {new_state}")

    def check_all_robots_finished(self):
        """
        Check if all robots are in no_task state.
        If yes, publish global completion message.
        """
        # Count active (non-broken) robots
        active_robots = [name for name in self.robot_names if name not in self.broke_agents]
        
        # Check if all active robots are in no_task state
        all_finished = all(robot in self.no_task_robots for robot in active_robots)
        
        if all_finished and not self.all_finished_published:
            self.get_logger().info("="*60)
            self.get_logger().info("ALL ROBOTS FINISHED - Publishing global completion message")
            self.get_logger().info(f"No task robots: {list(self.no_task_robots.keys())}")
            self.get_logger().info("="*60)
            
            # Output total task assignment time statistics
            self.get_logger().info("")
            self.get_logger().info("="*60)
            self.get_logger().info("TASK ASSIGNMENT TIME STATISTICS")
            self.get_logger().info("="*60)
            self.get_logger().info(f"Total task assignment operations: {self.task_assignment_count}")
            self.get_logger().info(f"Total time spent on task assignment: {self.total_task_assignment_time:.4f} seconds")
            if self.task_assignment_count > 0:
                avg_time = self.total_task_assignment_time / self.task_assignment_count
                self.get_logger().info(f"Average time per assignment: {avg_time:.4f} seconds")
            self.get_logger().info("="*60)
            self.get_logger().info("")
            
            # Publish global completion message
            completion_msg = String()
            completion_msg.data = json.dumps({
                'type': 'all_robots_finished',
                'robots': list(self.no_task_robots.keys()),
                'timestamp': time.time()
            })
            self.all_robots_finished_pub.publish(completion_msg)
            self.all_finished_published = True
            
            # Also publish to timing_data topic for logging
            timing_msg = String()
            timing_msg.data = json.dumps({
                'type': 'global_completion',
                'all_robots_finished': True,
                'robot_count': len(active_robots),
                'total_task_assignment_time': self.total_task_assignment_time,
                'task_assignment_count': self.task_assignment_count,
                'timestamp': time.time()
            })
            self.timing_pub.publish(timing_msg)
        elif not all_finished:
            # Reset the flag if not all robots are finished (some robot got new task)
            self.all_finished_published = False

    def auction_new_task_to_idle_robots(self, task_location, task_label, task_type):
        """
        Auction a new task to idle (no_task) robots based on distance.
        The closest robot that can complete the task wins.
        """
        # Start timing for task assignment
        assignment_start_time = time.monotonic()
        
        self.get_logger().info(f"\n=== AUCTIONING NEW TASK '{task_label}' TO IDLE ROBOTS ===")
        self.get_logger().info(f"Task location: {task_location}, Task type: {task_type}")
        self.get_logger().info(f"Available no_task robots: {list(self.no_task_robots.keys())}")
        
        is_special_task = task_type == 'special' or task_label in self.special_labels
        
        # Calculate distance for each idle robot and filter by capability
        candidates = []
        for robot_name, robot_pos in self.no_task_robots.items():
            robot_index = self.robot_names.index(robot_name)
            robot_type = self.robot_types[robot_index]
            
            # Check if robot is broken
            if robot_name in self.broke_agents:
                self.get_logger().info(f"  {robot_name}: SKIPPED (broken)")
                continue
            
            # Check capability: normal robots cannot do special tasks
            if is_special_task and robot_type == 'normal':
                self.get_logger().info(f"  {robot_name}: SKIPPED (normal robot cannot do special task)")
                continue
            
            # Calculate distance
            dist = np.linalg.norm(np.array(robot_pos) - np.array(task_location))
            candidates.append((robot_name, robot_pos, dist, robot_index))
            self.get_logger().info(f"  {robot_name} at {robot_pos}: distance = {dist:.2f}")
        
        if not candidates:
            self.get_logger().warn("No eligible robots found for new task auction")
            # Record task assignment time even for failed auction
            assignment_duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += assignment_duration
            self.task_assignment_count += 1
            self.get_logger().info(f"[TIMING] Failed auction completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")
            return
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[2])
        winner_name, winner_pos, winner_dist, winner_index = candidates[0]
        
        self.get_logger().info(f"AUCTION WINNER: {winner_name} (distance: {winner_dist:.2f})")
        
        # Remove winner from no_task_robots
        del self.no_task_robots[winner_name]
        
        # IMMEDIATELY publish state update to showmove_node (before planner builds automaton)
        self.publish_robot_state_update(winner_name, 'waiting')
        
        # Create a single-task cluster for this robot
        single_task_cluster = [task_location]
        
        if not hasattr(self, 'robot_cluster_map'):
            self.robot_cluster_map = {}
        if not hasattr(self, 'robot_cluster_indices'):
            self.robot_cluster_indices = {}
        
        self.robot_cluster_map[winner_name] = single_task_cluster
        
        # Use a unique high index for this new task cluster
        new_task_cluster_idx = len(getattr(self, 'cluster_centers', [])) + 2000 + len(self.assigned_points_global)
        self.robot_cluster_indices[winner_name] = new_task_cluster_idx
        
        # Update cluster task labels and delivery labels
        if not hasattr(self, 'cluster_task_labels'):
            self.cluster_task_labels = {}
        if not hasattr(self, 'cluster_delivery_labels'):
            self.cluster_delivery_labels = {}
        
        self.cluster_task_labels[new_task_cluster_idx] = [task_label]
        delivery_label = self.task_to_delivery.get(task_label, None)
        self.cluster_delivery_labels[new_task_cluster_idx] = [delivery_label] if delivery_label else [None]
        
        # Update robot pose for planning
        self.robot_poses[winner_index] = winner_pos
        
        self.last_assigned_robot = winner_name
        
        # Generate task sequence and publish
        self.generate_task_sequences(robot_names=[winner_name])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()
        
        # Reset all_finished state since a robot is now active again
        self.check_all_robots_finished()
        
        # Record task assignment time
        assignment_duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += assignment_duration
        self.task_assignment_count += 1
        self.get_logger().info(f"[TIMING] New task auction for {winner_name} completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")
        
        self.get_logger().info(f"Successfully assigned new task '{task_label}' to {winner_name}")

    def change_task_priority_callback(self, msg):
        # Record response start time (monotonic for accurate duration)
        response_start_mono = time.monotonic()
        response_start_time = time.time()
        
        task_label = msg.task_label
        priority = msg.priority
        
        self.get_logger().info(f"=== CHANGE_TASK_PRIORITY_CALLBACK TRIGGERED ===")
        self.get_logger().info(f"Received priority change request: task '{task_label}' to priority '{priority}'")
        
        if task_label not in [label for label in self.points_with_label.values()]:
            self.get_logger().error(f"Task label '{task_label}' not found in current tasks")
            return
        
        if priority not in ['high', 'low', 'normal']:
            self.get_logger().error(f"Invalid priority '{priority}'. Must be 'high', 'low', or 'normal'")
            return
        
        old_priority = self.task_priorities.get(task_label, 'normal')
        
        if old_priority == 'high' and priority == 'high':
            self.get_logger().info(f"Task '{task_label}' is already high priority, skipping duplicate processing")
            return
        
        self.task_priorities[task_label] = priority
        
        self.get_logger().info(f"Updated task '{task_label}' priority from '{old_priority}' to '{priority}'")
        self.get_logger().info(f"Current task priorities: {self.task_priorities}")
        
        if priority == 'high':
            # Add to high priority pending set - will be assigned to next requesting robot
            self.high_priority_pending_tasks.add(task_label)
            self.get_logger().info(f"Task '{task_label}' added to high priority pending queue. Will be assigned to next compatible robot.")
            self.get_logger().info(f"Current high priority pending tasks: {self.high_priority_pending_tasks}")
        else:
            # Remove from high priority pending if it was there
            self.high_priority_pending_tasks.discard(task_label)
            self.handle_priority_change_reassignment(task_label, priority)
        
        # Record response end time and publish timing (monotonic for accurate duration)
        response_end_mono = time.monotonic()
        response_duration = response_end_mono - response_start_mono
        self.get_logger().info(f'[TIMING] change_task_priority response completed. Duration: {response_duration:.4f}s')
        
        timing_msg = String()
        timing_msg.data = json.dumps({
            'type': 'system_response',
            'command': 'change_task_priority',
            'task_label': task_label,
            'old_priority': old_priority,
            'new_priority': priority,
            'start_time': response_start_time,
            'end_time': time.time(),
            'duration': response_duration,
            'timestamp': time.time()
        })
        self.timing_pub.publish(timing_msg)

    def handle_priority_change_reassignment(self, task_label, priority):
        affected_clusters = []
        
        if hasattr(self, 'cluster_task_labels'):
            for cluster_idx, task_labels in self.cluster_task_labels.items():
                if task_label in task_labels:
                    affected_clusters.append(cluster_idx)
        
        if affected_clusters:
            self.get_logger().info(f"Task '{task_label}' found in clusters: {affected_clusters}")
        else:
            self.get_logger().info(f"Task '{task_label}' not currently in any assigned cluster")

    def find_available_robot(self):

        for robot_name in self.robot_names:
            if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
                continue
            
            current_tasks = getattr(self, 'robot_task_sequence', {}).get(robot_name, [])
            if not current_tasks:
                self.get_logger().info(f"Found available robot with no current tasks: {robot_name}")
                return robot_name
        
        min_tasks = float('inf')
        best_robot = None
        for robot_name in self.robot_names:
            if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
                continue
            
            current_tasks = getattr(self, 'robot_task_sequence', {}).get(robot_name, [])
            task_count = len(current_tasks)
            if task_count < min_tasks:
                min_tasks = task_count
                best_robot = robot_name
        
        if best_robot:
            self.get_logger().info(f"Found available robot with fewest tasks ({min_tasks}): {best_robot}")
            return best_robot
        
        for robot_name in self.robot_names:
            if not (hasattr(self, 'broke_agents') and robot_name in self.broke_agents):
                self.get_logger().warn(f"All robots busy, assigning to first available: {robot_name}")
                return robot_name
        
        self.get_logger().error("No available robots found!")
        return None

    def remove_task_from_current_assignment(self, task_label, task_coord):

        self.assigned_points_global.discard(task_coord)
        
        task_index = self.label_to_index.get(task_label)
        if task_index:
            for robot_name, task_sequence in getattr(self, 'robot_task_sequence', {}).items():
                if task_index in task_sequence:
                    task_sequence.remove(task_index)
                    self.get_logger().info(f"Removed task {task_label} (index {task_index}) from {robot_name}'s task sequence")
                    break
        
        for robot_name, route_labels in getattr(self, 'robot_route_labels', {}).items():
            if task_label in route_labels:
                route_labels.remove(task_label)
                self.get_logger().info(f"Removed task {task_label} from {robot_name}'s route labels")
                break
        
        for cluster_idx, task_labels in getattr(self, 'cluster_task_labels', {}).items():
            if task_label in task_labels:
                task_labels.remove(task_label)

                if not task_labels and cluster_idx in getattr(self, 'cluster_delivery_labels', {}):
                    del self.cluster_delivery_labels[cluster_idx]
                self.get_logger().info(f"Removed task {task_label} from cluster {cluster_idx}")
                break

    def handle_high_priority_task(self, task_label):
        self.get_logger().info(f"\n=== HANDLING HIGH PRIORITY TASK: {task_label} ===")
        
        for cluster_idx, task_labels in getattr(self, 'cluster_task_labels', {}).items():
            if task_label in task_labels and cluster_idx >= 1000:
                assigned_robot = None
                for robot_name, robot_cluster_idx in getattr(self, 'robot_cluster_indices', {}).items():
                    if robot_cluster_idx == cluster_idx:
                        assigned_robot = robot_name
                        break
                
                self.get_logger().info(f"Task {task_label} is already in high priority cluster {cluster_idx} assigned to {assigned_robot}")
                self.get_logger().info("Skipping duplicate high priority processing")
                return
        
        task_coord = None
        for coord, label in self.points_with_label.items():
            if label == task_label:
                task_coord = coord
                break
        
        if not task_coord:
            self.get_logger().error(f"Could not find coordinates for task {task_label}")
            return
        
        if task_coord in self.assigned_points_global:
            self.get_logger().info(f"Task {task_label} is already assigned, removing from current assignment for high priority handling")
            self.remove_task_from_current_assignment(task_label, task_coord)
        
        available_robot = self.find_available_robot()
        if not available_robot:
            self.get_logger().error(f"No available robot found for high priority task {task_label}")
            return
        
        single_task_cluster = [task_coord]
        cluster_center = task_coord
        
        if not hasattr(self, 'robot_cluster_map'):
            self.robot_cluster_map = {}
        if not hasattr(self, 'robot_cluster_indices'):
            self.robot_cluster_indices = {}
        
        self.robot_cluster_map[available_robot] = single_task_cluster
        
        high_priority_cluster_idx = len(getattr(self, 'cluster_centers', [])) + 1000
        self.robot_cluster_indices[available_robot] = high_priority_cluster_idx
        
        if not hasattr(self, 'cluster_task_labels'):
            self.cluster_task_labels = {}
        if not hasattr(self, 'cluster_delivery_labels'):
            self.cluster_delivery_labels = {}
        
        self.cluster_task_labels[high_priority_cluster_idx] = [task_label]
        delivery_label = self.task_to_delivery.get(task_label, None)
        self.cluster_delivery_labels[high_priority_cluster_idx] = [delivery_label] if delivery_label else [None]
        
        self.get_logger().info(f"Created high priority cluster {high_priority_cluster_idx} for task {task_label}")
        self.get_logger().info(f"Assigned high priority task to robot: {available_robot}")
        
        self.last_assigned_robot = available_robot
        
        self.generate_task_sequences(robot_names=[available_robot])
        
        self.publish_task_reassignments()
        
        self.update_clusters_after_assignment()
        
        self.get_logger().info(f"High priority task {task_label} successfully assigned to {available_robot}")

    def _try_assign_high_priority_task(self, robot_name, robot_type, robot_index):
        """
        Try to assign a high priority task to the requesting robot.
        Returns True if a task was assigned, False otherwise.
        """
        self.get_logger().info(f"\n=== CHECKING HIGH PRIORITY TASKS FOR {robot_name} (type: {robot_type}) ===")
        self.get_logger().info(f"High priority pending tasks: {self.high_priority_pending_tasks}")
        
        # Find a matching high priority task
        for task_label in list(self.high_priority_pending_tasks):
            # Check if task is special and robot can handle it
            is_special_task = task_label in self.special_labels
            
            if is_special_task and robot_type == 'normal':
                self.get_logger().info(f"Task '{task_label}' is special but robot '{robot_name}' is normal, skipping...")
                continue
            
            # Find task coordinates
            task_coord = None
            for coord, label in self.points_with_label.items():
                if label == task_label:
                    task_coord = coord
                    break
            
            if not task_coord:
                self.get_logger().error(f"Could not find coordinates for high priority task '{task_label}'")
                self.high_priority_pending_tasks.discard(task_label)
                continue
            
            # Check if task is already assigned
            if task_coord in self.assigned_points_global:
                self.get_logger().info(f"High priority task '{task_label}' is already assigned, removing from pending...")
                self.high_priority_pending_tasks.discard(task_label)
                continue
            
            # Found a matching task - assign it directly
            self.get_logger().info(f"=== DIRECTLY ASSIGNING HIGH PRIORITY TASK '{task_label}' TO {robot_name} ===")
            
            # Remove from pending set
            self.high_priority_pending_tasks.discard(task_label)
            
            # Create a single-task cluster for this high priority task
            single_task_cluster = [task_coord]
            
            if not hasattr(self, 'robot_cluster_map'):
                self.robot_cluster_map = {}
            if not hasattr(self, 'robot_cluster_indices'):
                self.robot_cluster_indices = {}
            
            self.robot_cluster_map[robot_name] = single_task_cluster
            
            # Use a high index to distinguish high priority clusters
            high_priority_cluster_idx = len(getattr(self, 'cluster_centers', [])) + 1000 + len(self.assigned_points_global)
            self.robot_cluster_indices[robot_name] = high_priority_cluster_idx
            
            if not hasattr(self, 'cluster_task_labels'):
                self.cluster_task_labels = {}
            if not hasattr(self, 'cluster_delivery_labels'):
                self.cluster_delivery_labels = {}
            
            self.cluster_task_labels[high_priority_cluster_idx] = [task_label]
            delivery_label = self.task_to_delivery.get(task_label, None)
            self.cluster_delivery_labels[high_priority_cluster_idx] = [delivery_label] if delivery_label else [None]
            
            self.last_assigned_robot = robot_name
            
            # Generate task sequence and publish
            self.generate_task_sequences(robot_names=[robot_name])
            self.publish_task_reassignments()
            self.update_clusters_after_assignment()
            
            self.get_logger().info(f"Successfully assigned high priority task '{task_label}' to {robot_name}")
            return True
        
        self.get_logger().info(f"No compatible high priority task found for {robot_name}")
        return False

    def apply_priority_adjustment(self, base_score, cluster_index):
        if base_score <= 0:
            return base_score
            
        cluster_task_labels = []
        if hasattr(self, 'cluster_task_labels') and cluster_index in self.cluster_task_labels:
            cluster_task_labels = self.cluster_task_labels[cluster_index]
        elif hasattr(self, 'cluster_points') and cluster_index < len(self.cluster_points):
            for coord in self.cluster_points[cluster_index]:
                for point_coord, label in self.points_with_label.items():
                    if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                        cluster_task_labels.append(label)
                        break
        
        if not cluster_task_labels:
            return base_score
        

        has_high_priority = False
        has_low_priority = False
        
        for task_label in cluster_task_labels:
            priority = self.task_priorities.get(task_label, 'normal')
            if priority == 'high':
                has_high_priority = True
            elif priority == 'low':
                has_low_priority = True
        
        adjusted_score = base_score
        if has_high_priority:
            adjusted_score = base_score / 10.0
            self.get_logger().info(f"Cluster {cluster_index} has high priority tasks, score adjusted: {base_score} -> {adjusted_score}")
        elif has_low_priority:
            adjusted_score = base_score * 10.0 
            self.get_logger().info(f"Cluster {cluster_index} has low priority tasks, score adjusted: {base_score} -> {adjusted_score}")
        
        return adjusted_score

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
        
        # Start timing for task assignment
        assignment_start_time = time.monotonic()

        robot_id = msg.robot_id  # robot_id is an integer (1,2,3,4)
        robot_index = robot_id - 1
        robot_name = self.robot_names[robot_index]
        robot_type = self.robot_types[robot_index]

        # Update the robot's position using the position information from the message.
        new_position = tuple(msg.position)
        self.robot_poses[robot_index] = new_position
        self.get_logger().info(f"Updated position for {robot_name} to {new_position}.")

        # Remove robot from no_task_robots if it was there (robot is requesting a new task)
        if robot_name in self.no_task_robots:
            del self.no_task_robots[robot_name]
            self.get_logger().info(f"Removed {robot_name} from no_task_robots (requesting new task)")

        # Check for high priority tasks first
        if self.high_priority_pending_tasks:
            assigned_high_priority = self._try_assign_high_priority_task(robot_name, robot_type, robot_index)
            if assigned_high_priority:
                # Record task assignment time for high priority task
                assignment_duration = time.monotonic() - assignment_start_time
                self.total_task_assignment_time += assignment_duration
                self.task_assignment_count += 1
                self.get_logger().info(f"[TIMING] High priority task assignment for {robot_name} completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")
                return

        all_points = list(self.points_with_label.keys())
        unassigned_points = [pt for pt in all_points if pt not in self.assigned_points_global]
        if not unassigned_points:
            self.get_logger().info("No unassigned tasks left, publishing NoTask message.")
            
            # Track this robot as no_task with its current position
            self.no_task_robots[robot_name] = new_position
            self.get_logger().info(f"Added {robot_name} to no_task_robots at position {new_position}")
            
            no_task_msg = NoTask()
            no_task_msg.robot_id = robot_name
            self.no_task_pubs[robot_name].publish(no_task_msg)
            self.get_logger().info(f"Published NoTask message to {robot_name}")
            
            # Record task assignment time (even for no-task case)
            assignment_duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += assignment_duration
            self.task_assignment_count += 1
            self.get_logger().info(f"[TIMING] No-task check for {robot_name} completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")
            
            # Check if all robots are now in no_task state
            self.check_all_robots_finished()
            return
        
        # Initialize robot_cluster_indices if it does not exist.
        if not hasattr(self, 'robot_cluster_indices'):
            self.robot_cluster_indices = {}

        # Prepare clustering information.
        all_clusters = self.cluster_points
        cluster_types = self.cluster_types
        self.get_logger().info(f"all_clusters: {all_clusters}")
        self.get_logger().info(f"cluster_types: {cluster_types}")
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
                adjusted_score = self.apply_priority_adjustment(score, j)
                scores.append(adjusted_score)
                self.get_logger().info(f"base score: {score}, adjusted score: {adjusted_score}")

        # Select the best candidate index for this robot.
        best_index = -1
        best_score = float('inf')  # Changed to inf since lower scores are better
        for j, sc in enumerate(scores):
            if sc > 0 and sc < best_score:  # Only consider valid scores (non-zero) and lower is better
                best_score = sc
                best_index = j

        if best_index == -1 or best_score == float('inf'):
            # No valid cluster found - publish NoTask message instead of empty task list
            self.get_logger().info(f"No valid cluster found for {robot_name}. Publishing NoTask message.")
            
            # Track this robot as no_task with its current position
            self.no_task_robots[robot_name] = new_position
            self.get_logger().info(f"Added {robot_name} to no_task_robots at position {new_position}")
            
            no_task_msg = NoTask()
            no_task_msg.robot_id = robot_name
            self.no_task_pubs[robot_name].publish(no_task_msg)
            self.get_logger().info(f"Published NoTask message to {robot_name}")
            
            # Record task assignment time (even for no valid cluster case)
            assignment_duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += assignment_duration
            self.task_assignment_count += 1
            self.get_logger().info(f"[TIMING] No valid cluster check for {robot_name} completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")
            
            # Check if all robots are now in no_task state
            self.check_all_robots_finished()
            return
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

        # Record this robot as the one that was just re-assigned.
        self.last_assigned_robot = robot_name
        # self.get_logger().info(f"robot_cluster_map: {self.robot_cluster_map}")
        self.generate_task_sequences(robot_names=[robot_name])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()
        
        # Record task assignment time
        assignment_duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += assignment_duration
        self.task_assignment_count += 1
        self.get_logger().info(f"[TIMING] Cluster reassignment for {robot_name} completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")

        

    def finish_callback(self):
        # self.get_logger().info(f"Received finish_building_auto from {msg.robot_id}")
        # self.finished_robots.add(msg.robot_id)
        # # When all robots have finished building, start task assignment.
        # if all(name in self.finished_robots for name in self.robot_names):
        self.get_logger().info("Starting task assignment...")
        
        # Start timing for initial task assignment
        assignment_start_time = time.monotonic()
        
        self.init_cluster_assign()
        self.generate_task_sequences()
        self.publish_task_assignments()
        
        # Record task assignment time
        assignment_duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += assignment_duration
        self.task_assignment_count += 1
        self.get_logger().info(f"[TIMING] Initial task assignment completed in {assignment_duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)")

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
                
                if self.robot_types[i] == 'normal':
                    score = base_score * gamma_param
                else:  # special robot
                    if gamma_param > 0:
                        score = base_score / gamma_param
                    else:
                        score = base_score  # 当γ=0时使用base_score
                
                adjusted_score = self.apply_priority_adjustment(score, j)
                robot_scores.append(adjusted_score)
            scores.append(robot_scores)

        # self.get_logger().info(f"scores: {scores}")

        # Call the auction algorithm for initial cluster assignment.
        assigned, _ = self.cwa_algorithm.initial_cluster_assignment(scores, self.robot_types, self.cluster_types)
        # self.get_logger().info(f"assigned: {assigned}")
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

        # self.get_logger().info("\n=== Final Cluster Assignment ===")
        # for robot, clusters in self.robot_cluster_map.items():
        #     self.get_logger().info(f"{robot}: {clusters}")
        
        # self.get_logger().info("\n=== Cluster Task and Delivery Information ===")
        # for cluster_index in range(len(self.cluster_points)):
        #     self.get_logger().info(f"Cluster {cluster_index}:")
        #     self.get_logger().info(f"  Task labels: {self.cluster_task_labels.get(cluster_index, [])}")
        #     self.get_logger().info(f"  Delivery labels: {self.cluster_delivery_labels.get(cluster_index, [])}")


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
                self.robot_route_labels[robot_name] = []  # IMPORTANT: Also clear route_labels!
                continue
            
            # Get robot information
            robot_index = self.robot_names.index(robot_name)
            robot_start = self.robot_poses[robot_index]
            
            # Get cluster index for this robot
            cluster_index = self.robot_cluster_indices.get(robot_name, -1)
            if cluster_index == -1:
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []  # IMPORTANT: Also clear route_labels!
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
            
            # self.get_logger().info(f"\n=== MILP Planning for {robot_name} ===")
            # self.get_logger().info(f"Robot start: {robot_start}")
            # self.get_logger().info(f"Pickup points: {pickup_points}")
            # self.get_logger().info(f"Pickup labels: {pickup_labels}")
            # self.get_logger().info(f"Delivery points: {delivery_points}")
            # self.get_logger().info(f"Delivery labels: {delivery_labels_unique}")
            # self.get_logger().info(f"Pickup to delivery mapping: {pickup_to_delivery}")
            
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
                    
                    nn_route_labels = self.robot_route_labels[robot_name]
                    pickup_labels_from_nn = [label for label in nn_route_labels if label in pickup_labels]
                    selected_nn_points = []
                    for pickup_label in pickup_labels_from_nn:
                        for pt, label in self.points_with_label.items():
                            if label == pickup_label:
                                selected_nn_points.append(pt)
                                self.assigned_points_global.add(pt)
                                break
                    self.get_logger().info(f"NN selected {len(pickup_labels_from_nn)} pickup tasks: {pickup_labels_from_nn}")
                    self.get_logger().info(f"Marked {len(selected_nn_points)} NN selected tasks as assigned")
                else:
                    # Log MILP selection details
                    self.get_logger().info(f"\n=== MILP Selection Details for {robot_name} ===")
                    self.get_logger().info(f"Cluster had {len(pickup_labels)} pickup tasks: {pickup_labels}")
                    self.get_logger().info(f"MILP selected {len(chosen_pickups)} pickups: {chosen_pickups}")
                    unselected_pickups = [label for label in pickup_labels if label not in chosen_pickups]
                    if unselected_pickups:
                        self.get_logger().warn(f"MILP did NOT select these pickups: {unselected_pickups}")
                        self.get_logger().warn(f"These unselected tasks will remain in unassigned list for future allocation")
                    
                    # Convert route labels to task indices
                    task_indices = []
                    for label in route_labels:
                        if label in self.label_to_index:
                            task_indices.append(self.label_to_index[label])
                    
                    self.robot_task_sequence[robot_name] = task_indices
                    self.robot_route_labels[robot_name] = route_labels
                    
                    selected_pickup_points = []
                    for pickup_label in chosen_pickups:
                        for pt, label in self.points_with_label.items():
                            if label == pickup_label:
                                selected_pickup_points.append(pt)
                                self.assigned_points_global.add(pt)
                                break
                    self.get_logger().info(f"Marked {len(selected_pickup_points)} selected pickup tasks as assigned")
                    
                    # self.get_logger().info(f"Chosen pickups: {chosen_pickups}")
                    # self.get_logger().info(f"Chosen deliveries: {chosen_deliveries}")
                    # self.get_logger().info(f"Optimal route: {route_labels}")
                    # self.get_logger().info(f"Task indices: {task_indices}")
                    # self.get_logger().info(f"Total cost: {total_cost:.3f}")
                
            except Exception as e:
                # self.get_logger().error(f"Error in MILP planning for {robot_name}: {e}")
                # Fallback to nearest neighbor
                self.robot_task_sequence[robot_name], self.robot_route_labels[robot_name] = self._nearest_neighbor_fallback(
                    robot_start, task_points, task_labels
                )
                
                fallback_route_labels = self.robot_route_labels[robot_name]
                pickup_labels_from_fallback = [label for label in fallback_route_labels if label in task_labels]
                selected_fallback_points = []
                for pickup_label in pickup_labels_from_fallback:
                    for pt, label in self.points_with_label.items():
                        if label == pickup_label:
                            selected_fallback_points.append(pt)
                            self.assigned_points_global.add(pt)
                            break
                self.get_logger().info(f"Exception fallback selected {len(pickup_labels_from_fallback)} pickup tasks: {pickup_labels_from_fallback}")
                self.get_logger().info(f"Marked {len(selected_fallback_points)} fallback selected tasks as assigned")

        self.get_logger().info("\n=== Final Task Execution Sequences (by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            self.get_logger().info(f"{robot}: {seq}")

        self.log_unassigned_tasks()

        # self.get_logger().info(f"assigned_points_global: {self.assigned_points_global}")

    def log_unassigned_tasks(self):
        """记录所有未分配的任务"""
        all_task_points = set(self.points_with_label.keys())
        unassigned_points = all_task_points - self.assigned_points_global
        
        if unassigned_points:
            unassigned_labels = []
            for pt in unassigned_points:
                label = self.points_with_label.get(pt)
                if label:
                    unassigned_labels.append(label)
            
            self.get_logger().warn(f"\n=== UNASSIGNED TASKS DETECTED ===")
            self.get_logger().warn(f"The number of unassigned tasks: {len(unassigned_points)}")
            self.get_logger().warn(f"The labels of unassigned tasks: {unassigned_labels}")
            self.get_logger().warn(f"The coordinations of unassigned tasks: {list(unassigned_points)}")
            
            for cluster_idx, task_labels in getattr(self, 'cluster_task_labels', {}).items():
                cluster_unassigned = [label for label in task_labels if label in unassigned_labels]
                if cluster_unassigned:
                    assigned_robot = None
                    for robot_name, robot_cluster_idx in getattr(self, 'robot_cluster_indices', {}).items():
                        if robot_cluster_idx == cluster_idx:
                            assigned_robot = robot_name
                            break
                    
                    self.get_logger().warn(f"Cluster {cluster_idx} (is assigned to {assigned_robot}) including: {cluster_unassigned}")
                    self.get_logger().warn(f"The task label in the cluster: {task_labels}")
        else:
            self.get_logger().info("\n=== ALL TASKS ASSIGNED ===")

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

        while remaining:
            nearest_idx = min(remaining, key=lambda i: np.linalg.norm(np.array(current_pos) - np.array(task_points[i])))
            pickup_sequence.append(nearest_idx)
            current_pos = task_points[nearest_idx]
            remaining.remove(nearest_idx)

        pickup_labels_ordered = [task_labels[idx] for idx in pickup_sequence]
        
        delivery_labels_set = set()
        for label in pickup_labels_ordered:
            if label in self.task_to_delivery:
                delivery_labels_set.add(self.task_to_delivery[label])
        
        delivery_labels_ordered = list(delivery_labels_set)
        full_labels = pickup_labels_ordered + delivery_labels_ordered

        # 3. 转为task indices
        task_indices = [self.label_to_index[label] for label in full_labels if label in self.label_to_index]
        # self.get_logger().info(f"pickup_labels_ordered: {pickup_labels_ordered}")
        # self.get_logger().info(f"delivery_labels_ordered (deduplicated): {delivery_labels_ordered}")
        # self.get_logger().info(f"full_labels: {full_labels}")
        # self.get_logger().info(f"task_indices: {task_indices}")
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
            self.update_clusters_after_assignment()
        else:
            self.get_logger().warn("No new assignment available to publish.")

    def update_clusters_after_assignment(self):
        all_points = list(self.points_with_label.keys())
        unassigned_points = [pt for pt in all_points if pt not in self.assigned_points_global]
        unassigned_labels = [self.points_with_label[pt] for pt in unassigned_points]
        # self.get_logger().info(f"unassigned_points: {unassigned_points}")
        if unassigned_points:
            self.cluster_centers, self.cluster_points = self.clusterer.cluster(
                task_points=unassigned_labels,
                task_coords=unassigned_points
            )
            self.cluster_gamma = []
            self.cluster_types = []
            
            # Clear and rebuild cluster_task_labels and cluster_delivery_labels for new clusters
            self.cluster_task_labels = {}
            self.cluster_delivery_labels = {}
            
            for i, (center, points) in enumerate(zip(self.cluster_centers, self.cluster_points)):
                normal_count = 0
                special_count = 0
                task_labels = []
                delivery_labels = []
                
                for coord in points:
                    for point_coord, label in self.points_with_label.items():
                        if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                            if label in self.special_labels:
                                special_count += 1
                            else:
                                normal_count += 1
                            # Build task and delivery labels for this cluster
                            task_labels.append(label)
                            if label in self.task_to_delivery:
                                delivery_labels.append(self.task_to_delivery[label])
                            else:
                                delivery_labels.append(None)
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
                
                # Store task and delivery labels for this cluster
                self.cluster_task_labels[i] = task_labels
                self.cluster_delivery_labels[i] = delivery_labels
                
            self.valid_cluster = [1] * len(self.cluster_centers)
        else:
            self.cluster_centers = []
            self.cluster_points = []
            self.cluster_gamma = []
            self.cluster_types = []
            self.valid_cluster = []
            self.cluster_task_labels = {}
            self.cluster_delivery_labels = {}

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
