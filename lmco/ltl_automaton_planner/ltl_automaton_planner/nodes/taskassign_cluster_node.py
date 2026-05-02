import os
import numpy as np
import time
import json
import re
import yaml
import array
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_planner.MILP import ClusterTaskPlanner
from ltl_automaton_msgs.msg import (
    ClusterTaskassign,
    ClusterRequest,
    AgentFailTask,
    TaskFail,
    AddTask,
    NoTask,
    ChangeTaskPriority,
)


# -------------------- CWA Algorithm --------------------
class CWA:
    """
    Consensus-based auction algorithm.

    Score semantics (lower is better):
      - score == 0  : invalid (robot cannot bid on this cluster)
      - score >  0  : valid (smaller is preferred)

    For multi-type heterogeneous matching the score should be set to 0 when the
    capability-cluster compatibility (gamma_rk = <psi_k, zeta_r>) is zero.
    """

    def select_cluster(self, scores_list, y, robot_index, robot_types, cluster_types):
        robot_scores = scores_list[robot_index]
        valid_cluster = [
            (1 if score > 0 and (y_value == 0 or score < y_value) else 0)
            for score, y_value in zip(robot_scores, y)
        ]

        if sum(valid_cluster) > 0:
            valid_indices = [i for i, valid in enumerate(valid_cluster) if valid == 1]
            # Optional legacy hard filter (kept for backward-compat). With multi-type
            # capability scoring we set robot_types/cluster_types to 'normal' so this
            # filter is a no-op; capability matching is enforced through the score.
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
        cluster_count = len(scores_list[0]) if scores_list else 0
        num_robots = len(scores_list)
        x = [[0] * cluster_count for _ in range(num_robots)]
        y = [0] * cluster_count
        assigned_clusters = []
        unassigned_robots = set()

        # Bound the number of iterations to guard against pathological inputs
        max_iters = max(1, num_robots * (cluster_count + 1) * 2)
        iters = 0
        while any(sum(row) == 0 and r not in unassigned_robots for r, row in enumerate(x)) and iters < max_iters:
            iters += 1
            for robot_index in range(num_robots):
                if robot_index in unassigned_robots:
                    continue
                if sum(x[robot_index]) == 0:
                    cluster_index = self.select_cluster(
                        scores_list, y, robot_index, robot_types, cluster_types
                    )
                    if cluster_index != -1:
                        x, assigned_clusters = self.conflict_resolve(
                            cluster_index, assigned_clusters, x
                        )
                        x[robot_index][cluster_index] = 1
                        assigned_clusters.append((robot_index, cluster_index))
                    else:
                        unassigned_robots.add(robot_index)

        return assigned_clusters, sorted(unassigned_robots)


# -------------------- Task Assign Node --------------------
class TaskAssignNode(Node):
    """
    Heterogeneous multi-type task assignment node.

    Supports an arbitrary number of task types (1..N). Each robot has a binary
    capability vector U_r in {0,1}^N. Cluster k has a type-composition vector
    psi_k computed as the (smoothed) normalized count of task types it contains.
    The robot-cluster score follows the multi-type formulation:

        zeta_r   = U_r / ||U_r||_1
        gamma_rk = <psi_k, zeta_r>
        s_rk     = delta_rk / gamma_rk        (gamma_rk > 0)
        s_rk     = 0 (invalid)                (gamma_rk == 0)

    where delta_rk is the Euclidean distance from robot r's pose to cluster
    center k.
    """

    THETA = 1e-3  # Laplace smoothing for psi_k
    EPSILON = 1e-6  # avoid score == 0 when delta == 0 with positive gamma

    def __init__(self):
        super().__init__('taskassign_node')

        # ----- File paths -----
        package_share = get_package_share_directory('ltl_automaton_planner')
        self.task_points_yaml_path = os.path.join(
            package_share, 'config', 'Task_Points.yaml'
        )
        self.wall_path = os.path.join(package_share, 'config', 'wall.yaml')
        self.halton_points_csv = os.path.join(
            package_share, 'ltl_automaton_planner', 'all_points_in_Halton.csv'
        )
        self.precomputed_distances_csv = os.path.join(
            package_share, 'ltl_automaton_planner', 'multi_source_dijkstra_distances.csv'
        )

        with open(self.task_points_yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # ----- Robot positions and names -----
        robot_positions_dict = yaml_data.get('robot_positions', {})
        self.robot_names = list(robot_positions_dict.keys())
        self.robot_poses = []
        for coord_str in robot_positions_dict.values():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                self.robot_poses.append((float(match.group(1)), float(match.group(2))))
        self.robot_count = len(self.robot_names)

        # ----- Multi-type definitions -----
        self.type_labels_map, self.num_task_types = self._parse_task_types(yaml_data)
        # label -> integer type id (1..N)
        self.task_label_to_type = {}
        for type_id, labels in self.type_labels_map.items():
            for lbl in labels:
                self.task_label_to_type[lbl] = type_id
        self.get_logger().info(
            f"Loaded {self.num_task_types} task types: "
            f"{ {tid: lbls for tid, lbls in self.type_labels_map.items()} }"
        )

        # ----- Robot capabilities U_r and normalized zeta_r -----
        robot_caps_dict = yaml_data.get('robot_capabilities', {})
        self.robot_capabilities = []
        for name in self.robot_names:
            if name in robot_caps_dict:
                cap = np.array(robot_caps_dict[name], dtype=int)
                if len(cap) < self.num_task_types:
                    cap = np.concatenate([cap, np.zeros(self.num_task_types - len(cap), dtype=int)])
                elif len(cap) > self.num_task_types:
                    cap = cap[: self.num_task_types]
            else:
                cap = np.ones(self.num_task_types, dtype=int)
                self.get_logger().warn(
                    f"No capability defined for {name}, defaulting to all-ones {cap.tolist()}"
                )
            self.robot_capabilities.append(cap)
        self.robot_zetas = [self._normalize_capability(cap) for cap in self.robot_capabilities]
        self.get_logger().info(
            "Robot capabilities: "
            + ", ".join(
                f"{n}={c.tolist()}" for n, c in zip(self.robot_names, self.robot_capabilities)
            )
        )

        # Legacy types kept as 'normal' so CWA's hard filter is a no-op; capability
        # matching is enforced via the score (gamma_rk == 0 -> score == 0 -> invalid).
        self.robot_types = ['normal'] * self.robot_count

        # ----- Publishers -----
        self.task_pubs = {}
        self.no_task_pubs = {}
        for robot in self.robot_names:
            self.task_pubs[robot] = self.create_publisher(
                ClusterTaskassign, f"/{robot}/ClusterTaskassign", 10
            )
            self.no_task_pubs[robot] = self.create_publisher(NoTask, f"/{robot}/no_task", 10)

        self.robot_state_update_pub = self.create_publisher(String, '/robot_state_update', 10)

        # ----- Subscriptions -----
        self.new_cluster_request_subs = []
        for robot in self.robot_names:
            self.new_cluster_request_subs.append(
                self.create_subscription(
                    ClusterRequest, f"/{robot}/cluster_request", self.assign_new_cluster, 10
                )
            )
        self.robot_fail_subs = []
        for robot in self.robot_names:
            self.robot_fail_subs.append(
                self.create_subscription(
                    AgentFailTask, f"/{robot}/agent_fail_task", self.agent_fail_callback, 10
                )
            )
        self.task_fail_subs = []
        for robot in self.robot_names:
            self.task_fail_subs.append(
                self.create_subscription(
                    TaskFail, f"/{robot}/task_failure_cluster", self.task_fail_callback, 10
                )
            )
        self.llm_command_parser_subs = self.create_subscription(
            AddTask, '/add_task', self.add_task_callback, 10
        )
        self.get_logger().info("Successfully created subscription to /add_task topic")

        self.change_task_priority_subs = []
        for robot in self.robot_names:
            self.change_task_priority_subs.append(
                self.create_subscription(
                    ChangeTaskPriority,
                    f"/{robot}/change_task_priority",
                    self.change_task_priority_callback,
                    10,
                )
            )
        self.global_change_task_priority_sub = self.create_subscription(
            ChangeTaskPriority, '/change_task_priority', self.change_task_priority_callback, 10
        )
        self.get_logger().info("Successfully created subscriptions to change_task_priority topics")

        # ----- Auxiliary publishers -----
        self.timing_pub = self.create_publisher(String, '/timing_data', 10)
        self.all_robots_finished_pub = self.create_publisher(String, '/all_robots_finished', 10)

        # ----- State containers -----
        self.broke_agents = []
        self.task_priorities = {}  # task_label -> 'high' / 'low' / 'normal'
        self.high_priority_pending_tasks = set()
        self.no_task_robots = {}
        self.all_finished_published = False
        self.total_task_assignment_time = 0.0
        self.task_assignment_count = 0

        # ----- Task points and delivery info -----
        self.points_with_label = {}
        for k, v in yaml_data.get('task_points', {}).items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                self.points_with_label[(float(match.group(1)), float(match.group(2)))] = v
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        # delivery_label -> (x, y)
        self.delivery_points_data = {}
        for coord_str, label in yaml_data.get('delivery_points', {}).items():
            match = re.match(r"([\d\.]+),([\d\.]+)", coord_str)
            if match:
                self.delivery_points_data[label] = (
                    float(match.group(1)),
                    float(match.group(2)),
                )

        # pickup_label -> delivery_label  (flatten the nested mapping in the yaml)
        self.task_to_delivery = {}
        for delivery_label, pickup_list in yaml_data.get('task_to_delivery', {}).items():
            for p in pickup_list:
                self.task_to_delivery[p] = delivery_label

        # ----- Clustering -----
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=self.wall_path,
            num_clusters=4,
            halton_points_csv=self.halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=self.precomputed_distances_csv,
        )
        self.cluster_centers, self.cluster_points = self.clusterer.cluster()

        # cluster_psi[k]: type composition vector (length = num_task_types)
        # cluster_types[k]: kept as 'normal' for CWA backward compatibility
        self.cluster_psi = []
        self.cluster_types = []
        self.cluster_task_labels = {}
        self.cluster_delivery_labels = {}
        self._refresh_cluster_metadata()

        self.valid_cluster = [1] * len(self.cluster_centers)
        self.robot_cluster_indices = {}
        self.robot_cluster_map = {}
        self.failed_task_owners = {}

        # CWA auction
        self.cwa_algorithm = CWA()

        # MILP
        self.cluster_planner = ClusterTaskPlanner(
            dijkstra_distances_csv_path=self.precomputed_distances_csv
        )

        self.assigned_points_global = set()

        self.get_logger().info("TaskAssignNode initialization completed successfully")
        self.finish_callback()

    # ===================== Helpers =====================

    @staticmethod
    def _parse_task_types(yaml_data):
        """Return (type_labels_map: {type_id -> [labels]}, num_task_types)."""
        type_labels_map = {}
        max_type_id = 0
        for key, value in yaml_data.items():
            match = re.match(r"type(\d+)_labels", key)
            if match:
                type_id = int(match.group(1))
                type_labels_map[type_id] = list(value) if value else []
                max_type_id = max(max_type_id, type_id)
        if max_type_id == 0:
            raise RuntimeError(
                "No 'typeN_labels' entries found in Task_Points_multitype.yaml"
            )
        return type_labels_map, max_type_id

    @staticmethod
    def _normalize_capability(U):
        s = float(np.sum(np.abs(U)))
        if s <= 0:
            return np.zeros(len(U), dtype=float)
        return U.astype(float) / s

    def _coord_to_label(self, coord):
        """Find the label whose stored coordinate matches `coord` (within tolerance)."""
        for point_coord, label in self.points_with_label.items():
            if abs(point_coord[0] - coord[0]) < 1e-6 and abs(point_coord[1] - coord[1]) < 1e-6:
                return label
        return None

    def _compute_psi_for_cluster(self, points):
        """Compute the type-composition vector psi_k = (N_k + theta) / ||N_k + theta||_1."""
        counts = np.zeros(self.num_task_types, dtype=float)
        for coord in points:
            label = self._coord_to_label(coord)
            if label is None:
                continue
            type_id = self.task_label_to_type.get(label)
            if type_id is None or type_id < 1 or type_id > self.num_task_types:
                continue
            counts[type_id - 1] += 1.0
        vec = counts + self.THETA
        s = float(np.sum(vec))
        if s > 0:
            return vec / s
        return vec

    def _refresh_cluster_metadata(self):
        """Recompute cluster_psi, cluster_types, cluster_task_labels and cluster_delivery_labels."""
        self.cluster_psi = []
        self.cluster_types = []
        self.cluster_task_labels = {}
        self.cluster_delivery_labels = {}
        for i, points in enumerate(self.cluster_points):
            self.cluster_psi.append(self._compute_psi_for_cluster(points))
            self.cluster_types.append('normal')

            task_labels = []
            delivery_labels = []
            for coord in points:
                label = self._coord_to_label(coord)
                if label is None:
                    continue
                task_labels.append(label)
                delivery_labels.append(self.task_to_delivery.get(label))
            self.cluster_task_labels[i] = task_labels
            self.cluster_delivery_labels[i] = delivery_labels

    def _gamma_rk(self, robot_index, cluster_index):
        if cluster_index < 0 or cluster_index >= len(self.cluster_psi):
            return 0.0
        psi = self.cluster_psi[cluster_index]
        zeta = self.robot_zetas[robot_index]
        return float(np.dot(psi, zeta))

    def _robot_can_handle_cluster(self, robot_index, cluster_index):
        """True iff the cluster contains at least one task the robot can handle.

        This is a hard capability filter independent of the smoothed psi vector,
        so that robots without any capability overlap never win a bid on a cluster
        even though theta-smoothing keeps gamma_rk strictly positive.
        """
        if cluster_index < 0 or cluster_index >= len(self.cluster_points):
            return False
        cap = self.robot_capabilities[robot_index]
        for coord in self.cluster_points[cluster_index]:
            label = self._coord_to_label(coord)
            if label is None:
                continue
            type_id = self.task_label_to_type.get(label)
            if type_id is None or type_id < 1 or type_id > len(cap):
                continue
            if int(cap[type_id - 1]) == 1:
                return True
        return False

    def _robot_cluster_score(self, robot_pose, robot_index, cluster_index):
        """
        Compute the multi-type score s_rk = delta / gamma_rk.

        Returns 0.0 (invalid for the auction) when the robot has no capability
        overlap with the cluster. Otherwise returns the (smoothed) score using the
        psi-zeta inner product.
        """
        if not self._robot_can_handle_cluster(robot_index, cluster_index):
            return 0.0
        gamma = self._gamma_rk(robot_index, cluster_index)
        if gamma <= 0:
            return 0.0
        center = self.cluster_centers[cluster_index]
        delta = float(np.linalg.norm(np.array(robot_pose) - np.array(center)))
        return (delta + self.EPSILON) / gamma

    def _robot_can_handle_label(self, robot_index, task_label):
        """True iff the robot's capability vector has a 1 at the slot for the task's type."""
        type_id = self.task_label_to_type.get(task_label)
        if type_id is None:
            return True  # unknown type defaults to allowed
        cap = self.robot_capabilities[robot_index]
        if type_id < 1 or type_id > len(cap):
            return False
        return bool(cap[type_id - 1] == 1)

    @staticmethod
    def _parse_task_type_field(task_type_str):
        """Parse the AddTask.task_type string into an integer type id (1..N).

        Accepts:
          - decimal strings: "1", "2", ...
          - patterns like "type1", "type2", ...
        Returns None if the string cannot be parsed.
        """
        if task_type_str is None:
            return None
        s = str(task_type_str).strip().lower()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            pass
        m = re.match(r"type\s*(\d+)", s)
        if m:
            return int(m.group(1))
        return None

    # ===================== Callbacks =====================

    def add_task_callback(self, msg):
        response_start_mono = time.monotonic()
        response_start_time = time.time()

        self.get_logger().info("=== ADD_TASK_CALLBACK TRIGGERED ===")
        self.get_logger().info(
            f"Received add task command from LLM: type={msg.task_type}, "
            f"label={msg.task_label}, location={msg.location}, delivery={msg.delivery_point}"
        )
        self.add_task(msg.location, msg.task_type, msg.task_label, msg.delivery_point)

        response_end_mono = time.monotonic()
        response_duration = response_end_mono - response_start_mono
        self.get_logger().info(
            f"[TIMING] add_task response completed. Duration: {response_duration:.4f}s"
        )

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
            'timestamp': time.time(),
        })
        self.timing_pub.publish(timing_msg)

    def add_task(self, location, task_type, task_label, delivery_point):
        self.get_logger().info(
            f"Adding task {task_label} (type={task_type}) at {location} -> delivery {delivery_point}"
        )

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

        type_id = self._parse_task_type_field(task_type)
        if type_id is None or type_id < 1:
            self.get_logger().warn(
                f"Could not parse task_type='{task_type}', defaulting to type 1"
            )
            type_id = 1
        if type_id > self.num_task_types:
            self.get_logger().info(
                f"New task type {type_id} exceeds current num_task_types={self.num_task_types}; "
                f"expanding capability vectors with zeros."
            )
            self._expand_num_task_types(type_id)

        location_tuple = (float(location[0]), float(location[1]))
        if location_tuple in self.points_with_label:
            self.get_logger().warn(
                f"Task at {location_tuple} already exists, overwriting label to {task_label}"
            )
        self.points_with_label[location_tuple] = task_label

        # Update type membership: drop label from any previous type list, then add to new one
        for tid, labels in self.type_labels_map.items():
            if task_label in labels:
                labels.remove(task_label)
        self.type_labels_map.setdefault(type_id, []).append(task_label)
        self.task_label_to_type[task_label] = type_id

        # Update task_to_delivery
        self.task_to_delivery[task_label] = delivery_point

        # Regenerate label_to_index with consistent ordering
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}

        # Re-init clusterer with the updated task list
        self.clusterer = CostMapClusterer(
            points_with_label=self.points_with_label,
            wall_yaml_path=self.wall_path,
            num_clusters=4,
            halton_points_csv=self.halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=self.precomputed_distances_csv,
        )

        # Recluster what is still unassigned
        self.update_clusters_after_assignment()

        self.get_logger().info(
            f"Task {task_label} added. points_with_label size = {len(self.points_with_label)}; "
            f"task_label_to_type size = {len(self.task_label_to_type)}"
        )

        # If there are idle robots, run a single-task auction so the new task is picked up
        if self.no_task_robots:
            self.get_logger().info(
                f"Found {len(self.no_task_robots)} no_task robots; auctioning the new task..."
            )
            self.auction_new_task_to_idle_robots(location_tuple, task_label, type_id)

    def _expand_num_task_types(self, new_size):
        """Grow capability vectors and psi vectors when a new type id is introduced."""
        if new_size <= self.num_task_types:
            return
        delta = new_size - self.num_task_types
        for i in range(len(self.robot_capabilities)):
            self.robot_capabilities[i] = np.concatenate(
                [self.robot_capabilities[i], np.zeros(delta, dtype=int)]
            )
        self.robot_zetas = [self._normalize_capability(c) for c in self.robot_capabilities]
        self.num_task_types = new_size

    def publish_robot_state_update(self, robot_name, new_state):
        """Publish robot state update to showmove_node before planner builds automaton."""
        state_msg = String()
        state_msg.data = json.dumps({
            'type': 'robot_state_update',
            'robot_id': robot_name,
            'new_state': new_state,
            'timestamp': time.time(),
        })
        self.robot_state_update_pub.publish(state_msg)
        self.get_logger().info(f"Published state update: {robot_name} -> {new_state}")

    def check_all_robots_finished(self):
        active_robots = [name for name in self.robot_names if name not in self.broke_agents]
        all_finished = all(robot in self.no_task_robots for robot in active_robots)

        if all_finished and not self.all_finished_published:
            self.get_logger().info("=" * 60)
            self.get_logger().info("ALL ROBOTS FINISHED - Publishing global completion message")
            self.get_logger().info(f"No task robots: {list(self.no_task_robots.keys())}")
            self.get_logger().info("=" * 60)

            self.get_logger().info("")
            self.get_logger().info("=" * 60)
            self.get_logger().info("TASK ASSIGNMENT TIME STATISTICS")
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"Total task assignment operations: {self.task_assignment_count}")
            self.get_logger().info(
                f"Total time spent on task assignment: {self.total_task_assignment_time:.4f} seconds"
            )
            if self.task_assignment_count > 0:
                avg_time = self.total_task_assignment_time / self.task_assignment_count
                self.get_logger().info(f"Average time per assignment: {avg_time:.4f} seconds")
            self.get_logger().info("=" * 60)
            self.get_logger().info("")

            completion_msg = String()
            completion_msg.data = json.dumps({
                'type': 'all_robots_finished',
                'robots': list(self.no_task_robots.keys()),
                'timestamp': time.time(),
            })
            self.all_robots_finished_pub.publish(completion_msg)
            self.all_finished_published = True

            timing_msg = String()
            timing_msg.data = json.dumps({
                'type': 'global_completion',
                'all_robots_finished': True,
                'robot_count': len(active_robots),
                'total_task_assignment_time': self.total_task_assignment_time,
                'task_assignment_count': self.task_assignment_count,
                'timestamp': time.time(),
            })
            self.timing_pub.publish(timing_msg)
        elif not all_finished:
            self.all_finished_published = False

    def auction_new_task_to_idle_robots(self, task_location, task_label, task_type_id):
        """Auction a single new task to idle (no_task) robots based on capability + distance."""
        assignment_start_time = time.monotonic()

        self.get_logger().info(f"\n=== AUCTIONING NEW TASK '{task_label}' (type={task_type_id}) TO IDLE ROBOTS ===")
        self.get_logger().info(f"Task location: {task_location}")
        self.get_logger().info(f"Available no_task robots: {list(self.no_task_robots.keys())}")

        candidates = []
        for robot_name, robot_pos in self.no_task_robots.items():
            robot_index = self.robot_names.index(robot_name)

            if robot_name in self.broke_agents:
                self.get_logger().info(f"  {robot_name}: SKIPPED (broken)")
                continue

            if not self._robot_can_handle_label(robot_index, task_label):
                cap = self.robot_capabilities[robot_index].tolist()
                self.get_logger().info(
                    f"  {robot_name}: SKIPPED (capability {cap} cannot handle type {task_type_id})"
                )
                continue

            dist = float(np.linalg.norm(np.array(robot_pos) - np.array(task_location)))
            candidates.append((robot_name, robot_pos, dist, robot_index))
            self.get_logger().info(f"  {robot_name} at {robot_pos}: distance = {dist:.2f}")

        if not candidates:
            self.get_logger().warn("No eligible robots found for new task auction")
            assignment_duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += assignment_duration
            self.task_assignment_count += 1
            self.get_logger().info(
                f"[TIMING] Failed auction completed in {assignment_duration:.4f}s "
                f"(Total: {self.total_task_assignment_time:.4f}s)"
            )
            return

        candidates.sort(key=lambda c: c[2])
        winner_name, winner_pos, winner_dist, winner_index = candidates[0]
        self.get_logger().info(f"AUCTION WINNER: {winner_name} (distance: {winner_dist:.2f})")

        del self.no_task_robots[winner_name]
        self.publish_robot_state_update(winner_name, 'waiting')

        single_task_cluster = [task_location]
        self.robot_cluster_map[winner_name] = single_task_cluster

        new_task_cluster_idx = (
            len(getattr(self, 'cluster_centers', []))
            + 2000
            + len(self.assigned_points_global)
        )
        self.robot_cluster_indices[winner_name] = new_task_cluster_idx

        self.cluster_task_labels[new_task_cluster_idx] = [task_label]
        delivery_label = self.task_to_delivery.get(task_label)
        self.cluster_delivery_labels[new_task_cluster_idx] = [delivery_label] if delivery_label else [None]

        self.robot_poses[winner_index] = winner_pos
        self.last_assigned_robot = winner_name

        self.generate_task_sequences(robot_names=[winner_name])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()
        self.check_all_robots_finished()

        assignment_duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += assignment_duration
        self.task_assignment_count += 1
        self.get_logger().info(
            f"[TIMING] New task auction for {winner_name} completed in {assignment_duration:.4f}s "
            f"(Total: {self.total_task_assignment_time:.4f}s)"
        )
        self.get_logger().info(f"Successfully assigned new task '{task_label}' to {winner_name}")

    def change_task_priority_callback(self, msg):
        response_start_mono = time.monotonic()
        response_start_time = time.time()

        task_label = msg.task_label
        priority = msg.priority

        self.get_logger().info("=== CHANGE_TASK_PRIORITY_CALLBACK TRIGGERED ===")
        self.get_logger().info(
            f"Received priority change request: task '{task_label}' to priority '{priority}'"
        )

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
            self.high_priority_pending_tasks.add(task_label)
            self.get_logger().info(
                f"Task '{task_label}' added to high priority pending queue. "
                "Will be assigned to next compatible robot."
            )
        else:
            self.high_priority_pending_tasks.discard(task_label)
            self.handle_priority_change_reassignment(task_label, priority)

        response_end_mono = time.monotonic()
        response_duration = response_end_mono - response_start_mono
        self.get_logger().info(
            f'[TIMING] change_task_priority response completed. Duration: {response_duration:.4f}s'
        )

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
            'timestamp': time.time(),
        })
        self.timing_pub.publish(timing_msg)

    def handle_priority_change_reassignment(self, task_label, priority):
        affected_clusters = []
        for cluster_idx, task_labels in self.cluster_task_labels.items():
            if task_label in task_labels:
                affected_clusters.append(cluster_idx)
        if affected_clusters:
            self.get_logger().info(f"Task '{task_label}' found in clusters: {affected_clusters}")
        else:
            self.get_logger().info(f"Task '{task_label}' not currently in any assigned cluster")

    def find_available_robot(self, task_label=None):
        """Pick a working robot, preferring those with no current tasks. If `task_label` is given,
        only robots that can handle that label's type are considered."""
        candidates = []
        for robot_name in self.robot_names:
            if robot_name in self.broke_agents:
                continue
            robot_index = self.robot_names.index(robot_name)
            if task_label is not None and not self._robot_can_handle_label(robot_index, task_label):
                continue
            candidates.append(robot_name)

        if not candidates:
            self.get_logger().error("No available robots found!")
            return None

        for robot_name in candidates:
            current_tasks = getattr(self, 'robot_task_sequence', {}).get(robot_name, [])
            if not current_tasks:
                self.get_logger().info(f"Found available robot with no current tasks: {robot_name}")
                return robot_name

        best_robot = min(
            candidates,
            key=lambda r: len(getattr(self, 'robot_task_sequence', {}).get(r, [])),
        )
        self.get_logger().info(f"Found available robot with fewest tasks: {best_robot}")
        return best_robot

    def remove_task_from_current_assignment(self, task_label, task_coord):
        self.assigned_points_global.discard(task_coord)

        task_index = self.label_to_index.get(task_label)
        if task_index:
            for robot_name, task_sequence in getattr(self, 'robot_task_sequence', {}).items():
                if task_index in task_sequence:
                    task_sequence.remove(task_index)
                    self.get_logger().info(
                        f"Removed task {task_label} (index {task_index}) from {robot_name}'s task sequence"
                    )
                    break

        for robot_name, route_labels in getattr(self, 'robot_route_labels', {}).items():
            if task_label in route_labels:
                route_labels.remove(task_label)
                self.get_logger().info(f"Removed task {task_label} from {robot_name}'s route labels")
                break

        for cluster_idx, task_labels in self.cluster_task_labels.items():
            if task_label in task_labels:
                task_labels.remove(task_label)
                if not task_labels and cluster_idx in self.cluster_delivery_labels:
                    del self.cluster_delivery_labels[cluster_idx]
                self.get_logger().info(f"Removed task {task_label} from cluster {cluster_idx}")
                break

    def handle_high_priority_task(self, task_label):
        self.get_logger().info(f"\n=== HANDLING HIGH PRIORITY TASK: {task_label} ===")

        for cluster_idx, task_labels in self.cluster_task_labels.items():
            if task_label in task_labels and cluster_idx >= 1000:
                assigned_robot = None
                for robot_name, robot_cluster_idx in self.robot_cluster_indices.items():
                    if robot_cluster_idx == cluster_idx:
                        assigned_robot = robot_name
                        break
                self.get_logger().info(
                    f"Task {task_label} is already in high priority cluster {cluster_idx} "
                    f"assigned to {assigned_robot}; skipping duplicate processing"
                )
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
            self.get_logger().info(
                f"Task {task_label} is already assigned, removing from current assignment for high priority handling"
            )
            self.remove_task_from_current_assignment(task_label, task_coord)

        available_robot = self.find_available_robot(task_label=task_label)
        if not available_robot:
            self.get_logger().error(f"No available robot found for high priority task {task_label}")
            return

        single_task_cluster = [task_coord]
        self.robot_cluster_map[available_robot] = single_task_cluster

        high_priority_cluster_idx = len(self.cluster_centers) + 1000
        self.robot_cluster_indices[available_robot] = high_priority_cluster_idx

        self.cluster_task_labels[high_priority_cluster_idx] = [task_label]
        delivery_label = self.task_to_delivery.get(task_label)
        self.cluster_delivery_labels[high_priority_cluster_idx] = [delivery_label] if delivery_label else [None]

        self.get_logger().info(
            f"Created high priority cluster {high_priority_cluster_idx} for task {task_label}"
        )
        self.get_logger().info(f"Assigned high priority task to robot: {available_robot}")
        self.last_assigned_robot = available_robot

        self.generate_task_sequences(robot_names=[available_robot])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()

        self.get_logger().info(f"High priority task {task_label} successfully assigned to {available_robot}")

    def _try_assign_high_priority_task(self, robot_name, robot_index):
        """Try to assign a pending high-priority task to the requesting robot. Returns True if assigned."""
        self.get_logger().info(f"\n=== CHECKING HIGH PRIORITY TASKS FOR {robot_name} ===")
        self.get_logger().info(f"High priority pending tasks: {self.high_priority_pending_tasks}")

        for task_label in list(self.high_priority_pending_tasks):
            if not self._robot_can_handle_label(robot_index, task_label):
                cap = self.robot_capabilities[robot_index].tolist()
                ttype = self.task_label_to_type.get(task_label)
                self.get_logger().info(
                    f"Task '{task_label}' (type {ttype}) cannot be handled by '{robot_name}' "
                    f"(capability {cap}); skipping..."
                )
                continue

            task_coord = None
            for coord, label in self.points_with_label.items():
                if label == task_label:
                    task_coord = coord
                    break
            if not task_coord:
                self.get_logger().error(f"Could not find coordinates for high priority task '{task_label}'")
                self.high_priority_pending_tasks.discard(task_label)
                continue

            if task_coord in self.assigned_points_global:
                self.get_logger().info(
                    f"High priority task '{task_label}' is already assigned, removing from pending..."
                )
                self.high_priority_pending_tasks.discard(task_label)
                continue

            self.get_logger().info(
                f"=== DIRECTLY ASSIGNING HIGH PRIORITY TASK '{task_label}' TO {robot_name} ==="
            )
            self.high_priority_pending_tasks.discard(task_label)

            single_task_cluster = [task_coord]
            self.robot_cluster_map[robot_name] = single_task_cluster

            high_priority_cluster_idx = (
                len(self.cluster_centers) + 1000 + len(self.assigned_points_global)
            )
            self.robot_cluster_indices[robot_name] = high_priority_cluster_idx

            self.cluster_task_labels[high_priority_cluster_idx] = [task_label]
            delivery_label = self.task_to_delivery.get(task_label)
            self.cluster_delivery_labels[high_priority_cluster_idx] = (
                [delivery_label] if delivery_label else [None]
            )

            self.last_assigned_robot = robot_name
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

        cluster_task_labels = self.cluster_task_labels.get(cluster_index, [])
        if not cluster_task_labels and cluster_index < len(self.cluster_points):
            for coord in self.cluster_points[cluster_index]:
                lbl = self._coord_to_label(coord)
                if lbl is not None:
                    cluster_task_labels.append(lbl)

        if not cluster_task_labels:
            return base_score

        has_high_priority = any(
            self.task_priorities.get(lbl, 'normal') == 'high' for lbl in cluster_task_labels
        )
        has_low_priority = any(
            self.task_priorities.get(lbl, 'normal') == 'low' for lbl in cluster_task_labels
        )

        if has_high_priority:
            adjusted = base_score / 10.0
            self.get_logger().info(
                f"Cluster {cluster_index} has high priority tasks, score adjusted: {base_score} -> {adjusted}"
            )
            return adjusted
        if has_low_priority:
            adjusted = base_score * 10.0
            self.get_logger().info(
                f"Cluster {cluster_index} has low priority tasks, score adjusted: {base_score} -> {adjusted}"
            )
            return adjusted
        return base_score

    def task_fail_callback(self, msg):
        """Handle a single-task failure: build a 1-task cluster owned by the failing robot."""
        failed_label = msg.task_label
        owner_robot = msg.agent_id

        label_to_pt = {lbl: pt for pt, lbl in self.points_with_label.items()}
        pt = label_to_pt.get(failed_label)
        if pt is None:
            self.get_logger().error(f"Failed task label {failed_label} not found.")
            return

        # Append this point as a new single-task cluster
        new_idx = len(self.cluster_centers)
        self.cluster_centers.append(tuple(pt))
        self.cluster_points.append([tuple(pt)])
        self.cluster_psi.append(self._compute_psi_for_cluster([pt]))
        self.cluster_types.append('normal')
        self.cluster_task_labels[new_idx] = [failed_label]
        self.cluster_delivery_labels[new_idx] = [self.task_to_delivery.get(failed_label)]
        if len(self.valid_cluster) <= new_idx:
            self.valid_cluster.extend([1] * (new_idx + 1 - len(self.valid_cluster)))
        self.valid_cluster[new_idx] = 1

        self.failed_task_owners[new_idx] = owner_robot
        self.get_logger().info(
            f"Created failed-task cluster {new_idx} for label '{failed_label}' owned by {owner_robot}."
        )

    def agent_fail_callback(self, msg):
        self.get_logger().info(f'Task assignment node receive agent fail msg from {msg.robot_id}....')
        fail_task = msg.task_id
        fail_agent = msg.robot_id

        # Find which cluster contains the failed task index
        fail_cluster_idx = None
        for idx, points in enumerate(self.cluster_points):
            indices = []
            for pt in points:
                label = self.points_with_label.get(tuple(map(float, pt)))
                if label is not None and label in self.label_to_index:
                    indices.append(self.label_to_index[label])
            if fail_task in indices:
                fail_cluster_idx = idx
                break

        if fail_cluster_idx is not None and fail_cluster_idx < len(self.valid_cluster):
            self.valid_cluster[fail_cluster_idx] = 1

        # Trim the failing robot's task sequence to keep only the failure point onwards
        if 1 <= fail_agent <= len(self.robot_names):
            robot_name = self.robot_names[fail_agent - 1]
        else:
            robot_name = None

        if robot_name is not None:
            seq = getattr(self, 'robot_task_sequence', {}).get(robot_name, [])
            if fail_task in seq:
                pos = seq.index(fail_task)
                onward_seq = seq[pos:]
                self.robot_task_sequence[robot_name] = onward_seq

                # Trim the cluster to only contain onward task points (so future
                # auctions will not re-issue completed work)
                if fail_cluster_idx is not None:
                    label_to_point = {lbl: pt for pt, lbl in self.points_with_label.items()}
                    onward_labels = []
                    for task_id in onward_seq:
                        for lbl, idx in self.label_to_index.items():
                            if idx == task_id:
                                onward_labels.append(lbl)
                                break
                    new_cluster_pts = [
                        label_to_point[lbl] for lbl in onward_labels if lbl in label_to_point
                    ]
                    if new_cluster_pts:
                        self.cluster_points[fail_cluster_idx] = new_cluster_pts
                        self.cluster_centers[fail_cluster_idx] = tuple(
                            np.mean(new_cluster_pts, axis=0)
                        )
                        self.cluster_psi[fail_cluster_idx] = self._compute_psi_for_cluster(new_cluster_pts)

            if robot_name not in self.broke_agents:
                self.broke_agents.append(robot_name)

    def assign_new_cluster(self, msg):
        self.get_logger().info(f"Received new cluster assign request from Robot{msg.robot_id}.")
        assignment_start_time = time.monotonic()

        robot_id = msg.robot_id
        robot_index = robot_id - 1
        if robot_index < 0 or robot_index >= len(self.robot_names):
            self.get_logger().error(f"robot_id {robot_id} is out of range")
            return
        robot_name = self.robot_names[robot_index]

        new_position = tuple(msg.position)
        self.robot_poses[robot_index] = new_position
        self.get_logger().info(f"Updated position for {robot_name} to {new_position}.")

        if robot_name in self.no_task_robots:
            del self.no_task_robots[robot_name]
            self.get_logger().info(f"Removed {robot_name} from no_task_robots (requesting new task)")

        # 1) High priority queue first
        if self.high_priority_pending_tasks:
            assigned_high_priority = self._try_assign_high_priority_task(robot_name, robot_index)
            if assigned_high_priority:
                duration = time.monotonic() - assignment_start_time
                self.total_task_assignment_time += duration
                self.task_assignment_count += 1
                self.get_logger().info(
                    f"[TIMING] High priority assignment for {robot_name} completed in "
                    f"{duration:.4f}s (Total: {self.total_task_assignment_time:.4f}s)"
                )
                return

        # 2) No tasks left at all -> NoTask
        all_points = list(self.points_with_label.keys())
        unassigned_points = [pt for pt in all_points if pt not in self.assigned_points_global]
        if not unassigned_points:
            self.get_logger().info("No unassigned tasks left, publishing NoTask message.")
            self.no_task_robots[robot_name] = new_position
            no_task_msg = NoTask()
            no_task_msg.robot_id = robot_name
            self.no_task_pubs[robot_name].publish(no_task_msg)
            self.get_logger().info(f"Published NoTask message to {robot_name}")

            duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += duration
            self.task_assignment_count += 1
            self.get_logger().info(
                f"[TIMING] No-task check for {robot_name} completed in {duration:.4f}s "
                f"(Total: {self.total_task_assignment_time:.4f}s)"
            )
            self.check_all_robots_finished()
            return

        # 3) Compute scores for the requesting robot across current clusters
        scores = []
        if robot_name in self.broke_agents:
            scores = [0.0] * len(self.cluster_centers)
        else:
            for j in range(len(self.cluster_centers)):
                # Failed-task cluster owned by this robot must be skipped
                if self.failed_task_owners.get(j) == robot_id:
                    scores.append(0.0)
                    continue
                base_score = self._robot_cluster_score(new_position, robot_index, j)
                if base_score <= 0:
                    scores.append(0.0)
                    continue
                adjusted = self.apply_priority_adjustment(base_score, j)
                scores.append(adjusted)
                self.get_logger().info(
                    f"Cluster {j}: base score={base_score:.3f}, adjusted={adjusted:.3f} "
                    f"(psi={self.cluster_psi[j].tolist()}, zeta={self.robot_zetas[robot_index].tolist()})"
                )

        # Pick the best (lowest positive) score
        best_index = -1
        best_score = float('inf')
        for j, sc in enumerate(scores):
            if sc > 0 and sc < best_score:
                best_score = sc
                best_index = j

        if best_index == -1:
            self.get_logger().info(f"No valid cluster found for {robot_name}. Publishing NoTask message.")
            self.no_task_robots[robot_name] = new_position
            no_task_msg = NoTask()
            no_task_msg.robot_id = robot_name
            self.no_task_pubs[robot_name].publish(no_task_msg)
            self.get_logger().info(f"Published NoTask message to {robot_name}")

            duration = time.monotonic() - assignment_start_time
            self.total_task_assignment_time += duration
            self.task_assignment_count += 1
            self.get_logger().info(
                f"[TIMING] No valid cluster for {robot_name} completed in {duration:.4f}s "
                f"(Total: {self.total_task_assignment_time:.4f}s)"
            )
            self.check_all_robots_finished()
            return

        # 4) Commit the assignment
        self.robot_cluster_map[robot_name] = self.cluster_points[best_index]
        self.robot_cluster_indices[robot_name] = best_index
        self.get_logger().info(
            f"Assigned cluster {best_index} (score {best_score:.2f}) to {robot_name}."
        )

        task_labels = []
        delivery_labels = []
        for coord in self.cluster_points[best_index]:
            label = self._coord_to_label(coord)
            if label is None:
                continue
            task_labels.append(label)
            delivery_labels.append(self.task_to_delivery.get(label))
        self.cluster_task_labels[best_index] = task_labels
        self.cluster_delivery_labels[best_index] = delivery_labels
        self.get_logger().info(f"Cluster {best_index} task labels: {task_labels}")
        self.get_logger().info(f"Cluster {best_index} delivery labels: {delivery_labels}")

        self.last_assigned_robot = robot_name
        self.generate_task_sequences(robot_names=[robot_name])
        self.publish_task_reassignments()
        self.update_clusters_after_assignment()

        duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += duration
        self.task_assignment_count += 1
        self.get_logger().info(
            f"[TIMING] Cluster reassignment for {robot_name} completed in {duration:.4f}s "
            f"(Total: {self.total_task_assignment_time:.4f}s)"
        )

    def finish_callback(self):
        self.get_logger().info("Starting task assignment...")
        assignment_start_time = time.monotonic()

        self.init_cluster_assign()
        self.generate_task_sequences()
        self.publish_task_assignments()

        duration = time.monotonic() - assignment_start_time
        self.total_task_assignment_time += duration
        self.task_assignment_count += 1
        self.get_logger().info(
            f"[TIMING] Initial task assignment completed in {duration:.4f}s "
            f"(Total: {self.total_task_assignment_time:.4f}s)"
        )

    def init_cluster_assign(self):
        """Initial cluster assignment using multi-type capability scoring + CWA auction."""
        # Build R x K score matrix
        scores = []
        K = len(self.cluster_centers)
        for i, robot_pose in enumerate(self.robot_poses):
            robot_name = self.robot_names[i]
            if robot_name in self.broke_agents:
                scores.append([0.0] * K)
                continue
            row = []
            for j in range(K):
                base = self._robot_cluster_score(robot_pose, i, j)
                if base <= 0:
                    row.append(0.0)
                    continue
                row.append(self.apply_priority_adjustment(base, j))
            scores.append(row)

        self.get_logger().info(f"Initial scores (R x K): {scores}")

        assigned, _ = self.cwa_algorithm.initial_cluster_assignment(
            scores, self.robot_types, self.cluster_types
        )

        self.robot_cluster_map = {name: [] for name in self.robot_names}
        self.robot_cluster_indices = {}
        for robot_index, cluster_index in assigned:
            robot_name = self.robot_names[robot_index]
            if cluster_index == -1:
                self.robot_cluster_map[robot_name] = []
                self.robot_cluster_indices[robot_name] = -1
            else:
                self.robot_cluster_map[robot_name] = self.cluster_points[cluster_index]
                self.robot_cluster_indices[robot_name] = cluster_index
                if cluster_index < len(self.valid_cluster):
                    self.valid_cluster[cluster_index] = 0

        # Refresh cluster task/delivery labels
        for cluster_index, points in enumerate(self.cluster_points):
            task_labels = []
            delivery_labels = []
            for coord in points:
                label = self._coord_to_label(coord)
                if label is None:
                    continue
                task_labels.append(label)
                delivery_labels.append(self.task_to_delivery.get(label))
            self.cluster_task_labels[cluster_index] = task_labels
            self.cluster_delivery_labels[cluster_index] = delivery_labels

    def generate_task_sequences(self, robot_names=None):
        """Use MILP to optimize task sequences within each cluster."""
        self.robot_task_sequence = getattr(self, 'robot_task_sequence', {})
        self.robot_route_labels = getattr(self, 'robot_route_labels', {})

        if robot_names is None:
            robot_names = self.robot_names

        for robot_name in robot_names:
            task_points = self.robot_cluster_map.get(robot_name, [])
            if not task_points:
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue

            robot_index = self.robot_names.index(robot_name)
            robot_start = self.robot_poses[robot_index]

            cluster_index = self.robot_cluster_indices.get(robot_name, -1)
            if cluster_index == -1:
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue

            task_labels = list(self.cluster_task_labels.get(cluster_index, []))
            if not task_labels:
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue

            # ----- Filter pickups by robot capability -----
            label_to_coord = {lbl: pt for pt, lbl in self.points_with_label.items()}
            pickup_points = []
            pickup_labels = []
            for label in task_labels:
                if label not in label_to_coord:
                    continue
                if not self._robot_can_handle_label(robot_index, label):
                    self.get_logger().info(
                        f"Skipping task '{label}' for {robot_name} (capability mismatch, "
                        f"type={self.task_label_to_type.get(label)})"
                    )
                    continue
                pickup_points.append(label_to_coord[label])
                pickup_labels.append(label)

            if not pickup_labels:
                self.get_logger().warn(
                    f"{robot_name} has no executable pickups in cluster {cluster_index}"
                )
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue

            # ----- Build delivery point list from pickup_to_delivery -----
            pickup_to_delivery = {}
            for pl in pickup_labels:
                if pl in self.task_to_delivery:
                    pickup_to_delivery[pl] = self.task_to_delivery[pl]

            unique_delivery_labels = []
            seen = set()
            for pl in pickup_labels:
                d = pickup_to_delivery.get(pl)
                if d is not None and d not in seen:
                    seen.add(d)
                    unique_delivery_labels.append(d)

            delivery_points = []
            delivery_labels_unique = []
            for d in unique_delivery_labels:
                if d in self.delivery_points_data:
                    delivery_points.append(self.delivery_points_data[d])
                    delivery_labels_unique.append(d)
                else:
                    self.get_logger().warn(
                        f"Delivery label '{d}' is not in delivery_points; "
                        f"associated pickups will be dropped."
                    )

            # Drop pickups whose delivery is missing
            valid_delivery_set = set(delivery_labels_unique)
            filtered_pickup_points = []
            filtered_pickup_labels = []
            for pt, lbl in zip(pickup_points, pickup_labels):
                d = pickup_to_delivery.get(lbl)
                if d in valid_delivery_set:
                    filtered_pickup_points.append(pt)
                    filtered_pickup_labels.append(lbl)
            pickup_points = filtered_pickup_points
            pickup_labels = filtered_pickup_labels

            if not pickup_labels or not delivery_labels_unique:
                self.get_logger().warn(
                    f"No feasible pickup/delivery pair for {robot_name} after filtering."
                )
                self.robot_task_sequence[robot_name] = []
                self.robot_route_labels[robot_name] = []
                continue

            try:
                chosen_pickups, chosen_deliveries, route_labels, total_cost = (
                    self.cluster_planner.plan_cluster_tasks(
                        robot_start=robot_start,
                        pickup_points=pickup_points,
                        pickup_labels=pickup_labels,
                        delivery_points=delivery_points,
                        delivery_labels=delivery_labels_unique,
                        pickup_to_delivery=pickup_to_delivery,
                        robot_capacity=3,
                    )
                )

                if not chosen_pickups:
                    self.get_logger().warn(f"MILP failed for {robot_name}, using nearest neighbor fallback")
                    seq, full_labels = self._nearest_neighbor_fallback(
                        robot_start, pickup_points, pickup_labels
                    )
                    self.robot_task_sequence[robot_name] = seq
                    self.robot_route_labels[robot_name] = full_labels

                    nn_pickup_labels = [lbl for lbl in full_labels if lbl in pickup_labels]
                    selected_pts = []
                    for pl in nn_pickup_labels:
                        for pt, lbl in self.points_with_label.items():
                            if lbl == pl:
                                selected_pts.append(pt)
                                self.assigned_points_global.add(pt)
                                break
                    self.get_logger().info(
                        f"NN selected {len(nn_pickup_labels)} pickup tasks: {nn_pickup_labels}"
                    )
                    self.get_logger().info(
                        f"Marked {len(selected_pts)} NN selected tasks as assigned"
                    )
                else:
                    self.get_logger().info(f"\n=== MILP Selection Details for {robot_name} ===")
                    self.get_logger().info(
                        f"Cluster had {len(pickup_labels)} pickup tasks: {pickup_labels}"
                    )
                    self.get_logger().info(
                        f"MILP selected {len(chosen_pickups)} pickups: {chosen_pickups}"
                    )
                    unselected_pickups = [
                        label for label in pickup_labels if label not in chosen_pickups
                    ]
                    if unselected_pickups:
                        self.get_logger().warn(
                            f"MILP did NOT select these pickups: {unselected_pickups}"
                        )
                        self.get_logger().warn(
                            "These unselected tasks will remain in unassigned list for future allocation"
                        )

                    task_indices = []
                    for label in route_labels:
                        if label in self.label_to_index:
                            task_indices.append(self.label_to_index[label])
                    self.robot_task_sequence[robot_name] = task_indices
                    self.robot_route_labels[robot_name] = route_labels

                    selected_pts = []
                    for pl in chosen_pickups:
                        for pt, lbl in self.points_with_label.items():
                            if lbl == pl:
                                selected_pts.append(pt)
                                self.assigned_points_global.add(pt)
                                break
                    self.get_logger().info(
                        f"Marked {len(selected_pts)} selected pickup tasks as assigned. "
                        f"MILP cost = {total_cost:.3f}, deliveries = {chosen_deliveries}"
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"MILP planning raised {type(e).__name__}: {e}; using nearest neighbor fallback"
                )
                seq, full_labels = self._nearest_neighbor_fallback(
                    robot_start, pickup_points, pickup_labels
                )
                self.robot_task_sequence[robot_name] = seq
                self.robot_route_labels[robot_name] = full_labels

                fb_pickup_labels = [lbl for lbl in full_labels if lbl in pickup_labels]
                selected_pts = []
                for pl in fb_pickup_labels:
                    for pt, lbl in self.points_with_label.items():
                        if lbl == pl:
                            selected_pts.append(pt)
                            self.assigned_points_global.add(pt)
                            break
                self.get_logger().info(
                    f"Exception fallback selected {len(fb_pickup_labels)} pickup tasks: "
                    f"{fb_pickup_labels}"
                )
                self.get_logger().info(
                    f"Marked {len(selected_pts)} fallback selected tasks as assigned"
                )

        self.get_logger().info("\n=== Final Task Execution Sequences (by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            self.get_logger().info(f"{robot}: {seq}")

        self.log_unassigned_tasks()

    def log_unassigned_tasks(self):
        all_task_points = set(self.points_with_label.keys())
        unassigned_points = all_task_points - self.assigned_points_global

        if unassigned_points:
            unassigned_labels = []
            for pt in unassigned_points:
                label = self.points_with_label.get(pt)
                if label:
                    unassigned_labels.append(label)

            self.get_logger().warn("\n=== UNASSIGNED TASKS DETECTED ===")
            self.get_logger().warn(f"The number of unassigned tasks: {len(unassigned_points)}")
            self.get_logger().warn(f"The labels of unassigned tasks: {unassigned_labels}")
            self.get_logger().warn(
                f"The coordinations of unassigned tasks: {list(unassigned_points)}"
            )

            for cluster_idx, task_labels in self.cluster_task_labels.items():
                cluster_unassigned = [label for label in task_labels if label in unassigned_labels]
                if cluster_unassigned:
                    assigned_robot = None
                    for robot_name, robot_cluster_idx in self.robot_cluster_indices.items():
                        if robot_cluster_idx == cluster_idx:
                            assigned_robot = robot_name
                            break
                    self.get_logger().warn(
                        f"Cluster {cluster_idx} (assigned to {assigned_robot}): {cluster_unassigned}"
                    )
                    self.get_logger().warn(f"All task labels in cluster: {task_labels}")
        else:
            self.get_logger().info("\n=== ALL TASKS ASSIGNED ===")

    def _nearest_neighbor_fallback(self, robot_start, task_points, task_labels, robot_capacity=3):
        """Greedy fallback: nearest-neighbor pickups followed by their (deduped) deliveries."""
        current_pos = robot_start
        remaining = list(range(len(task_points)))
        pickup_sequence = []

        while remaining and len(pickup_sequence) < robot_capacity:
            nearest_idx = min(
                remaining,
                key=lambda i: np.linalg.norm(np.array(current_pos) - np.array(task_points[i])),
            )
            pickup_sequence.append(nearest_idx)
            current_pos = task_points[nearest_idx]
            remaining.remove(nearest_idx)

        pickup_labels_ordered = [task_labels[idx] for idx in pickup_sequence]
        delivery_labels_set = []
        seen = set()
        for label in pickup_labels_ordered:
            d = self.task_to_delivery.get(label)
            if d is not None and d not in seen:
                seen.add(d)
                delivery_labels_set.append(d)

        full_labels = pickup_labels_ordered + delivery_labels_set
        task_indices = [self.label_to_index[l] for l in full_labels if l in self.label_to_index]
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

        if unassigned_points:
            self.cluster_centers, self.cluster_points = self.clusterer.cluster(
                task_points=unassigned_labels, task_coords=unassigned_points
            )
            self._refresh_cluster_metadata()
            self.valid_cluster = [1] * len(self.cluster_centers)
        else:
            self.cluster_centers = []
            self.cluster_points = []
            self.cluster_psi = []
            self.cluster_types = []
            self.valid_cluster = []
            self.cluster_task_labels = {}
            self.cluster_delivery_labels = {}


def main(args=None):
    rclpy.init(args=args)
    task_assign_node = None
    try:
        task_assign_node = TaskAssignNode()
        rclpy.spin(task_assign_node)
    except KeyboardInterrupt:
        if task_assign_node is not None:
            task_assign_node.get_logger().info("KeyboardInterrupt detected, shutting down...")
    finally:
        if task_assign_node is not None:
            task_assign_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
