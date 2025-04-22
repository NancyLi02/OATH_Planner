import os
import numpy as np
import rclpy
from rclpy.node import Node
from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_msgs.msg import ClusterTaskassign, RobotID, ClusterRequest, AgentFailTask, TaskFail
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

# -------------------- CBAA Algorithm --------------------
class CBAA:
    def __init__(self):
        pass

    def select_task(self, scores_list, y, robot_index, robot_types, task_types):
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

    def conflict_resolve(self, task_index, assigned_tasks, x):
        for robot, assigned_task in assigned_tasks:
            if assigned_task == task_index:
                print(f"Conflict detected for task {task_index}. Removing previous assignment for Robot {robot + 1}.")
                x[robot][task_index] = 0
                assigned_tasks.remove((robot, assigned_task))
                break
        return x, assigned_tasks

    def initial_task_assignment(self, scores_list, robot_types, task_types):
        """
        Continue the auction process until every robot has been assigned a task.
        """
        task_count = len(scores_list[0])
        num_robots = len(scores_list)
        x = [[0] * task_count for _ in range(num_robots)]
        y = [0] * task_count
        assigned_tasks = []
        unassigned_robots = []

        # Continue until each robot has a task assignment.
        while any(sum(row) == 0 for row in x):
            for robot_index in range(num_robots):
                if sum(x[robot_index]) == 0:
                    task_index = self.select_task(scores_list, y, robot_index, robot_types, task_types)
                    if task_index != -1:
                        x, assigned_tasks = self.conflict_resolve(task_index, assigned_tasks, x)
                        x[robot_index][task_index] = 1
                        assigned_tasks.append((robot_index, task_index))
                    else:
                        unassigned_robots.append(robot_index)

        print("\nFinal y list:", y)
        print("x list (task assignment status for each robot):")
        for robot_index, task_assignments in enumerate(x):
            print(f"Robot {robot_index + 1}: {task_assignments}")
        print(f"\nUnassigned Robots: {unassigned_robots}")

        return assigned_tasks, unassigned_robots


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

        # Load wall data from YAML file and perform task clustering, passing in special labels.
        self.clusterer = CostMapClusterer(
            wall_yaml_path=wall_path, 
            points_with_label=self.points_with_label,
            special_labels=special_labels
        )
        (self.centers_other, self.clusters_other), (self.centers_special, self.clusters_special) = self.clusterer.cluster()
        # Add self.valid_cluster: an array with length equal to the total number of clusters, each initialized to 1.
        self.valid_cluster = [1] * (len(self.clusters_other) + len(self.clusters_special))
        self.robot_cluster_indices = {}
        self.failed_task_owners = {}

        # Create CBAA auction algorithm object.
        self.cbaa_algorithm = CBAA()

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
        all_clusters = self.clusters_other + self.clusters_special
        task_types = ['normal'] * len(self.centers_other) + ['special'] * len(self.centers_special)
        
        # Compute the score for the requested robot only.
        scores = []

         # If this robot has broken down, give zero score for all clusters
        if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
            scores = [0.0] * len(all_clusters)
        else:
            for j, (center, t_type) in enumerate(zip(all_clusters, task_types)):
                # If this is a failed-task cluster owned by this robot, score = 0
                if self.failed_task_owners.get(j) == robot_id:
                    scores.append(0.0)
                    continue
                # If the cluster is invalid (already assigned), set the score to 0.
                if self.valid_cluster[j] == 0:
                    scores.append(0.0)
                    continue
                dist = np.linalg.norm(np.array(new_position) - np.array(center))
                base_score = 100.0 / (dist + 1e-5)
                if t_type == 'special':
                    if self.robot_types[robot_index] == 'special':
                        score = base_score * 1.5  # Bonus for special tasks
                    else:
                        score = 0.0  # Normal robot gets 0 for special tasks
                else:
                    score = base_score
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
        # Combine normal and special task centers and construct the task type list.
        all_centers = self.centers_other + self.centers_special
        self.all_clusters = all_centers
        task_types = ['normal'] * len(self.centers_other) + ['special'] * len(self.centers_special)
        self.task_types = task_types

        # Compute each robot's score for every task center.
        # Score = 100 / (distance + epsilon).
        # For special tasks: if the robot is special, the score is multiplied by 1.5; if normal, score is 0.
        scores = []
        for i, robot_pose in enumerate(self.robot_poses):
            robot_name = self.robot_names[i]
            # If this robot has broken down, all its scores are zero
            if hasattr(self, 'broke_agents') and robot_name in self.broke_agents:
                scores.append([0.0] * len(self.all_clusters))
                continue

            robot_scores = []
            for j, (center, t_type) in enumerate(zip(self.all_clusters, self.task_types)):
                dist = np.linalg.norm(np.array(robot_pose) - np.array(center))
                base_score = 100.0 / (dist + 1e-5)
                if t_type == 'special':
                    if self.robot_types[i] == 'special':
                        score = base_score * 1.5
                    else:
                        score = 0.0
                else:
                    score = base_score
                robot_scores.append(score)
            scores.append(robot_scores)

        # Call the auction algorithm for initial task assignment.
        assigned, _ = self.cbaa_algorithm.initial_task_assignment(scores, self.robot_types, task_types)

        # Map the auction results with the clustering data:
        # Combine the clustering results from both parts.
        all_clusters = self.clusters_other + self.clusters_special
        self.robot_cluster_map = {name: [] for name in self.robot_names}
        for robot_index, task_index in assigned:
            robot_name = self.robot_names[robot_index]
            if task_index == -1:
                # If a normal robot wins a special task (or no valid task), mark assignment as empty.
                self.robot_cluster_map[robot_name] = []
            else:
                self.robot_cluster_map[robot_name] = all_clusters[task_index]
                # Update valid_cluster: mark the assigned cluster as used by setting its valid_cluster flag to 0.
                self.valid_cluster[task_index] = 0

        print("\n=== Final Cluster Assignment ===")
        for robot, tasks in self.robot_cluster_map.items():
            print(f"{robot}: {tasks}")     

    def generate_task_sequences(self):
        self.robot_task_sequence = {}
        # Build mappings: label to coordinate, and coordinate to label.
        label_to_point = {v: k for k, v in self.points_with_label.items()}  # label -> point
        point_to_label = {k: v for k, v in self.points_with_label.items()}  # point -> label

        for robot_name, task_points in self.robot_cluster_map.items():
            current_pos = self.robot_poses[self.robot_names.index(robot_name)]
            # If the cluster is empty, assign an empty sequence.
            remaining = task_points[:] if isinstance(task_points, list) else []
            sequence = []

            while remaining:
                # Select the nearest task point to the current position.
                nearest = min(remaining, key=lambda pt: np.linalg.norm(np.array(current_pos) - np.array(pt)))
                label = point_to_label.get(tuple(map(float, nearest)))
                if label is None:
                    self.get_logger().error(f"Point {nearest} not found in task label dictionary!")
                    remaining.remove(nearest)
                    continue

                sequence.append(self.label_to_index[label])
                current_pos = nearest
                remaining.remove(nearest)

            self.robot_task_sequence[robot_name] = sequence

        print("\n=== Task Execution Sequences (by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            print(f"{robot}: {seq}")

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
            msg.robot_id = idx
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
