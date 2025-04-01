import os
import numpy as np
import rclpy
from rclpy.node import Node
from ltl_automaton_planner.CostMapClusterer import CostMapClusterer
from ltl_automaton_msgs.msg import ClusterTaskassign, RobotID

# -------------------- CBAA Algorithm --------------------
class CBAA:
    def __init__(self):
        pass

    def select_task(self, scores_list, y, robot_index):
        robot_scores = scores_list[robot_index]
        valid_task = [(1 if score > y_value else 0) for score, y_value in zip(robot_scores, y)]

        if sum(valid_task) > 0:
            valid_indices = [i for i, valid in enumerate(valid_task) if valid == 1]
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

    def initial_task_assignment(self, scores_list):
        task_count = len(scores_list[0])
        x = [[0] * task_count for _ in range(len(scores_list))]
        y = [0] * task_count
        assigned_tasks = []
        unassigned_robots = []

        while any(sum(row) == 0 for row in x):
            for robot_index in range(len(scores_list)):
                if sum(x[robot_index]) == 0:
                    task_index = self.select_task(scores_list, y, robot_index)
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

        # ----- Static robot poses -----
        self.robot_count = 4
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        self.robot_names_ = [f'robot_{i}' for i in range(1, self.robot_count + 1)]
        self.robot_poses = [
            (1, 19),  # robot_1
            (11, 19), # robot_2
            (9, 11),  # robot_3
            (11, 9)   # robot_4
        ]

        # Publishers per robot namespace
        self.task_pubs = {}
        for idx, robot in enumerate(self.robot_names):
            topic = f"/{robot}/ClusterTaskassign"
            self.task_pubs[robot] = self.create_publisher(ClusterTaskassign, topic, 10)

        # Subscriber to finish_building_auto
        self.finished_robots = set()
        self.finish_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/finish_building_auto"
            sub = self.create_subscription(RobotID, topic, self.finish_callback, 10)
            self.finish_subs.append(sub)

        # ----- Load wall & task info -----
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner'))
        wall_path = os.path.join(parent_dir, 'config', 'wall.yaml')

        points_with_label = {
            (1.0, 4): 'bb', (6, 6): 'cb', (1, 6.5): 'db', (5.5, 9.5): 'eb', (9, 6.5): 'fb',
            (1.0, 13.5): 'bc', (9, 16.5): 'cc', (1, 16): 'dc', (6, 13): 'ec',
            (11.0, 13.5): 'bd', (11, 16.5): 'cd', (19, 16.5): 'dd', (19, 19): 'ed', (15, 19.5): 'fd',
            (11.0, 4.0): 'be', (19, 6.5): 'ce', (16, 3): 'de', (19, 1): 'ee'
        }

        self.points_with_label = {
            tuple(map(float, k)): v for k, v in points_with_label.items()
        }
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(self.points_with_label.values())}
        self.clusterer = CostMapClusterer(wall_yaml_path=wall_path, points_with_label=self.points_with_label)
        self.centers, self.clusters = self.clusterer.cluster()
        self.cbaa_algorithm = CBAA()

    def finish_callback(self, msg):
        self.get_logger().info(f"Received finish_building_auto from {msg.robot_id}")
        self.finished_robots.add(msg.robot_id)
        if all(name in self.finished_robots for name in self.robot_names_):
            self.get_logger().info("All robots finished building, starting task assignment...")
            self.init_task_assign()
            self.generate_task_sequences()
            self.publish_task_assignments()

    def init_task_assign(self):
        scores = []
        for robot_pose in self.robot_poses:
            robot_scores = []
            for center in self.centers:
                dist = np.linalg.norm(np.array(robot_pose) - np.array(center))
                score = 100.0 / (dist + 1e-5)
                robot_scores.append(score)
            scores.append(robot_scores)

        assigned, _ = self.cbaa_algorithm.initial_task_assignment(scores)

        self.robot_cluster_map = {name: [] for name in self.robot_names}
        for robot_index, cluster_index in assigned:
            robot_name = self.robot_names[robot_index]
            cluster_labels = self.clusters[cluster_index]
            self.robot_cluster_map[robot_name] = cluster_labels

        print("\n=== Final Cluster Assignment ===")
        for robot, tasks in self.robot_cluster_map.items():
            print(f"{robot}: {tasks}")

    def generate_task_sequences(self):
        self.robot_task_sequence = {}
        label_to_point = {v: k for k, v in self.points_with_label.items()}  # label -> point
        point_to_label = {k: v for k, v in self.points_with_label.items()}  # point -> label ✅


        for robot_name, task_points in self.robot_cluster_map.items():
            current_pos = self.robot_poses[self.robot_names.index(robot_name)]
            remaining = task_points[:]  # These are points like (1.0, 13.5)
            sequence = []

            while remaining:
                # Find the closest point
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

        print("\n=== Task Execution Sequences (NNH, by task index) ===")
        for robot, seq in self.robot_task_sequence.items():
            print(f"{robot}: {seq}")


    def publish_task_assignments(self):
        for idx, robot in enumerate(self.robot_names):
            msg = ClusterTaskassign()
            msg.robot_id = idx + 1
            msg.task_sequence = self.robot_task_sequence[robot]
            self.task_pubs[robot].publish(msg)
            print(f"Published task sequence to {robot}: {msg.task_sequence}")


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
