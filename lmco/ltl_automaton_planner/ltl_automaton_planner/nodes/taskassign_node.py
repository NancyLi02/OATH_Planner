import os
import rclpy
from rclpy.node import Node
from ltl_automaton_msgs.msg import ClusterTaskassign, ClusterRequest, NoTask
import yaml
import re
from ament_index_python.packages import get_package_share_directory
import time


# -------------------- Task Assign Node --------------------
class TaskAssignNode(Node):
    def __init__(self):
        super().__init__('taskassign_node')

        # ----- 预定义的任务和路线列表 -----
        # 每个机器人是一组组任务（列表的列表），我们会按“组”为单位发布
        self.predefined_tasks = {
            # 'robot1': [
            #     ['cb load']
            # ],
            # 'robot2': [
            #     ['bc load', 'c unload'],
            #     ['dc load', 'c unload'],
            #     ['cc load', 'c unload']
            # ],
            'robot3': [
                ['ec load']
            ]
            # 'robot4': [
            #     ['fb load']
            # ]
        }

        # 从YAML文件获取机器人名称
        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        robot_positions_dict = yaml_data.get('robot_positions', {})
        self.robot_names = list(robot_positions_dict.keys())

        # 创建任务标签到索引的映射（包括 task_points 和 delivery_points，再加单字母标签）
        points_with_label = {}

        # 更鲁棒的坐标正则：支持负号、小数与空格
        coord_re = re.compile(r"\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*$")

        # 添加 task_points
        for k, v in yaml_data.get('task_points', {}).items():
            match = coord_re.match(k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
            else:
                self.get_logger().warn(f"task_points 键 '{k}' 不是 'x,y' 坐标格式，已跳过")

        # 添加 delivery_points
        for k, v in yaml_data.get('delivery_points', {}).items():
            match = coord_re.match(k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
            else:
                self.get_logger().warn(f"delivery_points 键 '{k}' 不是 'x,y' 坐标格式，已跳过")

        # 无条件把单字母类型（如 b/c）加入映射，保证能取到索引
        task_to_delivery = yaml_data.get('task_to_delivery', {})
        for delivery_type in task_to_delivery.keys():
            points_with_label[delivery_type] = delivery_type

        # 去重后再编号，保证索引稳定
        labels = []
        for v in points_with_label.values():
            if v not in labels:
                labels.append(v)
        self.label_to_index = {label: idx + 1 for idx, label in enumerate(labels)}

        # Publishers for each robot namespace
        self.task_pubs = {}
        self.no_task_pubs = {}
        for idx, robot in enumerate(self.robot_names):
            topic = f"/{robot}/ClusterTaskassign"
            self.task_pubs[robot] = self.create_publisher(ClusterTaskassign, topic, 10)
            # NoTask publisher with namespace
            no_task_topic = f"/{robot}/no_task"
            self.no_task_pubs[robot] = self.create_publisher(NoTask, no_task_topic, 10)

        # Subscribers for task completion requests
        self.task_completion_subs = []
        for robot in self.robot_names:
            topic = f"/{robot}/cluster_request"
            sub = self.create_subscription(ClusterRequest, topic, self.handle_task_completion_request, 10)
            self.task_completion_subs.append(sub)

        # 任务进度与数据结构（按组发布）
        # robot_group_progress[robot] -> 当前发布到第几组（组索引）
        # robot_group_sequences[robot] -> List[List[int]]，每组的任务索引列表
        # robot_group_labels[robot]    -> List[List[str]]，每组的任务标签列表
        self.robot_group_progress = {}
        self.robot_group_sequences = {}
        self.robot_group_labels = {}

        # 初始化任务组
        for robot in self.robot_names:
            group_indices_list = []
            group_labels_list = []

            robot_tasks = self.predefined_tasks.get(robot, [])
            if not robot_tasks:
                self.get_logger().warn(f"机器人 {robot} 没有预定义任务")
                self.robot_group_progress[robot] = 0
                self.robot_group_sequences[robot] = []
                self.robot_group_labels[robot] = []
                continue

            for task_group in robot_tasks:
                group_indices = []
                group_labels = []

                for task_label in task_group:
                    # 解析任务标签，提取位置部分（去掉 load/unload）
                    if task_label.endswith(' load'):
                        location_label = task_label[:-5]
                    elif task_label.endswith(' unload'):
                        location_label = task_label[:-7]
                    else:
                        location_label = task_label

                    if location_label in self.label_to_index:
                        group_indices.append(self.label_to_index[location_label])
                        group_labels.append(task_label)  # 保持原始格式用于 LTL 生成
                    else:
                        self.get_logger().warn(f"位置标签 '{location_label}' 未找到对应的索引")

                # 跳过空组（如果组里没有任何有效索引）
                if group_indices:
                    group_indices_list.append(group_indices)
                    group_labels_list.append(group_labels)
                else:
                    self.get_logger().warn(f"机器人 {robot} 的某一任务组为空，已跳过")

            self.robot_group_sequences[robot] = group_indices_list
            self.robot_group_labels[robot] = group_labels_list
            self.robot_group_progress[robot] = 0

        self.get_logger().info("TaskAssignNode initialization completed successfully")

        time.sleep(10)

        self.publish_first_groups()

    # 首次对每个机器人发布其第一个任务组
    def publish_first_groups(self):
        self.get_logger().info("开始按组发布初始任务...")
        for robot in self.robot_names:
            self.publish_next_group(robot)
        self.get_logger().info("初始任务组发布完毕")

    def publish_next_group(self, robot_name):
        """按组发布下一个任务组给指定机器人"""
        if robot_name not in self.robot_group_sequences:
            self.get_logger().warn(f"机器人 {robot_name} 没有任务组数据")
            return

        group_idx = self.robot_group_progress[robot_name]
        group_sequences = self.robot_group_sequences[robot_name]
        group_labels = self.robot_group_labels[robot_name]

        if group_idx >= len(group_sequences):
            # 所有任务完成，发布 no_task
            self.publish_no_task(robot_name)
            return

        # 取当前组
        seq_batch = group_sequences[group_idx]   # List[int]
        label_batch = group_labels[group_idx]    # List[str]

        # 发布当前组（一次性把该组的 load/unload/… 全部发出）
        robot_idx = self.robot_names.index(robot_name)
        msg = ClusterTaskassign()
        msg.robot_id = robot_idx + 1
        msg.task_sequence = seq_batch
        msg.route_labels = label_batch

        self.task_pubs[robot_name].publish(msg)
        self.get_logger().info(
            f"已向 {robot_name} 发布第 {group_idx + 1} 组任务: 序列={msg.task_sequence}"
        )
        self.get_logger().info(f"任务标签: {msg.route_labels}")

    def publish_no_task(self, robot_name):
        """发布 no_task 消息给指定机器人"""
        robot_idx = self.robot_names.index(robot_name)
        msg = NoTask()
        msg.robot_id = str(robot_idx + 1)  # 若 NoTask.robot_id 为 string，保持字符串

        self.no_task_pubs[robot_name].publish(msg)
        self.get_logger().info(f"已发布 no_task 给 {robot_name}，所有任务完成")

    def handle_task_completion_request(self, msg):
        """处理机器人完成任务的请求（按组推进）"""
        robot_id = int(msg.robot_id)  # 转换为整数
        robot_name = self.robot_names[robot_id - 1]  # robot_id 从 1 开始，索引从 0 开始

        self.get_logger().info(f"收到 {robot_name} 的任务完成请求")

        # 完成一整组后，推进到下一组
        self.robot_group_progress[robot_name] += 1

        # 发布下一组
        self.publish_next_group(robot_name)


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
