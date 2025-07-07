#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import pygame
import numpy as np
from shapely.geometry import LineString, Polygon
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers, build_graph_halton, check_in_block, check_in_bump
import sys
import cv2
from ltl_automaton_msgs.msg import ShowPosition, UpdateValidTasks, TaskFail
from enum import Enum
import threading
import yaml
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import re

#=======================================================================
#  Interfaces between ShowMoveNode and other nodes
#                       -----------------
# This node is responsible for receiving robot position messages
# from multiple publishers, updating the shared robot state data,
# and periodically refreshing the display using pygame.
# It decouples data updates from visualization by using a timer-based
# simulation loop and ensures thread-safe data access with locks.
#=======================================================================

# Color definitions
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREY   = (190, 190, 190)
RED    = (255, 0, 0)
YELLOW = (152, 251, 152)
BLUE   = (0, 0, 128)

# Additional colors for tasks (loaded/unloaded)
GREEN    = (107, 142, 35)      # For unload task points
SKY_BLUE = (135, 206, 235)     # For unfinished load task points
# notask_color already defined as grey, used to indicate finished (or no task) task points
NOTASK_COLOR = (0, 255, 0)
FINISHED_TASK = (190, 190, 190)
FAIL = (0, 0, 0)

# ---------- Walls ----------
def load_lines_from_yaml():
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner')
    )
    file_path = os.path.join(parent_dir, 'config', 'wall.yaml')

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    line_coords = yaml_data.get('lines', [])
    return [LineString(coords) for coords in line_coords]

class GridWorld(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.width, self.height = 800, 800
        self.cell_size = self.width // self.grid_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont('timesnewroman', 20)
        pygame.display.set_caption("Multiagent Task Planner")
        self.clock = pygame.time.Clock()
        # Initialize video writer (if video saving is needed)
        # self.output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))

class ShowMoveNode(Node):
    def __init__(self, env):
        super().__init__('showmove_node')
        self.world = env

        self.world.block = []  # List for drawing blocked lines
        self.world.bump = []   # List for drawing bump lines

        # Load transition system from configuration file (if exists)
        transition_system_textfile = self.declare_parameter('transition_system_textfile', '').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.nodes, self.actions = build_graph_halton(20, 20, 1000)
        self.transition_system['state_models']['2d_pose_region']['nodes'] = self.nodes
        self.transition_system['actions'].update(self.actions)
        # The process to integrate self.nodes and self.actions into the transition system is omitted

        self.color_mapping = {
            'robot1': {'loaded': (0, 0, 255),     # Blue
                       'unloaded': (83, 77, 255)},  # Light Blue
            'robot2': {'loaded': (255, 0, 0),
                       'unloaded': (255, 92, 92)},
            'robot3': {'loaded': (255, 0, 0),
                       'unloaded': (255, 92, 92)},
            'robot4': {'loaded': (0, 0, 255),      # Pink
                       'unloaded': (83, 77, 255)},   # Light Pink
            # 'robot5': {'loaded': (0, 0, 255),
            #            'unloaded': (83, 77, 255)},
            # 'robot6': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot7': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot8': {'loaded': (0, 0, 255),
            #            'unloaded': (83, 77, 255)},
            # 'robot9': {'loaded': (0, 0, 255),     # Blue
            #            'unloaded': (83, 77, 255)},
            # 'robot10': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot11': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot12': {'loaded': (0, 0, 255),     # Blue
            #            'unloaded': (83, 77, 255)},
            # 'robot13': {'loaded': (0, 0, 255),     # Blue
            #            'unloaded': (83, 77, 255)},
            # 'robot14': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot15': {'loaded': (255, 0, 0),
            #            'unloaded': (255, 92, 92)},
            # 'robot16': {'loaded': (0, 0, 255),     # Blue
            #            'unloaded': (83, 77, 255)}
        }


        self.waiting_color = (255, 165, 0)  # Orange
        self.notask_color = NOTASK_COLOR     # Grey color for finished tasks
        self.finished_tasks_color = FINISHED_TASK
        self.fail_color = FAIL

        self.lines = load_lines_from_yaml()        

        self.obstacles = [line.buffer(distance=0.1, cap_style=3) for line in self.lines]

        self.check_in_blocks = [LineString([(5, 4), (6, 4)]),
                                LineString([(4, 15), (5, 15)])]
        self.blocks = [block.buffer(distance=0.1, cap_style=3) for block in self.check_in_blocks]

        # Dictionary to store all robot states in the format:
        # {'robot_id': {'pose': (x, y), 'mode': (R, G, B)}}
        self.robot_positions = {}
        # Create a lock for thread-safe access to shared resources
        self.lock = threading.Lock()

        # List of robot IDs
        self.robot_ids = ['robot1', 'robot2', 'robot3', 'robot4', 'robot5', 'robot6', 'robot7', 'robot8', 'robot9', 'robot10', 'robot11', 'robot12', 'robot13', 'robot14', 'robot15', 'robot16']
        self.special_robot_ids = ['robot2', 'robot3', 'robot6', 'robot7', 'robot10', 'robot11', 'robot14', 'robot15']

        self.failed_task_list = []

        # === 从YAML读取任务点、delivery点 ===
        parent_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner')
        )
        task_points_yaml = os.path.join(parent_dir, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # 解析任务点
        points_with_label = {}
        for k, v in yaml_data['task_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
        # 解析delivery点
        delivery_points = {}
        if 'delivery_points' in yaml_data:
            for k, v in yaml_data['delivery_points'].items():
                match = re.match(r"([\d\.]+),([\d\.]+)", k)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    delivery_points[(x, y)] = v
        # 合并所有点
        self.points = {**points_with_label, **delivery_points}
        # Set of unload points for quick lookup
        self.unloaded_points = set(delivery_points.keys())
        # Precompute loading-task labels in order to map between label and index
        self.loaded_labels = [
            label for pt, label in self.points.items()
            if pt not in self.unloaded_points
        ]
        
        # Initialize subscriptions lists
        self.position_subscriptions = []
        self.update_valid_tasks_subs = []
        self.task_failure_subs = []
        
        # Initialize finished tasks set to store completed task indices
        self.finished_tasks = set()

        # Create subscribers for each robot topic (e.g., "/robot1/show_position")
        for robot_id in self.robot_ids:
            topic = f"/{robot_id}/show_position"
            position_subscription = self.create_subscription(
                ShowPosition,
                topic,
                self.position_callback,
                10
            )
            # self.get_logger().info(f"Subscribed to topic: {topic}")
            self.position_subscriptions.append(position_subscription)

        # Create subscribers for update_valid_tasks on separate list
        for robot_id in self.robot_ids:
            topic = f"/{robot_id}/update_valid_tasks"
            valid_tasks_subscription = self.create_subscription(
                UpdateValidTasks,
                topic,
                self.update_valid_tasks,
                10
            )
            # self.get_logger().info(f"Subscribed to topic: {topic}")
            self.update_valid_tasks_subs.append(valid_tasks_subscription)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        for robot_id in self.robot_ids:
            topic = f"/{robot_id}/task_failure"
            task_fail_sub = self.create_subscription(
                TaskFail,
                topic,
                self.task_fail_callback,
                qos
            )
            # self.get_logger().info(f"Subscribed to topic: {topic}")
            self.task_failure_subs.append(task_fail_sub)

        # Create a timer for periodic simulation updates (e.g., every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.simulate)
    
    def task_fail_callback(self, msg):
        # 直接用label字符串
        try:
            label = msg.task_label
            # 如果之前标记为完成，移除
            if label in self.finished_tasks:
                self.finished_tasks.remove(label)
            # 加入失败列表
            if label not in self.failed_task_list:
                self.failed_task_list.append(label)
        except Exception:
            pass

    def update_valid_tasks(self, msg):
        """
        Callback for update_valid_tasks topic.
        现在直接用label字符串，不再用数字。
        """
        label = msg.loaded_task
        # 如果在失败列表，移除
        if label in self.failed_task_list:
            self.failed_task_list.remove(label)
        # 加入完成列表
        self.finished_tasks.add(label)

    def position_callback(self, msg):
        """
        Callback function: update self.robot_positions upon receiving a robot position message.
        This function only updates the state; the simulate() function is called periodically by the timer.
        """
        # Directly use msg.pose to get the position (expected as [x, y])
        pos = msg.pose
        # Get the corresponding color based on robot ID and mode
        color = self.get_color(robot_id=msg.robot_id, mode=msg.mode)
        # Use lock to ensure thread-safe update of shared data
        with self.lock:
            self.robot_positions[msg.robot_id] = {'pose': pos, 'mode': color}
        # self.get_logger().info(f"Received {msg.robot_id}: position {pos}, mode {msg.mode}")

    def get_color(self, robot_id, mode):
        """
        Returns the corresponding color based on robot ID and current mode.
        """
        if mode == "Waiting":
            return self.waiting_color
        elif mode == 'NoTask':
            return self.notask_color
        elif mode == "loaded":
            return self.color_mapping.get(robot_id, {}).get("loaded", (255, 255, 255))
        elif mode == "unloaded":
            return self.color_mapping.get(robot_id, {}).get("unloaded", (211, 211, 211))
        elif mode == "Fail":
            return self.fail_color
        else:
            # Default color: white
            return (255, 255, 255)
        
    def transform_coords(self, coord):
        """Convert shapely coordinates to pygame coordinates."""
        x, y = coord
        return int(x * self.world.cell_size), int(-y * self.world.cell_size + self.world.height)

    def simulate(self):
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                pygame.quit()
                sys.exit()
        # Clear the screen
        self.world.screen.fill(WHITE)
        
        # Draw obstacles (buffered polygons)
        for obstacle in self.obstacles:
            if obstacle.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in obstacle.exterior.coords]
                pygame.draw.polygon(self.world.screen, BLACK, polygon_coords, 0)  # Filled polygon
        
        # Draw blocks for check-in areas
        for block in self.blocks:
            if block.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in block.exterior.coords]
                pygame.draw.polygon(self.world.screen, RED, polygon_coords, 0)  # Filled polygon

        # If there are bump coordinates, you can construct bump polygons here (currently, coords is empty)
        bumps = []
        coords = [
            [(4.1, 1.1), (4.1, 2.0), (2.5, 2.0), (2.5, 1.1)],
            [(17.5, 15), (20, 15), (20, 13), (17.5, 13)],
            [(17.5, 5), (20, 5), (20, 3), (17.5, 3)]
        ]
        for coord in coords:
            polygon = Polygon(coord)
            bumps.append(polygon)

        # Define all task points and their labels, where some points are marked as unload points
        points = self.points
        unloaded_points = self.unloaded_points
        loaded_labels = self.loaded_labels
        special_points = {'db', 'eb', 'bb', 'cd', 'fd'}

        # Iterate through all points, choose color based on status:
        for pt, label in points.items():
            pixel_pos = self.transform_coords(pt)
            side = self.world.cell_size // 2

            if pt in unloaded_points:
                color = GREEN
            else:
                # 直接用label判断是否完成
                if label in self.finished_tasks:
                    color = self.finished_tasks_color
                    special_color = self.finished_tasks_color
                else:
                    color = SKY_BLUE
                    special_color = RED

            # Draw the task point (special tasks as triangles, others as rectangles)
            if label in special_points:
                top = (pixel_pos[0], pixel_pos[1] - side // 2)
                left = (pixel_pos[0] - side // 2, pixel_pos[1] + side // 2)
                right = (pixel_pos[0] + side // 2, pixel_pos[1] + side // 2)
                pygame.draw.polygon(self.world.screen, special_color, [top, left, right], 0)
            else:
                rect = pygame.Rect(pixel_pos[0] - side // 2, pixel_pos[1] - side // 2, side, side)
                pygame.draw.rect(self.world.screen, color, rect)

            # Draw red circle around failed loading tasks
            if pt not in unloaded_points:
                if label in self.failed_task_list:
                    pygame.draw.circle(self.world.screen, RED, pixel_pos, side, 2)

            # Draw the label
            font = pygame.font.SysFont("Arial", 16)
            text_surface = font.render(label, True, BLACK)
            self.world.screen.blit(text_surface, (pixel_pos[0] + 5, pixel_pos[1] + 5))

        for bump in bumps:
            if bump.geom_type == "Polygon":
                bump_coords = [self.transform_coords(pt) for pt in bump.exterior.coords]
                pygame.draw.polygon(self.world.screen, YELLOW, bump_coords, 0)
        
        for action in self.actions:
            pose_ab = extract_numbers(str(action))
            pose_a = pose_ab[0]
            pose_b = pose_ab[1]
            
            start_pos = (
                int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
            )
            end_pos = (
                int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
            )

            pygame.draw.line(self.world.screen, GREY, start_pos, end_pos, 1)
        
        # Draw blocked lines
        for action in self.world.block:
            pose_ab = extract_numbers(action)
            pose_a = pose_ab[0]
            pose_b = pose_ab[1]
            
            start_pos = (
                int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
            )
            end_pos = (
                int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
            )

            pygame.draw.line(self.world.screen, RED, start_pos, end_pos, 3)
        
        # Draw bump lines
        for action in self.world.bump:
            pose_ab = extract_numbers(action)
            pose_a = pose_ab[0]
            pose_b = pose_ab[1]
            
            start_pos = (
                int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
            )
            end_pos = (
                int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
            )

            pygame.draw.line(self.world.screen, YELLOW, start_pos, end_pos, 3)
            
        # Draw nodes
        for node in self.nodes:
            pygame.draw.circle(
                self.world.screen, BLUE,
                (self.nodes[node]['attr']['pose'][0] * self.world.cell_size,
                 self.world.height - (self.nodes[node]['attr']['pose'][1] * self.world.cell_size)),
                3
            )
    
        # Draw all robot positions (reading shared data under lock)
        with self.lock:
            for robot_id, info in self.robot_positions.items():
                pos = info['pose']
                color = info['mode']
                pixel_pos = ((pos[0] * self.world.cell_size), (self.world.height - pos[1] * self.world.cell_size))
                # Larger circle for special robots
                radius = self.world.cell_size // 3
                if robot_id in self.special_robot_ids:
                    radius = self.world.cell_size // 2
                pygame.draw.circle(self.world.screen, color, pixel_pos, radius)
                font = pygame.font.SysFont("Arial", 16)
                text_surface = font.render(robot_id, True, BLACK)
                self.world.screen.blit(text_surface, (pixel_pos[0] + 5, pixel_pos[1] + 5))

        pygame.display.flip()
        self.world.clock.tick(30)

def main(args=None):
    pygame.init()
    rclpy.init(args=args)
    # Create a node for obtaining parameters
    main_node = rclpy.create_node('showmove_node_main')
    grid_size = main_node.declare_parameter('N', 20).get_parameter_value().integer_value
    main_node.get_logger().info(f"grid_size: {grid_size}")
    
    env = GridWorld(grid_size)
    main_node.get_logger().info("Starting showmove_node ...")
    # Create the actual node for subscription and display
    showposition = ShowMoveNode(env)
    try:
        rclpy.spin(showposition)
    except KeyboardInterrupt:
        pass
    finally:
        showposition.destroy_node()
        main_node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
