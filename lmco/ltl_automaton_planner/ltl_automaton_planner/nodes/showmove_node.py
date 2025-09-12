#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import pygame
import numpy as np
from shapely.geometry import LineString, Polygon
from ltl_automaton_planner.ltl_automaton_utilities import (
    import_ts_from_file, 
    extract_numbers, 
    build_graph_halton, 
    load_lines_from_yaml,
    update_graph_with_obstacle,
    add_bump_polygon,
    add_block_polygon,
    BUMP_POLYGONS,
    BLOCK_POLYGONS
)
import sys
import cv2
from ltl_automaton_msgs.msg import ShowPosition, UpdateValidTasks, TaskFail, AddTask, ObstacleUpdate
from enum import Enum
import threading
import yaml
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import re
from ament_index_python.packages import get_package_share_directory
import time

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
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)         # For newly added walls
# notask_color already defined as grey, used to indicate finished (or no task) task points
NOTASK_COLOR = (0, 255, 0)
FINISHED_TASK = (190, 190, 190)
FAIL = (0, 0, 0)

# ---------- Walls ----------
# The local load_lines_from_yaml function is removed.
# We will use the one from ltl_automaton_utilities.

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

        self.waiting_color = (255, 165, 0)  # Orange
        self.notask_color = NOTASK_COLOR     # Grey color for finished tasks
        self.finished_tasks_color = FINISHED_TASK
        self.fail_color = FAIL

        # --- Separate Obstacle Lists for Drawing ---
        # 1. Obstacles from wall.yaml (Known) - To be drawn in BLACK
        self.initial_wall_polygons = [line.buffer(distance=0.1, cap_style=3) for line in load_lines_from_yaml()]

        # 2. Obstacles from utility file (Unknown) - To be drawn in RED
        self.unknown_block_polygons = BLOCK_POLYGONS.copy() # Copy initial state

        # 3. Obstacles added at runtime (New) - To be drawn in MAGENTA
        self.new_wall_polygons = []

        # --- Unify All Obstacles for Logic Nodes (like benchmark_node) ---
        # Clear the global list and repopulate it to ensure a single source of truth for collision checking
        BLOCK_POLYGONS.clear()
        for poly in self.initial_wall_polygons:
            BLOCK_POLYGONS.append(poly)
        for poly in self.unknown_block_polygons:
            BLOCK_POLYGONS.append(poly)

        # Dictionary to store all robot states in the format:
        # {'robot_id': {'pose': (x, y), 'mode': (R, G, B)}}
        self.robot_positions = {}
        # Create a lock for thread-safe access to shared resources
        self.lock = threading.Lock()

        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        points_with_label = {}
        for k, v in yaml_data['task_points'].items():
            match = re.match(r"([\d\.]+),([\d\.]+)", k)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                points_with_label[(x, y)] = v
        delivery_points = {}
        if 'delivery_points' in yaml_data:
            for k, v in yaml_data['delivery_points'].items():
                match = re.match(r"([\d\.]+),([\d\.]+)", k)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    delivery_points[(x, y)] = v
        self.points = {**points_with_label, **delivery_points}
        # Set of unload points for quick lookup
        self.unloaded_points = set(delivery_points.keys())
        # Precompute loading-task labels in order to map between label and index
        self.loaded_labels = [
            label for pt, label in self.points.items()
            if pt not in self.unloaded_points
        ]
        # special robot
        self.special_robot_ids = yaml_data.get('special_robot', [])
        # special labels
        self.special_labels = set(yaml_data.get('special_labels', []))

        self.robot_ids = list(yaml_data.get('robot_positions', {}).keys())

        PINK_LOADED = (255, 105, 180)
        PINK_UNLOADED = (255, 182, 193)
        BLUE_LOADED = (0, 0, 255)       
        BLUE_UNLOADED = (135, 206, 250)   
        self.color_mapping = {}
        for robot_id in self.robot_ids:
            if robot_id in self.special_robot_ids:
                self.color_mapping[robot_id] = {
                    'loaded': PINK_LOADED,
                    'unloaded': PINK_UNLOADED
                }
            else:
                self.color_mapping[robot_id] = {
                    'loaded': BLUE_LOADED,
                    'unloaded': BLUE_UNLOADED
                }
        
        self.failed_task_list = []

        self.new_task_points = []
        self.map_needs_update = False
        self.processed_obstacles = set()

        # Initialize subscriptions lists
        self.position_subscriptions = []
        self.update_valid_tasks_subs = []
        self.task_failure_subs = []
        
        # Initialize finished tasks set to store completed task indices
        self.finished_tasks = set()


        self.start_time = time.time()  
        self.end_time = None  
        self.timing_completed = False 
        self.get_logger().info(f"Start timing at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")

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

        self.add_task_sub = self.create_subscription(
            AddTask,
            "/add_task",
            self.add_task_callback,
            10
        )

        self.obstacle_update_sub = self.create_subscription(
            ObstacleUpdate,
            '/obstacle_update',
            self.obstacle_update_callback,
            10
        )

        # Create a timer for periodic simulation updates (e.g., every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.simulate)

    def _action_nodes_exist(self, action):
        try:
            pose_ab = extract_numbers(action)
            pose_a = str(pose_ab[0])
            pose_b = str(pose_ab[1])
            return pose_a in self.nodes and pose_b in self.nodes
        except:
            return False

    def check_all_robots_notask(self):
        if not self.timing_completed and len(self.robot_positions) >= len(self.robot_ids):
            all_notask = True
            for robot_id in self.robot_ids:
                if robot_id in self.robot_positions:
                    if self.robot_positions[robot_id]['mode'] != self.notask_color:
                        all_notask = False
                        break
                else:
                    all_notask = False
                    break
            
            if all_notask:
                self.end_time = time.time()
                self.timing_completed = True
                total_time = self.end_time - self.start_time
                self.get_logger().info("="*60)
                self.get_logger().info("🎉 All robots have finished tasks!")
                self.get_logger().info(f"📊 Time:")
                self.get_logger().info(f"   Starting Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
                self.get_logger().info(f"   Ending Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}")
                self.get_logger().info(f"   Total Time: {total_time:.2f} 秒")
                self.get_logger().info(f"   Robot number: {len(self.robot_ids)}")
                self.get_logger().info("="*60)
                return True
        return False

    def obstacle_update_callback(self, msg):
        obstacle_key = tuple(msg.obstacle_location)
        if obstacle_key in self.processed_obstacles:
            self.get_logger().info(f"Ignoring duplicate obstacle update for location: {msg.obstacle_location}")
            return

        self.get_logger().info(f"Received obstacle update: {msg.obstacle_type} at {msg.obstacle_location}")
        self.processed_obstacles.add(obstacle_key)

        # Convert flattened coordinates back to list of tuples
        coords_flat = msg.obstacle_location
        if len(coords_flat) % 2 != 0:
            self.get_logger().error("Received obstacle with an odd number of coordinates.")
            return
        
        vertices = []
        for i in range(0, len(coords_flat), 2):
            vertices.append((coords_flat[i], coords_flat[i+1]))

        if msg.obstacle_type == 'wall' or msg.obstacle_type == 'block':
            new_polygon = Polygon(vertices)
            # 1. Add to global list for other nodes' collision checking
            add_block_polygon(vertices)

            # 2. Add to local list for magenta drawing
            self.new_wall_polygons.append(new_polygon)
            
            # 3. Set flag to locally update the navigation graph
            self.map_needs_update = True
            # Store the new obstacle to be processed in the main loop
            self.last_added_obstacle = new_polygon 
            self.get_logger().info("New permanent obstacle received. It will be drawn in magenta. Map will be updated locally.")
        elif msg.obstacle_type == 'bush' or msg.obstacle_type == 'bump':
            # Add to the global BUMP_POLYGONS list for dynamic visualization
            add_bump_polygon(vertices)
            self.get_logger().info("New bump received for visualization.")

    def add_task_callback(self, msg):
        self.get_logger().info(f"Received add task command from LLM, new {msg.task_type} task appears at {msg.location}, updating map......")
        point = tuple(msg.location)
        label = msg.task_label
        
        if point in self.points:
            self.get_logger().info(f"Task point already exists at {point}, skipping...")
            return
            
        self.points[point] = label
    
        if msg.task_type == "special":
            self.special_labels.add(label)
        

    
    def task_fail_callback(self, msg):
        try:
            label = msg.task_label
            if label in self.finished_tasks:
                self.finished_tasks.remove(label)
            if label not in self.failed_task_list:
                self.failed_task_list.append(label)
        except Exception:
            pass

    def update_valid_tasks(self, msg):
        label = msg.loaded_task
        if label in self.failed_task_list:
            self.failed_task_list.remove(label)
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
            self.check_all_robots_notask()
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
        
        # Check if map needs update due to a new wall
        if self.map_needs_update and hasattr(self, 'last_added_obstacle'):
            self.get_logger().info("Locally updating map visualization due to new wall...")
            # Perform a local update instead of a full rebuild
            update_graph_with_obstacle(self.nodes, self.actions, self.last_added_obstacle)
            
            self.world.block = [action for action in self.world.block 
                               if self._action_nodes_exist(action)]
            self.world.bump = [action for action in self.world.bump 
                              if self._action_nodes_exist(action)]
            
            self.map_needs_update = False
            del self.last_added_obstacle # Clear after processing
            self.get_logger().info("Map visualization updated locally.")

        # Clear the screen
        self.world.screen.fill(WHITE)
        
        # --- Draw Obstacles with Different Colors ---
        # 1. Draw initial, known walls in BLACK
        for obstacle in self.initial_wall_polygons:
            if obstacle.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in obstacle.exterior.coords]
                pygame.draw.polygon(self.world.screen, BLACK, polygon_coords, 0)

        # 2. Draw unknown, pre-defined blocks in RED
        for obstacle in self.unknown_block_polygons:
            if obstacle.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in obstacle.exterior.coords]
                pygame.draw.polygon(self.world.screen, RED, polygon_coords, 0)

        # 3. Draw newly added walls in MAGENTA
        for obstacle in self.new_wall_polygons:
            if obstacle.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in obstacle.exterior.coords]
                pygame.draw.polygon(self.world.screen, MAGENTA, polygon_coords, 0)

        # Define all task points and their labels, where some points are marked as unload points
        points = self.points
        unloaded_points = self.unloaded_points
        loaded_labels = self.loaded_labels
        special_points = self.special_labels

        # Iterate through all points, choose color based on status:
        for pt, label in points.items():
            pixel_pos = self.transform_coords(pt)
            side = self.world.cell_size // 2

            if pt in unloaded_points:
                color = GREEN
            else:
                if label in self.finished_tasks:
                    color = self.finished_tasks_color
                    special_color = self.finished_tasks_color
                else:
                    color = CYAN
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

        # Draw bumps (e.g., bushes) in YELLOW
        bumps = BUMP_POLYGONS
        for bump in bumps:
            if bump.geom_type == "Polygon":
                bump_coords = [self.transform_coords(pt) for pt in bump.exterior.coords]
                pygame.draw.polygon(self.world.screen, YELLOW, bump_coords, 0)
        
        for action in self.actions:
            pose_ab = extract_numbers(str(action))
            pose_a = pose_ab[0]
            pose_b = pose_ab[1]
            
            if str(pose_a) not in self.nodes or str(pose_b) not in self.nodes:
                continue
            
            start_pos = (
                int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
            )
            end_pos = (
                int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
                int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
            )

            pygame.draw.line(self.world.screen, GREY, start_pos, end_pos, 1)
        
        # The drawing loops for self.world.block and self.world.bump seem to draw detected edges, not polygons.
        # This is different from the obstacle drawing above. This logic can remain.
        # Draw blocked lines
        for action in self.world.block:
            pose_ab = extract_numbers(action)
            pose_a = pose_ab[0]
            pose_b = pose_ab[1]
            
            if str(pose_a) not in self.nodes or str(pose_b) not in self.nodes:
                continue
            
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
            
            if str(pose_a) not in self.nodes or str(pose_b) not in self.nodes:
                continue
            
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
