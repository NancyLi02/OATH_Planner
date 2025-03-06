#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import pygame
import numpy as np
from shapely.geometry import LineString, Polygon
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers, build_graph_hilton, check_in_block, check_in_bump
import sys
import cv2
from ltl_automaton_msgs.msg import ShowPosition
from enum import Enum
import threading

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
RED    = (255, 0, 0)
YELLOW = (255, 255, 222)
BLUE   = (0, 0, 128)

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
        self.nodes, self.actions = build_graph_hilton(20, 20, 700)
        self.transition_system['state_models']['2d_pose_region']['nodes'] = self.nodes
        self.transition_system['actions'].update(self.actions)
        # The process to integrate self.nodes and self.actions into the transition system is omitted

        self.color_mapping = {
            'robot_1': {'loaded': (0, 0, 255),     # Blue
                       'unloaded': (83, 77, 255)},# Light Blue
            'robot_2': {'loaded': (0, 255, 0),       # Green
                       'unloaded': (107, 255, 107)},# Light Green
            'robot_3': {'loaded': (255, 0, 0),     # Purple
                       'unloaded': (255, 92, 92)},# Light Purple
            'robot_4': {'loaded': (255, 6, 231),   # Pink
                       'unloaded': (255, 155, 246)} # Light Pink
        }
        self.waiting_color = (255, 165, 0)  # Orange

        # Define obstacles: using a single LineString as an example
        self.lines = [LineString([(0, 3), (2, 3), (2, 4)]),
                LineString([(0, 5), (2, 5)]),
                LineString([(0, 7), (2, 7), (2, 6)]),
                LineString([(4, 9), (6, 9), (6, 10)]),
                LineString([(6, 7), (4, 7), (4, 5)]),
                LineString([(5, 5), (7, 5), (7, 7)]),
                LineString([(8, 5), (8, 7), (10, 7)]),
                LineString([(4, 2), (4, 4), (5, 4)]),
                LineString([(6, 4), (7, 4), (7, 2), (5, 2)]),

                LineString([(10, 3), (12, 3), (12, 4)]),
                LineString([(10, 5), (12, 5)]),
                LineString([(10, 7), (12, 7), (12, 6)]),
                LineString([(14, 9), (16, 9), (16, 10)]),
                LineString([(16, 7), (14, 7), (14, 5)]),
                LineString([(15, 5), (17, 5), (17, 7)]),
                LineString([(18, 5), (18, 7), (20, 7)]),
                LineString([(14, 2), (14, 4), (15, 4)]),
                LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),

                LineString([(0, 13), (2, 13), (2, 14)]),
                LineString([(0, 15), (2, 15)]),
                LineString([(0, 17), (2, 17), (2, 16)]),
                LineString([(4, 19), (6, 19), (6, 20)]),
                LineString([(6, 17), (4, 17), (4, 15)]),
                LineString([(5, 15), (7, 15), (7, 17)]),
                LineString([(8, 15), (8, 17), (10, 17)]),
                LineString([(4, 12), (4, 14), (5, 14)]),
                LineString([(6, 14), (7, 14), (7, 12), (5, 12)]),
                
                LineString([(10, 13), (12, 13), (12, 14)]),
                LineString([(10, 15), (12, 15)]),
                LineString([(10, 17), (12, 17), (12, 16)]),
                LineString([(14, 19), (16, 19), (16, 20)]),
                LineString([(16, 17), (14, 17), (14, 15)]),
                LineString([(15, 15), (17, 15), (17, 17)]),
                LineString([(18, 15), (18, 17), (20, 17)]),
                LineString([(14, 12), (14, 14), (15, 14)]),
                LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
                
                LineString([(0, 10), (6, 10)]),
                LineString([(10, 0), (10, 7)]),
                LineString([(14, 10), (20, 10)]),
                LineString([(10, 13), (10, 20)])]
        

        self.obstacles = [line.buffer(distance=0.1, cap_style=3) for line in self.lines]

        self.check_in_blocks = [LineString([(5, 4), (6, 4)]),
                                LineString([(4, 15), (5, 15)])]
        self.blocks= [block.buffer(distance=0.1, cap_style=3) for block in self.check_in_blocks]

        # Dictionary to store all robot states in the format:
        # {'robot_id': {'pose': (x, y), 'mode': (R, G, B)}}
        self.robot_positions = {}
        # Create a lock for thread-safe access to shared resources
        self.lock = threading.Lock()

        # List of robot IDs
        self.robot_ids = ['robot1', 'robot2', 'robot3', 'robot4']
        # Create subscribers for each robot topic (e.g., "/robot1/show_position")
        # Create subscribers for each robot topic (e.g., "/robot1/show_position")
        self.position_subscriptions = []
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

        
        # Create a timer for periodic simulation updates (e.g., every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.simulate)

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
        elif mode == "loaded":
            return self.color_mapping.get(robot_id, {}).get("loaded", (255, 255, 255))
        elif mode == "unloaded":
            return self.color_mapping.get(robot_id, {}).get("unloaded", (211, 211, 211))
        else:
            # Default color: white
            return (255, 255, 255)
        
    def transform_coords(self, coord):
        """Convert shapely coordinates to pygame coordinates."""
        x, y = coord
        return int(x * self.world.cell_size ), int(-y * self.world.cell_size + self.world.height)

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
        
        for block in self.blocks:
            if block.geom_type == "Polygon":
                polygon_coords = [self.transform_coords(coord) for coord in block.exterior.coords]
                pygame.draw.polygon(self.world.screen, RED, polygon_coords, 0)  # Filled polygon

        bumps = []
        # coords = [[(13.0, 10.5), (13, 12), (15, 12), (15, 10.5)], \
        #         [(12, 3), (12, 7), (14, 7), (14, 3)], \
        #         [(7, 1), (7, 3), (8, 3), (8, 1)]]
        coords = []
        for coord in coords:
            polygon = Polygon(coord)
            bumps.append(polygon)


        # Define the points dictionary with labels.
        points = {
            (1, 6.5): 'b', 
            (5.5, 9.5): 'c',
            (9, 6.5): 'd',
            (6, 3): 'e', # unload
            (1, 13.5): 'f',
            (9, 16.5): 'g',
            (5, 16): 'h',     # unload
            (11, 13.5): 'i',
            (11, 16.5): 'j',
            (19, 16.5): 'k',
            (16, 13): 'l',    # unload
            (11, 4): 'm',
            (19, 6.5): 'n',
            (16, 5.5): 'o'    # unload
        }

        # Define the set of points that are marked as unloaded.
        unloaded_points = {(6, 3), (5, 16), (16, 13), (16, 5.5)}

        # Define colors.
        GREEN = (108, 247, 80)
        SKY_BLUE = (135, 206, 235)  # 天蓝色

       # Iterate through all points, choose color based on unloaded status, and draw a square.
        for pt, label in points.items():
            if pt in unloaded_points:
                color = GREEN
            else:
                color = SKY_BLUE
            # Convert logical coordinate to pixel coordinate using transform_coords method.
            pixel_pos = self.transform_coords(pt)
            # Define square side length.
            side = self.world.cell_size // 2
            # Create a rectangle with center at pixel_pos.
            rect = pygame.Rect(pixel_pos[0] - side // 2, pixel_pos[1] - side // 2, side, side)
            pygame.draw.rect(self.world.screen, color, rect)
            # Draw the label next to the square.
            font = pygame.font.SysFont("Arial", 16)
            text_surface = font.render(label, True, BLACK)
            self.world.screen.blit(text_surface, (pixel_pos[0] + 5, pixel_pos[1] + 5))

        for bump in bumps:
            if bump.geom_type == "Polygon":
                bump_coords = [self.transform_coords(pt) for pt in bump.exterior.coords]
                pygame.draw.polygon(self.world.screen, YELLOW, bump_coords, 0)
        
        # Draw actions (if available, drawing lines between nodes)
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

            pygame.draw.line(self.world.screen, BLACK, start_pos, end_pos, 1)
        
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
            pygame.draw.circle(self.world.screen, BLUE, (self.nodes[node]['attr']['pose'][0]* self.world.cell_size, \
                            self.world.height - (self.nodes[node]['attr']['pose'][1]* self.world.cell_size)), 3)
    
        # Draw all robot positions (reading shared data under lock)
        with self.lock:
            for robot_id, info in self.robot_positions.items():
                pos = info['pose']
                color = info['mode']
                pixel_pos = ((pos[0] * self.world.cell_size), (self.world.height - pos[1] * self.world.cell_size))
                pygame.draw.circle(self.world.screen, color, pixel_pos, self.world.cell_size // 3)
                font = pygame.font.SysFont("Arial", 16)
                text_surface = font.render(robot_id, True, BLACK)
                self.world.screen.blit(text_surface, (pixel_pos[0] + 5, pixel_pos[1] + 5))
        
        # Capture the screen image for video saving (if needed)
        # pygame_surface = pygame.display.get_surface()
        # pygame_pixels = pygame.surfarray.array3d(pygame_surface)
        # image = np.flipud(pygame_pixels)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # self.world.output_video.write(image)
        
        pygame.display.flip()
        self.world.clock.tick(30)

def main(args=None):
    pygame.init()
    rclpy.init(args=args)
    # Create a node for obtaining parameters
    main_node = rclpy.create_node('showmove_node_main')
    grid_size = main_node.declare_parameter('N', 8).get_parameter_value().integer_value
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
