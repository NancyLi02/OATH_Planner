#!/usr/bin/env python3
import os
import math
import rclpy
from rclpy.node import Node
import pygame
from shapely.geometry import Polygon
from ltl_automaton_planner.ltl_automaton_utilities import (
    import_ts_from_file,
    extract_numbers,
    build_graph_halton,
    load_lines_from_yaml,
    update_graph_with_obstacle,
    add_bump_polygon,
    add_block_polygon,
    BUMP_POLYGONS,
    BLOCK_POLYGONS,
)
import sys
from ltl_automaton_msgs.msg import (
    ShowPosition,
    UpdateValidTasks,
    TaskFail,
    AddTask,
    ObstacleUpdate,
    ChangeTaskPriority,
)
from std_msgs.msg import String
import threading
import yaml
import json
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
LEGEND_BG = (245, 245, 250)
LEGEND_BORDER = (180, 180, 190)

# ---- Multi-type task palette ----
# Distinct colors for up to 12 task types; cycled if there are more.
TYPE_COLORS = [
    (220, 50, 60),     # Red
    (60, 180, 75),     # Green
    (50, 100, 235),    # Blue
    (255, 130, 30),    # Orange
    (155, 60, 200),    # Purple
    (0, 160, 200),     # Teal
    (240, 90, 175),    # Pink
    (140, 100, 50),    # Brown
    (190, 190, 50),    # Olive
    (90, 90, 90),      # Dark Grey
    (50, 200, 200),    # Aqua
    (255, 200, 80),    # Amber
]

# Distinct shapes for up to 8 task types; cycled if there are more.
TYPE_SHAPES = [
    'square',
    'circle',
    'triangle_up',
    'diamond',
    'triangle_down',
    'pentagon',
    'hexagon',
    'star',
]

# Distinct robot base colors (cycled if more robots than entries)
ROBOT_BASE_COLORS = [
    (255, 20, 147),    # DeepPink
    (0, 130, 220),     # Strong Blue
    (50, 180, 80),     # LimeGreen
    (255, 175, 0),     # Gold/Amber
    (155, 80, 220),    # Violet
    (255, 90, 90),     # Tomato Red
    (60, 200, 200),    # Cyan
    (160, 110, 60),    # Brown
]


def _shape_for_type(type_id):
    if type_id is None or type_id < 1:
        return 'square'
    return TYPE_SHAPES[(type_id - 1) % len(TYPE_SHAPES)]


def _color_for_type(type_id):
    if type_id is None or type_id < 1:
        return GREY
    return TYPE_COLORS[(type_id - 1) % len(TYPE_COLORS)]


def _color_for_robot(index):
    return ROBOT_BASE_COLORS[index % len(ROBOT_BASE_COLORS)]


def _lighten(rgb, amount=80):
    return tuple(min(255, c + amount) for c in rgb)


def _draw_marker(screen, shape, center, size, fill_color,
                 border_color=BLACK, border_width=2):
    """Draw a marker shape centered at `center` with bounding size `size`."""
    cx, cy = int(center[0]), int(center[1])
    half = max(2, size // 2)

    if shape == 'square':
        rect = pygame.Rect(cx - half, cy - half, half * 2, half * 2)
        pygame.draw.rect(screen, fill_color, rect)
        if border_color:
            pygame.draw.rect(screen, border_color, rect, border_width)
    elif shape == 'circle':
        pygame.draw.circle(screen, fill_color, (cx, cy), half)
        if border_color:
            pygame.draw.circle(screen, border_color, (cx, cy), half, border_width)
    elif shape == 'triangle_up':
        pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    elif shape == 'triangle_down':
        pts = [(cx, cy + half), (cx - half, cy - half), (cx + half, cy - half)]
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    elif shape == 'diamond':
        pts = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    elif shape == 'pentagon':
        pts = []
        for i in range(5):
            angle = -math.pi / 2 + i * 2 * math.pi / 5
            pts.append((cx + half * math.cos(angle), cy + half * math.sin(angle)))
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    elif shape == 'hexagon':
        pts = []
        for i in range(6):
            angle = i * math.pi / 3
            pts.append((cx + half * math.cos(angle), cy + half * math.sin(angle)))
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    elif shape == 'star':
        pts = []
        for i in range(10):
            r = half if i % 2 == 0 else half // 2
            angle = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        pygame.draw.polygon(screen, fill_color, pts)
        if border_color:
            pygame.draw.polygon(screen, border_color, pts, border_width)
    else:
        # Fallback square
        rect = pygame.Rect(cx - half, cy - half, half * 2, half * 2)
        pygame.draw.rect(screen, fill_color, rect)
        if border_color:
            pygame.draw.rect(screen, border_color, rect, border_width)

# ---------- Walls ----------
# The local load_lines_from_yaml function is removed.
# We will use the one from ltl_automaton_utilities.

class GridWorld(object):
    def __init__(self, grid_size, legend_width=320):
        self.grid_size = grid_size
        # The map area stays 800x800 (same as before); a side panel hosts the legend.
        self.grid_width = 800
        self.height = 800
        self.legend_width = legend_width
        self.width = self.grid_width + self.legend_width
        self.cell_size = self.grid_width // self.grid_size
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

        # ---- Multi-type task definitions ----
        # type_labels_map: {type_id -> [task_labels]} parsed from typeN_labels keys
        self.type_labels_map = {}
        max_type_id = 0
        for key, value in yaml_data.items():
            m = re.match(r"type(\d+)_labels", key)
            if m and value:
                tid = int(m.group(1))
                self.type_labels_map[tid] = list(value)
                max_type_id = max(max_type_id, tid)
        self.num_task_types = max_type_id if max_type_id > 0 else 0

        # task_label -> type_id
        self.task_label_to_type = {}
        for tid, labels in self.type_labels_map.items():
            for lbl in labels:
                self.task_label_to_type[lbl] = tid

        # robot_capabilities: list[list[int]] aligned with robot_ids
        self.robot_ids = list(yaml_data.get('robot_positions', {}).keys())
        robot_caps_dict = yaml_data.get('robot_capabilities', {}) or {}
        self.robot_capabilities = {}
        for rid in self.robot_ids:
            cap = robot_caps_dict.get(rid)
            if cap is None:
                cap = [1] * max(1, self.num_task_types)
            else:
                cap = list(cap)
                if self.num_task_types > 0 and len(cap) < self.num_task_types:
                    cap = cap + [0] * (self.num_task_types - len(cap))
                elif self.num_task_types > 0 and len(cap) > self.num_task_types:
                    cap = cap[: self.num_task_types]
            self.robot_capabilities[rid] = cap

        # Per-robot color (loaded/unloaded shade) determined by index in robot_ids.
        self.color_mapping = {}
        for idx, robot_id in enumerate(self.robot_ids):
            base = _color_for_robot(idx)
            self.color_mapping[robot_id] = {
                'loaded': base,
                'unloaded': _lighten(base, 80),
                'base': base,
            }

        # Backward-compat shims (kept so old code paths don't break)
        self.special_robot_ids = []
        self.special_labels = set()
        
        self.failed_task_list = []

        self.new_task_points = []
        self.map_needs_update = False
        self.processed_obstacles = set()
        
        # High priority task tracking for visual indication
        self.high_priority_tasks = set()  # Set of task labels with high priority
        self.blink_state = True  # Toggle state for blinking effect
        self.blink_counter = 0  # Counter for controlling blink speed
        
        # Track robots that have been assigned a new task but haven't started executing yet
        # This prevents position_callback from overwriting the 'waiting' state with 'NoTask'
        self.robots_waiting_for_plan = set()

        # Initialize subscriptions lists
        self.position_subscriptions = []
        self.update_valid_tasks_subs = []
        self.task_failure_subs = []
        
        # Initialize finished tasks set to store completed task indices
        self.finished_tasks = set()

        # ========== TIMING DATA COLLECTION ==========
        # Timing data storage
        self.timing_data = {
            'ui_inputs': [],           # UI input timing records
            'llm_processing': [],      # LLM processing timing records
            'command_published': [],   # Command parsing/publishing timing records
            'system_responses': [],    # System response timing records (add_task, obstacle, priority)
            'robot_finished': {},      # Robot completion records {agent_name: {total_steps, timestamp}}
        }
        self.timing_summary_printed = False

        self.start_time = time.time()  
        self.end_time = None  
        self.timing_completed = False 
        self.get_logger().info(f"Start timing at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")

        # Dictionary to store waypoint history for each robot
        # Format: {'robot_id': [[x1, y1], [x2, y2], ...]}
        self.robot_waypoints = {}
        self.waypoints_saved = False
        # Save to package share directory (install/ltl_automaton_planner/share/ltl_automaton_planner/config/)
        self.waypoints_file = os.path.join(package_share, 'config', 'robot_waypoints.yaml')
        self.get_logger().info(f"📁 Waypoints will be saved to: {os.path.abspath(self.waypoints_file)}")

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

        # Subscribe to change_task_priority topic
        self.change_task_priority_sub = self.create_subscription(
            ChangeTaskPriority,
            '/change_task_priority',
            self.change_task_priority_callback,
            10
        )

        # Subscribe to timing data from all nodes
        self.timing_data_sub = self.create_subscription(
            String,
            '/timing_data',
            self.timing_data_callback,
            10
        )

        # Subscribe to immediate robot state updates from taskassign_cluster_node
        self.robot_state_update_sub = self.create_subscription(
            String,
            '/robot_state_update',
            self.robot_state_update_callback,
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

    def robot_state_update_callback(self, msg):
        """
        Handle immediate robot state update from taskassign_cluster_node.
        This is called RIGHT AFTER task assignment, BEFORE planner builds automaton.
        """
        try:
            data = json.loads(msg.data)
            if data.get('type') != 'robot_state_update':
                return
            
            robot_id = data.get('robot_id')
            new_state = data.get('new_state')
            
            if not robot_id:
                return
            
            self.get_logger().info(f"📥 Received immediate state update: {robot_id} -> {new_state}")
            
            with self.lock:
                # Check if robot was in no_task state
                was_notask = False
                if robot_id in self.robot_positions:
                    if self.robot_positions[robot_id]['mode'] == self.notask_color:
                        was_notask = True
                
                # Update robot state to waiting color immediately
                if new_state == 'waiting':
                    # Add to waiting set to prevent position_callback from overwriting
                    self.robots_waiting_for_plan.add(robot_id)
                    
                    if robot_id in self.robot_positions:
                        self.robot_positions[robot_id]['mode'] = self.waiting_color
                    else:
                        # Robot not in positions yet, create entry with waiting color
                        # Use last known position or default
                        self.robot_positions[robot_id] = {
                            'pose': [0, 0],  # Will be updated by next position message
                            'mode': self.waiting_color
                        }
                    
                    self.get_logger().info(f"🔄 [{robot_id}] Immediately changed to WAITING (task assigned, waiting for plan)")
                
                # Reset timing if robot was in no_task state
                if was_notask:
                    self.timing_completed = False
                    self.timing_summary_printed = False
                    self.get_logger().info(f"⏱️ [{robot_id}] Reset timing - continuing to count...")
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse robot state update: {e}")

    def timing_data_callback(self, msg):
        """Process incoming timing data from various nodes"""
        try:
            data = json.loads(msg.data)
            timing_type = data.get('type', 'unknown')
            
            if timing_type == 'ui_input':
                duration = data.get('duration', 0)
                if duration < 0:
                    self.get_logger().warn(f"[TIMING WARNING] Negative UI input duration detected: {duration:.4f}s (clock sync issue?)")
                    data['duration'] = max(0.0, duration)
                self.timing_data['ui_inputs'].append(data)
                self.get_logger().info(f"[TIMING COLLECTED] UI input: {data.get('duration', 0):.4f}s")
                
            elif timing_type == 'llm_processing':
                duration = data.get('duration', 0)
                if duration < 0:
                    self.get_logger().warn(f"[TIMING WARNING] Negative LLM processing duration detected: {duration:.4f}s (clock sync issue?)")
                    data['duration'] = max(0.0, duration)
                self.timing_data['llm_processing'].append(data)
                self.get_logger().info(f"[TIMING COLLECTED] LLM processing: {data.get('duration', 0):.4f}s, intent: {data.get('intent', 'unknown')}")
                
            elif timing_type == 'command_published':
                total_parse_duration = data.get('total_parse_duration', 0)
                llm_call_duration = data.get('llm_call_duration', 0)
                if total_parse_duration < 0:
                    self.get_logger().warn(f"[TIMING WARNING] Negative parse duration detected: {total_parse_duration:.4f}s (clock sync issue?)")
                    data['total_parse_duration'] = max(0.0, total_parse_duration)
                if llm_call_duration < 0:
                    self.get_logger().warn(f"[TIMING WARNING] Negative LLM call duration detected: {llm_call_duration:.4f}s (clock sync issue?)")
                    data['llm_call_duration'] = max(0.0, llm_call_duration)
                self.timing_data['command_published'].append(data)
                self.get_logger().info(f"[TIMING COLLECTED] Command published: {data.get('command', 'unknown')}, parse duration: {data.get('total_parse_duration', 0):.4f}s")
                
            elif timing_type == 'system_response':
                duration = data.get('duration', 0)
                if duration < 0:
                    self.get_logger().warn(f"[TIMING WARNING] Negative system response duration detected: {duration:.4f}s (clock sync issue?)")
                    data['duration'] = max(0.0, duration)
                self.timing_data['system_responses'].append(data)
                self.get_logger().info(f"[TIMING COLLECTED] System response: {data.get('command', 'unknown')}, duration: {data.get('duration', 0):.4f}s")
                
            elif timing_type == 'robot_finished':
                agent_name = data.get('agent_name', 'unknown')
                self.timing_data['robot_finished'][agent_name] = data
                dist = data.get('total_distance', 0.0)
                self.get_logger().info(f"[TIMING COLLECTED] Robot {agent_name} finished: {data.get('total_steps', 0)} steps, total distance = {dist:.4f}")
                
            elif timing_type == 'instruction_sent':
                # Store instruction sent timing for correlation
                pass  # Can be used for more detailed analysis if needed
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse timing data: {e}")

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
                self.get_logger().info("📊 Planner Running Time:")
                self.get_logger().info(f"   Starting Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
                self.get_logger().info(f"   Ending Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}")
                self.get_logger().info(f"   Total Time: {total_time:.2f} seconds")
                self.get_logger().info(f"   Robot number: {len(self.robot_ids)}")
                self.get_logger().info("="*60)
                
                # Print comprehensive timing summary
                self.print_timing_summary()
                
                # Save waypoints when all tasks are completed
                # self.save_waypoints_to_yaml()
                
                return True
        return False

    def print_timing_summary(self):
        """Print comprehensive timing summary when all tasks are completed"""
        if self.timing_summary_printed:
            return
        self.timing_summary_printed = True
        
        self.get_logger().info("")
        self.get_logger().info("="*80)
        self.get_logger().info("📊 COMPREHENSIVE TIMING SUMMARY")
        self.get_logger().info("="*80)
        
        # 1. UI Input Timing Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 1. UI INPUT TIMING (User typing to message sent)                           │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        if self.timing_data['ui_inputs']:
            ui_durations = [d['duration'] for d in self.timing_data['ui_inputs']]
            self.get_logger().info(f"│   Total UI inputs: {len(ui_durations):<58}│")
            self.get_logger().info(f"│   Average input time: {sum(ui_durations)/len(ui_durations):.4f}s{' '*48}│")
            self.get_logger().info(f"│   Min input time: {min(ui_durations):.4f}s{' '*52}│")
            self.get_logger().info(f"│   Max input time: {max(ui_durations):.4f}s{' '*52}│")
            self.get_logger().info(f"│   Total input time: {sum(ui_durations):.4f}s{' '*50}│")
        else:
            self.get_logger().info("│   No UI input timing data collected                                         │")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        
        # 2. LLM Processing Timing Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 2. LLM PROCESSING TIMING (Processing user input and generating response)   │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        if self.timing_data['llm_processing']:
            llm_durations = [d['duration'] for d in self.timing_data['llm_processing']]
            self.get_logger().info(f"│   Total LLM calls: {len(llm_durations):<58}│")
            self.get_logger().info(f"│   Average processing time: {sum(llm_durations)/len(llm_durations):.4f}s{' '*42}│")
            self.get_logger().info(f"│   Min processing time: {min(llm_durations):.4f}s{' '*46}│")
            self.get_logger().info(f"│   Max processing time: {max(llm_durations):.4f}s{' '*46}│")
            self.get_logger().info(f"│   Total processing time: {sum(llm_durations):.4f}s{' '*44}│")
            # Group by intent
            intent_times = {}
            for d in self.timing_data['llm_processing']:
                intent = d.get('intent', 'unknown')
                if intent not in intent_times:
                    intent_times[intent] = []
                intent_times[intent].append(d['duration'])
            self.get_logger().info("│   By intent:                                                                │")
            for intent, times in intent_times.items():
                avg_time = sum(times) / len(times)
                self.get_logger().info(f"│     - {intent}: count={len(times)}, avg={avg_time:.4f}s{' '*(51-len(intent)-len(str(len(times))))}│")
        else:
            self.get_logger().info("│   No LLM processing timing data collected                                   │")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        
        # 3. Command Parsing/Publishing Timing Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 3. COMMAND PARSING & PUBLISHING TIMING (LLM parser to system)              │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        if self.timing_data['command_published']:
            parse_durations = [d['total_parse_duration'] for d in self.timing_data['command_published']]
            llm_call_durations = [d.get('llm_call_duration', 0) for d in self.timing_data['command_published']]
            self.get_logger().info(f"│   Total commands parsed: {len(parse_durations):<52}│")
            self.get_logger().info(f"│   Average total parse time: {sum(parse_durations)/len(parse_durations):.4f}s{' '*40}│")
            self.get_logger().info(f"│   Average LLM call time: {sum(llm_call_durations)/len(llm_call_durations):.4f}s{' '*43}│")
            # Group by command type
            cmd_times = {}
            for d in self.timing_data['command_published']:
                cmd = d.get('command', 'unknown')
                if cmd not in cmd_times:
                    cmd_times[cmd] = []
                cmd_times[cmd].append(d['total_parse_duration'])
            self.get_logger().info("│   By command type:                                                          │")
            for cmd, times in cmd_times.items():
                avg_time = sum(times) / len(times)
                self.get_logger().info(f"│     - {cmd}: count={len(times)}, avg={avg_time:.4f}s{' '*(51-len(cmd)-len(str(len(times))))}│")
        else:
            self.get_logger().info("│   No command parsing timing data collected                                  │")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        
        # 4. System Response Timing Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 4. SYSTEM RESPONSE TIMING (Robot system executing commands)                │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        if self.timing_data['system_responses']:
            # Group by command type
            response_by_cmd = {}
            for d in self.timing_data['system_responses']:
                cmd = d.get('command', 'unknown')
                if cmd not in response_by_cmd:
                    response_by_cmd[cmd] = []
                response_by_cmd[cmd].append(d)
            
            for cmd, responses in response_by_cmd.items():
                durations = [r['duration'] for r in responses]
                self.get_logger().info(f"│   {cmd}:                                                                  │"[:78] + "│")
                self.get_logger().info(f"│     - Count: {len(durations):<62}│")
                self.get_logger().info(f"│     - Average response time: {sum(durations)/len(durations):.4f}s{' '*38}│")
                self.get_logger().info(f"│     - Min response time: {min(durations):.4f}s{' '*42}│")
                self.get_logger().info(f"│     - Max response time: {max(durations):.4f}s{' '*42}│")
                self.get_logger().info(f"│     - Total response time: {sum(durations):.4f}s{' '*40}│")
        else:
            self.get_logger().info("│   No system response timing data collected                                  │")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        
        # 5. Robot Step Count & Distance Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 5. ROBOT STEP COUNT & DISTANCE SUMMARY                                       │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        if self.timing_data['robot_finished']:
            total_steps = 0
            total_distance_all = 0.0
            for agent_name, data in sorted(self.timing_data['robot_finished'].items()):
                steps = data.get('total_steps', 0)
                dist = data.get('total_distance', 0.0)
                total_steps += steps
                total_distance_all += dist
                self.get_logger().info(f"│   {agent_name}: {steps} steps, distance = {dist:.4f}{' '*(38-len(agent_name)-len(str(steps))-len(f'{dist:.4f}'))}│")
            self.get_logger().info("│   ─────────────────────────────────────────────────────────────────────── │")
            self.get_logger().info(f"│   TOTAL STEPS (ALL ROBOTS): {total_steps:<48}│")
            self.get_logger().info(f"│   TOTAL DISTANCE (ALL ROBOTS): {total_distance_all:.4f}{' '*(44-len(f'{total_distance_all:.4f}'))}│")
        else:
            self.get_logger().info("│   No robot step/distance data collected                                    │")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        
        # 6. Overall Summary
        self.get_logger().info("")
        self.get_logger().info("┌─────────────────────────────────────────────────────────────────────────────┐")
        self.get_logger().info("│ 6. OVERALL SUMMARY                                                          │")
        self.get_logger().info("├─────────────────────────────────────────────────────────────────────────────┤")
        total_task_time = self.end_time - self.start_time if self.end_time else 0
        self.get_logger().info(f"│   Total task completion time: {total_task_time:.2f} seconds{' '*(39-len(f'{total_task_time:.2f}'))}│")
        self.get_logger().info(f"│   Number of robots: {len(self.robot_ids):<56}│")
        
        total_steps_all = sum(d.get('total_steps', 0) for d in self.timing_data['robot_finished'].values())
        total_distance_all = sum(d.get('total_distance', 0.0) for d in self.timing_data['robot_finished'].values())
        self.get_logger().info(f"│   Total steps (all robots): {total_steps_all:<48}│")
        self.get_logger().info(f"│   Total distance (all robots): {total_distance_all:.4f}{' '*(42-len(f'{total_distance_all:.4f}'))}│")
        
        total_ui_time = sum(d['duration'] for d in self.timing_data['ui_inputs']) if self.timing_data['ui_inputs'] else 0
        total_llm_time = sum(d['duration'] for d in self.timing_data['llm_processing']) if self.timing_data['llm_processing'] else 0
        total_parse_time = sum(d['total_parse_duration'] for d in self.timing_data['command_published']) if self.timing_data['command_published'] else 0
        total_response_time = sum(d['duration'] for d in self.timing_data['system_responses']) if self.timing_data['system_responses'] else 0
        
        self.get_logger().info(f"│   Total UI input time: {total_ui_time:.4f}s{' '*(46-len(f'{total_ui_time:.4f}'))}│")
        self.get_logger().info(f"│   Total LLM processing time: {total_llm_time:.4f}s{' '*(40-len(f'{total_llm_time:.4f}'))}│")
        self.get_logger().info(f"│   Total command parsing time: {total_parse_time:.4f}s{' '*(39-len(f'{total_parse_time:.4f}'))}│")
        self.get_logger().info(f"│   Total system response time: {total_response_time:.4f}s{' '*(39-len(f'{total_response_time:.4f}'))}│")
        self.get_logger().info("└─────────────────────────────────────────────────────────────────────────────┘")
        self.get_logger().info("")
        self.get_logger().info("="*80)

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
        self.get_logger().info(
            f"Received add task command from LLM, new task type='{msg.task_type}' appears at {msg.location}, updating map......"
        )
        point = tuple(msg.location)
        label = msg.task_label

        if point in self.points:
            self.get_logger().info(f"Task point already exists at {point}, skipping...")
            return

        self.points[point] = label

        # Try to parse the task_type as a type id (int or 'typeN'); fall back to type 1.
        type_id = self._parse_task_type_field(msg.task_type)
        if type_id is None or type_id < 1:
            type_id = 1
        if self.num_task_types > 0 and type_id > self.num_task_types:
            self.num_task_types = type_id
            # Pad existing robot capability vectors with zeros for the new type
            for rid in self.robot_ids:
                cap = self.robot_capabilities.get(rid, [])
                if len(cap) < self.num_task_types:
                    self.robot_capabilities[rid] = list(cap) + [0] * (self.num_task_types - len(cap))
        self.type_labels_map.setdefault(type_id, []).append(label)
        self.task_label_to_type[label] = type_id

        # Remove from finished_tasks if exists (this is a new task, should not be marked as finished)
        if label in self.finished_tasks:
            self.finished_tasks.discard(label)
            self.get_logger().info(f"Removed '{label}' from finished_tasks as it's a new task")

    @staticmethod
    def _parse_task_type_field(task_type_str):
        """Convert AddTask.task_type string into a 1-based int type id; None if unparsable."""
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

    def change_task_priority_callback(self, msg):
        """Handle task priority changes for visual indication"""
        task_label = msg.task_label
        priority = msg.priority
        
        self.get_logger().info(f"Received task priority change: '{task_label}' to '{priority}'")
        
        if priority == 'high':
            self.high_priority_tasks.add(task_label)
            self.get_logger().info(f"Task '{task_label}' marked as HIGH PRIORITY for visual indication")
        else:
            self.high_priority_tasks.discard(task_label)
            self.get_logger().info(f"Task '{task_label}' removed from high priority visual indication")
        

    
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
        # Remove from high priority set when task is completed
        if label in self.high_priority_tasks:
            self.high_priority_tasks.discard(label)
            self.get_logger().info(f"Task '{label}' completed, removed from high priority visual indication")

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
            # Check if this robot was previously in no_task state and is now active again
            previous_color = None
            if msg.robot_id in self.robot_positions:
                previous_color = self.robot_positions[msg.robot_id]['mode']
            
            # Check if robot is waiting for plan (has been assigned task but planner hasn't finished)
            if msg.robot_id in self.robots_waiting_for_plan:
                if msg.mode == 'NoTask':
                    # Robot is still waiting for plan, keep waiting color, only update position
                    self.robot_positions[msg.robot_id] = {'pose': pos, 'mode': self.waiting_color}
                    # Don't update color, don't record waypoint, don't check completion
                    return
                else:
                    # Robot received plan and started executing, remove from waiting set
                    self.robots_waiting_for_plan.discard(msg.robot_id)
                    self.get_logger().info(f"✅ [{msg.robot_id}] Received plan and started executing (mode: {msg.mode})")
            
            self.robot_positions[msg.robot_id] = {'pose': pos, 'mode': color}
            
            # If robot was in no_task state (notask_color) and now has a different state,
            # it means the robot received a new task - reset timing to continue counting
            if previous_color == self.notask_color and color != self.notask_color:
                self.get_logger().info(f"🔄 [{msg.robot_id}] Recovered from no_task state (received new task), continuing timing...")
                self.timing_completed = False
                self.timing_summary_printed = False
            
            # Record waypoint for this robot
            self._record_waypoint(msg.robot_id, pos)
            
            self.check_all_robots_notask()
        # self.get_logger().info(f"Received {msg.robot_id}: position {pos}, mode {msg.mode}")

    def _record_waypoint(self, robot_id, pos):
        """
        Record the waypoint for a robot. Only adds if position has changed significantly.
        """
        # Convert pos to list format [x, y]
        waypoint = [float(pos[0]), float(pos[1])]
        
        # Initialize list for this robot if not exists
        if robot_id not in self.robot_waypoints:
            self.robot_waypoints[robot_id] = []
        
        # Only add if this is a new position (avoid duplicates)
        if len(self.robot_waypoints[robot_id]) == 0:
            self.robot_waypoints[robot_id].append(waypoint)
        else:
            last_waypoint = self.robot_waypoints[robot_id][-1]
            # Check if position has changed (with small tolerance to avoid floating point issues)
            distance = ((waypoint[0] - last_waypoint[0])**2 + (waypoint[1] - last_waypoint[1])**2)**0.5
            if distance > 0.01:  # Only record if moved more than 0.01 units
                self.robot_waypoints[robot_id].append(waypoint)

    def save_waypoints_to_yaml(self):
        """
        Save all recorded robot waypoints to a YAML file.
        """
        if self.waypoints_saved:
            return
        
        self.waypoints_saved = True
        
        # Prepare data in the required format
        waypoints_data = {}
        for robot_id in sorted(self.robot_waypoints.keys()):
            waypoints_data[robot_id] = self.robot_waypoints[robot_id]
        
        # Write to YAML file with custom formatting
        try:
            with open(self.waypoints_file, 'w') as f:
                f.write("# Waypoints configuration for robots\n")
                f.write("# Each robot has a list of [x, y] waypoints to follow sequentially\n")
                f.write(f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for robot_id in sorted(waypoints_data.keys()):
                    f.write(f"{robot_id}:\n")
                    for waypoint in waypoints_data[robot_id]:
                        f.write(f"  - [{waypoint[0]:.4f}, {waypoint[1]:.4f}]\n")
                    f.write("\n")
            
            abs_path = os.path.abspath(self.waypoints_file)
            self.get_logger().info("="*60)
            self.get_logger().info("📁 WAYPOINTS FILE SAVED!")
            self.get_logger().info(f"📍 Full path: {abs_path}")
            self.get_logger().info("="*60)
            self.get_logger().info(f"   Total robots: {len(waypoints_data)}")
            for robot_id, waypoints in waypoints_data.items():
                self.get_logger().info(f"   {robot_id}: {len(waypoints)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Failed to save waypoints: {e}")

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
        
        # Update blink state for high priority task indication (toggle every 5 frames)
        self.blink_counter += 1
        if self.blink_counter >= 5:
            self.blink_counter = 0
            self.blink_state = not self.blink_state
        
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

        marker_size = max(12, self.world.cell_size // 2 + 4)
        label_font = pygame.font.SysFont("Arial", 16)

        # Iterate through all points, choose color/shape based on type and status:
        for pt, label in points.items():
            pixel_pos = self.transform_coords(pt)
            is_high_priority = label in self.high_priority_tasks
            is_finished = label in self.finished_tasks

            if pt in unloaded_points:
                # Delivery points: keep a clear, fixed look (green diamond with black border)
                _draw_marker(
                    self.world.screen,
                    'diamond',
                    pixel_pos,
                    marker_size + 4,
                    GREEN,
                    border_color=BLACK,
                    border_width=2,
                )
            else:
                type_id = self.task_label_to_type.get(label)
                shape = _shape_for_type(type_id)
                fill_color = self.finished_tasks_color if is_finished else _color_for_type(type_id)
                _draw_marker(
                    self.world.screen,
                    shape,
                    pixel_pos,
                    marker_size,
                    fill_color,
                    border_color=BLACK,
                    border_width=2,
                )

            # High priority indicator: blinking red ring
            if is_high_priority and not is_finished:
                if self.blink_state:
                    pygame.draw.circle(
                        self.world.screen, (255, 0, 0), pixel_pos, marker_size + 6, 3
                    )

            # Failed-task indicator: red ring around the marker
            if pt not in unloaded_points and label in self.failed_task_list:
                pygame.draw.circle(self.world.screen, RED, pixel_pos, marker_size, 2)

            # Label text
            text_surface = label_font.render(label, True, BLACK)
            self.world.screen.blit(text_surface, (pixel_pos[0] + 5, pixel_pos[1] + 5))

        # Draw bumps (e.g., bushes) in YELLOW
        bumps = BUMP_POLYGONS
        for bump in bumps:
            if bump.geom_type == "Polygon":
                bump_coords = [self.transform_coords(pt) for pt in bump.exterior.coords]
                pygame.draw.polygon(self.world.screen, YELLOW, bump_coords, 0)
        
        edge_surface = pygame.Surface((self.world.width, self.world.height), pygame.SRCALPHA)
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

            pygame.draw.line(edge_surface, (200, 200, 200, 50), start_pos, end_pos, 1)
        self.world.screen.blit(edge_surface, (0, 0))
        
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
            
        # Draw nodes (subtle, semi-transparent)
        node_surface = pygame.Surface((self.world.width, self.world.height), pygame.SRCALPHA)
        for node in self.nodes:
            pos = (int(self.nodes[node]['attr']['pose'][0] * self.world.cell_size),
                   int(self.world.height - (self.nodes[node]['attr']['pose'][1] * self.world.cell_size)))
            pygame.draw.circle(node_surface, (160, 160, 200, 60), pos, 2)
        self.world.screen.blit(node_surface, (0, 0))
    
        # Draw all robot positions (reading shared data under lock).
        # All robots share the same circular shape; the body color reflects
        # mode (loaded/unloaded/waiting/notask/fail) and a numeric id label is
        # drawn on top of the circle.
        radius = max(10, self.world.cell_size // 2)
        with self.lock:
            for robot_id, info in self.robot_positions.items():
                pos = info['pose']
                color = info['mode']
                pixel_pos = (
                    int(pos[0] * self.world.cell_size),
                    int(self.world.height - pos[1] * self.world.cell_size),
                )

                # Filled body
                pygame.draw.circle(self.world.screen, color, pixel_pos, radius)
                # Black outline
                pygame.draw.circle(self.world.screen, BLACK, pixel_pos, radius, 2)

                # Numeric id (extract trailing digits, e.g. 'robot3' -> '3')
                rid_match = re.search(r'(\d+)$', robot_id)
                rid_text = rid_match.group(1) if rid_match else robot_id
                id_font = pygame.font.SysFont("Arial", 18, bold=True)
                # Use white-ish text on darker fills, black text otherwise
                text_color = WHITE if sum(color) < 380 else BLACK
                text_surface = id_font.render(rid_text, True, text_color)
                text_rect = text_surface.get_rect(center=pixel_pos)
                self.world.screen.blit(text_surface, text_rect)

        # Side panel: legend for task types and robot capabilities
        self._draw_legend()

        pygame.display.flip()
        self.world.clock.tick(30)

    # ===================== Legend =====================

    def _draw_legend(self):
        """Render a side-panel legend covering task types and robot capabilities."""
        screen = self.world.screen
        x0 = self.world.grid_width
        panel_rect = pygame.Rect(x0, 0, self.world.legend_width, self.world.height)
        pygame.draw.rect(screen, LEGEND_BG, panel_rect)
        pygame.draw.line(screen, LEGEND_BORDER, (x0, 0), (x0, self.world.height), 2)

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        section_font = pygame.font.SysFont("Arial", 16, bold=True)
        item_font = pygame.font.SysFont("Arial", 14)

        margin_x = 14
        cur_y = 14

        title = title_font.render("Legend", True, BLACK)
        screen.blit(title, (x0 + margin_x, cur_y))
        cur_y += 28

        # ---- Task Types ----
        section = section_font.render("Task Types", True, BLACK)
        screen.blit(section, (x0 + margin_x, cur_y))
        cur_y += 22

        # Show each declared type with marker + count of labels
        type_ids_sorted = sorted(self.type_labels_map.keys()) if self.type_labels_map else []
        marker_x = x0 + margin_x + 10
        text_x = x0 + margin_x + 36

        for tid in type_ids_sorted:
            if cur_y > self.world.height - 200:
                break  # keep room for the robot legend section
            shape = _shape_for_type(tid)
            color = _color_for_type(tid)
            _draw_marker(screen, shape, (marker_x, cur_y + 8), 18, color,
                         border_color=BLACK, border_width=2)
            count = len(self.type_labels_map.get(tid, []))
            line = item_font.render(f"Type {tid}  ({count} tasks)", True, BLACK)
            screen.blit(line, (text_x, cur_y + 1))
            cur_y += 22

        # Delivery point marker reference
        cur_y += 4
        _draw_marker(screen, 'diamond', (marker_x, cur_y + 8), 18, GREEN,
                     border_color=BLACK, border_width=2)
        screen.blit(item_font.render("Delivery point", True, BLACK), (text_x, cur_y + 1))
        cur_y += 24

        # ---- Robots ----
        cur_y += 6
        section = section_font.render("Robots & Capabilities", True, BLACK)
        screen.blit(section, (x0 + margin_x, cur_y))
        cur_y += 22

        rid_font = pygame.font.SysFont("Arial", 14, bold=True)
        for idx, rid in enumerate(self.robot_ids):
            if cur_y > self.world.height - 28:
                break
            base = _color_for_robot(idx)
            # Filled circle with id digit
            pygame.draw.circle(screen, base, (marker_x, cur_y + 8), 10)
            pygame.draw.circle(screen, BLACK, (marker_x, cur_y + 8), 10, 2)
            rid_match = re.search(r'(\d+)$', rid)
            rid_digit = rid_match.group(1) if rid_match else str(idx + 1)
            text_color = WHITE if sum(base) < 380 else BLACK
            id_surf = rid_font.render(rid_digit, True, text_color)
            id_rect = id_surf.get_rect(center=(marker_x, cur_y + 8))
            screen.blit(id_surf, id_rect)

            # Capability list, e.g. "R3: T1, T3, T5"
            cap = self.robot_capabilities.get(rid, [])
            allowed_types = [str(i + 1) for i, v in enumerate(cap) if int(v) == 1]
            cap_text = ", ".join(f"T{t}" for t in allowed_types) if allowed_types else "—"
            line = item_font.render(f"{rid}: {cap_text}", True, BLACK)
            screen.blit(line, (text_x, cur_y + 1))
            cur_y += 22

            # If capability text is long, also draw small type marker chips on next line
            if cap and any(int(v) == 1 for v in cap):
                chip_x = text_x
                chip_y = cur_y + 2
                for tid_idx, v in enumerate(cap):
                    if int(v) != 1:
                        continue
                    tid = tid_idx + 1
                    _draw_marker(
                        screen,
                        _shape_for_type(tid),
                        (chip_x + 7, chip_y + 7),
                        12,
                        _color_for_type(tid),
                        border_color=BLACK,
                        border_width=1,
                    )
                    chip_x += 18
                cur_y += 18

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
        # Save waypoints before shutting down (in case not saved yet)
        # showposition.save_waypoints_to_yaml()
        showposition.destroy_node()
        main_node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
