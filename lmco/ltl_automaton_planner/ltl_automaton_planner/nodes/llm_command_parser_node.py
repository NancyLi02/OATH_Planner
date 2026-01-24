import os
import json
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ltl_automaton_msgs.msg import AddTask, ObstacleUpdate, ChangeTaskPriority
from openai import OpenAI
import yaml
from ament_index_python.packages import get_package_share_directory

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def expand_line_to_rectangle(vertices, thickness=0.2):
    """
    If vertices define a line segment (2 points), expand it to a rectangle with given thickness.
    For example: [[7, 5], [8, 5]] with thickness 0.2 becomes a rectangle:
    [[7, 4.9], [8, 4.9], [8, 5.1], [7, 5.1]]
    """
    if len(vertices) != 2:
        return vertices  # Not a line segment, return as-is
    
    p1 = vertices[0]
    p2 = vertices[1]
    
    # Direction vector from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Length of the line
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return vertices  # Points are the same, can't create rectangle
    
    # Normalized perpendicular vector (rotate 90 degrees)
    # Perpendicular to (dx, dy) is (-dy, dx)
    perp_x = -dy / length
    perp_y = dx / length
    
    # Half thickness offset
    offset = thickness / 2.0
    
    # Create 4 corners of the rectangle
    # Order: counter-clockwise to form a proper polygon
    corner1 = [p1[0] - perp_x * offset, p1[1] - perp_y * offset]  # p1 - offset
    corner2 = [p2[0] - perp_x * offset, p2[1] - perp_y * offset]  # p2 - offset
    corner3 = [p2[0] + perp_x * offset, p2[1] + perp_y * offset]  # p2 + offset
    corner4 = [p1[0] + perp_x * offset, p1[1] + perp_y * offset]  # p1 + offset
    
    return [corner1, corner2, corner3, corner4]

class LLMCommandParserNode(Node):
    def __init__(self):
        super().__init__('llm_command_parser_node')

        self.subscription = self.create_subscription(
            String,
            '/human_instruction',
            self.instruction_callback,
            10
        )

        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        robot_names = list(yaml_data.get('robot_positions', {}).keys())
        self.task_pubs = {}
        for robot in robot_names:
            topic = f'/{robot}/add_task'
            self.task_pubs[robot] = self.create_publisher(AddTask, topic, 10)
        
        self.obstacle_pubs = {}
        for robot in robot_names:
            topic = f'/{robot}/obstacle_update'
            self.obstacle_pubs[robot] = self.create_publisher(ObstacleUpdate, topic, 10)

        self.change_task_priority_pubs = {}
        for robot in robot_names:
            topic = f'/{robot}/change_task_priority'
            self.change_task_priority_pubs[robot] = self.create_publisher(ChangeTaskPriority, topic, 10)

        self.task_pub = self.create_publisher(AddTask, '/add_task', 10)

        self.obstacle_pub = self.create_publisher(ObstacleUpdate, '/obstacle_update', 10)

        self.change_task_priority_pub = self.create_publisher(ChangeTaskPriority, '/change_task_priority', 10)

        # Timing data publisher
        self.timing_pub = self.create_publisher(String, '/timing_data', 10)

        self.get_logger().info('LLM Command Parser Node initialized.')

    def instruction_callback(self, msg):
        instruction = msg.data
        self.get_logger().info(f'Received instruction: "{instruction}"')
        
        # Record parsing start time using monotonic clock (not affected by system clock changes)
        parse_start_mono = time.monotonic()
        parse_start_time = time.time()  # For timestamp reference

        try:
            parsed_intents = self.call_llm(instruction)
            
            # Record LLM call completion time using monotonic clock
            llm_call_end_mono = time.monotonic()
            llm_call_duration = llm_call_end_mono - parse_start_mono
            self.get_logger().info(f'[TIMING] LLM call completed. Duration: {llm_call_duration:.4f}s')

            if not isinstance(parsed_intents, list):
                raise ValueError("LLM response is not a list")

            for intent_obj in parsed_intents:
                intent = intent_obj.get("intent", "")
                parameters = intent_obj.get("parameters", {})
                
                # Record command publish start time
                publish_start_time = time.time()

                if intent == "add_task":
                    msg_out = AddTask()
                    msg_out.location = parameters.get("location", [0.0, 0.0])
                    msg_out.task_label = parameters.get("task_label", "")
                    msg_out.delivery_point = parameters.get("delivery_point", "")
                    msg_out.task_type = parameters.get("task_type", "normal")
                    self.publish_add_task(msg_out)
                    self.get_logger().info(f'Published AddTask: {msg_out}')
                    
                    # Publish timing for add_task command (using monotonic time for accurate duration)
                    publish_end_mono = time.monotonic()
                    total_parse_duration = publish_end_mono - parse_start_mono
                    timing_msg = String()
                    timing_msg.data = json.dumps({
                        'type': 'command_published',
                        'command': 'add_task',
                        'task_label': msg_out.task_label,
                        'parse_start_time': parse_start_time,
                        'llm_call_duration': llm_call_duration,
                        'publish_time': time.time(),
                        'total_parse_duration': total_parse_duration,
                        'timestamp': time.time()
                    })
                    self.timing_pub.publish(timing_msg)
                    self.get_logger().info(f'[TIMING] add_task total_parse_duration: {total_parse_duration:.4f}s')

                elif intent == "obstacle_update":
                    msg_out = ObstacleUpdate()
                    obstacle_vertices = parameters.get("obstacle_location", [(0.0, 0.0)])
                    
                    # If only 2 points (a line segment), expand to rectangle with thickness 0.2
                    if len(obstacle_vertices) == 2:
                        self.get_logger().info(f'Detected line segment wall: {obstacle_vertices}')
                        obstacle_vertices = expand_line_to_rectangle(obstacle_vertices, thickness=0.2)
                        self.get_logger().info(f'Expanded to rectangle: {obstacle_vertices}')
                    
                    flattened_coords = []
                    for vertex in obstacle_vertices:
                        if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                            flattened_coords.extend([float(vertex[0]), float(vertex[1])])
                        else:
                            self.get_logger().warn(f'Invalid vertex format: {vertex}')
                    msg_out.obstacle_location = flattened_coords
                    msg_out.obstacle_type = parameters.get("obstacle_type", "unknown")
                    self.publish_obstacle_update(msg_out)
                    self.get_logger().info(f'Published ObstacleUpdate: {msg_out}')
                    
                    # Publish timing for obstacle_update command (using monotonic time for accurate duration)
                    publish_end_mono = time.monotonic()
                    total_parse_duration = publish_end_mono - parse_start_mono
                    timing_msg = String()
                    timing_msg.data = json.dumps({
                        'type': 'command_published',
                        'command': 'obstacle_update',
                        'obstacle_type': msg_out.obstacle_type,
                        'parse_start_time': parse_start_time,
                        'llm_call_duration': llm_call_duration,
                        'publish_time': time.time(),
                        'total_parse_duration': total_parse_duration,
                        'timestamp': time.time()
                    })
                    self.timing_pub.publish(timing_msg)
                    self.get_logger().info(f'[TIMING] obstacle_update total_parse_duration: {total_parse_duration:.4f}s')

                elif intent == "change_task_priority":
                    msg_out = ChangeTaskPriority()
                    msg_out.task_label = parameters.get("task_label", "")
                    msg_out.priority = parameters.get("priority", "normal")
                    self.publish_change_task_priority(msg_out)
                    self.get_logger().info(f'Published ChangeTaskPriority: {msg_out}')
                    
                    # Publish timing for change_task_priority command (using monotonic time for accurate duration)
                    publish_end_mono = time.monotonic()
                    total_parse_duration = publish_end_mono - parse_start_mono
                    timing_msg = String()
                    timing_msg.data = json.dumps({
                        'type': 'command_published',
                        'command': 'change_task_priority',
                        'task_label': msg_out.task_label,
                        'priority': msg_out.priority,
                        'parse_start_time': parse_start_time,
                        'llm_call_duration': llm_call_duration,
                        'publish_time': time.time(),
                        'total_parse_duration': total_parse_duration,
                        'timestamp': time.time()
                    })
                    self.timing_pub.publish(timing_msg)
                    self.get_logger().info(f'[TIMING] change_task_priority total_parse_duration: {total_parse_duration:.4f}s')
                else:
                    self.get_logger().warn(f'Unknown intent: {intent}')

        except Exception as e:
            self.get_logger().error(f'Parsing or publishing failed: {e}')

    def call_llm(self, instruction_text):
        prompt = f"""
You are a command parser. Convert the following instruction into a JSON array, where each element represents one intent.
Possible intents: "add_task" or "obstacle_update" or "change_task_priority".

Each element should follow one of the following formats:

For "add_task":
{{
  "intent": "add_task",
  "parameters": {{
    "location": [1.0, 2.0],
    "delivery_point": "b",
    "task_label": "fb",
    "task_type": "special"
  }}
}}

For "obstacle_update":
{{
  "intent": "obstacle_update",
  "parameters": {{
    "obstacle_location": [[4.1, 1.1], [4.1, 2.0], [2.5, 2.0], [2.5, 1.1]],
    "obstacle_type": "wall"
  }}
}}

For "change_task_priority":
{{
  "intent": "change_task_priority",
  "parameters": {{
    "task_label": "fb",
    "priority": "high"
  }}
}}

Instruction: {instruction_text}

Return ONLY the JSON array. Do NOT add any explanation.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a command parser that outputs structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return json.loads(response.choices[0].message.content)

    def publish_add_task(self, msg):
        for pub in self.task_pubs.values():
            pub.publish(msg)
        self.task_pub.publish(msg)

    def publish_obstacle_update(self, msg):
        for pub in self.obstacle_pubs.values():
            pub.publish(msg)
        self.obstacle_pub.publish(msg)

    def publish_change_task_priority(self, msg):
        for pub in self.change_task_priority_pubs.values():
            pub.publish(msg)
        self.change_task_priority_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = LLMCommandParserNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()