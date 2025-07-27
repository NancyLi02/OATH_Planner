import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ltl_automaton_msgs.msg import AddTask, ObstacleUpdate
from openai import OpenAI
import yaml
from ament_index_python.packages import get_package_share_directory

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMCommandParserNode(Node):
    def __init__(self):
        super().__init__('llm_command_parser_node')

        self.subscription = self.create_subscription(
            String,
            '/human_instruction',
            self.instruction_callback,
            10
        )

        # === 新增：为每个robot创建带namespace的add_task publisher ===
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

        self.task_pub = self.create_publisher(AddTask, '/add_task', 10)

        self.obstacle_pub = self.create_publisher(ObstacleUpdate, '/obstacle_update', 10)

        self.get_logger().info('LLM Command Parser Node initialized.')

    def instruction_callback(self, msg):
        instruction = msg.data
        self.get_logger().info(f'Received instruction: "{instruction}"')

        try:
            parsed_intents = self.call_llm(instruction)

            if not isinstance(parsed_intents, list):
                raise ValueError("LLM response is not a list")

            for intent_obj in parsed_intents:
                intent = intent_obj.get("intent", "")
                parameters = intent_obj.get("parameters", {})

                if intent == "add_task":
                    msg_out = AddTask()
                    msg_out.location = parameters.get("location", [0.0, 0.0])
                    msg_out.task_label = parameters.get("task_label", "")
                    msg_out.delivery_point = parameters.get("delivery_point", "")
                    msg_out.task_type = parameters.get("task_type", "normal")
                    self.publish_add_task(msg_out)
                    self.get_logger().info(f'Published AddTask: {msg_out}')

                elif intent == "obstacle_update":
                    msg_out = ObstacleUpdate()
                    # 处理多边形顶点坐标，将 [(x1,y1), (x2,y2), ...] 格式转换为 [x1, y1, x2, y2, ...] 格式
                    obstacle_vertices = parameters.get("obstacle_location", [(0.0, 0.0)])
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

                else:
                    self.get_logger().warn(f'Unknown intent: {intent}')

        except Exception as e:
            self.get_logger().error(f'Parsing or publishing failed: {e}')

    def call_llm(self, instruction_text):
        prompt = f"""
You are a command parser. Convert the following instruction into a JSON array, where each element represents one intent.
Possible intents: "add_task" or "obstacle_update".

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
