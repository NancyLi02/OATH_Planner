import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ltl_automaton_msgs.msg import AddTask, ObstacleUpdate
import openai
import json

openai.api_key = "YOUR_OPENAI_API_KEY"

class LLMCommandParserNode(Node):
    def __init__(self):
        super().__init__('llm_command_parser_node')

        self.subscription = self.create_subscription(
            String,
            '/human_instruction',
            self.instruction_callback,
            10
        )

        self.task_pub = self.create_publisher(AddTask, '/add_task', 10)
        self.obstacle_pub = self.create_publisher(ObstacleUpdate, '/obstacle_update', 10)

        self.get_logger().info('LLM Command Parser Node initialized.')

    def instruction_callback(self, msg):
        instruction = msg.data
        self.get_logger().info(f'Received: "{instruction}"')

        try:
            parsed = self.call_llm(instruction)
            intent = parsed.get("intent", "")

            if intent == "add_task":
                msg_out = AddTask()
                msg_out.location = parsed["parameters"].get("location", "unknown")
                msg_out.task_type = parsed["parameters"].get("task_type", "normal")
                self.task_pub.publish(msg_out)
                self.get_logger().info(f'Published AddTask: {msg_out}')

            elif intent == "obstacle_update":
                msg_out = ObstacleUpdate()
                msg_out.obstacle_location = parsed["parameters"].get("obstacle_location", "unknown")
                self.obstacle_pub.publish(msg_out)
                self.get_logger().info(f'Published ObstacleUpdate: {msg_out}')

            else:
                self.get_logger().warn(f'Unknown intent: {intent}')

        except Exception as e:
            self.get_logger().error(f'Parsing or publishing failed: {e}')

    def call_llm(self, instruction_text):
        prompt = f"""
You are a command parser. Convert the following instruction into a JSON object.
Possible intents: "add_task" or "obstacle_update".

For "add_task":
{{
  "intent": "add_task",
  "parameters": {{
    "location": "Zone A",
    "task_type": "special"
  }}
}}

For "obstacle_update":
{{
  "intent": "obstacle_update",
  "parameters": {{
    "obstacle_location": "Zone B"
  }}
}}

Instruction: {instruction_text}
Return ONLY the JSON.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a command parser that outputs structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response['choices'][0]['message']['content']
        return json.loads(content)

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
