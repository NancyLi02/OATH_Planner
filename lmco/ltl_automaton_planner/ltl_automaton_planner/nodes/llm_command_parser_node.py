import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ltl_automaton_msgs.msg import AddTask, ObstacleUpdate
from openai import OpenAI

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
                    msg_out.task_type = parameters.get("task_type", "normal")
                    self.task_pub.publish(msg_out)
                    self.get_logger().info(f'Published AddTask: {msg_out}')

                elif intent == "obstacle_update":
                    msg_out = ObstacleUpdate()
                    msg_out.obstacle_location = parameters.get("obstacle_location", [0.0, 0.0])
                    msg_out.obstacle_type = parameters.get("obstacle_type", "unknown")
                    self.obstacle_pub.publish(msg_out)
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
    "task_type": "special"
  }}
}}

For "obstacle_update":
{{
  "intent": "obstacle_update",
  "parameters": {{
    "obstacle_location": [3.5, 4.2],
    "obstacle_type": "wall"
  }}
}}

Instruction: {instruction_text}

Return ONLY the JSON array. Do NOT add any explanation.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a command parser that outputs structured JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return json.loads(response.choices[0].message.content)

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
