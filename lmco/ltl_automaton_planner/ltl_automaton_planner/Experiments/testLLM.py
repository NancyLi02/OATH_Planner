import os
import json
from openai import OpenAI
from ltl_automaton_msgs.msg import AddTask, ObstacleUpdate

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(instruction_text):
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

def test_instruction(instruction):
    print(f"\n🧪 Instruction: {instruction}")
    intents = call_llm(instruction)
    print("✅ Parsed JSON:")
    print(json.dumps(intents, indent=2))

    for intent in intents:
        if intent["intent"] == "add_task":
            msg = AddTask()
            msg.location = intent["parameters"].get("location", [0.0, 0.0])
            msg.task_type = intent["parameters"].get("task_type", "normal")
            print(f"➡️ AddTask: location={msg.location}, task_type={msg.task_type}")

        elif intent["intent"] == "obstacle_update":
            msg = ObstacleUpdate()
            msg.obstacle_location = intent["parameters"].get("obstacle_location", [0.0, 0.0])
            msg.obstacle_type = intent["parameters"].get("obstacle_type", "unknown")
            print(f"➡️ ObstacleUpdate: location={msg.obstacle_location}, type={msg.obstacle_type}")

        else:
            print(f"⚠️ Unknown intent: {intent['intent']}")

if __name__ == '__main__':
    test_instruction("New special task appears at [1.0, 2.0] and there's a bush at [3.5, 4.2].")
