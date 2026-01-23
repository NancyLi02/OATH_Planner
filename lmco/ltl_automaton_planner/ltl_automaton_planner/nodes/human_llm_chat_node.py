import os
import json
import traceback
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI
import threading
import tkinter as tk
from tkinter import scrolledtext

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define required parameters for each intent
INTENT_PARAMS = {
    "add_task": {
        "required": ["location", "task_label", "delivery_point"],
        "optional": ["task_type"],
        "descriptions": {
            "location": "coordinates [x, y] (e.g., [5.0, 19.5])",
            "task_label": "two-letter code chosen by user (e.g., 'fb', 'aa', 'xy')",
            "delivery_point": "single letter: 'b', 'c', 'd', or 'e'",
            "task_type": "type of task: 'normal' or 'special' (default: 'normal')"
        }
    },
    "obstacle_update": {
        "required": ["obstacle_location"],
        "optional": ["obstacle_type"],
        "descriptions": {
            "obstacle_location": "list of vertices - can be 2 points for a line segment (e.g., [[7,5], [8,5]]) or 4+ points for a polygon",
            "obstacle_type": "type of obstacle (e.g., 'wall', 'block', 'unknown')"
        }
    },
    "change_task_priority": {
        "required": ["task_label", "priority"],
        "optional": [],
        "descriptions": {
            "task_label": "two-letter task code to change (e.g., 'fb', 'aa')",
            "priority": "new priority level: 'high', 'normal', or 'low'"
        }
    }
}


class HumanLLMChatNode(Node):
    def __init__(self):
        super().__init__('human_llm_chat_node')
        
        self.instruction_pub = self.create_publisher(String, '/human_instruction', 10)
        
        self.conversation_history = []
        self.current_intent = None
        self.collected_params = {}
        
        self.get_logger().info('Human-LLM Chat Node initialized.')
        
        # Start GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.start_gui, daemon=True)
        self.gui_thread.start()

    def start_gui(self):
        self.root = tk.Tk()
        self.root.title("Human-LLM Command Interface")
        self.root.geometry("800x700")
        self.root.configure(bg='#1e1e2e')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e1e2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title label
        title_label = tk.Label(
            main_frame, 
            text="Robot Task Command Interface",
            font=('Consolas', 20, 'bold'),
            fg='#89b4fa',
            bg='#1e1e2e'
        )
        title_label.pack(pady=(0, 10))
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=('Consolas', 14),
            bg='#313244',
            fg='#cdd6f4',
            insertbackground='#f5e0dc',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure tags for different message types
        self.chat_display.tag_config('system', foreground='#a6e3a1')
        self.chat_display.tag_config('user', foreground='#89b4fa')
        self.chat_display.tag_config('assistant', foreground='#f9e2af')
        self.chat_display.tag_config('error', foreground='#f38ba8')
        self.chat_display.tag_config('success', foreground='#94e2d5')
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg='#1e1e2e')
        input_frame.pack(fill=tk.X)
        
        # Input entry
        self.input_entry = tk.Entry(
            input_frame,
            font=('Consolas', 14),
            bg='#45475a',
            fg='#cdd6f4',
            insertbackground='#f5e0dc',
            relief=tk.FLAT
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.input_entry.bind('<Return>', self.on_send)
        
        # Send button
        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            font=('Consolas', 13, 'bold'),
            bg='#89b4fa',
            fg='#1e1e2e',
            activebackground='#b4befe',
            activeforeground='#1e1e2e',
            relief=tk.FLAT,
            cursor='hand2',
            command=lambda: self.on_send(None)
        )
        self.send_btn.pack(side=tk.RIGHT, ipadx=18, ipady=8)
        
        # Reset button
        self.reset_btn = tk.Button(
            input_frame,
            text="Reset",
            font=('Consolas', 13),
            bg='#f38ba8',
            fg='#1e1e2e',
            activebackground='#eba0ac',
            activeforeground='#1e1e2e',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.reset_conversation
        )
        self.reset_btn.pack(side=tk.RIGHT, ipadx=12, ipady=8, padx=(0, 10))
        
        # Display welcome message
        welcome_msg = """Welcome! I'm your robot command assistant.

Just tell me what you need, for example:
• "Send a robot to pick something up at position 3, 4"
• "Make the delivery task more urgent"
"""
        self.append_message(welcome_msg, 'system')
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def append_message(self, message, tag='system'):
        self.chat_display.config(state=tk.NORMAL)
        
        prefix = ""
        if tag == 'user':
            prefix = "You: "
        elif tag == 'assistant':
            prefix = "Assistant: "
        elif tag == 'success':
            prefix = "✓ "
        elif tag == 'error':
            prefix = "✗ "
            
        self.chat_display.insert(tk.END, f"{prefix}{message}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def on_send(self, event):
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.input_entry.delete(0, tk.END)
        self.append_message(user_input, 'user')
        
        # Process in a separate thread to avoid GUI freeze
        threading.Thread(target=self.process_input, args=(user_input,), daemon=True).start()
    
    def process_input(self, user_input):
        self.get_logger().info(f'Processing user input: "{user_input}"')
        try:
            result = self.analyze_with_llm(user_input)
            self.get_logger().info(f'Analysis result status: {result.get("status")}')
            
            if result.get("status") == "complete":
                # All parameters collected, send the instruction
                final_instruction = result.get("instruction", user_input)
                self.get_logger().info(f'Command complete, sending instruction: {final_instruction}')
                self.send_instruction(final_instruction)
                self.root.after(0, lambda: self.append_message(
                    f"Command sent successfully!\nInstruction: {final_instruction}", 
                    'success'
                ))
                self.reset_state()
            elif result.get("status") == "need_more_info":
                # Missing parameters, ask user
                missing_info = result.get("message", "Please provide more information.")
                self.get_logger().info(f'Need more info. Missing params: {result.get("missing_params")}')
                self.root.after(0, lambda: self.append_message(missing_info, 'assistant'))
            elif result.get("status") == "clarification":
                # Need clarification
                clarification = result.get("message", "Could you please clarify your request?")
                self.get_logger().info(f'Requesting clarification from user')
                self.root.after(0, lambda: self.append_message(clarification, 'assistant'))
            else:
                self.get_logger().warn(f'Unknown status returned: {result.get("status")}')
                self.root.after(0, lambda: self.append_message(
                    "I couldn't understand that. Please try again.", 
                    'error'
                ))
                
        except Exception as e:
            error_msg = str(e)  # Capture error message before lambda
            self.get_logger().error(f'Error processing input: {type(e).__name__}: {e}')
            self.get_logger().error(f'Full traceback: {traceback.format_exc()}')
            self.root.after(0, lambda msg=error_msg: self.append_message(
                f"Error: {msg}", 
                'error'
            ))
    
    def analyze_with_llm(self, user_input):
        """Use LLM to analyze user input and check for missing parameters."""
        
        # Build conversation context
        self.conversation_history.append({"role": "user", "content": user_input})
        
        system_prompt = f"""You are a friendly and intelligent assistant helping users control robots. You can understand casual, vague, or informal language and interpret the user's intent.

YOUR CAPABILITIES:
You help users create 3 types of robot commands:
1. add_task - Send a robot to do something at a location
   - REQUIRED: 
     * location: [x, y] coordinates (e.g., [5.0, 19.5])
     * task_label: a two-letter code chosen by the user (e.g., 'fb', 'aa', 'xy')
     * delivery_point: a single letter indicating the delivery room - ONLY 'b', 'c', 'd', or 'e' are valid
       - 'b' = location [6, 3]
       - 'c' = location [5, 16]
       - 'd' = location [16, 13]
       - 'e' = location [16, 5.5]
   - OPTIONAL: task_type ("normal" or "special")
   
2. obstacle_update - Report an obstacle in the environment
   - REQUIRED: obstacle_location - can be specified as:
     * A line segment with 2 points: [[x1,y1], [x2,y2]] - will be auto-expanded to a wall with 0.2 thickness
     * A polygon with 4+ vertices: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
   - OPTIONAL: obstacle_type (e.g., "wall", "block", "unknown")
   
3. change_task_priority - Change how urgent a task is
   - REQUIRED: task_label (the two-letter task code), priority ("high", "normal", or "low")

UNDERSTANDING CASUAL LANGUAGE:
- "go grab something from point A" → add_task intent
- "there's something blocking the way near coordinates X,Y" → obstacle_update intent  
- "make task X more urgent" → change_task_priority with high priority
- "robot needs to pick up item at 3,4" → add_task at location [3.0, 4.0]
- "found a wall blocking area" → obstacle_update intent
- "hurry up with the delivery task" → change_task_priority intent

CONVERSATION STYLE:
- Be conversational and friendly, not robotic
- If the user is vague, make reasonable guesses and confirm with them
- Understand context from previous messages in the conversation
- Accept approximate descriptions and help users refine them
- DO NOT use markdown formatting (no **bold**, no *italic*, no headers). Use plain text only.

WHEN INFORMATION IS MISSING:
When you identify the intent but some required parameters are missing, your message MUST:
1. Acknowledge what you understood (the intent and any params already provided)
2. Clearly list ALL missing required parameters
3. Give examples of what format each missing parameter should be

Example response when intent is add_task but missing location, task_label, and delivery_point:
"Got it, you want to add a new task! To complete this command, I still need:
• location: Where should the robot go? (e.g., [5.0, 19.5])
• task_label: A two-letter code for this task (e.g., 'fb', 'aa', 'xy')
• delivery_point: Which delivery room? Choose from 'b', 'c', 'd', or 'e'
Please provide these details."

Example response when intent is obstacle_update but missing obstacle_location:
"I understand you want to report an obstacle. I need:
• obstacle_location: The coordinates - either 2 points for a line/wall (e.g., [[7,5], [8,5]]) or 4 points for a rectangle
Where exactly is this obstacle?"

RESPONSE FORMAT (JSON):
{{
    "status": "complete" | "need_more_info" | "clarification",
    "intent": "add_task" | "obstacle_update" | "change_task_priority" | null,
    "collected_params": {{...all params gathered so far...}},
    "missing_params": ["list of still missing required params"],
    "message": "Your friendly response to the user",
    "instruction": "Final formatted instruction (ONLY when status is complete)"
}}

When status is "complete", format the instruction clearly:
- add_task: "Add a task at location [x, y] with label 'name', delivery point 'dest', task type 'type'"
- obstacle_update: "Obstacle update at [[x1,y1], [x2,y2], ...] with type 'type'"
- change_task_priority: "Change priority of task 'name' to 'level'"

Remember: Be helpful and understanding. Users may not know exact formats - help them!

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no explanation, just pure JSON.
"""
        
        try:
            self.get_logger().info(f'Sending request to LLM with {len(self.conversation_history)} messages in history')
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Fast and capable - good balance
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history
                ],
                temperature=0.3,  # Lower temperature for more consistent JSON output
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            self.get_logger().info('Received response from LLM')
            
            # Check if response has content
            if not response.choices or not response.choices[0].message.content:
                self.get_logger().error('LLM returned empty response!')
                self.get_logger().error(f'Full response object: {response}')
                return {
                    "status": "clarification",
                    "message": "I received an empty response. Please try again."
                }
            
            response_text = response.choices[0].message.content
            self.get_logger().info(f'Raw LLM response (first 500 chars): {response_text[:500]}')
            
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            original_response = response_text  # Keep original for logging
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
                self.get_logger().debug('Extracted JSON from ```json block')
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
                self.get_logger().debug('Extracted JSON from ``` block')
            
            self.get_logger().info(f'Attempting to parse JSON: {response_text.strip()[:300]}...')
            
            result = json.loads(response_text.strip())
            self.get_logger().info(f'Successfully parsed JSON. Status: {result.get("status")}, Intent: {result.get("intent")}')
            
            # Add assistant response to history
            if result.get("message"):
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": result.get("message", "")
                })
            
            return result
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parsing failed at position {e.pos}: {e.msg}')
            self.get_logger().error(f'Original LLM response was: {original_response if "original_response" in locals() else "N/A"}')
            self.get_logger().error(f'Text being parsed: "{response_text if "response_text" in locals() else "N/A"}"')
            return {
                "status": "clarification",
                "message": "I'm having trouble understanding. Could you please rephrase your request?"
            }
        except Exception as e:
            self.get_logger().error(f'LLM API error: {type(e).__name__}: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            raise
    
    def send_instruction(self, instruction):
        """Publish the complete instruction to /human_instruction topic."""
        msg = String()
        msg.data = instruction
        self.instruction_pub.publish(msg)
        self.get_logger().info(f'Published instruction: {instruction}')
    
    def reset_state(self):
        """Reset the conversation state for a new command."""
        self.conversation_history = []
        self.current_intent = None
        self.collected_params = {}
    
    def reset_conversation(self):
        """Reset button handler."""
        self.reset_state()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        welcome_msg = """Conversation reset. Ready for new instructions!

Just tell me what you need:
• "Send a robot somewhere to do a task"
• "Report an obstacle in the environment"  
• "Change how urgent a task is"

I'll help figure out the details!
"""
        self.append_message(welcome_msg, 'system')
    
    def on_closing(self):
        """Handle window close event."""
        self.get_logger().info("Chat window closing...")
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = HumanLLMChatNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
