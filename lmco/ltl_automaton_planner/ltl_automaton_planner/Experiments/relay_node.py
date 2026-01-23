import rclpy
from rclpy.node import Node

from ltl_automaton_msgs.srv import TaskReplanningDelete, TaskReplanningModify
from ltl_automaton_msgs.msg import RelayRequest, RelayResponse

class Relay(Node):

    def __init__(self):
        super().__init__('relay_node')
        
        self.delete_client = self.create_client(TaskReplanningDelete, 'replanning_delete')
        if not self.delete_client.wait_for_service(timeout_sec=1000.0):  # Set your desired timeout in seconds
            self.get_logger().error('Service /replanning_delete not available after waiting')
        else:
            self.get_logger().info('Service /replanning_delete is available')
            
        self.modify_client = self.create_client(TaskReplanningModify, 'replanning_modify')
        if not self.modify_client.wait_for_service(timeout_sec=1000.0):  # Set your desired timeout in seconds
            self.get_logger().error('Service /replanning_modify not available after waiting')
        else:
            self.get_logger().info('Service /replanning_modify is available')
        
        self.subscriber_ = self.create_subscription(
            RelayRequest,
            'replanning_request',
            self.listener_callback,
            10)
        
        self.publisher_ = self.create_publisher(RelayResponse, 'replanning_response', 10)

    def listener_callback(self, msg):
        self.get_logger().info("listen to call 0.1")
        if msg.type == "modify":
            task_replanning_srv = TaskReplanningModify.Request()
            task_replanning_srv.current_state = msg.current_state
            task_replanning_srv.mod_from = msg.from_pose
            task_replanning_srv.mod_to = msg.to_pose
            task_replanning_srv.exec_index = msg.exec_index
            task_replanning_srv.cost = msg.cost
            
            future = self.modify_client.call_async(task_replanning_srv)
            rclpy.spin_until_future_complete(self, future)
            response = self.future.result()
            if response.success:
                self.get_logger().info("successful received service!")
                publish_msg = RelayResponse()
                publish_msg.success = True
                publish_msg.new_plan_prefix = response.new_plan_prefix
                publish_msg.new_plan_suffix = response.new_plan_suffix
                self.publisher_.publish(publish_msg)
            else:
                self.get_logger().error('Failed to call service')
                
        elif msg.type == "delete":
            self.get_logger().info("listen to call 0.2")
            task_replanning_srv = TaskReplanningDelete.Request()
            task_replanning_srv.current_state = msg.current_state
            task_replanning_srv.delete_from = msg.from_pose
            task_replanning_srv.delete_to = msg.to_pose
            task_replanning_srv.exec_index = msg.exec_index
            self.get_logger().info("listen to call 0.3")
            self.future = self.delete_client.call_async(task_replanning_srv)
            self.get_logger().info("listen to call 0.4")
            rclpy.spin_until_future_complete(self, self.future)
            self.get_logger().info("listen to call 0.5")
            response = self.future.result()
            self.get_logger().info("listen to call 0.5")
            if response.success:
                self.get_logger().info("successful received service!")
                publish_msg = RelayResponse()
                publish_msg.success = True
                publish_msg.new_plan_prefix = response.new_plan_prefix
                publish_msg.new_plan_suffix = response.new_plan_suffix
                self.publisher_.publish(publish_msg)
            else:
                self.get_logger().error('Failed to call service')
        else:
            self.get_logger().error('Failed to call service')

    
def main(args=None):
    rclpy.init(args=args)

    relay_node = Relay()

    rclpy.spin(relay_node)

    relay_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()