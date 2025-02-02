#!/usr/bin/env python
import numpy
import rclpy
from rclpy.node import Node
import sys
import importlib
import matplotlib.pyplot as plt

from copy import deepcopy

import std_msgs

#import matplotlib.pyplot as plt
import networkx as nx
from ltl_automaton_planner.ltl_automaton_utilities import state_models_from_ts, import_ts_from_file, handle_ts_state_msg, extract_numbers

# Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TransitionSystemStateStamped, TransitionSystemState, LTLPlan, RelayRequest, RelayResponse
from ltl_automaton_msgs.srv import * #TaskPlanning, TaskPlanningResponse, TaskReplanningAdd, TaskReplanningDelete, TaskReplanningRelabel, TaskReplanningAddResponse, TaskReplanningDeleteResponse

# Import dynamic reconfigure components for dynamic parameters (see dynamic_reconfigure and dynamic_params package)

from ltl_automaton_planner.ltl_tools.ts import TSModel
from ltl_automaton_planner.ltl_tools.ltl_planner import LTLPlanner

import time
import yaml
from example_interfaces.srv import AddTwoInts

def show_automaton(automaton_graph):
    pos=nx.spring_layout(automaton_graph)
    nx.draw(automaton_graph, pos)
    nx.draw_networkx_labels(automaton_graph, pos)
    edge_labels = nx.get_edge_attributes(automaton_graph, 'guard')
    nx.draw_networkx_edge_labels(automaton_graph, pos, edge_labels = edge_labels)
    plt.show()
    return

class MainPlanner(Node):
    def __init__(self):
        super().__init__("planner_node")
        # init parameters, automaton, etc...
        self.init_params()
        print(self.get_node_names_and_namespaces())
        self.setup_pub_sub()
        self.build_automaton()
        time.sleep(1)
        self.publish_plan()

        self.get_logger().info("MainPlanner node started")
        # self.timer = self.create_timer(1.0, self.loop_callback)
        
    def init_params(self):
        self.declare_parameter('agent_name', 'agent')  
        self.declare_parameter('initial_beta', 1000)  
        self.declare_parameter('gamma', 10)
        self.declare_parameter('hard_task', "")
        self.declare_parameter('soft_task', "")
        self.declare_parameter('transition_system_textfile', "")  
        self.declare_parameter('algo_type', 'dstar')  
        self.declare_parameter('N', 10)
        self.declare_parameter('init_state', 'c0_r0')

        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        self.initial_beta = self.get_parameter('initial_beta').get_parameter_value().integer_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().integer_value
        self.algo_type = self.get_parameter('algo_type').get_parameter_value().string_value
        self.grid_size = self.get_parameter('N').get_parameter_value().integer_value
        param_list = [parameter.name for parameter in self._parameters.values()]
        print("param_list", param_list)

        # Get LTL hard task and raise error if it doesn't exist
        if self.get_parameter('hard_task').get_parameter_value().string_value:
            self.hard_task = self.get_parameter('hard_task').get_parameter_value().string_value
            print("hard_task", self.hard_task)
        else:
            raise ValueError("Cannot initialize LTL planner, no hard_task defined")

        # Get LTL soft task and transition system
        self.soft_task = self.get_parameter('soft_task').get_parameter_value().string_value
        transition_system_textfile = self.get_parameter('transition_system_textfile').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.initial_state_ts_dict = {'2d_pose_region': self.get_parameter('init_state').get_parameter_value().string_value,
                                      'Drone_state': 'unloaded'}
        print("**** inital state dict:", self.initial_state_ts_dict)

    
    def build_automaton(self):
        # Import state models from TS
        state_models = state_models_from_ts(self.transition_system, self.initial_state_ts_dict)
     
        # Here we take the product of each element of state_models to define the full TS
        self.robot_model = TSModel(state_models)
        self.ltl_planner = LTLPlanner(self.robot_model, self.hard_task, self.soft_task, self.initial_beta, self.gamma)
        self.ltl_planner.optimal(algo=self.algo_type, N=self.grid_size)
        # Get first value from set
        self.ltl_planner.curr_ts_state = list(self.ltl_planner.product.graph['ts'].graph['initial'])[0]

        # initialize storage of set of possible runs in product
        self.ltl_planner.posb_runs = set([(n,) for n in self.ltl_planner.product.graph['initial']])

        #show_automaton(self.robot_model)
        #show_automaton(self.ltl_planner.product.graph['buchi'])
        #show_automaton(self.ltl_planner.product)


    def setup_pub_sub(self):
        
        # Set up publishers (replace YourMsgType with the correct message type)
        self.prefix_plan_pub = self.create_publisher(LTLPlan, 'prefix_plan', 10)
        self.suffix_plan_pub = self.create_publisher(LTLPlan, 'suffix_plan', 10)
        
        # # Prefix plan publisher
        # self.prefix_plan_pub = rospy.Publisher('/prefix_plan', LTLPlan, queue_size = 1)
        # self.suffix_plan_pub = rospy.Publisher('/suffix_plan', LTLPlan, queue_size = 1)

        # Initialize services for task replanning modifications
        # self.replan_srv_add = self.create_service(TaskReplanningModify, 'replanning_modify', self.replanning_modify_callback)
        # self.replan_srv_delete = self.create_service(TaskReplanningDelete, 'replanning_delete', self.replanning_delete_callback)
        # self.replan_srv_relabel = self.create_service(TaskReplanningRelabel, 'replanning_relabel', self.replanning_relabel_callback)
        
        # Initialize services 
        self.subscriber_ = self.create_subscription(
            RelayRequest,
            'replanning_request',
            self.listener_callback,
            10)
        
        self.publisher_ = self.create_publisher(RelayResponse, 'replanning_response', 10)
    
        # # Initialize task replanning service for ADD
        # self.replan_srv_add = rospy.Service('replanning_mod', TaskReplanningModify, self.replanning_modify_callback)
        # self.replan_srv_delete = rospy.Service('replanning_delete', TaskReplanningDelete, self.replanning_delete_callback)
        # self.replan_srv_relabel = rospy.Service('replanning_relabel', TaskReplanningRelabel, self.replanning_relabel_callback)
        
    #----------------------------------------------
    # Publish prefix and suffix plans from planner
    #----------------------------------------------
    def publish_plan(self):
        # If plan exists
        self.get_logger().info("in push plan")
        
        if not (self.ltl_planner.run == None):
            self.get_logger().info("in push plan2")
            # Prefix plan
            #-------------
            self.prefix_plan_msg = LTLPlan()
            self.prefix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.prefix_plan_msg.action_sequence = self.ltl_planner.run.pre_plan
            self.prefix_plan_msg.ts_state_sequence = []
            # # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planner.run.line:
                ts_state_msg = TransitionSystemState()
                # ts_state_msg.state_dimension_names = self.ltl_planner.product.graph['ts'].graph['ts_state_format']
                # If TS state is more than 1 dimension (is a tuple)
                if type(ts_state) is tuple:
                    ts_state_msg.states = list(ts_state)
                # Else state is a single string
                else:
                    ts_state_msg.states = [ts_state]
                # Add to plan TS state sequence
                self.prefix_plan_msg.ts_state_sequence.append(ts_state_msg)

            # Publish
            self.get_logger().info("Publish Prefix Plan")
            # print(prefix_plan_msg.action_sequence)
            self.prefix_plan_pub.publish(self.prefix_plan_msg)

            # Suffix plan
            #-------------
            self.suffix_plan_msg = LTLPlan()
            self.suffix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.suffix_plan_msg.action_sequence = self.ltl_planner.run.suf_plan
            self.suffix_plan_msg.ts_state_sequence = []
            # # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planner.run.loop:
                ts_state_msg = TransitionSystemState()
                # ts_state_msg.state_dimension_names = self.ltl_planner.product.graph['ts'].graph['ts_state_format']
                # If TS state is more than 1 dimension (is a tuple)
                if type(ts_state) is tuple:
                    ts_state_msg.states = list(ts_state)
                # Else state is a single string
                else:
                    ts_state_msg.states = [ts_state]

                # Add to plan TS state sequence
                self.suffix_plan_msg.ts_state_sequence.append(ts_state_msg)

            # Publish
            self.get_logger().info("Publish Suffix Plan")
            self.suffix_plan_pub.publish(self.suffix_plan_msg)


    def replanning_modify_callback(self, task_replanning_req):
        if task_replanning_req:
            self.get_logger().info("Replanning [modify] Callback")
            update_info = dict()
            update_info["modified"] = set()
            update_info["deleted"] = set()
            update_info["relabel"] = set()
            # TODO: check both from_pose and to_pose have only two elements
            # change position in tuple to ts node of the format ('c0_r5', 'unloaded')
            for node in self.ltl_planner.product.graph['ts'].nodes():
                if tuple(task_replanning_req.from_pose) == extract_numbers(node[0]):
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.to_pose) == extract_numbers(succ_node[0]):
                            update_info["modified"].add((node, succ_node, task_replanning_req.cost))
                if tuple(task_replanning_req.to_pose) == extract_numbers(node[0]):
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.from_pose) == extract_numbers(succ_node[0]):
                            update_info["modified"].add((node, succ_node, task_replanning_req.cost))
            # print(update_info["modified"])
            modified_edges_dict = self.ltl_planner.revise_product(update_info)
            self.get_logger().info("Finished revise")
            
            success = False
            if self.algo_type == 'dstar' or self.algo_type =="dstar-relaxed":
                if self.ltl_planner.dstar_rewire(task_replanning_req.exec_index, modified_edges_dict, update_info):
                    success = True
            elif self.algo_type == 'local':
                if self.ltl_planner.local_rewire(task_replanning_req.exec_index):
                    success = True
            elif self.algo_type == 'brute-force' or self.algo_type == "relaxed":
                if self.ltl_planner.dijkstra_rewire(task_replanning_req.exec_index):
                    success = True
            
            res = RelayResponse()
            if success:
                # print("new_prefix", self.ltl_planner.prefix)
                # print("new_suffix", self.ltl_planner.suffix)
                # res = TaskReplanningModifyResponse()
                res.success = True
                res.new_plan_prefix = LTLPlan()
                res.new_plan_prefix.header.stamp = self.get_clock().now().to_msg()
                res.new_plan_prefix.action_sequence = self.ltl_planner.run.pre_plan
                # # Go through all TS state in plan and add it as TransitionSystemState message
                for ts_state in self.ltl_planner.run.line:
                    ts_state_msg = TransitionSystemState()
                    # If TS state is more than 1 dimension (is a tuple)
                    if type(ts_state) is tuple:
                        ts_state_msg.states = list(ts_state)
                    # Else state is a single string
                    else:
                        ts_state_msg.states = [ts_state]
                    # Add to plan TS state sequence
                    res.new_plan_prefix.ts_state_sequence.append(ts_state_msg)
                    
                res.new_plan_suffix = LTLPlan()
                res.new_plan_suffix.header.stamp = self.get_clock().now().to_msg()
                res.new_plan_suffix.action_sequence = self.ltl_planner.run.suf_plan
                # # Go through all TS state in plan and add it as TransitionSystemState message
                for ts_state in self.ltl_planner.run.loop:
                    ts_state_msg = TransitionSystemState()
                    # If TS state is more than 1 dimension (is a tuple)
                    if type(ts_state) is tuple:
                        ts_state_msg.states = list(ts_state)
                    # Else state is a single string
                    else:
                        ts_state_msg.states = [ts_state]
                    # Add to plan TS state sequence
                    res.new_plan_suffix.ts_state_sequence.append(ts_state_msg)
                self.get_logger().info("service has been transmitted")
                self.publisher_.publish(res)
                return
            
        self.get_logger().error("Error in replanning modify callback")
        res.success = False
        self.publisher_.publish(res)
        return 
        
    def replanning_delete_callback(self, task_replanning_req):
        if task_replanning_req:
            self.get_logger().info("Replanning [Delete] Callback")
            update_info = dict()
            update_info["modified"] = set()
            update_info["deleted"] = set()
            update_info["relabel"] = set()
            # TODO: check both from_pose and to_pose have only two elements
            # change position in tuple to ts node of the format ('c0_r5', 'unloaded')
            for node in self.ltl_planner.product.graph['ts'].nodes():
                if tuple(task_replanning_req.from_pose) == extract_numbers(node[0]):
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.to_pose) == extract_numbers(succ_node[0]):
                            update_info["deleted"].add((node, succ_node))
                if tuple(task_replanning_req.to_pose) == extract_numbers(node[0]):
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.from_pose) == extract_numbers(succ_node[0]):
                            update_info["deleted"].add((node, succ_node))
            # print(update_info["deleted"])
            modified_edges_dict = self.ltl_planner.revise_product(update_info)
            self.get_logger().info("finished revise")
            
            success = False
            if self.algo_type == 'dstar' or self.algo_type =="dstar-relaxed":
                if self.ltl_planner.dstar_rewire(task_replanning_req.exec_index, modified_edges_dict, update_info):
                    success = True
            elif self.algo_type == 'local':
                if self.ltl_planner.local_rewire(task_replanning_req.exec_index):
                    success = True
            elif self.algo_type == 'brute-force' or self.algo_type == "relaxed":
                if self.ltl_planner.dijkstra_rewire(task_replanning_req.exec_index):
                    success = True
            self.get_logger().info("finished revise successfully")
            
            res = RelayResponse()
            if success:
                self.get_logger().info("start preparing for the ")
                # print("new_prefix", self.ltl_planner.prefix)
                res.success = True
                res.new_plan_prefix = LTLPlan()
                res.new_plan_prefix.header.stamp = self.get_clock().now().to_msg()
                res.new_plan_prefix.action_sequence = self.ltl_planner.run.pre_plan
                # # Go through all TS state in plan and add it as TransitionSystemState message
                for ts_state in self.ltl_planner.run.line:
                    ts_state_msg = TransitionSystemState()
                    # If TS state is more than 1 dimension (is a tuple)
                    if type(ts_state) is tuple:
                        ts_state_msg.states = list(ts_state)
                    # Else state is a single string
                    else:
                        ts_state_msg.states = [ts_state]
                    # Add to plan TS state sequence
                    res.new_plan_prefix.ts_state_sequence.append(ts_state_msg)
                    
                res.new_plan_suffix = LTLPlan()
                res.new_plan_suffix.header.stamp = self.get_clock().now().to_msg()
                res.new_plan_suffix.action_sequence = self.ltl_planner.run.suf_plan
                # # Go through all TS state in plan and add it as TransitionSystemState message
                for ts_state in self.ltl_planner.run.loop:
                    ts_state_msg = TransitionSystemState()
                    # If TS state is more than 1 dimension (is a tuple)
                    if type(ts_state) is tuple:
                        ts_state_msg.states = list(ts_state)
                    # Else state is a single string
                    else:
                        ts_state_msg.states = [ts_state]
                    # Add to plan TS state sequence
                    res.new_plan_suffix.ts_state_sequence.append(ts_state_msg)
                self.publisher_.publish(res)
                self.get_logger().info("service has been transmitted ")
                return
            
        self.get_logger().error("Error in replanning modify callback")
        res.success = False
        self.publisher_.publish(res)
        return

    
    def replanning_relabel_callback(self):
        pass

    def listener_callback(self, msg):
        if msg.type == "modify":
            self.replanning_modify_callback(msg)
            return
        if msg.type == "delete":
            self.replanning_delete_callback(msg)
            return
        self.get_logger.error("wrong replanning request type")
        
    # #-------------------------
    # # Publish possible states
    # #-------------------------
    # def publish_possible_states(self):
    #     # Create message
    #     possible_states_msg = LTLStateArray()
    #     # For all possible state, add to the message list
    #     for ltl_state in self.ltl_planner.product.possible_states:
    #         ltl_state_msg = LTLState()
    #         # If TS state is more than 1 dimension (is a tuple)
    #         if type(ltl_state[0]) is tuple:
    #             ltl_state_msg.ts_state.states = list(ltl_state[0])
    #         # Else state is a single string
    #         else:
    #             ltl_state_msg.ts_state.states = [ltl_state[0]]

    #         ltl_state_msg.ts_state.state_dimension_names = self.ltl_planner.product.graph['ts'].graph['ts_state_format']
    #         ltl_state_msg.buchi_state = str(ltl_state[1])
    #         possible_states_msg.ltl_states.append(ltl_state_msg)

    #     # Publish
    #     self.possible_states_pub.publish(possible_states_msg)
#==============================
#             Main
#==============================

def main(args=None):
    rclpy.init(args=args)
    try:
        ltl_planner_node = MainPlanner()
        rclpy.spin(ltl_planner_node)
    except ValueError as e:
        ltl_planner_node.get_logger().error("LTL Planner: " + str(e))
        ltl_planner_node.get_logger().error("LTL Planner: shutting down...")
    finally:
        # ltl_planner_node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':

    main()