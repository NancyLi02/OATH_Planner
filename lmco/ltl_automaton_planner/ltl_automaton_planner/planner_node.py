#!/usr/bin/env python
import numpy as np
import rclpy
from rclpy.node import Node
import sys
import importlib
import matplotlib.pyplot as plt
import os
from copy import deepcopy

import std_msgs

#import matplotlib.pyplot as plt
import networkx as nx
from ltl_automaton_planner.ltl_automaton_utilities import state_models_from_ts, import_ts_from_file, handle_ts_state_msg, extract_numbers

# Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TransitionSystemStateStamped, TransitionSystemState, LTLPlan, RelayRequest, RelayResponse, TaskAssignment, TaskReAssignment, ScoreRequest, ScoreList
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
        self.task_data = self.load_tasks(self.ltl_formula_file)
        self.get_logger().info("MainPlanner node started")

        time.sleep(1)
        self.init_score_list()

        
    def init_params(self):
        self.declare_parameter('agent_name', '')  
        self.declare_parameter('initial_beta', 1000)  
        self.declare_parameter('gamma', 10)
        self.declare_parameter('transition_system_textfile', "")  
        self.declare_parameter('algo_type', 'dstar')  
        self.declare_parameter('N', 10)
        self.declare_parameter('init_state', 'c0_r0')
        self.declare_parameter('ltl_formula_file','')


        self.ltl_formula_file = self.get_parameter('ltl_formula_file').get_parameter_value().string_value
        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        self.get_logger().info(f'Robot name is {self.agent_name}')

        self.initial_beta = self.get_parameter('initial_beta').get_parameter_value().integer_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().integer_value
        self.algo_type = self.get_parameter('algo_type').get_parameter_value().string_value
        self.grid_size = self.get_parameter('N').get_parameter_value().integer_value
        param_list = [parameter.name for parameter in self._parameters.values()]
        print("param_list", param_list)

        transition_system_textfile = self.get_parameter('transition_system_textfile').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.initial_state_ts_dict = {'2d_pose_region': self.get_parameter('init_state').get_parameter_value().string_value,
                                      'Drone_state': 'unloaded'}
        print("**** inital state dict:", self.initial_state_ts_dict)
        self.init_state = self.get_parameter('init_state').value
        self.init_pose = np.array(list(map(int, self.init_state[1:].split('_r'))), dtype=np.int32)
        self.score_list = []

        # workspace_dir = os.path.join('/home/nanli/ros2_ws/', 'src/lmco')
        # package_src_dir = os.path.join(workspace_dir, 'ltl_automaton_planner')
        # config_dir = os.path.join(package_src_dir, 'config')
        # ltl_formula_file = os.path.join(config_dir, 'task_ltl.yaml')

        # # Get LTL hard task and raise error if it doesn't exist
        # if self.get_parameter('hard_task').get_parameter_value().string_value:
        #     self.hard_task = self.get_parameter('hard_task').get_parameter_value().string_value
        #     print("hard_task", self.hard_task)
        # else:
        #     raise ValueError("Cannot initialize LTL planner, no hard_task defined")
        # # Get LTL soft task and transition system
        # self.soft_task = self.get_parameter('soft_task').get_parameter_value().string_value


    def load_tasks(self, yaml_file):
        try:
            with open(yaml_file, 'r') as file:
                yaml_content = yaml.safe_load(file)
            tasks = yaml_content.get('tasks', {})
            task_data = {task_id: {'hard_task': task.get('hard_task', ""), 'soft_task': task.get('soft_task', "")}
                         for task_id, task in tasks.items()}
            return task_data
        except Exception as e:
            self.get_logger().error(f"Failed to read or parse the YAML file: {e}")
            raise
    
    def build_automaton(self, task_id):
        # Import state models from TS
        state_models = state_models_from_ts(self.transition_system, self.initial_state_ts_dict)

        # Get task ltl specification
        hard_task = self.task_data[task_id]['hard_task']
        soft_task = self.task_data[task_id]['soft_task']

        # Build product automaton
        self.robot_model = TSModel(state_models)
        self.ltl_planner = LTLPlanner(self.robot_model, hard_task, soft_task, self.initial_beta, self.gamma)
        self.ltl_planner.optimal(algo=self.algo_type, N=self.grid_size)

        # initialize storage of set of possible runs in product
        self.ltl_planner.curr_ts_state = list(self.ltl_planner.product.graph['ts'].graph['initial'])[0]
        self.ltl_planner.posb_runs = set([(n,) for n in self.ltl_planner.product.graph['initial']])

        #show_automaton(self.robot_model)
        #show_automaton(self.ltl_planner.product.graph['buchi'])
        #show_automaton(self.ltl_planner.product)


    def setup_pub_sub(self):
        
        # Set up publishers (replace YourMsgType with the correct message type)
        self.prefix_plan_pub = self.create_publisher(LTLPlan, 'prefix_plan', 10)
        self.suffix_plan_pub = self.create_publisher(LTLPlan, 'suffix_plan', 10)
        self.publisher_ = self.create_publisher(RelayResponse, 'replanning_response', 10)   
        self.score_list_pub = self.create_publisher(ScoreList, 'score_list', 10)     
        
        # Initialize services 
        self.subscriber_ = self.create_subscription(
            RelayRequest,
            'replanning_request',
            self.listener_callback,
            10)
        
        self.taskassignment_sub = self.create_subscription(
            TaskAssignment,
            'task_assignment',
            self.taskassignment_callback,
            10
        )

        self.new_task_sub = self.create_subscription(
            TaskReAssignment,
            'task_reassignment',
            self.new_task_callback,
            10
        )

        self.score_request_sub = self.create_subscription(
            ScoreRequest,
            'score_request',
            self.get_score_list,
            10
        )

    def get_score_list(self, msg):
        current_pos = msg.position
        formatted_pose = f'c{current_pos[0]}_r{current_pos[1]}'
        initial_state = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }
        task_index = [1, 2, 3, 4]
        self.score_list = []

        for i in task_index:
            task_id = f'task{i}'
            self.build_score_automaton(task_id, initial_state)
            self.score_list.append(self.build_score_list())  # Append scores to the list

        self.pub_score()

    def init_score_list(self):
        formatted_pose = self.init_state
        initial_state = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }
        task_index = [1, 2, 3, 4]
        self.score_list = []

        for i in task_index:
            task_id = f'task{i}'
            self.build_score_automaton(task_id, initial_state)
            self.score_list.append(self.build_score_list())  # Append scores to the list

        self.pub_score()


    def pub_score(self):
        self.get_logger().info(f'Finish calculating score for {self.agent_name}: {self.score_list}')
        score_msg = ScoreList()
        score_msg.robot_id = self.agent_name
        score_msg.score_list = self.score_list
        self.score_list_pub.publish(score_msg)
        self.get_logger().info(f'Publish score list for {self.agent_name}...')

    def build_score_list(self):
        if self.score_est.run is not None:
            
            # Prefix Score
            self.cal_score_pre = LTLPlan()
            self.cal_score_pre.action_sequence = self.score_est.run.pre_plan
            self.pre_score = len(self.cal_score_pre.action_sequence)  # Ensure correct length calculation
            # Suffix Score
            self.cal_score_suf = LTLPlan()
            self.cal_score_suf.action_sequence = self.score_est.run.suf_plan
            self.suf_score = len(self.cal_score_suf.action_sequence)  # Ensure correct length calculation
            
            score = max(0, 50 - (self.pre_score + self.suf_score))  # Prevent negative scores
            return score
        return 0  # Return 0 if no score is calculated



    def build_score_automaton(self, task_id, initial_state):
        # Import state models from TS
        state_models = state_models_from_ts(self.transition_system, initial_state)

        # Get task ltl specification
        hard_task = self.task_data[task_id]['hard_task']
        soft_task = self.task_data[task_id]['soft_task']

        # Build product automaton
        robot_model = TSModel(state_models)
        self.score_est = LTLPlanner(robot_model, hard_task, soft_task, self.initial_beta, self.gamma)
        self.score_est.optimal(algo=self.algo_type, N=self.grid_size)

        # initialize storage of set of possible runs in product
        self.score_est.curr_ts_state = list(self.score_est.product.graph['ts'].graph['initial'])[0]
        self.score_est.posb_runs = set([(n,) for n in self.score_est.product.graph['initial']])
    
    def new_task_callback(self, msg):
        self.get_logger().info('---------------Task Reassignment Received---------------')

        # Determine the agent name and extract the corresponding task
        if self.agent_name == 'robot_1':
            task_index = msg.robot_1_task
            new_initial_pose = msg.robot_1_pos
            self.get_logger().info(f'Robot 1 has been assigned to task {task_index}')
        elif self.agent_name == 'robot_2':
            task_index = msg.robot_2_task
            new_initial_pose = msg.robot_2_pos
            self.get_logger().info(f'Robot 2 has been assigned to task {task_index}')
        else:
            self.get_logger().error(f"Invalid agent name: {self.agent_name}")
            return

        # Ensure the task index is valid
        if task_index is None:
            self.get_logger().info(f"No task assigned to {self.agent_name}")
            return
        
        # Format the position into 'cx_ry' format
        formatted_pose = f'c{new_initial_pose[0]}_r{new_initial_pose[1]}'

        # Update initial state dictionary
        self.initial_state_ts_dict = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }

        # Check if the task index is within a valid range
        task_id = f'task{int(task_index)}'
        if task_id not in self.task_data:
            self.get_logger().error(f"Invalid task index received: {task_index}")
            return

        # Build the automaton for the assigned task
        self.get_logger().info(f"Building automaton for {task_id} assigned to {self.agent_name}")
        self.build_automaton(task_id)

        # Publish the plan
        self.publish_plan()
        
    
    def taskassignment_callback(self, msg):
        self.get_logger().info('---------------start taskassignment callback function---------------')
        # Determine the agent_name and extract the corresponding task
        if self.agent_name == 'robot_1':
            task_index = msg.robot_1_task
            self.get_logger().info(f'Robot 1 has been assigned to task{task_index}')
        elif self.agent_name == 'robot_2':
            task_index = msg.robot_2_task
            self.get_logger().info(f'Robot 2 has been assigned to task{task_index}')
        else:
            self.get_logger().error(f"Invalid agent name: {self.agent_name}")
            return

        # Ensure the task index is valid
        if task_index is None:
            self.get_logger().info(f"No task assigned to {self.agent_name}")
            return

        # Check if the task index is within a valid range
        task_id = f'task{int(task_index)}'
        if task_id not in self.task_data:
            self.get_logger().error(f"Invalid task index received: {task_index}")
            return

        # Call the corresponding build_automaton method
        self.get_logger().info(f"Building automaton for {task_id} assigned to {self.agent_name}")
        self.build_automaton(task_id)

        # Call the self.publish_plan() method to publish the plan
        self.publish_plan()
    
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