#!/usr/bin/env python
import numpy as np
import rclpy
from rclpy.node import Node
import sys
import importlib
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from ltl_automaton_planner.ltl_tools.product import ProdAut
from ltl_automaton_planner.ltl_tools.buchi import mission_to_buchi
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import std_msgs

#import matplotlib.pyplot as plt
import networkx as nx
from ltl_automaton_planner.ltl_automaton_utilities import state_models_from_ts, import_ts_from_file, handle_ts_state_msg, extract_numbers, build_graph_halton

# Import LTL automaton message definitions
from ltl_automaton_msgs.msg import AddTask, AgentFail, ClusterRequest, NoTask, TaskRequestCluster, ClusterTaskassign, TransitionSystemStateStamped, TransitionSystemState, LTLPlan, RelayRequest, RelayResponse, TaskAssignment, TaskReAssignment, ScoreRequest, ScoreList, RobotID
from ltl_automaton_msgs.srv import * #TaskPlanning, TaskPlanningResponse, TaskReplanningAdd, TaskReplanningDelete, TaskReplanningRelabel, TaskReplanningAddResponse, TaskReplanningDeleteResponse

# Import dynamic reconfigure components for dynamic parameters (see dynamic_reconfigure and dynamic_params package)

from ltl_automaton_planner.ltl_tools.ts import TSModel
from ltl_automaton_planner.ltl_tools.ltl_planner import LTLPlanner

import time
import yaml
from example_interfaces.srv import AddTwoInts
import re
from ltl_automaton_planner.Gen_LTL import generate_ltl_formula

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

        self.nodes, self.actions = build_graph_halton(20, 20, 1000)

        # start_time = time.time()
        # self.initialize_automaton()
        # end_time = time.time()
        # build_automaton_time = end_time - start_time
        # self.get_logger().info(f'Building Automaton for {self.agent_name} cost {build_automaton_time} seconds.')
        
        # robotid_msg = RobotID()
        # robotid_msg.robot_id = self.agent_name
        # self.finish_build_auto_pub.publish(robotid_msg)
        # time.sleep(1)
        # self.init_score_list()
        
        
    def init_params(self):
        self.declare_parameter('agent_name', '')  
        self.declare_parameter('initial_beta', 1000)  
        self.declare_parameter('gamma', 10)
        self.declare_parameter('transition_system_textfile', "")  
        self.declare_parameter('algo_type', 'dstar')  
        self.declare_parameter('N', 40)
        self.declare_parameter('init_state', 0)
        self.declare_parameter('ltl_formula_file','')


        self.ltl_formula_file = self.get_parameter('ltl_formula_file').get_parameter_value().string_value
        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        # self.get_logger().info(f'Robot name is {self.agent_name}')

        self.initial_beta = self.get_parameter('initial_beta').get_parameter_value().integer_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().integer_value
        self.algo_type = self.get_parameter('algo_type').get_parameter_value().string_value
        self.grid_size = self.get_parameter('N').get_parameter_value().integer_value
        param_list = [parameter.name for parameter in self._parameters.values()]
        # print("param_list", param_list)

        transition_system_textfile = self.get_parameter('transition_system_textfile').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.init_state = self.get_parameter('init_state').value
        self.initial_state_ts_dict = {'2d_pose_region': f'{self.init_state}',
                                      'Drone_state': 'unloaded'}
        print("**** inital state dict:", self.initial_state_ts_dict)
        
        self.score_list = []
        self.task_index = list(range(1, 21))  # [1, 2, ..., 18]
        self.cur_task = ''
        self.task_number = 0
        self.current_task_list = []
        self.pose_index = self.init_state
        self.position = []
        self.current_ltl_formula = ''
        self.add_point_to_map = False
        self.new_task_points = []


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

    def generate_ltl_formula(self, route_labels):
        """
        根据route_labels生成LTL公式
        从route_labels中推断pickup和delivery标签，然后调用Gen_LTL.py中的函数
        """
        if not route_labels:
            self.get_logger().warn("Empty route_labels received")
            return ""
        
        # 从route_labels中推断pickup和delivery标签
        # pickup标签通常是两个字母（如'bc', 'cc', 'dc', 'ec'）
        # delivery标签通常是单个字母（如'b', 'c', 'd', 'e'）
        pickup_labels = []
        delivery_labels = []
        
        for label in route_labels:
            if len(label) == 2:  # 两个字母的标签通常是pickup
                pickup_labels.append(label)
            elif len(label) == 1:  # 单个字母的标签通常是delivery
                delivery_labels.append(label)
        
        # 去重
        pickup_labels = list(set(pickup_labels))
        delivery_labels = list(set(delivery_labels))
        
        self.get_logger().info(f"Inferred pickup labels: {pickup_labels}")
        self.get_logger().info(f"Inferred delivery labels: {delivery_labels}")
    
        
        # 调用Gen_LTL.py中的函数生成LTL公式
        ltl_formula = generate_ltl_formula(route_labels, pickup_labels, delivery_labels)
        
        self.get_logger().info(f"Generated LTL formula: {ltl_formula}")
        
        return ltl_formula

    # def initialize_automaton(self):
    #     # Import state models from TS
    #     state_models = state_models_from_ts(self.transition_system, self.initial_state_ts_dict)

    #     # Maintain multiple `Product Automaton` and `LTLPlanner` instances, each corresponding to a predefined task
    #     if not hasattr(self, 'product_automata'):
    #         self.product_automata = {}  # Initialize dictionary to store `ProdAut`
    #     if not hasattr(self, 'ltl_planners'):
    #         self.ltl_planners = {}  # Store LTLPlanner instances per task
    #         self.score_planners = {}

    #     # Define four tasks, each with its own LTL specification
    #     for task_id in self.task_index:
    #         task_id = f'task{task_id}'
    #         hard_task = self.task_data[task_id]['hard_task']
    #         soft_task = ''

    #         # Create Task Product Automaton
    #         self.get_logger().info(f"{self.agent_name} creating new Product Automaton for task {task_id}...")
    #         robot_model = TSModel(state_models)
    #         product_automaton = ProdAut(robot_model, mission_to_buchi(hard_task, soft_task), self.initial_beta)
    #         # self.get_logger().info(f"Step 4: ProdAut 完成，product nodes: {len(product_automaton.nodes)}, edges: {len(product_automaton.edges)}")
    #         product_automaton.graph['ts'].build_full()  # Fully initialize the automaton during construction
    #         # self.get_logger().info(f"Step 5: build_full 完成，product nodes: {len(product_automaton.nodes)}, edges: {len(product_automaton.edges)}")
    #         product_automaton.build_full_relaxed()
    #         # self.get_logger().info(f"Step 6: build_full_relaxed 完成，product nodes: {len(product_automaton.nodes)}, edges: {len(product_automaton.edges)}")


    #         # self.get_logger().info(f"Product Automaton Nodes before calling optimal: {product_automaton.graph['initial']}")

    #         # Initialize LTL Planner for each task
    #         # self.get_logger().info(f"Step 7: LTLPlanner 初始化 开始")
    #         ltl_planner = LTLPlanner(robot_model, hard_task, soft_task, self.initial_beta, self.gamma)
    #         # self.get_logger().info(f"Step 8: LTLPlanner 初始化 完成")
    #         ltl_planner.optimal(product_automaton, algo=self.algo_type, N=self.grid_size)
    #         # self.get_logger().info(f"Step 9: LTLPlanner.optimal 完成")

    #         # Store the newly created product automaton and LTL planner
    #         self.product_automata[task_id] = product_automaton
    #         self.ltl_planners[task_id] = ltl_planner  # Store LTLPlanner for later use
            
    #         # Initialize storage for the set of possible runs in the product
    #         self.ltl_planners[task_id].curr_ts_state = list(product_automaton.graph['ts'].graph['initial'])[0]
    #         self.ltl_planners[task_id].posb_runs = set([(n,) for n in product_automaton.graph['initial']])

    #         # Seperate planner for score calculating
    #         score_planner = LTLPlanner(robot_model, hard_task, soft_task, self.initial_beta, self.gamma)
    #         score_planner.optimal(product_automaton, algo=self.algo_type, N=self.grid_size)

    #         # Store the newly created product automaton and LTL planner
    #         self.score_planners[task_id] = score_planner  # Store LTLPlanner for later use
            
    #         # Initialize storage for the set of possible runs in the product
    #         self.score_planners[task_id].curr_ts_state = list(product_automaton.graph['ts'].graph['initial'])[0]
    #         self.score_planners[task_id].posb_runs = set([(n,) for n in product_automaton.graph['initial']])


    # def update_and_run_automaton(self, task_id, new_initial_ts_state):
    #     # Check if `ProdAut` for `task_id` exists
    #     if task_id not in self.product_automata:
    #         raise ValueError(f"Product Automaton for task {task_id} has not been initialized. Call initialize_automaton first.")
        
    #     if task_id not in self.ltl_planners:
    #         raise ValueError(f"LTLPlanner for task {task_id} has not been initialized. Call initialize_automaton first.")

    #     product_automaton = self.product_automata[task_id]

    #     # Ensure correct format if input is a dictionary
    #     if isinstance(new_initial_ts_state, dict):
    #         new_initial_ts_state = (new_initial_ts_state['2d_pose_region'], new_initial_ts_state['Drone_state'])

    #     # Update the initial state in the expected format
    #     product_automaton.graph['ts'].graph['initial'] = {new_initial_ts_state}
    #     product_automaton.build_initial()  # Only update the initial state

    #     # Update LTL Planner's current state and possible runs
    #     self.ltl_planners[task_id].curr_ts_state = list(product_automaton.graph['ts'].graph['initial'])[0]
    #     self.ltl_planners[task_id].posb_runs = set([(n,) for n in product_automaton.graph['initial']])

    #     # Use the corresponding LTLPlanner instance for this task
    #     self.ltl_planners[task_id].optimal(product_automaton, algo=self.algo_type, N=self.grid_size)

    
    def build_and_run_automaton(self, ltl_formula, initial_state_ts_dict):
        self.get_logger().info(f"Building and running automaton for {self.agent_name}.")

        if self.add_point_to_map:
            self.add_point_to_map = False
            self.get_logger().info(f"New task points: {self.new_task_points} appear, start updating map......")
            state_models = state_models_from_ts(self.transition_system, initial_state_ts_dict, self.new_task_points)
            self.get_logger().info(f"Finish updating map, building new automaton......")
        else:
            state_models = state_models_from_ts(self.transition_system, initial_state_ts_dict, self.new_task_points)

        # Update self.nodes and self.actions to be consistent with the newly built graph
        self.nodes = self.transition_system['state_models']['2d_pose_region']['nodes']
        self.actions = self.transition_system['actions']

        # Get task ltl specification
        hard_task = ltl_formula
        soft_task = ''


        buchi = mission_to_buchi(hard_task, soft_task)
        self.robot_model = TSModel(state_models)

        self.product_automaton = ProdAut(self.robot_model, buchi, self.initial_beta)

        self.product_automaton.graph['ts'].build_full()
        self.product_automaton.build_full_relaxed()

        self.ltl_planner = LTLPlanner(self.robot_model, hard_task, soft_task, self.initial_beta, self.gamma)

        self.ltl_planner.optimal(self.product_automaton, algo=self.algo_type, N=self.grid_size)

        # initialize storage of set of possible runs in product
        self.ltl_planner.curr_ts_state = list(self.ltl_planner.product.graph['ts'].graph['initial'])[0]
        self.ltl_planner.posb_runs = set([(n,) for n in self.ltl_planner.product.graph['initial']])
        self.get_logger().info(f"LTL Planner for {self.agent_name} built and run.")


    def setup_pub_sub(self):

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Set up publishers (replace YourMsgType with the correct message type)
        self.prefix_plan_pub = self.create_publisher(LTLPlan, 'prefix_plan', 10)
        self.suffix_plan_pub = self.create_publisher(LTLPlan, 'suffix_plan', 10)
        self.publisher_ = self.create_publisher(RelayResponse, 'replanning_response', 10)   
        self.score_list_pub = self.create_publisher(ScoreList, 'score_list', qos_profile)
        self.finish_build_auto_pub = self.create_publisher(RobotID, 'finish_building_auto', 10)  
        self.request_cluster_pub = self.create_publisher(ClusterRequest, 'cluster_request', 10)   
        self.no_task_pub = self.create_publisher(NoTask, 'no_task', 10)
        
        # Initialize services 
        self.subscriber_ = self.create_subscription(
            RelayRequest,
            'replanning_request',
            self.listener_callback,
            10)
        
        # self.taskassignment_sub = self.create_subscription(
        #     TaskAssignment,
        #     'task_assignment',
        #     self.taskassignment_callback,
        #     10
        # )

        # self.new_task_sub = self.create_subscription(
        #     TaskReAssignment,
        #     'task_reassignment',
        #     self.new_task_callback,
        #     10
        # )

        self.cluster_task_sub = self.create_subscription(
            ClusterTaskassign,
            'ClusterTaskassign',
            self.cluster_task_callback,
            10
        )

        self.score_request_sub = self.create_subscription(
            ScoreRequest,
            'score_request',
            self.get_score_list,
            10
        )

        self.task_request_sub = self.create_subscription(
            TaskRequestCluster, 
            'task_assignment_request', 
            self.assign_new_task,
            10
        )

        self.add_task_sub = self.create_subscription(
            AddTask,
            'add_task',
            self.add_task_callback,
            10
        )

        # self.agent_fail_sub = self.create_subscription(
        #     AgentFail,
        #     'agent_failure',
        #     self.agent_fail_callback,
        #     10
        # )
    

    def add_task_callback(self, msg):
        self.get_logger().info(f"Received add task command from LLM, new {msg.task_type} task appears at {msg.location}, updating map......")
        self.add_point_to_map = True
        # 传递(label, 坐标)
        point = tuple(msg.location)  # 转成tuple更保险
        label = msg.task_label
        
        # 检查是否已经存在相同的point和label组合
        new_task = (point, label)
        if new_task not in self.new_task_points:
            self.new_task_points.append(new_task)
            self.get_logger().info(f"Added new task point: {new_task}")
        else:
            self.get_logger().info(f"Task point already exists, skipping: {new_task}")



    def assign_new_task(self, msg):
        # Update current robot pose
        self.pose_index = msg.pose_index
        self.position = msg.position

        self.request_cluster_msg = ClusterRequest()
        self.request_cluster_msg.robot_id = int(re.findall(r'\d+', self.agent_name)[0])
        self.request_cluster_msg.position = self.position
        self.request_cluster_pub.publish(self.request_cluster_msg)
        # self.get_logger().info(f"No more tasks remaining for {self.agent_name}, requesting for new cluster.......................................")



    def cluster_task_callback(self, msg):
        self.get_logger().info(f"Received task list for {self.agent_name}: {msg.route_labels}")

        self.current_task_list = list(msg.route_labels)
        # 只保留pickup的点（两个字母的label），并保持顺序
        self.pickup_labels = [label for label in self.current_task_list if len(label) == 2]

        self.current_ltl_formula = self.generate_ltl_formula(msg.route_labels)

        new_initial_pose = self.pose_index
        formatted_pose = f'{new_initial_pose}'

        # Prepare initial state
        self.initial_state_ts_dict = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }

        self.build_and_run_automaton(self.current_ltl_formula, self.initial_state_ts_dict)
        
        # Publish the generated plan
        self.publish_cluster_plan()

    # def handle_task(self, task_index):
    #     # Get and format current pose
    #     new_initial_pose = self.pose_index
    #     formatted_pose = f'{new_initial_pose}'

    #     # Prepare initial state
    #     self.initial_state_ts_dict = {
    #         '2d_pose_region': formatted_pose,
    #         'Drone_state': 'unloaded'
    #     }

    #     # Verify task existence
    #     task_id = f'task{int(task_index)}'
    #     self.cur_task = task_id
    #     if task_id not in self.task_data:
    #         self.get_logger().error(f"Invalid task index received: {task_index}")
    #         return

    #     # Run automaton and plan
    #     # self.get_logger().info(f"Calculating plan for {task_id} assigned to {self.agent_name}")
    #     self.update_and_run_automaton(task_id, self.initial_state_ts_dict)

    #     # Publish result
    #     self.publish_plan(task_id)

    def get_score_list(self, msg):
        formatted_pose = f'{msg.pose_index}'
        initial_state = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }

        self.score_list = []

        for i in self.task_index:
            task_id = f'task{i}'
            self.update_score_automaton(task_id, initial_state)
            self.score_list.append(self.build_score_list(task_id))  # Append scores to the list

        self.pub_score()

    def init_score_list(self):
        formatted_pose = f'{self.init_state}'
        initial_state = {
            '2d_pose_region': formatted_pose,
            'Drone_state': 'unloaded'
        }

        self.score_list = []
        for i in self.task_index:
            task_id = f'task{i}'
            self.update_score_automaton(task_id, initial_state)
            self.score_list.append(self.build_score_list(task_id))  # Append scores to the list
        self.pub_score()


    def pub_score(self):
        self.get_logger().info(f'Finish calculating score for {self.agent_name}: {self.score_list}')
        score_msg = ScoreList()
        score_msg.robot_id = self.agent_name
        score_msg.score_list = self.score_list
        self.score_list_pub.publish(score_msg)
        self.get_logger().info(f'Publish score list for {self.agent_name}...')

    def build_score_list(self, task_id):
        if self.score_planners[task_id].run is not None:
            # Prefix Score
            cal_score_pre = LTLPlan()
            cal_score_pre.action_sequence = self.score_planners[task_id].run.pre_plan
            pre_score = len(cal_score_pre.action_sequence)  # Ensure correct length calculation
            
            # Suffix Score
            cal_score_suf = LTLPlan()
            cal_score_suf.action_sequence = self.score_planners[task_id].run.suf_plan
            suf_score = len(cal_score_suf.action_sequence)  # Ensure correct length calculation
            
            score = max(0, 100 - (pre_score + suf_score))  # Prevent negative scores
            return score
        return 0  # Return 0 if no score is calculated
        
    
    def publish_cluster_plan(self):
        """
        发布从cluster_task_callback生成的计划
        使用self.ltl_planner（单数形式）
        """
        self.get_logger().info(f"Publishing cluster plan for {self.agent_name}.")
        if self.ltl_planner.run is not None:
            # Prefix plan
            #-------------
            self.prefix_plan_msg = LTLPlan()
            self.prefix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.prefix_plan_msg.action_sequence = self.ltl_planner.run.pre_plan
            self.prefix_plan_msg.ts_state_sequence = []
            # 只发布pickup_labels
            self.prefix_plan_msg.route_labels = self.pickup_labels if hasattr(self, 'pickup_labels') else []
            
            # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planner.run.line:
                ts_state_msg = TransitionSystemState()
                # If TS state is more than 1 dimension (is a tuple)
                if type(ts_state) is tuple:
                    ts_state_msg.states = list(ts_state)
                # Else state is a single string
                else:
                    ts_state_msg.states = [ts_state]
                # Add to plan TS state sequence
                self.prefix_plan_msg.ts_state_sequence.append(ts_state_msg)

            # Publish
            self.get_logger().info("Publishing Prefix Plan")
            self.get_logger().info(f"Prefix Plan: {self.prefix_plan_msg.action_sequence}")
            self.prefix_plan_pub.publish(self.prefix_plan_msg)

            # Suffix plan
            #-------------
            self.suffix_plan_msg = LTLPlan()
            self.suffix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.suffix_plan_msg.action_sequence = self.ltl_planner.run.suf_plan
            self.suffix_plan_msg.ts_state_sequence = []
            # 只发布pickup_labels
            self.suffix_plan_msg.route_labels = self.pickup_labels if hasattr(self, 'pickup_labels') else []
            
            # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planner.run.loop:
                ts_state_msg = TransitionSystemState()
                # If TS state is more than 1 dimension (is a tuple)
                if type(ts_state) is tuple:
                    ts_state_msg.states = list(ts_state)
                # Else state is a single string
                else:
                    ts_state_msg.states = [ts_state]

                # Add to plan TS state sequence
                self.suffix_plan_msg.ts_state_sequence.append(ts_state_msg)

            # Publish
            self.get_logger().info("Publishing Suffix Plan")
            self.get_logger().info(f"Suffix Plan: {self.suffix_plan_msg.action_sequence}")
            self.suffix_plan_pub.publish(self.suffix_plan_msg)
        else:
            self.get_logger().warn("No plan available to publish")

    #----------------------------------------------
    # Publish prefix and suffix plans from planner
    #----------------------------------------------
    def publish_plan(self, task_id):
        
        if self.ltl_planners[task_id].run is not None:
            # self.get_logger().info("in push plan...")
            # Prefix plan
            #-------------
            self.prefix_plan_msg = LTLPlan()
            self.prefix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.prefix_plan_msg.action_sequence = self.ltl_planners[task_id].run.pre_plan
            self.prefix_plan_msg.ts_state_sequence = []
            self.prefix_plan_msg.cur_task = self.task_number
            # # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planners[task_id].run.line:
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
            # self.get_logger().info("Publish Prefix Plan")
            # print(prefix_plan_msg.action_sequence)
            self.prefix_plan_pub.publish(self.prefix_plan_msg)

            # Suffix plan
            #-------------
            self.suffix_plan_msg = LTLPlan()
            self.suffix_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.suffix_plan_msg.action_sequence = self.ltl_planners[task_id].run.suf_plan
            self.suffix_plan_msg.ts_state_sequence = []
            # # Go through all TS state in plan and add it as TransitionSystemState message
            for ts_state in self.ltl_planners[task_id].run.loop:
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
            # self.get_logger().info("Publish Suffix Plan")
            self.suffix_plan_pub.publish(self.suffix_plan_msg)


    def replanning_modify_callback(self, task_replanning_req):
        if task_replanning_req:
            # self.get_logger().info("Replanning [modify] Callback")
            update_info = dict()
            update_info["modified"] = set()
            update_info["deleted"] = set()
            update_info["relabel"] = set()
            # TODO: check both from_pose and to_pose have only two elements
            # change position in tuple to ts node of the format ('c0_r5', 'unloaded')
            for node in self.ltl_planner.product.graph['ts'].nodes():
                if tuple(task_replanning_req.from_pose) == self.nodes[node[0]]['attr']['pose']:
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.to_pose) == self.nodes[succ_node[0]]['attr']['pose']:
                            update_info["modified"].add((node, succ_node, task_replanning_req.cost))
                if tuple(task_replanning_req.to_pose) == self.nodes[node[0]]['attr']['pose']:
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.from_pose) == self.nodes[succ_node[0]]['attr']['pose']:
                            update_info["modified"].add((node, succ_node, task_replanning_req.cost))
            # print(update_info["modified"])
            modified_edges_dict = self.ltl_planner.revise_product(update_info)
            # self.get_logger().info("Finished revise")
            
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
                # self.get_logger().info("service has been transmitted")
                self.publisher_.publish(res)
                return
            
        self.get_logger().error("Error in replanning modify callback")
        res.success = False
        self.publisher_.publish(res)
        return 
        
    def replanning_delete_callback(self, task_replanning_req):
        if task_replanning_req:
            # self.get_logger().info("Replanning [Delete] Callback")
            update_info = dict()
            update_info["modified"] = set()
            update_info["deleted"] = set()
            update_info["relabel"] = set()
            # TODO: check both from_pose and to_pose have only two elements
            # change position in tuple to ts node of the format ('c0_r5', 'unloaded')
            for node in self.ltl_planner.product.graph['ts'].nodes():
                if tuple(task_replanning_req.from_pose) == self.nodes[node[0]]['attr']['pose']:
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.to_pose) == self.nodes[succ_node[0]]['attr']['pose'] :
                            update_info["deleted"].add((node, succ_node))
                if tuple(task_replanning_req.to_pose) == self.nodes[node[0]]['attr']['pose']:
                    for succ_node in self.ltl_planner.product.graph['ts'].successors(node):
                        if tuple(task_replanning_req.from_pose) == self.nodes[succ_node[0]]['attr']['pose']:
                            update_info["deleted"].add((node, succ_node))
            # print(update_info["deleted"])
            modified_edges_dict = self.ltl_planner.revise_product(update_info)
            # self.get_logger().info("finished revise")
            
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
            # self.get_logger().info("finished revise successfully")
            
            res = RelayResponse()
            if success:
                # self.get_logger().info("start preparing for the ")
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
                # self.get_logger().info("service has been transmitted ")
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