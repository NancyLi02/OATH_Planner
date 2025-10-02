###################################
##### Simple Drone Simulation #####
#####     using Isaac Sim     #####
###################################
#####      Haris Miller       #####
#####     GA Tech - LIDAR     #####
###################################

## Start Simulation
import argparse
import sys
import time
import numpy as np
import csv
import yaml
import os

import rclpy
from rclpy.node import Node
from interfaces_hmm_sim.msg import AgentPoses, Status, PathPlan, ReplanStatus, AgentGoTo
from ltl_automaton_msgs.msg import TransitionSystemState, TransitionSystemStateStamped, LTLPlan, RelayResponse

class NumAgentsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            # Ensure the first value is an integer (number of agents)
            num_agents = int(values[0])
            # Parse the remaining values as a list of (x, y) tuples
            points = []
            for i in range(1, len(values), 2):
                x = float(values[i])
                y = float(values[i + 1])
                points.append((x, y))
            if len(points) != num_agents:
                raise ValueError(f"Number of agents does not match the number of (x, y) points provided for {self.dest}.")
            # Store the parsed values in the namespace
            setattr(namespace, self.dest, {"num_agents": num_agents, "points": points})
        except (ValueError, IndexError):
            raise argparse.ArgumentError(self, f"Invalid format for {self.dest}. Expected: <int> <x1> <y1> <x2> <y2> ...")

def get_parser():
    parser = argparse.ArgumentParser(description="Isaac Sim Multi-Agent Halton Simulation")
    parser.add_argument("--world", type=str, default="Halton_env.usd", help="World to load")
    parser.add_argument("--halton_file", type=str, default="halton_points.csv", help="Halton file to load")
    parser.add_argument("--environment_scale", type=float, default=1.0, help="Scale of the environment")
    parser.add_argument(
        "--num_quads",
        nargs="+",
        action=NumAgentsAction,
        help="Number of quadcopters followed by their (x, y) initial positions. Example: --num_quads 2 0.0 0.0 1.0 1.0",
    )
    parser.add_argument(
        "--num_turtle",
        nargs="+",
        action=NumAgentsAction,
        help="Number of TurtleBots followed by their (x, y) initial positions. Example: --num_turtle 2 2.0 2.0 3.0 3.0",
    )
    parser.add_argument("--setup_file", type=str, help="Setup to load")
    return parser

def keyboard_event(event, *args, **kwargs):
    global start

    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.P:
            if start:
                print("P key pressed: Pausing simulation.")
            else:
                print("P key pressed: Starting simulation.")
            start = not start
        if event.input == carb.input.KeyboardInput.X:
            print("X key pressed: Exiting simulation.")
            simulation_app.close()
            rclpy.shutdown()

                
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    setup_file_flag = False

    # Check if --setup_file is used with other arguments
    if args.setup_file:
        setup_file_flag = True
        args_dict = vars(args)
        # List of arguments with default values to ignore
        default_values = {
            "world": "Halton_env.usd",
            "halton_file": "halton_points.csv",
            "environment_scale": 1.0,
            "num_quads": None,
            "num_turtle": None,
        }
        # Filter out arguments that are not explicitly provided by the user
        other_args = {
            key: value for key, value in args_dict.items()
            if key != "setup_file" and value != default_values.get(key)
        }
        if other_args:
            print(args_dict)
            raise ValueError("The --setup_file argument cannot be used with any other arguments.")

    # Setup Isaac Sim
    import isaacsim
    from omni.isaac.kit import SimulationApp

    CONFIG = {"width": 2560, "height": 1440, "window_width": 3840, "window_height": 2160, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
    simulation_app = SimulationApp(CONFIG)

    import carb
    import omni

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage

    from omni.isaac.dynamic_control import _dynamic_control
    from omni.isaac.core.articulations import Articulation

    from pxr import Gf, UsdPhysics, Usd, UsdGeom, Sdf
    import omni.usd

    # Import custom modules
    import drone
    import helper as hp
    from helper import elapsed_time, parse_action_sequence
    
    ####################################################################################################
    ##### Initialize Setup #####

    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    env_directory = os.path.join(current_directory, "env")

    halton_file_name = args.halton_file
    usd_file_name = args.world
    num_quads = 0
    num_turtle = 0
    environment_scale = args.environment_scale
    agents = []
    if setup_file_flag:
        print("Setup file provided. Loading setup...")
        # Load setup file
        setup_file_path = os.path.join(current_directory, args.setup_file)
        if not os.path.exists(setup_file_path):
            raise FileNotFoundError(f"Setup file '{setup_file_path}' does not exist.")
        
        with open(setup_file_path, 'r') as f:
            setup_data = yaml.safe_load(f)
        
        files = setup_data.get("files", [])
        for file_entry in files:
            if "halton" in file_entry:
                halton_file_name = file_entry["halton"]
            if "environment" in file_entry:
                usd_file_name = file_entry["environment"]
        
        setup = setup_data.get("setup", [])
        for setup_entry in setup:
            if "environment_scale" in setup_entry:
                environment_scale = setup_entry["environment_scale"]
            if "num_quads" in setup_entry:
                num_quads = setup_entry["num_quads"]
            if "num_turtle" in setup_entry:
                num_turtle = setup_entry["num_turtle"]

        # print(f"Number of quadcopters: {num_quads}")
        # print(f"Number of TurtleBots: {num_turtle}")

        if (num_quads > 0) or (num_turtle > 0):
            agents = setup_data.get("agents", [])
            for agent in agents:
                agent_id = agent.get("id")
                agent_type = agent.get("type")
                initial_position = agent.get("initial_position")
                agent_path = f"/World/quads/quad{agent_id}/quad" if agent_type == "quad" else f"/World/turtles/turtlebot{agent_id}/turtlebot"
                agent["path"] = agent_path
                print(f"Agent ID: {agent_id}, Type: {agent_type}, Initial Position: {initial_position}, Path : {agent_path}")
    else:
        if args.num_quads:
            num_quads = args.num_quads["num_agents"]
            print(f"Number of quadcopters: {args.num_quads['num_agents']}")
            print(f"Initial positions (quads): {args.num_quads['points']}")
            for i, position in enumerate(args.num_quads["points"], start=1):
                agents.append({
                    "id": i,
                    "type": "quad",
                    "initial_position": position,
                    "path": f"/World/quads/quad{i}/quad"
                })

        if args.num_turtle:
            num_turtle = args.num_turtle["num_agents"]
            print(f"Number of TurtleBots: {args.num_turtle['num_agents']}")
            print(f"Initial positions (turtles): {args.num_turtle['points']}")
            for i, position in enumerate(args.num_turtle["points"], start=num_quads + 1):
                agents.append({
                    "id": i,
                    "type": "turtle",
                    "initial_position": position,
                    "path": f"/World/turtles/turtlebot{i}/turtlebot"
                })

        for agent in agents:
            print(f"Agent ID: {agent['id']}, Type: {agent['type']}, Initial Position: {agent['initial_position']}, Path : {agent['path']}")
        
    halton_path = os.path.join(env_directory, halton_file_name)
    print("Halton file path:", halton_path)
    if not os.path.exists(halton_path):
        raise FileNotFoundError(f"Halton file '{halton_path}' does not exist.")
    
    
    num_agents = num_quads + num_turtle
    print("Number of agents:", num_agents)

    ####################################################################################################
    ##### ROS2 Setup #####

    replan_flag = []
    for i in range(num_agents):
        replan_flag.append(False)

    rclpy.init()

    # RelayResponse callback
    def relay_response_callback(msg, agent_node):
        idx = agent_node["number"]
        replan_flag[idx] = True
        agent_node["new_prefix_actions"] = msg.new_plan_prefix.action_sequence
        agent_node["node"].get_logger().info(f"Received prefix actions")
        agent_node["new_suffix_actions"] = msg.new_plan_suffix.action_sequence
        agent_node["node"].get_logger().info(f"Received suffix actions")

    # PrefixPlan callback
    def prefix_plan_callback(msg, agent_node):
        agent_node["prefix_actions"] = msg.action_sequence
        agent_node["node"].get_logger().info(f"Received prefix plan actions")

    # SuffixPlan callback
    def suffix_plan_callback(msg, agent_node):
        agent_node["suffix_actions"] = msg.action_sequence
        agent_node["node"].get_logger().info(f"Received suffix plan actions")

    def next_step_callback(msg, agent_node):
        agent_node["next_step"] = msg.next_step
        agent_node["next_flag"] = msg.next_flag
        agent_node["node"].get_logger().info(f"Received next step command")

    # Create agent nodes
    def create_agent_nodes(num_agents):
        agent_nodes = []
        for i in range(num_agents):
            agent_name = f"robot{i + 1}"
            namespace = f"/{agent_name}"
            print(f"Creating node for {agent_name} in namespace {namespace}")
            # node = rclpy.create_node(f"isaac_node", namespace=namespace)
            try:
                node = rclpy.create_node(f"isaac_node", namespace=namespace)
            except Exception as e:
                print(f"Error creating node for {agent_name}: {e}")
                continue

            # Publishers
            pose_publisher = node.create_publisher(AgentPoses, f"{namespace}/poses", 10)
            status_publisher = node.create_publisher(Status, f"{namespace}/status", 10)

            # Initialize agent-specific data
            agent_node = {
                "name": agent_name,
                "node": node,
                "number": i,
                "pose_publisher": pose_publisher,
                "status_publisher": status_publisher,
                "new_prefix_actions": [],
                "new_suffix_actions": [],
                "prefix_actions": [],
                "suffix_actions": [],
                "next_step": [],
                "next_flag": 'i',
            }

            # Closures to bind agent-specific data
            def relay_callback(msg, agent_node=agent_node):
                relay_response_callback(msg, agent_node)

            def prefix_callback(msg, agent_node=agent_node):
                prefix_plan_callback(msg, agent_node)

            def suffix_callback(msg, agent_node=agent_node):
                suffix_plan_callback(msg, agent_node)

            # Subscribers
            relay_response_subscriber = node.create_subscription(
                RelayResponse, f"{namespace}/replanning_response", relay_callback, 10
            )
            prefix_plan_subscriber = node.create_subscription(
                LTLPlan, f"{namespace}/prefix_plan", prefix_callback, 10
            )
            suffix_plan_subscriber = node.create_subscription(
                LTLPlan, f"{namespace}/suffix_plan", suffix_callback, 10
            )

            next_step_subscriber = node.create_subscription(
                AgentGoTo,
                f"{namespace}/agent_next",
                lambda msg, agent_node=agent_node: next_step_callback(msg, agent_node),
                10
            )

            agent_node["relay_response_subscriber"] = relay_response_subscriber
            agent_node["prefix_plan_subscriber"] = prefix_plan_subscriber
            agent_node["suffix_plan_subscriber"] = suffix_plan_subscriber
            agent_node["next_step_subscriber"] = next_step_subscriber

            agent_nodes.append(agent_node)

        return agent_nodes


    agent_nodes = create_agent_nodes(num_agents)
    print("ROS2 Nodes initialized!")


    ####################################################################################################
    ##### Open World #####

    halton_points = []
    labelled_points = {}

    with open(halton_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                x, y = float(row[0]), float(row[1])
                label = row[2].strip() if row[2] else None
                halton_points.append((x, y))
                if label:
                    labelled_points[label] = (x, y)

    x_max = np.ceil(max(point[0] for point in halton_points))
    y_max = np.ceil(max(point[1] for point in halton_points))

    x_offset = x_max / 2
    y_offset = y_max / 2

    # print("Parsed Halton Points:", halton_points)
    # print("Label-to-Point Mapping:", label_to_point)

    usd_path = os.path.join(current_directory,"environments", usd_file_name)

    open_stage(usd_path)
    world = World()

    ####################################################################################################
    ##### Drone Controller Parameters #####

    ## Simple PID Gains
    kp = 9
    kp_z = 3
    kd = 2
    kd_z = 0.5

    PID = [kp,0,kd]
    zPID = [kp_z,0,kd_z]

    ####################################################################################################
    ##### Initializations #####

    wait_val = 25

    start = False
    init_paths = False
    goal_reached = False

    stage = omni.usd.get_context().get_stage()
    dc=_dynamic_control.acquire_dynamic_control_interface()

    ## Create drone object
    robot_list = []
    finalgoal_check = []
    init_start = []
    wait_count = []
    stay_flag = []
    robot_count = 1
    for agent in agents:
        drone_new = drone.Drone(agent["id"],agent["path"],agent["type"],PID,zPID,environment_scale,x_offset)
        # drone_new.setPath(x_grid_list[robot_count-1],y_grid_list[robot_count-1],dz_list[robot_count-1],key)
        drone_new.setNext(agent["initial_position"][0], agent["initial_position"][1], 0.0, 'i')

        robot_new = {
            "info": drone_new,
            "prim": Articulation(drone_new.path, name=drone_new.name),
            "rigidbody": UsdPhysics.RigidBodyAPI.Get(stage, drone_new.path)
        }
        finalgoal_check.append(False)
        init_start.append(False)
        robot_list.append(robot_new)
        wait_count.append(wait_val)
        stay_flag.append(False)
        robot_count += 1

    ####################################################################################################
    ##### Execution #####
    
    world.reset()
    # rate = node.create_rate(1000) # Settings a rate too low may cause the simulation to crash
                                    # It is probably better to keep this off unless needed
    
    while rclpy.ok():
        for agent in agent_nodes:
            rclpy.spin_once(agent["node"], timeout_sec=0)
        appwindow = omni.appwindow.get_default_app_window()
        input = carb.input.acquire_input_interface()
        input.subscribe_to_keyboard_events(appwindow.get_keyboard(), keyboard_event)
        
        if start:
            if not init_paths:
                for agent in agent_nodes:
                    num = agent["number"]
                    if agent["next_flag"] == 'g':
                        init_status = Status()
                        init_status.agent = agent["name"]
                        init_status.start = True
                        agent["status_publisher"].publish(init_status)
                    else:
                        init_start[num] = True
                        init_x = agent["next_step"][0]
                        init_y = agent["next_step"][1]
                        # print(f"Agent {num} - Type: {agents[num]['type']})")
                        if agents[num]["type"] == "quad":
                            init_z = 3.0
                        else:
                            init_z = 0.0
                        init_flag = agent["next_flag"]
                        robot_list[num]["info"].setNext(init_x, init_y, init_z, init_flag)


        if (np.all(np.array(init_start))):
            init_paths = True   
        # print(args.world)
        # print(args.num_quads)
        # print(args.num_turtle)

        if start and init_paths:
            for agent in agent_nodes:
                num = agent["number"]
                next_x = agent["next_step"][0]
                next_y = agent["next_step"][1]
                next_flag = agent["next_flag"]
                # print(f"Agent {num} - Flag: {next_flag})")
                if agents[num]["type"] == "quad":
                    if next_flag == 'g':
                        next_z = 3.0
                    elif next_flag == 'l' or next_flag == 'u':
                        next_z = 2.0
                    elif next_flag == 's':
                        next_z = 1.0
                    elif next_flag == 'b':
                        next_z = 0.0
                    else:
                        next_z = 3.0
                robot_list[num]["info"].setNext(next_x, next_y, next_z, next_flag)
                robot_obj = robot_list[num]["info"]

                robot_status = Status()

                object = dc.get_rigid_body(robot_obj.path)
                object_pose=dc.get_rigid_body_pose(object)
                global_pose,global_orient = robot_list[num]["prim"].get_world_pose()
                robot_rb = robot_list[num]["rigidbody"]

                xpose = global_pose[0]
                ypose = global_pose[1]
                zpose = global_pose[2]

                robot_obj.pose = [xpose,ypose,zpose]
                pose_prev = robot_obj.pose_prev
                del_prev = robot_obj.del_prev

                xgoal = robot_obj.xgoal
                ygoal = robot_obj.ygoal
                zgoal = robot_obj.zgoal
                flag = robot_obj.flag

                # print(f"Agent {num} - Current Pose: {robot_obj.pose}, Goal: ({xgoal}, {ygoal}, {zgoal}), Flag: {flag}")
                if robot_rb and not stay_flag[num]:
                    delx = xgoal - xpose
                    dely = ygoal - ypose
                    delz = zgoal - zpose

                    kp_mod = kp
                    if delz > 0.5:
                        kp_mod = 0.25*kp
                    elif delz < -0.5 and (np.abs(delx) < 0.5 and np.abs(dely) < 0.5):
                        kp_mod = 0.25*kp
                    vx = kp_mod*delx + kd*(delx - del_prev[0])
                    vy = kp_mod*dely + kd*(dely - del_prev[1])
                    vz = kp_z*(delz) + kd_z*(delz - del_prev[2])
                    v_curr = [vx,vy,vz]

                    if np.abs(delx) <= 0.5 and np.abs(dely) <= 0.5 and np.abs(delz) <= 0.05:
                        print(f"Agent {num} - Flag:  {flag}")
                        if flag == 's' and not stay_flag[num]:
                            # stay_flag[num] = True
                            robot_status.agent = agent["name"]
                            robot_status.arrived = True
                            agent["status_publisher"].publish(robot_status)
                        elif flag == 'b' and not stay_flag[num]:
                            # stay_flag[num] = True
                            robot_status.agent = agent["name"]
                            robot_status.arrived = True
                            agent["status_publisher"].publish(robot_status)
                        elif flag != 's' or flag != 'b':
                            robot_status.agent = agent["name"]
                            robot_status.arrived = True
                            agent["status_publisher"].publish(robot_status)

                    vel = Gf.Vec3f(vx,vy,vz)
                    robot_rb.GetVelocityAttr().Set(vel)
                    # if np.all(np.array(finalgoal_check)):
                    #     print("All goals reach!")
                    #     # start = not start
                else:
                    # finalgoal_check[num] = True

                    vel = Gf.Vec3f(0,0,0)
                    robot_rb.GetVelocityAttr().Set(vel)
        else:
            vel = Gf.Vec3f(0,0,0)
            for robot in robot_list:
                robot["rigidbody"].GetVelocityAttr().Set(vel)

        try:
            # Simulate the world
            world.step(render=True)
        except (KeyboardInterrupt, SystemExit):
            print("Simulation interrupted. Closing simulation...")

    simulation_app.close()
    rclpy.shutdown()
