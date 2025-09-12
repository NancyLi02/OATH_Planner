# LLM-OATH: Obstacle-Aware Multi-Agent Task Assignment and Planning

> **LLM-OATH** is an advanced framework for obstacle-aware multi-agent task assignment and planning in large, dynamic environments, enabling intelligent robot team collaboration through integration with large language models.


## Core Functionality
- **Intelligent Task Assignment**: Scalable task allocation algorithm based on cluster-auction architecture
- **Adaptive Map Generation**: Generates adaptive Halton maps with increased sampling density near obstacles
- **Obstacle-Aware Distance Computation**: Uses multi-source Dijkstra algorithm to compute obstacle-aware task-to-task distances
- **Dynamic Path Planning**: Integrates LTL-D* algorithm for dynamic, obstacle-aware path planning
- **LLM Integration**: Leverages large language models to translate human instructions into formal specifications and adapt to changing intent in real time


## Quick Start

### Prerequisites

- **ROS2 Humble** - [Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
  - Learn how to build a ROS2 Humble Workspace - [Building ROS2 Humble Workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)
- **Python 3.8+**
- **Ubuntu 22.04**

### 1. Clone Repository

```bash
git clone https://github.com/NancyLi02/multiagent_planner.git
cd multiagent_planner
# Optional: rename folder to 'src'
mv multiagent_planner src
```

### 2. Install Dependencies

#### Python Dependencies
```bash
pip install numpy pandas matplotlib shapely scikit-learn scipy pyyaml
```

#### ROS2 Dependencies
Please refer to the [LTL-D* Planner](https://github.gatech.edu/jren313/lmco/tree/ros2) installation guide.

### 3. Build Workspace

```bash
# Execute in ros2_ws directory
colcon build
source install/setup.bash
```

## Configuration

### Configuration Files

The system uses two main configuration files:

#### Task_Points.yaml
```
lmco/ltl_automaton_planner/config/Task_Points.yaml
```
This file contains all robot parameters, task points, and delivery points.

#### wall.yaml
```
lmco/ltl_automaton_planner/config/wall.yaml
```
This file defines the map layout and obstacle positions.

### Configurable Parameters

Based on the `Task_Points.yaml` file, you can configure the following parameters:

#### Robot Configuration
- **Robot Positions** (`robot_positions`): Set initial positions for each robot
- **Special Robots** (`special_robot`): Designate specific robots with special capabilities

#### Task Configuration
- **Task Points** (`task_points`): Define task execution locations with coordinates and labels
  - Format: `"x,y": 'label'` (e.g., `"1,4": 'bb'`)
- **Task to Delivery Mapping** (`task_to_delivery`): Map task categories to specific task points
- **Special Labels** (`special_labels`): Define special task points that require specific handling

#### Delivery Configuration
- **Delivery Points** (`delivery_points`): Set item delivery locations
  - Format: `"x,y": 'delivery_id'` (e.g., `"6,3": 'b'`)

#### Map Configuration (wall.yaml)
- **Wall Lines** (`lines`): Define obstacle boundaries using line segments
  - Format: `[[x1, y1], [x2, y2], ...]` - Each line segment connects consecutive points
  - Example: `[[0, 3], [2, 3], [2, 4]]` - Creates a wall from (0,3) to (2,3) to (2,4)
- **Map Layout**: Modify the `lines` array to change obstacle positions and create different environments


> **Tip**: All parameter modifications are done through the YAML files without code changes! Every time you change the task number, robot number, positions, or map layout, you need to run Usage Guide steps 1 and 2 to make it work.

## Usage Guide

### Step 1: Generate Adaptive Halton Map
```bash
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/Adaptive_Halton.py
```

### Step 2: Compute Obstacle-Aware Task Distance Matrix
```bash
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/multi_source_dijkstra.py
```

### Step 3: (Optional) Visualize Task Clustering
```bash
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/multi_source_cluster.py
```

### Step 4: Launch Multi-Agent Task Assignment and Planning System
```bash
# Ensure ROS2 environment is properly configured
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch the system
ros2 launch ltl_automaton_planner ltl_planner_isaac_launch.py
```

## Project Structure

```
ros2_ws/
├── build/                    # Build directory
├── install/                  # Install directory
├── log/                      # Log directory
└── src/                      # Source code directory
    ├── interfaces_hmm_sim/   # Interface package
    └── lmco/                 # Main functionality package
        ├── ltl_automaton_msgs/      # Message definitions
        └── ltl_automaton_planner/   # Core planner
            ├── config/
            │   ├── Task_Points.yaml # Task configuration file
            │   └── wall.yaml        # Map layout configuration file
            └── ltl_automaton_planner/
                ├── Adaptive_Halton.py
                ├── multi_source_dijkstra.py
                └── multi_source_cluster.py
```


## Contact

- **Maintainer**: nan.li@gatech.edu
---


For questions or issues, please open an issue or contact the maintainer (nan.li@gatech.edu).
