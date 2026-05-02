# LLM-OATH: Obstacle-Aware Multi-Agent Task Assignment and Planning

> **LLM-OATH** is an advanced framework for obstacle-aware multi-agent task assignment and planning in large, dynamic environments. It enables intelligent multi-robot collaboration over heterogeneous task types and integrates large language models (LLMs) so that human operators can adjust the mission in real time using natural language.

- **Project website**: <https://llm-oath.github.io/>
- **Paper / video / additional results**: see the project website above.


## Core Functionality

- **Heterogeneous Task Assignment**: Cluster–auction–selection framework that supports an arbitrary number of task types and per-robot capability vectors. Robots only bid on clusters whose task composition matches their capabilities, and intra-cluster sequences are optimized via MILP (Gurobi).
- **Adaptive Halton Map**: Builds a Halton-sampled roadmap whose density is increased near obstacles for better spatial resolution where it matters.
- **Obstacle-Aware Distance Computation**: Pre-computes task-to-task distances with a multi-source Dijkstra over the obstacle-aware graph (cached to CSV).
- **Dynamic Path Planning**: LTL-D* based local replanning when new obstacles are detected mid-execution.
- **LLM-Guided Interaction**: A natural-language interface (Tk GUI + LLM parser) translates human instructions into structured commands at runtime — currently `add_task`, `obstacle_update`, and `change_task_priority` are supported.


## Quick Start

### Prerequisites

- **ROS2 Humble** — [Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
  - Workspace setup: [Building a ROS2 Humble Workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)
- **Python 3.8+**
- **Ubuntu 22.04**
- **Gurobi** (used by the MILP intra-cluster solver) — academic license available at <https://www.gurobi.com/academia/>
- **OpenAI API key** (only required for the LLM interaction nodes) — exported as `OPENAI_API_KEY`

### 1. Clone the Repository

```bash
git clone https://github.com/NancyLi02/multiagent_planner.git
cd multiagent_planner
# Optional: rename folder to 'src' to fit a standard ROS2 workspace layout
mv multiagent_planner src
```

### 2. Install Dependencies

#### Python Dependencies

```bash
pip install numpy pandas matplotlib shapely scikit-learn scipy pyyaml \
            networkx pygame openai gurobipy
```

#### ROS2 Dependencies

Please refer to the [LTL-D* Planner](https://github.gatech.edu/jren313/lmco/tree/ros2) installation guide for the underlying ROS2 packages and message types.

### 3. Build the Workspace

```bash
# Execute in ros2_ws directory
colcon build
source install/setup.bash
```


## Configuration

### Configuration Files

The system uses two main configuration files:

#### `Task_Points.yaml`

```
lmco/ltl_automaton_planner/config/Task_Points.yaml
```

Defines task points, delivery points, robot initial positions, the **type partition of tasks** (`typeN_labels`), and **per-robot capability vectors** (`robot_capabilities`).

#### `wall.yaml`

```
lmco/ltl_automaton_planner/config/wall.yaml
```

Defines the map layout and obstacle positions.

### Configurable Parameters

`Task_Points.yaml` is the single source of truth for the runtime configuration of all nodes (task assigner, planner, benchmark, visualizer, LLM parsers).

#### Task Configuration

- **Task points** (`task_points`): Task pickup locations.
  - Format: `"x,y": 'label'` (e.g., `"1,4": 'bb'`)
- **Delivery points** (`delivery_points`): Locations where pickups must be delivered.
  - Format: `"x,y": 'delivery_id'` (e.g., `"6,3": 'b'`)
- **Task-to-delivery mapping** (`task_to_delivery`): Which task labels go to which delivery point.

#### Heterogeneous Task Types (multi-type system)

The framework supports an **arbitrary number of task types**. Each type is declared by adding a `typeN_labels` list, which partitions the `task_points` labels by type:

```yaml
type1_labels: ['bb', 'fb', 'gb', 'he', 'ec']
type2_labels: ['db', 'bc', 'dc', 'fc', 'gc']
type3_labels: ['cc', 'eb', 'gd', 'ce']
type4_labels: ['cd', 'dd', 'ed', 'fd', 'de']
type5_labels: ['bd', 'be', 'ee', 'fe', 'ge', 'cb']
```

You can add `type6_labels`, `type7_labels`, … as needed; the nodes auto-discover the count.

#### Robot Configuration

- **Initial positions** (`robot_positions`):

  ```yaml
  robot_positions:
    "robot1": "1,19"
    "robot2": "11,19"
    "robot3": "9,11"
    "robot4": "11,9"
  ```

- **Capability vectors** (`robot_capabilities`): Binary vector per robot, of length equal to the number of task types. Position *i* is `1` if the robot can execute type *i+1*:

  ```yaml
  robot_capabilities:
    "robot1": [1, 0, 1, 1, 0]   # can do type 1, 3, 4
    "robot2": [1, 0, 0, 1, 1]   # can do type 1, 4, 5
    "robot3": [1, 0, 1, 0, 1]   # can do type 1, 3, 5
    "robot4": [1, 1, 1, 1, 0]   # can do type 1, 2, 3, 4
  ```

  These vectors drive the cluster–robot compatibility score `s_rk = (Δ + ε) / γ_rk`, where `γ_rk = ⟨ψ_k, ζ_r⟩` (cluster type composition · normalized capability), and they also serve as a hard filter — a robot will never be assigned a cluster containing only tasks it cannot perform.

#### Map Configuration (`wall.yaml`)

- **Wall lines** (`lines`): Define obstacle boundaries with line segments.
  - Format: `[[x1, y1], [x2, y2], ...]`. Each line segment connects consecutive points.
  - Example: `[[0, 3], [2, 3], [2, 4]]` creates a wall going (0,3) → (2,3) → (2,4).

> **Tip**: All parameter modifications are done through YAML files without code changes. Whenever you change the task number, robot number, type partition, capability vectors, or map layout, you need to re-run **Steps 1 and 2** of the Usage Guide so the precomputed maps and distance matrices stay consistent.


## Usage Guide

### Step 1: Generate Adaptive Halton Map

```bash
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/Adaptive_Halton.py
```

### Step 2: Compute the Obstacle-Aware Task Distance Matrix

```bash
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/multi_source_dijkstra.py
```

This produces `multi_source_dijkstra_distances.csv` used by the clusterer and the MILP solver.

### Step 3: Launch the Multi-Agent Task Assignment and Planning System

```bash
# Ensure the ROS2 environment is properly configured
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch the full pipeline (planner + benchmark + showmove visualizer)
ros2 launch ltl_automaton_planner ltl_planner_isaac_launch.py
```

The visualizer window now includes a **legend panel** on the right that shows:

- the marker / color used for each declared task type, plus the number of tasks it currently owns,
- each robot’s id (drawn as a colored circle with a number on top), the list of types it can complete (`T1, T3, T5`, …), and small chip markers in matching colors / shapes for quick recognition.

### Step 4 (optional): LLM-Guided Online Interaction

If you also want to control the running mission with natural language, start the two LLM nodes (see the dedicated **LLM-Guided Interface** section below for full details):

```bash
# In separate terminals (after sourcing the workspace)
ros2 run ltl_automaton_planner human_llm_chat_node
ros2 run ltl_automaton_planner llm_command_parser_node
```


## LLM-Guided Interface

The framework includes an end-to-end natural-language interface that lets a human operator change the mission **online**, while the robots are still executing tasks. No restart, no manual yaml editing — just type in plain English.

### How it Works

```
┌──────────────────────┐       ┌──────────────────────────┐       ┌──────────────────────┐
│  human_llm_chat_node │  ───► │ llm_command_parser_node  │  ───► │  Planner / Assigner  │
│   (Tk GUI + GPT-4)   │  free │  (GPT-4o-mini → ROS msg) │ ROS   │  (online replanning) │
│                      │  text │                          │ topic │                      │
└──────────────────────┘       └──────────────────────────┘       └──────────────────────┘
```

1. **`human_llm_chat_node`** — a small Tk chat window. Talks to GPT-4 with a system prompt that is **dynamically populated from `Task_Points.yaml`** (it learns how many task types exist, the example labels of each type, and each robot’s capability vector). It collects missing parameters through follow-up questions, then sends a **single completed instruction string** to the topic `/human_instruction`.
2. **`llm_command_parser_node`** — a deterministic parser using GPT-4o-mini. It listens on `/human_instruction`, converts the instruction to structured JSON, and publishes the corresponding ROS2 message:
   - `AddTask` → `/add_task` and `/<robot>/add_task`
   - `ObstacleUpdate` → `/obstacle_update` and `/<robot>/obstacle_update`
   - `ChangeTaskPriority` → `/change_task_priority` and `/<robot>/change_task_priority`
3. **The downstream nodes** (`taskassign_cluster_node`, `planner_cluster_node`, `benchmark_cluster_node`, `showmove_node`) all subscribe to those topics and re-cluster, replan, or update the visualization in place.

### Setup

1. Export your OpenAI key (the chat node uses `gpt-4o`, the parser uses `gpt-4o-mini`):

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

2. Make sure the main pipeline (Step 3) is already running, so the topics exist.

3. Start the two LLM nodes in two separate terminals:

   ```bash
   # Terminal A — chat GUI
   ros2 run ltl_automaton_planner human_llm_chat_node

   # Terminal B — parser
   ros2 run ltl_automaton_planner llm_command_parser_node
   ```

   The chat GUI window will pop up. Type your instruction and press *Enter*.

### Supported Intents

| Intent                  | What it does                                            | Example utterance                                                                  |
| ----------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `add_task`              | Spawn a new pickup task at a given location.            | *“Add a type-2 task at (5, 19) called `fb`, deliver to room c.”*                   |
| `obstacle_update`       | Inject a new wall / blocking polygon into the map.      | *“There is a wall blocking from (7, 5) to (8, 5).”*                                |
| `change_task_priority`  | Mark a task as `high`, `normal`, or `low` priority.     | *“Bump the priority of `fb` to high.”* / *“Make `cd` low priority.”*               |

For `add_task`, the LLM understands the **multi-type heterogeneous setup** — task type ids (`"1"`, `"2"`, …) are inferred from the user’s wording (e.g. *“the red ones”*, *“type 3”*, *“delivery task”*) using the type/capability information injected from `Task_Points.yaml`. The capability filter in `taskassign_cluster_node` then guarantees that only robots that can actually perform the type will bid on it.

### Conversational Behaviour

You don't need to give all the fields in one go. The chat node is designed to **ask follow-up questions** when something is missing. Example session:

```
You:        Add a new task somewhere around 5,19.
Assistant:  Got it, you want to add a new task. To complete this command,
            I still need:
              • task_label: a two-letter code for the task (e.g. 'fb')
              • delivery_point: which room? choose 'b', 'c', 'd', or 'e'
              • task_type (optional): integer id of the type, default '1'
You:        label fb, delivery c, type 2
Assistant:  ✓ Command sent successfully!
            Instruction: Add a task at location [5, 19] with label 'fb',
                         delivery point 'c', task type '2'.
```

The window also has a **Reset** button that clears the conversation history, in case you want to start a new command from scratch.

### Visual Feedback in the Pygame Window

Once an instruction is published, the visualizer reflects it immediately:

- **`add_task`** — a new marker (with the colour / shape of the announced task type) appears on the grid; the task type counter in the right-hand legend goes up.
- **`obstacle_update`** — the new wall is drawn in **magenta** so you can distinguish it from the pre-known black walls and the unknown red blocks.
- **`change_task_priority`** — high-priority tasks get a **blinking red ring** until they are completed.

### Tips & Troubleshooting

- The chat node prints a `[TIMING]` line for every step (UI input → LLM reply → instruction sent → system response). These are also published on `/timing_data`, which `showmove_node` aggregates into a final summary table at the end of the run.
- If the LLM produces malformed JSON the parser logs it and asks you to rephrase; the underlying ROS pipeline never sees a half-formed message.
- To extend the interface with a new intent, edit the `INTENT_PARAMS` table in `human_llm_chat_node.py`, the system prompt in `analyze_with_llm`, and the dispatch in `llm_command_parser_node.py`.
- Costs: the chat node is the more expensive of the two (GPT-4o). If you only want machine-friendly commands, you can talk directly to the parser by publishing to `/human_instruction` from the command line:

  ```bash
  ros2 topic pub --once /human_instruction std_msgs/String \
    "{data: 'Add a task at location [5,19] with label fb, delivery point c, task type 2'}"
  ```


## Project Structure

```
ros2_ws/
├── build/                    # colcon build output
├── install/                  # colcon install output
├── log/                      # colcon log output
└── src/                      # Source code
    ├── interfaces_hmm_sim/   # Lower-level simulator interface package
    └── lmco/                 # Main framework package
        ├── ltl_automaton_msgs/        # Custom message definitions
        └── ltl_automaton_planner/
            ├── config/
            │   ├── Task_Points.yaml        # Tasks, types, capabilities, robot start poses
            │   ├── wall.yaml               # Obstacles / walls
            │   └── ...                     # Variants for benchmarking (different task counts)
            ├── launch/
            │   └── ltl_planner_isaac_launch.py
            └── ltl_automaton_planner/
                ├── Adaptive_Halton.py            # Step 1: roadmap generation
                ├── multi_source_dijkstra.py      # Step 2: distance matrix
                ├── CostMapClusterer.py           # Obstacle-aware task clustering
                ├── MILP.py                       # Intra-cluster MILP task ordering
                ├── ltl_automaton_utilities.py
                ├── Gen_LTL.py
                ├── ltl_tools/                    # LTL/Buchi/product automaton helpers
                ├── Experiments/                  # Offline scripts (benchmarks, plots)
                └── nodes/
                    ├── taskassign_cluster_node.py    # Heterogeneous task assigner (CWA + ψ/ζ)
                    ├── planner_cluster_node.py       # Per-robot LTL planner with replanning
                    ├── benchmark_cluster_node.py     # Simulated execution / metrics
                    ├── showmove_node.py              # Pygame visualizer + legend
                    ├── llm_command_parser_node.py    # Natural language → structured ROS msgs
                    └── human_llm_chat_node.py        # Tk chat GUI for human-LLM interaction
```


## Citation

If you use **LLM-OATH** in academic work, please cite the project page:

> *OATH: Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming*, project website <https://llm-oath.github.io/>.


## Contact

- **Maintainer**: <nan.li@gatech.edu>

For questions or issues, please open an issue or contact the maintainer (<nan.li@gatech.edu>).
