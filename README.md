# LLM-OATH: Obstacle-Aware Multi-Agent Task Assignment and Planning

## Abstract

Multi-Agent Task Assignment and Planning (MATP) remains challenging in large-scale, obstacle-rich, and dynamic environments. As the number of robots and tasks increases, traditional methods often struggle with scalability, realistic spatial reasoning, and adaptability to failures. To address these issues, we propose **LLM-OATH**: a novel framework for LLM-guided Obstacle-Aware Task Assignment and Planning from Human Instruction, designed specifically for heterogeneous robot teams.

LLM-OATH combines two strategies to enable obstacle-aware task allocation. First, it constructs an adaptive Halton sequence map, which adjusts sample point density based on obstacle distribution. Second, it applies a multi-source Dijkstra algorithm on the adaptive map to compute task-to-task distance matrices that explicitly incorporate obstacle information, enabling more accurate task allocation. For task assignment, we introduce a cluster-auction-task selection architecture, which improves scalability while maintaining optimality in task allocation. For path planning, we integrate LTL-D* to support dynamic planning. Unlike prior work that applies large language models (LLMs) only during task interpretation before execution, LLM-OATH integrates the LLM throughout the entire MATP process. During execution, it continuously translates human commands into formal specifications and routes them to the appropriate modules, enabling real-time responsiveness to dynamic human intent.

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/NancyLi02/multiagent_planner.git
cd multiagent_planner
```

### 2. Install Dependencies

This project requires ROS2, Python 3, and the following Python packages:
- numpy
- pandas
- matplotlib
- shapely
- scikit-learn
- scipy
- pyyaml

You can install the Python dependencies with:

```sh
pip install numpy pandas matplotlib shapely scikit-learn scipy pyyaml
```

For ROS2 dependencies, please refer to the [LTL-D* Planner](https://github.com/JimingLi/LTL-Dstar) installation guide.

---

## Parameter Configuration

**All robot parameters, task points, delivery points, and special robot settings are managed in:**

```
lmco/ltl_automaton_planner/config/Task_Points.yaml
```

You can flexibly adjust the number of robots, their initial positions, task points, delivery points, and special robots by editing this YAML file. No code modification is required for these changes.

---

## Usage Guide

### 1. Generate the Adaptive Halton Map

```sh
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/Adaptive_Halton.py
```

### 2. Compute Obstacle-Aware Task Distance Matrix

```sh
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/multi_source_dijkstra.py
```

### 3. (Optional) Visualize Task Clustering

```sh
python3 lmco/ltl_automaton_planner/ltl_automaton_planner/multi_source_cluster.py
```

### 4. Launch the Multi-Agent Task Assignment and Planning System

Make sure your ROS2 environment is properly sourced, then run:

```sh
ros2 launch ltl_automaton_planner ltl_planner_isaac_launch.py
```

---

## Notes

- **All robot, task, and delivery parameters are managed in `Task_Points.yaml`.**
- Always run `Adaptive_Halton.py` and `multi_source_dijkstra.py` first to generate the required map and distance matrix before launching the main system.
- `multi_source_cluster.py` is provided for visual verification of clustering results.
- The system is designed for scalability and easy adaptation to different robot team sizes and task sets.

For questions or issues, please open an issue or contact the maintainer.
