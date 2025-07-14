# LLM-OATH: Obstacle-Aware Multi-Agent Task Assignment and Planning

## Overview

This repository implements **LLM-OATH**, a framework for obstacle-aware multi-agent task assignment and planning in large, dynamic environments. LLM-OATH enables scalable and realistic task allocation for heterogeneous robot teams by:

- Generating adaptive Halton maps that increase sampling density near obstacles.
- Computing obstacle-aware task-to-task distances using a multi-source Dijkstra algorithm.
- Assigning tasks via a cluster-auction architecture for improved scalability.
- Integrating LTL-D* for dynamic, obstacle-aware path planning.
- Leveraging large language models (LLMs) throughout the process to translate human instructions into formal specifications and adapt to changing intent in real time.

All robot and task parameters—including robot count, initial positions, task points, and special robot designations—are managed in a single YAML file (`Task_Points.yaml`), making the system easy to configure and extend.

---

## Installation

This project is based on **ROS2 Humble**. Please ensure you have installed ROS2 Humble before proceeding.

- Official ROS2 workspace creation tutorial: [Creating a ROS2 Workspace (Humble)](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)

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

For ROS2 dependencies, please refer to the [LTL-D* Planner](https://github.gatech.edu/jren313/lmco/tree/ros2) installation guide.

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

For questions or issues, please open an issue or contact the maintainer (nan.li@gatech.edu).
