# multiagent_planner

### Multi-Agent Task Assignment and D* Path Planner

This repository includes an implementation of **CBAA Task Assignment** and **D\* Planning and Replanning** for multi-agent systems.

## Features
- Task assignment is using Consensus-based Auction Algorithm (CBAA).
- Task assignment using different scoring schemes, both works fine for now, but 'dstar' scheme need to be modified later.
- D* planning for dynamic obstacle avoidance.

## Configuration

### 1. Modifying the Environment (`isaac_block.yaml`)
By default, the configuration includes a commented-out wall, making **Room D** reachable. The relevant section in `isaac_block.yaml` is:

```yaml
  blocks:
  - - - 2
      - 1
    - - 2
      - 2
  # - - - 0
  #     - 4
  #   - - 1
  #     - 4
```

If you uncomment it, the path accessibility will change accordingly.

### 2. Changing the Scoring Scheme
To modify the task scoring scheme, update the `ltl_planner_issac_launch.py` file.

You can choose between:
- **D\* (`dstar`)**: Uses the D* algorithm for scoring.
- **Manhattan (`manhattan`)**: Uses Manhattan distance as the scoring metric.

Example in `ltl_planner_issac_launch.py`:
```python
score_scheme = 'dstar'  # dstar or manhattan
```

## Installation and Running the Code

### 1. Clone the Repository
```sh
git clone https://github.com/NancyLi02/multiagent_planner.git
```

### 2. Install Dependencies
Should be same as Jiming's LTL-D* Planner.

### 3. Run the Planner
```python
ros2 launch ltl_automaton_planner ltl_planner_isaac_launch.py
```
