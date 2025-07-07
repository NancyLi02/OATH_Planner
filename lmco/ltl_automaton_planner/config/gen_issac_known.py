import yaml
import os

# 你的工作区绝对路径
base_dir = "/home/nanli/ros2_ws/src/lmco/ltl_automaton_planner/config"

task_points_path = os.path.join(base_dir, "Task_Points.yaml")
isaac_known_path = os.path.join(base_dir, "isaac_known.yaml")

# 1. 读取 Task_Points.yaml
with open(task_points_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# 2. 收集所有 label
all_labels = set()
for d in ['task_points']:
    for v in data.get(d, {}).values():
        all_labels.add(v.strip())

# 3. 收集所有 delivery label
delivery_labels = set()
for v in data.get('delivery_points', {}).values():
    delivery_labels.add(v.strip())

# 4. 生成 load.guard（所有 task_points label 用 || 连接）
load_guard = ' || '.join(sorted(all_labels))

# 5. 生成 unload.guard（所有 delivery label 用 || 连接）
unload_guard = ' || '.join(sorted(delivery_labels))

# 6. 生成 isaac_known.yaml 内容
isaac_yaml = f"""actions:
  load:
    guard: '{load_guard}'
    type: load
    weight: 0
  stay:
    guard: '1'
    synchronized_transition:
      guard: '1'
      type: synchronized_transition
      weight: 0
    type: stay
    weight: 0
  unload:
    guard: '{unload_guard}'
    type: unload
    weight: 0
state_dim:
- 2d_pose_region
- Drone_state
state_models:
  2d_pose_region:
    initial: '0'
    ts_type: 2d_pose_region
  Drone_state:
    initial: "unloaded"
    nodes:
      loaded1:
        attr:
          labels: ['loaded1']
        connected_to:
          unloaded: unload
          loaded1: load
          loaded2: load
      loaded2:
        attr:
          labels: ['loaded2']
        connected_to:
          unloaded: unload
          loaded2: load
          loaded3: load
      loaded3:
        attr:
          labels: ['loaded3']
        connected_to:
          unloaded: unload
          loaded3: load
      unloaded:
        attr:
          labels: ['unloaded']
        connected_to:
          loaded1: load
          unloaded: unload
    ts_type: Drone_state
"""

# 7. 写入 isaac_known.yaml
with open(isaac_known_path, 'w', encoding='utf-8') as f:
    f.write(isaac_yaml)

print("isaac_known.yaml 已自动生成！")