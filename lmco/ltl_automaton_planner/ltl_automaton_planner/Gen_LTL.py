def generate_ltl_formula(route_with_states):
    """
    Generate a nested LTL formula from a route that includes load/unload states.
    Args:
        route_with_states (List[str]): Sequence of nodes with states, e.g. ['bb load', 'b unload', 'cb load', 'b unload']
    Returns:
        str: Nested LTL formula as a string.
    """
    # Load delivery points from Task_Points.yaml
    import os
    import yaml
    from ament_index_python.packages import get_package_share_directory

    package_share = get_package_share_directory('ltl_automaton_planner')
    task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
    
    with open(task_points_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    delivery_points = yaml_data.get('delivery_points', {})
    task_to_delivery = yaml_data.get('task_to_delivery', {})

    # 解析route中的节点和状态
    parsed_route = []
    for item in route_with_states:
        if ' load' in item:
            node = item.replace(' load', '')
            state = 'loaded'
        elif ' unload' in item:
            node = item.replace(' unload', '')
            state = 'unloaded'
        else:
            # 如果没有明确的状态，保持原样
            node = item
            state = None
        parsed_route.append((node, state))
    
    # 为loaded状态分配编号
    loaded_count = 1
    state_mapping = {}
    
    for node, state in parsed_route:
        if state == 'loaded':
            state_mapping[(node, state)] = f'loaded{loaded_count}'
            loaded_count += 1
        elif state == 'unloaded':
            # Check if it's a delivery point
            if node in delivery_points or any(node in delivery_list for delivery_list in task_to_delivery.values()):
                state_mapping[(node, state)] = 'delivery_unloaded'
            else:
                state_mapping[(node, state)] = 'unloaded'
    
    formula = ""
    
    # 按route正序嵌套（从后往前构建）
    for idx, (node, state) in enumerate(reversed(parsed_route)):
        real_idx = len(parsed_route) - 1 - idx
        
        if state is None:
            # 没有明确状态的情况，只检查位置
            part = f"({node})"
        else:
            # 有明确状态的情况
            mapped_state = state_mapping.get((node, state), state)
            
            # 构建状态条件
            if state == 'loaded':
                # 对于loaded状态，需要确保不是其他loaded状态
                other_loaded_conditions = []
                for other_node, other_state in parsed_route:
                    if other_state == 'loaded' and other_node != node:
                        other_loaded_idx = list(state_mapping.keys()).index((other_node, other_state)) + 1
                        other_loaded_conditions.append(f"!loaded{other_loaded_idx}")
                
                state_condition = mapped_state
                if other_loaded_conditions:
                    state_condition += " && " + " && ".join(other_loaded_conditions)
                
                part = f"({node} && {state_condition})"
            else:  # unloaded or delivery_unloaded
                part = f"({node} && {mapped_state})"
        
        if formula:
            part += f" && ({formula})"
        formula = f"<>({part})"
    
    return formula

# Example usage:
# 示例1: 包含load和unload的路线
route_with_states = ['bb load', 'b unload', 'cb load', 'b unload']
print("示例1 - 包含load/unload的路线:")
print(f"输入: {route_with_states}")
print(f"LTL公式: {generate_ltl_formula(route_with_states)}")
print()

# 示例2: 先去多个地点再load的情况
route_with_states2 = ['aa', 'bb', 'cc load', 'd unload', 'ee load', 'f unload']
print("示例2 - 先去多个地点再load:")
print(f"输入: {route_with_states2}")
print(f"LTL公式: {generate_ltl_formula(route_with_states2)}")
print()

# 示例3: 只有位置没有状态的情况
route_with_states3 = ['aa', 'bb', 'cc', 'dd']
print("示例3 - 只有位置没有状态:")
print(f"输入: {route_with_states3}")
print(f"LTL公式: {generate_ltl_formula(route_with_states3)}")
