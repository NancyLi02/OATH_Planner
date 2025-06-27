def generate_ltl_formula(route, pickup_labels, delivery_labels):
    """
    Generate a nested LTL formula where the state (loaded/unloaded/loaded1/loaded2/...) is inferred.
    Args:
        route (List[str]): Sequence of nodes, e.g. ['ec', 'bc', 'dc', 'c']
        pickup_labels (List[str]): Pickups, e.g. ['bc', 'cc', 'dc', 'ec']
        delivery_labels (List[str]): Deliveries, e.g. ['c']
    Returns:
        str: Nested LTL formula as a string.
    """
    # 先统计每个pickup在route中的顺序编号
    pickup_order = {}
    count = 1
    for node in route:
        if node in pickup_labels and node not in pickup_order:
            pickup_order[node] = f'loaded{count}'
            count += 1

    def get_state(node):
        if node in pickup_labels:
            return pickup_order[node]
        elif node in delivery_labels:
            return 'unloaded'
        else:
            raise ValueError(f"Node {node} not in pickup or delivery labels")

    formula = ""
    # 找到第一个pickup点
    first_pickup_idx = None
    for idx, node in enumerate(route):
        if node in pickup_labels:
            first_pickup_idx = idx
            break
    # 按route正序嵌套
    for idx, node in enumerate(reversed(route)):
        real_idx = len(route) - 1 - idx
        state = get_state(node)
        if real_idx == first_pickup_idx and node in pickup_labels:
            # 第一个pickup点，必须是loaded1，且不是loaded2和loaded3
            part = f"({node} && loaded1 && !loaded2 && !loaded3)"
        else:
            part = f"({node} && {state})"
        if formula:
            part += f" && ({formula})"
        formula = f"<>({part})"
    return formula

# Example usage:
route = ['ec', 'bc', 'dc', 'c']
pickup_labels = ['bc', 'cc', 'dc', 'ec']
delivery_labels = ['c']

print(generate_ltl_formula(route, pickup_labels, delivery_labels))
