import os
import pandas as pd
from MILP import ClusterTaskPlanner

def main():
    # 使用实际的测试数据
    robot_start = (1, 19)
    pickup_points = [(1.0, 13.5), (9.0, 16.5), (1.0, 16.0), (6.0, 13.0)]
    pickup_labels = ['bc', 'cc', 'dc', 'ec']
    delivery_points = [(15, 15)]
    delivery_labels = ['c']
    pickup_to_delivery = {'bc': 'c', 'cc': 'c', 'dc': 'c', 'ec': 'c'}
    robot_capacity = 3  # 设置容量为4，因为只有4个pickup点

    # 使用正确的距离文件路径
    dijkstra_path = "multi_source_dijkstra_distances.csv"
    
    # 检查文件是否存在
    if not os.path.exists(dijkstra_path):
        print(f"错误：距离文件 {dijkstra_path} 不存在")
        return
    
    print(f"使用距离文件: {dijkstra_path}")
    print(f"Pickup labels: {pickup_labels}")
    print(f"Delivery labels: {delivery_labels}")

    # 实例化 planner 并运行
    planner = ClusterTaskPlanner(dijkstra_distances_csv_path=dijkstra_path)
    
    # 检查距离数据是否正确加载
    print(f"加载的距离数据数量: {len(planner.dijkstra_distances)}")
    
    # 检查测试标签的距离是否存在
    test_labels = pickup_labels + delivery_labels
    for label in test_labels:
        found_distances = 0
        for other_label in test_labels:
            if (label, other_label) in planner.dijkstra_distances:
                found_distances += 1
        print(f"标签 {label} 的距离数据: {found_distances} 个")
    
    # 显示一些具体的距离值
    print("一些具体的距离值:")
    for pickup in pickup_labels:
        for delivery in delivery_labels:
            key = (pickup, delivery)
            if key in planner.dijkstra_distances:
                print(f"  {pickup} -> {delivery}: {planner.dijkstra_distances[key]}")
    
    chosen_pickups, chosen_deliveries, route, total_cost = planner.plan_cluster_tasks(
        robot_start, pickup_points, pickup_labels,
        delivery_points, delivery_labels,
        pickup_to_delivery, robot_capacity
    )
    print("==== TEST RESULT ====")
    print("Chosen pickups:", chosen_pickups)
    print("Chosen deliveries:", chosen_deliveries)
    print("Optimal route:", route)
    print("Total cost:", total_cost)
    print("=====================")

if __name__ == '__main__':
    main()
