import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from networkx.classes.digraph import DiGraph
import math
import time
import re

def extract_two_integers(s):
    numbers = re.findall(r'\d+', s)  # Find all numbers in the string
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])  # Return the first two numbers as integers
    else:
        return None  # Return None if there are not enough numbers

print(extract_two_integers("from_12_to_35"))


def halton_sequence(size, base=2):
    sequence = []
    for i in range(1, size+1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i = i // base
        sequence.append(r)
    return np.array(sequence)

points_with_label = {(0.5, 0.3): 'A', 
                     (0.5, 1.7): 'F', 
                     (0.5, 2.7): 'D',
                     (2.5, 1.7): 'E',
                     (5.5, 0.3): 'B',
                     (5.5, 2.7): 'C'}

x_length = 6
y_length = 3

n_points = 120
x = halton_sequence(n_points, 2) * 6
y = halton_sequence(n_points, 3) * 3
points = np.vstack((x, y)).T

# 2. Filter points (pseudo-code)
obstacles = []  # List of Shapely polygons
lines = [LineString([(0, 1), (1, 1)]),
        LineString([(0, 2), (1, 2), (1, 1.5)]),
        LineString([(1, 0), (1, 0.3)]),
        LineString([(1, 2.5), (1, 3)]),
        LineString([(2, 1), (2, 2), (3, 2)]),
        LineString([(3, 1), (4, 1), (4, 2)]),
        LineString([(5, 0), (5, 1)]),
        LineString([(5, 2), (5, 3)])]
for line in lines:
    buffered = line.buffer(distance=0.1, cap_style=3)
    obstacles.append(buffered)
    
valid_points = [Point(p) for p in points if not any(poly.contains(Point(p)) for poly in obstacles)]
nodes = dict()
actions = dict()
index = 0

for p in valid_points:
    cell_key = f'{index}'
    position = [p.x, p.y]
    nodes[cell_key] = {
        'attr': {
            'pose': tuple(position),
            'labels': [],
        },
        'connected_to': {f'{index}_to_{index}':'stay'}
    }
    index = index + 1
    
for key, value in points_with_label.items():
    cell_key = f'{index}'
    nodes[cell_key] = {
        'attr': {
            'pose': key,
            'labels': [points_with_label[key]],
        },
        'connected_to': {f'{index}_to_{index}':'stay'}
    }
    valid_points.append(Point(key))
    index = index + 1

print(valid_points)
print("checkpoint A")

tri = Delaunay([(p.x, p.y) for p in valid_points])
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i+1)%3]
        # print("a: ", a)
        if a < b:  # Avoid duplicates
            line = LineString([valid_points[a], valid_points[b]])
            if not any(line.intersects(obstacle) for obstacle in obstacles):
                edges.add(line)
                nodes[f'{a}']["connected_to"][f'{b}'] = f'{a}_to_{b}'
                nodes[f'{b}']["connected_to"][f'{a}'] = f'{b}_to_{a}'
                actions[f'{a}_to_{b}'] = {
                    'guard': '1',
                    'type': 'move',
                    'weight':math.dist([valid_points[a].x, valid_points[a].y], 
                                       [valid_points[b].x, valid_points[b].y])
                }
                actions[f'{b}_to_{a}'] = {
                    'guard': '1',
                    'type': 'move',
                    'weight':math.dist([valid_points[a].x, valid_points[a].y], 
                                       [valid_points[b].x, valid_points[b].y])
                }

               
# def state_models_from_ts(valid_points, edges):
#     state_model = DiGraph(initial=set(), ts_state_format=["2d_pose_region"])
#     for node in valid_points:
#         # state_model.add_node(tuple([node]), label=set([str(node)]))
#         state_model.add_node(tuple([node]), label=set())
#     for node in state_model_dict['nodes']:
#         # Go through all connected node
#         for connected_node in state_model_dict['nodes'][node]['connected_to']:
#             # Add edge between node and connected node
#             # Get associated action from "connected_to" tag of state node
#             act = TS_dict['state_models'][model_dim]["nodes"][node]['connected_to'][connected_node]
#             # Use action to retrieve weight and guard from action dictionnary
#             act_guard = "1"
#             act_weight = 10
#             state_model.add_edge(tuple([node]), tuple([connected_node]), action = act, guard = act_guard, weight = act_weight)
        
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the buffered polygon (dilation)
for buffered in obstacles:
    x_buffered, y_buffered = buffered.exterior.xy
    ax.fill(x_buffered, y_buffered, alpha=0.3, color='blue')

# plot
for p in valid_points:
    plt.scatter(p.x, p.y, color='red', s=0.1)

# Plot the original line
for line in edges:
    x_line, y_line = line.xy
    ax.plot(x_line, y_line, 'r-', linewidth=2)

# Customize the plot
ax.set_title("Dilation with cap_style=3 (Square Cap)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.legend()
ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)
plt.show()
# 4. Graph ready for pathfinding (nodes: valid_points, edges: edges)

# from shapely.geometry import LineString, Polygon
# import matplotlib.pyplot as plt

# # Original line segment from (-1, 0) to (4, 0)
# line = LineString([(0, 0), (1, 1), (0, 2),(2,2), (3,1), (1,0)])

# # Apply dilation (buffer) with distance=1 and cap_style=3 (square)
# buffered = line.buffer(distance=0.2, cap_style=3)

# fig, ax = plt.subplots(figsize=(10, 4))

# # Plot the buffered polygon (dilation)
# x_buffered, y_buffered = buffered.exterior.xy
# ax.fill(x_buffered, y_buffered, alpha=0.3, color='blue', label='Dilated (cap_style=3)')

# # Plot the original line
# x_line, y_line = line.xy
# ax.plot(x_line, y_line, 'r-', linewidth=2, label='Original Line')

# # Customize the plot
# ax.set_title("Dilation with cap_style=3 (Square Cap)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.grid(True)
# ax.legend()
# ax.set_xlim(-2, 5)
# ax.set_ylim(-1.5, 4)
# plt.show()