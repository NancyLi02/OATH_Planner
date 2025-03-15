import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import scale


map_size = (8, 8)  
grid_size = (15, 15)  

x_centers = np.linspace(0, map_size[0], grid_size[0], endpoint=False) + (map_size[0] / grid_size[0]) / 2
y_centers = np.linspace(0, map_size[1], grid_size[1], endpoint=False) + (map_size[1] / grid_size[1]) / 2
X, Y = np.meshgrid(x_centers, y_centers)
points = np.vstack([X.ravel(), Y.ravel()]).T 


obstacles = []
lines = [
    LineString([(0, 2), (1, 2), (1, 3)]),
    LineString([(0, 4), (1, 4)]),
    LineString([(0, 6), (1, 6), (1, 5)]),
    LineString([(0, 7), (2, 7)]),
    LineString([(3, 7), (5, 7), (5, 8)]),
    LineString([(5, 6), (3, 6), (3, 4)]),
    LineString([(4, 4), (6, 4), (6, 6)]),
    LineString([(7, 3), (7, 5), (8, 5)]),
    LineString([(7, 0), (7, 2)]),
    LineString([(3, 1), (3, 3), (4, 3)]),
    LineString([(5, 3), (6, 3), (6, 1), (4, 1)])
]


for line in lines:
    buffered = line.buffer(distance=0.1, cap_style=3) 
    obstacles.append(buffered)


fig, ax = plt.subplots(figsize=(6, 6))


for i in range(map_size[0] + 1):
    ax.plot([i, i], [0, map_size[1]], 'k-', linewidth=1) 
for j in range(map_size[1] + 1):
    ax.plot([0, map_size[0]], [j, j], 'k-', linewidth=1)  


for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, 'red', alpha=0.7) 


ax.scatter(points[:, 0], points[:, 1], color='blue', s=10)


ax.set_xticks(range(map_size[0] + 1))
ax.set_yticks(range(map_size[1] + 1))
ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])
ax.set_aspect('equal') 
ax.set_title("8x8 Grid with 256 Grid Centers and Obstacles")
ax.legend()

plt.show()
