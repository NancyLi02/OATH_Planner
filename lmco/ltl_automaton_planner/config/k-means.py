import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString
from sklearn.cluster import KMeans

# ----------- Parameters -----------
WALL_WIDTH = 0.3
MAP_BOUNDS = (0, 0, 20, 20)
kmeans_clusters = 7

# ----------- Task Points -----------
points_with_label = {
    (1, 4): '', (6, 6): '',
    (1, 6.5): 'b', (5.5, 9.5): 'c', (9, 6.5): 'd',
    (1, 13.5): 'f', (9, 16.5): 'g', (1, 16): '', (6, 13): '',
    (11, 13.5): 'i', (11, 16.5): 'j', (19, 16.5): 'k',
    (19, 19): '', (15, 19.5): '', (11, 4): 'm',
    (19, 6.5): 'n', (16, 3): '', (19, 1): ''
}
task_points = list(points_with_label.keys())
task_coords = np.array(task_points)

# ----------- Wall Definitions -----------
lines = [
    LineString([(0, 3), (2, 3), (2, 4)]), LineString([(0, 5), (2, 5)]),
    LineString([(0, 7), (2, 7), (2, 6)]), LineString([(4, 9), (6, 9), (6, 10)]),
    LineString([(6, 7), (4, 7), (4, 5)]), LineString([(5, 5), (7, 5), (7, 7)]),
    LineString([(8, 5), (8, 7), (10, 7)]), LineString([(4, 2), (4, 4), (5, 4)]),
    LineString([(6, 4), (7, 4), (7, 2), (5, 2)]), LineString([(10, 3), (12, 3), (12, 4)]),
    LineString([(10, 5), (12, 5)]), LineString([(10, 7), (12, 7), (12, 6)]),
    LineString([(14, 9), (16, 9), (16, 10)]), LineString([(16, 7), (14, 7), (14, 5)]),
    LineString([(15, 5), (17, 5), (17, 7)]), LineString([(18, 5), (18, 7), (20, 7)]),
    LineString([(14, 2), (14, 4), (15, 4)]), LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),
    LineString([(0, 13), (2, 13), (2, 14)]), LineString([(0, 15), (2, 15)]),
    LineString([(0, 17), (2, 17), (2, 16)]), LineString([(4, 19), (6, 19), (6, 20)]),
    LineString([(6, 17), (4, 17), (4, 15)]), LineString([(5, 15), (7, 15), (7, 17)]),
    LineString([(8, 15), (8, 17), (10, 17)]), LineString([(4, 12), (4, 14), (5, 14)]),
    LineString([(6, 14), (7, 14), (7, 12), (5, 12)]), LineString([(10, 13), (12, 13), (12, 14)]),
    LineString([(10, 15), (12, 15)]), LineString([(10, 17), (12, 17), (12, 16)]),
    LineString([(14, 19), (16, 19), (16, 20)]), LineString([(16, 17), (14, 17), (14, 15)]),
    LineString([(15, 15), (17, 15), (17, 17)]), LineString([(18, 15), (18, 17), (20, 17)]),
    LineString([(14, 12), (14, 14), (15, 14)]), LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
    LineString([(0, 10), (6, 10)]), LineString([(10, 0), (10, 7)]),
    LineString([(14, 10), (20, 10)]), LineString([(10, 13), (10, 20)])
]

# ----------- Run KMeans Clustering -----------
kmeans = KMeans(n_clusters=kmeans_clusters, random_state=0, n_init=10)
kmeans_labels = kmeans.fit_predict(task_coords)
kmeans_centers = kmeans.cluster_centers_

# ----------- Plotting -----------
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
cmap = plt.get_cmap("tab10")

for i in range(kmeans_clusters):
    cluster_indices = np.where(kmeans_labels == i)[0]
    cluster_coords = task_coords[cluster_indices]
    color = cmap(i % 10)
    plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=100, c=[color],
                edgecolors='white', linewidths=1)
    center = kmeans_centers[i]
    radius = np.max(np.linalg.norm(cluster_coords - center, axis=1)) + 1.0
    circle = patches.Circle(center, radius, facecolor=color, edgecolor='black', alpha=0.3)
    ax.add_patch(circle)

# Draw walls
for line in lines:
    poly = line.buffer(WALL_WIDTH / 2, cap_style=2, join_style=2)
    x, y = poly.exterior.xy
    plt.fill(x, y, color='gray', alpha=0.8, zorder=0)

# Draw map border
x_min, y_min, x_max, y_max = MAP_BOUNDS
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], color='black', linewidth=2)

plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title(f"Vanilla KMeans Clustering ({kmeans_clusters} Clusters)", fontsize=14)
plt.xticks([]); plt.yticks([])
plt.axis('equal')
plt.tight_layout()
plt.show()
