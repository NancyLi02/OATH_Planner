import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, MultiPolygon

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions to finite polygons.
    Returns:
      regions: list of regions as indices of vertices
      vertices: array of vertex coordinates
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Function supports only 2D inputs.")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # map each point to its ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # reconstruct regions
    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        # region has infinite vertices
        ridges = all_ridges[p1]
        new_region = [v for v in region if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            # compute direction vector
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


# ————— Parameters —————
robot_positions = np.array([
    [1, 19],   # robot_1
    [11, 19],  # robot_2
    [9, 11],   # robot_3
    [11, 9],   # robot_4
])

xmin, xmax = 0, 20
ymin, ymax = 0, 20
boundary = box(xmin, ymin, xmax, ymax)

# ————— Generate Voronoi and make polygons finite —————
vor = Voronoi(robot_positions)
# pass a large radius so that the infinite regions extend beyond our 20×20 box
regions, vertices = voronoi_finite_polygons_2d(vor, radius=100)

# ————— Clip each region to the 20×20 box —————
clipped = []
for region in regions:
    poly = Polygon(vertices[region])
    poly_clip = poly.intersection(boundary)
    if poly_clip.is_empty:
        continue
    # handle both Polygon and MultiPolygon
    if isinstance(poly_clip, MultiPolygon):
        for sub in poly_clip.geoms:
            clipped.append(sub)
    else:
        clipped.append(poly_clip)

# ————— Visualization —————

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(6,6))
for poly in clipped:
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.4)

ax.plot(robot_positions[:,0], robot_positions[:,1],
        'o', color='red', markersize=8, label='Robots')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('4 Robots Voronoi Partition on 20×20 Map')
ax.legend()

plt.show()
