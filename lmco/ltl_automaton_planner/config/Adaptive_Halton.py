import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import yaml

# Global parameter definitions
d_min = 0.3       # Minimum allowed distance to avoid points being too close to obstacles
d_opt = 0.4       # Optimal distance (highest sampling probability)
sigma = 0.5       # Controls the width of the probability distribution
floor_prob = 0.2  # Minimum sampling probability in open areas
wall_thick = 0.1  # Thickness of the walls

# Halton sequence generation function
def halton_sequence(size, base=2):
    sequence = []
    for i in range(1, size+1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        sequence.append(r)
    return np.array(sequence)

points_with_label = {
    (1, 19): 'a',
    (11, 19): '' ,
    (9, 11): '' ,
    (11, 9): '' ,
    (17, 14.5): '',
    (1, 4): 'bb', 
    (6, 6): 'cb',
    (1, 6.5): 'db', 
    (5.5, 9.5): 'eb',
    (9, 6.5): 'fb',
    (6, 3): 'b', # unload
    (1, 13.5): 'bc',
    (9, 16.5): 'cc',
    (1, 16): 'dc', 
    (6, 13): 'ec',
    (5, 16): 'c', # unload
    (11, 13.5): 'bd',
    (11, 16.5): 'cd',
    (19, 16.5): 'dd',
    (19, 19): 'ed', 
    (15, 19.5): 'fd',
    (16, 13): 'd', # unload
    (11, 4): 'be',
    (19, 6.5): 'ce',
    (16, 3): 'de', 
    (19, 1): 'ee',
    (16, 5.5): 'e' # unload
}

x_length, y_length = 20, 20

# ---------- Walls ----------
def load_lines_from_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    line_coords = data.get('lines', [])
    return [LineString(coords) for coords in line_coords]

filepath = "lmco/ltl_automaton_planner/config/wall.yaml"
lines = load_lines_from_yaml(filepath)

# # Definition of obstacles (constructed using LineString)
# lines = [
#     LineString([(0, 3), (2, 3), (2, 4)]),
#     LineString([(0, 5), (2, 5)]),
#     LineString([(0, 7), (2, 7), (2, 6)]),
#     LineString([(4, 9), (6, 9), (6, 10)]),
#     LineString([(6, 7), (4, 7), (4, 5)]),
#     LineString([(5, 5), (7, 5), (7, 7)]),
#     LineString([(8, 5), (8, 7), (10, 7)]),
#     LineString([(4, 2), (4, 4), (5, 4)]),
#     LineString([(6, 4), (7, 4), (7, 2), (5, 2)]),
#     LineString([(10, 3), (12, 3), (12, 4)]),
#     LineString([(10, 5), (12, 5)]),
#     LineString([(10, 7), (12, 7), (12, 6)]),
#     LineString([(14, 9), (16, 9), (16, 10)]),
#     LineString([(16, 7), (14, 7), (14, 5)]),
#     LineString([(15, 5), (17, 5), (17, 7)]),
#     LineString([(18, 5), (18, 7), (20, 7)]),
#     LineString([(14, 2), (14, 4), (15, 4)]),
#     LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),
#     LineString([(0, 13), (2, 13), (2, 14)]),
#     LineString([(0, 15), (2, 15)]),
#     LineString([(0, 17), (2, 17), (2, 16)]),
#     LineString([(4, 19), (6, 19), (6, 20)]),
#     LineString([(6, 17), (4, 17), (4, 15)]),
#     LineString([(5, 15), (7, 15), (7, 17)]),
#     LineString([(8, 15), (8, 17), (10, 17)]),
#     LineString([(4, 12), (4, 14), (5, 14)]),
#     LineString([(6, 14), (7, 14), (7, 12), (5, 12)]),
#     LineString([(10, 13), (12, 13), (12, 14)]),
#     LineString([(10, 15), (12, 15)]),
#     LineString([(10, 17), (12, 17), (12, 16)]),
#     LineString([(14, 19), (16, 19), (16, 20)]),
#     LineString([(16, 17), (14, 17), (14, 15)]),
#     LineString([(15, 15), (17, 15), (17, 17)]),
#     LineString([(18, 15), (18, 17), (20, 17)]),
#     LineString([(14, 12), (14, 14), (15, 14)]),
#     LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
#     LineString([(0, 10), (6, 10)]),
#     LineString([(10, 0), (10, 7)]),
#     LineString([(14, 10), (20, 10)]),
#     LineString([(10, 13), (10, 20)])
# ]

# Function to calculate the minimum distance from a point to a set of lines
def point_to_lines_distance(point, lines):
    return min(line.distance(point) for line in lines)

# Probability density function based on distance
def density_probability(d, d_min, d_opt, sigma, floor):
    if d < d_min:
        return 0
    return floor + (1 - floor) * np.exp(-((d - d_opt) ** 2) / (2 * sigma ** 2))

# Rejection sampling algorithm
def rejection_sampling(n_samples, lines, area_size, d_min, d_opt, sigma, floor):
    samples = []
    multiplier = 10
    while len(samples) < n_samples:
        halton_x = halton_sequence(n_samples * multiplier, 2) * area_size
        halton_y = halton_sequence(n_samples * multiplier, 3) * area_size
        for x, y in zip(halton_x, halton_y):
            if len(samples) >= n_samples:
                break
            p = Point(x, y)
            d = point_to_lines_distance(p, lines)
            if d < d_min:
                continue
            prob = density_probability(d, d_min, d_opt, sigma, floor)
            if np.random.rand() < prob:
                samples.append(p)
        multiplier += 5
    return samples[:n_samples]

# Generate valid sampling points
valid_points = rejection_sampling(1000, lines, x_length, d_min, d_opt, sigma, floor_prob)

# Add labeled points to the valid points
for key in points_with_label.keys():
    valid_points.append(Point(key))

# Delaunay triangulation
tri = Delaunay([(p.x, p.y) for p in valid_points])
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        a, b = simplex[i], simplex[(i+1)%3]
        if a < b:
            line_edge = LineString([valid_points[a], valid_points[b]])
            if not any(line_edge.intersects(obs) for obs in [line.buffer(wall_thick, cap_style=3) for line in lines]):
                edges.add(line_edge)

# Visualization
obstacles = [line.buffer(wall_thick, cap_style=3) for line in lines]

fig, ax = plt.subplots(figsize=(8, 8))

for buffered in obstacles:
    x_buffered, y_buffered = buffered.exterior.xy
    ax.fill(x_buffered, y_buffered, alpha=0.6, color='red')

for edge in edges:
    ax.plot(*edge.xy, color='lightblue', linewidth=0.8)

ax.scatter([p.x for p in valid_points], [p.y for p in valid_points], s=5, color='blue')

ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)
ax.set_aspect('equal')
plt.title('Adaptive Halton Sequence Map')
plt.grid(True)
plt.show()
