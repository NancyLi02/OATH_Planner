#!/usr/bin/env python3
"""
Find optimal number of clusters using the Elbow Method
Based on Dijkstra distances between task points (using K-medoids)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import yaml
import os

def load_distance_matrix(csv_path):
    """Load pairwise Dijkstra distances and create a full distance matrix"""
    df = pd.read_csv(csv_path)
    
    # Get all unique nodes
    nodes = set(df['from'].unique()) | set(df['to'].unique())
    nodes = sorted(list(nodes))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create distance matrix (symmetric)
    dist_matrix = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = node_to_idx[row['from']]
        j = node_to_idx[row['to']]
        dist_matrix[i, j] = row['distance']
        dist_matrix[j, i] = row['distance']  # Symmetric
    
    return dist_matrix, nodes, node_to_idx

def load_task_positions(yaml_path):
    """Load task positions from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    task_points = data.get('task_points', {})
    positions = {}
    for coord, name in task_points.items():
        x, y = map(float, coord.split(','))
        positions[name] = (x, y)
    
    return positions, data

def load_walls(yaml_path):
    """Load wall information from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('lines', [])

def load_halton_points(csv_path):
    """Load Halton sequence points"""
    df = pd.read_csv(csv_path)
    points = []
    labels = []
    for _, row in df.iterrows():
        points.append((row['x'], row['y']))
        labels.append(row['label'] if pd.notna(row['label']) else '')
    return points, labels, df

def kmedoids(dist_matrix, n_clusters, max_iter=300, random_state=42):
    """
    K-medoids (PAM) algorithm using precomputed distance matrix
    Returns: medoid indices, labels, total cost
    """
    np.random.seed(random_state)
    n = dist_matrix.shape[0]
    
    # Initialize medoids randomly
    medoids = np.random.choice(n, n_clusters, replace=False)
    
    for _ in range(max_iter):
        # Assign points to nearest medoid
        distances_to_medoids = dist_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        
        # Update medoids
        new_medoids = []
        for k in range(n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                new_medoids.append(medoids[k])
                continue
            
            # Find the point that minimizes total distance to other cluster members
            min_cost = np.inf
            best_medoid = medoids[k]
            for idx in cluster_indices:
                cost = np.sum(dist_matrix[idx, cluster_indices])
                if cost < min_cost:
                    min_cost = cost
                    best_medoid = idx
            new_medoids.append(best_medoid)
        
        new_medoids = np.array(new_medoids)
        
        # Check for convergence
        if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
            break
        medoids = new_medoids
    
    # Final assignment
    distances_to_medoids = dist_matrix[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    
    # Calculate total cost (WCSS using Dijkstra distances)
    total_cost = 0
    for k in range(n_clusters):
        cluster_mask = labels == k
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 0:
            # Sum of squared distances to medoid
            total_cost += np.sum(dist_matrix[cluster_indices, medoids[k]] ** 2)
    
    return medoids, labels, total_cost

def elbow_method_dijkstra(dist_matrix, nodes, max_clusters=10, n_init=10):
    """
    Use K-medoids with Dijkstra distance matrix for elbow method
    """
    wcss = []
    k_range = range(1, min(max_clusters + 1, len(nodes)))
    
    for k in k_range:
        best_cost = np.inf
        for seed in range(n_init):
            _, _, cost = kmedoids(dist_matrix, k, random_state=42 + seed)
            if cost < best_cost:
                best_cost = cost
        wcss.append(best_cost)
        print(f"  k={k}: WCSS (Dijkstra) = {best_cost:.2f}")
    
    return list(k_range), wcss

def calculate_elbow_point(k_range, wcss):
    """
    Calculate the elbow point using the maximum curvature method
    """
    x = np.array(k_range)
    y = np.array(wcss)
    
    # Create line from first to last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Calculate perpendicular distance from each point to the line
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return k_range[elbow_idx]

def plot_elbow(k_range, wcss, elbow_k, save_path=None):
    """Plot the elbow curve"""
    plt.figure(figsize=(10, 5))
    
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=elbow_k, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal k = {elbow_k}')
    
    plt.xlabel('Number of Clusters (k)', fontsize=16)
    plt.ylabel('Within-Cluster Sum of Squares', fontsize=16)
    # plt.title('Elbow Method for Optimal Cluster Number', fontsize=16)
    plt.xticks(k_range)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Elbow plot saved to: {save_path}")
    
    plt.show()

def plot_clusters_on_map(positions, nodes, labels, medoids, elbow_k, walls, 
                          halton_points=None, save_path=None):
    """Plot the clustering result on the actual map with walls"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, elbow_k))
    
    # Plot Halton points (light gray background)
    if halton_points is not None:
        halton_x = [p[0] for p in halton_points]
        halton_y = [p[1] for p in halton_points]
        ax.scatter(halton_x, halton_y, c='lightgray', s=5, alpha=0.3, zorder=1)
    
    # Plot walls
    for wall in walls:
        wall_x = [p[0] for p in wall]
        wall_y = [p[1] for p in wall]
        ax.plot(wall_x, wall_y, 'k-', linewidth=2, zorder=2)
    
    # Plot task points by cluster
    for cluster_id in range(elbow_k):
        cluster_nodes = [nodes[i] for i in range(len(nodes)) if labels[i] == cluster_id]
        cluster_x = [positions[node][0] for node in cluster_nodes if node in positions]
        cluster_y = [positions[node][1] for node in cluster_nodes if node in positions]
        
        ax.scatter(cluster_x, cluster_y, c=[colors[cluster_id]], s=200, 
                   alpha=0.8, edgecolors='black', linewidth=1.5,
                   label=f'Cluster {cluster_id + 1}', zorder=4)
        
        # Annotate task points
        for node in cluster_nodes:
            if node in positions:
                x, y = positions[node]
                ax.annotate(node, (x, y), fontsize=9, fontweight='bold',
                           ha='center', va='bottom', 
                           xytext=(0, 8), textcoords='offset points',
                           zorder=5)
    
    # Highlight medoids
    for k, medoid_idx in enumerate(medoids):
        medoid_node = nodes[medoid_idx]
        if medoid_node in positions:
            x, y = positions[medoid_node]
            ax.scatter(x, y, c=[colors[k]], s=400, marker='*', 
                      edgecolors='black', linewidth=2, zorder=6)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Task Clustering Result (k={elbow_k}) with Dijkstra Distances\n(Stars = Medoids)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cluster map saved to: {save_path}")
    
    plt.show()

def print_cluster_details(nodes, labels, medoids, elbow_k, dist_matrix):
    """Print detailed cluster information"""
    print("\n" + "="*60)
    print(f"CLUSTERING RESULT (k={elbow_k}) - Using Dijkstra Distances")
    print("="*60)
    
    for cluster_id in range(elbow_k):
        cluster_indices = [i for i in range(len(nodes)) if labels[i] == cluster_id]
        cluster_nodes = [nodes[i] for i in cluster_indices]
        medoid_node = nodes[medoids[cluster_id]]
        
        print(f"\nCluster {cluster_id + 1}:")
        print(f"  Medoid: {medoid_node}")
        print(f"  Members: {cluster_nodes}")
        print(f"  Size: {len(cluster_nodes)} tasks")
        
        # Print distances from medoid to each member
        print(f"  Dijkstra distances from medoid:")
        for idx in cluster_indices:
            node = nodes[idx]
            dist = dist_matrix[medoids[cluster_id], idx]
            print(f"    {medoid_node} -> {node}: {dist:.3f}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths
    csv_path = os.path.join(script_dir, 'multi_source_dijkstra_distances.csv')
    yaml_path = os.path.join(script_dir, '..', 'config', 'Task_Points.yaml')
    wall_path = os.path.join(script_dir, '..', 'config', 'wall.yaml')
    halton_path = os.path.join(script_dir, 'all_points_in_Halton.csv')
    
    print("Loading Dijkstra distance matrix...")
    dist_matrix, nodes, node_to_idx = load_distance_matrix(csv_path)
    print(f"Found {len(nodes)} task nodes: {nodes}")
    
    print("\nLoading task positions...")
    positions, yaml_data = load_task_positions(yaml_path)
    
    # Add positions from Halton CSV for nodes not in yaml
    print("Loading Halton points...")
    halton_points, halton_labels, halton_df = load_halton_points(halton_path)
    
    # Update positions from Halton CSV
    for _, row in halton_df.iterrows():
        label = row['label']
        if pd.notna(label) and label != '' and label not in positions:
            positions[label] = (row['x'], row['y'])
    
    print(f"Task positions loaded for: {list(positions.keys())}")
    
    print("\nLoading walls...")
    walls = load_walls(wall_path)
    print(f"Loaded {len(walls)} wall segments")
    
    print("\n" + "="*60)
    print("Running Elbow Method with Dijkstra Distances")
    print("="*60)
    
    max_clusters = min(15, len(nodes) - 1)
    k_range, wcss = elbow_method_dijkstra(dist_matrix, nodes, max_clusters)
    
    # Find optimal k
    elbow_k = calculate_elbow_point(k_range, wcss)
    print(f"\n*** Optimal number of clusters: {elbow_k} ***")
    
    # Print WCSS values
    print("\nWCSS values (Dijkstra distances) for each k:")
    for k, w in zip(k_range, wcss):
        marker = " <-- ELBOW" if k == elbow_k else ""
        print(f"  k={k:2d}: WCSS = {w:.2f}{marker}")
    
    # Perform final clustering with optimal k (multiple runs to get best)
    best_cost = np.inf
    best_medoids = None
    best_labels = None
    for seed in range(20):
        medoids, labels, cost = kmedoids(dist_matrix, elbow_k, random_state=42 + seed)
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids
            best_labels = labels
    
    # Print cluster details
    print_cluster_details(nodes, best_labels, best_medoids, elbow_k, dist_matrix)
    
    # Plot results
    elbow_plot_path = os.path.join(script_dir, 'elbow_method_dijkstra.png')
    cluster_plot_path = os.path.join(script_dir, 'cluster_visualization_dijkstra.png')
    
    plot_elbow(k_range, wcss, elbow_k, elbow_plot_path)
    # plot_clusters_on_map(positions, nodes, best_labels, best_medoids, elbow_k, 
    #                      walls, halton_points, cluster_plot_path)
    
    return elbow_k, nodes, best_labels, best_medoids

if __name__ == '__main__':
    optimal_k, nodes, labels, medoids = main()
