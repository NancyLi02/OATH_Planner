#!/usr/bin/env python3
"""
Combined plot: Elbow Method (top) and Cluster Number Sensitivity (bottom)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# Part 1: Elbow Method Data (from find_optimal_clusters.py)
# =============================================================================

def load_distance_matrix(csv_path):
    """Load pairwise Dijkstra distances and create a full distance matrix"""
    df = pd.read_csv(csv_path)
    
    nodes = set(df['from'].unique()) | set(df['to'].unique())
    nodes = sorted(list(nodes))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    dist_matrix = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = node_to_idx[row['from']]
        j = node_to_idx[row['to']]
        dist_matrix[i, j] = row['distance']
        dist_matrix[j, i] = row['distance']
    
    return dist_matrix, nodes, node_to_idx

def kmedoids(dist_matrix, n_clusters, max_iter=300, random_state=42):
    """K-medoids (PAM) algorithm using precomputed distance matrix"""
    np.random.seed(random_state)
    n = dist_matrix.shape[0]
    
    medoids = np.random.choice(n, n_clusters, replace=False)
    
    for _ in range(max_iter):
        distances_to_medoids = dist_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        
        new_medoids = []
        for k in range(n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                new_medoids.append(medoids[k])
                continue
            
            min_cost = np.inf
            best_medoid = medoids[k]
            for idx in cluster_indices:
                cost = np.sum(dist_matrix[idx, cluster_indices])
                if cost < min_cost:
                    min_cost = cost
                    best_medoid = idx
            new_medoids.append(best_medoid)
        
        new_medoids = np.array(new_medoids)
        
        if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
            break
        medoids = new_medoids
    
    distances_to_medoids = dist_matrix[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    
    total_cost = 0
    for k in range(n_clusters):
        cluster_mask = labels == k
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 0:
            total_cost += np.sum(dist_matrix[cluster_indices, medoids[k]] ** 2)
    
    return medoids, labels, total_cost

def elbow_method_dijkstra(dist_matrix, nodes, max_clusters=10, n_init=10):
    """Use K-medoids with Dijkstra distance matrix for elbow method"""
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
    """Calculate the elbow point using the maximum curvature method"""
    x = np.array(k_range)
    y = np.array(wcss)
    
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return k_range[elbow_idx]

# =============================================================================
# Part 2: Sensitivity Data (from cluster_number_sensitivity.py)
# =============================================================================

sensitivity_data = {
    "Cluster Number": (
        [4]*5 +
        [5]*5 +
        [6]*5 +
        [7]*5 +
        [8]*5
    ),
    "Total Steps": (
        [636]*5 +
        [661]*5 +
        [675]*5 +
        [691]*5 +
        [691]*5
    ),
    "Total Time": [
        100.78, 99.38, 100.09, 99.02, 98.02,
        104.41, 101.91, 103.93, 102.54, 104.10,
        98.89, 97.34, 99.50, 100.50, 100.21,
        95.90, 95.30, 96.62, 97.31, 95.24,
        95.20, 94.84, 94.30, 96.34, 97.84
    ]
}

# =============================================================================
# Main: Combined Plot
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Load elbow method data ---
    csv_path = os.path.join(script_dir, 'multi_source_dijkstra_distances.csv')
    
    print("Loading Dijkstra distance matrix...")
    dist_matrix, nodes, node_to_idx = load_distance_matrix(csv_path)
    print(f"Found {len(nodes)} task nodes")
    
    print("\nRunning Elbow Method with Dijkstra Distances...")
    max_clusters = min(15, len(nodes) - 1)
    k_range, wcss = elbow_method_dijkstra(dist_matrix, nodes, max_clusters)
    elbow_k = calculate_elbow_point(k_range, wcss)
    print(f"\n*** Optimal number of clusters: {elbow_k} ***")
    
    # --- Prepare sensitivity data ---
    df_sens = pd.DataFrame(sensitivity_data)
    clusters = sorted(df_sens["Cluster Number"].unique())
    steps_per_cluster = df_sens.groupby("Cluster Number")["Total Steps"].first().loc[clusters]
    time_by_cluster = [
        df_sens[df_sens["Cluster Number"] == c]["Total Time"].values
        for c in clusters
    ]
    
    # --- Create combined figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # =====================
    # Top subplot: Elbow Method
    # =====================
    ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal K = {elbow_k}')
    
    ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=14)
    ax1.set_xticks(k_range)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # =====================
    # Bottom subplot: Sensitivity
    # =====================
    # Total Time boxplot
    ax2.boxplot(
        time_by_cluster,
        positions=clusters,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#1f77b4", edgecolor="black", alpha=0.7),
        medianprops=dict(color="#ff7f0e", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black")
    )
    
    ax2.set_ylabel("Total Time (s)", fontsize=14, color="#1f77b4")
    ax2.set_ylim(80, 120)
    ax2.tick_params(axis="y", labelsize=14, colors="#1f77b4")
    ax2.tick_params(axis="x", labelsize=14)
    
    # Total Steps line (secondary y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        clusters,
        steps_per_cluster.values,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Total Steps"
    )
    
    ax2_twin.set_ylabel("Total Steps", fontsize=14, color="#d62728")
    ax2_twin.set_ylim(600, 750)
    ax2_twin.tick_params(axis="y", labelsize=14, colors="#d62728")
    
    # X-axis label only on bottom subplot
    ax2.set_xlabel("Number of Clusters (K)", fontsize=14)
    ax2.set_xticks(clusters)
    
    plt.tight_layout()
    
    # Save as PDF
    save_path = os.path.join(script_dir, 'combined_cluster_analysis.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nCombined plot saved to: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
