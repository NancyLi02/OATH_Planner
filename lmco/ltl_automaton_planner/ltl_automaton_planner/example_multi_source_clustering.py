#!/usr/bin/env python3
"""
Example script: Demonstrates how to use the improved CostMapClusterer for multi-source clustering
"""

import os
import sys
from CostMapClusterer import CostMapClusterer

def main():
    # Define task points and their labels
    points_with_label = {
        (1, 4): 'bb', (6, 6): 'cb', (1, 6.5): 'db', (5.5, 9.5): 'eb', (9, 6.5): 'fb',
        (1, 13.5): 'bc', (9, 16.5): 'cc', (1, 16): 'dc', (6, 13): 'ec',
        (11.0, 13.5): 'bd', (11, 16.5): 'cd', (19, 16.5): 'dd', (19, 19): 'ed', (15, 19.5): 'fd',
        (11.0, 4.0): 'be', (19, 6.5): 'ce', (16, 3): 'de', (19, 1): 'ee'
    }
    
    # File paths
    current_dir = os.path.dirname(__file__)
    wall_yaml_path = os.path.join(current_dir, '..', 'config', 'wall.yaml')
    halton_points_csv = os.path.join(current_dir, 'all_points_in_Halton.csv')
    precomputed_distances_csv = os.path.join(current_dir, 'multi_source_dijkstra_distances.csv')
    
    print("=== Multi-Source Clustering Example ===")
    print("Number of task points:", len(points_with_label))
    
    # Choose whether to use precomputed distance matrix or real-time computation
    use_precomputed = True  # Set to True to use precomputed distance matrix, False for real-time computation
    
    if use_precomputed:
        print("Using precomputed distance matrix...")
        clusterer = CostMapClusterer(
            points_with_label=points_with_label,
            wall_yaml_path=wall_yaml_path,
            num_clusters=4,
            halton_points_csv=halton_points_csv,
            wall_thick=0.1,
            precomputed_distances_csv=precomputed_distances_csv
        )
    else:
        print("Using real-time distance matrix computation...")
        clusterer = CostMapClusterer(
            points_with_label=points_with_label,
            wall_yaml_path=wall_yaml_path,
            num_clusters=4,
            halton_points_csv=halton_points_csv,
            wall_thick=0.1
        )
    
    print("\nComputing distance matrix...")
    clusterer.compute_distance_matrix()
    print("Distance matrix shape:", clusterer.distance_matrix.shape)
    
    print("\nPerforming clustering...")
    cluster_centers, cluster_points = clusterer.cluster()
    
    print("\n=== Clustering Results ===")
    for i, (center, points) in enumerate(zip(cluster_centers, cluster_points)):
        print(f"  Cluster {i+1}: Center {center}, Contains {len(points)} task points")
        for point in points:
            for label, coord in points_with_label.items():
                if coord == point:
                    print(f"    - {label}: {point}")
                    break
    
    print("\nPlotting clustering results...")
    clusterer.plot_clusters()
    print("\nMulti-source clustering completed!")

if __name__ == "__main__":
    main() 