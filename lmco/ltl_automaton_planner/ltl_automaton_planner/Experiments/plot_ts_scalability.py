#!/usr/bin/env python
"""
Plotting script for transition system scalability benchmark results.
Creates box plots showing the relationship between sampling points and computation time.

This script reads the JSON results from benchmark_ts_scalability.py and generates:
- Box plot for product automaton build time
- Box plot for optimal run time
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_latest_results(results_dir):
    """Load the most recent benchmark results file"""
    json_files = [f for f in os.listdir(results_dir) if f.startswith('ts_scalability_results_') and f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No benchmark results found in {results_dir}")
    
    # Sort by timestamp in filename
    json_files.sort(reverse=True)
    latest_file = os.path.join(results_dir, json_files[0])
    
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file


def load_results_from_file(filepath):
    """Load results from a specific file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_data_for_boxplot(all_results, metric_key):
    """Extract data for box plot from results"""
    sampling_points = sorted([int(k) for k in all_results['results'].keys()])
    data = []
    
    for n_points in sampling_points:
        values = [r[metric_key] for r in all_results['results'][str(n_points)] if metric_key in r]
        data.append(values)
    
    return sampling_points, data


def create_boxplot(ax, sampling_points, data, ylabel, color='steelblue', show_xlabel=True):
    """Create a single box plot without title and without outliers, with correct x-axis spacing"""
    # Calculate box width based on the minimum gap between sampling points
    min_gap = min(sampling_points[i+1] - sampling_points[i] for i in range(len(sampling_points)-1))
    box_width = min_gap * 0.6  # 60% of minimum gap
    
    # Use positions parameter to place boxes at actual x values
    bp = ax.boxplot(data, patch_artist=True, positions=sampling_points, 
                    widths=box_width, showfliers=False)
    
    # Customize box plot appearance
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5, linestyle='--')
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)
    
    # Set x-axis ticks at actual sampling point values
    ax.set_xticks(sampling_points)
    ax.set_xticklabels([str(n) for n in sampling_points])
    
    # Add some padding to x-axis limits
    x_range = sampling_points[-1] - sampling_points[0]
    ax.set_xlim(sampling_points[0] - x_range * 0.05, sampling_points[-1] + x_range * 0.05)
    
    if show_xlabel:
        ax.set_xlabel('Number of Sampling Points', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def create_combined_figure(all_results, output_path):
    """Create a combined figure with two subplots: product build time and optimal run time"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1 (top): Product Automaton Build Time - no x-axis label
    sampling_points, data = extract_data_for_boxplot(all_results, 'product_build_time')
    create_boxplot(axes[0], sampling_points, data, 'Time (seconds)', 'steelblue', show_xlabel=False)
    
    # Plot 2 (bottom): Optimal Run Time - with x-axis label
    sampling_points, data = extract_data_for_boxplot(all_results, 'optimal_run_time')
    create_boxplot(axes[1], sampling_points, data, 'Time (seconds)', 'forestgreen', show_xlabel=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def print_summary_table(all_results):
    """Print a summary table of results"""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    
    sampling_points = sorted([int(k) for k in all_results['results'].keys()])
    
    print(f"{'Points':<8} {'Nodes':<8} {'Edges':<10} {'Graph(s)':<12} {'Product(s)':<12} {'Optimal(s)':<12} {'Total(s)':<12}")
    print("-" * 100)
    
    for n_points in sampling_points:
        results = all_results['results'][str(n_points)]
        
        avg_nodes = np.mean([r['actual_nodes'] for r in results])
        avg_edges = np.mean([r['actual_edges'] for r in results])
        avg_graph = np.mean([r['graph_build_time'] for r in results])
        avg_product = np.mean([r['product_build_time'] for r in results])
        avg_optimal = np.mean([r['optimal_run_time'] for r in results])
        avg_total = np.mean([r['total_time'] for r in results])
        
        print(f"{n_points:<8} {avg_nodes:<8.0f} {avg_edges:<10.0f} {avg_graph:<12.3f} {avg_product:<12.3f} {avg_optimal:<12.3f} {avg_total:<12.3f}")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Plot transition system scalability benchmark results')
    parser.add_argument('--input', '-i', type=str, help='Path to specific results JSON file')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Determine results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, '..', 'config', 'evaluation_results')
    
    # Load results
    if args.input:
        all_results = load_results_from_file(args.input)
        results_file = args.input
    else:
        all_results, results_file = load_latest_results(default_results_dir)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp from config or create new one
    timestamp = all_results['config'].get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Print summary
    print_summary_table(all_results)
    
    # Create combined figure with two subplots
    print("\nGenerating plot...")
    
    combined_path = os.path.join(output_dir, f'ts_scalability_{timestamp}.pdf')
    create_combined_figure(all_results, combined_path)
    
    print(f"\nPlot saved to: {output_dir}")
    print(f"  - {os.path.basename(combined_path)}")


if __name__ == '__main__':
    main()
