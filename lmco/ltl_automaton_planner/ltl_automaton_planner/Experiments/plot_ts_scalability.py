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
    
    # Don't set x-axis ticks here - will be set in create_combined_figure
    # Only set y-axis label and styling
    ax.set_ylabel(ylabel, fontsize=16, color=color)
    ax.tick_params(axis='y', labelsize=16, labelcolor=color)
    ax.tick_params(axis='x', labelsize=16)
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return bp


def fit_and_plot_trend(ax, sampling_points, data, color, linestyle='--', linewidth=2):
    """Fit a polynomial trend to median values and plot as dashed line"""
    # Calculate median for each sampling point
    medians = [np.median(values) for values in data]
    
    # Fit polynomial (degree 2 for smooth curve)
    if len(sampling_points) >= 3:
        degree = min(2, len(sampling_points) - 1)
    else:
        degree = 1
    
    coeffs = np.polyfit(sampling_points, medians, degree)
    poly = np.poly1d(coeffs)
    
    # Generate smooth x values for plotting
    x_smooth = np.linspace(sampling_points[0], sampling_points[-1], 200)
    y_smooth = poly(x_smooth)
    
    # Plot fitted line
    ax.plot(x_smooth, y_smooth, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8, label='Trend')


def create_combined_figure(all_results, output_path):
    """Create a single figure with dual y-axes: product build time (left) and optimal run time (right)"""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Extract data for both metrics
    sampling_points_product, data_product = extract_data_for_boxplot(all_results, 'product_build_time')
    sampling_points_optimal, data_optimal = extract_data_for_boxplot(all_results, 'optimal_run_time')
    
    # Plot product build time on left y-axis
    bp1 = create_boxplot(ax1, sampling_points_product, data_product, 
                        'Product Automaton Build Time (s)', 'steelblue', show_xlabel=True)
    
    # Fit and plot trend for product build time
    fit_and_plot_trend(ax1, sampling_points_product, data_product, 'steelblue')
    
    # Create second y-axis for optimal run time
    ax2 = ax1.twinx()
    
    # Plot optimal run time on right y-axis
    bp2 = create_boxplot(ax2, sampling_points_optimal, data_optimal, 
                        'Optimal Run Time (s)', 'forestgreen', show_xlabel=False)
    
    # Fit and plot trend for optimal run time
    fit_and_plot_trend(ax2, sampling_points_optimal, data_optimal, 'forestgreen')
    
    # Set x-axis limits to match the data range
    all_sampling_points = sorted(set(sampling_points_product + sampling_points_optimal))
    x_range = all_sampling_points[-1] - all_sampling_points[0]
    x_min = all_sampling_points[0] - x_range * 0.05
    x_max = all_sampling_points[-1] + x_range * 0.05
    
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Set x-axis ticks to match the data positions (let matplotlib auto-format the labels)
    ax1.set_xticks(all_sampling_points)
    ax1.set_xlabel('Number of Sampling Points', fontsize=16)
    
    # Set custom y-axis ranges
    ax1.set_ylim(0, 65)  # Product Build Time: 0 to 70
    ax2.set_ylim(0, 5)   # Optimal Run Time: 0 to 5
    
    # # Add legend
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
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
    default_results_dir = os.path.join(script_dir, '..', '..', 'config', 'evaluation_results')
    
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
