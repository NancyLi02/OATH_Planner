"""
Combined plot: Elbow Method (top) + Cluster Number Sensitivity (bottom).
Shared x-axis concept (Number of Clusters), label only on bottom subplot.
Publication-quality style.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from find_optimal_clusters import (
    load_distance_matrix,
    elbow_method_dijkstra,
    calculate_elbow_point,
)

# ── 科研绘图常量 ────────────────────────────────────────────────
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 15
LINE_WIDTH = 1.2
GRID_ALPHA = 0.35
SPINE_WIDTH = 1.0

COLOR_ELBOW_LINE = '#5C83B3'
COLOR_ELBOW_VLINE = '#D06A75'
COLOR_BOX = '#5C83B3'
COLOR_BOX_EDGE = '#3E6490'
COLOR_MEDIAN = '#B22222'
COLOR_STEPS_LINE = '#D18358'


def _apply_rc():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': FONT_TICK,
        'axes.labelsize': FONT_LABEL,
        'axes.titlesize': FONT_LABEL,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'axes.linewidth': SPINE_WIDTH,
        'axes.edgecolor': '0.15',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': LINE_WIDTH,
        'ytick.major.width': LINE_WIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ── 上图: Elbow Method ─────────────────────────────────────────

def compute_elbow_data():
    csv_path = os.path.join(script_dir, '..', 'multi_source_dijkstra_distances.csv')
    print('Loading Dijkstra distance matrix...')
    dist_matrix, nodes, _ = load_distance_matrix(csv_path)

    max_clusters = min(15, len(nodes) - 1)
    print('Running Elbow Method...')
    k_range, wcss = elbow_method_dijkstra(dist_matrix, nodes, max_clusters)
    elbow_k = calculate_elbow_point(k_range, wcss)
    print(f'Optimal k = {elbow_k}')
    return k_range, wcss, elbow_k


def plot_elbow(ax, k_range, wcss, elbow_k):
    ax.plot(k_range, wcss, color=COLOR_ELBOW_LINE, marker='o',
            linewidth=2, markersize=7, markeredgecolor='white',
            markeredgewidth=1.2, zorder=3)
    ax.axvline(x=elbow_k, color=COLOR_ELBOW_VLINE, linestyle='--',
               linewidth=2, label=f'Optimal k = {elbow_k}', zorder=2)

    ax.set_ylabel('Within-Cluster\nSum of Squares', fontsize=FONT_LABEL)
    ax.set_xticks(k_range)
    ax.tick_params(axis='x', labelsize=FONT_TICK)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=GRID_ALPHA, color='0.75', linewidth=0.8)
    ax.xaxis.grid(False)

    leg = ax.legend(loc='upper right', frameon=True, framealpha=1,
                    edgecolor='0.3', fancybox=False, fontsize=FONT_LEGEND)
    leg.get_frame().set_linewidth(0.8)


# ── 下图: Sensitivity ──────────────────────────────────────────

SENSITIVITY_DATA = {
    'Cluster Number': [4]*5 + [5]*5 + [6]*5 + [7]*5 + [8]*5,
    'Total Steps': [636]*5 + [661]*5 + [675]*5 + [691]*5 + [691]*5,
    'Total Time': [
        100.78, 99.38, 100.09, 99.02, 98.02,
        104.41, 101.91, 103.93, 102.54, 104.10,
        98.89, 97.34, 99.50, 100.50, 100.21,
        95.90, 95.30, 96.62, 97.31, 95.24,
        95.20, 94.84, 94.30, 96.34, 97.84,
    ],
}


def plot_sensitivity(ax):
    df = pd.DataFrame(SENSITIVITY_DATA)
    clusters = sorted(df['Cluster Number'].unique())
    steps_per_cluster = df.groupby('Cluster Number')['Total Steps'].first().loc[clusters]
    time_by_cluster = [
        df[df['Cluster Number'] == c]['Total Time'].values for c in clusters
    ]

    bp = ax.boxplot(
        time_by_cluster,
        positions=clusters,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for box in bp['boxes']:
        box.set_facecolor(COLOR_BOX)
        box.set_edgecolor(COLOR_BOX_EDGE)
        box.set_alpha(0.72)
        box.set_linewidth(LINE_WIDTH)
    for w in bp['whiskers']:
        w.set(color='0.45', linewidth=LINE_WIDTH, linestyle='--')
    for c in bp['caps']:
        c.set(color='0.45', linewidth=LINE_WIDTH)
    for m in bp['medians']:
        m.set(color=COLOR_MEDIAN, linewidth=2)

    ax.set_ylabel('Total Time (s)', fontsize=FONT_LABEL, color=COLOR_BOX)
    ax.tick_params(axis='y', labelcolor=COLOR_BOX)
    ax.set_ylim(88, 112)
    ax.set_xlabel('Number of Clusters', fontsize=FONT_LABEL)
    ax.set_xticks(clusters)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=GRID_ALPHA, color='0.75', linewidth=0.8)
    ax.xaxis.grid(False)

    # 右 Y 轴: Total Steps 折线
    ax2 = ax.twinx()
    ax2.plot(clusters, steps_per_cluster.values, color=COLOR_STEPS_LINE,
             marker='s', linewidth=2, markersize=7,
             markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    ax2.set_ylabel('Total Steps', fontsize=FONT_LABEL, color=COLOR_STEPS_LINE)
    ax2.tick_params(axis='y', labelcolor=COLOR_STEPS_LINE, labelsize=FONT_TICK)
    ax2.set_ylim(600, 750)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_edgecolor('0.15')
    ax2.spines['right'].set_linewidth(SPINE_WIDTH)
    ax2.spines['top'].set_visible(False)

    return ax2


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    _apply_rc()

    k_range, wcss, elbow_k = compute_elbow_data()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), constrained_layout=True,
    )

    plot_elbow(ax_top, k_range, wcss, elbow_k)
    ax_bot_right = plot_sensitivity(ax_bot)

    # 统一轴线样式
    for ax in [ax_top, ax_bot]:
        for sp in ax.spines.values():
            sp.set_edgecolor('0.15')
            sp.set_linewidth(SPINE_WIDTH)
    for sp in ax_bot_right.spines.values():
        sp.set_edgecolor('0.15')
        sp.set_linewidth(SPINE_WIDTH)

    # 上图不需要右轴
    ax_top.spines['right'].set_visible(False)

    out_pdf = os.path.join(script_dir, 'combined_cluster_analysis.pdf')
    out_png = os.path.join(script_dir, 'combined_cluster_analysis.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.show()
    print(f'图表已保存: {out_pdf}, {out_png}')


if __name__ == '__main__':
    main()
