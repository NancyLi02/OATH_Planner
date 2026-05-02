"""
Task allocation comparison: grouped bar charts for Total Steps and Total Time.
Two vertically stacked subplots, publication-quality style.
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(script_dir, 'Results comparsion.xlsx')

# ── 科研绘图常量（与其他脚本保持一致）──────────────────────────
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 14
LINE_WIDTH = 1.2
GRID_ALPHA = 0.35
SPINE_WIDTH = 1.0

COLORS = ['#BDBADB', '#8DD1C6', '#FBC99A', '#EF98A1']
EDGE_COLORS = ['#8A86B8', '#5FB3A0', '#D9A56E', '#D06A75']

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


def plot_grouped_bar(ax, sheet_name, ylabel):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    task_numbers = df['Task Number']
    methods = df.columns[1:]

    x = np.arange(len(task_numbers))
    n = len(methods)
    width = 0.18

    for i, method in enumerate(methods):
        bars = ax.bar(
            x + i * width - width * (n - 1) / 2,
            df[method],
            width,
            color=COLORS[i],
            edgecolor=EDGE_COLORS[i],
            linewidth=0.9,
            label=method,
        )

    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(task_numbers.astype(int))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=GRID_ALPHA, color='0.75', linewidth=0.8)
    ax.xaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor('0.15')
        spine.set_linewidth(SPINE_WIDTH)


fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)

plot_grouped_bar(axes[0], 'Total Steps', 'Total Running Steps')
plot_grouped_bar(axes[1], 'Total Time', 'Total Running Time (s)')

axes[1].set_xlabel('Task Number', fontsize=FONT_LABEL)

# 只在上方子图放图例（两子图共享相同的 method）
# handles, labels = axes[0].get_legend_handles_labels()
# axes[0].legend(
#     handles, labels,
#     loc='upper left',
#     frameon=True,
#     framealpha=1,
#     edgecolor='0.3',
#     fancybox=False,
#     fontsize=FONT_LEGEND,
#     ncol=2,
# )

out_pdf = os.path.join(script_dir, 'comparison_results.pdf')
out_png = os.path.join(script_dir, 'comparison_results.png')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.show()
print(f'图表已保存: {out_pdf}, {out_png}')
