"""
用户研究：Human Formulation Time vs System Reaction Time，
箱线图+散点图。科研出版级绘图样式。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(script_dir, 'User_study.xlsx')

# 科研绘图样式常量（与 plot_llm_system_time 一致）
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 15
LINE_WIDTH = 1.2
GRID_ALPHA = 0.4
SPINE_WIDTH = 1.0

# 配色：与 plot_llm_system_time 一致（蓝 / 橙）
PALETTE = {
    "Human Formulation Time": "#5C83B3",
    "System Reaction Time": "#D18358",
}

# 读取数据并进行预处理
all_data = []

# 获取所有 sheet 的名称
try:
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    print(f"成功读取 Excel 文件，找到 sheets: {sheet_names}")
except FileNotFoundError:
    print(f"错误: 文件 '{excel_file}' 未找到。请确保文件与脚本在同一目录下，或提供完整路径。")
    exit()

# 检查是否有至少一个 sheet
if not sheet_names:
    print(f"错误: Excel 文件 '{excel_file}' 中没有找到任何 sheets。")
    exit()

for sheet_name in sheet_names:
    try:
        # 读取当前 Sheet 的前两列
        df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=[0, 1])
        
        # 检查读取到的数据是否有效
        if df.shape[1] < 2:
            print(f"警告: Sheet '{sheet_name}' 的列数少于2，已跳过。")
            continue
            
        col1_name = df.columns[0]
        col2_name = df.columns[1]

        # 将“宽格式”转换为“长格式”以便 Seaborn 绘图
        # 假设第一列是用户操作时间，第二列是系统反应时间
        
        # 处理第一列数据 (e.g., Human Formulation Time)
        h_df = pd.DataFrame({
            'Instruction Type': sheet_name,
            'Category': 'Human Formulation Time',
            'Time (s)': df[col1_name]
        })
        
        # 处理第二列数据 (e.g., System Reaction Time)
        s_df = pd.DataFrame({
            'Instruction Type': sheet_name,
            'Category': 'System Reaction Time',
            'Time (s)': df[col2_name]
        })
        
        all_data.append(h_df)
        all_data.append(s_df)

    except Exception as e:
        print(f"处理 Sheet '{sheet_name}' 时出错: {e}")


# 合并所有数据
if not all_data:
    print("错误: 未能从 Excel 文件中加载任何有效数据，绘图程序终止。")
    exit()

final_df = pd.concat(all_data, ignore_index=True)

# 科研出版级全局样式
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

fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='-', alpha=GRID_ALPHA, color='0.75', linewidth=0.8)
ax.xaxis.grid(False)

# 箱线图
bp = sns.boxplot(
    x='Instruction Type',
    y='Time (s)',
    hue='Category',
    data=final_df,
    palette=PALETTE,
    width=0.6,
    linewidth=LINE_WIDTH,
    showfliers=False,
    ax=ax,
)
for artist in bp.artists:
    artist.set_edgecolor('0.25')
    artist.set_linewidth(LINE_WIDTH)

# 叠加散点（与箱线同色、白边，不重复图例）
sns.stripplot(
    x='Instruction Type',
    y='Time (s)',
    hue='Category',
    data=final_df,
    dodge=True,
    alpha=0.65,
    palette=PALETTE,
    size=4.5,
    linewidth=0.4,
    edgecolor='white',
    ax=ax,
    legend=False,
)

ax.set_ylabel('Time (seconds)', fontsize=FONT_LABEL, fontweight='normal')
ax.set_xlabel('')
ax.set_ylim(0, final_df['Time (s)'].max() * 1.08)

# 图例：右上角、白底细框
handles, labels = ax.get_legend_handles_labels()
if handles:
    leg = ax.legend(
        handles[0:2], labels[0:2],
        loc='upper right',
        frameon=True,
        framealpha=1,
        edgecolor='0.3',
        fancybox=False,
        fontsize=FONT_LEGEND,
    )
    leg.get_frame().set_linewidth(0.8)

for spine in ax.spines.values():
    spine.set_edgecolor('0.15')
    spine.set_linewidth(SPINE_WIDTH)

out_pdf = os.path.join(script_dir, 'LLM_experiment_results.pdf')
out_png = os.path.join(script_dir, 'LLM_experiment_results.png')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.show()

print(f"绘图完成，结果已保存为: {out_pdf}, {out_png}")
