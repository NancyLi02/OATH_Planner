import matplotlib.pyplot as plt
import numpy as np

# Data
task_num = [10, 15, 20, 25]
baseline = [199, 497, 687, 868]
llm_oath = [248, 389, 554, 603]
improvement = [-49, 108, 133, 265]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(task_num))

# 使用低饱和度的柔和颜色
soft_colors = ['#A6CEE3', '#B2DF8A', '#FDBF6F']  # pastel blue, green, orange

# 设置全局字体大小，统一调整坐标轴、图例、刻度和标注字体
plt.rcParams.update({
    'font.size': 14,         # 基础字体大小（影响坐标轴、legend等）
    'axes.labelsize': 16,    # 坐标轴标签字体
    'xtick.labelsize': 14,   # x轴刻度字体
    'ytick.labelsize': 14,   # y轴刻度字体
    'legend.fontsize': 14,   # 图例字体
})

# 重新绘图，应用字体设置
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - bar_width, baseline, width=bar_width, label='Baseline', color=soft_colors[0])
bars2 = plt.bar(x, llm_oath, width=bar_width, label='LLM-OATH', color=soft_colors[1])
bars3 = plt.bar(x + bar_width, improvement, width=bar_width, label='Improvement', color=soft_colors[2])

# 添加放大的柱状图顶部数值标注
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height}', 
                 ha='center', va='bottom', fontsize=14)

# 设置坐标轴标签和图例
plt.xlabel('Number of tasks')
plt.ylabel('Total Steps')
plt.legend(loc='upper left')
plt.xticks(x, task_num)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# 显示图像
plt.show()



