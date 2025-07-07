import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体为 Calibri（前提是系统中已安装该字体）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri'],  # fallback 可加 'Arial' 等
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'text.usetex': False  # 禁用 LaTeX，防止错误
})

# 数据
task_numbers = [25, 50, 75, 100]
total_steps = [917, 1614, 2074, 2543]
total_time_seconds = [109, 153, 183, 255]

# 创建双Y轴图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：Total Steps
color_steps = 'tab:blue'
ax1.set_xlabel('Number of Tasks')
ax1.set_ylabel('Total Steps', color=color_steps)
line1, = ax1.plot(task_numbers, total_steps, marker='o', color=color_steps, label='Total Steps')
ax1.tick_params(axis='y', labelcolor=color_steps)
ax1.set_ylim(bottom=0)

# 添加步数标注
for x, y in zip(task_numbers, total_steps):
    ax1.text(x, y + 50, str(y), color=color_steps, fontsize=14, ha='center')

# 右轴：Total Time
ax2 = ax1.twinx()
color_time = 'tab:red'
ax2.set_ylabel('Total Time (s)', color=color_time)
line2, = ax2.plot(task_numbers, total_time_seconds, marker='s', color=color_time, label='Total Time (s)')
ax2.tick_params(axis='y', labelcolor=color_time)
ax2.set_ylim(100, 280)

# 添加时间标注
for x, y in zip(task_numbers, total_time_seconds):
    ax2.text(x, y + 5, str(y), color=color_time, fontsize=14, ha='center')

# 图例
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

# 仅保留纵向网格线
ax1.grid(True, axis='x', linestyle='--', linewidth=0.5)
ax2.grid(False)

# 显示图形（无标题）
fig.tight_layout()
plt.show()
