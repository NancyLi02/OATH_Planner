"""
分析 LLM_System.xlsx 中三种任务场景的 LLM parsing time 与 system response time，
绘制箱线图+散点图，对数刻度 Y 轴。科研出版级绘图样式。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os

# 脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(script_dir, 'LLM_System.xlsx')

# 三种任务在图表上的显示名称
SCENARIO_NAMES = {
    'Add New Task': 'Add New Task',
    'New obstacles detected': 'Obstacle Detected',
    'Change Task Priority': 'Change Priority',
}

# 主色（与示例一致）
PALETTE = {
    'LLM Parsing Time': '#5C83B3',
    'Planner Update Time': '#D18358',
}

# 科研绘图字体与尺寸
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 15
LINE_WIDTH = 1.2
GRID_ALPHA = 0.4
SPINE_WIDTH = 1.0


def load_and_reshape():
    """读取 Excel 各 sheet，统一列名并转为长格式。"""
    xls = pd.ExcelFile(excel_file)
    rows_list = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        # 列名可能略有不同（大小写），统一为 LLM / system
        cols = [c for c in df.columns]
        llm_col = next(c for c in cols if 'llm' in c.lower() and 'parsing' in c.lower())
        sys_col = next(c for c in cols if 'system' in c.lower() and 'response' in c.lower())

        scenario = SCENARIO_NAMES.get(sheet_name, sheet_name)

        for _, row in df.iterrows():
            llm_val = row[llm_col]
            sys_val = row[sys_col]
            if pd.notna(llm_val):
                rows_list.append({
                    'Instruction Scenario': scenario,
                    'System Component': 'LLM Parsing Time',
                    'Time (seconds)': float(llm_val),
                })
            if pd.notna(sys_val):
                rows_list.append({
                    'Instruction Scenario': scenario,
                    'System Component': 'Planner Update Time',
                    'Time (seconds)': float(sys_val),
                })

    return pd.DataFrame(rows_list)


def main():
    df = load_and_reshape()
    if df.empty:
        print('未读取到有效数据。')
        return

    # 保证 X 轴顺序与示例一致
    scenario_order = ['Add New Task', 'Obstacle Detected', 'Change Priority']
    df = df[df['Instruction Scenario'].isin(scenario_order)]
    df['Instruction Scenario'] = pd.Categorical(
        df['Instruction Scenario'], categories=scenario_order, ordered=True
    )
    df = df.sort_values('Instruction Scenario')

    # 过滤掉非正值，避免 log 报错
    df = df[df['Time (seconds)'] > 0].copy()

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

    # 箱线图：清晰轮廓、不显示离群点（由散点体现分布）
    bp = sns.boxplot(
        x='Instruction Scenario',
        y='Time (seconds)',
        hue='System Component',
        data=df,
        palette=PALETTE,
        width=0.6,
        linewidth=LINE_WIDTH,
        showfliers=False,
        ax=ax,
    )
    # 为箱线设置略深边框，增强可读性
    for artist in bp.artists:
        artist.set_edgecolor('0.25')
        artist.set_linewidth(LINE_WIDTH)

    # 叠加散点：适度透明、小抖动，不重复图例
    sns.stripplot(
        x='Instruction Scenario',
        y='Time (seconds)',
        hue='System Component',
        data=df,
        dodge=True,
        alpha=0.65,
        palette=PALETTE,
        size=4.5,
        linewidth=0.4,
        edgecolor='white',
        ax=ax,
        legend=False,
    )

    ax.set_yscale('log')
    ax.set_ylabel('Time (seconds, Log Scale)', fontsize=FONT_LABEL, fontweight='normal')
    ax.set_xlabel('')
    ax.set_title('')
    # 图例：左下角、白底细框、与正文协调
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[0:2], labels[0:2],
        loc='lower left',
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

    out_pdf = os.path.join(script_dir, 'LLM_System_time_analysis.pdf')
    out_png = os.path.join(script_dir, 'LLM_System_time_analysis.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.show()
    print(f'图表已保存: {out_pdf}, {out_png}')


if __name__ == '__main__':
    main()
