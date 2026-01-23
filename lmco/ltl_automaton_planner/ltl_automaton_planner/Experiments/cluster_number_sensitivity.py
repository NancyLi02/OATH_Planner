import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Raw data
# -----------------------------
data = {
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

df = pd.DataFrame(data)

clusters = sorted(df["Cluster Number"].unique())
steps_per_cluster = df.groupby("Cluster Number")["Total Steps"].first().loc[clusters]
time_by_cluster = [
    df[df["Cluster Number"] == c]["Total Time"].values
    for c in clusters
]

# -----------------------------
# Plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))

# Total Time boxplot
ax1.boxplot(
    time_by_cluster,
    positions=clusters,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="#1f77b4", edgecolor="black", alpha=0.7),
    medianprops=dict(color="#ff7f0e", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black")
)



ax1.set_xlabel("Cluster Number", fontsize=16)
ax1.set_ylabel("Total Time", fontsize=16)
ax1.set_ylim(80, 120)   # 放大 Time 纵轴范围
ax1.tick_params(axis="y", labelsize=16)
ax1.tick_params(axis="x", labelsize=16)

# Total Steps line (secondary y-axis)
ax2 = ax1.twinx()
ax2.plot(
    clusters,
    steps_per_cluster.values,
    color="#d62728",
    marker="o",
    linewidth=2
)

ax2.set_ylabel("Total Steps", fontsize=16)
ax2.set_ylim(600, 750)  # 放大 Steps 纵轴范围
ax2.tick_params(axis="y", labelsize=16)
ax2.tick_params(axis="x", labelsize=16)
# plt.title("Total Time (Boxplot) and Total Steps (Line)", fontsize=16)

script_dir = os.path.dirname(os.path.abspath(__file__))

cluster_number_sensitivity_plot_path = os.path.join(script_dir, 'cluster_number_sensitivity.png')

if cluster_number_sensitivity_plot_path:
    plt.savefig(cluster_number_sensitivity_plot_path, dpi=150, bbox_inches='tight')
    print(f"Cluster number sensitivity plot saved to: {cluster_number_sensitivity_plot_path}")

plt.show()
