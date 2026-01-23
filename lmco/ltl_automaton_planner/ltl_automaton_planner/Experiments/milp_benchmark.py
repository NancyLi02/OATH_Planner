"""
MILP Benchmark Script (Decoupled Version)

- Run MILP benchmark experiments and save raw results
- Load saved results independently
- Generate publication-ready PDF boxplots
"""

import time
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import gurobipy as gp

from MILP import ClusterTaskPlanner


# =========================
# Configuration Utilities
# =========================

def load_task_config(yaml_path: str) -> Dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_coordinate(coord_str: str) -> Tuple[float, float]:
    x, y = coord_str.split(",")
    return float(x), float(y)


def prepare_experiment_data(config: Dict):
    pickup_points, pickup_labels = [], []
    for coord_str, label in config["task_points"].items():
        pickup_points.append(parse_coordinate(coord_str))
        pickup_labels.append(label)

    delivery_points, delivery_labels = [], []
    for coord_str, label in config["delivery_points"].items():
        delivery_points.append(parse_coordinate(coord_str))
        delivery_labels.append(label)

    pickup_to_delivery = {}
    for delivery_label, task_list in config["task_to_delivery"].items():
        for task_label in task_list:
            pickup_to_delivery[task_label] = delivery_label

    first_robot = list(config["robot_positions"].keys())[0]
    robot_start = parse_coordinate(config["robot_positions"][first_robot])

    return (
        pickup_points,
        pickup_labels,
        delivery_points,
        delivery_labels,
        pickup_to_delivery,
        robot_start,
    )


def get_solver_info() -> Dict:
    return {
        "solver_type": "MILP (Mixed Integer Linear Programming)",
        "solver_name": "Gurobi",
        "gurobi_version": gp.gurobi.version(),
    }


# =========================
# Experiment Core
# =========================

def run_single_experiment(
    planner: ClusterTaskPlanner,
    robot_start: Tuple[float, float],
    pickup_points: List[Tuple[float, float]],
    pickup_labels: List[str],
    delivery_points: List[Tuple[float, float]],
    delivery_labels: List[str],
    pickup_to_delivery: Dict[str, str],
    capacity: int,
) -> Tuple[float, bool, float, List[str]]:
    start_time = time.time()
    try:
        chosen_pickups, chosen_deliveries, route, cost = planner.plan_cluster_tasks(
            robot_start=robot_start,
            pickup_points=pickup_points,
            pickup_labels=pickup_labels,
            delivery_points=delivery_points,
            delivery_labels=delivery_labels,
            pickup_to_delivery=pickup_to_delivery,
            robot_capacity=capacity,
        )
        solve_time = time.time() - start_time
        success = cost < float("inf")
        return solve_time, success, cost, route
    except Exception as e:
        print(f"Optimization failed: {e}")
        return time.time() - start_time, False, float("inf"), []


def run_benchmark_experiment(
    config_path: str,
    dijkstra_csv_path: str,
    output_dir: str,
    min_capacity: int = 3,
    max_capacity: int = None,
    num_trials: int = 5,
) -> pd.DataFrame:
    solver_info = get_solver_info()

    config = load_task_config(config_path)
    (
        pickup_points,
        pickup_labels,
        delivery_points,
        delivery_labels,
        pickup_to_delivery,
        robot_start,
    ) = prepare_experiment_data(config)

    total_tasks = len(pickup_labels)
    if max_capacity is None:
        max_capacity = total_tasks
    else:
        max_capacity = min(max_capacity, total_tasks)

    planner = ClusterTaskPlanner(dijkstra_csv_path)

    results = []

    for capacity in range(min_capacity, max_capacity + 1):
        for trial in range(num_trials):
            solve_time, success, cost, route = run_single_experiment(
                planner,
                robot_start,
                pickup_points,
                pickup_labels,
                delivery_points,
                delivery_labels,
                pickup_to_delivery,
                capacity,
            )

            results.append(
                {
                    "capacity": capacity,
                    "trial": trial + 1,
                    "solve_time": solve_time,
                    "success": success,
                    "cost": cost,
                    "solver": solver_info["solver_name"],
                    "method": "MILP",
                    "route": " → ".join(["START"] + route),
                }
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    excel_path = output_path / "milp_benchmark_results.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"Benchmark results saved to: {excel_path}")
    return df


# =========================
# Result Loading
# =========================

def load_benchmark_results(result_path: str) -> pd.DataFrame:
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")


# =========================
# Visualization
# =========================

def plot_benchmark_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    filename: str = "milp_benchmark_boxplot.pdf",
):
    df_success = df[df["success"]]

    capacities, data = [], []
    for c in sorted(df_success["capacity"].unique()):
        values = df_success.loc[df_success["capacity"] == c, "solve_time"].values
        if len(values) > 0:
            capacities.append(c)
            data.append(values)

    if len(data) == 0:
        raise ValueError("No successful runs to plot")

    fig, ax = plt.subplots(figsize=(8, 4))

    # Calculate statistics for trend shading
    positions = np.arange(1, len(data) + 1)
    means = np.array([np.mean(d) for d in data])
    mins = np.array([np.min(d) for d in data])
    maxs = np.array([np.max(d) for d in data])

    # Polynomial fitting for smooth trend curve (degree 3 for rise-then-fall pattern)
    poly_degree = min(3, len(positions) - 1)
    
    # Fit polynomial to means
    coeffs_mean = np.polyfit(positions, means, poly_degree)
    poly_mean = np.poly1d(coeffs_mean)
    
    # Add padding to min/max before fitting to ensure coverage of all data points
    data_range = maxs.max() - mins.min()
    padding = data_range * 0.2  # 20% padding
    maxs_padded = maxs + padding
    mins_padded = mins - padding
    
    # Fit polynomial to padded upper and lower bounds
    coeffs_upper = np.polyfit(positions, maxs_padded, poly_degree)
    coeffs_lower = np.polyfit(positions, mins_padded, poly_degree)
    poly_upper = np.poly1d(coeffs_upper)
    poly_lower = np.poly1d(coeffs_lower)
    
    # Generate smooth curve points
    x_smooth = np.linspace(positions.min(), positions.max(), 100)
    y_mean_smooth = poly_mean(x_smooth)
    y_upper_smooth = poly_upper(x_smooth)
    y_lower_smooth = poly_lower(x_smooth)
    
    # Ensure lower bound doesn't go negative
    y_lower_smooth = np.maximum(y_lower_smooth, 0)

    # Add light purple shaded area to show fitted trend range (covers all data points)
    ax.fill_between(
        x_smooth,
        y_lower_smooth,
        y_upper_smooth,
        color="#9370DB",  # Medium purple
        alpha=0.15,
        zorder=1,
    )

    # Add smooth trend line through fitted means (dashed)
    ax.plot(
        x_smooth,
        y_mean_smooth,
        color="#8A2BE2",  # Blue violet
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        zorder=2,
    )

    bp = ax.boxplot(
        data,
        labels=capacities,
        patch_artist=True,
        showfliers=True,
        zorder=3,
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Number of Tasks (Capacity)", fontsize=16)
    ax.set_ylabel("Solve Time (seconds)", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    ax.scatter(
        positions,
        means,
        marker="D",
        s=20,
        color="red",
        zorder=5,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / filename

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf")
    plt.close()

    print(f"Boxplot saved to: {pdf_path}")


# =========================
# Summary (Optional)
# =========================

def print_summary(df: pd.DataFrame):
    df_success = df[df["success"]]

    summary = (
        df_success.groupby("capacity")
        .agg(
            mean_time=("solve_time", "mean"),
            std_time=("solve_time", "std"),
            min_time=("solve_time", "min"),
            max_time=("solve_time", "max"),
            mean_cost=("cost", "mean"),
            success_count=("success", "count"),
        )
        .round(4)
    )

    print("\nSUMMARY STATISTICS")
    print(summary)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "benchmark_results"
    result_file = output_dir / "milp_benchmark_results.xlsx"

    # -------- MODE 1: Run experiments --------
    # run_benchmark_experiment(
    #     config_path=str(script_dir.parent / "config" / "Task_Points.yaml"),
    #     dijkstra_csv_path=str(script_dir / "multi_source_dijkstra_distances.csv"),
    #     output_dir=str(output_dir),
    #     min_capacity=3,
    #     max_capacity=None,
    #     num_trials=5,
    # )

    # -------- MODE 2: Plot only --------
    df = load_benchmark_results(result_file)
    plot_benchmark_boxplot(df, output_dir)
    print_summary(df)
