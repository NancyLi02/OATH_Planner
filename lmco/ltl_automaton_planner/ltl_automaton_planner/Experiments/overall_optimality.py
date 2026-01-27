#!/usr/bin/env python3
"""
Visualization script for OATH vs. MILP comparison.

Generates:
1) Computation Time vs. Number of Tasks (log scale)
2) Solution Cost vs. Number of Tasks
3) MILP Optimality Gap w.r.t. Lower Bound

All data are taken from experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # =======================
    # Data
    # =======================
    tasks = np.array([9, 12, 15, 18, 21])

    # Objective values
    oath_cost = np.array([164, 197, 316, 399, 440])
    milp_cost = np.array([159, 197, 224, 300, 354])

    # Computation time (seconds)
    oath_time = np.array([0.0716, 0.0837, 0.2011, 0.3237, 0.3663])
    milp_time = np.array([3.9627, 32.9447, 600.0, 600.0, 600.0])  # time limit

    # MILP time-limit indicator
    milp_timeout = np.array([False, False, True, True, True])

    # MILP gap w.r.t. lower bound (%)
    milp_gap_lb = np.array([0.0, 0.0, 6.07, 16.69, 22.12])

    # =======================
    # Figure 1: Computation Time
    # =======================
    plt.figure(figsize=(7, 4.5))

    plt.plot(tasks, oath_time, marker='o', linewidth=2, label='OATH')
    plt.plot(tasks, milp_time, marker='s', linestyle='--', linewidth=2,
             label='MILP')

    # Mark MILP time-limit points
    plt.scatter(
        tasks[milp_timeout],
        milp_time[milp_timeout],
        marker='x',
        s=80,
        label='MILP (time limit reached)'
    )

    plt.yscale('log')
    plt.xlabel('Number of Tasks', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # =======================
    # Figure 2: Solution Cost
    # =======================
    plt.figure(figsize=(7, 4.5))

    plt.plot(tasks, oath_cost, marker='o', linewidth=2, label='OATH')
    plt.plot(tasks, milp_cost, marker='s', linestyle='--', linewidth=2,
             label='MILP (best found)')

    plt.xlabel('Number of Tasks', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # =======================
    # Figure 3: MILP Gap vs. Lower Bound
    # =======================
    plt.figure(figsize=(7, 4.5))

    plt.plot(tasks, milp_gap_lb, marker='d', linewidth=2)
    plt.xlabel('Number of Tasks', fontsize=12)
    plt.ylabel('Optimality Gap w.r.t. Lower Bound (%)', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
