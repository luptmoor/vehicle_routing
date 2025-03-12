import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

from sample_case_2 import run

M = 5
N = 9


# ------------------------ SPEED SENSITIVITY ANALYSIS ------------------------

def run_speed_analysis(N=9, M=5, trials=1, lower=0.4, upper=1.6, speed_steps=30):
    """
    Runs speed sensitivity analysis, recording the multipliers where the solution changes.
    """
    left_changes, right_changes = [], []
    no_change_lower, no_change_upper = 0, 0
    infeasible_base = 0

    for seed in range(trials):
        np.random.seed(seed)
        base_result = run(N=N, M=M, speed_multiplier=1.0, random_fleet=True, seed=seed)
        if base_result["solution_x"] is None:
            print(f"WARNING: Base case infeasible for seed {seed}. Skipping trial.")
            infeasible_base += 1
            continue

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]
        base_edges = base_result["solution_x"]


        left_changed, right_changed = False, False

        # Test lower multipliers (speed decrease)
        for multiplier in np.linspace(1.0, lower, speed_steps):
            result = run(N, M, speed_multiplier=multiplier, random_fleet=False, fleet_composition=fixed_fleet,
                         seed=seed, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)
            if result and result["solution_x"] != base_edges:
                left_changes.append(multiplier)
                left_changed = True
                break  # Stop after first change

        # Test upper multipliers (speed increase)
        for multiplier in np.linspace(1.0, upper, speed_steps):
            result = run(N, M, speed_multiplier=multiplier, random_fleet=False, fleet_composition=fixed_fleet,
                         seed=seed, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)
            if result and result["solution_x"] != base_edges:
                right_changes.append(multiplier)
                right_changed = True
                break  # Stop after first change

        # If no change was observed for a trial, increment counter
        if not left_changed:
            no_change_lower += 1
        if not right_changed:
            no_change_upper += 1

    print("\nFinal Counts for Speed Sensitivity:")
    print(f"Speed Down Changes: {len(left_changes), Counter(left_changes)}")
    print(f"Speed Up Changes: {len(right_changes), (right_changes)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials with No Lower Change: {no_change_lower}")
    print(f"Trials with No Upper Change: {no_change_upper}")

    return Counter(left_changes), Counter(right_changes)


def plot_speed_analysis(N=9, M=5, speed_steps=30, lower=0.4, upper=1.6, trials=8):
    """Plots the distribution of speed multipliers where the optimal solution changes."""
    left_counts, right_counts = run_speed_analysis(N, M, trials, lower, upper, speed_steps)

    # Ensure we include **all possible tested multipliers** in the x-axis
    left_multipliers = np.round(np.linspace(1.0, lower, speed_steps), decimals=4)
    right_multipliers = np.round(np.linspace(1.0, upper, speed_steps), decimals=4)
    tested_multipliers = sorted(set(left_multipliers).union(set(right_multipliers)))

    # Get counts including zero occurrences for all tested multipliers
    left_frequencies = [left_counts.get(m, 0) for m in tested_multipliers]
    right_frequencies = [right_counts.get(m, 0) for m in tested_multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots for changes
    ax.bar(tested_multipliers, left_frequencies, width=0.02, label="Decrease", zorder = 1)
    ax.bar(tested_multipliers, right_frequencies, width=0.02, label="Increase", zorder = 1)

    # Set labels and title
    ax.set_xlabel("Speed Multiplier Where Optimal Solution Changes", fontsize=14)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize=14)
    ax.set_title(f"Distribution of Critical Speed Values (M={M}, N={N}, Trials={trials})", fontsize=16)

    # Ensure all x-ticks are displayed
    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    ax.set_xlim(lower - (1-lower)/(speed_steps-1), upper + (upper-1)/(speed_steps-1))

    # Move grid behind bars
    # Grid styling
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add dashed vertical lines at each x-tick position
    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.6, zorder = 0)

    ax.set_axisbelow(True)

    ax.legend()

    plt.savefig("Figures/characteristics/distribution_speed.png", dpi = 300,
                bbox_inches='tight' )
    plt.show()


# ------------------------ CAPACITY SENSITIVITY ANALYSIS ------------------------

def run_capacity_infeasibility_analysis(N=9, M=5, trials=8, lower=0.4, capacity_steps=30):
    """Finds the first lower multiplier where the model becomes infeasible, and checks if it ever does."""
    lower_infeasible = []
    always_feasible = 0  # Count how many trials never became infeasible
    infeasible_base = 0
    for seed in range(trials):
        base_result = run(N, M, capacity_multiplier=1.0, random_fleet=True, seed=seed)
        if base_result["solution_x"] is None:
            print(f"WARNING: Base case infeasible for seed {seed}. Skipping trial.")
            infeasible_base += 1
            continue

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        found_infeasible = False  # Track if we found infeasibility

        for multiplier in np.round(np.linspace(1.0, lower, capacity_steps), 4):
            result = run(N, M, capacity_multiplier=multiplier, random_fleet=False,
                         fleet_composition=fixed_fleet, seed=seed,
                         fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)
            if result["solution_x"] is None:
                lower_infeasible.append(multiplier)
                found_infeasible = True
                break  # Stop at first infeasible case

        if not found_infeasible:
            always_feasible += 1  # If never infeasible, increment counter

    print("\nFinal Counts for Capacity Sensitivity:")
    print(f"Capacity Infeasibility Multipliers: {Counter(lower_infeasible)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials that never became infeasible: {always_feasible}")

    return Counter(lower_infeasible)



def plot_capacity_analysis(N=9, M=5, capacity_steps=30, lower=0.4, trials=8):
    """Plots the distribution of capacity multipliers where infeasibility occurs (lower only)."""
    lower_counts = run_capacity_infeasibility_analysis(N, M, trials, lower, capacity_steps)

    # Ensure we include all tested multipliers
    tested_multipliers = np.round(np.linspace(1.0, lower, capacity_steps), decimals=4)
    tested_multipliers = sorted(set(tested_multipliers))

    # Get counts including zero occurrences
    lower_frequencies = [lower_counts.get(m, 0) for m in tested_multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Move grid behind bars
    ax.set_axisbelow(True)

    # Add dashed vertical lines at each x-tick
    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.6, zorder=0)

    # Bar plot for infeasibility multipliers
    ax.bar(tested_multipliers, lower_frequencies, width=0.02, label="Decrease", color="tab:blue", zorder=2)

    # Labels and title
    ax.set_xlabel("Capacity Multiplier Where Model Becomes Infeasible", fontsize = 15)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize = 15)
    ax.set_title(f"Distribution of Capacity Infeasibility Thresholds (M={M}, N={N}, Trials={trials})", fontsize = 16)

    # Set x-ticks to ensure all multipliers are displayed
    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    # Grid styling
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(lower, 1)
    ax.legend()

    plt.savefig(f"Figures/characteristics/distribution_capacity_N{N}_M{M}.png", dpi=300,
                bbox_inches='tight' )
    plt.show()



# ------------------------ DEMAND SENSITIVITY ANALYSIS ------------------------

def run_demand_infeasibility_analysis(N=9, M=5, trials=8, upper=2.0, demand_steps=30):
    """Finds the first higher multiplier where the model becomes infeasible, and checks if it ever does."""
    upper_infeasible = []
    always_feasible = 0  # Count how many trials never became infeasible

    for seed in range(trials):
        np.random.seed(seed)
        infeasible_base = 0
        base_result = run(N, M, capacity_multiplier=1.0, random_fleet=True, seed=seed)
        if base_result["solution_x"] is None:
            print(f"WARNING: Base case infeasible for seed {seed}. Skipping trial.")
            infeasible_base += 1
            continue

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        found_infeasible = False  # Track if we found infeasibility

        for multiplier in np.round(np.linspace(1.0, upper, demand_steps), 4):
            if multiplier > 1.0:
                increased_demand = [int(d * multiplier) for d in fixed_demand]
                result = run(N, M, capacity_multiplier=1.0, random_fleet=False,
                             fleet_composition=fixed_fleet, seed=seed,
                             fixed_demand=increased_demand, fixed_nodes=fixed_nodes)
                if result["solution_x"] is None:
                    upper_infeasible.append(multiplier)
                    found_infeasible = True
                    break  # Stop at first infeasible case

        if not found_infeasible:
            always_feasible += 1  # If never infeasible, increment counter

    print("\nFinal Counts for Demand Sensitivity:")
    print(f"Demand Infeasibility Multipliers: {Counter(upper_infeasible)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials that never became infeasible: {always_feasible}")

    return Counter(upper_infeasible)



def plot_demand_infeasibility_analysis(N=9, M=5, demand_steps=51, upper=2.0, trials=8):
    """Plots the distribution of demand multipliers where infeasibility occurs (higher only)."""
    upper_counts = run_demand_infeasibility_analysis(N, M, trials, upper, demand_steps)
    print(upper_counts)
    # Ensure we include all tested multipliers
    tested_multipliers = np.round(np.linspace(1.0, upper, demand_steps), decimals=4)
    tested_multipliers = sorted(set(tested_multipliers))

    # Get counts including zero occurrences
    upper_frequencies = [upper_counts.get(m, 0) for m in tested_multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Move grid behind bars
    ax.set_axisbelow(True)

    # Add dashed vertical lines at each x-tick
    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha = 0.6, zorder=0)

    # Bar plot for infeasibility multipliers
    ax.bar(tested_multipliers, upper_frequencies, width=0.02, label="Increase", color="tab:orange", zorder=2)

    # Labels and title
    ax.set_xlabel("Demand Multiplier Where Model Becomes Infeasible", fontsize=15)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize=15)
    ax.set_title(f"Distribution of Demand Infeasibility Thresholds (M={M}, N={N}, Trials={trials})", fontsize=16)

    # Set x-ticks to ensure all multipliers are displayed
    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    # Grid styling
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(1, upper)
    ax.legend()
    plt.savefig("Figures/characteristics/distribution_demand.png", dpi=300,
                bbox_inches='tight' )
    plt.show()


# plot_speed_analysis(N=9, M=5, lower=0.6, upper=1.4, speed_steps=21, trials=100)

plot_capacity_analysis(N=9, M=5, lower=0.5, capacity_steps=26, trials=100)

# plot_demand_infeasibility_analysis(N=9, M=5, upper=2, demand_steps=51, trials=100)

