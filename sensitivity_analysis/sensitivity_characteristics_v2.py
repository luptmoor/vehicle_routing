import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

from sample_case_2 import run

M = 5
N = 9

# ------------------------ SPEED SENSITIVITY ANALYSIS ------------------------
def dicts_equal(dict1, dict2):
    """
    Compares two dictionaries to check if they contain the same keys
    and if their values are numerically close (ignoring order).
    """
    tol = 1e-6
    # Different keys mean different dictionaries
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    return all(abs(dict1[k] - dict2[k]) < tol for k in dict1)


def compute_distance_per_drone_type(route_set, distance_matrix, fleet):
    """
    Computes total travel distance per drone type for a given route set with a certain fleet.
    Used to identify routing changes.
    """
    drone_type_distances = {}

    for vehicle, start, end in route_set:
        # Retrieve the drone type for the current vehicle
        drone_type = fleet[vehicle]
        # Get the travel distance from the matrix
        distance = distance_matrix[start, end]

        # Initialize the drone type in the dictionary if not already present
        if drone_type not in drone_type_distances:
            drone_type_distances[drone_type] = 0.0

        # Accumulate the total travel distance for this drone type
        drone_type_distances[drone_type] += distance

    return drone_type_distances


def run_speed_analysis(N=9, M=5, trials=1, lower=0.4, upper=1.6, speed_steps=30):
    """
    Runs speed sensitivity analysis, recording the multipliers where the solution changes.
    Uses verification of routing changes based on total distance per drone type.
    """
    left_changes, right_changes = [], []
    no_change_lower, no_change_upper = 0, 0
    infeasible_base = 0

    for seed in range(trials):
        base_result = run(N=N, M=M, speed_multiplier=1.0, random_fleet=True, seed=seed)

        if base_result["solution_x"] is None:
            print(f"WARNING: Base case infeasible for seed {seed}. Skipping trial.")
            infeasible_base += 1
            continue

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]
        base_edges = base_result["solution_x"]
        base_distance_per_type = compute_distance_per_drone_type(base_edges, base_result["distance_matrix"],
                                                                 base_result["fleet"])

        # Flags to track if a change occurs
        left_changed, right_changed = False, False

        # Test lower multipliers (speed decrease)
        for multiplier in np.linspace(1.0, lower, speed_steps):
            # Keep node locations, fleet composition and demand the same as the base case
            # to isolate the effect of the speed multiplier
            result = run(N, M, speed_multiplier=multiplier, random_fleet=False, fleet_composition=fixed_fleet,
                         seed=seed, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)

            if result["solution_x"] is not None:
                new_distance_per_type = compute_distance_per_drone_type(result["solution_x"], result["distance_matrix"],
                                                                        result["fleet"])

                # Only record real routing changes (not just drone swaps) by comparing new
                # distances per drone type versus the base case
                if not dicts_equal(base_distance_per_type, new_distance_per_type):
                    left_changes.append(multiplier)
                    left_changed = True
                    break  # Stop after first change

        # Test upper multipliers (speed increase)
        for multiplier in np.linspace(1.0, upper, speed_steps):
            # Keep node locations, fleet composition and demand the same as the base case
            # to isolate the effect of the speed multiplier
            result = run(N, M, speed_multiplier=multiplier, random_fleet=False, fleet_composition=fixed_fleet,
                         seed=seed, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)

            if result["solution_x"] is not None:
                new_distance_per_type = compute_distance_per_drone_type(result["solution_x"], result["distance_matrix"],
                                                                        result["fleet"])

                # Only record real routing changes (not just drone swaps) by comparing new
                # distances per drone type versus the base case
                if not dicts_equal(base_distance_per_type, new_distance_per_type):
                    right_changes.append(multiplier)
                    right_changed = True
                    break  # Stop after first change

        # If no change was observed for a trial, increment counter
        if not left_changed:
            no_change_lower += 1
        if not right_changed:
            no_change_upper += 1

    print("\nFinal Counts for Speed Sensitivity:")
    print(f"Speed Down Changes: {len(left_changes)}, {Counter(left_changes)}")
    print(f"Speed Up Changes: {len(right_changes)}, {Counter(right_changes)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials with No Lower Change: {no_change_lower}")
    print(f"Trials with No Upper Change: {no_change_upper}")

    return Counter(left_changes), Counter(right_changes)



def plot_speed_analysis(N=9, M=5, speed_steps=30, lower=0.4, upper=1.6, trials=8):
    """
    Plots the distribution of speed multipliers where the optimal solution changes.
    """
    left_counts, right_counts = run_speed_analysis(N, M, trials, lower, upper, speed_steps)

    # Ensure plot spans entire range of tested multipliers
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

    ax.set_xlabel("Speed Multiplier Where Optimal Solution Changes", fontsize=14)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize=14)
    ax.set_title(f"Distribution of Critical Speed Values (M={M}, N={N}, Trials={trials})", fontsize=16)

    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    ax.set_xlim(lower - (1-lower)/(speed_steps-1), upper + (upper-1)/(speed_steps-1))

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.6, zorder = 0)

    ax.set_axisbelow(True)
    ax.legend()
    plt.savefig("figures/characteristics/distribution_speed.png", dpi = 300,
                bbox_inches='tight' )
    plt.show()


# ------------------------ CAPACITY SENSITIVITY ANALYSIS ------------------------

def run_capacity_infeasibility_analysis(N=9, M=5, trials=8, lower=0.4, capacity_steps=30):
    """
    Finds the first lower multiplier for capacity where the model becomes infeasible.
    """
    lower_infeasible = []

    # Count how many trials never become infeasible
    always_feasible = 0
    # Count infeasible base cases
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

        found_infeasible = False

        for multiplier in np.linspace(1.0, lower, capacity_steps):
            # Keep node locations, fleet composition and demand the same as the base case
            # to isolate the effect of the capacity multiplier
            result = run(N, M, capacity_multiplier=multiplier, random_fleet=False,
                         fleet_composition=fixed_fleet, seed=seed,
                         fixed_demand=fixed_demand, fixed_nodes=fixed_nodes)
            if result["solution_x"] is None:
                lower_infeasible.append(multiplier)
                found_infeasible = True
                break  # Stop at first infeasible case

        if not found_infeasible:
            always_feasible += 1

    print("\nFinal Counts for Capacity Sensitivity:")
    print(f"Capacity Infeasibility Multipliers: {Counter(lower_infeasible)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials that never became infeasible: {always_feasible}")

    return Counter(lower_infeasible)



def plot_capacity_analysis(N=9, M=5, capacity_steps=30, lower=0.4, trials=8):
    """
    Plots the distribution of capacity multipliers where infeasibility occurs
    """
    lower_counts = run_capacity_infeasibility_analysis(N, M, trials, lower, capacity_steps)

    # Ensure plot spans entire range of tested multipliers
    tested_multipliers = np.round(np.linspace(1.0, lower, capacity_steps), decimals=4)
    tested_multipliers = sorted(set(tested_multipliers))

    # Get counts
    lower_frequencies = [lower_counts.get(m, 0) for m in tested_multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_axisbelow(True)

    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.6, zorder=0)

    # Bar plot for infeasibility multipliers
    ax.bar(tested_multipliers, lower_frequencies, width=0.02, label="Decrease", color="tab:blue", zorder=2)

    ax.set_xlabel("Capacity Multiplier Where Model Becomes Infeasible", fontsize = 14)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize = 14)
    ax.set_title(f"Distribution of Capacity Infeasibility Thresholds (M={M}, N={N}, Trials={trials})", fontsize = 16)

    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(lower, 1)
    ax.legend()

    plt.savefig("figures/characteristics/distribution_capacity.png", dpi=300,
                bbox_inches='tight' )
    plt.show()



# ------------------------ DEMAND SENSITIVITY ANALYSIS ------------------------

def run_demand_infeasibility_analysis(N=9, M=5, trials=8, upper=2.0, demand_steps=30):
    """
    Finds the first higher multiplier for demand where the model becomes infeasible.
    """
    upper_infeasible = []

    # Count how many trials never become infeasible
    always_feasible = 0
    # Count infeasible base cases
    infeasible_base = 0

    for seed in range(trials):
        base_result = run(N, M, capacity_multiplier=1.0, random_fleet=True, seed=seed)
        if base_result["solution_x"] is None:
            infeasible_base += 1
            continue

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        found_infeasible = False

        for multiplier in np.round(np.linspace(1.0, upper, demand_steps), 4):
            # Skip the base case
            if multiplier > 1.0:
                # Increase demand while keeping node locations, fleet composition
                # and demand the same as the base case
                increased_demand = [int(d * multiplier) for d in fixed_demand]
                result = run(N, M, capacity_multiplier=1.0, random_fleet=False,
                             fleet_composition=fixed_fleet, seed=seed,
                             fixed_demand=increased_demand, fixed_nodes=fixed_nodes)

                if result["solution_x"] is None:
                    upper_infeasible.append(multiplier)
                    found_infeasible = True
                    break  # Stop at first infeasible case

        if not found_infeasible:
            always_feasible += 1

    print("\nFinal Counts for Demand Sensitivity:")
    print(f"Demand Infeasibility Multipliers: {Counter(upper_infeasible)}")
    print(f"Infeasible Base Cases: {infeasible_base}")
    print(f"Trials that never became infeasible: {always_feasible}")

    return Counter(upper_infeasible)



def plot_demand_infeasibility_analysis(N=9, M=5, demand_steps=30, upper=2.0, trials=8):
    """
    Plots the distribution of demand multipliers where infeasibility occurs (higher only).
    """
    upper_counts = run_demand_infeasibility_analysis(N, M, trials, upper, demand_steps)
    print(upper_counts)

    # Ensure plot spans entire range of tested multipliers
    tested_multipliers = np.round(np.linspace(1.0, upper, demand_steps), decimals=4)
    tested_multipliers = sorted(set(tested_multipliers))

    # Get counts
    upper_frequencies = [upper_counts.get(m, 0) for m in tested_multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_axisbelow(True)

    for tick in tested_multipliers:
        ax.axvline(x=tick, color='gray', linestyle='dashed', linewidth=0.5, alpha = 0.6, zorder=0)

    # Bar plot for infeasibility multipliers
    ax.bar(tested_multipliers, upper_frequencies, width=0.02, label="Increase", color="tab:orange", zorder=2)

    ax.set_xlabel("Demand Multiplier Where Model Becomes Infeasible", fontsize=14)
    ax.set_ylabel("Frequency (Number of Trials)", fontsize=14)
    ax.set_title(f"Distribution of Demand Infeasibility Thresholds (M={M}, N={N}, Trials={trials})", fontsize=16)

    ax.set_xticks(tested_multipliers)
    ax.set_xticklabels(tested_multipliers, rotation=45)

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(1, upper)
    ax.legend()
    plt.savefig("figures/characteristics/distribution_demand.png", dpi=300,
                bbox_inches='tight' )
    plt.show()


plot_speed_analysis(N=9, M=5, lower=0.6, upper=1.4, speed_steps=21, trials=100)

plot_capacity_analysis(N=9, M=5, lower=0.5, capacity_steps=26, trials=100)
plot_demand_infeasibility_analysis(N=9, M=5, upper=2, demand_steps=51, trials=100)

