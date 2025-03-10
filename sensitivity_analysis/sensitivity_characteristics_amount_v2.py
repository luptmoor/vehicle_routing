
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sample_case_2 import run

M = 5
N = 9
TRIALS = 100

def compute_distance_per_drone_type(route_set, distance_matrix, fleet):
    """Computes total travel distance per drone type."""
    drone_type_distances = {}

    for vehicle, start, end in route_set:
        drone_type = fleet[vehicle]  # Get drone type based on vehicle index
        distance = distance_matrix[start, end]

        if drone_type not in drone_type_distances:
            drone_type_distances[drone_type] = 0.0
        drone_type_distances[drone_type] += distance  # Accumulate distance for that drone type

    return drone_type_distances


def compute_speed_sensitivity(N=9, M=5, trials=100, lower=0.94, upper=1.06, steps=4):
    """
    Computes the percentage changes in objective value for speed sensitivity.
    - Filters out cases where the solution change is just a swap between identical drones.
    - Stores the base and changed routes for comparison.
    """
    multipliers_decrease = np.round(np.linspace(1.0, lower, steps), decimals=4)
    multipliers_increase = np.round(np.linspace(1.0, upper, steps), decimals=4)
    results = []

    for seed in range(trials):
        np.random.seed(seed)

        # Run base case to fix fleet, demand, and customer locations
        base_result = run(N=N, M=M, speed_multiplier=1.0, random_fleet=True, seed=seed)

        if base_result is None or base_result["objective_value"] is None:
            continue  # Skip infeasible base cases

        base_obj = base_result["objective_value"]
        base_norm_obj = base_result["normalized_objective_value"]
        base_edges = base_result["solution_x"]  # Store optimal solution structure
        base_time_per_type = compute_distance_per_drone_type(base_edges, base_result["distance_matrix"], base_result["fleet"])
        print("base time type")
        print(base_time_per_type)
        # Fix fleet, demand, and nodes for controlled sensitivity analysis
        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        # Iterate over decreasing multipliers
        for multiplier in multipliers_decrease:
            result = run(
                N=N, M=M, speed_multiplier=multiplier, random_fleet=False,
                fleet_composition=fixed_fleet, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes, seed=seed
            )

            if result["objective_value"] is None:
                continue  # Skip infeasible results

            new_time_per_type = compute_distance_per_drone_type(result["solution_x"], result["distance_matrix"], result["fleet"])
            print("new time per type")
            print(new_time_per_type)
            # Only record real routing changes (not just drone swaps)
            if base_time_per_type != new_time_per_type:
                results.append({
                    "Trial": seed,
                    "Multiplier": multiplier,
                    "Change Type": "Decrease",
                    "Base Obj": base_obj,
                    "New Obj": result["objective_value"],
                    "Objective Value Change %": (result["objective_value"] - base_obj) / base_obj * 100,
                    "Normalized Objective Value Change %": (result["normalized_objective_value"] - base_norm_obj) / base_norm_obj * 100,
                    "Fleet Composition": fixed_fleet,
                    "Base Route": base_edges,
                    "Changed Route": result["solution_x"],
                    "Base time type": base_time_per_type,
                    "New time type": new_time_per_type
                })
                break  # Stop checking once a true optimal solution change is detected

        # Iterate over increasing multipliers
        for multiplier in multipliers_increase:
            result = run(
                N=N, M=M, speed_multiplier=multiplier, random_fleet=False,
                fleet_composition=fixed_fleet, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes, seed=seed
            )

            if result["objective_value"] is None:
                continue  # Skip infeasible results

            new_time_per_type = compute_distance_per_drone_type(result["solution_x"], result["distance_matrix"], result["fleet"])
            print("new time per type")
            print(new_time_per_type)
            # Only record real routing changes
            if base_time_per_type != new_time_per_type:
                results.append({
                    "Trial": seed,
                    "Multiplier": multiplier,
                    "Change Type": "Increase",
                    "Base Obj": base_obj,
                    "New Obj": result["objective_value"],
                    "Objective Value Change %": (result["objective_value"] - base_obj) / base_obj * 100,
                    "Normalized Objective Value Change %": (result["normalized_objective_value"] - base_norm_obj) / base_norm_obj * 100,
                    "Fleet Composition": fixed_fleet,
                    "Base Route": base_edges,
                    "Changed Route": result["solution_x"],
                    "Base time type": base_time_per_type,
                    "New time type": new_time_per_type
                })
                break  # Stop checking once a true optimal solution change is detected

    df_results = pd.DataFrame(results)
    df_results.to_csv("speed_sensitivity_results_filtered.csv", index=False)
    return df_results


def plot_speed_sensitivity(df):
    """Plots speed sensitivity using scatter plots with multipliers on the x-axis."""

    # Ensure correct ordering of multipliers
    unique_multipliers = sorted(df["Multiplier"].unique())

    # Define colors based on change type
    color_map = {"Decrease": "blue", "Increase": "orange"}

    # Objective Value Change
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="Multiplier", y="Objective Value Change %", hue="Change Type",
        data=df, palette=color_map, s=100  # Increase dot size
    )
    plt.xticks(unique_multipliers)  # Ensure all multipliers appear on the x-axis
    plt.xlabel("Speed Multiplier", fontsize=14)
    plt.ylabel("Objective Value Change (%)", fontsize=14)
    plt.title("Objective Value Change vs Speed Multiplier", fontsize=16)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.legend(title="Change Type")
    plt.show()

    # Normalized Objective Value Change (Saved)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="Multiplier", y="Normalized Objective Value Change %", hue="Change Type",
        data=df, palette=color_map, s=100  # Increase dot size
    )
    plt.xticks(unique_multipliers)
    plt.xlabel("Speed Multiplier", fontsize=14)
    plt.ylabel("(Normalized) Objective Value Change (%)", fontsize=14)
    plt.title("Normalized Objective Change vs Speed Multiplier", fontsize=16)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.legend(title="Change Type")
    plt.savefig("Figures/speed_normalized_objective_change_scatter.png", dpi=300,
                bbox_inches='tight' )
    plt.show()

def compute_sensitivity_changes(N=9, M=5, trials=100, lower=0.4, upper=1.6, steps=21, parameter="speed"):
    """
    Computes percentage changes in objective value and total distance for trials where the solution changed.
    - parameter: "speed" or "capacity" to indicate what is being analyzed.
    - Returns a dataframe for boxplot visualization.
    """
    # Select correct multiplier type
    lower_multipliers = np.round(np.linspace(1.0, lower, steps), decimals=4)
    upper_multipliers = np.round(np.linspace(1.0, upper, steps), decimals=4)

    results = []  # Store changes for plotting

    for seed in range(trials):
        np.random.seed(seed)
        base_result = run(N=N, M=M, speed_multiplier=1.0, capacity_multiplier=1.0, random_drones=True, seed=seed)

        if base_result is None:
            continue  # Skip infeasible trials

        base_obj = base_result["objective_value"]
        base_dist = base_result["total_distance"]
        base_edges = base_result["solution_x"]

        # Iterate over lower multipliers
        for multiplier in lower_multipliers:
            result = run(
                N=N, M=M, speed_multiplier=multiplier if parameter == "speed" else 1.0,
                capacity_multiplier=multiplier if parameter == "capacity" else 1.0,
                random_drones=True, seed=seed
            )
            if result and result["solution_x"] != base_edges:
                obj_change = (result["objective_value"] - base_obj) / base_obj * 100
                dist_change = (result["total_distance"] - base_dist) / base_dist * 100
                results.append({"Multiplier": multiplier, "Change Type": "Decrease", "Objective % Change": obj_change, "Distance % Change": dist_change})
                break  # Stop at first change

        # Iterate over upper multipliers
        for multiplier in upper_multipliers:
            result = run(
                N=N, M=M, speed_multiplier=multiplier if parameter == "speed" else 1.0,
                capacity_multiplier=multiplier if parameter == "capacity" else 1.0,
                random_drones=True, seed=seed
            )
            if result and result["solution_x"] != base_edges:
                obj_change = (result["objective_value"] - base_obj) / base_obj * 100
                dist_change = (result["total_distance"] - base_dist) / base_dist * 100
                results.append({"Multiplier": multiplier, "Change Type": "Increase", "Objective % Change": obj_change, "Distance % Change": dist_change})
                break  # Stop at first change

    return pd.DataFrame(results)

def plot_capacity_demand_sensitivity(df, parameter="capacity"):
    """Plots capacity/demand sensitivity using boxplots with multipliers on the x-axis."""

    df["Multiplier"] = df["Multiplier"].astype(str)  # Convert multipliers to categorical

    # Objective Value Change
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Multiplier", y="Objective Value Change %", hue="Change Type", data=df,
                palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier", fontsize=14)
    plt.ylabel("Objective Value Change (%)", fontsize=14)
    plt.title(f"Objective Value Change vs {parameter.capitalize()} Multiplier", fontsize=16)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()

    # Normalized Objective Value Change (Saved)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Multiplier", y="Normalized Objective Value Change %", hue="Change Type", data=df,
                palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier", fontsize=14)
    plt.ylabel("Normalized Objective Value Change (%)", fontsize=14)
    plt.title(f"Normalized Objective Change vs {parameter.capitalize()} Multiplier", fontsize=16)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(f"Figures/{parameter}_normalized_objective_change_boxplot.png", dpi=300)
    plt.show()

df_speed = compute_speed_sensitivity(N=9, M=5, trials=100, lower=0.92, upper=1.08, steps=5)
plot_speed_sensitivity(df_speed)

# # Run Capacity Sensitivity Analysis (Only Lower Multipliers)
# df_capacity = compute_sensitivity_changes(N=9, M=5, trials=100, lower=0.6, upper=1.0, steps=3, parameter="capacity")
# plot_capacity_demand_sensitivity(df_capacity, parameter="capacity")
#
# # Run Demand Sensitivity Analysis (Only Upper Multipliers)
# df_demand = compute_sensitivity_changes(N=9, M=5, trials=100, lower=1.0, upper=1.4, steps=3, parameter="demand")
# plot_capacity_demand_sensitivity(df_demand, parameter="demand")
