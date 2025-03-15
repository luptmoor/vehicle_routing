import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sample_case_2 import run

M = 5
N = 9
TRIALS = 100

# ------------------------ SPEED SENSITIVITY ANALYSIS ------------------------
def compute_speed_sensitivity(N=9, M=5, trials=100, lower=0.94, upper=1.06, steps=4):
    """
    Computes the percentage changes in objective value and normalized objective value for speed sensitivity.
    Records changes for all cases, not just when the optimal solution changes.
    """
    multipliers_decrease = np.linspace(1.0, lower, steps)
    multipliers_increase = np.linspace(1.0, upper, steps)
    results = []

    for seed in range(trials):
        # Run base case to fix fleet, demand, and node locations
        base_result = run(N=N, M=M, speed_multiplier=1.0, random_fleet=True, seed=seed)

        if base_result["objective_value"] is None:
            continue

        base_obj = base_result["objective_value"]
        base_norm_obj = base_result["normalized_objective_value"]
        base_edges = base_result["solution_x"]

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        # Iterate over decreasing multipliers
        for multiplier in multipliers_decrease:
            if multiplier == 1:
                continue

            # Run for speed multiplier, fixing fleet, demand and nodes
            result = run(
                N=N, M=M,
                speed_multiplier=multiplier,
                random_fleet=False, fleet_composition=fixed_fleet,
                fixed_demand=fixed_demand, fixed_nodes=fixed_nodes,
                seed=seed
            )

            # Skip infeasible results
            if result["objective_value"] is None:
                continue

            # Record changes for all cases (not just when the solution changes)
            solution_changed = result["solution_x"] != base_edges  # Boolean check

            results.append({
                "Trial": seed,
                "Multiplier": multiplier,
                "Change Type": "Decrease",
                "Base Obj": base_obj,
                "New Obj": result["objective_value"],
                "Objective Value Change %": (result["objective_value"] - base_obj) / base_obj * 100,
                "Objective Value Change (Abs)": result["objective_value"] - base_obj,
                "Normalized Objective Value Change %": (result["normalized_objective_value"] - base_norm_obj) / base_norm_obj * 100,
                "Normalized Objective Value Change (Abs)": result["normalized_objective_value"] - base_norm_obj,
                "Solution Changed": solution_changed,
                "Fleet Composition": fixed_fleet,
                "Base Route": base_edges,
                "Changed Route": result["solution_x"]
            })

        # Iterate over increasing multipliers
        for multiplier in multipliers_increase:
            if multiplier == 1:
                continue

            # Run for speed multiplier, fixing fleet, demand and nodes
            result = run(
                N=N, M=M,
                speed_multiplier=multiplier,
                random_fleet=False, fleet_composition=fixed_fleet,
                fixed_demand=fixed_demand, fixed_nodes=fixed_nodes,
                seed=seed
            )

            # Skip infeasible results
            if result["objective_value"] is None:
                continue

            # Record changes for all cases
            solution_changed = result["solution_x"] != base_edges

            results.append({
                "Trial": seed,
                "Multiplier": multiplier,
                "Change Type": "Increase",
                "Base Obj": base_obj,
                "New Obj": result["objective_value"],
                "Objective Value Change %": (result["objective_value"] - base_obj) / base_obj * 100,
                "Objective Value Change (Abs)": result["objective_value"] - base_obj,
                "Normalized Objective Value Change %": (result["normalized_objective_value"] - base_norm_obj) / base_norm_obj * 100,
                "Normalized Objective Value Change (Abs)": result["normalized_objective_value"] - base_norm_obj,
                "Solution Changed": solution_changed,
                "Fleet Composition": fixed_fleet,
                "Base Route": base_edges,
                "Changed Route": result["solution_x"]
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv("results/characteristics/speed_sensitivity_results.csv", index=False)
    return df_results


def plot_speed_sensitivity(df):
    """
    Plots (normalized) objective value change for speed multipliers.
    """

    # Ensure correct ordering of multipliers
    unique_multipliers = sorted(df["Multiplier"].unique())

    color_map = {"Decrease": "blue", "Increase": "orange"}

    # Objective Value Change
    plt.figure(figsize=(8, 5))
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

    # Normalized Objective Value Change
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="Multiplier", y="Normalized Objective Value Change %", hue="Change Type",
        data=df, palette=color_map, s=100  # Increase dot size
    )
    plt.xticks(unique_multipliers)
    plt.xlabel("Speed Multiplier", fontsize=14)
    plt.ylabel("(Normalized) Objective Value Change (%)", fontsize=14)
    plt.title(f"Normalized Objective Change vs Speed Multiplier (N={N}, M={M})", fontsize=16)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.legend(title="Change Type")
    plt.savefig(f"figures/characteristics/speed_normalized_objective_change_scatter_N{N}_M{M}.png", dpi=300,
                bbox_inches='tight' )
    plt.show()

# ------------------------ CAPACITY AND DEMAND SENSITIVITY ANALYSIS ------------------------
def compute_normalized_obj_value_change(
        N=9, M=5, trials=100, lower=1, upper=1, steps=21, parameter="capacity"):
    """
    Computes (normalized) objective value change before infeasibility.
    - If `parameter="capacity"`, it tests lower multipliers.
    - If `parameter="demand"`, it tests upper multipliers.
    """

    # Select correct multipliers based on parameter
    multipliers = np.linspace(1.0, lower, steps) if parameter == "capacity" else np.linspace(1.0, upper, steps)
    results = []
    always_feasible = 0
    infeasible_base = 0

    for seed in range(trials):
        # Run base case
        base_result = run(N=N, M=M, speed_multiplier=1.0, capacity_multiplier=1.0, random_fleet=True, seed=seed)

        if base_result is None:
            infeasible_base += 1
            continue  # Skip infeasible trials

        base_obj = base_result["objective_value"]
        base_norm_obj = base_result["normalized_objective_value"]
        base_demand = base_result["demand_list"]
        base_edges = base_result["solution_x"]

        fixed_fleet = base_result["fleet"]
        fixed_demand = base_result["demand_list"]
        fixed_nodes = base_result["nodes"]

        found_feasible = False

        # Iterate over multipliers
        for multiplier in multipliers:
            if parameter == "capacity":
                result = run(N=N, M=M, capacity_multiplier=multiplier, random_fleet=False,
                             fleet_composition=fixed_fleet, fixed_demand=fixed_demand, fixed_nodes=fixed_nodes,
                                seed=seed)
            elif parameter == "demand":
                increased_demand = [d * multiplier for d in fixed_demand]
                result = run(N=N, M=M, capacity_multiplier=1.0, random_fleet=False,
                             fleet_composition=fixed_fleet, fixed_demand=increased_demand, fixed_nodes=fixed_nodes,
                                seed=seed)

            if result["solution_x"] is None:
                found_infeasible = True
                break  # Stop when infeasible

            # Compute changes
            obj_change = result["objective_value"] - base_obj
            norm_obj_change = result["normalized_objective_value"] - base_norm_obj
            obj_change_pct = (obj_change / base_obj) * 100
            norm_obj_change_pct = (norm_obj_change / base_norm_obj) * 100

            results.append({
                "Multiplier": multiplier,
                "Trial": seed,
                "Objective Change": obj_change,
                "Normalized Objective Change": norm_obj_change,
                "Objective Change (%)": obj_change_pct,
                "Normalized Objective Change (%)": norm_obj_change_pct,
                "Fleet Composition": result["fleet"],
                "Base Route": base_edges,
                "New Route": result["solution_x"],
            })

    if not found_infeasible:
        always_feasible += 1

    pd.DataFrame(results).to_csv(f"results/characteristics/sensitivity_characteristics_{parameter}.csv", index=False)
    return pd.DataFrame(results)

def plot_normalized_obj_value(df, trials, parameter="capacity"):
    """
    Plots the normalized objective value change before infeasibility for capacity or demand.
    Includes:
    - Line Plot with Mean and Standard Deviation
    - Scatter Plot
    - Box Plot
    """

    # Get unique multipliers, sorted for consistent x-axis
    unique_multipliers = np.sort(df["Multiplier"].unique())

    if parameter == "capacity":
        min = np.min(unique_multipliers)
        max = 1
        xtick_fontsize = 10
        rotation = 45
        color = "tab:blue"
    elif parameter == "demand":
        min = 1
        max = np.max(unique_multipliers)
        xtick_fontsize = 8
        rotation = 60
        color = "tab:orange"

    ### Line Plot with Mean and Standard Deviation ###
    plt.figure(figsize=(8, 6))
    multipliers = df.groupby("Multiplier")["Normalized Objective Change (%)"]
    means = multipliers.mean()
    stds = multipliers.std()

    plt.plot(means.index, means, marker='o', label="Mean Change", color = color)
    plt.fill_between(means.index, means - stds, means + stds, alpha=0.3, label="±1 Std Dev",color=color)

    plt.xlabel(f"{parameter.capitalize()} Multiplier", fontsize = 14)
    plt.ylabel("Normalized Objective Value Change (%)", fontsize = 14)
    plt.title(f"Mean and Spread of Normalized Objective Value Change vs \n"
              f"{parameter.capitalize()} Multiplier (N={N}, M={M}, trials={trials})", fontsize = 16)
    plt.xticks(unique_multipliers, rotation=rotation, fontsize=xtick_fontsize)  # Rotate x-ticks
    plt.xlim(min, max)
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.savefig(f"figures/characteristics/{parameter}_obj_value_line_N{N}_M{M}.png", dpi=300)
    plt.show()

    ### Scatter Plot ###
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Multiplier"], df["Normalized Objective Change (%)"], alpha=0.7, color = color,
                zorder = 0)

    plt.xlabel(f"{parameter.capitalize()} Multiplier", fontsize = 14)
    plt.ylabel("Normalized Objective Change (%)", fontsize =14 )
    plt.title(f"Scatter of Normalized Objective Value Change vs {parameter.capitalize()}\n"
              f"{N}, M={M}, trials={trials})", fontsize =16)
    plt.xticks(unique_multipliers, rotation=rotation, fontsize=xtick_fontsize)  # Rotate x-ticks
    plt.xlim(min, max)
    plt.grid(alpha = 0.5, zorder = 1)
    plt.savefig(f"figures/characteristics/{parameter}_obj_value_scatter_N{N}_M{M}.png", dpi=300)
    plt.show()

    ### Box Plot ###
    plt.figure(figsize=(8, 6))
    df.boxplot(column="Normalized Objective Change (%)", by="Multiplier", grid=False, zorder = 0)

    plt.xlabel(f"{parameter.capitalize()} Multiplier", fontsize=14)
    plt.ylabel("Normalized Objective Value Change (%)", fontsize = 14)
    plt.title(f"Distribution of Normalized Objective Value Change vs {parameter.capitalize()} Multiplier \n"
              f"(N={N}, M={M}, trials={trials})", fontsize = 16)
    plt.xticks(rotation=45)
    plt.suptitle("")
    plt.grid(zorder = 1)
    plt.savefig(f"figures/characteristics/{parameter}_obj_value_boxplot_N{N}_M{M}.png", dpi=300)
    plt.show()


df_capacity = compute_normalized_obj_value_change(lower=0.5, steps = 26, trials = 100, parameter="capacity")
plot_normalized_obj_value(df_capacity, trials = 100, parameter="capacity")

df_demand = compute_normalized_obj_value_change(N = 9, M = 5, upper = 2, steps = 51, trials = 100, parameter="demand")
plot_normalized_obj_value(df_demand, trials = 100, parameter="demand")

# df_speed = compute_speed_sensitivity(N=9, M=5, trials=100, lower=0.92, upper=1.08, steps=5)
# plot_speed_sensitivity(df_speed)


