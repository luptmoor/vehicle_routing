import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sample_case_2 import run

M = 5
N = 9
TRIALS = 100
LOWER = 0.91
UPPER = 1.09
STEPS = 4


def compute_sensitivity_changes(N=9, M=5, trials=100, lower=0.91, upper=1.09, steps=4, parameter="speed"):
    """
    Computes percentage changes in objective value and total distance for trials where the solution changed.
    - `parameter`: "speed" or "capacity" to indicate what is being analyzed.
    - Returns a dataframe for visualization.
    """
    lower_multipliers = np.round(np.linspace(1.0, lower, steps), decimals=4)
    upper_multipliers = np.round(np.linspace(1.0, upper, steps), decimals=4)

    results = []

    for seed in range(trials):
        np.random.seed(seed)
        base_result = run(N=N, M=M, speed_multiplier=1.0, capacity_multiplier=1.0, random_fleet=True, seed=seed)

        if base_result is None or base_result["objective_value"] is None:
            continue

        base_obj = base_result["objective_value"]
        base_dist = base_result["total_distance"]
        base_edges = base_result["solution_x"]
        fleet = tuple(base_result["fleet"])

        # Iterate over multipliers
        for change_type, multipliers in zip(["Decrease", "Increase"], [lower_multipliers, upper_multipliers]):
            for multiplier in multipliers:
                result = run(
                    N=N, M=M,
                    speed_multiplier=multiplier if parameter == "speed" else 1.0,
                    capacity_multiplier=multiplier if parameter == "capacity" else 1.0,
                    random_fleet=True, seed=seed
                )

                if result["objective_value"] and result["solution_x"] != base_edges:
                    obj_change_abs = result["objective_value"] - base_obj
                    obj_change_pct = (obj_change_abs / base_obj) * 100
                    dist_change_abs = result["total_distance"] - base_dist
                    dist_change_pct = (dist_change_abs / base_dist) * 100

                    results.append({
                        "Trial": seed,
                        "Multiplier": multiplier,
                        "Change Type": change_type,
                        "Objective Value Change %": obj_change_pct,
                        "Objective Value Change (Abs)": obj_change_abs,
                        "Total Distance Change %": dist_change_pct,
                        "Total Distance Change (Abs)": dist_change_abs,
                        "Fleet Composition": fleet
                    })
                    break  # Stop at first detected change

    return pd.DataFrame(results)


def plot_sensitivity_boxplots(df, parameter="speed"):
    """
    Generates boxplots for objective value and distance changes due to speed or capacity sensitivity.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Multiplier", y="Objective Value Change %", hue="Change Type", data=df,
                palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier")
    plt.ylabel("Objective Value Change (%)")
    plt.title(f"Objective Value Change vs {parameter.capitalize()} Multiplier")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(f"Figures/{parameter}_objective_change_boxplot.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Multiplier", y="Total Distance Change %", hue="Change Type", data=df,
                palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier")
    plt.ylabel("Total Distance Change (%)")
    plt.title(f"Total Distance Change vs {parameter.capitalize()} Multiplier")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(f"Figures/{parameter}_distance_change_boxplot.png", dpi=300)
    plt.show()


def plot_sensitivity_scatter(df, parameter="speed"):
    """
    Generates scatter plots for objective value and distance changes due to speed or capacity sensitivity.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Multiplier", y="Objective Value Change %", hue="Change Type", data=df,
                    palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier")
    plt.ylabel("Objective Value Change (%)")
    plt.title(f"Objective Value Change vs {parameter.capitalize()} Multiplier")
    plt.xticks(sorted(df["Multiplier"].unique()))
    plt.grid(alpha=0.7)
    plt.savefig(f"Figures/{parameter}_objective_change_scatter.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Multiplier", y="Total Distance Change %", hue="Change Type", data=df,
                    palette={"Decrease": "blue", "Increase": "red"})
    plt.xlabel(f"{parameter.capitalize()} Multiplier")
    plt.ylabel("Total Distance Change (%)")
    plt.title(f"Total Distance Change vs {parameter.capitalize()} Multiplier")
    plt.xticks(sorted(df["Multiplier"].unique()))
    if parameter == "speed":
        plt.ylim(-1, 1)
    plt.grid(alpha=0.7)
    plt.savefig(f"Figures/{parameter}_distance_change_scatter.png", dpi=300)
    plt.show()


# Compute changes for speed
df_speed = compute_sensitivity_changes(N=9, M=5, trials=TRIALS, lower=LOWER, upper=UPPER, steps=STEPS,
                                       parameter="speed")
df_speed.to_csv("Figures/speed_sensitivity_results.csv", index=False)
plot_sensitivity_boxplots(df_speed, parameter="speed")
plot_sensitivity_scatter(df_speed, parameter="speed")

# Compute changes for capacity
df_capacity = compute_sensitivity_changes(N=9, M=5, trials=TRIALS, lower=LOWER, upper=UPPER, steps=STEPS,
                                          parameter="capacity")
df_capacity.to_csv("Figures/capacity_sensitivity_results.csv", index=False)
plot_sensitivity_boxplots(df_capacity, parameter="capacity")
plot_sensitivity_scatter(df_capacity, parameter="capacity")
