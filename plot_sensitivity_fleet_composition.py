import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import seaborn as sns

# Generate heatmaps for fleet composition sensitivity
def plot_fleet_composition_heatmaps(df_results):
    """
    Generates heatmaps showing the impact of different fleet compositions on the objective value.
    Averages objective values over compositions that have the same percentage of a given drone type.
    """

    # Ensure compositions with the same % type but different mixes are averaged
    df_avg = df_results.groupby(["fleet_size", "% Fixed-Wing"])["objective_value"].mean().reset_index()
    pivot_fw = df_avg.pivot(index="fleet_size", columns="% Fixed-Wing", values="objective_value")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_fw, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title("Fleet Size vs. Fixed-Wing % Impact on Objective Value (Averaged)")
    plt.xlabel("Percentage of Fixed-Wing Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig("Figures/heatmap_fixedwing.png")
    plt.show()

    df_avg = df_results.groupby(["fleet_size", "% Quadcopter"])["objective_value"].mean().reset_index()
    pivot_qc = df_avg.pivot(index="fleet_size", columns="% Quadcopter", values="objective_value")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_qc, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title("Fleet Size vs. Quadcopter % Impact on Objective Value (Averaged)")
    plt.xlabel("Percentage of Quadcopter Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig("Figures/heatmap_quadcopter.png")
    plt.show()

    df_avg = df_results.groupby(["fleet_size", "% Blimp"])["objective_value"].mean().reset_index()
    pivot_blimp = df_avg.pivot(index="fleet_size", columns="% Blimp", values="objective_value")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_blimp, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title("Fleet Size vs. Blimp % Impact on Objective Value (Averaged)")
    plt.xlabel("Percentage of Blimp Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig("Figures/heatmap_blimp.png")

    plt.show()


def plot_fleet_composition_perM(df_results):
    """
    Plots the impact of fleet composition on the objective value for each fleet size (M).
    - X-axis: Percentage of a specific drone type (only applicable values for each M).
    - Y-axis: The three drone types.
    - Heatmap values: Average objective value.
    """

    df_results["fleet_size"] = df_results["fleet_size"].astype(int)

    # Round percentages properly to avoid duplicate pivot errors
    df_results["% Fixed-Wing"] = df_results["% Fixed-Wing"].round(1)
    df_results["% Quadcopter"] = df_results["% Quadcopter"].round(1)
    df_results["% Blimp"] = df_results["% Blimp"].round(1)

    fleet_sizes = sorted(df_results["fleet_size"].unique())

    for M in fleet_sizes:
        df_M = df_results[df_results["fleet_size"] == M].copy()

        # Compute averages across compositions with the same % distribution
        df_M = df_M.groupby(["% Fixed-Wing", "% Quadcopter", "% Blimp"]).agg({"objective_value": "mean"}).reset_index()

        # Create a dataframe for plotting
        heatmap_data = pd.DataFrame(index=["Fixed-Wing", "Quadcopter", "Blimp"])

        for drone_type, percentage_column in zip(["Fixed-Wing", "Quadcopter", "Blimp"],
                                                 ["% Fixed-Wing", "% Quadcopter", "% Blimp"]):

            valid_percentages = sorted(df_M[percentage_column].unique())  # Get applicable % values
            avg_values = [df_M[df_M[percentage_column] == p]["objective_value"].mean() for p in valid_percentages]
            heatmap_data.loc[drone_type, valid_percentages] = avg_values  # Fill data

        # Plot heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".1f")

        plt.title(f"Impact of Drone Type % on Objective Value for Fleet Size M={M}")
        plt.xlabel("Percentage of Drone Type")
        plt.ylabel("Drone Type")
        plt.xticks(rotation=45)

        plt.savefig(f"Figures/fleet_composition_perM_M{M}.png", dpi=300, bbox_inches="tight")
        plt.show()



df_results = pd.read_csv('fleet_composition_results.csv')
print(df_results)
# Generate heatmaps
plot_fleet_composition_heatmaps(df_results)
plot_fleet_composition_perM(df_results)