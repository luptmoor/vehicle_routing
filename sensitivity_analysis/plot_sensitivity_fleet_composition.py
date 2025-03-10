import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import seaborn as sns

# Generate heatmaps for fleet composition sensitivity
def plot_fleet_composition_heatmaps(df_results, metric="Objective Value"):
    """
    Generates heatmaps showing the impact of different fleet compositions on the given metric.
    """

    df_avg = df_results.groupby(["Fleet Size", "% Fixed-Wing"])[metric].mean().reset_index()
    pivot_fw = df_avg.pivot(index="Fleet Size", columns="% Fixed-Wing", values=metric)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_fw, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title(f"Fleet Size vs. Fixed-Wing % Impact on {metric}")
    plt.xlabel("Percentage of Fixed-Wing Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig(f"Figures/Fleet_composition/heatmap_fixedwing_{metric.replace(' ', '_')}.png")
    plt.show()

    df_avg = df_results.groupby(["Fleet Size", "% Quadcopter"])[metric].mean().reset_index()
    pivot_qc = df_avg.pivot(index="Fleet Size", columns="% Quadcopter", values=metric)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_qc, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title(f"Fleet Size vs. Quadcopter % Impact on {metric}")
    plt.xlabel("Percentage of Quadcopter Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig(f"Figures/Fleet_composition/heatmap_quadcopter_{metric.replace(' ', '_')}.png")
    plt.show()

    df_avg = df_results.groupby(["Fleet Size", "% Blimp"])[metric].mean().reset_index()
    pivot_blimp = df_avg.pivot(index="Fleet Size", columns="% Blimp", values=metric)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_blimp, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title(f"Fleet Size vs. Blimp % Impact on {metric}")
    plt.xlabel("Percentage of Blimp Drones")
    plt.ylabel("Fleet Size (M)")
    plt.savefig(f"Figures/Fleet_composition/heatmap_blimp_{metric.replace(' ', '_')}.png")
    plt.show()


def plot_fleet_composition_perM(df_results, metric="Objective Value"):
    """
    Plots the impact of fleet composition on the selected metric for each fleet size (M).
    - X-axis: Percentage of a specific drone type.
    - Y-axis: Drone types (Fixed-Wing, Quadcopter, Blimp).
    - Heatmap values: Average of the selected metric (Objective Value or Normalized Objective Value).
    """

    df_results["Fleet Size"] = df_results["Fleet Size"].astype(int)

    # Ensure percentage values are properly rounded
    df_results["% Fixed-Wing"] = df_results["% Fixed-Wing"].round(4)
    df_results["% Quadcopter"] = df_results["% Quadcopter"].round(4)
    df_results["% Blimp"] = df_results["% Blimp"].round(4)

    fleet_sizes = sorted(df_results["Fleet Size"].unique())

    for M in fleet_sizes:
        df_M = df_results[df_results["Fleet Size"] == M].copy()

        # Compute averages for the selected metric across compositions with the same % distribution
        df_M = df_M.groupby(["% Fixed-Wing", "% Quadcopter", "% Blimp"]).agg({metric: "mean"}).reset_index()

        # Create a dataframe for plotting
        heatmap_data = pd.DataFrame(index=["Fixed-Wing", "Quadcopter", "Blimp"])

        for drone_type, percentage_column in zip(["Fixed-Wing", "Quadcopter", "Blimp"],
                                                 ["% Fixed-Wing", "% Quadcopter", "% Blimp"]):

            valid_percentages = sorted(df_M[percentage_column].unique())  # Get applicable % values
            avg_values = [df_M[df_M[percentage_column] == p][metric].mean() for p in valid_percentages]
            heatmap_data.loc[drone_type, valid_percentages] = avg_values  # Fill data

        # Plot heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".4f")

        plt.title(f"Impact of Drone Type % on {metric} for Fleet Size M={M}")
        plt.xlabel("Percentage of Drone Type")
        plt.ylabel("Drone Type")
        plt.xticks(rotation=45)

        plt.savefig(f"Figures/Fleet_composition/fleet_composition_perM_M{M}_{metric.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.show()



df_results_normalized = pd.read_csv('fleet_composition_results_normalized.csv')
print(df_results_normalized.columns)
# Generate heatmaps
# plot_fleet_composition_heatmaps(df_results_normalized)
plot_fleet_composition_perM(df_results_normalized, metric="Average Normalized Objective Value")

