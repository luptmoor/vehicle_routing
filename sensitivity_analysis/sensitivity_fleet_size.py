import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_case_2 import run

# List of fleet sizes
M_list = range(2, 10)

# N
N = 13


def sensitivity_fleet(fleet_composition=None, random_fleet=True, trials=1):
    """
    Runs the optimization for different fleet sizes and stores all trial results in a DataFrame.
    """
    results_list = []

    for M in M_list:
        # Get homogeneous fleet for M
        fleet_subset = fleet_composition[:M].copy() if fleet_composition else None

        trial_objective_values = []
        trial_total_distances = []
        trial_runtimes = []
        trial_results = []


        for seed in range(trials):
            result = run(N=N, M=M, random_fleet=random_fleet, fleet_composition=fleet_subset, seed=seed)

            if result and result["objective_value"] is not None:
                feasibility = "Yes"
                obj_value = result["objective_value"]
                total_dist = result["total_distance"]
                runtime = result["runtime"]
                trial_objective_values.append(obj_value)
                trial_total_distances.append(total_dist)
                trial_runtimes.append(runtime)
            else:
                feasibility = "No"
                obj_value = np.nan
                total_dist = np.nan
                runtime = np.nan

            trial_results.append({
                "Fleet Size": M,
                "Fleet Composition": tuple(result["fleet"]),
                "Seed": seed,
                "Objective Value": obj_value,
                "Total Distance": total_dist,
                "Runtime": runtime,
                "Feasible": feasibility
            })

        # Compute averages across trials
        avg_obj_value = np.nanmean(trial_objective_values) if trial_objective_values else np.nan
        avg_total_distance = np.nanmean(trial_total_distances) if trial_total_distances else np.nan
        avg_runtime = np.nanmean(trial_runtimes) if trial_runtimes else np.nan
        std_obj_value = np.nanstd(trial_objective_values) if len(trial_objective_values) > 1 else np.nan
        std_total_distance = np.nanstd(trial_total_distances) if len(trial_total_distances) > 1 else np.nan
        std_runtime = np.nanstd(trial_runtimes) if len(trial_runtimes) > 1 else np.nan

        for entry in trial_results:
            entry["Avg Objective Value"] = avg_obj_value
            entry["Avg Total Distance"] = avg_total_distance
            entry["Avg Runtime"] = avg_runtime
            entry["Std Objective Value"] = std_obj_value
            entry["Std Total Distance"] = std_total_distance
            entry["Std Runtime"] = std_runtime
            results_list.append(entry)

    df_results = pd.DataFrame(results_list)
    return df_results


def compute_sensitivity_results(trials=1, random_fleet=False):
    """Runs the sensitivity analysis for different fleet sizes and returns a compiled DataFrame."""
    fleet_compositions = {
        'Fixed-Wing (Type 1)': [0] * max(M_list),
        'Quadcopter (Type 2)': [1] * max(M_list),
        'Cargo Blimp (Type 3)': [2] * max(M_list)
    }

    results_df_list = []
    if random_fleet:
        results_df_list.append(sensitivity_fleet(fleet_composition=None, random_fleet=True, trials=trials))
    else:
        for type_name, fleet_comp in fleet_compositions.items():
            df = sensitivity_fleet(fleet_composition=fleet_comp, random_fleet=False, trials=trials)
            df["Fleet Composition"] = type_name
            results_df_list.append(df)

    final_results_df = pd.concat(results_df_list, ignore_index=True)

    # Save full data including infeasible cases
    if random_fleet:
        final_results_df.to_csv(f"sensitivity_analysis_random_{N}.csv", index=False)
    else:
        final_results_df.to_csv(f"sensitivity_analysis_fixed_{N}.csv", index=False)

    # Create a filtered dataframe excluding infeasible cases for plotting
    # df_filtered = final_results_df[final_results_df["Feasible"] == "Yes"]
    return final_results_df


def plot_metric_fleet(df_filtered, metric, ylabel, title, filename, random_fleet):
    """ Generalized function to plot different metrics vs Fleet Size """
    plt.figure(figsize=(8, 6))

    if random_fleet:
        avg_results = df_filtered.groupby("Fleet Size")[metric].mean()
        std_results = df_filtered.groupby("Fleet Size")[f"Std {metric}"].mean()
        plt.errorbar(avg_results.index, avg_results.values, yerr=std_results.values, fmt="o-", capsize=4)
    else:
        fleet_types = ["Fixed-Wing (Type 1)", "Quadcopter (Type 2)", "Cargo Blimp (Type 3)"]
        for fleet_type in fleet_types:
            subset = df_filtered[df_filtered["Fleet Composition"] == fleet_type]
            if not subset.empty:
                plt.errorbar(subset["Fleet Size"], subset[f"Avg {metric}"],
                             yerr=subset[f"Std {metric}"], fmt="o-", capsize=4, label=fleet_type)

    plt.xlabel("Fleet Size (M)")
    plt.ylabel(ylabel)
    plt.title(title)
    if not random_fleet:
        plt.legend()
    plt.grid()

    file_prefix = "fixed" if not random_fleet else "random"
    filename = filename.replace("Fleet_size", f"Fleet_size/{file_prefix}")

    plt.savefig(filename, dpi=300)

    plt.show()


def plot_results(df_filtered, random_fleet, N):
    """Plots all required metrics using filtered data."""
    if random_fleet:
        X = "random"
    else:
        X = "fixed"
    plot_metric_fleet(df_filtered, "Objective Value", "Objective Value (Cost)", "Averaged Objective Value vs. Fleet Size",
                      f"Figures/Fleet_size/{N}/objective_fleet_size_{X}.png", random_fleet)
    plot_metric_fleet(df_filtered, "Total Distance", "Total Travel Distance", "Averaged Total Distance vs. Fleet Size",
                      f"Figures/Fleet_size/{N}/distance_fleet_size_{X}.png", random_fleet)
    plot_metric_fleet(df_filtered, "Runtime", "Computation Runtime (s)", "Averaged Computation Time vs. Fleet Size ",
                      f"Figures/Fleet_size/{N}/runtime_fleet_size_{X}.png", random_fleet)


# Run analysis
results_fixed = compute_sensitivity_results(trials=10, random_fleet=False)
results_random = compute_sensitivity_results(trials=10, random_fleet=True)

# Plot for fleets of a single type
plot_results(results_fixed, random_fleet=False, N=N)

# Plot for random fleets (averaged over trials)
plot_results(results_random, random_fleet=True, N=N)