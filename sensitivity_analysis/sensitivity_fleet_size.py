import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sample_case_2 import run

# Range of fleet sizes
M_list = range(2, 10)

# List of customer counts for which to produce the plots
N_list = range(9, 14)

def sensitivity_fleetsize(N, fleet_composition=None, random_fleet=True, trials=1):
    """
    Runs the optimization for different fleet sizes and stores all trial results in a DataFrame.
    """
    results_list = []
    infeasibility_summary = {}

    for M in M_list:
        # Skip instances that are certainly infeasible
        if M > N:
            continue

        # Create subset for homogeneous fleets
        fleet_subset = fleet_composition[:M].copy() if fleet_composition else None


        infeasible_trials = []

        trial_objective_values = []
        trial_normalized_values = []
        trial_total_distances = []
        trial_runtimes = []
        trial_results = []

        # Run trials
        for seed in range(trials):
            result = run(N=N, M=M, random_fleet=random_fleet, fleet_composition=fleet_subset, seed=seed)

            # If trial is feasible extract metrics
            if result["objective_value"] is not None:
                feasibility = "Yes"
                obj_value = result["objective_value"]
                total_dist = result["total_distance"]
                runtime = result["runtime"]
                total_demand = result["total_demand"]
                norm_value = obj_value / total_demand

                trial_objective_values.append(obj_value)
                trial_normalized_values.append(norm_value)
                trial_total_distances.append(total_dist)
                trial_runtimes.append(runtime)
            else:
                feasibility = "No"
                obj_value = np.nan
                norm_value = np.nan
                total_dist = np.nan
                runtime = np.nan
                infeasible_trials.append(seed)

            # Store the trial result in a list of dictionaries
            trial_results.append({
                "Nodes": N,
                "Fleet Size": M,
                "Fleet Composition": tuple(result["fleet"]),
                "Seed": seed,
                "Objective Value": obj_value,
                "Normalized Objective Value": norm_value,
                "Total Distance": total_dist,
                "Runtime": runtime,
                "Feasible": feasibility
            })

        # Store infeasibility summary
        infeasibility_summary[(N, M, tuple(result["fleet"]))] = (infeasible_trials, len(infeasible_trials) / trials)

        # Compute average and standard deviation metrics across trials
        avg_obj_value = np.nanmean(trial_objective_values) if trial_objective_values else np.nan
        avg_norm_value = np.nanmean(trial_normalized_values) if trial_normalized_values else np.nan
        avg_total_distance = np.nanmean(trial_total_distances) if trial_total_distances else np.nan
        avg_runtime = np.nanmean(trial_runtimes) if trial_runtimes else np.nan

        std_obj_value = np.nanstd(trial_objective_values) if len(trial_objective_values) > 1 else np.nan
        std_norm_value = np.nanstd(trial_normalized_values) if len(trial_normalized_values) > 1 else np.nan
        std_total_distance = np.nanstd(trial_total_distances) if len(trial_total_distances) > 1 else np.nan
        std_runtime = np.nanstd(trial_runtimes) if len(trial_runtimes) > 1 else np.nan

        # Append aggregated results to the trial entries
        for entry in trial_results:
            entry["Avg Objective Value"] = avg_obj_value
            entry["Std Objective Value"] = std_obj_value
            entry["Avg Normalized Objective Value"] = avg_norm_value
            entry["Std Normalized Objective Value"] = std_norm_value
            entry["Avg Total Distance"] = avg_total_distance
            entry["Std Total Distance"] = std_total_distance
            entry["Avg Runtime"] = avg_runtime
            entry["Std Runtime"] = std_runtime
            results_list.append(entry)

    # df_results consists of all the trials for a specific N for the range of M
    df_results = pd.DataFrame(results_list)
    return df_results, infeasibility_summary


def compute_sensitivity_fleet(N, trials=1, random_fleet=False):
    """
    Runs sensitivity analysis for different fleet compositions (fixed or random).
    """

    # Homogeneous fleet compositions
    fleet_compositions = {
        'Fixed-Wing (Type 1)': [0] * max(M_list),
        'Quadcopter (Type 2)': [1] * max(M_list),
        'Cargo Blimp (Type 3)': [2] * max(M_list)
    }

    # List for the outputs of sensitivity_fleet()
    results_df_list = []

    infeasibility_data = {}

    # Run sensitivity analysis with a random fleet composition
    if random_fleet:
        df, infeasibility = sensitivity_fleetsize(N=N, fleet_composition=None, random_fleet=True, trials=trials)
        results_df_list.append(df)
        infeasibility_data.update(infeasibility)

    # Run sensitivity analysis for each homogeneous fleet composition
    else:
        for type_name, fleet_comp in fleet_compositions.items():
            df, infeasibility = sensitivity_fleetsize(N=N, fleet_composition=fleet_comp, random_fleet=False,
                                                      trials=trials)
            df["Fleet Composition"] = type_name
            results_df_list.append(df)
            infeasibility_data.update(infeasibility)

    # Add all the results for the specific N to a dataframe
    final_results_df = pd.concat(results_df_list, ignore_index=True)

    file_prefix = "random" if random_fleet else "fixed"
    final_results_df.to_csv(f"results/fleet_size/sensitivity_analysis_{file_prefix}_N{N}_fleet_size.csv", index=False)

    # Convert infeasibility summary to a dataframe
    infeasibility_df = pd.DataFrame.from_dict(
        {key: [val[0], val[1]] for key, val in infeasibility_data.items()},
        orient='index',
        columns=['Infeasible Trials', 'Infeasibility Rate']
    )
    infeasibility_df.index = pd.MultiIndex.from_tuples(infeasibility_df.index, names=["Nodes", "Fleet Size", "Fleet Composition"])

    infeasibility_df.to_csv(f"results/fleet_size/infeasibility/infeasibility_summary_{file_prefix}_N{N}_fleet_size.csv")

    return final_results_df

def plot_metric_fleet(df_results, metric, ylabel, title, filename, random_fleet, N):
    """
    Generalized function to plot different metrics vs fleet size.
    """
    plt.figure(figsize=(8, 6))

    # If using a random fleet composition, plot average results for each fleet size
    if random_fleet:
        avg_results = df_results.set_index("Fleet Size")[f"Avg {metric}"]
        std_results = df_results.set_index("Fleet Size")[f"Std {metric}"].fillna(0)
        if avg_results.empty:
            return

        # Plot mean metric values with error bars representing standard deviation
        plt.errorbar(avg_results.index, avg_results.values, yerr=std_results.values,
                     fmt="o-", capsize=4)

    # If fleet composition is predefined (homogeneous fleets), plot each type separately
    else:
        fleet_types = ["Fixed-Wing (Type 1)", "Quadcopter (Type 2)", "Cargo Blimp (Type 3)"]

        for i, fleet_type in enumerate(fleet_types):
            subset = df_results[df_results["Fleet Composition"] == fleet_type]

            if subset.dropna(subset=[f"Avg {metric}"]).empty:
                continue

            std_values = subset[f"Std {metric}"].fillna(0).values

            # Plot mean values with error bars
            plt.errorbar(subset["Fleet Size"] + 0.03 * i, subset[f"Avg {metric}"],
                         yerr=std_values, fmt="o-", capsize=4, label=fleet_type)

    plt.xlabel("Fleet Size (M)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Determine valid fleet sizes
    valid_fleet_sizes = df_results[df_results["Fleet Size"] <= df_results["Nodes"].max()]["Fleet Size"].unique()

    plt.xticks(sorted(valid_fleet_sizes))
    plt.xlim(min(valid_fleet_sizes) - 0.5, max(valid_fleet_sizes) + 0.5)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if not random_fleet:
        plt.legend()
    plt.grid()

    file_prefix = "random" if random_fleet else "fixed"
    plt.savefig(f"figures/fleet_size/{file_prefix}/N{N}/{filename}_N{N}_{file_prefix}_size.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_results_fleet(df_results, random_fleet, N):
    """
    Plots metrics for fleet size sensitivity analysis.
    """
    plot_metric_fleet(df_results, "Objective Value", "Objective Value (Cost)",
                      f"Averaged Objective Value vs. Fleet Size (N = {N})", "objective_fleet_size", random_fleet, N)
    plot_metric_fleet(df_results, "Normalized Objective Value", "Total Hours per KG Demand",
                      f"Averaged Normalized Objective Value vs. Fleet Size (N = {N})", "normalized_objective_fleet_size", random_fleet, N)
    plot_metric_fleet(df_results, "Total Distance", "Total Travel Distance",
                      f"Averaged Total Distance vs. Fleet Size (N = {N})", "distance_fleet_size", random_fleet, N)
    plot_metric_fleet(df_results, "Runtime", "Computation Runtime (s)",
                      f"Averaged Computation Time vs. Fleet Size (N = {N})", "runtime_fleet_size", random_fleet, N)

# Run analysis for multiple values of N
for N in N_list:
    results_fixed = compute_sensitivity_fleet(N=N, trials=20, random_fleet=False)
    results_random = compute_sensitivity_fleet(N=N, trials=20, random_fleet=True)
    plot_results_fleet(results_fixed, random_fleet=False, N=N)
    plot_results_fleet(results_random, random_fleet=True, N=N)