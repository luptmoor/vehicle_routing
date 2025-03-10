import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sample_case_2 import run

# List of fleet sizes to iterate over
M_list = range(2, 8)

# Define the range of N (number of customers)
N_list = range(3, 14)

def sensitivity_nodes(M, fleet_composition=None, random_fleet=True, trials=1):
    """
    Runs the optimization for different numbers of customers (N) and stores all trial results in a DataFrame.
    """
    results_list = []
    infeasibility_summary = {}


    for N in N_list:
        if M > N:
            continue
        print(f"Running trials for M = {M}, N = {N}")

        infeasible_trials = []

        trial_objective_values = []
        trial_normalized_values = []
        trial_total_distances = []
        trial_runtimes = []
        trial_results = []

        for seed in range(trials):
            result = run(N=N, M=M, random_fleet=random_fleet, fleet_composition=fleet_composition, seed=seed)

            if result and result["objective_value"] is not None:
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

        if len(set(result["fleet"])) == 1:
            fleet_type = result["fleet"][0]
        else:
            fleet_type = "random"
        infeasibility_summary[(N, M, fleet_type)] = (infeasible_trials, len(infeasible_trials) / trials)

        avg_obj_value = np.nanmean(trial_objective_values) if trial_objective_values else np.nan
        avg_norm_value = np.nanmean(trial_normalized_values) if trial_normalized_values else np.nan
        avg_total_distance = np.nanmean(trial_total_distances) if trial_total_distances else np.nan
        avg_runtime = np.nanmean(trial_runtimes) if trial_runtimes else np.nan

        std_obj_value = np.nanstd(trial_objective_values) if len(trial_objective_values) > 1 else np.nan
        std_norm_value = np.nanstd(trial_normalized_values) if len(trial_normalized_values) > 1 else np.nan
        std_total_distance = np.nanstd(trial_total_distances) if len(trial_total_distances) > 1 else np.nan
        std_runtime = np.nanstd(trial_runtimes) if len(trial_runtimes) > 1 else np.nan

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

    df_results = pd.DataFrame(results_list)
    return df_results, infeasibility_summary


def compute_node_sensitivity_results(M, trials=1, random_fleet=False):
    fleet_compositions = {
        'Fixed-Wing (Type 1)': [0] * M,
        'Quadcopter (Type 2)': [1] * M,
        'Cargo Blimp (Type 3)': [2] * M
    }

    results_df_list = []
    infeasibility_data = {}

    if random_fleet:
        df, infeasibility = sensitivity_nodes(M = M, fleet_composition = None, random_fleet=True, trials=trials)
        results_df_list.append(df)
        infeasibility_data.update(infeasibility)
    else:
        for type_name, fleet_comp in fleet_compositions.items():
            df, infeasibility = sensitivity_nodes(M, fleet_composition=fleet_comp, random_fleet=False, trials=trials)
            df["Fleet Composition"] = type_name
            results_df_list.append(df)
            infeasibility_data.update(infeasibility)

    final_results_df = pd.concat(results_df_list, ignore_index=True)

    file_prefix = "random" if random_fleet else "fixed"
    final_results_df.to_csv(f"nodes/sensitivity_analysis_{file_prefix}_M{M}_nodes.csv", index=False)

    infeasibility_df = pd.DataFrame.from_dict(
        {key: [val[0], val[1]] for key, val in infeasibility_data.items()},
        orient='index',
        columns=['Infeasible Trials', 'Infeasibility Rate']
    )
    infeasibility_df.index = pd.MultiIndex.from_tuples(infeasibility_df.index, names=["Nodes", "Fleet Size", "Fleet Composition"])

    infeasibility_df.to_csv(f"Infeasibility/infeasibility_summary_{file_prefix}_{M}M_nodes.csv")


    return final_results_df



def plot_metric_nodes(df_results, metric, ylabel, title, filename, random_fleet, M):
    """ Generalized function to plot different metrics vs Number of Nodes """
    plt.figure(figsize=(8, 6))

    if random_fleet:
        avg_results = df_results.groupby("Nodes")[metric].mean()
        std_results = df_results.groupby("Nodes")[f"Std {metric}"].mean().fillna(0)  # Avoid NaN in std deviation
        if avg_results.empty:
            print(f"Skipping {metric}: No valid data")
            return
        plt.errorbar(avg_results.index, avg_results.values, yerr=std_results.values, fmt="o-", capsize=4)
    else:
        fleet_types = ["Fixed-Wing (Type 1)", "Quadcopter (Type 2)", "Cargo Blimp (Type 3)"]
        for i, fleet_type in enumerate(fleet_types):
            subset = df_results[df_results["Fleet Composition"] == fleet_type]
            if subset.dropna(subset=[f"Avg {metric}"]).empty:
                continue  # Skip fleet type if no valid data

            std_values = subset[f"Std {metric}"].fillna(0).values
            plt.errorbar(subset["Nodes"] + 0.03*i, subset[f"Avg {metric}"], yerr=std_values, fmt="o-", capsize=4, label=fleet_type)

    plt.xlabel("Number of Nodes (N)", fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    valid_nodes = df_results[df_results["Nodes"] >= df_results["Fleet Size"].min()][
        "Nodes"].unique()
    plt.xlim(min(valid_nodes) - 0.5, max(valid_nodes) + 0.5)
    plt.xticks(sorted(valid_nodes))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(title, fontsize = 16)
    if not random_fleet:
        plt.legend()
    plt.grid()
    file_prefix = "random" if random_fleet else "fixed"
    plt.savefig(f"Figures/Customer_size/{file_prefix}/M{M}/{filename}_M{M}_{file_prefix}_nodes.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_results(df_results, random_fleet, M):
    """Plots all required metrics using filtered data."""
    plot_metric_nodes(df_results, "Objective Value", "Objective Value (Cost)",
                      f"Averaged Objective Value vs. Number of Nodes (M = {M})", "objective_node_size",
                      random_fleet, M)
    plot_metric_nodes(df_results, "Normalized Objective Value", "Total Hours per KG Demand",
                      f"Averaged Normalized Objective Value vs. Number of Nodes (M = {M})",
                      "normalized_objective_node_size", random_fleet, M)
    plot_metric_nodes(df_results, "Total Distance", "Total Travel Distance",
                      f"Averaged Total Distance vs. Number of Nodes (M = {M})", "distance_node_size",
                      random_fleet, M)
    plot_metric_nodes(df_results, "Runtime", "Computation Runtime (s)",
                      f"Averaged Computation Time vs. Number of Nodes (M = {M})", "runtime_node_size",
                      random_fleet, M)

# Run analysis for multiple values of M
for M in M_list:
    results_fixed = compute_node_sensitivity_results(M=M, trials=10, random_fleet=False)
    results_random = compute_node_sensitivity_results(M=M, trials=10, random_fleet=True)

    plot_results(results_fixed, random_fleet=False, M=M)
    plot_results(results_random, random_fleet=True, M=M)
