import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

from sample_case_2 import run


# Fleet types (0 = Fixed Wing, 1 = Quadcopter, 2 = Blimp)
# Generate all unique fleet compositions for a given fleet size M
def generate_fleet_compositions(M):
    fleet_types = [0, 1, 2]
    return [list(comp) for comp in combinations_with_replacement(fleet_types, M)]


# Run multiple trials and store individual results
def run_trials(N, M, fleet_composition, num_trials=3):
    trial_results = {}  # Stores raw objective values
    trial_results_normalized = {}  # Stores normalized objective values
    objective_values = []
    normalized_objective_values = []

    for seed in range(num_trials):
        result = run(N=N, M=M, fleet_composition=fleet_composition, random_fleet=False, seed=seed)
        if result and result["objective_value"] is not None:
            total_demand = result["total_demand"] if "total_demand" in result else 1  # Ensure demand is never zero
            normalized_value = result["objective_value"] / total_demand  # Normalize

            # Store values separately in respective dictionaries
            trial_results[f"Seed {seed}"] = result["objective_value"]
            trial_results_normalized[f"Seed {seed}"] = normalized_value

            objective_values.append(result["objective_value"])
            normalized_objective_values.append(normalized_value)
        else:
            trial_results[f"Seed {seed}"] = np.nan
            trial_results_normalized[f"Seed {seed}"] = np.nan

    avg_objective_value = np.nanmean(objective_values) if objective_values else np.nan
    avg_normalized_value = np.nanmean(normalized_objective_values) if normalized_objective_values else np.nan

    return avg_objective_value, avg_normalized_value, trial_results, trial_results_normalized


# Run experiments for each fleet size and composition
def run_compositions(N, M_list, num_trials=3):
    results_objective = []
    results_normalized = []

    for M in M_list:
        compositions = generate_fleet_compositions(M)
        for fleet_composition in compositions:
            avg_objective_value, avg_normalized_value, trial_results, trial_results_normalized = run_trials(
                N=N, M=M, fleet_composition=fleet_composition, num_trials=num_trials
            )

            # Create base entry
            base_entry = {
                "Fleet Size": M,
                "Fleet Composition": tuple(fleet_composition),
                "% Fixed-Wing": round(fleet_composition.count(0) / M * 100, 1),
                "% Quadcopter": round(fleet_composition.count(1) / M * 100, 1),
                "% Blimp": round(fleet_composition.count(2) / M * 100, 1)
            }

            # Create separate result entries
            result_entry_objective = base_entry.copy()
            result_entry_objective["Average Objective Value"] = avg_objective_value
            result_entry_objective.update(trial_results)

            result_entry_normalized = base_entry.copy()
            result_entry_normalized["Average Normalized Objective Value"] = avg_normalized_value
            result_entry_normalized.update(trial_results_normalized)

            results_objective.append(result_entry_objective)
            results_normalized.append(result_entry_normalized)

    # Convert to DataFrame
    df_results_objective = pd.DataFrame(results_objective)
    df_results_normalized = pd.DataFrame(results_normalized)

    # Save to CSV
    df_results_objective.to_csv("fleet_composition_results_objective.csv", index=False)
    df_results_normalized.to_csv("fleet_composition_results_normalized.csv", index=False)

    # Filter out NaN rows for further analysis
    df_filtered_objective = df_results_objective.dropna(subset=["Average Objective Value"])
    df_filtered_normalized = df_results_normalized.dropna(subset=["Average Normalized Objective Value"])

    return df_results_objective, df_results_normalized, df_filtered_objective, df_filtered_normalized


# Define fleet size range and trials
M_list = range(2, 8)
N = 9
num_trials = 10

df_results_objective, df_results_normalized, df_filtered_objective, df_filtered_normalized = run_compositions(N,
    M_list, num_trials
)
#
# # Print to confirm the structure
# print("Objective Value DataFrame:")
# print(df_results_objective.head())
#
# print("\nNormalized Objective Value DataFrame:")
# print(df_results_normalized.head())

