import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import seaborn as sns

from sample_case_2 import run


# Fleet types (0 = Fixed Wing, 1 = Quadcopter, 2 = Blimp)
# Generate all unique fleet compositions for a given fleet size M
def generate_fleet_compositions(M):
    fleet_types = [0, 1, 2]
    return [list(comp) for comp in combinations_with_replacement(fleet_types, M)]


# Run multiple trials and store individual results
def run_trials(N, M, fleet_composition, num_trials=3):
    trial_results = {}
    objective_values = []

    for seed in range(num_trials):
        result = run(N=N, M=M, fleet_composition=fleet_composition, random_fleet=False, seed=seed)
        if result and result["objective_value"] is not None:
            trial_results[f"Seed {seed}"] = result["objective_value"]
            objective_values.append(result["objective_value"])
        else:
            # Store NaN for infeasible cases
            trial_results[f"Seed {seed}"] = np.nan

    # Compute average, ignoring NaNs
    avg_objective_value = np.nanmean(objective_values) if objective_values else np.nan
    return avg_objective_value, trial_results


# Run experiments for each fleet size and composition
def run_compositions(M_list, num_trials=3):
    results = []

    for M in M_list:
        compositions = generate_fleet_compositions(M)
        for fleet_composition in compositions:
            avg_objective_value, trial_results = run_trials(N=9, M=M, fleet_composition=fleet_composition,
                                                            num_trials=num_trials)

            result_entry = {
                "Fleet Size": M,
                "Fleet Composition": tuple(fleet_composition),
                "% Fixed-Wing": round(fleet_composition.count(0) / M * 100, 1),
                "% Quadcopter": round(fleet_composition.count(1) / M * 100, 1),
                "% Blimp": round(fleet_composition.count(2) / M * 100, 1),
                "Objective Value": avg_objective_value
            }

            # Merge seed-specific results into the row
            result_entry.update(trial_results)

            results.append(result_entry)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv("fleet_composition_results.csv", index=False)

    # Create a filtered DataFrame excluding infeasible cases for plotting
    df_filtered = df_results.dropna(subset=["Objective Value"])
    return df_results, df_filtered


# Define fleet size range and trials
M_list = range(2, 7)
num_trials = 8

df_results, df_filtered = run_compositions(M_list, num_trials)



