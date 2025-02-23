import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_case_2 import run

M_list = range(2, 5)
N = 15  # Number of customers

def sensitivity_fleet(fleet_composition=None, random_drones=True, trials=1):
    """
    Runs the optimization for different fleet sizes and returns multiple performance metrics:
    - `objective_values`: Optimization cost (averaged over trials if random)
    - `total_distances`: Total travel distance (averaged over trials if random)
    - `runtimes`: Computation time (averaged over trials if random)
    """
    objective_values, total_distances, runtimes = [], [], []

    for M in M_list:
        print(f"Running optimization for M={M}, N={N}")

        fleet_subset = fleet_composition[:M].copy() if fleet_composition else None

        trial_obj_values, trial_distances, trial_runtimes = [], [], []

        for seed in range(trials):  # Multiple trials only for random fleets
            if random_drones:
                np.random.seed(seed)  # Ensure different randomness for each trial
            result = run(N=N, M=M, random_drones=random_drones, fleet_composition=fleet_subset)

            if result:
                trial_obj_values.append(result["objective_value"])
                trial_distances.append(result["total_distance"])
                trial_runtimes.append(result["runtime"])
            else:
                trial_obj_values.append(np.nan)
                trial_distances.append(np.nan)
                trial_runtimes.append(np.nan)

        # Compute averages across trials
        objective_values.append(np.nanmean(trial_obj_values))
        total_distances.append(np.nanmean(trial_distances))
        runtimes.append(np.nanmean(trial_runtimes))

    return {
        "objective_values": objective_values,
        "total_distances": total_distances,
        "runtimes": runtimes
    }


def compute_sensitivity_results(trials=1, random_fleet=False):
    """Runs the sensitivity analysis for different fleet sizes and returns all metrics."""
    fleet_compositions = {
        'Fixed-Wing (Type 1)': [0] * max(M_list),
        'Quadcopter (Type 2)': [1] * max(M_list),
        'Cargo Blimp (Type 3)': [2] * max(M_list)
    }

    results = {}
    if random_fleet:
        results["Random Fleet"] = sensitivity_fleet(fleet_composition=None, random_drones=True, trials=trials)
    else:
        for type_name, fleet_comp in fleet_compositions.items():
            results[type_name] = sensitivity_fleet(fleet_composition=fleet_comp, random_drones=False, trials=1)

    return results


def plot_objective_fleet(results, random_fleet=False):
    """Plots Objective Value vs Fleet Size using precomputed results."""
    plt.figure(figsize=(8, 6))

    for type_name, result in results.items():
        plt.plot(M_list, result["objective_values"], marker="o", label=type_name if not random_fleet else None)

    plt.xlabel("Fleet Size (M)")
    plt.ylabel("Objective Value (Cost)")
    plt.title("Objective Value vs. Fleet Size")
    if not random_fleet:
        plt.legend()
    plt.grid()
    plt.savefig(f"Figures/sensitivity_fleet_size_{'random' if random_fleet else 'per_type'}.png", dpi=300)
    plt.show()


def plot_distance_fleet(results, random_fleet=False):
    """Plots Total Distance vs Fleet Size using precomputed results."""
    plt.figure(figsize=(8, 6))

    for type_name, result in results.items():
        plt.plot(M_list, result["total_distances"], marker="o", label=type_name if not random_fleet else None)

    plt.xlabel("Fleet Size (M)")
    plt.ylabel("Total Travel Distance")
    plt.title("Total Distance vs. Fleet Size")
    if not random_fleet:
        plt.legend()
    plt.grid()
    plt.savefig(f"Figures/distance_fleet_size_{'random' if random_fleet else 'per_type'}.png", dpi=300)
    plt.show()


# Run analysis
results_fixed = compute_sensitivity_results()
results_random = compute_sensitivity_results(trials=8, random_fleet=True)

# Plot for fleets of a single type
plot_objective_fleet(results_fixed)
plot_distance_fleet(results_fixed)

# Plot for **random fleets** (averaged over trials)
plot_objective_fleet(results_random, random_fleet=True)
plot_distance_fleet(results_random, random_fleet=True)