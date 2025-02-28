import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_case_2 import run

M = 5
N = 9


def run_speed_analysis(N=9, M=5, trials=8, lower=0.4, upper=1.6, speed_steps=30):
    """
    Runs speed sensitivity analysis and records:
    - The exact speed multipliers where the solution changes for each trial.
    - Trials where the solution NEVER changed for all speed steps (lower or upper).
    """
    left_changes = []
    right_changes = []
    fleet_compositions = {}

    # Track trials where the solution never changed
    no_change_trials = {"lower": [], "upper": []}

    # Store per-trial change points for detailed printing
    per_trial_changes = {}

    for seed in range(trials):
        np.random.seed(seed)

        left_multipliers = np.round(np.linspace(1.0, lower, speed_steps), decimals=4)
        right_multipliers = np.round(np.linspace(1.0, upper, speed_steps), decimals=4)

        base_result = run(N=N, M=M, speed_multiplier=1.0, random_drones=True, seed=seed)

        # Ensure every trial is recorded, even if it fails
        if base_result is None:
            per_trial_changes[seed] = {
                "fleet": 'Infeasible',
                "lower_change": "Infeasible",
                "upper_change": "Infeasible"
            }
            continue

        base_edges = base_result["solution_x"]
        fleet_compositions[seed] = base_result["fleet"]

        left_change_point = None
        right_change_point = None

        # Check when solution changes for lower multipliers
        for multiplier in left_multipliers:
            result = run(N=N, M=M, speed_multiplier=multiplier, random_drones=True, seed=seed)
            if result and result["solution_x"] != base_edges:
                left_changes.append(multiplier)
                left_change_point = multiplier
                break

        # Check when solution changes for upper multipliers
        for multiplier in right_multipliers:
            result = run(N=N, M=M, speed_multiplier=multiplier, random_drones=True, seed=seed)
            if result and result["solution_x"] != base_edges:
                right_changes.append(multiplier)
                right_change_point = multiplier
                break

        # Store the change points for detailed output
        per_trial_changes[seed] = {
            "fleet": fleet_compositions[seed],
            "lower_change": left_change_point if left_change_point is not None else "No Change",
            "upper_change": right_change_point if right_change_point is not None else "No Change",
        }

        # If no change occurred for all speed steps, record the trial
        if left_change_point is None:
            no_change_trials["lower"].append((seed, fleet_compositions[seed]))
        if right_change_point is None:
            no_change_trials["upper"].append((seed, fleet_compositions[seed]))

    # Count occurrences of each speed change value
    left_change_counts = Counter(left_changes)
    right_change_counts = Counter(right_changes)

    return left_change_counts, right_change_counts, fleet_compositions, no_change_trials, per_trial_changes


def plot_speed_analysis(N=9, M=5, speed_steps=30, lower=0.4, upper=1.6, trials=8):
    """
    Plots speed sensitivity results with frequency-based bars.
    - Grid is behind the bars.
    - Vertical grid lines at every tested speed multiplier (both lower and upper).
    - X-axis labels only at multipliers with at least one bar.
    - Lighter and clearer grid lines.
    """
    left_change_counts, right_change_counts, fleet_compositions, no_change_trials, per_trial_changes = run_speed_analysis(
        N=N, M=M, speed_steps=speed_steps, lower=lower, upper=upper, trials=trials
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the exact multipliers tested (for vertical grid lines)
    lower_multipliers = np.round(np.linspace(1.0, lower, speed_steps), decimals=4)
    upper_multipliers = np.round(np.linspace(1.0, upper, speed_steps), decimals=4)
    step_values = np.concatenate((lower_multipliers, upper_multipliers))

    # Sort unique speed multipliers (for bars & x-axis labels)
    all_unique_multipliers = sorted(set(left_change_counts.keys()).union(right_change_counts.keys()))

    # Get corresponding heights (frequencies) for each unique multiplier
    left_frequencies = [left_change_counts.get(multiplier, 0) for multiplier in all_unique_multipliers]
    right_frequencies = [right_change_counts.get(multiplier, 0) for multiplier in all_unique_multipliers]

    # Bar width set dynamically for consistent visualization
    bar_width = (upper - lower) / speed_steps * 0.5

    # Bar plots
    ax.bar(all_unique_multipliers, left_frequencies, width=bar_width, alpha=0.7,
           label="Decrease (Left of 1.0)")
    ax.bar(all_unique_multipliers, right_frequencies, width=bar_width, alpha=0.7,
           label="Increase (Right of 1.0)")

    # Ensure grid is behind the bars
    ax.set_axisbelow(True)

    # Add vertical grid lines at **all tested multipliers**
    ax.set_xticks(step_values, minor=True)  # Minor ticks for vertical grid at every tested multiplier
    ax.grid(which='minor', axis='x', linestyle='-', linewidth=0.5, alpha=0.5)  # Lighter minor grid lines

    # Only show x-axis labels where there's at least one bar (frequency > 0)
    ax.set_xticks(all_unique_multipliers)
    ax.set_xticklabels(all_unique_multipliers, rotation=45)

    # Improve grid visibility
    ax.grid(which = 'major', linestyle='-', linewidth=0.5, alpha=0.5)  # Slightly stronger grid for better clarity

    ax.set_xlabel("Speed Multiplier Where Optimal Solution Changes")
    ax.set_ylabel("Frequency (Number of Trials)")
    ax.set_title(f"Distribution of Critical Speed Values (M={M}, N={N}, Trials={trials})")
    ax.legend()

    plt.savefig(f"speed_sensitivity_analysis_M{M}_N{N}.png", dpi=300)
    plt.show()

    print("\n Detailed Trial Results:")
    for trial in range(trials):
        details = per_trial_changes.get(trial, {"fleet": "Missing", "lower_change": "Missing", "upper_change": "Missing"})
        print(f"Trial {trial}: Fleet {details['fleet']} → Lower Change: {details['lower_change']}, Upper Change: {details['upper_change']}")

    print("\n Trials Where the Solution NEVER Changed:")
    if no_change_trials["lower"]:
        print("\n Trials where no change occurred for lower speed multipliers:")
        for trial, fleet in no_change_trials["lower"]:
            print(f"Trial {trial}: Fleet {fleet}")

    if no_change_trials["upper"]:
        print("\n Trials where no change occurred for higher speed multipliers:")
        for trial, fleet in no_change_trials["upper"]:
            print(f"Trial {trial}: Fleet {fleet}")




def run_capacity_analysis(N=9, M=5, trials=8, lower=0.4, upper=1.6, capacity_steps=30):
    """
    Runs payload (capacity) sensitivity analysis:
    - Tracks multipliers where the optimal solution changes.
    - Identifies trials where the solution never changed.
    """
    lower_changes = []
    upper_changes = []
    fleet_compositions = {}

    # Track trials where the solution never changed
    no_change_trials = {"lower": [], "upper": []}

    # Store per-trial change points for detailed printing
    per_trial_changes = {}

    for seed in range(trials):
        np.random.seed(seed)

        lower_multipliers = np.round(np.linspace(1.0, lower, capacity_steps), decimals=4)
        upper_multipliers = np.round(np.linspace(1.0, upper, capacity_steps), decimals=4)

        base_result = run(N=N, M=M, capacity_multiplier=1.0, random_drones=True, seed=seed)

        # Ensure every trial is recorded, even if it fails
        if base_result is None:
            per_trial_changes[seed] = {
                "fleet": "Infeasible",
                "lower_change": "Infeasible",
                "upper_change": "Infeasible"
            }
            continue

        base_edges = base_result["solution_x"]
        fleet_compositions[seed] = base_result["fleet"]

        lower_change_point = None
        upper_change_point = None

        # Check when solution changes for lower capacity multipliers
        for multiplier in lower_multipliers:
            result = run(N=N, M=M, capacity_multiplier=multiplier, random_drones=True, seed=seed)
            if result and result["solution_x"] != base_edges:
                lower_changes.append(multiplier)
                lower_change_point = multiplier
                break

        # Check when solution changes for upper capacity multipliers
        for multiplier in upper_multipliers:
            result = run(N=N, M=M, capacity_multiplier=multiplier, random_drones=True, seed=seed)
            if result and result["solution_x"] != base_edges:
                upper_changes.append(multiplier)
                upper_change_point = multiplier
                break

        # Store the change points for detailed output
        per_trial_changes[seed] = {
            "fleet": fleet_compositions[seed],
            "lower_change": lower_change_point if lower_change_point is not None else "No Change",
            "upper_change": upper_change_point if upper_change_point is not None else "No Change",
        }

        # If no change occurred for all capacity multipliers, record the trial
        if lower_change_point is None:
            no_change_trials["lower"].append((seed, fleet_compositions[seed]))
        if upper_change_point is None:
            no_change_trials["upper"].append((seed, fleet_compositions[seed]))

    # Count occurrences of each capacity change value
    lower_change_counts = Counter(lower_changes)
    upper_change_counts = Counter(upper_changes)

    return lower_change_counts, upper_change_counts, fleet_compositions, no_change_trials, per_trial_changes

def plot_capacity_analysis(N=9, M=5, capacity_steps=30, lower=0.4, upper=1.6, trials=8):
    """
    Plots capacity sensitivity results with frequency-based bars.
    - Grid is behind the bars.
    - Vertical grid lines at every tested capacity multiplier (both lower and upper).
    - X-axis labels only at multipliers with at least one bar.
    - Lighter and clearer grid lines.
    """
    lower_change_counts, upper_change_counts, fleet_compositions, no_change_trials, per_trial_changes = run_capacity_analysis(
        N=N, M=M, capacity_steps=capacity_steps, lower=lower, upper=upper, trials=trials
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the exact multipliers tested (for vertical grid lines)
    lower_multipliers = np.round(np.linspace(1.0, lower, capacity_steps), decimals=4)
    upper_multipliers = np.round(np.linspace(1.0, upper, capacity_steps), decimals=4)
    step_values = np.concatenate((lower_multipliers, upper_multipliers))

    # Sort unique capacity multipliers (for bars & x-axis labels)
    all_unique_multipliers = sorted(set(lower_change_counts.keys()).union(upper_change_counts.keys()))

    # Get corresponding heights (frequencies) for each unique multiplier
    lower_frequencies = [lower_change_counts.get(multiplier, 0) for multiplier in all_unique_multipliers]
    upper_frequencies = [upper_change_counts.get(multiplier, 0) for multiplier in all_unique_multipliers]

    bar_width = (upper - lower) / capacity_steps * 0.5

    ax.bar(all_unique_multipliers, lower_frequencies, width=bar_width, alpha=0.7,
           label="Decrease (Lower than 1.0)")
    ax.bar(all_unique_multipliers, upper_frequencies, width=bar_width, alpha=0.7,
           label="Increase (Higher than 1.0)")

    ax.set_axisbelow(True)

    ax.set_xticks(step_values, minor=True)
    ax.grid(which='minor', axis='x', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xticks(all_unique_multipliers)
    ax.set_xticklabels(all_unique_multipliers, rotation=45)

    ax.grid(which = 'major', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Capacity Multiplier Where Optimal Solution Changes")
    ax.set_ylabel("Frequency (Number of Trials)")
    ax.set_title(f"Distribution of Critical Capacity Values (M={M}, N={N}, Trials={trials})")
    ax.legend()

    plt.savefig(f"capacity_sensitivity_analysis_M{M}_N{N}.png", dpi=300)
    plt.show()

    print("\n Detailed Trial Results:")
    for trial in range(trials):
        details = per_trial_changes.get(trial, {"fleet": "Missing", "lower_change": "Missing", "upper_change": "Missing"})
        print(f"Trial {trial}: Fleet {details['fleet']} → Lower Change: {details['lower_change']}, Upper Change: {details['upper_change']}")

    print("\n Trials Where the Solution NEVER Changed:")
    if no_change_trials["lower"]:
        print("\n Trials where no change occurred for lower capacity multipliers:")
        for trial, fleet in no_change_trials["lower"]:
            print(f"Trial {trial}: Fleet {fleet}")

    if no_change_trials["upper"]:
        print("\n Trials where no change occurred for higher capacity multipliers:")
        for trial, fleet in no_change_trials["upper"]:
            print(f"Trial {trial}: Fleet {fleet}")


# Run and plot results
plot_capacity_analysis(N=9, M=5, lower=0.4, upper=1.6, capacity_steps=21, trials=100)
# Run and plot results
plot_speed_analysis(N=9, M=5, lower=0.4, upper=1.6, speed_steps=21, trials=100)

