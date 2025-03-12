import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sample_case_3 import run

MAP_SIZE = 100
GRID_STEPS = 10

# Define depot locations to test
depot_locations = [
    (0,0),
    (25, 25),
    (50, 50),
    (75, 75),
    (100, 100),
]

# Number of trials
num_trials = 20

# Define number of customers N and vehicles M
M = 5
N = 9

# Store results in a DataFrame
results = []

# Store results in a dictionary for plotting the routes without rerunning the model
all_results = {}

# fleet_compositions = {
#         'Fixed-Wing (Type 1)': [0] * (M),
#         'Quadcopter (Type 2)': [1] * (M),
#         'Cargo Blimp (Type 3)': [2] * (M)
#     }

for depot_index, depot in enumerate(depot_locations):
    for seed in range(num_trials):
        result = run(N=N, M=M, random_fleet= True, depot_location = depot, seed=seed)
        all_results[(depot_index, seed)] = result

        # If feasible extract results
        if result["objective_value"] is not None:
            feasibility = "Yes"
            obj_value = result["objective_value"]
            total_dist = result["total_distance"]
            runtime = result["runtime"]
            norm_obj_value = obj_value/sum(result["demand_list"])
        else:
            feasibility = "No"
            obj_value = np.nan
            total_dist = np.nan
            runtime = np.nan
            norm_obj_value = np.nan

        results.append({
            "Depot Location Index": depot_index,
            "Nodes": N,
            "Fleet Size": M,
            "Fleet Composition": str(tuple(result["fleet"])),
            "Seed": seed,
            "Objective Value": obj_value,
            "Normalized Objective Value": norm_obj_value,
            "Total Distance": total_dist,
            "Runtime": runtime,
            "Feasible": feasibility
        })


# Plot function for vehicle routes
def plot_solution(result, depot_index, seed):
    plt.figure(figsize=(10, 8))

    depot = np.array(result["depot_location"])
    customers = np.array(result["customer_locations"])
    solution_x = result["solution_x"]
    fleet = result["fleet"]
    demand_list = result["demand_list"]

    vehicle_types = {0: "Fixed Wing", 1: "Quadcopter", 2: "Cargo Blimp"}

    # Generate colors for vehicles
    cmap = plt.colormaps.get_cmap('tab10')
    vehicle_colors = [cmap(i % M) for i in range(M)]

    # Plot depot
    plt.scatter(depot[0], depot[1], s=100, c='red', marker='s', label="Depot")
    plt.annotate('Depot', (depot[0] + 1, depot[1] + 1), fontsize=12, color='red', fontweight='bold')

    # Plot customers
    for i, (x, y) in enumerate(customers):
        plt.scatter(x, y, s=50, c='blue', marker='o')
        plt.annotate(f"C{i + 1} [{demand_list[i]}kg]", (x + 1, y + 1), fontsize=12, color='black')

    # Track total delivered capacities for each vehicle
    delivered_capacities = {i: 0 for i in range(M)}

    # Plot vehicle paths
    for (i, j, k) in solution_x:
        start_node = customers[j] if j < len(customers) else depot
        end_node = customers[k] if k < len(customers) else depot

        # Add delivered demand if going to a customer (not depot)
        if k < len(customers):
            delivered_capacities[i] += result["demand_list"][k]

        dx = end_node[0] - start_node[0]
        dy = end_node[1] - start_node[1]
        plt.arrow(
            start_node[0], start_node[1], dx, dy,
            color=vehicle_colors[i % M],
            head_width=1.5, head_length=2.5,
            length_includes_head=True
        )

    # Add legend entries for each vehicle with its type and delivered capacity
    for i in range(M):
        vehicle_type = vehicle_types[fleet[i]]
        plt.scatter([], [], color=vehicle_colors[i],
                    label=f"Vehicle {i+1} ({vehicle_type}): {delivered_capacities[i]}/{result['capacity_list'][i]} kg")

    plt.grid()
    plt.xlabel('X Position [km]', fontsize = 16)
    plt.ylabel('Y Position [km]', fontsize = 16)
    plt.xlim(-2, MAP_SIZE+2)
    plt.ylim(-2, MAP_SIZE+2)
    plt.title(f'Optimized Routes for Depot {depot_index}, Seed {seed}', fontsize = 18)
    plt.xticks(fontsize=14)  # Increase x-axis tick font size
    plt.yticks(fontsize=14)  # Increase y-axis tick font size
    if depot_index == 3 or depot_index == 4:
        plt.legend(fontsize=12, loc = "lower left")
    else:
        plt.legend(fontsize=12)
    plt.savefig(f"Figures/depot/solution_plot_depot{depot_index}_seed{seed}.png", bbox_inches = "tight", dpi = 300)
    plt.show()

def plot_depot_sensitivity_boxplot(df_results):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Depot Location Index", y="Normalized Objective Value", data=df_results)
    plt.xlabel("Depot Location", fontsize = 14)
    plt.ylabel("Total Hours per KG Demand", fontsize = 14)
    plt.title(f"Sensitivity of Normalized Objective Value to Depot Location \n"
              f" (N={N}, M={M}, trials={num_trials})", fontsize = 16)
    plt.grid()
    plt.xticks(fontsize=14)  # Increase x-axis tick font size
    plt.yticks(fontsize=14)

    plt.savefig(f"Figures/depot/boxplot_depot_sensitivity_boxplot_N{N}_M{M}.png", bbox_inches = "tight", dpi = 300)
    plt.show()

# for (depot_index, seed), result in all_results.items():
#     if result and result["solution_x"]:
#         plot_solution(result, depot_index, seed)

# Convert results to a DataFrame and save
df_results = pd.DataFrame(results)
df_results.to_csv("results/depot/depot_sensitivity_results.csv", index=False)

plot_depot_sensitivity_boxplot(df_results)