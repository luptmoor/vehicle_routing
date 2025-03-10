
import numpy as np
import matplotlib.pyplot as plt
import sys
import os



def plot_solution():
    plt.figure(figsize=(8, 8))

    # Generate colors dynamically for vehicles
    cmap = plt.cm.get_cmap('tab10', M)
    vehicle_colors = [cmap(i) for i in range(M)]

    # Plot depot
    plt.scatter(depot[0], depot[1], s=100, c='red', marker='s', label="Depot")
    plt.annotate('Depot', (depot[0] + 1, depot[1] + 1), fontsize=10, color='red', fontweight='bold')

    # Plot customers with demand labels
    for i, (x, y) in enumerate(customers):
        plt.scatter(x, y, s=50, c='blue', marker='o')
        plt.annotate(f"C{i+1} [{demand_list[i]}kg]", (x + 1, y + 1), fontsize=10, color='black')

    # Track total delivered capacities for each vehicle
    delivered_capacities = [0] * M

    # Plot vehicle paths
    for i in range(M):  # Loop over each vehicle
        for j in range(N+1):  # From-node
            for k in range(N+1):  # To-node
                if solution[i, j, k] > 0.5:  # If the route is used
                    start_node = (nodes[j, 0], nodes[j, 1])
                    end_node = (nodes[k, 0], nodes[k, 1])

                    # Add delivered demand if going to a customer (not the depot)
                    if k < N:  # Ensure it's not the depot
                        delivered_capacities[i] += demand_list[k]

                    # Apply offset if another vehicle uses the same path
                    route_offset = 1.0  # Offset distance for overlapping paths (between different vehicles)

                    if any((solution[v, j, k] > 0.5 or solution[v, k, j] > 0.5) and v != i for v in range(M)):
                        start_node = (start_node[0] + route_offset, start_node[1] + route_offset)
                        end_node = (end_node[0] + route_offset, end_node[1] + route_offset)

                    dx = end_node[0] - start_node[0]
                    dy = end_node[1] - start_node[1]

                    # Draw arrow
                    plt.arrow(
                        start_node[0], start_node[1], dx, dy,
                        color=vehicle_colors[i],
                        head_width=1.5, head_length=2.5,  # Slightly larger arrowheads
                        length_includes_head=True
                    )

        # Add legend entry for this vehicle with delivered and total capacity
        plt.scatter([], [], color=vehicle_colors[i], label=f"Vehicle {i+1}: ({delivered_capacities[i]}/{capacity_list[i]}) [KG]")

    # Configure plot
    plt.xticks(range(0, MAP_SIZE, GRID_STEPS))
    plt.yticks(range(0, MAP_SIZE, GRID_STEPS))
    plt.xlim([0, MAP_SIZE])
    plt.ylim([0, MAP_SIZE])
    plt.grid()

    plt.xlabel('X Position [km]')
    plt.ylabel('Y Position [km]')
    plt.title('Optimized Routes for Vehicle Routing Problem')
    plt.legend()

    # Ensure the "results" directory exists
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directory '{results_dir}' created.")

    # Save the figure
    plt.savefig(f"{results_dir}/solution_plot_with_capacity.png")
    plt.show()

if __name__ == "__main__":
    plot_solution()
