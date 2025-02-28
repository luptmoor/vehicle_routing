import matplotlib.pyplot as plt
from gurobipy import *
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sensitivity_analysis.sample_case_2 import run
# Define the range for number of nodes (N) and fleet size (M)
N_range = range(5, 15)  # Number of nodes (customers + depot)
M_range = range(1, 9)   # Number of vehicles in fleet

def generate_feasibility_heatmap():
    results = np.zeros((len(N_range), len(M_range)))  # Initialize matrix

    for i, N in enumerate(N_range):
        for j, M in enumerate(M_range):
            print(f"Running optimization for N={N}, M={M}")
            cost = run(N=N, M=M)

            # If the optimization is feasible (returns an objective value), mark as 1 (green)
            if cost is not None:
                results[i, j] = 1  # Feasible
            else:
                results[i, j] = 0  # Infeasible

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(results, cmap="RdYlGn", aspect="auto", origin="lower")

    # Label axes
    plt.xticks(ticks=np.arange(len(M_range)), labels=[str(m) for m in M_range])
    plt.yticks(ticks=np.arange(len(N_range)), labels=[str(n) for n in N_range])

    plt.xlabel("Fleet Size (M)")
    plt.ylabel("Number of Nodes (N)")
    plt.title("Feasibility Heatmap (Green=Feasible, Red=Infeasible)")

    # Show color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Feasible'),
                       Patch(facecolor='red', label='Infeasible')]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    generate_feasibility_heatmap()