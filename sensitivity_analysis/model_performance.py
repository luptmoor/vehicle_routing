import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *

from sample_case_2 import run

N_max = 15
M_max = 13
trials = 20

objective_values = np.full((N_max + 1, M_max + 1, trials), np.nan)
normalized_values = np.full((N_max + 1, M_max + 1, trials), np.nan)
runtimes = np.full((N_max + 1, M_max + 1, trials), np.nan)
feasibility_counts = np.zeros((N_max + 1, M_max + 1))

# Run optimization for each combination
for N in range(2, N_max + 1):
    for M in range(1, M_max + 1):
        for t in range(trials):
            result = run(N=N, M=M, seed=t)  # Use different seeds for trials
            if result["objective_value"] is not None:
                objective_values[N, M, t] = result["objective_value"]
                normalized_values[N, M, t] = result["normalized_objective_value"]
                runtimes[N, M, t] = result["runtime"]
                feasibility_counts[N, M] += 1


mean_objective = np.full((N_max + 1, M_max + 1), np.nan)
mean_normalized_objective = np.full((N_max + 1, M_max + 1), np.nan)
mean_runtime = np.full((N_max + 1, M_max + 1), np.nan)
feasibility_ratio = feasibility_counts / trials

for N in range(N_max + 1):
    for M in range(M_max + 1):
        if feasibility_counts[N, M] > 0:  # Only compute if at least one feasible trial exists
            mean_objective[N, M] = np.nanmean(objective_values[N, M, :])
            mean_normalized_objective[N, M] = np.nanmean(normalized_values[N, M, :])
            mean_runtime[N, M] = np.nanmean(runtimes[N, M, :])

df_mean_objective = pd.DataFrame(mean_objective)
df_mean_normalized_objective = pd.DataFrame(mean_normalized_objective)
df_mean_runtime = pd.DataFrame(mean_runtime)
df_feasibility_ratio = pd.DataFrame(feasibility_ratio)

df_mean_objective.to_csv("results/model_performance/mean_objective.csv", index=False)
df_mean_normalized_objective.to_csv("results/model_performance/mean_normalized_objective.csv", index=False)
df_mean_runtime.to_csv("results/model_performance/mean_runtime.csv", index=False)
df_feasibility_ratio.to_csv("results/model_performance/feasibility_ratio.csv", index=False)


def plot_heatmap(data, title, filename, cmap="coolwarm", mask=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        cmap=cmap,
        linewidths=0.5,
        linecolor="black",
        cbar=True,
        fmt=".2f",
        annot_kws={"size": 8},
        mask=mask
    )
    plt.xlabel("Fleet Size (M)", fontsize = 14)
    plt.ylabel("Number of Nodes (N)", fontsize = 14)
    plt.title(title)
    plt.savefig(f"figures/model_performance/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

plot_heatmap(mean_objective, f"Mean Objective Value (trials = {trials})", "mean_objective_heatmap.png")
plot_heatmap(mean_normalized_objective, f"Mean Objective Value per kg of Demand Served (trials = {trials})", "mean_normalized_objective_heatmap.png")
plot_heatmap(mean_runtime, f"Mean Runtime (seconds) (trials = {trials})", "mean_runtime_heatmap.png", cmap="Blues", mask=np.isnan(mean_runtime))
plot_heatmap(feasibility_ratio, f"Feasibility Ratio (trials = {trials})", "feasibility_heatmap.png", cmap="RdYlGn")