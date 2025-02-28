import numpy as np
import pandas as pd

df_fixed = pd.read_csv("sensitivity_analysis_fixed.csv")

df_random = pd.read_csv("sensitivity_analysis_random.csv")

feasibility_pivot_random = df_random.pivot(index='Fleet Size', columns='Seed', values='Feasible')
feasibility_pivot_fixed = df_fixed.pivot(index='Fleet Size', columns=['Fleet Composition', 'Seed'], values='Feasible')

print(feasibility_pivot_random)
print(feasibility_pivot_fixed)