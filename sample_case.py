import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *

# Scenario Parameters
np.random.seed(45);
MAP_SIZE = 100;
GRID_STEPS = 10;
N = 12;
M = 3;
MAX_DEMAND = 10;

MAX_VEHICLES = 4;
VEHICLE_CAPS = [6, 15, 40];
VEHICLE_SPEEDS = [170, 80, 40];

# Generate 100 x 100 km map
side_array = np.arange(MAP_SIZE);
x, y = np.meshgrid(side_array, side_array);
coordinate_grid = np.vstack([x.ravel(), y.ravel()]).T;

# Choose N map points as depot and customers
nodes = coordinate_grid[np.random.choice(range(len(coordinate_grid)), N, replace=False)];
depot, customers = nodes[0, :], nodes[1:, :];

# Generate distance matrix
distance_matrix = np.zeros((N, N));
for i, j in [(i, j) for i in range(N) for j in range(N)]:
    distance_matrix[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0])**2 + (nodes[i, 1] - nodes[j, 1])**2);

# Generate delivery fleet
fleet_numbers = np.random.randint(0, MAX_VEHICLES, 3);
fleet_cap = sum(fleet_numbers * VEHICLE_CAPS);


# Generate demand list
demand_list = [0] * (N-1);
rem_fleet_cap = fleet_cap;
max_demand = rem_fleet_cap // (N-1) * 2;

while rem_fleet_cap > 0:
    for i in range(N-1):
        demand = min(np.random.randint(1, max_demand), rem_fleet_cap);
        demand_list[i] += demand;
        rem_fleet_cap -= demand;
        if rem_fleet_cap == 0: break;
demand_list = [int(x) for x in demand_list];

# Report
print();
print('SCENARIO OVERVIEW');
print();
print(f'Fleet Report');
print(f'Fast fixed-wings (C {VEHICLE_CAPS[0]}, V {VEHICLE_SPEEDS[0]}): {fleet_numbers[0]}');
print(f'Quadrotors (C {VEHICLE_CAPS[1]}, V {VEHICLE_SPEEDS[1]}): {fleet_numbers[1]}');
print(f'Cargo Blimps: (C {VEHICLE_CAPS[2]}, V {VEHICLE_SPEEDS[2]}) {fleet_numbers[2]}');
print(f'Total capacity: {fleet_cap}');
print();
print(f'A maximum demand of {max_demand} is used in distributing the demands.')
print(f'{N-1} customers will be served with following demands:');
print(f'{demand_list} ({sum(demand_list)} in total).')
print();

# Visualise
plt.scatter(depot.T[0], depot.T[1], s=30.0, c='b');
plt.scatter(customers.T[0], customers.T[1], s=15.0, c='g');

plt.xticks(range(0, MAP_SIZE, GRID_STEPS));
plt.yticks(range(0, MAP_SIZE, GRID_STEPS));
plt.xlim([0, MAP_SIZE]);
plt.ylim([0, MAP_SIZE]);
plt.grid();

plt.xlabel('X Position [km]');
plt.ylabel('Y Position [km]');
for i in range(N-1):
    plt.annotate(demand_list[i], [x + 1 for x in customers[i, :]]);
plt.annotate('Depot', [x + 1 for x in depot]);

plt.show();




# Gurobi Model definition
m = Model('Vehicle Routing');
m.update();
m.modelSense = GRB.MINIMIZE;

# TODO: test
xij = m.addVars(N, N, M, vtype = GRB.BINARY, name='x_ij');

m.setObjective(sum(xij * distance_matrix)); 


# Constraints

# 1. Customers are only visited once by a vehicle

# 2. Depot capacity is not exceeded

# 3. Vehicles leave at depot

# 4. Vehicles end at depot

# 5. Vehicles move between customers

# 6. Subtour elimination


m.optimize();
m.write('vehicle_routing.lp')