import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *

# Scenario Parameters
np.random.seed(42);
MAP_SIZE = 100; # km (in a square)
GRID_STEPS = 10;
N = 8;

MAX_VEHICLES = 5;
VEHICLE_CAPACITIES = [6, 15, 40];
VEHICLE_VELOCITIES = [170, 80, 40];

# Generate 100 x 100 km map
side_array = np.arange(MAP_SIZE);
x, y = np.meshgrid(side_array, side_array);
coordinate_grid = np.vstack([x.ravel(), y.ravel()]).T;


# Choose N map points as depot and customers
nodes = coordinate_grid[np.random.choice(range(len(coordinate_grid)), N, replace=False)];
depot, customers = nodes[0, :], nodes[1:, :];


# Generate delivery fleet
fleet_numbers = np.random.randint(0, MAX_VEHICLES, 3);  # number of vehicles for each type

fleet_list = [fleet_numbers[i] * [i] for i in range(3)];  # give the type of all vehicles in a list for encoding
fleet_list = [item for sublist in fleet_list for item in sublist]; # flatten the list

velocity_list = [VEHICLE_VELOCITIES[x] for x in fleet_list];
capacity_list = [VEHICLE_CAPACITIES[x] for x in fleet_list];

fleet_capacity = sum(capacity_list); # Total capacity of fleet
M = len(fleet_list); # Number of vehicles


# Generate demand list (N-1)
demand_list = [0] * (N-1);
rem_fleet_capacity = fleet_capacity;
max_demand = rem_fleet_capacity // (N-1) * 2;

while rem_fleet_capacity > 0:
    for i in range(N-1):
        demand = min(np.random.randint(1, max_demand), rem_fleet_capacity);
        demand_list[i] += demand;
        rem_fleet_capacity -= demand;
        if rem_fleet_capacity == 0: break;
demand_list = [int(x) for x in demand_list];  # convert to pure integer list
demand_list.insert(0, 0); # add 0 for demand of depot


# Generate distance matrix (N x N)
distance_matrix = np.zeros((N, N));
for i, j in [(i, j) for i in range(N) for j in range(N)]:
    distance_matrix[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0])**2 + (nodes[i, 1] - nodes[j, 1])**2);


# Generate Traveltime Matrix (M x N x N)
velocity_matrix = [np.full((N, N), 1) * velocity for velocity in velocity_list];
traveltime_matrix = np.array([distance_matrix / velocities for velocities in velocity_matrix]);



#___________________________________________________________________________________________________________________________________________




# Report
print();
print('SCENARIO OVERVIEW');
print();
print(f'Fleet Report');
print(f'Total fleet size M: {M}')
print(f'Fast fixed-wings (C {VEHICLE_CAPACITIES[0]}kg, V {VEHICLE_VELOCITIES[0]}km/h): {fleet_numbers[0]}');
print(f'Quadrotors (C {VEHICLE_CAPACITIES[1]}kg, V {VEHICLE_VELOCITIES[1]}km/h): {fleet_numbers[1]}');
print(f'Cargo Blimps (C {VEHICLE_CAPACITIES[2]}kg, V {VEHICLE_VELOCITIES[2]}km/h): {fleet_numbers[2]}');
print(f'Fleet list for encoding: {fleet_list}');
print(f'Velocity list [km/h]: {velocity_list}');
print(f'Capacity list [kg]: {capacity_list}');
print(f'Total capacity [kg]: {fleet_capacity}');
print();
print(f'A maximum demand of {max_demand}kg is used in distributing the demands.')
print(f'{N-1} customers will be served with following demands [kg]:');
print(f'{demand_list} ({sum(demand_list)}kg in total).')
print();
print('Distance Matrix [km]');
print(distance_matrix);
print()
print()


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
    plt.annotate(demand_list[i+1], [x + 1 for x in customers[i, :]]);
plt.annotate('Depot', [x + 1 for x in depot]);

plt.show();



#_________________________________________________________________________________________________________________________________________



# Gurobi Model definition
m = Model('Vehicle Routing');
m.update();
m.modelSense = GRB.MINIMIZE;


# Binary decision variable (M x N x N)
x = m.addVars(M, N, N, vtype = GRB.BINARY, name='x_ijk');

# Total travel time [h] (i.e. hours that pilots have to be paid for, NOT duration of delivery)
m.setObjective(sum(x[i, j, k] * traveltime_matrix[i, j, k] for i in range(M) for j in range(N) for k in range(N))); 


# Constraints

# 1. Customers k are only visited once by a vehicle
for k in range(N-1):
    m.addConstr(sum(x[i, j, k] for i in range(M) for j in range(N)) == 1);

# 2. Vehicle i's capacity is not exceeded
for i in range(M):
    m.addConstr(sum(x[i, j, k] * demand_list[k] for j in range(N) for k in range(N)) <= capacity_list[i]);


# 3. Vehicles i leave at depot 0 and do not come back
for i in range(M):
    m.addConstr(sum(x[i, 0, k] for k in range(N)) == 1); # yes: from 0 to k
    m.addConstr(sum(x[i, j, 0] for j in range(N)) == 0); # no: from j to 0

# 4. Vehicles end at depot
#for i in range(M):
    

# 5. Vehicles move between customers

# 6. Subtour elimination


m.optimize();
m.write('vehicle_routing.lp')