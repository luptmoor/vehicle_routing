import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verification_functions import verify_all_constraints


# Scenario Parameters
np.random.seed(42);
MAP_SIZE = 100; # km (in a square)
GRID_STEPS = 10;

# Vehicle type characteristics [Fixed wing (fast but little capactiy), quadcopter (balanced), cargo blimp (slow but high capacity)]
VEHICLE_CAPACITIES = [6, 15, 40]; # kg
VEHICLE_VELOCITIES = [170, 80, 40]; # km/h

N = 5; # number of nodes including depot
M = 2; # number of vehicles

# Generate 100 x 100 km map
side_array = np.arange(MAP_SIZE);
x, y = np.meshgrid(side_array, side_array);
coordinate_grid = np.vstack([x.ravel(), y.ravel()]).T;


# Choose N map points as depot and customers
nodes = np.zeros((N+1, 2));
nodes[:N, :] = coordinate_grid[np.random.choice(range(len(coordinate_grid)), N, replace=False)];
customers, depot = nodes[:N-1, :], nodes[N-1, :];
nodes[N, :] = depot; # duplicate depot node to mimic problem formulation from paper

# Generate delivery fleet
fleet_list = np.random.randint(0, 3, M);  # randomly decide vehicle type (0-2) for M vehicles

n_0, n_1, n_2 = 0, 0, 0; # Counters for vehicle types
for i in fleet_list:
    if i == 0: n_0 +=1;
    elif i == 1: n_1 +=1;
    elif i == 2: n_2 +=1;

velocity_list = [VEHICLE_VELOCITIES[x] for x in fleet_list]; # useful later
capacity_list = [VEHICLE_CAPACITIES[x] for x in fleet_list];

fleet_capacity = sum(capacity_list); # Total capacity of fleet


# Generate demand list (N+1) (depot counted twice: as start and end)
demand_list = [0] * (N+1);
rem_fleet_capacity = fleet_capacity;

max_demand = rem_fleet_capacity // (N-1) * 7 // 4;
for i in range(N-1):
    demand = min(np.random.randint(1, max_demand), rem_fleet_capacity-M-1);
    demand_list[i] += demand;
    rem_fleet_capacity -= demand;
    if rem_fleet_capacity == 0: break;
demand_list = [int(x) for x in demand_list];  # convert to pure integer list



# Generate distance matrix (N+1 x N+1)
distance_matrix = np.zeros((N+1, N+1)); # [km]

for i, j in [(i, j) for i in range(N+1) for j in range(N+1)]:
    distance_matrix[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0])**2 + (nodes[i, 1] - nodes[j, 1])**2); # Pythagorean direct distance


# Generate Traveltime Matrix (M x N+1 x N+1)
velocity_matrix = [np.full((N+1, N+1), 1) * velocity for velocity in velocity_list]; # [km/h]
traveltime_matrix = np.array([distance_matrix / velocities for velocities in velocity_matrix]); # [h]



#___________________________________________________________________________________________________________________________________________




# Report for verification
print();
print('SCENARIO OVERVIEW');
print();
print(f'Fleet Report');
print(f'Total fleet size M: {M}')
print(f'Fast fixed-wings (C {VEHICLE_CAPACITIES[0]}kg, V {VEHICLE_VELOCITIES[0]}km/h): {n_0}');
print(f'Quadrotors (C {VEHICLE_CAPACITIES[1]}kg, V {VEHICLE_VELOCITIES[1]}km/h): {n_1}');
print(f'Cargo Blimps (C {VEHICLE_CAPACITIES[2]}kg, V {VEHICLE_VELOCITIES[2]}km/h): {n_2}');
print(f'Fleet list for encoding: {fleet_list}');
print(f'Velocity list [km/h]: {velocity_list}');
print(f'Capacity list [kg]: {capacity_list}');
print(f'Total capacity [kg]: {fleet_capacity}');
print();
print(f'A maximum demand of {max_demand}kg is used in distributing the demands.')
print(f'{N-1} customers will be served with following demands [kg]:');
print(f'{demand_list[:-2]} ({sum(demand_list)}kg in total).')
print();
# print('Distance Matrix [km]');
# print(distance_matrix);
# print()
# print()
# print('Traveltime Matrix [h]:');
# print(traveltime_matrix);


# Visualise scenario without solution
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
    plt.annotate(str(int(demand_list[i])) + ' kg', [x + 1 for x in customers[i, :]]);
plt.annotate('Depot', [x + 1 for x in depot]);
plt.legend(['Depot', 'Customers']);




#_________________________________________________________________________________________________________________________________________



# Gurobi Model definition
m = Model('Vehicle Routing');
m.update();
m.modelSense = GRB.MINIMIZE;


# Binary decision variable (M x N+1 x N+1) (N+1 because depot counted twice: depotstart and depotend)
x = m.addVars(range(M), range(N+1), range(N+1), vtype = GRB.BINARY, name='x_ijk');

# Total travel time [h] (i.e. hours that pilots have to be paid for, NOT duration of delivery)
m.setObjective(sum(x[i, j, k] * traveltime_matrix[i, j, k] for i in range(M) for j in range(N) for k in range(N+1))); 



# Constraints:  Vehicle i goes from node j to k

# 1. Every customer k is visited once by a vehicle and drones also leave from every customer
for k in range(N-1):
    m.addConstr(sum(x[i, j, k] for i in range(M) for j in range(N)) == 1);


# 2. Vehicles i leave at depotStart N and do not come back
for i in range(M):
    m.addConstr(sum(x[i, N  -1, k] for k in range(N-1)) == 1); # yes: from N to k until N-1 (customers)
    m.addConstr(sum(x[i, j, N  -1] for j in range(N)) == 0); # no: from j to N


# 3. Vehicles i do not leave at depotEnd N+1 but always come back
for i in range(M):
    m.addConstr(sum(x[i, N+1  -1, k] for k in range(N+1)) == 0);
    m.addConstr(sum(x[i, j, N+1  -1] for j in range(N-1)) == 1);
    

# 4. If a vehicle i goes to a node, it also needs to leave from that node
for i in range(M):
    for j in range(N-1):
        m.addConstr(sum(x[i, j, k] for k in list(range(N-1)) + [N] if k!=j) - sum(x[i, k, j] for k in range(N)) == 0);
#                       outgoing arcs from j                    -   incoming arcs to j                == 0
        #for k in range(N-1):
           # m.addConstr(x[i, j, k] + x[i, k, j] < 2);


# 5. Subtour elimination
u = m.addVars(range(M), range(N-1), vtype=GRB.CONTINUOUS, lb=0, ub=N-1, name="u");
for i in range(M):
    for j in range(N-1):
        for k in range(N-1):
            if j != k:
                m.addConstr(u[i, j] - u[i, k] + (N - 1) * x[i, j, k] <= N-2);



# 6. Vehicle i's capacity is not exceeded.
for i in range(M):
    m.addConstr(sum(x[i, j, k] * demand_list[k] for j in range(N) for k in range(N-1)) <= capacity_list[i]);



m.optimize();
m.write('vehicle_routing.lp') # Only writes simplex problem formulation, not solution.



#_______________________________________________________________________________________________________________________________________________________-


# Convert solution to np.array for output print
solution = np.zeros((M, N+1, N+1));

for i in range(M):
    for j in range(N+1):
        for k in range(N+1):
            solution[i, j, k] = x[i, j, k].X;


print()
print()
print(solution);
print()
print('Total visit matrix');
print(sum(solution[i, :, :] for i in range(M)));
print()
print('Total time needed:', m.ObjVal, 'person hours.')
#plt.show(); # only plots scenario, not solution

# verify_all_constraints(solution, demand_list, capacity_list)