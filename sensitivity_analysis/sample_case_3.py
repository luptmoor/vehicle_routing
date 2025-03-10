import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import sys
import os

def run(N=5, M=2, MAP_SIZE=100, GRID_STEPS=10, speed_multiplier=1, capacity_multiplier=1,
        random_fleet=True, fleet_composition=None, seed=2, depot_location=(50,50)):
    np.random.seed(seed)

    VEHICLE_CAPACITIES = [6, 15, 40]  # kg
    VEHICLE_VELOCITIES = [170, 80, 40]  # km/h

    VEHICLE_VELOCITIES = [v * speed_multiplier for v in VEHICLE_VELOCITIES]
    VEHICLE_CAPACITIES = [c * capacity_multiplier for c in VEHICLE_CAPACITIES]

    side_array = np.arange(MAP_SIZE)
    x, y = np.meshgrid(side_array, side_array)
    coordinate_grid = np.vstack([x.ravel(), y.ravel()]).T

    # Generate customer locations, keeping them fixed across depot placements
    customer_indices = np.random.choice(range(len(coordinate_grid)), N - 1, replace=False)
    customer_locations = coordinate_grid[customer_indices]

    # Set depot location explicitly
    depot = np.array(depot_location)

    # Combine customers and depot into nodes array
    nodes = np.zeros((N + 1, 2))
    nodes[:N - 1, :] = customer_locations
    nodes[N - 1, :] = depot
    nodes[N, :] = depot  # Duplicate depot node

    # Fleet composition handling
    if random_fleet:
        fleet_list = np.random.randint(0, 3, M)
    else:
        if fleet_composition is None or len(fleet_composition) != M:
            raise ValueError(
                f"fleet_composition must be a list of {M} elements (0=Fixed Wing, 1=Quadcopter, 2=Cargo Blimp).")
        fleet_list = np.array(fleet_composition)

    velocity_list = [VEHICLE_VELOCITIES[x] for x in fleet_list]
    capacity_list = [VEHICLE_CAPACITIES[x] for x in fleet_list]
    fleet_capacity = sum(capacity_list)

    demand_list = [0] * (N + 1)
    rem_fleet_capacity = fleet_capacity
    max_demand = max(2, rem_fleet_capacity // (N - 1) * 7 // 4);

    for i in range(N - 1):
        demand = min(np.random.randint(1, max_demand), rem_fleet_capacity - M - 1)
        demand_list[i] += demand
        rem_fleet_capacity -= demand
        if rem_fleet_capacity == 0:
            break
    demand_list = [int(x) for x in demand_list]

    distance_matrix = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            distance_matrix[i, j] = np.sqrt(
                (nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2
            )

    velocity_matrix = [np.full((N + 1, N + 1), 1) * velocity for velocity in velocity_list]
    traveltime_matrix = np.array([distance_matrix / velocities for velocities in velocity_matrix])

    m = Model('Vehicle Routing')
    m.modelSense = GRB.MINIMIZE
    # Binary decision variable (M x N+1 x N+1) (N+1 because depot counted twice: depotstart and depotend)
    x = m.addVars(range(M), range(N + 1), range(N + 1), vtype=GRB.BINARY, name='x_ijk');

    # Total travel time [h] (i.e. hours that pilots have to be paid for, NOT duration of delivery)
    m.setObjective(
        sum(x[i, j, k] * traveltime_matrix[i, j, k] for i in range(M) for j in range(N) for k in range(N + 1)));

    # 1. Every customer k is visited once by a vehicle and drones also leave from every customer
    for k in range(N - 1):
        m.addConstr(sum(x[i, j, k] for i in range(M) for j in range(N)) == 1);

    # 2. Vehicles i leave at depotStart N and do not come back
    for i in range(M):
        m.addConstr(sum(x[i, N - 1, k] for k in range(N - 1)) == 1);  # yes: from N to k until N-1 (customers)
        m.addConstr(sum(x[i, j, N - 1] for j in range(N)) == 0);  # no: from j to N

    # 3. Vehicles i do not leave at depotEnd N+1 but always come back
    for i in range(M):
        m.addConstr(sum(x[i, N + 1 - 1, k] for k in range(N + 1)) == 0);
        m.addConstr(sum(x[i, j, N + 1 - 1] for j in range(N - 1)) == 1);

    # 4. If a vehicle i goes to a node, it also needs to leave from that node
    for i in range(M):
        for j in range(N - 1):
            m.addConstr(
                sum(x[i, j, k] for k in list(range(N - 1)) + [N] if k != j) - sum(x[i, k, j] for k in range(N)) == 0);
    #                       outgoing arcs from j                    -   incoming arcs to j                == 0
    # for k in range(N-1):
    # m.addConstr(x[i, j, k] + x[i, k, j] < 2);

    # 5. Subtour elimination
    u = m.addVars(range(M), range(N - 1), vtype=GRB.CONTINUOUS, lb=0, ub=N - 1, name="u");
    for i in range(M):
        for j in range(N - 1):
            for k in range(N - 1):
                if j != k:
                    m.addConstr(u[i, j] - u[i, k] + (N - 1) * x[i, j, k] <= N - 2);

    # 6. Vehicle i's capacity is not exceeded.
    for i in range(M):
        m.addConstr(sum(x[i, j, k] * demand_list[k] for j in range(N) for k in range(N - 1)) <= capacity_list[i]);

    m.optimize()

    if m.status == GRB.OPTIMAL:
        total_distance = sum(
            distance_matrix[j][k]
            for i in range(M) for j in range(N + 1) for k in range(N + 1)
            if x[i, j, k].X > 0.5
        )
        selected_edges = {(i, j, k) for i in range(M) for j in range(N + 1) for k in range(N + 1) if x[i, j, k].X > 0.5}
        runtime = m.Runtime

        return {
            "objective_value": m.ObjVal,
            "total_distance": total_distance,
            "fleet": fleet_list,
            "runtime": runtime,
            "depot_location": depot,
            "demand_list": demand_list,
            "capacity_list": capacity_list,
            "customer_locations": customer_locations,
            "solution_x": selected_edges,
        }

    else:
        return {
            "objective_value": None,
            "total_distance": None,
            "fleet": fleet_list,
            "runtime": None,
            "depot_location": depot,
            "customer_locations": customer_locations,
            "solution_x": None
        }


run(5, 2, seed = 42, depot_location = (21, 42))


