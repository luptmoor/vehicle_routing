from gurobipy import *
import numpy as np
import pandas as pd


def run(N=5, M=2, MAP_SIZE=100, GRID_STEPS=10, speed_multiplier = 1, capacity_multiplier = 1, random_fleet=True, fleet_composition=None, fixed_demand = None, fixed_nodes = None, seed = 2):
    """
    Function to run the model for speed, capacity, fleet size and number of nodes sensitivity
    """

    np.random.seed(seed)

    print(f"Running for N = {N}, M = {M}, seed = {seed}")

    # Vehicle type characteristics
    VEHICLE_CAPACITIES = [6, 15, 40]  # kg
    VEHICLE_VELOCITIES = [170, 80, 40]  # km/h

    VEHICLE_VELOCITIES = [v * speed_multiplier for v in VEHICLE_VELOCITIES]
    VEHICLE_CAPACITIES = [c * capacity_multiplier for c in VEHICLE_CAPACITIES ]

    # Generate 100 x 100 km map
    if fixed_nodes is None:
        side_array = np.arange(MAP_SIZE)
        x, y = np.meshgrid(side_array, side_array)
        coordinate_grid = np.vstack([x.ravel(), y.ravel()]).T
        nodes = np.zeros((N + 1, 2))
        nodes[:N, :] = coordinate_grid[np.random.choice(range(len(coordinate_grid)), N, replace=False)]
        customers, depot = nodes[:N - 1, :], nodes[N - 1, :]
        nodes[N, :] = depot  # Duplicate depot for consistency
    else:
        nodes = fixed_nodes  # Use predefined locations


    # Assign fleet composition
    if random_fleet:
        fleet_list = np.random.randint(0, 3, M)  # Randomly assign vehicle types (0, 1, 2)
    else:
        if fleet_composition is None or len(fleet_composition) != M:
            raise ValueError(f"fleet_composition must be a list of {M} elements (0=Fixed Wing, 1=Quadcopter, 2=Cargo Blimp).")
        fleet_list = np.array(fleet_composition)

    # Count vehicle types
    n_0, n_1, n_2 = np.bincount(fleet_list, minlength=3)

    # Get velocity and capacity for each vehicle
    velocity_list = [VEHICLE_VELOCITIES[x] for x in fleet_list]
    capacity_list = [VEHICLE_CAPACITIES[x] for x in fleet_list]
    fleet_capacity = sum(capacity_list)


    # if fleet_capacity < FIXED_TOTAL_DEMAND:
    #     raise ValueError(
    #         f"Fleet capacity {fleet_capacity} is too small for fixed demand {FIXED_TOTAL_DEMAND}. Increase M or use larger drones.")

    # Generate demand list (N+1) (depot counted twice: as start and end)
    if fixed_demand is None:
        demand_list = [0] * (N + 1)
        rem_fleet_capacity = fleet_capacity
        # Compute a maximum demand based on fleet capacity (as in your original model)
        max_demand = max(2, rem_fleet_capacity // (N - 1) * 7 // 4)
        print(f"Fleet composition = {fleet_list} with capacity {fleet_capacity}")
        for i in range(N - 1):
            # Ensure we don't exceed remaining capacity too much
            demand = min(np.random.randint(1, max_demand), rem_fleet_capacity - M - 1)
            demand_list[i] += demand
            rem_fleet_capacity -= demand
            if rem_fleet_capacity <= 0:
                break
    else:
        demand_list = fixed_demand


    # Generate distance matrix (N+1 x N+1)
    distance_matrix = np.zeros((N + 1, N + 1))
    for i, j in [(i, j) for i in range(N + 1) for j in range(N + 1)]:
        distance_matrix[i, j] = np.sqrt(
            (nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2
        )  # Pythagorean distance

    # Generate travel time matrix (M x N+1 x N+1)
    velocity_matrix = [np.full((N + 1, N + 1), velocity) for velocity in velocity_list]
    traveltime_matrix = np.array([distance_matrix / velocities for velocities in velocity_matrix])

    # Gurobi Model definition
    m = Model('Vehicle Routing')
    m.modelSense = GRB.MINIMIZE

    # Decision variable (M x N+1 x N+1)
    x = m.addVars(range(M), range(N + 1), range(N + 1), vtype=GRB.BINARY, name='x_ijk')

    # Objective function: Minimize total travel time
    m.setObjective(sum(x[i, j, k] * traveltime_matrix[i, j, k] for i in range(M) for j in range(N) for k in range(N + 1)))

    # Constraints
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
            for i in range(M) for j in range(N + 1) for k in range(N + 1))

        runtime = m.Runtime  # Extract computation time

        selected_edges = {(i, j, k) for i in range(M) for j in range(N + 1) for k in range(N + 1) if x[i, j, k].X > 0.5}

        return {
            "objective_value": m.ObjVal,
            "normalized_objective_value": m.ObjVal/ sum(demand_list),
            "total_distance": total_distance,
            "runtime": runtime,
            "fleet": fleet_list,
            "demand_list": demand_list,
            "total_demand": sum(demand_list),
            "solution_x": selected_edges,
            "nodes": nodes,
            "distance_matrix": distance_matrix,
            "velocity_list": velocity_list

        }

    else:
        return {
            "objective_value": None,
            "normalized_objective_value": None,
            "total_distance": None,
            "runtime": None,
            "fleet": fleet_list,
            "demand_list": demand_list,
            "total_demand": sum(demand_list),
            "solution_x": None,
            "nodes": nodes,
            "distance_matrix": distance_matrix,
            "velocity_list": velocity_list
        }

print(run())




