import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import sys
import os
import time
import seaborn as sns
import pdb
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verification_functions import verify_all_constraints


# Scenario Parameters
#np.random.seed(42);
MAP_SIZE = 100; # km (in a square)
GRID_STEPS = 10

# Vehicle type characteristics [Fixed wing (fast but little capactiy), quadcopter (balanced), cargo blimp (slow but high capacity)]
VEHICLE_CAPACITIES = [6, 15, 40]; # kg
VEHICLE_VELOCITIES = [170, 80, 40]; # km/h


def run_optimisation(N,M):

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



    m.optimize()

    # # Convert solution to np.array for output print
    solution = np.zeros((M, N+1, N+1));

    for i in range(M):
        for j in range(N+1):
            for k in range(N+1):            
                solution[i, j, k] = x[i, j, k].X;
    
    verify_all_constraints(solution, demand_list, capacity_list) # Verify all constraints
    best_time = round(m.ObjVal, 2)  # Total time needed
    return best_time

   

#total feastibility test


N = 16   # Max number of customers + 1x depot
M = 13  # Max number of vehicles
seed_values = [42, 1337, 9001, 1234, 421, 1, 2342, 7]  # Seed values
previous_file = None
test_matrix = np.zeros((4, N, M+1))  # Initialize the test matrix


# test matrix 0 = feasibility
# test matrix 1 = elapsed time
# test matrix 2 = best time
# test matrix 3 = infeasibility

#  **Check if the last seed & last combo file exist**
expected_file = f"feasibility_test_matrix_lastseed{seed_values[-1]}_lastcombo({N-1}, {M}).pkl"
if os.path.exists(expected_file):
    with open(expected_file, "rb") as f:
        test_matrix = pickle.load(f)  # Load test matrix
    print(f"✅ Resuming from saved file: {expected_file}")
    # Skip computation & go straight to results
else:
    # 🔄 **Run Optimization If Not Resumed**
    for s in seed_values:
        seed_progress = s
        np.random.seed(s)  # Set seed for reproducibility
        print(f"🚀 Processing seed {s}...")

        for n in range(2,N):  
            for m in range(M+1):  
                start_time = time.time()
                try:
                    solution = run_optimisation(n, m)  # Run optimization
                    test_matrix[0, n, m] += 1  # Store feasibility
                    print(f"✅ Feasibility test for {n} customers, {m} vehicles PASSED")
                    test_matrix[2, n, m] += solution  # Store best time
                    end_time = time.time()
                    test_matrix[1, n, m] = end_time - start_time 

                except AttributeError as e:
                    test_matrix[3, n, m] += 1  # Store infeasibility
                    print(f"⚠️ Infeasible for {n} customers, {m} vehicles: {e}")

                except Exception as e:
                    test_matrix[3, n, m] += 1  # Store infeasibility
                    print(f"❌ Unexpected error for {n} customers and {m} vehicles: {e}")
                    continue  

               

            last_combo = (n, m)  

            # Define new file name
            new_file = f"feasibility_test_matrix_lastseed{seed_progress}_lastcombo{last_combo}.pkl"

            # Remove the previous pickle file before saving the new one
            if previous_file and os.path.exists(previous_file):
                os.remove(previous_file)
                print(f"🗑️ Removed previous file: {previous_file}")

            # Save the new progress
            with open(new_file, "wb") as f:
                pickle.dump(test_matrix, f)
            print(f"💾 Progress saved: {new_file}")

            # Update previous_file
            previous_file = new_file

print(" Feasibility test matrix saved!")

# Ensure safe division without NaN, Inf, or divide-by-zero errors
best_average_time_matrix = np.divide(
    test_matrix[2], 
    test_matrix[0], 
    where=(test_matrix[0] > 0),  # Perform division only where feasible
    out=np.zeros_like(test_matrix[2])  # Fill invalid places with 0
)

compute_time_matrix = np.divide(
    test_matrix[1], 
    test_matrix[0], 
    where=(test_matrix[0] > 0),  # Only divide where feasible
    out=np.full_like(test_matrix[1], np.nan)  # Fill invalid places with NaN
)

# Optional: Replace NaN values with 0 for cleaner visualization
compute_time_matrix = np.nan_to_num(compute_time_matrix, nan=0.0)


# Normalize by number of customers (n-wise division)
best_time_n_customer_ratio = np.zeros((N, M+1))
for n in np.arange(2,N):
    for m in np.arange(M+1):
        if best_average_time_matrix[n, m] > 0:  # Only divide if time is > 0
                best_time_n_customer_ratio[n, m] = best_average_time_matrix[n, m] / n
        else:
            best_time_n_customer_ratio[n, m] = 0  # Assign 0 if not valid

best_time_m_customer_ratio = np.zeros((N, M+1))
for n in np.arange(2,N):
    for m in np.arange(M+1):   
        if best_average_time_matrix[n, m] > 0:
            best_time_m_customer_ratio[n, m] = best_average_time_matrix[n, m] / m
        else:
            best_time_m_customer_ratio[n, m] = 0

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)



#  **Feasibility Heatmap**
plt.figure(figsize=(8, 6))
sns.heatmap(
    test_matrix[0, :, :],  
    annot=True,  
    cmap="RdYlGn",  
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    vmin=0,
    vmax=np.max(test_matrix[0, :, :])  
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Customers")
plt.title("Feasibility Test Matrix (Feasible Count)")
plt.savefig(os.path.join(results_dir, "feasibility_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

#  **Best Average Time Heatmap**
plt.figure(figsize=(8, 6))
sns.heatmap(
    best_average_time_matrix,  
    annot=True,  
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="black",
    cbar=True
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Customers")
plt.title("Best Average Solution Time (s)")
plt.savefig(os.path.join(results_dir, "best_average_time_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Replace values smaller than 0.009 with NaN
filtered_compute_time_matrix = np.where(compute_time_matrix < 0.009, np.nan, compute_time_matrix)

#  **Average Computation Time Heatmap**
plt.figure(figsize=(8, 6))
sns.heatmap(
    filtered_compute_time_matrix,  
    annot=True,  
    cmap="Blues",
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    fmt=".2f",
    annot_kws={"size": 10},
    mask=np.isnan(filtered_compute_time_matrix)  # Hide values < 0.009
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Customers")
plt.title("Average Computation Time (s)")
plt.savefig(os.path.join(results_dir, "average_computation_time_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()


#  **Heatmap for Best Time per Customer Ratio**
plt.figure(figsize=(8, 6))
sns.heatmap(
    best_time_n_customer_ratio,  
    annot=True,  
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    fmt=".2f",
    annot_kws={"size": 10},
    mask=best_time_n_customer_ratio == 0
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Customers")
plt.title("Best Average Time per Customer (s/customer)")
plt.savefig(os.path.join(results_dir, "best_time_per_customer_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

#  **Heatmap for Best Time per Vehicle Ratio**
plt.figure(figsize=(8, 6))
sns.heatmap(
    best_time_m_customer_ratio,  
    annot=True,  
    cmap="coolwarm",
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    fmt=".2f",
    annot_kws={"size": 10},
    mask=best_time_m_customer_ratio == 0
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Customers")
plt.title("Best Average Time per Vehicle (s/vehicle)")
plt.savefig(os.path.join(results_dir, "best_time_per_vehicle_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()