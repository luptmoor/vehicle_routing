"""
Functions to verify the constrains
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from sample_case import nodes, customers, depot, solution, MAP_SIZE, GRID_STEPS, M, N, demand_list, capacity_list, traveltime_matrix

def verify_customer_visit_constraint(solution): 
    """
    Constraint 1: Verify that every customer is visited exactly once.
    """
    M, N, _ = solution.shape # M is amount of vehicles, N is amount of customers + depot-start + depot-end
    for k in range(N-2):  # For each customer
        visits = solution[:, k, :].sum() # Total visits by all vehicles
        if visits != 1:
            raise ValueError(f"Constraint 1 FAILED: Customer {k} is visited {visits} times, expected 1.")
    print("Constraint 1 PASSED: Every customer is visited exactly once.")
    


def verify_depot_start_constraint(solution):
    """
    Constraint 2: Verify that veicles leave from the depot start exactly once.
    """
    M, N, _ = solution.shape # M is amount of vehicles, N is amount of customers + depot-start + depot-end
    for i in range(M):  # For each vehicle
        depot_start = solution[i, -2, :].sum() # Vehicle leaves depot at the start
        if depot_start != 1:
            raise ValueError(f"Constraint 2 FAILED: {depot_start} vehicles did not leave depot_start properly.")
    print(f"Constraint 2 PASSED: All {M} vehicles successfully departed from the depot at the start.") 






def verify_depot_end_constraint(solution):
    """
    Constraint 3: Verify that vehicles return to the depot end exactly once.
    """
    M, N, _ = solution.shape  # M: Number of vehicles, N: Customers + depot-start + depot-end

    # Summing up all returns to the depot at the end
    for i in range(M):
        depot_end = solution[i, :, -1].sum()  # -1 selects the last row (depot end)
        if depot_end != 1:
                raise ValueError(f"Constraint 3 FAILED: {depot_end} vehicles did not return to the depot_end properly (expected {M}).")

    print(f"Constraint 3 PASSED: All {M} vehicles successfully returned to the depot at the end.")

    

def verify_vehicle_flow_constraint(solution):
    """
    Constr aint 4: Verify that if a vehicle enters a customer node, it must also leave.
    """
    M, N, _ = solution.shape # M is amount of vehicles, N is amount of customers + depot-start + depot-end
    for i in range(M):  # For each vehicle
        for j in range(N-2):  # For each customer
            incoming = solution[i, :, j].sum()
            outgoing = solution[i, j, :].sum()
            if outgoing != incoming:
                raise ValueError(f"Constraint 4 FAILED: Flow conservation violated at customer {j} for vehicle {i}.")
            if incoming > 1 or outgoing > 1:
                raise ValueError(f"Constraint 4 FAILED: Vehicle {i} has more than one incoming or outgoing route at customer {j}.")
    print("Constraint 4 PASSED: Flow conservation at each node.")



def verify_capacity_constraint(solution, demand_list, capacity_list):
    """
    Constraint 6: Verify that no vehicle exceeds its capacity.
    """
    M, N, _ = solution.shape # M is amount of vehicles, N is amount of customers + depot-start + depot-end
    for i in range(M):  # For each vehicle
        customer_visit_indexes = np.where(solution[i,:,:].any(axis=0))[0] #-1 because it always arrives at the last node
        #print(f"columns_with_ones for vehichle {i} = {   customer_visit_indexes}")
        #print(f"demand list = {demand_list}")
        total_load = np.sum(np.array(demand_list)[customer_visit_indexes])
        if total_load > capacity_list[i]:
            raise ValueError(f"Constraint 6 FAILED: Vehicle {i} exceeds capacity! Load: {total_load}, Capacity: {capacity_list[i]}.")
    print("Constraint 6 PASSED: Vehicle capacities not exceeded.")


def verify_binary_variables(solution):
    """
    Verify that all variables are binary
    """
    if not np.all(np.isin(solution, [0, 1])):
        raise ValueError("Binary constraint FAILED: All variables must be binary.")
    print("Binary constraint PASSED: All variables are binary.")




def verify_all_constraints(solution, demand_list, capacity_list):
    """
    Verify all constraints
    """
    verify_customer_visit_constraint(solution)
    verify_depot_start_constraint(solution)
    verify_depot_end_constraint(solution)
    verify_vehicle_flow_constraint(solution)
    verify_capacity_constraint(solution, demand_list, capacity_list)
    verify_binary_variables(solution)
    print("All constraints PASSED.")
    return True

#verify_all_constraints(solution, demand_list, capacity_list)