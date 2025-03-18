# Vehicle Routing Problem (VRP) Solver

This repository contains an implementation of the Vehicle Routing Problem (VRP), where a fleet of vehicles must deliver goods to customers while optimizing routes and adhering to constraints.

## 📌 Overview

The VRP formulation is implemented in `sample_case.py`, where:

- The number of customers and vehicles is defined.
- The solution is computed using optimization techniques.
- A verification step ensures that the solution meets all constraints.

Constraint verification tests are implemented in `verification_functions.py` to validate the correctness of the VRP solution.

Visualizations are generated using `plot_results.py`, which plots the vehicle routes for easy visual inspection.

## 📂 Project Structure

```
📁 vehicle_routing/
│-- 📂 results/                      
│-- 📂 sensitivity_analysis/         # Contains sensitivity analysis scripts
│   │-- 📂 figures/                  # Stores generated sensitivity figures
│   │-- 📂 results/                  # Results from sensitivity analysis
│   │-- model_performance.py         # Evaluates model performance
│   │-- plot_sensitivity_fleet_composition.py  # Plots fleet composition sensitivity
│   │-- route_equal.py               # Manual route comparison 
│   │-- route_plot.py                # Manual route plot
│   │-- sample_case_2.py             # Model as a function for sensitivity analysis
│   │-- sample_case_3.py             # Model as a function for depot sensitivity analysis
│   │-- sensitivity_characteristics.py         
│   │-- sensitivity_characteristics_amount.py  
│   │-- sensitivity_characteristics_amount_v2.py    # Sensitivity of normalized objective value for speed, capacity and demand
│   │-- sensitivity_characteristics_v2.py           # Sensitivity for solution change for speed, infeasibility for capacity and demand
│   │-- sensitivity_depot.py          # Depot sensitivity analysis
│   │-- sensitivity_fleet_composition.py  # Fleet composition sensitivity
│   │-- sensitivity_fleet_size.py     # Fleet size sensitivity
│   │-- sensitivity_nodes.py          # Number of nodes sensitivity analysis
│-- plot_results.py                   # Visualization of vehicle routes
│-- README.md                          # Project documentation
│-- sample_case.py                     # Main VRP problem formulation and verification
│-- vehicle_routing.lp                  # VRP model definition
│-- verification_functions.py           # Constraint validation functions

```

## 🛠️ VRP Constraint Verification

To ensure that the computed solution is valid, the following five constraints are checked in `verification_functions.py`:

1️⃣ **Depot Departure** – Each vehicle must leave the depot exactly once.
2️⃣ **Depot Arrival** – Each vehicle must return to the depot exactly once.
3️⃣ **Customer Flow** – If a vehicle arrives at a customer, it must also leave that customer.
4️⃣ **Customer Coverage** – Every customer must be visited exactly once.
5️⃣ **Capacity Constraint** – The total demand of customers assigned to a vehicle must not exceed its capacity.

🔹 If any constraint is violated, an error is raised.

## Sensitivity Analysis 
The sensitivity analysis evaluates how variations in parameters affect the performance - measured by the normalized objective value - of the Vehicle Routing Problem (VRP).
Analysis was done for the following parameters:
- Fleet size M
- Number of nodes N
- Fleet composition
- Drone characteristics: speed and capacity, demand
- Depot location

## 📊 Visualization

The `plot_results.py` script plots the optimized routes.

It displays:

- The depot and customer locations.
- The paths taken by each vehicle.
- The demand distribution across customers.

🖥️ Results depend on the input values defined in `sample_case.py`.

## 

## 📌 Notes

- This project uses **Gurobi** as the optimization solver.
- Ensure you have a valid Gurobi license to run the optimization model.
- The number of vehicles and customers can be modified in `sample_case.py`.

## 🛠️ Future Improvements

- Implement real-time constraint checking inside the model.
- Add alternative solvers for non-Gurobi users.
- Support for different vehicle types and constraints.

## 📜 License

This project is licensed under the **MIT License**.

## 📬 Contact

For questions or contributions, feel free to reach out or submit an issue.
