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
│-- sample_case.py                # Main VRP problem formulation and solution verification
│-- verification_functions.py     # Constraint validation functions
│-- plot_results.py               # Visualization of routes
│-- total_verification.py         # Itterates over many different scenarios and performs verification and sensitivity analysis
│-- feasibility_test_matrix_lastseed7-lastcombo(12, 13).pkl        # This is pre computed matrix for 8 different seeds 
│-- README.md                     # Project documentation


```

## 🛠️ VRP Constraint Verification

To ensure that the computed solution is valid, the following five constraints are checked in `verification_functions.py`:

1️⃣ **Depot Departure** – Each vehicle must leave the depot exactly once.
2️⃣ **Depot Arrival** – Each vehicle must return to the depot exactly once.
3️⃣ **Customer Flow** – If a vehicle arrives at a customer, it must also leave that customer.
4️⃣ **Customer Coverage** – Every customer must be visited exactly once.
5️⃣ **Capacity Constraint** – The total demand of customers assigned to a vehicle must not exceed its capacity.

🔹 If any constraint is violated, an error is raised.

## 📊 Total Verification & Feasibility Analysis

The script `total_verification.py` iterates over different problem scenarios, testing feasibility and optimizing routes under various conditions. It saves results in structured matrices and produces multiple visual outputs.

### Key Functionalities:

- **Feasibility Testing** – Runs multiple simulations with different seeds to test solution feasibility.
- **Performance Metrics** – Stores computation time, best solutions, and infeasibility counts.
- **Automatic Progress Saving** – Uses pickle to save and resume long computations.

### 🔍 Output Visualizations

The results include:

- **Feasibility Heatmap** – Displays how many times a scenario was feasible.
- **Best Average Solution Time** – Heatmap of the optimal solution time.
- **Computation Time Analysis** – Shows the average runtime per scenario.
- **Time Per Customer & Vehicle** – Evaluates solution efficiency per customer and per vehicle.

📁 All results are saved in the `results/` folder.
 

## 📊 Visualization

The `plot_results.py` script plots the optimized routes.

It displays:

- The depot and customer locations.
- The paths taken by each vehicle.
- The demand distribution across customers.

🖥️ Results depend on the input values defined in `sample_case.py`.
  The plot will be saved in a folder called 'results'


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
