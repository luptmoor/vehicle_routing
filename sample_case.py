import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42);
MAP_SIZE = 100;
GRID_STEPS = 10;
N = 12;
MAX_DEMAND = 10;

# Generate 100 x 100 km map
side_array = np.arange(MAP_SIZE);
x, y = np.meshgrid(side_array, side_array);
coordinate_list = np.vstack([x.ravel(), y.ravel()]).T;

# Choose N map points as depot and customers
chosen_coordinates = coordinate_list[np.random.choice(range(len(coordinate_list)), N, replace=False)];
depot, customers = chosen_coordinates[0, :], chosen_coordinates[1:, :];

# Generate distance matrix
distance_matrix = np.zeros((N, N));
for i, j in [(i, j) for i in range(N) for j in range(N)]:
    distance_matrix[i, j] = np.sqrt((chosen_coordinates[i, 0] - chosen_coordinates[j, 0])**2 + (chosen_coordinates[i, 1] - chosen_coordinates[j, 1])**2);


# Generate demand list
demand_list = np.random.randint(1, MAX_DEMAND, N-1);


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

