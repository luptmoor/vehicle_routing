import matplotlib.pyplot as plt
import networkx as nx

edges = {(1, 3, 6), (0, 5, 7), (1, 6, 9), (2, 8, 2), (4, 8, 4), (2, 1, 9), (3, 0, 9), (3, 8, 0), (0, 7, 9), (1, 8, 3), (2, 2, 1), (0, 8, 5), (4, 4, 9)}

edges = {(2, 2, 9), (0, 8, 7), (3, 6, 3), (4, 8, 4), (0, 5, 9), (3, 8, 6), (2, 8, 1), (2, 1, 2), (1, 0, 9), (0, 7, 5), (1, 8, 0), (3, 3, 9), (4, 4, 9)}
# Adjusted plot to merge nodes 8 and 9 as the depot

# Create directed graph
G = nx.DiGraph()

# Adjust the edges to merge 8 and 9
adjusted_edges = set()
for i, j, k in edges:
    j = 8 if j in {8, 9} else j  # Merge nodes 8 and 9
    k = 8 if k in {8, 9} else k
    adjusted_edges.add((i, j, k))

# Add edges
for i, j, k in adjusted_edges:
    G.add_edge(j, k, label=f"V{i}")

# Position nodes using spring layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray", font_size=10)
edge_labels = {(j, k): f"V{i}" for i, j, k in adjusted_edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Vehicle Movements (i, j → k) with Merged Depot")
plt.show()
