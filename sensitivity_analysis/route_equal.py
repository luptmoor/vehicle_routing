r1 = {(1, 3, 6), (0, 5, 7), (1, 6, 9), (2, 8, 2), (4, 8, 4), (2, 1, 9), (3, 0, 9), (3, 8, 0), (0, 7, 9), (1, 8, 3), (2, 2, 1), (0, 8, 5), (4, 4, 9)}
r2 = {(2, 2, 9), (0, 8, 7), (3, 6, 3), (4, 8, 4), (0, 5, 9), (3, 8, 6), (2, 8, 1), (2, 1, 2), (1, 0, 9), (0, 7, 5), (1, 8, 0), (3, 3, 9), (4, 4, 9)}


def normalize_route(route_set):
    return { (v, min(a, b), max(a, b)) for (v, a, b) in route_set }

# Normalize both sets
normalized_set_1 = normalize_route(r1)
normalized_set_2 = normalize_route(r2)

# Compare the two sets
are_equal = normalized_set_1 == normalized_set_2
print("Are the two route sets equivalent?", are_equal)

print(normalized_set_1)
print(normalized_set_2)