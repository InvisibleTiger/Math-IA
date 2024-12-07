import numpy as np

edge_matrix = np.loadtxt('edge_matrix.txt')

rows, cols = edge_matrix.shape

edge_points = []

for x in range(rows):
    for y in range(cols):
        if edge_matrix[x, y] == 1:
            edge_points.append((y, x))  # Store as (x, y)

with open('feature_matrix.txt', 'w') as f:
    for point in edge_points:
        f.write(f"{point}, ")

print("Feature matrix created and saved to 'feature_matrix.txt'")