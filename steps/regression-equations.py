import numpy as np
import re
from sklearn.linear_model import LinearRegression

with open("feature_matrix.txt", "r") as f:
    points_text = f.read()

points = [(int(x), int(y)) for x, y in re.findall(r"\((\d+),\s*(\d+)\)", points_text)]
point_set = set(points)

def get_neighbors(point):
    x, y = point
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

def regression_line(point1, point2):
    x_values = np.array([point1[0], point2[0]]).reshape(-1, 1)
    y_values = np.array([point1[1], point2[1]])
    model = LinearRegression().fit(x_values, y_values)
    m = model.coef_[0]
    b = model.intercept_
    lower_x_bound = min(point1[0], point2[0])
    upper_x_bound = max(point1[0], point2[0])
    return f"y = {m:.2f}x + {b:.2f} {{{lower_x_bound} < x < {upper_x_bound}}}" # might have to change to ≤ to fit vertical lines

lines = []
for point in points:
    neighbors = get_neighbors(point)
    for neighbor in neighbors:
        if neighbor in point_set:
            line_eq = regression_line(point, neighbor)
            lines.append(line_eq)

with open("regression_lines.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

print("Regression lines saved to regression_lines.txt")