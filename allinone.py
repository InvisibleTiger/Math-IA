import numpy as np
import cv2
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

image_path = 'nyc.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_matrix = np.array(image)
np.savetxt('image_matrix.txt', image_matrix, fmt='%d')

image = cv2.flip(image, 0)  # fix flip
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gradient_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
gradient_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_magnitude = np.uint8(gradient_magnitude)
np.savetxt('gradient_matrix.txt', gradient_magnitude, fmt='%d')

gradient_matrix = np.loadtxt('gradient_matrix.txt')
threshold = 150
edge_matrix = np.where(gradient_matrix >= threshold, 1, 0)
np.savetxt('edge_matrix.txt', edge_matrix, fmt='%d')

rows, cols = edge_matrix.shape
edge_points = [(y, x) for x in range(rows) for y in range(cols) if edge_matrix[x, y] == 1]
with open('feature_matrix.txt', 'w') as f:
    for point in edge_points:
        f.write(f"{point}, ")

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
    return f"y = {m:.2f}x + {b:.2f} {{{lower_x_bound} < x < {upper_x_bound}}}"

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

def parse_line(line):
    match = re.match(r"y = ([-\d.]+)x \+ ([-\d.]+) \{(\d+) < x < (\d+)\}", line)
    if match:
        m = float(match.group(1))
        b = float(match.group(2))
        x_min = int(match.group(3))
        x_max = int(match.group(4))
        return m, b, x_min, x_max
    return None

lines = []
with open("regression_lines.txt", "r") as file:
    lines = file.readlines()

plt.figure(figsize=(8, 8))
for line in lines:
    parsed_line = parse_line(line.strip())
    if parsed_line:
        m, b, x_min, x_max = parsed_line
        x_values = list(range(x_min, x_max + 1))
        y_values = [m * x + b for x in x_values]
        plt.plot(x_values, y_values, marker='o', label=f"y = {m}x + {b}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression Lines")
plt.grid(True)
plt.show()
