import matplotlib.pyplot as plt
import re

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