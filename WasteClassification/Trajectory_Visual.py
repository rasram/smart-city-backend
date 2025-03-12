import numpy as np
import matplotlib.pyplot as plt

# Given link lengths
L1 = 0.5  # Length of Link 1
L2 = 0.4  # Length of Link 2

# Optimized joint angles from GA
optimized_trajectory = np.array([
    [-0.17953312,  1.98231341],
    [-4.83672954,  4.38691937],
    [ 1.42291011, -1.80757067],
    [ 1.3960342,  -1.71574538],
    [-0.02192139,  1.62019879],
    [ 1.06222641, -0.92013862],
    [ 0.06971221, -4.86857872],
    [ 0.12019902,  1.30214288],
    [ 1.2069928,  -1.18073291],
    [ 0.23520025,  1.04720063]
])

# Compute end-effector positions
x_positions = []
y_positions = []

for theta1, theta2 in optimized_trajectory:
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    x_positions.append(x)
    y_positions.append(y)

# Plot the trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_positions, y_positions, 'ro-', markersize=5, label="End-Effector Path")

# Mark start and end points
plt.scatter(x_positions[0], y_positions[0], color='green', s=100, label="Start")
plt.scatter(x_positions[-1], y_positions[-1], color='blue', s=100, label="End")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Optimized End-Effector Trajectory")
plt.legend()
plt.grid(True)
plt.show()
