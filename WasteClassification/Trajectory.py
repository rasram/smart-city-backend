import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize
import random

# SCARA Robot Parameters
L1, L2 = 0.5, 0.4  # Arm lengths (meters)
workspace_radius = L1 + L2

# Check if the target is within reach
def is_within_workspace(target):
    return np.linalg.norm(target) <= workspace_radius

# Forward Kinematics
def forward_kinematics(theta1, theta2):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

# 1️⃣ Analytical Inverse Kinematics
def analytical_ik(target):
    x, y = target
    d = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    
    if abs(d) > 1:
        print("Target is out of reach for analytical solution!")
        return None

    theta2 = np.arccos(d)  
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return np.array([theta1, theta2])

# Cost Function for Optimization
def cost_function(angles, target):
    x, y = forward_kinematics(angles[0], angles[1])
    return np.linalg.norm(target - np.array([x, y]))

# 2️⃣ Genetic Algorithm for IK
def genetic_algorithm(target, population_size=100, generations=50, mutation_rate=0.1):
    population = np.random.uniform(-np.pi, np.pi, (population_size, 2))
    
    for _ in range(generations):
        fitness = np.array([cost_function(ind, target) for ind in population])
        best_indices = np.argsort(fitness)[:population_size // 2]  # Select top individuals
        parents = population[best_indices]

        offspring = []
        for _ in range(population_size // 2):
            p1, p2 = random.choice(parents), random.choice(parents)
            child = (p1 + p2) / 2  # Crossover
            if random.random() < mutation_rate:
                child += np.random.uniform(-0.1, 0.1, size=2)  # Mutation
            offspring.append(child)

        population = np.vstack((parents, offspring))  # New generation
    
    best_solution = population[np.argmin([cost_function(ind, target) for ind in population])]
    return best_solution

# 3️⃣ L-BFGS-B for Fine-tuning after GA
def optimize_angles(initial_angles, target):
    result = minimize(cost_function, initial_angles, args=(target,), method='L-BFGS-B', bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
    return result.x

# Get Target Input
x_target, y_target = map(float, input("Enter target (x, y) in meters: ").split())
target = np.array([x_target, y_target])

if not is_within_workspace(target):
    print("Target is out of the workspace!")
    exit()

# Solve Inverse Kinematics
analytical_solution = analytical_ik(target)
ga_solution = genetic_algorithm(target)
optimized_angles = optimize_angles(ga_solution, target)

print("\nIK Solutions:")
print("Analytical:", analytical_solution)
print("GA Solution:", ga_solution)
print("GA + L-BFGS-B Optimized:", optimized_angles)

# Choose which solution to visualize
if analytical_solution is not None:
    trajectory = np.linspace(analytical_solution, optimized_angles, num=50)
else:
    trajectory = np.linspace(ga_solution, optimized_angles, num=50)

# Compute End-effector Path
end_effector_trajectory = np.array([forward_kinematics(theta1, theta2) for theta1, theta2 in trajectory])

# Plot Setup
fig, ax = plt.subplots()
ax.set_xlim(-workspace_radius, workspace_radius)
ax.set_ylim(-workspace_radius, workspace_radius)
ax.set_xlabel("X-axis (meters)")
ax.set_ylabel("Y-axis (meters)")
ax.set_title("SCARA Robot Inverse Kinematics with Trajectory")
ax.grid(True)

# Plot Target
ax.scatter([x_target], [y_target], color="red", marker="x", label="Target")

# Plot Full Trajectory
ax.plot(end_effector_trajectory[:, 0], end_effector_trajectory[:, 1], 'g--', lw=1, label="Trajectory")

# Initialize Arms and Moving Dot
arm1, = ax.plot([], [], 'ro-', lw=2)
arm2, = ax.plot([], [], 'bo-', lw=2)
end_effector_dot, = ax.plot([], [], 'go', markersize=5)

def update(frame):
    theta1, theta2 = trajectory[frame]
    x1, y1 = L1 * np.cos(theta1), L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    
    arm1.set_data([0, x1], [0, y1])
    arm2.set_data([x1, x2], [y1, y2])
    
    end_effector_dot.set_data([x2], [y2])  # Move dot along trajectory

    return arm1, arm2, end_effector_dot

ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=True)
plt.legend()
plt.show()
