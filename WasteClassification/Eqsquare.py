import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import random
import pandas as pd

# === Parameters ===
n_joints = 4
N = 100
T = 2.0
h = T / (N - 1)

lambda_energy = 9.0
lambda_jerk = 20.0
lambda_time = 2.0
rho = 10.0

v_max = 2.5
a_max = 5.0

q_start = np.array([0.0, 0.0, 0.1, 0.0])

# Define possible q_end positions (7 possible ones for selection)
q_end_positions = [
    np.array([0.5, 0.3, 0.05, 0.2]), 
    np.array([0.6, 0.4, 0.06, 0.3]), 
    np.array([0.7, 0.5, 0.07, 0.4]),  # Example for position 3
    np.array([0.8, 0.6, 0.08, 0.5]),  # Example for position 4
    np.array([0.9, 0.7, 0.09, 0.6]),  # Example for position 5
    np.array([1.0, 0.8, 0.1, 0.7]),   # Example for position 6
    np.array([1.1, 0.9, 0.11, 0.8]),  # Example for position 7
    np.array([1.2, 1.0, 0.12, 0.9])   # Example for position 8
]

# Prompt user to select which q_end position to use
print("Please choose the q_end position from the following options:")
for i, q_end_option in enumerate(q_end_positions, 1):
    print(f"{i}: {q_end_option}")

# Get the user's selection
selection = int(input("Enter a number between 1 and 7 to choose q_end: ")) - 1

# Ensure the input is valid
if 0 <= selection < len(q_end_positions):
    q_end = q_end_positions[selection]
    print(f"You selected q_end: {q_end}")
else:
    print("Invalid selection. Using default q_end.")
    q_end = q_end_positions[0]  # Default to the first option if invalid input


# === Genetic Algorithm Parameters ===
pop_size = 100  # reduced for speed
generations = 50  # reduced for speed
elite_frac = 0.2
mutation_rate_init = 0.4
mutation_rate_final = 0.1

np.random.seed(42)
random.seed(42)

def generate_individual(q_end):
    t = np.linspace(0, 1, N)
    traj = np.zeros((n_joints, N))
    for i in range(n_joints):
        traj[i] = (2 * t**3 - 3 * t**2 + 1) * q_start[i] + (-2 * t**3 + 3 * t**2) * q_end[i]
    traj += 0.005 * np.random.randn(*traj.shape)
    traj[:, 0] = q_start
    traj[:, -1] = q_end
    return traj

def mutate(traj, rate, q_end):
    mutation = rate * np.random.randn(*traj.shape)
    traj_new = traj + mutation
    traj_new[:, 0] = q_start
    traj_new[:, -1] = q_end
    return traj_new

def crossover(parent1, parent2, q_end):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    child[:, 0] = q_start
    child[:, -1] = q_end
    return child

def compute_cost(q, penalty_weight=1000):
    cost_energy = 0.0
    cost_jerk = 0.0
    penalty = 0.0

    for k in range(N - 1):
        dq = (q[:, k + 1] - q[:, k]) / h
        cost_energy += np.sum(dq**2)
        penalty += np.sum(np.maximum(0, np.linalg.norm(dq) - v_max)**2)

    for k in range(1, N - 2):
        jerk = (q[:, k + 2] - 3 * q[:, k + 1] + 3 * q[:, k] - q[:, k - 1]) / h**3
        cost_jerk += np.sum(jerk**2)

    for k in range(1, N - 1):
        ddq = (q[:, k + 1] - 2 * q[:, k] + q[:, k - 1]) / h**2
        penalty += np.sum(np.maximum(0, np.linalg.norm(ddq) - a_max)**2)

    return lambda_energy * h * cost_energy + lambda_jerk * h * cost_jerk + lambda_time * T + penalty_weight * penalty

# === Initial Population ===
population = [generate_individual(q_end) for _ in range(pop_size)]
cost_history = []

for gen in range(generations):
    mutation_rate = mutation_rate_init * (1 - gen / generations) + mutation_rate_final * (gen / generations)
    costs = [compute_cost(ind) for ind in population]
    ranked = sorted(zip(costs, population), key=lambda x: x[0])
    elites = [ind for _, ind in ranked[:int(elite_frac * pop_size)]]

    new_population = elites.copy()
    while len(new_population) < pop_size:
        p1, p2 = random.sample(elites, 2)
        child = crossover(p1, p2, q_end)
        if np.random.rand() < mutation_rate:
            child = mutate(child, mutation_rate, q_end)
        if not any(np.allclose(child, other) for other in new_population):
            new_population.append(child)

    population = new_population
    best_cost = compute_cost(ranked[0][1])
    cost_history.append(best_cost)

# Best individual for initial guess
q = ranked[0][1]
z = np.copy(q)
u = np.zeros((n_joints, N))

def projection(z):
    z_proj = np.copy(z)
    for k in range(N - 1):
        dq = (z_proj[:, k + 1] - z_proj[:, k]) / h
        dq = np.clip(dq, -v_max, v_max)
        z_proj[:, k + 1] = z_proj[:, k] + h * dq

    for k in range(1, N - 1):
        ddq = (z_proj[:, k + 1] - 2 * z_proj[:, k] + z_proj[:, k - 1]) / h**2
        ddq = np.clip(ddq, -a_max, a_max)
        z_proj[:, k] = 0.5 * (z_proj[:, k - 1] + z_proj[:, k + 1] - h**2 * ddq)

    z_proj[:, 0] = q_start
    z_proj[:, -1] = q_end
    return z_proj

# === ADMM Loop ===
max_iters = 100
tolerance = 1e-4

for it in range(max_iters):
    q = (z - u)
    q[:, 0] = q_start
    q[:, -1] = q_end

    z_old = np.copy(z)
    z = projection(q + u)
    u = u + q - z

    primal_residual = np.linalg.norm(q - z)
    dual_residual = np.linalg.norm(z - z_old)

    if primal_residual < tolerance and dual_residual < tolerance:
        break

# === Post-process with Spline ===
time = np.linspace(0, T, N)
q_smooth = np.zeros_like(q)
for i in range(n_joints):
    spline = CubicSpline(time, q[i])
    q_smooth[i] = spline(time)

# === Apply Savitzky-Golay Smoothing ===
window_length = 91  # Odd number
polyorder = 3
q_smooth = savgol_filter(q_smooth, window_length, polyorder, axis=1)
dq = np.gradient(q_smooth, h, axis=1)
ddq = np.gradient(dq, h, axis=1)
jerk = np.gradient(ddq, h, axis=1)

# === Apply Savitzky-Golay Filtering to Derivatives ===
dq_smooth = savgol_filter(dq, window_length, polyorder, axis=1)  # Smoothing velocity
ddq_smooth = savgol_filter(ddq, window_length, polyorder, axis=1)  # Smoothing acceleration
jerk_smooth = savgol_filter(jerk, window_length, polyorder, axis=1)  # Smoothing jerk

# === Plotting ===
titles = ['Position', 'Velocity', 'Acceleration', 'Jerk']
data = [q_smooth, dq_smooth, ddq_smooth, jerk_smooth]

fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
for i, (title, d) in enumerate(zip(titles, data)):
    ax = axs1[i // 2, i % 2]
    for j in range(n_joints):
        ax.plot(time, d[j], label=f'Joint {j+1}')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend()
fig1.tight_layout()

# === GA Convergence Plot ===
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(cost_history, marker='o')
ax2.set_title("GA Convergence")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Cost")
ax2.grid(True)
fig2.tight_layout()

plt.show()

# Create a DataFrame to store the joint positions over time
# Assuming 'q_smooth' is the final smooth trajectory matrix
time = np.linspace(0, T, N)

# Create a DataFrame for each joint trajectory
df = pd.DataFrame(q_smooth.T, columns=[f'Joint {i+1}' for i in range(n_joints)])
df['Time'] = time  # Add time as a column

# Save the DataFrame to a CSV file
output_file = "trajectory.csv"
df.to_csv(output_file, index=False)

print(f"Trajectory saved to {output_file}")


print("Loading the animation of the Trajectory")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Load the trajectory
df = pd.read_csv("trajectory.csv")
time = df["Time"].values
q = df[[f"Joint {i+1}" for i in range(4)]].values.T  # Shape: (4, N)

# Robot parameters
L1 = 0.5  # Length of link 1
L2 = 0.3  # Length of link 2

# Forward Kinematics for SCARA (R-R-P-R)
def forward_kinematics(q1, q2, q3, q4):
    # Base
    x0, y0, z0 = 0, 0, 0

    # Joint 1 end
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    z1 = 0

    # Joint 2 end
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    z2 = 0

    # Prismatic joint moves vertically (along -Z)
    x3 = x2
    y3 = y2
    z3 = -q3  # Typically prismatic SCARA moves down

    # End-effector (no spatial offset for joint 4 here)
    x4 = x3
    y4 = y3
    z4 = z3

    return np.array([[x0, x1, x2, x3, x4],
                     [y0, y1, y2, y3, y4],
                     [z0, z1, z2, z3, z4]])

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.5, 0.5)
ax.set_title("4-DOF SCARA Robot Trajectory")
line, = ax.plot([], [], [], 'o-', lw=3)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    q1, q2, q3, q4 = q[:, frame]
    pos = forward_kinematics(q1, q2, q3, q4)
    line.set_data(pos[0], pos[1])
    line.set_3d_properties(pos[2])
    return line,

ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                    blit=True, interval=30)

plt.show()

from matplotlib.animation import PillowWriter

writer = PillowWriter(fps=30)
ani.save("scara_trajectory.gif", writer=writer)

