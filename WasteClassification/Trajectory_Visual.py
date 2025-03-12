import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# SCARA Robot Parameters
L1, L2 = 0.5, 0.4  # Arm lengths (meters)
workspace_radius = L1 + L2

def is_within_workspace(target):
    return np.linalg.norm(target) <= workspace_radius

# Inverse Kinematics
def inverse_kinematics(target):
    x, y = target
    d = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(d) > 1:
        return None  # No solution exists
    
    theta2 = np.arctan2(np.sqrt(1 - d**2), d)  # Elbow up configuration
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return np.array([theta1, theta2])

# Interactive Target Input
x_target, y_target = map(float, input("Enter target (x, y) in meters: ").split())
target = np.array([x_target, y_target])

if not is_within_workspace(target):
    print("Target is out of the workspace!")
    exit()

# Solve Inverse Kinematics
angles = inverse_kinematics(target)
if angles is None:
    print("No valid inverse kinematics solution!")
    exit()

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(-workspace_radius, workspace_radius)
ax.set_ylim(-workspace_radius, workspace_radius)
ax.set_xlabel("X-axis (meters)")
ax.set_ylabel("Y-axis (meters)")
ax.set_title("SCARA Robot Inverse Kinematics Simulation")
ax.grid(True)
ax.scatter([x_target], [y_target], color="red", marker="x", label="Target")

arm1, = ax.plot([], [], 'ro-', lw=2)
arm2, = ax.plot([], [], 'bo-', lw=2)

def forward_kinematics(theta1, theta2):
    x1, y1 = L1 * np.cos(theta1), L1 * np.sin(theta1)
    x2, y2 = x1 + L2 * np.cos(theta1 + theta2), y1 + L2 * np.sin(theta1 + theta2)
    return np.array([[0, x1, x2], [0, y1, y2]])

frames = np.linspace(0, 1, 30)  # Smooth interpolation
theta1_vals = np.linspace(0, angles[0], len(frames))
theta2_vals = np.linspace(0, angles[1], len(frames))

def update(frame):
    theta1, theta2 = theta1_vals[frame], theta2_vals[frame]
    positions = forward_kinematics(theta1, theta2)
    arm1.set_data(positions[0, :2], positions[1, :2])
    arm2.set_data(positions[0, 1:], positions[1, 1:])
    return arm1, arm2

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
plt.legend()
plt.show()