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
