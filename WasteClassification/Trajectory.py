import numpy as np

# SCARA arm link lengths
L1, L2 = 0.5, 0.4  # Example link lengths

def inverse_kinematics(x, y):
    """Compute joint angles (theta1, theta2) given end-effector position (x, y)"""
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))  # Clip to avoid numerical errors
    
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return theta1, theta2

def forward_kinematics(theta1, theta2):
    """Compute (x, y) given joint angles (theta1, theta2)"""
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

# Define waypoints in Cartesian space
waypoints = np.linspace([0, 0], [1, 0.6], num=10)

# Compute joint space trajectory using inverse kinematics
joint_angles = np.array([inverse_kinematics(x, y) for x, y in waypoints])

# Extract theta1 and theta2
theta1_vals, theta2_vals = joint_angles[:, 0], joint_angles[:, 1]

# Forward kinematics to verify correctness
fk_positions = np.array([forward_kinematics(t1, t2) for t1, t2 in zip(theta1_vals, theta2_vals)])

# Compute the Euclidean error between waypoints and forward kinematics results
errors = np.linalg.norm(waypoints - fk_positions, axis=1)
max_error = np.max(errors)
avg_error = np.mean(errors)

max_error, avg_error
