import sim
import time
import numpy as np
from Trajectory import analytical_ik, genetic_algorithm, optimize_angles

# Connect to CoppeliaSim
sim.simxFinish(-1)  # Close previous connections
client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if client_id != -1:
    print("✅ Connected to CoppeliaSim")

    # Start simulation
    sim.simxStartSimulation(client_id, sim.simx_opmode_blocking)
    time.sleep(0.5)

    # Get joint handles (Make sure these names match the scene!)
    _, joint1 = sim.simxGetObjectHandle(client_id, 'shoulder_joint', sim.simx_opmode_blocking)
    _, joint2 = sim.simxGetObjectHandle(client_id, 'slider_joint', sim.simx_opmode_blocking)

    # Get target input from user
    x_target = float(input("Enter target x: "))
    y_target = float(input("Enter target y: "))
    target = np.array([x_target, y_target])

    # Solve inverse kinematics
    initial_solution = genetic_algorithm(target)
    final_angles = optimize_angles(initial_solution, target)

    print("🎯 Final joint angles (rad):", final_angles)

    # OPTIONAL: Log current joint positions
    _, pos1 = sim.simxGetJointPosition(client_id, joint1, sim.simx_opmode_blocking)
    _, pos2 = sim.simxGetJointPosition(client_id, joint2, sim.simx_opmode_blocking)
    print("Initial joint positions:", pos1, pos2)

    # Move robot using simxSetJointPosition for passive joints
    sim.simxSetJointPosition(client_id, joint1, final_angles[0], sim.simx_opmode_oneshot)
    sim.simxSetJointPosition(client_id, joint2, final_angles[1], sim.simx_opmode_oneshot)

    # Wait for motion to complete
    time.sleep(5)

    # Stop simulation
    sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
    sim.simxFinish(client_id)

else:
    print("❌ Failed to connect to CoppeliaSim.")
