# Simulation.py
import pybullet as p
import pybullet_data
import time
import random
import os  # Added for path joining

# Connect to PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity
p.setGravity(0, 0, -9.8)

# Load a plane for visualization
plane_id = p.loadURDF("plane.urdf")

# --- Robot Setup ---
# Define robot start position and orientation
robot_start_pos = [-0.8, 0, 0.01]  # Positioned left of the first bins
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
# Define fixed checkpoints for the robot base (y=0, x is average of bin x-coordinates)
# Checkpoint 0: Start/Pickup area (x=-0.8, y=0)
# Checkpoint 1: Average of bins 1-4 (x=0.8, y=0) [plastic, metal, paper, glass]
# Checkpoint 2: Average of bins 5-8 (x=2.0, y=0) [organic, electronics, textile, ewaste]
# Checkpoint 3: Average of bins 9-12 (x=3.2, y=0) [battery, rubber, medicine, mixed]
robot_base_checkpoints = [-0.8, 0.8, 2.0, 3.2]  # x-coordinates only (y=0 is handled in movement)
# Ensure the URDF file path is correct
script_dir = os.path.dirname(__file__)
urdf_path = os.path.join(script_dir, "scara.urdf")

try:
    # Load the SCARA robot with a fixed base initially (will be moved programmatically)
    robot_id = p.loadURDF(urdf_path, robot_start_pos, robot_start_orientation, useFixedBase=False)  # Keep useFixedBase=False to allow sliding
    print(f"Successfully loaded robot from: {urdf_path}")
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints.")
    # Find the end effector link index dynamically
    end_effector_link_index = -1
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode('UTF-8')
        if link_name == 'end_effector':  # Match the link name in URDF
            end_effector_link_index = i
            break

    if end_effector_link_index == -1:
        print("Warning: Could not find link named 'end_effector'. Using default index 2.")
        end_effector_link_index = 2  # Default based on typical structure
    else:
        print(f"Found end effector link index: {end_effector_link_index}")

except Exception as e:
    print(f"Error loading robot URDF: {e}")
    print(f"Attempted to load from: {urdf_path}")
    p.disconnect()
    exit()

# --- Bin Setup ---
# Define 12 bins for waste sorting (Adjusted positions slightly)
bins = {
    "plastic": [0.5, -1.0, 0.01],
    "metal": [0.5, 1.0, 0.01],
    "paper": [1.1, -1.0, 0.01],
    "glass": [1.1, 1.0, 0.01],
    "organic": [1.7, -1.0, 0.01],
    "electronics": [1.7, 1.0, 0.01],
    "textile": [2.3, -1.0, 0.01],
    "ewaste": [2.3, 1.0, 0.01],
    "battery": [2.9, -1.0, 0.01],
    "rubber": [2.9, 1.0, 0.01],
    "medicine": [3.5, -1.0, 0.01],
    "mixed": [3.5, 1.0, 0.01],
}

# Define colors for each bin
bin_colors = {
    "plastic": [1, 0, 0, 1],
    "metal": [0.7, 0.7, 0.7, 1],
    "paper": [0, 0, 1, 1],
    "glass": [0, 1, 1, 1],
    "organic": [0.5, 0.25, 0, 1],
    "electronics": [1, 1, 0, 1],
    "textile": [1, 0, 1, 1],
    "ewaste": [0.3, 0.3, 0.3, 1],
    "battery": [0.6, 0.4, 0.2, 1],
    "rubber": [0.1, 0.1, 0.1, 1],
    "medicine": [1, 0.8, 0.6, 1],
    "mixed": [0.5, 0.5, 1, 1],
}

# Create bins
bin_half_extents = [0.25, 0.25, 0.1]  # Slightly larger bins


def create_bin(position, color):
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX, halfExtents=bin_half_extents, rgbaColor=color)
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX, halfExtents=bin_half_extents)
    bin_id = p.createMultiBody(
        baseMass=0,  # Static bins
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )
    return bin_id


bin_ids = {}
for bin_type, position in bins.items():
    bin_ids[bin_type] = create_bin(position, bin_colors[bin_type])

# --- Conveyor Belt Setup ---
conveyor_start_pos = [-2.0, 0.5, 0.05]
conveyor_length = 2.5
conveyor_width = 0.4
conveyor_height = 0.05
conveyor_half_extents = [conveyor_length / 2.0, conveyor_width / 2.0, conveyor_height / 2.0]
conveyor_pos = [conveyor_start_pos[0] + conveyor_length / 2.0, conveyor_start_pos[1], conveyor_start_pos[2]]
conveyor_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=conveyor_half_extents, rgbaColor=[0.4, 0.4, 0.4, 1])
conveyor_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=conveyor_half_extents)
conveyor_id = p.createMultiBody(baseMass=0,  # Static
                                baseCollisionShapeIndex=conveyor_collision_shape_id,
                                baseVisualShapeIndex=conveyor_visual_shape_id,
                                basePosition=conveyor_pos,
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# --- Object and Simulation Parameters ---
object_urdf_path = os.path.join(script_dir, "cube_small.urdf")  # Make sure cube_small.urdf exists
conveyor_speed = 0.5  # meters per second
pickup_zone_x = -0.5  # X-coordinate where the robot should pick up items
object_state = "GENERATING"  # Possible states: GENERATING, MOVING, WAITING_PICKUP, PROCESSING, DONE
current_object = None
objects_processed = 0
objects_removed = 0
active_constraint = None  # To hold the grasp constraint ID

# Home position for SCARA (joint angles) - Adjust if needed
home_joint_angles = [0, 0, 0]  # Joint1, Joint2, Joint3(prismatic) - Ensure this is a safe retracted position
num_scara_joints = 3  # Typically 3 main joints for positioning

# Limits from URDF for joint3 (prismatic)
joint3_lower_limit = -0.2
joint3_upper_limit = 0.2

# --- Robot Control Functions ---


def wait_for_movement(target_angles, joint_indices, tolerance=0.01, max_wait_steps=240 * 5):
    """Steps simulation until robot joints reach target angles or timeout."""
    steps = 0
    while steps < max_wait_steps:
        current_joint_states = p.getJointStates(robot_id, joint_indices)
        current_joint_positions = [state[0] for state in current_joint_states]

        diff = [abs(a - b) for a, b in zip(target_angles, current_joint_positions)]

        if all(d < tolerance for d in diff):
            return True

        p.stepSimulation()
        steps += 1

    print(f"Warning: Movement timed out after {max_wait_steps} steps.")
    return False


def move_scara_joints(target_angles, speed_fraction=0.5):
    """Moves the robot to target joint angles."""
    joint_indices = list(range(min(len(target_angles), num_scara_joints)))
    if len(target_angles) > num_scara_joints:
        print(f"Warning: target_angles ({len(target_angles)}) longer than controlled joints ({num_scara_joints}). Using first {num_scara_joints}.")
        target_angles = target_angles[:num_scara_joints]

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_angles,
    )
    wait_for_movement(target_angles, joint_indices)


def move_scara_ik(target_world_position, speed_fraction=0.5):
    """Moves the robot's end effector to a target world position using IK."""
    joint_angles_raw = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=end_effector_link_index,
        targetPosition=target_world_position,
    )

    target_joint_angles = list(joint_angles_raw[:num_scara_joints])
    joint_indices = list(range(num_scara_joints))

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_joint_angles,
    )
    wait_for_movement(target_joint_angles, joint_indices)


def move_scara_joint_linear(target_angles, linear_joint_index, speed=0.1):
    """Moves a single specified joint linearly to its target angle while keeping others fixed."""
    control_joint_indices = list(range(num_scara_joints))  # Indices [0, 1, 2]

    if linear_joint_index >= num_scara_joints:
        print(f"Error: linear_joint_index {linear_joint_index} out of range.")
        return
    if len(target_angles) != num_scara_joints:
        print(f"Error: target_angles length mismatch.")
        return

    # Get current joint angles
    current_joint_states = p.getJointStates(robot_id, control_joint_indices)
    current_joint_angles = [state[0] for state in current_joint_states]

    # Calculate difference for the linear joint
    delta_angle = target_angles[linear_joint_index] - current_joint_angles[linear_joint_index]
    if abs(delta_angle) < 1e-6:  # Already there
        return

    # Calculate steps based on speed and simulation time step
    time_step = 1. / 240.
    duration = abs(delta_angle) / speed
    steps = max(1, int(duration / time_step))

    print(f"Linear joint move: Joint={linear_joint_index}, Delta={delta_angle:.3f}, Steps={steps}, Speed={speed:.2f}")

    # Keep other joints fixed at their target angles (from the input target_angles)
    intermediate_target = list(current_joint_angles)  # Start from current
    # Set non-linear joints to their final target immediately
    for i in range(num_scara_joints):
        if i != linear_joint_index:
            intermediate_target[i] = target_angles[i]

    for i in range(1, steps + 1):
        fraction = i / steps
        # Interpolate only the linear joint
        intermediate_target[linear_joint_index] = current_joint_angles[linear_joint_index] + delta_angle * fraction

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=control_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=intermediate_target,
        )
        p.stepSimulation()
        time.sleep(time_step)

    # Final command to ensure target is reached
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=control_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_angles,
    )
    wait_for_movement(target_angles, control_joint_indices, tolerance=0.015)  # Wait for final settling


def attach_object(obj_id):
    global active_constraint
    if active_constraint is not None:
        release_object()

    active_constraint = p.createConstraint(
        parentBodyUniqueId=robot_id, parentLinkIndex=end_effector_link_index,
        childBodyUniqueId=obj_id, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0.01],
    )
    print("Object attached.")


def release_object():
    global active_constraint
    if active_constraint is not None:
        p.removeConstraint(active_constraint)
        active_constraint = None
        print("Object released.")


def move_robot_base_to_checkpoint(checkpoint_x, base_speed=0.5, tolerance=0.01):
    """Moves the robot base to a checkpoint at constant velocity."""
    print(f"Moving robot base to checkpoint x = {checkpoint_x:.2f}")
    while True:
        current_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        current_x = current_pos[0]
        dx = checkpoint_x - current_x
        if abs(dx) < tolerance:
            # Snap to checkpoint and stop
            p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            p.resetBasePositionAndOrientation(robot_id, [checkpoint_x, current_pos[1], current_pos[2]], base_orn)
            for _ in range(10):
                p.stepSimulation()
                time.sleep(1./240.)
            print(f"Robot base arrived at checkpoint x = {checkpoint_x:.2f}")
            break
        direction = 1.0 if dx > 0 else -1.0
        move_velocity = direction * base_speed
        p.resetBaseVelocity(robot_id, linearVelocity=[move_velocity, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(1./240.)


def move_scara_ik_linear(target_world_position, speed=0.2, step_size=0.01):
    """Moves the robot's end effector to a target world position in a straight line at constant speed."""
    # Get current end effector position
    link_state = p.getLinkState(robot_id, end_effector_link_index)
    current_pos = list(link_state[0])
    target_pos = list(target_world_position)
    distance = ((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2 + (target_pos[2] - current_pos[2])**2) ** 0.5
    steps = max(1, int(distance / step_size))
    for i in range(1, steps + 1):
        interp = [current_pos[j] + (target_pos[j] - current_pos[j]) * i / steps for j in range(3)]
        joint_angles_raw = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=end_effector_link_index,
            targetPosition=interp,
        )
        target_joint_angles = list(joint_angles_raw[:num_scara_joints])
        joint_indices = list(range(num_scara_joints))
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_angles,
        )
        p.stepSimulation()
        time.sleep(step_size / speed)


def get_checkpoint_for_bin(bin_type):
    """Returns the checkpoint x for a given bin, with one checkpoint positioned equidistant from the boxes it serves."""
    # Group bins into sets of four, each mapped to a checkpoint index:
    # Checkpoint 1 (index 1): plastic, metal, paper, glass (avg x = 0.8)
    # Checkpoint 2 (index 2): organic, electronics, textile, ewaste (avg x = 2.0)
    # Checkpoint 3 (index 3): battery, rubber, medicine, mixed (avg x = 3.2)
    bin_to_checkpoint = {
        "plastic": 1, "metal": 1, "paper": 1, "glass": 1,
        "organic": 2, "electronics": 2, "textile": 2, "ewaste": 2,
        "battery": 3, "rubber": 3, "medicine": 3, "mixed": 3,
    }
    idx = bin_to_checkpoint.get(bin_type, 1)  # Default to first bin checkpoint if unknown
    return robot_base_checkpoints[idx]


def check_object_placement(obj_id, bin_type):
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    bin_pos = bins[bin_type]
    if (bin_pos[0] - bin_half_extents[0] <= obj_pos[0] <= bin_pos[0] + bin_half_extents[0] and
            bin_pos[1] - bin_half_extents[1] <= obj_pos[1] <= bin_pos[1] + bin_half_extents[1] and
            obj_pos[2] < bin_pos[2] + bin_half_extents[2] * 2):
        print(f"Object successfully placed in {bin_type} bin.")
        return True
    print(f"Object missed {bin_type} bin! Position: {obj_pos}, Bin Center: {bin_pos}")
    return False

# --- Conveyor and Object Logic ---


def reset_object_position(object_id, new_position, new_orientation=[0, 0, 0, 1]):
    p.resetBasePositionAndOrientation(object_id, new_position, new_orientation)


def move_object_on_conveyor(dt):
    global current_object, object_state
    if current_object and object_state == "MOVING":
        obj_id = current_object["id"]
        pos, orn = p.getBasePositionAndOrientation(obj_id)

        # Check if the object has reached or passed the pickup zone
        if pos[0] >= pickup_zone_x:
            object_state = "WAITING_PICKUP"
            print("Object reached pickup zone.")
            # Stop the object precisely at the pickup zone
            p.resetBasePositionAndOrientation(obj_id, [pickup_zone_x, pos[1], pos[2]], orn)
            p.resetBaseVelocity(obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        else:
            # Apply conveyor velocity
            # Ensure the object stays roughly aligned with the conveyor's Y position
            target_y = conveyor_pos[1]
            # Small correction force/velocity towards center if needed (optional)
            y_diff = target_y - pos[1]
            correction_vy = y_diff * 0.5  # Gentle pull towards center

            p.resetBaseVelocity(obj_id, linearVelocity=[conveyor_speed, correction_vy, 0], angularVelocity=[0, 0, 0])


def generate_new_object():
    global current_object, object_state
    if object_urdf_path and os.path.exists(object_urdf_path):
        bin_name = random.choice(list(bins.keys()))
        # Start slightly higher to ensure it's above the conveyor
        start_pos = [conveyor_start_pos[0] - 0.1, conveyor_start_pos[1], conveyor_pos[2] + conveyor_half_extents[2] + 0.05]
        try:
            obj_id = p.loadURDF(object_urdf_path, start_pos)
            # Set friction properties (optional, might help sliding)
            p.changeDynamics(obj_id, -1, lateralFriction=0.1)
            current_object = {"id": obj_id, "bin": bin_name}
            object_state = "MOVING"
            print(f"\nGenerated new object ({bin_name}) ID: {obj_id}")
            # Give it an initial velocity immediately
            p.resetBaseVelocity(obj_id, linearVelocity=[conveyor_speed, 0, 0], angularVelocity=[0, 0, 0])
            return True
        except Exception as e:
            print(f"Error loading object URDF '{object_urdf_path}': {e}")
            object_state = "DONE"  # Stop simulation if object loading fails
            return False
    else:
        print(f"Error: Object URDF not found at {object_urdf_path}")
        object_state = "DONE"
        return False

# --- Main Processing Function ---


def process_object():
    global current_object, object_state, objects_processed, objects_removed
    if not current_object or object_state != "WAITING_PICKUP":
        return

    object_state = "PROCESSING"
    obj_id = current_object["id"]
    bin_type = current_object["bin"]
    target_bin_pos = bins[bin_type]
    target_bin_x = target_bin_pos[0]

    print(f"Processing object for {bin_type} bin at x={target_bin_x:.2f}...")

    # --- Pickup Sequence ---
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    pickup_hover_z = obj_pos[2] + 0.20
    pickup_grasp_z = obj_pos[2] + 0.03

    # Always move base to checkpoint 0 for pickup
    pickup_checkpoint_x = robot_base_checkpoints[0]
    print(f"Moving robot base to pickup checkpoint {pickup_checkpoint_x:.2f}")
    move_robot_base_to_checkpoint(pickup_checkpoint_x)

    print("Moving to pickup hover position...")
    move_scara_ik_linear([pickup_zone_x, obj_pos[1], pickup_hover_z])
    print("Moving to grasp position...")
    move_scara_ik_linear([pickup_zone_x, obj_pos[1], pickup_grasp_z])

    print("Pausing before grasp...")
    time.sleep(0.5)

    attach_object(obj_id)
    print("Moving up after grasp...")
    move_scara_ik_linear([pickup_zone_x, obj_pos[1], pickup_hover_z])

    # --- Placement Sequence ---
    placement_checkpoint_x = get_checkpoint_for_bin(bin_type)
    print(f"Moving robot base to placement checkpoint {placement_checkpoint_x:.2f}")
    move_robot_base_to_checkpoint(placement_checkpoint_x)

    bin_top_z = target_bin_pos[2] + bin_half_extents[2]
    place_hover_z = bin_top_z + 0.20
    place_down_z = bin_top_z + 0.01  # Place closer to bin bottom (was 0.05)

    # Define placement position and orientation - ensure directly above bin center
    target_place_hover_world = [target_bin_pos[0], target_bin_pos[1], place_hover_z]
    
    print(f"Bin center position: {target_bin_pos}")
    print(f"Target hover position: {target_place_hover_world}")

    # Use a slower speed (0.2) for more precise positioning
    print("Moving to placement hover position (linear)...")
    move_scara_ik_linear(target_place_hover_world, speed=0.2)

    # Double pause at hover to ensure stability
    print("Pausing at hover position to stabilize...")
    time.sleep(1.0)  # Longer pause to ensure stability

    # --- Vertical Placement using Linear Joint Control ---
    print("Moving down to place (vertical only)...")
    # Get the ACTUAL joint angles after the hover move and pause
    current_joint_states_actual_hover = p.getJointStates(robot_id, list(range(num_scara_joints)))
    j1_actual_hover, j2_actual_hover, j3_actual_hover = [state[0] for state in current_joint_states_actual_hover]

    # Get the world position corresponding to the actual hover joint state
    link_state_actual_hover = p.getLinkState(robot_id, end_effector_link_index)
    current_world_pos_actual_hover = link_state_actual_hover[0]
    current_world_z_actual_hover = current_world_pos_actual_hover[2]
    
    print(f"Actual hover position: {current_world_pos_actual_hover}")

    # Calculate delta Z needed to reach the 'down' position from the actual hover Z
    delta_z_down = place_down_z - current_world_z_actual_hover

    # Calculate target prismatic joint value based on actual hover angle
    j3_target_down = j3_actual_hover + delta_z_down
    j3_target_down = max(joint3_lower_limit, min(joint3_upper_limit, j3_target_down))

    # Target angles keeping J1 and J2 fixed at their ACTUAL hover values
    target_angles_down = [j1_actual_hover, j2_actual_hover, j3_target_down]
    # Use the linear joint move function for the prismatic joint (index 2) with slower speed
    move_scara_joint_linear(target_angles_down, linear_joint_index=2, speed=0.05)  # Much slower for precise placement
    
    print("Pausing before release...")
    time.sleep(0.5)

    release_object()

    # --- Vertical Retract using Linear Joint Control ---
    print("Moving up after release (vertical only)...")
    # Calculate delta Z needed to go back up to the desired hover height
    delta_z_up = place_hover_z - place_down_z # Desired total vertical distance

    # Get current prismatic joint position after downward move and release
    current_joint_states_after_down = p.getJointStates(robot_id, list(range(num_scara_joints)))
    j3_current = current_joint_states_after_down[2][0]

    # Calculate target prismatic joint value relative to current position
    j3_target_up = j3_current + delta_z_up
    j3_target_up = max(joint3_lower_limit, min(joint3_upper_limit, j3_target_up))

    # Target angles keeping J1 and J2 fixed at their ACTUAL hover values
    target_angles_up = [j1_actual_hover, j2_actual_hover, j3_target_up]
    # Use the linear joint move function for the prismatic joint (index 2)
    move_scara_joint_linear(target_angles_up, linear_joint_index=2, speed=0.1) # Adjust speed as needed

    # --- Return to Home ---
    print("Returning arm to home joint angles...")
    move_scara_joints(home_joint_angles)

    print("Returning robot base to start checkpoint...")
    home_checkpoint_x = robot_base_checkpoints[0]
    move_robot_base_to_checkpoint(home_checkpoint_x)

    print("Waiting for object to settle...")
    for _ in range(120):
        p.stepSimulation()

    if check_object_placement(obj_id, bin_type):
        objects_processed += 1
    else:
        objects_removed += 1
        print(f"Removing missed object {obj_id}")
        try:
            p.getBodyInfo(obj_id)
            p.removeBody(obj_id)
        except p.error as e:
            print(f"Could not remove body {obj_id}, likely already removed or invalid: {e}")

    current_object = None
    object_state = "GENERATING"
    print(f"Processing complete. Total processed: {objects_processed}, Total removed: {objects_removed}")
    print("-" * 20)

# --- Main Simulation Loop ---
last_time = time.time()

print("Moving robot to initial home position...")
move_scara_joints(home_joint_angles, speed_fraction=1.0)
move_robot_base_to_checkpoint(robot_base_checkpoints[0])
time.sleep(0.5)

p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[1.5, 0, 0])

try:
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if object_state == "GENERATING":
            if not generate_new_object():
                break

        elif object_state == "MOVING":
            move_object_on_conveyor(dt)

        elif object_state == "WAITING_PICKUP":
            process_object()

        elif object_state == "PROCESSING":
            pass

        elif object_state == "DONE":
            print("Simulation marked as DONE.")
            break

        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred during simulation: {e}")
finally:
    print("\n--- Simulation Summary ---")
    print(f"Objects successfully placed: {objects_processed}")
    print(f"Objects missed/removed: {objects_removed}")
    print("Disconnecting from PyBullet.")
    p.disconnect()
