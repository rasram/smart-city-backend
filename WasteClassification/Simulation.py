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
# Define fixed checkpoints for the robot base X-coordinate (Reduced to 3)
# Checkpoint 1: Start/Pickup area
# Checkpoint 2: Middle area
# Checkpoint 3: End area
robot_base_checkpoints = [-0.8, 1.4, 3.2] 
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


def move_robot_base_to(target_x, base_speed=1.0, tolerance=0.01, max_wait_steps=240 * 10):
    """Moves the robot base along the X axis to a specific checkpoint using velocity."""
    print(f"Attempting to move robot base to checkpoint x = {target_x:.2f}")
    steps = 0
    while steps < max_wait_steps:
        current_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        current_x = current_pos[0]
        
        if abs(target_x - current_x) < tolerance:
            # Already at the target, ensure velocity is zero and break
            p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            print(f"Robot base reached checkpoint x = {target_x:.2f}")
            return True

        direction = 1.0 if target_x > current_x else -1.0
        move_velocity = direction * base_speed

        # Apply velocity
        p.resetBaseVelocity(robot_id, linearVelocity=[move_velocity, 0, 0], angularVelocity=[0, 0, 0])

        # Step simulation
        p.stepSimulation()
        time.sleep(1./240.) # Maintain simulation rate consistency
        
        # Check if we passed the target in this step
        next_pos, _ = p.getBasePositionAndOrientation(robot_id)
        next_x = next_pos[0]
        
        # If moving right and passed target OR moving left and passed target
        if (direction > 0 and next_x >= target_x) or (direction < 0 and next_x <= target_x):
            # Stop and place precisely at the target checkpoint
            p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            p.resetBasePositionAndOrientation(robot_id, [target_x, current_pos[1], current_pos[2]], base_orn) # Final precise placement
            print(f"Robot base arrived and stopped at checkpoint x = {target_x:.2f}")
            for _ in range(10):
                 p.stepSimulation()
                 time.sleep(1./240.)
            return True
            
        steps += 1

    # Timeout
    print(f"Warning: Robot base movement to {target_x:.2f} timed out after {max_wait_steps} steps.")
    p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]) # Stop on timeout
    return False


def get_closest_checkpoint(target_x):
    """Finds the checkpoint X-coordinate closest to the target_x."""
    closest_checkpoint = min(robot_base_checkpoints, key=lambda cp: abs(cp - target_x))
    return closest_checkpoint


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
    target_bin_x = target_bin_pos[0] # Get bin's X coordinate

    print(f"Processing object for {bin_type} bin at x={target_bin_x:.2f}...")

    # --- Pickup Sequence ---
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    pickup_hover_z = obj_pos[2] + 0.20
    pickup_grasp_z = obj_pos[2] + 0.03

    # Ensure base is at the starting checkpoint for pickup
    # Use get_closest_checkpoint for pickup as before
    pickup_checkpoint_x = get_closest_checkpoint(pickup_zone_x)
    print(f"Moving robot base to pickup checkpoint {pickup_checkpoint_x:.2f}")
    if not move_robot_base_to(pickup_checkpoint_x):
         print("Error: Failed to move base to pickup checkpoint. Skipping object.")
         # No object held yet, just reset state
         current_object = None
         object_state = "GENERATING"
         return

    print("Moving to pickup hover position...")
    move_scara_ik([pickup_zone_x, obj_pos[1], pickup_hover_z])
    print("Moving to grasp position...")
    move_scara_ik([pickup_zone_x, obj_pos[1], pickup_grasp_z])

    print("Pausing before grasp...")
    time.sleep(0.5) # Pause before grasping

    attach_object(obj_id)
    print("Moving up after grasp...")
    move_scara_ik([pickup_zone_x, obj_pos[1], pickup_hover_z])

    # --- Placement Sequence ---
    # Determine the placement checkpoint based on bin X-coordinate
    # Goal: Stop base further away, use arm reach more.
    if target_bin_x <= 1.1: # Bins at 0.5, 1.1
        placement_checkpoint_x = robot_base_checkpoints[0] # Use -0.8 (start checkpoint)
    elif target_bin_x <= 3.5: # Bins at 1.7, 2.3, 2.9, 3.5
        placement_checkpoint_x = robot_base_checkpoints[1] # Use 1.4 (middle checkpoint)
    else: # Fallback, should not be reached with current bin layout
        print(f"Warning: Bin X ({target_bin_x}) outside expected range, using middle checkpoint.")
        placement_checkpoint_x = robot_base_checkpoints[1]

    print(f"Selected placement checkpoint {placement_checkpoint_x:.2f} for bin {bin_type} at x={target_bin_x:.2f}")

    print(f"Moving robot base to placement checkpoint {placement_checkpoint_x:.2f}")
    if not move_robot_base_to(placement_checkpoint_x): # Check if movement succeeded
         print("Error: Failed to move base to placement checkpoint. Releasing object and skipping.")
         # Release the object before skipping
         release_object()
         # Optionally move arm up/home before skipping
         print("Moving arm up before skipping...")
         move_scara_ik([pickup_zone_x, obj_pos[1], pickup_hover_z]) # Move arm back up slightly relative to pickup pos
         print("Returning arm to home joints before skipping...")
         move_scara_joints(home_joint_angles) # Move arm home
         print("Returning base to home checkpoint before skipping...")
         move_robot_base_to(robot_base_checkpoints[0]) # Attempt to return base home

         current_object = None
         object_state = "GENERATING"
         return

    # Define placement heights relative to bin top
    bin_top_z = target_bin_pos[2] + bin_half_extents[2]
    place_hover_z = bin_top_z + 0.20
    place_down_z = bin_top_z + 0.05

    # Target world positions for IK (using the actual bin coordinates)
    target_place_hover_world = [target_bin_pos[0], target_bin_pos[1], place_hover_z]
    target_place_down_world = [target_bin_pos[0], target_bin_pos[1], place_down_z]

    print("Moving to placement hover position...")
    move_scara_ik(target_place_hover_world)

    print("Moving down to place...")
    move_scara_ik(target_place_down_world)

    print("Pausing before release...")
    time.sleep(0.5) # Pause before releasing

    release_object()

    print("Moving up after release...")
    move_scara_ik(target_place_hover_world)

    # --- Return to Home ---
    print("Returning arm to home joint angles...")
    move_scara_joints(home_joint_angles)

    print("Returning robot base to start checkpoint...")
    home_checkpoint_x = robot_base_checkpoints[0]
    if not move_robot_base_to(home_checkpoint_x): # Check if movement succeeded
        print("Error: Failed to return base to home checkpoint.")

    # Wait for object settling
    print("Waiting for object to settle...")
    for _ in range(120): # Simulate for ~0.5 seconds
        p.stepSimulation()
        # time.sleep(1./240.) # Optional sleep if needed for visual debugging

    # Check placement
    if check_object_placement(obj_id, bin_type):
        objects_processed += 1
    else:
        objects_removed += 1
        print(f"Removing missed object {obj_id}")
        # Check if body exists before removing, as physics might have already removed it
        try:
            p.getBodyInfo(obj_id) # Check if ID is valid
            p.removeBody(obj_id)
        except p.error as e:
            # If getBodyInfo fails, the body likely doesn't exist anymore
            print(f"Could not remove body {obj_id}, likely already removed or invalid: {e}")


    current_object = None
    object_state = "GENERATING"
    print(f"Processing complete. Total processed: {objects_processed}, Total removed: {objects_removed}")
    print("-" * 20)

# --- Main Simulation Loop ---
last_time = time.time()

print("Moving robot to initial home position...")
move_scara_joints(home_joint_angles, speed_fraction=1.0)
move_robot_base_to(robot_base_checkpoints[0])
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
