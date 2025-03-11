import pybullet as p
import pybullet_data
import time
import random

# Connect to PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity
p.setGravity(0, 0, -9.8)

# Load a plane for visualization
plane_id = p.loadURDF("plane.urdf")

# Load the SCARA robot
robot_id = p.loadURDF("scara.urdf", [0, 0, 0], useFixedBase=True)

# Define bins for waste sorting - moved closer for better reach
bins = {
    "plastic": [0.4, -0.4, 0.01],
    "metal": [0.4, 0.4, 0.01],
    "paper": [-0.4, -0.4, 0.01],
    "glass": [-0.4, 0.4, 0.01],
}

# Create bins as containers
def create_bin(position, color):
    # Create the base of the bin
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.2, 0.2, 0.1],
        rgbaColor=color,
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.2, 0.2, 0.1],
    )
    bin_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )

    # Define wall dimensions and positions relative to the bin
    wall_thickness = 0.02
    wall_height = 0.2
    wall_positions = [
        [position[0] + 0.2 + wall_thickness / 2, position[1], position[2] + wall_height / 2],  # Right wall
        [position[0] - 0.2 - wall_thickness / 2, position[1], position[2] + wall_height / 2],  # Left wall
        [position[0], position[1] + 0.2 + wall_thickness / 2, position[2] + wall_height / 2],  # Front wall
        [position[0], position[1] - 0.2 - wall_thickness / 2, position[2] + wall_height / 2],  # Back wall
    ]
    wall_half_extents = [
        [wall_thickness / 2, 0.2, wall_height / 2],  # Right and Left walls
        [0.2, wall_thickness / 2, wall_height / 2],  # Front and Back walls
    ]

    # Create walls
    for i, wall_pos in enumerate(wall_positions):
        wall_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=wall_half_extents[i // 2],
            rgbaColor=color,
        )
        wall_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=wall_half_extents[i // 2],
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_shape,
            baseVisualShapeIndex=wall_visual_shape,
            basePosition=wall_pos,
        )

    return bin_id

# Add bins
bin_colors = {
    "plastic": [1, 0, 0, 1],  # Red
    "metal": [0, 1, 0, 1],    # Green
    "paper": [0, 0, 1, 1],    # Blue
    "glass": [1, 1, 0, 1],    # Yellow
}
for bin_type, position in bins.items():
    create_bin(position, bin_colors[bin_type])

# Home position for SCARA robot
home_position = [0, 0, 0.3]

# Attach object to end effector with proper offset calculation
def attach_object(obj_id):
    # Get current position of the end effector
    end_effector_state = p.getLinkState(robot_id, 2)
    end_effector_pos = end_effector_state[0]
    end_effector_orn = end_effector_state[1]
    
    # Get current position of the object
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    
    # Calculate offset
    offset_x = obj_pos[0] - end_effector_pos[0]
    offset_y = obj_pos[1] - end_effector_pos[1]
    offset_z = obj_pos[2] - end_effector_pos[2]
    
    return p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=2,  # End effector link
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[offset_x, offset_y, offset_z],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=end_effector_orn,
        childFrameOrientation=obj_orn
    )

# Release object
def release_object(constraint_id):
    p.removeConstraint(constraint_id)

# Move SCARA using inverse kinematics
def move_scara(target_position):
    end_effector_index = 2
    joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=end_effector_index,
        targetPosition=target_position,
    )
    for joint_index, angle in enumerate(joint_angles[:3]):  # Limit to 3 joints
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
        )
    for _ in range(100):  # Simulate the movement
        p.stepSimulation()
        time.sleep(1 / 240)
    
    # Return the actual end effector position after movement
    return p.getLinkState(robot_id, end_effector_index)[0]

# Reset object position
def reset_object_position(object_id, new_position, new_orientation=[0, 0, 0, 1]):
    p.resetBasePositionAndOrientation(object_id, new_position, new_orientation)

# Move conveyor
def move_conveyor():
    global current_object
    if current_object is not None:
        pos, _ = p.getBasePositionAndOrientation(current_object["id"])
        new_pos = [pos[0] + 0.005, pos[1], pos[2]]
        reset_object_position(current_object["id"], new_pos)

# Generate a new object
def generate_new_object():
    new_bin = random.choice(list(bins.keys()))
    return {
        "id": p.loadURDF("cube_small.urdf", [-0.5, 0.2, 0.1]),
        "bin": new_bin,
    }

# Check if object is in the correct bin
def check_object_placement(obj_id, bin_type):
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    bin_pos = bins[bin_type]
    
    # Define bin boundaries
    bin_min_x = bin_pos[0] - 0.19  # Slightly smaller than bin to ensure it's inside
    bin_max_x = bin_pos[0] + 0.19
    bin_min_y = bin_pos[1] - 0.19
    bin_max_y = bin_pos[1] + 0.19
    
    # Check if object is within bin boundaries
    if (bin_min_x <= obj_pos[0] <= bin_max_x and 
        bin_min_y <= obj_pos[1] <= bin_max_y):
        print(f"Object correctly placed in {bin_type} bin.")
        return True
    
    # Object is not correctly placed, remove it
    print(f"Object not placed in {bin_type} bin! Removing object.")
    print(f"Object position: {obj_pos}, Bin position: {bin_pos}")
    p.removeBody(obj_id)
    return False

# Main simulation
current_object = generate_new_object()
objects_processed = 0
objects_removed = 0

def process_object():
    global current_object, objects_processed, objects_removed
    pos, _ = p.getBasePositionAndOrientation(current_object["id"])
    if 0.3 <= pos[0] <= 0.4:
        # Store object info before processing
        obj_id = current_object["id"]
        bin_type = current_object["bin"]
        
        # Move to pick-up position
        move_scara([pos[0], pos[1], pos[2] + 0.1])
        
        # Attach the object to the end effector
        constraint = attach_object(obj_id)
        
        # Get bin position and add random offset for placement
        bin_pos = bins[bin_type]
        # Add random variation to avoid stacking (within 0.15m of bin center)
        x_offset = random.uniform(-0.1, 0.1)
        y_offset = random.uniform(-0.1, 0.1)
        
        # Move to position above bin with offset
        move_scara([bin_pos[0] + x_offset, bin_pos[1] + y_offset, bin_pos[2] + 0.3])
        
        # Lower into bin with the same offset
        move_scara([bin_pos[0] + x_offset, bin_pos[1] + y_offset, bin_pos[2] + 0.15])
        
        # Release the object
        release_object(constraint)
        
        # Allow object to settle
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1/240)
        
        # Check if object is correctly placed and remove if not
        if check_object_placement(obj_id, bin_type):
            objects_processed += 1
        else:
            objects_removed += 1
        
        # Return to home position
        move_scara(home_position)
        
        # Mark as processed
        current_object = None

try:
    for step in range(10000):
        if current_object is not None:
            process_object()
        
        # Generate new object only after previous one is sorted
        if current_object is None:
            current_object = generate_new_object()
            print(f"\nProcessing new object for {current_object['bin']} bin")
        
        move_conveyor()
        p.stepSimulation()
        time.sleep(1 / 240)

finally:
    print("\nSimulation ended.")
    print(f"Objects successfully placed: {objects_processed}")
    print(f"Objects removed (incorrectly placed): {objects_removed}")
    p.disconnect()