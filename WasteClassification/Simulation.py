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

# Define bins for waste sorting
bins = {
    "plastic": [0.5, -0.5, 0.01],
    "metal": [1.0, -0.5, 0.01],
    "paper": [1.5, -0.5, 0.01],
    "glass": [1.75, -0.5, 0.01],
}

# Create bins as containers
def create_bin(position, color):
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

# Attach object to end effector
def attach_object(obj_id):
    return p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=2,  # End effector link
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
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

# Main simulation
current_object = generate_new_object()

def process_object():
    global current_object
    pos, _ = p.getBasePositionAndOrientation(current_object["id"])
    if 0.3 <= pos[0] <= 0.4:
        move_scara([pos[0], pos[1], pos[2] + 0.1])  # Pick-up position
        constraint = attach_object(current_object["id"])
        bin_pos = bins[current_object["bin"]]
        move_scara([bin_pos[0], bin_pos[1], bin_pos[2] + 0.1])  # Move to bin
        release_object(constraint)
        move_scara(home_position)  # Return home
        current_object = None  # Mark as processed

try:
    for step in range(10000):
        if current_object is not None:
            process_object()
        
        # Generate new object only after previous one is sorted
        if current_object is None:
            current_object = generate_new_object()
        
        move_conveyor()
        p.stepSimulation()
        time.sleep(1 / 240)

finally:
    p.disconnect()