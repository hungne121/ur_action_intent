import time
import math
import pybullet
import pybullet_data
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
physics_client = pybullet.connect(pybullet.GUI) 

pybullet.resetSimulation() # Reset the simulation space
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) # Add paths to necessary data for pybullet
pybullet.setGravity(0.0, 0.0, -9.8) # Standard gravity along -Z axis
time_step = 1./240.
pybullet.setTimeStep(time_step)

# Load the floor
plane_start_pos = [0, 0, 0]
plane_start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])  # Flat floor on XY plane
plane_id = pybullet.loadURDF("plane.urdf", plane_start_pos, plane_start_orientation)

# Load the box — place it within UR5 workspace (~0.85m reach) on the floor
mass = 0.5  # kg
box_half = 0.05  # half-size of the cube (10cm cube)
position = [0.5, 0.0, box_half]  # in front of robot, resting on floor
orientation = [0, 0, 0, 1]       # no rotation
box_collision_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[box_half]*3)
box_visual_id    = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[box_half]*3, rgbaColor=[1, 0, 0, 1])
box_body_id      = pybullet.createMultiBody(mass, box_collision_id, box_visual_id, position, orientation)

import subprocess
from pathlib import Path

# --- Auto compile Xacro to URDF if needed ---
URDF_DIR = Path(__file__).resolve().parent.parent / "urdf"
XACRO_FILE = URDF_DIR / "ur3e_susgrip.urdf.xacro"
URDF_FILE = URDF_DIR / "ur3e_susgrip.urdf"

if XACRO_FILE.exists():
    if not URDF_FILE.exists() or XACRO_FILE.stat().st_mtime > URDF_FILE.stat().st_mtime:
        print(f"[XACRO] Compiling {XACRO_FILE.name} -> {URDF_FILE.name}...")
        cmd = f"source /opt/ros/humble/setup.bash && source /home/hungdao/ur_ws/install/setup.bash 2>/dev/null; xacro '{XACRO_FILE}' > '{URDF_FILE}'"
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
        print("[XACRO] Compilation complete!")

# Load the robot
arm_start_pos = [0, 0, 0]    # Robot base at floor level
arm_start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])  # Upright orientation
arm_id = pybullet.loadURDF(str(URDF_FILE), arm_start_pos, arm_start_orientation, useFixedBase=True)

# Set the camera position and other parameters in GUI mode
camera_distance = 2.0
camera_yaw = 45.0   # deg
camera_pitch = -30  # deg
camera_target_position = [0.0, 0.0, 1.5]
pybullet.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)


# --- Discover joint indices by name (robust against URDF structure) ---
num_joints_total = pybullet.getNumJoints(arm_id)
joint_name_to_idx = {}
for _i in range(num_joints_total):
    _info = pybullet.getJointInfo(arm_id, _i)
    joint_name_to_idx[_info[1].decode('utf-8')] = _i

print("All joints found in URDF:")
for name, idx in sorted(joint_name_to_idx.items(), key=lambda x: x[1]):
    _type = ['REVOLUTE','PRISMATIC','SPHERICAL','PLANAR','FIXED'][pybullet.getJointInfo(arm_id, idx)[2]]
    print(f"  idx={idx:2d}  {_type:<8}  {name}")

# Correct arm joint indices (looked up by name)
shoulder_pan_joint_IDX  = joint_name_to_idx['shoulder_pan_joint']
shoulder_lift_joint_IDX = joint_name_to_idx['shoulder_lift_joint']
elbow_joint_IDX         = joint_name_to_idx['elbow_joint']
wrist_1_joint_IDX       = joint_name_to_idx['wrist_1_joint']
wrist_2_joint_IDX       = joint_name_to_idx['wrist_2_joint']
wrist_3_joint_IDX       = joint_name_to_idx['wrist_3_joint']
print(f"\nArm joint indices: pan={shoulder_pan_joint_IDX}, lift={shoulder_lift_joint_IDX}, "
      f"elbow={elbow_joint_IDX}, w1={wrist_1_joint_IDX}, w2={wrist_2_joint_IDX}, w3={wrist_3_joint_IDX}")

# Set the initial posture — robot arm angled to look down at the workspace
# Pose: arm reaching forward-down, EE camera pointing toward the floor
shoulder_pan_joint_init  =  0.0    # facing forward
shoulder_lift_joint_init = -1.0    # arm tilted forward
elbow_joint_init         =  1.2    # elbow bent
wrist_1_joint_init       = -1.8    # wrist pointing EE downward
wrist_2_joint_init       = -1.57   # wrist neutral
wrist_3_joint_init       =  0.0    # no roll
pybullet.resetJointState(arm_id, shoulder_pan_joint_IDX,  shoulder_pan_joint_init)
pybullet.resetJointState(arm_id, shoulder_lift_joint_IDX, shoulder_lift_joint_init)
pybullet.resetJointState(arm_id, elbow_joint_IDX,         elbow_joint_init)
pybullet.resetJointState(arm_id, wrist_1_joint_IDX,       wrist_1_joint_init)
pybullet.resetJointState(arm_id, wrist_2_joint_IDX,       wrist_2_joint_init)
pybullet.resetJointState(arm_id, wrist_3_joint_IDX,       wrist_3_joint_init)

# Simulation settings
time_length = 2000  # Number of time steps (~8 seconds at 240Hz)
save_img_freq = 4   # Capture image every 4 steps (60fps equivalent)

# Initial joint angles (same as resetJointState above)
initial_angles = np.array([
    shoulder_pan_joint_init,   # 0°
    shoulder_lift_joint_init,  # -90°
    elbow_joint_init,          # +90°
    wrist_1_joint_init,        # -90°
    wrist_2_joint_init,        # -90°
    wrist_3_joint_init,        # 0°
])

# Final angles — sweep pan left/right while maintaining downward-looking EE
final_angles = np.array([
    math.radians(90),    # shoulder_pan:  0° → +30° (sweep toward box)
    math.radians(-55),   # shoulder_lift: small tilt
    math.radians(70),    # elbow: adjust reach
    math.radians(-100),  # wrist_1: keep EE pointing down
    math.radians(-90),   # wrist_2: neutral
    math.radians(0),     # wrist_3: no roll
])

def smoothstep(t):
    """Smooth s-curve interpolation: slow start, fast middle, slow end."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

# ===============================================================

# Camera settings
projection_matrix = pybullet.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=10)

# List to store the obtained camera images
frames = []


# Use ee_link (child of ee_fixed_joint) as camera frame.
# In this UR5 URDF:
#   wrist_3_joint axis = Y  →  approach direction = local Y of wrist_3_link
#   ee_fixed_joint adds rpy=(0,0,π/2)  →  ee_link Y-axis = gripper approach direction
EE_LINK_IDX = joint_name_to_idx.get('ee_fixed_joint', joint_name_to_idx.get('flange-tool0', joint_name_to_idx['wrist_3_joint']))



# Disable PyBullet's default velocity motor damping on all revolute joints
for joint_idx in range(num_joints_total):
    if pybullet.getJointInfo(arm_id, joint_idx)[2] != pybullet.JOINT_FIXED:
        pybullet.setJointMotorControl2(arm_id, joint_idx, pybullet.VELOCITY_CONTROL, force=0)

pybullet.setRealTimeSimulation(0)
for t in range(time_length):
    # --- Compute interpolated target angles (smoothstep from initial → final) ---
    alpha = smoothstep(t / (time_length - 1))  # progress 0.0 → 1.0
    interp_angles = initial_angles + alpha * (final_angles - initial_angles)

    joint_IDXs  = [shoulder_pan_joint_IDX, shoulder_lift_joint_IDX, elbow_joint_IDX,
                   wrist_1_joint_IDX, wrist_2_joint_IDX, wrist_3_joint_IDX]

    # Control each joint toward the current interpolated target
    for jidx, target_pos in zip(joint_IDXs, interp_angles):
        pybullet.setJointMotorControl2(arm_id, jidx, pybullet.POSITION_CONTROL,
                                       targetPosition=float(target_pos),
                                       force=500, positionGain=0.5, velocityGain=1.0)
        
    # Advance the simulation by one time step (240Hz)
    pybullet.stepSimulation()

    # --- Logging: print joint states every 240 steps (= 1 simulated second) ---
    if t % 240 == 0:
        joint_indices = [shoulder_pan_joint_IDX, shoulder_lift_joint_IDX,
                         elbow_joint_IDX, wrist_1_joint_IDX,
                         wrist_2_joint_IDX, wrist_3_joint_IDX]
        names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
        print(f"\n[t={t:4d}  sim_time={t*time_step:.2f}s  alpha={alpha:.3f}]")
        print(f"  {'Joint':<16} {'IDX':>4} {'Current(deg)':>14} {'Interp.Tgt(deg)':>16} {'Error(deg)':>11}")
        print(f"  {'-'*65}")
        for idx, name, tgt in zip(joint_indices, names, interp_angles):
            cur = pybullet.getJointState(arm_id, idx)[0]
            err = math.degrees(tgt) - math.degrees(cur)
            print(f"  {name:<16} {idx:>4} {math.degrees(cur):>14.2f} {math.degrees(tgt):>16.2f} {err:>11.2f}")

    if t % save_img_freq == 0:
        # Get ee_link world state (position + orientation change with arm posture)
        ee_state = pybullet.getLinkState(arm_id, EE_LINK_IDX, computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4])   # world position of ee_link frame
        ee_orn = ee_state[5]             # world orientation quaternion (updates every step)

        # Convert quaternion → 3x3 rotation matrix.
        # Each column is a local axis expressed in world frame:
        #   rot_mat[:, 0] = local X in world
        #   rot_mat[:, 1] = local Y in world  ← gripper approach dir (out of fingers)
        #   rot_mat[:, 2] = local Z in world  ← camera "up" direction
        # This is the 3D equivalent of the original Rz-chain upvector calculation,
        # but works correctly for full 3D arm motion.
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)

        forward = rot_mat[:, 2]   # cameraEyePosition → cameraTargetPosition direction
        up      = rot_mat[:, 2]   # cameraUpVector — changes with arm posture automatically

        cam_eye    = ee_pos + forward * 0.06 + rot_mat[:, 1] * 0.05   # camera origin: 2cm along approach axis
        cam_target = ee_pos + forward * 0.5    # look 50cm ahead along approach axis

        # Compute view matrix — all three vectors are pose-dependent
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up.tolist()             # ← pose-dependent, not fixed!
        )

        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(
            300, 300, view_matrix, projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        frames.append(rgb_img)
# Display animation within Jupyter Notebook
def update(time, frames):
    plt.cla()
    frames_np = np.asarray(frames[time])
    plt.imshow(frames_np)

fig = plt.figure()
time_step_milli_sec = time_step * 1000
ani = FuncAnimation(fig, update, interval=time_step_milli_sec * save_img_freq, frames=len(frames), fargs=(frames,))
HTML(ani.to_jshtml()) # Display as HTML
# ani.save('robot_camera.mp4', writer="ffmpeg") # Save as mp4. Executing this will increase processing time
# ani.save('robot_camera.gif', writer="imagemagick") # Save as gif. Executing this will increase processing time
##try:
    #while True:
    #    pybullet.stepSimulation()
    #    time.sleep(time_step)
##except KeyboardInterrupt:
##    pybullet.disconnect()