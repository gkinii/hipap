import mujoco
import mujoco.viewer
import time
import numpy as np

# Your existing model loading code
XML_PATH = "scripts/mujoco/models/cell/cell.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)

# Create data structure
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


def wxyz_to_xyzw(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w])


def normalize_quaternion(q):
    """Normalize quaternion to unit length"""
    norm = np.linalg.norm(q)
    if norm < 1e-8:  # Avoid division by zero
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity quaternion
    return q / norm

def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions"""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    dot = np.dot(q1, q2)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result)
    
    theta_0 = np.arccos(np.abs(dot))
    theta = theta_0 * t
    
    q2_perp = q2 - q1 * dot
    q2_perp = normalize_quaternion(q2_perp)
    
    return q1 * np.cos(theta) + q2_perp * np.sin(theta)

def smooth_step(t):
    """Smooth S-curve interpolation function (0 to 1)"""
    if t <= 0:
        return 0
    elif t >= 1:
        return 1
    else:
        return t * t * (3 - 2 * t)

# Get initial states
# Mobile robot is at indices 0:7 [x, y, z, qw, qx, qy, qz]
mobile_initial_state = data.qpos[0:7].copy()
print(f"Mobile robot initial state: {mobile_initial_state}")

# AGV is at indices 7:14 [x, y, z, qw, qx, qy, qz] 
agv_initial_state = data.qpos[7:14].copy()
print(f"AGV initial state: {agv_initial_state}")

# Human is at indices 14:21 [x, y, z, qw, qx, qy, qz]
human_initial_state = data.qpos[14:21].copy()
print(f"Human initial state: {human_initial_state}")

# Store initial human joint positions for all joints beyond the main body
print(f"Total DOF in model: {model.nq}")
human_other_joints_initial = data.qpos[21:].copy()  # All joints after main body pose
print(f"Human other joints (beyond main body): {len(human_other_joints_initial)} joints")

# Set human to initial task position with z = 1.282
human_task_initial = np.array([2.0, 5.0, 1.282, 1.0, 0.0, 0.0, 0.0])
data.qpos[14:21] = human_task_initial
print(f"Human main body set to: {data.qpos[14:21]}")

# Keep other human joints fixed at their initial positions
data.qpos[21:] = human_other_joints_initial

print(data.qpos)

# Define AGV waypoints - move from current position to target while keeping orientation
agv_waypoints = [
    # Starting point (current AGV position)
    agv_initial_state.copy(),
    
    # Target position: move to [-4, 0, 0.25] with same orientation
    np.array([
        -4.0,                     # x position
        0.0,                      # y position  
        0.25,                     # z position
        agv_initial_state[3],     # qw (same orientation)
        agv_initial_state[4],     # qx (same orientation)
        agv_initial_state[5],     # qy (same orientation)
        agv_initial_state[6]      # qz (same orientation)
    ])
]

# Define mobile robot waypoints [x, y, z, qw, qx, qy, qz]
mobile_waypoints = [
    # Starting point (current position)
    mobile_initial_state.copy(),
    
    # Point 1: only x changes
    np.array([
        -0.8,                    # x position
        mobile_initial_state[1], # y position (unchanged)
        mobile_initial_state[2], # z position (unchanged)
        mobile_initial_state[3], # qw (unchanged)
        mobile_initial_state[4], # qx (unchanged)
        mobile_initial_state[5], # qy (unchanged)
        mobile_initial_state[6]  # qz (unchanged)
    ]),
    
    # Point 2: x, y, and qz change
    np.array([
        -2.0,                    # x position
        -1.6,                    # y position
        mobile_initial_state[2], # z position (unchanged)
        mobile_initial_state[3], # qw (unchanged for now)
        mobile_initial_state[4], # qx (unchanged)
        mobile_initial_state[5], # qy (unchanged)
        -0.5                     # qz orientation
    ]),
    
    # Point 3: final destination
    np.array([
        -2.2,                    # x position
        -0.2,                    # y position
        mobile_initial_state[2], # z position (unchanged)
        mobile_initial_state[3], # qw (unchanged for now)
        mobile_initial_state[4], # qx (unchanged)
        mobile_initial_state[5], # qy (unchanged)
        -1.0                     # qz orientation
    ])
]

# Define human waypoints for Task 1 [x, y, z, qw, qx, qy, qz] - Updated with z = 1.282
human_waypoints = [
    # Starting point: [2, 5, 1.282] with [1, 0, 0, 0] orientation
    np.array([2.0, 5.0, 1.282, -1.0, 0.0, 0.0, 1]),
    # Point 1: [2, 1.2, 1.282] with [1, 0, 0, 0] orientation
    np.array([2.0, 0.8, 1.282, 1.0, 0.0, 0.0, -1.0]),
    # Point 2: [0.0, 1.2, 1.282] with [1, 0, 0, -1] orientation
    np.array([0.0, 0.8, 1.282, -1.0, 0.0, 0.0, -1.0]),
    np.array([0.4, 0.0, 1.282, 1.0, 0.0, 0.0, -1.0]),
    np.array([0.4, 0.8, 1.282, -1.0, 0.0, 0.0, -1.0]),
    np.array([0.4, 0.0, 1.282, 1.0, 0.0, 0.0, -1.0]),
]

# Normalize quaternions for mobile robot waypoints 2 and 3
for i in range(2, len(mobile_waypoints)):
    mobile_waypoints[i][3:7] = normalize_quaternion(mobile_waypoints[i][3:7])

# Normalize quaternions for human waypoints (all waypoints)
for i in range(len(human_waypoints)):
    human_waypoints[i][3:7] = normalize_quaternion(human_waypoints[i][3:7])
    # Ensure no NaN or invalid values
    if np.any(np.isnan(human_waypoints[i][3:7])) or np.any(np.isinf(human_waypoints[i][3:7])):
        print(f"Warning: Invalid quaternion at waypoint {i}, using default [1,0,0,0]")
        human_waypoints[i][3:7] = np.array([1.0, 0.0, 0.0, 0.0])

print("\nAGV Waypoints:")
for i, wp in enumerate(agv_waypoints):
    print(f"  Point {i}: pos=[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}], quat=[{wp[3]:.3f}, {wp[4]:.3f}, {wp[5]:.3f}, {wp[6]:.3f}]")

print("\nMobile Robot Waypoints:")
for i, wp in enumerate(mobile_waypoints):
    print(f"  Point {i}: pos=[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}], quat=[{wp[3]:.3f}, {wp[4]:.3f}, {wp[5]:.3f}, {wp[6]:.3f}]")

print("\nHuman Task 1 Waypoints:")
for i, wp in enumerate(human_waypoints):
    print(f"  Point {i}: pos=[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}], quat=[{wp[3]:.3f}, {wp[4]:.3f}, {wp[5]:.3f}, {wp[6]:.3f}]")

# Trajectory parameters
agv_segment_duration = 4.0     # 4 seconds for AGV movement
mobile_segment_duration = 3.0  # 3 seconds per mobile robot segment
human_segment_duration = 3.0   # 3 seconds per human segment
wait_time = 1.0                # Wait 1 second before starting
pause_between_robots = 1.0     # Pause between robots

# State variables
current_phase = "waiting"      # "waiting", "human_moving", "pause1", "agv_moving", "pause2", "mobile_moving"
phase_start_time = 0.0
current_agv_segment = 0
current_mobile_segment = 0
current_human_segment = 0
agv_completed = False
mobile_completed = False
human_completed = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera to be further away
    viewer.cam.distance = 20.0     # Increase distance
    viewer.cam.azimuth = -90       # Horizontal rotation (degrees)
    viewer.cam.elevation = -90     # Vertical angle (degrees)
    # time.sleep(2)
    print(f"Camera set to distance: {viewer.cam.distance}")
    
    while viewer.is_running():
        step_start = time.time()
        
        # Phase management
        if current_phase == "waiting" and data.time > wait_time:
            current_phase = "human_moving"
            phase_start_time = data.time
            current_human_segment = 0
            print("Phase 1: Starting human movement (Task 1)...")
        
        elif current_phase == "human_moving" and human_completed:
            current_phase = "pause1"
            phase_start_time = data.time
            print("Phase 2: Human completed, pausing...")
        
        elif current_phase == "pause1" and (data.time - phase_start_time) > pause_between_robots:
            current_phase = "agv_moving"
            phase_start_time = data.time
            current_agv_segment = 0
            print("Phase 3: Starting AGV movement...")
        
        elif current_phase == "agv_moving" and agv_completed:
            current_phase = "pause2"
            phase_start_time = data.time
            print("Phase 4: AGV completed, pausing...")
        
        elif current_phase == "pause2" and (data.time - phase_start_time) > pause_between_robots:
            current_phase = "mobile_moving"
            phase_start_time = data.time
            current_mobile_segment = 0
            print("Phase 5: Starting mobile robot movement...")
        
        # Check for simulation stability
        if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)) or np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel)):
            print("Simulation became unstable! Resetting...")
            # Reset to a stable state
            mujoco.mj_resetData(model, data)
            data.qpos[14:21] = human_task_initial
            data.qpos[21:] = human_other_joints_initial
            mujoco.mj_forward(model, data)
            continue
        
        # Keep human other joints fixed throughout the simulation
        data.qpos[21:] = human_other_joints_initial
        
        # Execute human trajectory (first)
        if current_phase == "human_moving" and current_human_segment < len(human_waypoints) - 1:
            elapsed_time = data.time - phase_start_time
            segment_elapsed = elapsed_time - (current_human_segment * human_segment_duration)
            
            if segment_elapsed <= human_segment_duration:
                # Calculate interpolation parameter (0 to 1)
                t_raw = segment_elapsed / human_segment_duration
                t_smooth = smooth_step(t_raw)
                
                # Get current segment waypoints
                start_point = human_waypoints[current_human_segment]
                end_point = human_waypoints[current_human_segment + 1]
                
                # Interpolate position (linear)
                current_pos = start_point[0:3] + t_smooth * (end_point[0:3] - start_point[0:3])
                
                # Interpolate orientation (spherical)
                current_quat = quaternion_slerp(start_point[3:7], end_point[3:7], t_smooth)
                
                # Ensure quaternion is valid
                if np.any(np.isnan(current_quat)) or np.any(np.isinf(current_quat)) or np.linalg.norm(current_quat) < 0.1:
                    print(f"Warning: Invalid quaternion during interpolation, using start quaternion")
                    current_quat = normalize_quaternion(start_point[3:7])
                
                # Update human state (indices 14:21) - only main body
                data.qpos[14:17] = current_pos
                data.qpos[17:21] = current_quat
                
                # Print progress every 0.5 seconds
                if int(segment_elapsed * 2) != int((segment_elapsed - model.opt.timestep) * 2):
                    print(f"Human Task 1 Point {current_human_segment + 1}: {t_raw*100:.1f}% - Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            
            else:
                # Current segment complete
                current_human_segment += 1
                
                if current_human_segment < len(human_waypoints) - 1:
                    print(f"Human reached Point {current_human_segment}! Moving to Point {current_human_segment + 1}...")
                else:
                    # All segments complete
                    data.qpos[14:21] = human_waypoints[-1]  # Ensure exact final position
                    human_completed = True
                    print("All human Task 1 waypoints completed!")
                    print(f"Human final position: {data.qpos[14:17]}")
                    print(f"Human final quaternion: {data.qpos[17:21]}")
        
        # Execute AGV trajectory (second)
        if current_phase == "agv_moving" and not agv_completed:
            elapsed_time = data.time - phase_start_time
            
            if elapsed_time <= agv_segment_duration:
                # Calculate interpolation parameter (0 to 1)
                t_raw = elapsed_time / agv_segment_duration
                t_smooth = smooth_step(t_raw)
                
                # Get AGV waypoints
                start_point = agv_waypoints[0]
                end_point = agv_waypoints[1]
                
                # Interpolate position (linear)
                current_pos = start_point[0:3] + t_smooth * (end_point[0:3] - start_point[0:3])
                
                # Keep same orientation (no interpolation needed)
                current_quat = start_point[3:7]
                
                # Update AGV state (indices 7:14)
                data.qpos[7:10] = current_pos
                data.qpos[10:14] = current_quat
                
                # Print progress every 0.5 seconds
                if int(elapsed_time * 2) != int((elapsed_time - model.opt.timestep) * 2):
                    print(f"AGV: {t_raw*100:.1f}% - Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            else:
                # AGV movement complete
                data.qpos[7:14] = agv_waypoints[1]  # Ensure exact final position
                agv_completed = True
                print("AGV movement completed!")
                print(f"AGV final position: {data.qpos[7:10]}")
        
        # Execute mobile robot trajectory (third)
        if current_phase == "mobile_moving" and current_mobile_segment < len(mobile_waypoints) - 1:
            elapsed_time = data.time - phase_start_time
            segment_elapsed = elapsed_time - (current_mobile_segment * mobile_segment_duration)
            
            if segment_elapsed <= mobile_segment_duration:
                # Calculate interpolation parameter (0 to 1)
                t_raw = segment_elapsed / mobile_segment_duration
                t_smooth = smooth_step(t_raw)
                
                # Get current segment waypoints
                start_point = mobile_waypoints[current_mobile_segment]
                end_point = mobile_waypoints[current_mobile_segment + 1]
                
                # Interpolate position (linear)
                current_pos = start_point[0:3] + t_smooth * (end_point[0:3] - start_point[0:3])
                
                # Interpolate orientation (spherical)
                current_quat = quaternion_slerp(start_point[3:7], end_point[3:7], t_smooth)
                
                # Update mobile robot state (indices 0:7)
                data.qpos[0:3] = current_pos
                data.qpos[3:7] = current_quat
                
                # Print progress every 0.5 seconds
                if int(segment_elapsed * 2) != int((segment_elapsed - model.opt.timestep) * 2):
                    print(f"Mobile Robot Point {current_mobile_segment + 1}: {t_raw*100:.1f}% - Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            
            else:
                # Current segment complete
                current_mobile_segment += 1
                
                if current_mobile_segment < len(mobile_waypoints) - 1:
                    print(f"Mobile Robot reached Point {current_mobile_segment}! Moving to Point {current_mobile_segment + 1}...")
                else:
                    # All segments complete
                    data.qpos[0:7] = mobile_waypoints[-1]  # Ensure exact final position
                    mobile_completed = True
                    print("All mobile robot waypoints completed!")
                    print(f"Mobile robot final position: {data.qpos[0:3]}")
                    print(f"Mobile robot final quaternion: {data.qpos[3:7]}")
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Sync viewer
        viewer.sync()
        
        # Maintain real-time rate
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)