import json
import math
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import (
    Grid,
    spawn_goals,
    theta_star,
    smooth_path_with_beziers,
    create_padded,
    get_bounds,
    NoFreeCellsError
)

from generate_grid import generate_occupancy_grid
# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
# MODE SELECTION - Set one to True
GENERATE_LARGE_TRAJECTORY = True   # Generate large multi-station trajectories
GENERATE_TASK_LIBRARY = False      # Generate task library trajectories

# Large trajectory settings
NUM_LARGE_TRAJECTORIES = 1        # Number of large trajectories to generate
NUM_WAYPOINTS = 10                  # Number of waypoints per large trajectory
LARGE_CSV_PATH = "path_large.csv"

# Task library settings
NUM_PARTS   = 10
CSV_PATH    = "paths.csv"
SAVE_EVERY  = 100
DRAW        = False

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
x_min, y_min, x_max, y_max = -5.3, -6.3, 5.4, 6.3

MOBILE_WIDTH, MOBILE_HEIGHT = 1.75, 0.9
AGV_WIDTH, AGV_HEIGHT       = 1.75, 3.0
TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT = 1.0, 0.5
BATTERY_WIDTH, BATTERY_HEIGHT     = 1.2, 0.5
DYDYNAMIC_CENTER= (-0.05, 1.25)
DYNAMIC_WIDTH, DYNAMIC_HEIGHT = 1.0, 1.

DYDYNAMIC2_CENTER= (0.5, 3.5)
DYNAMIC2_WIDTH, DYNAMIC2_HEIGHT= 2., 2.


TOOL_VAR_X, TOOL_VAR_Y = 1.5, 0.5
BATTERY_VAR_X = 0.5

TOOL_ROT_RANGE = 90
AGV_ROT_RANGE  = 10
MOBILE_ROT_RANGE = 10

if GENERATE_TASK_LIBRARY:
    tasks = json.load(open(Path("task_library.json")))

# ───────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT + PARTITION
# ───────────────────────────────────────────────────────────────────────────────
def build_environment(ax, agv_key, mobile_key, draw=True):
    pads = []
    objects_no_pad = []

    def cp(ax, *args, **kwargs):
        return create_padded(ax, *args, draw=draw, **kwargs)

    # Dynamic object
    dynamic = cp(ax, DYDYNAMIC_CENTER, DYNAMIC_WIDTH, DYNAMIC_HEIGHT,
                 color="lightgrey", edgecolor="black",
                 jitter_x=(-0.5, 0.5), jitter_y=(-0.5, 0.5),
                 rotation_deg=random.uniform(-10, 10))
    pads.append(dynamic["collision_data"])
    objects_no_pad.append(dynamic["original_collision_data"])

    dynamic2 = cp(ax, DYDYNAMIC2_CENTER, DYNAMIC2_WIDTH, DYNAMIC2_HEIGHT,
                 color="lightgrey", edgecolor="black",
                 jitter_x=(-0.5, 0.5), jitter_y=(-0.5, 0.5),
                 rotation_deg=random.uniform(-10, 10))
    pads.append(dynamic2["collision_data"])
    objects_no_pad.append(dynamic2["original_collision_data"])

    # Desk
    desk = cp(ax, (2.8, -6.25), 2.5, 7.0, color="wheat", edgecolor="brown", jitter_x=(-1, 1), jitter_y=(-1, 4.5),)
    pads.append(desk["collision_data"])
    objects_no_pad.append(desk["original_collision_data"])

    # Battery
    battery = cp(ax, (-0.4, -3.6), BATTERY_WIDTH, BATTERY_HEIGHT,
                 color="lightblue", jitter_x=(0, BATTERY_VAR_X), edgecolor="blue")
    pads.append(battery["collision_data"])
    objects_no_pad.append(battery["original_collision_data"])

    # Tool
    tool = cp(ax, (-0.3, -1.35), TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT,
              color="lightgreen", edgecolor="green",
              jitter_x=(0, TOOL_VAR_X),
              jitter_y=(-TOOL_VAR_Y, TOOL_VAR_Y * 0.1),
              rotation_deg=random.uniform(-TOOL_ROT_RANGE, TOOL_ROT_RANGE))
    pads.append(tool["collision_data"])
    objects_no_pad.append(tool["original_collision_data"])

    # AGV
    agv_center = {"random": (-4.55, -2.75), "marriage_point_agv": (-4.55, -3.25)}[agv_key]
    agv_jitter_y = (-3, 5) if agv_key == "random" else (0., 1.)
    agv = cp(ax, agv_center, AGV_WIDTH, AGV_HEIGHT,
             color="lightcoral", edgecolor="red",
             jitter_y=agv_jitter_y, jitter_x=(0., 0.8),
             rotation_deg=random.uniform(-AGV_ROT_RANGE, AGV_ROT_RANGE))
    pads.append(agv["collision_data"])
    objects_no_pad.append(agv["original_collision_data"])

    # Mobile
    if mobile_key == "assembly":
        mob_pos = (battery["center"][0] - 0.6, battery["center"][1] - 1.395)
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 jitter_x=(-0.5, 0.3), jitter_y=(-0.2, 0.2),
                 rotation_deg=random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE))
    else:
        mob_pos = (agv["center"][0] + 0.33 * AGV_WIDTH, agv["center"][1])
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 rotation_deg=90 + random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE),
                 jitter_y=(-0.5, 0.5), jitter_x=(-0.1, 0.1))
    pads.append(mob["collision_data"])
    objects_no_pad.append(mob["original_collision_data"])

    centers = tool["center"], agv["center"], mob["center"], battery["center"]
    return pads, centers, objects_no_pad


def partition_goals(entries, n_parts):
    partitions = [[] for _ in range(n_parts)]
    for entry in entries:
        n = entry["n_goals"]
        slice_size = int(round(n * (1 / n_parts)))
        p = [slice_size for _ in range(n_parts - 1)]
        p.append(n - sum(p))
        for i in range(n_parts):
            partitions[i].append((entry["goal"], p[i]))
    return partitions

# ───────────────────────────────────────────────────────────────────────────────
# PATH GENERATION + CSV LOGGING
# ───────────────────────────────────────────────────────────────────────────────
def generate_paths_and_log(ax, starts, goals, grid, pads, writer,
                           task_id, grid_id, mobile_center, agv_center,
                           traj_id_start=0, frame_interval=0.4,
                           min_points=25, draw=True):
    """
    Generate paths with speed modeling, standing periods, trajectory IDs, and task/grid IDs.
    """
    traj_id = traj_id_start
    
    # Speed range for humans (units per second)
    MIN_SPEED = 0.8
    MAX_SPEED = 1.5
    SPEED_VARIABILITY = 0.05  # Small variability around base speed
    
    # Standing parameters
    MIN_STANDING_FRAMES = 5
    MAX_STANDING_FRAMES = 7
    STANDING_POSITION_NOISE = 0.02  # Small position variability while standing (2cm)
    
    for sx, sy in starts:
        for gx, gy in goals:
            raw = theta_star((sx, sy), (gx, gy), grid, pads)
            if not raw:
                continue
            path = smooth_path_with_beziers(raw)
            
            if len(path) < 2:
                continue
            
            # Reset frame_id for each trajectory
            frame_id = 0
            trajectory_frames = []  # Store all frames for this trajectory
            
            # Assign a base speed for this path (with small random variation)
            base_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
            
            # Calculate cumulative distances and times along the path
            distances = []
            cumulative_dist = [0]
            
            for i in range(1, len(path)):
                # Calculate distance between consecutive points
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(dist)
                cumulative_dist.append(cumulative_dist[-1] + dist)
            
            # Calculate time for each segment with small speed variations
            cumulative_time = [0]
            for dist in distances:
                # Add small random variation to speed for each segment
                segment_speed = base_speed * np.random.uniform(
                    1 - SPEED_VARIABILITY, 
                    1 + SPEED_VARIABILITY
                )
                segment_time = dist / segment_speed
                cumulative_time.append(cumulative_time[-1] + segment_time)
            
            # Total time for the path
            total_time = cumulative_time[-1]
            
            # Generate movement frames at fixed time intervals
            current_time = 0
            while current_time <= total_time:
                # Find position at current_time through linear interpolation
                x, y = interpolate_position_at_time(
                    path, cumulative_time, current_time
                )
                trajectory_frames.append((x, y))
                current_time += frame_interval
            
            # Ensure we record the final position if not already recorded
            if current_time - frame_interval < total_time:
                x, y = path[-1]
                trajectory_frames.append((x, y))
            
            # Check if we need to add standing frames
            num_movement_frames = len(trajectory_frames)
            
            if num_movement_frames < min_points:
                # Calculate how many standing frames to add
                frames_needed = min_points - num_movement_frames
                
                # Distribute standing frames between start and end
                # Slightly more at the beginning (preparing to move)
                start_standing_frames = min(
                    np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1),
                    int(frames_needed * 0.6)
                )
                end_standing_frames = min(
                    np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1),
                    frames_needed - start_standing_frames
                )
                
                # If still need more frames, add them to the beginning
                if start_standing_frames + end_standing_frames + num_movement_frames < min_points:
                    start_standing_frames = min_points - num_movement_frames - end_standing_frames
                
                # Generate standing frames at the beginning
                initial_standing_frames = []
                start_x, start_y = trajectory_frames[0] if trajectory_frames else (sx, sy)
                for _ in range(start_standing_frames):
                    # Add small random movement while standing
                    noise_x = np.random.normal(0, STANDING_POSITION_NOISE)
                    noise_y = np.random.normal(0, STANDING_POSITION_NOISE)
                    initial_standing_frames.append((start_x + noise_x, start_y + noise_y))
                
                # Generate standing frames at the end
                final_standing_frames = []
                end_x, end_y = trajectory_frames[-1] if trajectory_frames else (gx, gy)
                for _ in range(end_standing_frames):
                    # Add small random movement while standing
                    noise_x = np.random.normal(0, STANDING_POSITION_NOISE)
                    noise_y = np.random.normal(0, STANDING_POSITION_NOISE)
                    final_standing_frames.append((end_x + noise_x, end_y + noise_y))
                
                # Combine all frames: standing at start + movement + standing at end
                trajectory_frames = initial_standing_frames + trajectory_frames + final_standing_frames
            
            # Write all frames to CSV with task_id and grid_id
            for x, y in trajectory_frames:
                # Write data with task_id, grid_id, trajectory ID, etc.
                # Human position (agent_id=0, agent_type=1)
                writer.writerow([task_id, grid_id, traj_id, frame_id, 0, 1, x, y, 0])
                # AGV position (agent_id=1, agent_type=0)
                writer.writerow([task_id, grid_id, traj_id, frame_id, 1, 0, agv_center[0], agv_center[1], 0])
                # Mobile position (agent_id=2, agent_type=0)
                writer.writerow([task_id, grid_id, traj_id, frame_id, 2, 0, mobile_center[0], mobile_center[1], 0])
                
                frame_id += 1
            
            # Draw the path if needed (only draw the actual movement path, not standing)
            if draw and ax is not None:
                px, py = zip(*path)
                ax.plot(px, py, lw=1.2)
            
            # Increment trajectory ID for next trajectory
            traj_id += 1
    
    return traj_id


def interpolate_position_at_time(path, cumulative_time, target_time):
    """
    Interpolate position along path at a specific time.
    """
    # Handle edge cases
    if target_time <= 0:
        return path[0]
    if target_time >= cumulative_time[-1]:
        return path[-1]
    
    # Find the segment containing target_time
    for i in range(1, len(cumulative_time)):
        if cumulative_time[i] >= target_time:
            # Interpolate between path[i-1] and path[i]
            t0 = cumulative_time[i-1]
            t1 = cumulative_time[i]
            
            # Linear interpolation factor
            alpha = (target_time - t0) / (t1 - t0)
            
            x = path[i-1][0] + alpha * (path[i][0] - path[i-1][0])
            y = path[i-1][1] + alpha * (path[i][1] - path[i-1][1])
            
            return (x, y)
    
    return path[-1]


# ───────────────────────────────────────────────────────────────────────────────
# LARGE TRAJECTORY GENERATION
# ───────────────────────────────────────────────────────────────────────────────
def generate_large_trajectory(ax, grid, pads, writer, task_id, grid_id, 
                              mobile_center, agv_center, traj_id,
                              num_waypoints=6, frame_interval=0.4, 
                              min_points_per_segment=25, draw=True):
    """
    Generate a large trajectory that visits multiple stations in sequence.
    
    Waypoint distribution:
    - 2 waypoints in tool station
    - 2 waypoints in assembly area
    - 2 waypoints in random locations
    """
    # Speed parameters
    MIN_SPEED = 0.8
    MAX_SPEED = 1.5
    SPEED_VARIABILITY = 0.05
    
    # Standing parameters at each waypoint
    MIN_STANDING_FRAMES = 5
    MAX_STANDING_FRAMES = 10
    STANDING_POSITION_NOISE = 0.02
    
    # Generate waypoints from different areas
    waypoints = []
    
    # Tool station waypoints (2)
    tool_area = get_bounds((-0.3, -1.35), 1.05, 0.55, 0.0)
    for _ in range(2):
        pts = spawn_goals(1, grid, pads, tool_area["x_bounds"], tool_area["y_bounds"])
        waypoints.extend(pts)
    
    # Assembly area waypoints (2)
    assembly_area = get_bounds((-0.4, -3.15), 1.2, 0.5, 0.05)
    for _ in range(2):
        pts = spawn_goals(1, grid, pads, assembly_area["x_bounds"], assembly_area["y_bounds"])
        waypoints.extend(pts)
    
    # Random waypoints (remaining)
    random_area = get_bounds((0.0, 0.0), 10.0, 12.0, 0)
    remaining = num_waypoints - len(waypoints)
    for _ in range(remaining):
        pts = spawn_goals(1, grid, pads, random_area["x_bounds"], random_area["y_bounds"])
        waypoints.extend(pts)
    
    # Shuffle waypoints to create varied paths
    random.shuffle(waypoints)
    
    print(f"  Generated {len(waypoints)} waypoints for large trajectory {traj_id}")
    
    # Build the complete trajectory by connecting all waypoints
    all_trajectory_frames = []
    frame_id = 0
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]
        
        # Find path between waypoints
        raw = theta_star(start, goal, grid, pads)
        if not raw:
            print(f"  Warning: Could not find path between waypoint {i} and {i+1}")
            continue
        
        path = smooth_path_with_beziers(raw)
        if len(path) < 2:
            continue
        
        # Calculate distances and times
        distances = []
        cumulative_dist = [0]
        for j in range(1, len(path)):
            dx = path[j][0] - path[j-1][0]
            dy = path[j][1] - path[j-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
            cumulative_dist.append(cumulative_dist[-1] + dist)
        
        # Calculate time with speed variations
        base_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
        cumulative_time = [0]
        for dist in distances:
            segment_speed = base_speed * np.random.uniform(
                1 - SPEED_VARIABILITY, 
                1 + SPEED_VARIABILITY
            )
            segment_time = dist / segment_speed
            cumulative_time.append(cumulative_time[-1] + segment_time)
        
        total_time = cumulative_time[-1]
        
        # Generate movement frames
        segment_frames = []
        current_time = 0
        while current_time <= total_time:
            x, y = interpolate_position_at_time(path, cumulative_time, current_time)
            segment_frames.append((x, y))
            current_time += frame_interval
        
        # Ensure final position
        if current_time - frame_interval < total_time:
            x, y = path[-1]
            segment_frames.append((x, y))
        
        # Add standing period at waypoint (except at the very end)
        if i < len(waypoints) - 2:
            standing_frames_count = np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1)
            end_x, end_y = segment_frames[-1] if segment_frames else goal
            
            for _ in range(standing_frames_count):
                noise_x = np.random.normal(0, STANDING_POSITION_NOISE)
                noise_y = np.random.normal(0, STANDING_POSITION_NOISE)
                segment_frames.append((end_x + noise_x, end_y + noise_y))
        
        all_trajectory_frames.extend(segment_frames)
        
        # Draw segment
        if draw and ax is not None:
            px, py = zip(*path)
            ax.plot(px, py, lw=1.2, alpha=0.7)
    
    # Write all frames to CSV
    for x, y in all_trajectory_frames:
        writer.writerow([task_id, grid_id, traj_id, frame_id, 0, 1, x, y, 0])
        writer.writerow([task_id, grid_id, traj_id, frame_id, 1, 0, agv_center[0], agv_center[1], 0])
        writer.writerow([task_id, grid_id, traj_id, frame_id, 2, 0, mobile_center[0], mobile_center[1], 0])
        frame_id += 1
    
    # Draw waypoints
    if draw and ax is not None and waypoints:
        wx, wy = zip(*waypoints)
        ax.plot(wx, wy, 'ro', ms=8, label='waypoints')
        # Draw arrows between waypoints
        for i in range(len(waypoints) - 1):
            ax.annotate('', xy=waypoints[i+1], xytext=waypoints[i],
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))
    
    print(f"  Large trajectory {traj_id}: {len(all_trajectory_frames)} total frames")
    
    return traj_id + 1


# ───────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def main_large_trajectories():
    """Generate large multi-station trajectories"""
    traj_id_counter = 0
    grid_id_counter = 0
    FRAME_INTERVAL = 0.4
    
    print(f"=== LARGE TRAJECTORY MODE ===")
    print(f"Generating {NUM_LARGE_TRAJECTORIES} large trajectories with {NUM_WAYPOINTS} waypoints each")
    
    with open(LARGE_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id", "agent_id", "agent_type", "x", "y", "z"])
        
        for traj_num in range(NUM_LARGE_TRAJECTORIES):
            ax = None
            if DRAW:
                fig, ax = plt.subplots(figsize=(9, 7))
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"Large Trajectory {traj_num + 1}/{NUM_LARGE_TRAJECTORIES}")
            
            while True:
                try:
                    # Build environment with random configuration
                    agv_key = random.choice(["random", "marriage_point_agv"])
                    mobile_key = random.choice(["assembly", "other"])
                    
                    pads, (tool_c, agv_c, mob_c, batt_c), objects_no_pad = \
                        build_environment(ax, agv_key, mobile_key, draw=DRAW)
                    
                    grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)
                    
                    # Create grid_id
                    task_id = f"large_{traj_num}"
                    grid_id = f"large_{grid_id_counter}"
                    
                    # Generate PGM file
                    generate_occupancy_grid(grid, objects_no_pad, grid_id, grid_size=256)
                    grid_id_counter += 1
                    
                    print(f"[Large Trajectory {traj_num + 1}] Task ID: {task_id}, Grid ID: {grid_id}")
                    
                    # Generate large trajectory
                    traj_id_counter = generate_large_trajectory(
                        ax, grid, pads, writer,
                        task_id=task_id,
                        grid_id=grid_id,
                        mobile_center=mob_c,
                        agv_center=agv_c,
                        traj_id=traj_id_counter,
                        num_waypoints=NUM_WAYPOINTS,
                        frame_interval=FRAME_INTERVAL,
                        draw=DRAW
                    )
                    
                    break
                    
                except NoFreeCellsError as e:
                    print(f"{e} → rebuilding environment and retrying...")
                    if DRAW and ax is not None:
                        ax.cla()
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_title(f"Large Trajectory {traj_num + 1}/{NUM_LARGE_TRAJECTORIES}")
                    continue
            
            if DRAW and ax is not None:
                ax.legend(loc='upper right')
                plt.tight_layout()
                plt.show()
    
    print(f"\nDone! Generated {NUM_LARGE_TRAJECTORIES} large trajectories.")
    print(f"Paths saved to {LARGE_CSV_PATH}")
    print(f"Generated {grid_id_counter} occupancy grids as PGM files.")


def main_task_library():
    """Original task library trajectory generation"""
    traj_id_counter = 0
    grid_id_counter = 0
    FRAME_INTERVAL = 0.4
    MIN_POINTS = 25
    
    print(f"=== TASK LIBRARY MODE ===")
    
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id", "agent_id", "agent_type", "x", "y", "z"])

        for task in tasks:
            start_parts = partition_goals(task["start_pos"], NUM_PARTS)
            goal_parts  = partition_goals(task["goal_pos"], NUM_PARTS)

            for stage in range(NUM_PARTS):
                ax = None
                if DRAW:
                    fig, ax = plt.subplots(figsize=(9,7))
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: {task['title']}")

                while True:
                    try:
                        pads, (tool_c, agv_c, mob_c, batt_c), objects_no_pad = \
                            build_environment(ax, task["agv_pos"], task["mobile_pos"], draw=DRAW)
                        grid = Grid((x_min,x_max),(y_min,y_max),resolution=0.2)
                        
                        grid_id = f"{task['task_id']}_{stage+1}"
                        generate_occupancy_grid(grid, objects_no_pad, grid_id, grid_size=256)
                        grid_id_counter += 1

                        GOAL_AREA = {
                            "random":           get_bounds((0.0,0.0),10.0,12.0,0),
                            "tool_station":     get_bounds(tool_c,1.05,0.55,0.0),
                            "battery_assembly": get_bounds((batt_c[0],batt_c[1]+0.45), batt_c[0]+0.6,0.0,0.05),
                            "agv_ph1":          get_bounds((agv_c[0]+2,agv_c[1]+3.5),1.5,1.5,0.5),
                            "agv_ph2":          get_bounds(agv_c, AGV_WIDTH+0.2, AGV_HEIGHT+0.2,0.0),
                        }

                        starts, goals = [], []
                        for goal_type,n in start_parts[stage]:
                            if n>0:
                                area = GOAL_AREA[goal_type]
                                pts = spawn_goals(n, grid, pads, area["x_bounds"], area["y_bounds"])
                                starts.extend(pts)
                        for goal_type,n in goal_parts[stage]:
                            if n>0:
                                area = GOAL_AREA[goal_type]
                                pts = spawn_goals(n, grid, pads, area["x_bounds"], area["y_bounds"])
                                goals.extend(pts)

                        start_count = len(starts)
                        goal_count = len(goals)
                        num_trajectories = start_count * goal_count
                        print(f"[Task {task['task_id']} - Part {stage+1}] Generated {start_count} starts, {goal_count} goals ({num_trajectories} trajectories).")
                        print(f"  Grid ID: {grid_id}, Task ID: {task['task_id']}")
                        break

                    except NoFreeCellsError as e:
                        print(f"{e} → rebuilding environment and retrying...")
                        if DRAW and ax is not None:
                            ax.cla()
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_aspect('equal')
                            ax.grid(True, alpha=0.3)
                            ax.set_title(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: {task['title']}")
                        continue

                if DRAW and ax is not None:
                    if starts: ax.plot(*zip(*starts), 'bo', ms=4, label='start')
                    if goals:  ax.plot(*zip(*goals),  'rx', ms=5, label='goal')

                traj_id_counter = generate_paths_and_log(
                    ax, starts, goals, grid, pads, writer,
                    task_id=task['task_id'],
                    grid_id=grid_id,
                    mobile_center=mob_c,
                    agv_center=agv_c,
                    traj_id_start=traj_id_counter,
                    frame_interval=FRAME_INTERVAL,
                    min_points=MIN_POINTS,
                    draw=DRAW
                )

                if DRAW and ax is not None:
                    ax.legend(loc='upper right')
                    plt.tight_layout()
                    plt.show()

    print(f"\nDone. Generated {traj_id_counter} trajectories. Paths saved to {CSV_PATH}")
    print(f"Trajectories with < {MIN_POINTS} frames were padded with standing periods.")
    print(f"Generated {grid_id_counter} occupancy grids as PGM files.")


# ───────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────
def main():
    if GENERATE_LARGE_TRAJECTORY:
        main_large_trajectories()
    elif GENERATE_TASK_LIBRARY:
        main_task_library()
    else:
        print("ERROR: No mode selected! Set either GENERATE_LARGE_TRAJECTORY or GENERATE_TASK_LIBRARY to True.")


if __name__ == '__main__':
    main()