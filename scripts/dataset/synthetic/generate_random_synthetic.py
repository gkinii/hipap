import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

from utils import Grid, spawn_goals, get_bounds, NoFreeCellsError
from generate_grid import generate_occupancy_grid
from trajectory_utils import build_random_environment

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Origin Mode: "fixed" or "random"
ORIGIN_MODE = "random" 

# Generation Settings
NUM_RANDOM_MAPS = 300
STATIONS_PER_MAP = 7  # Number of stations (serve as both starts and goals)
CSV_PATH = "256/paths_random.csv"
GRID_OUTPUT_DIR = "256" 

# Masking Settings
MASK_PROBABILITY = 0.2 

# Obstacle Density Settings
OBSTACLE_DENSITY = 0.2
REGULAR_OBSTACLE_SIZE_RANGE = (0.7, 2) 
NUM_SMALL_OBSTACLES = 5
SMALL_OBSTACLE_SIZE_RANGE = (0.1, 0.3) 
NUM_NARROW_OBSTACLES = 2
NARROW_OBSTACLE_WIDTH_RANGE = (0.1, 0.3) 
NARROW_OBSTACLE_HEIGHT_RANGE = (1.5, 3) 

# General Settings
DRAW = False
FRAME_INTERVAL = 0.4
MIN_POINTS = 25

# Fixed Grid Settings
GRID_SIZE = 256
GRID_RESOLUTION = 0.05

# Randomization Ranges
ORIG_GRID_WIDTH_RANGE = (160, 256) 
ORIG_GRID_HEIGHT_RANGE = (160, 256) 
X_MIN_RANGE = (-4, -8)
Y_MIN_RANGE = (-4, -8)

# Origin Configuration
FIXED_MAP_ORIGIN_X = -14.4
FIXED_MAP_ORIGIN_Y = -5.2

ORIGIN_OFFSET_X_RANGE = (-2.0, 2.0) 
ORIGIN_OFFSET_Y_RANGE = (-2.0, 2.0) 

# -------------------------------------------------------------------------------
# NEW: Station Spawning Configuration
# Reduce this value to place goals closer to obstacle edges (in meters)
STATION_OFFSET_DISTANCE = 0.2
# -------------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_num_regular_obstacles(workspace_area, density=OBSTACLE_DENSITY):
    num_obstacles = int(np.sqrt(workspace_area) * density)
    return num_obstacles

def setup_plot(title, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return ax

def finalize_plot(ax):
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def calculate_workspace_bounds(map_origin_x, map_origin_y, workspace_width_px, workspace_height_px,
                               grid_size=GRID_SIZE, grid_resolution=GRID_RESOLUTION):
    pad_x_pixels = grid_size - workspace_width_px
    pad_y_pixels = grid_size - workspace_height_px
    pad_left = pad_x_pixels // 2
    pad_bottom = pad_y_pixels // 2
    
    x_min = map_origin_x + pad_left * grid_resolution
    y_min = map_origin_y + pad_bottom * grid_resolution
    
    workspace_width_m = workspace_width_px * grid_resolution
    workspace_height_m = workspace_height_px * grid_resolution
    
    x_max = x_min + workspace_width_m
    y_max = y_min + workspace_height_m
    
    return {
        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
        'width': workspace_width_m, 'height': workspace_height_m,
        'area': workspace_width_m * workspace_height_m,
        'orig_width': workspace_width_px, 'orig_height': workspace_height_px,
        'pad_x_pixels': pad_x_pixels, 'pad_y_pixels': pad_y_pixels
    }

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: STATION SPAWNING LOGIC (Handling Dictionary Pads)
# ═══════════════════════════════════════════════════════════════════════════════

def rotate_point(x, y, theta_deg):
    """Rotate a point (x,y) by theta_deg around (0,0)"""
    rad = math.radians(theta_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

def check_collision_with_pads(px, py, pads):
    """
    Checks if a point (px, py) is inside any of the obstacles (pads).
    Uses the inverse rotation to check against the obstacle's local AABB 
    for rectangles, and distance check for circles.
    
    This replaces grid.is_free, as grid.is_free is not available.
    """
    for pad in pads:
        shape_type = pad.get('type', 'rectangle')
        # Use 'original_center' as the rotation/size reference point
        cx, cy = pad['original_center']
        rot = pad.get('cumulative_rotation', 0)
        
        if shape_type == 'rectangle':
            # Use padded width and height which are stored in 'width'/'height' of the pad dictionary
            w = pad['width'] 
            h = pad['height']
            
            # 1. Translate point to the obstacle's center frame
            tx, ty = px - cx, py - cy
            
            # 2. Inverse Rotate the point (to get back to the unrotated local frame)
            rad = math.radians(-rot) # Inverse rotation
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a
            
            # 3. Check if the rotated point is within the unrotated AABB defined by W, H
            # The unrotated AABB is [-w/2, w/2] x [-h/2, h/2]
            if (-w/2 <= rx <= w/2) and (-h/2 <= ry <= h/2):
                return True # Collision
                
        elif shape_type == 'circle':
            # Use padded radius (radius is half of 'width')
            radius = pad['width'] / 2.0
            dist_sq = (px - cx)**2 + (py - cy)**2
            if dist_sq <= radius**2:
                return True # Collision
                
    return False


def spawn_goals_near_obstacles(num_goals, grid, pads, bounds, offset_dist):
    """
    Spawn goals (stations) close to the edges of obstacles.
    Handles rotated rectangles and circles based on the provided collision_data dict structure.
    
    Parameters:
    - pads: List of collision_data dictionaries.
    - offset_dist: How far from the obstacle edge to place the station (meters).
    """
    goals = []
    attempts = 0
    max_attempts = num_goals * 50
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # If no obstacles exist, fall back to random sampling
    if not pads:
        return spawn_goals(num_goals, grid, [], bounds[0], bounds[1])

    while len(goals) < num_goals and attempts < max_attempts:
        attempts += 1
        
        # 1. Pick a random obstacle
        pad = random.choice(pads)
        
        # Extract properties
        cx, cy = pad['original_center']
        rot = pad.get('cumulative_rotation', 0)
        shape_type = pad.get('type', 'rectangle')
        
        gx, gy = 0, 0
        
        if shape_type == 'rectangle':
            # Use the PADDED dimensions directly
            w_pad = pad['width']
            h_pad = pad['height']
            
            # Pick a random side in the LOCAL (unrotated) frame
            # 0: Top (y+), 1: Right (x+), 2: Bottom (y-), 3: Left (x-)
            side = random.randint(0, 3)
            
            local_x, local_y = 0, 0
            
            # Place the station OUTSIDE the padded obstacle by offset_dist
            if side == 0: # Top
                local_x = random.uniform(-w_pad/2, w_pad/2)
                local_y = h_pad/2 + offset_dist
            elif side == 1: # Right
                local_x = w_pad/2 + offset_dist
                local_y = random.uniform(-h_pad/2, h_pad/2)
            elif side == 2: # Bottom
                local_x = random.uniform(-w_pad/2, w_pad/2)
                local_y = -(h_pad/2 + offset_dist)
            elif side == 3: # Left
                local_x = -(w_pad/2 + offset_dist)
                local_y = random.uniform(-h_pad/2, h_pad/2)
                
            # Rotate local point to match obstacle orientation
            rx, ry = rotate_point(local_x, local_y, rot)
            
            # Translate to global position
            gx = cx + rx
            gy = cy + ry
            
        elif shape_type == 'circle':
            # Use padded radius for positioning
            radius = pad['width'] / 2.0
            angle = random.uniform(0, 2 * math.pi)
            dist = radius + offset_dist
            
            gx = cx + dist * math.cos(angle)
            gy = cy + dist * math.sin(angle)

        # 2. Validation
        # Check workspace bounds
        if not (x_min <= gx <= x_max and y_min <= gy <= y_max):
            continue

        # Check occupancy - make sure point is NOT inside any obstacle
        if check_collision_with_pads(gx, gy, pads):
            continue
            
        # Check distance from other generated goals (prevent stacking)
        if any(np.hypot(gx - ox, gy - oy) < 1.0 for (ox, oy) in goals):
            continue

        goals.append((gx, gy))

    # Fill remaining if failed
    if len(goals) < num_goals:
        print(f"    Warning: Could only place {len(goals)}/{num_goals} goals near obstacles. Filling randomly.")
        remaining = num_goals - len(goals)
        
        random_goals = spawn_goals(remaining, grid, pads, bounds[0], bounds[1])
        goals.extend(random_goals)

    return goals

# ═══════════════════════════════════════════════════════════════════════════════
# MODIFIED TRAJECTORY GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_paths_and_log(ax, stations, grid, pads, writer,
                           task_id, grid_id, mobile_center, agv_center,
                           station_coords_flat,
                           traj_id_start=0, frame_interval=0.25,
                           min_points=25, draw=True,
                           mask_prob=0.3, n_trajectories=20):   
    
    # Imports inside function to avoid circular deps if this was in a separate file
    from trajectory_utils import theta_star, smooth_path_with_beziers, interpolate_position_at_time
    
    traj_id = traj_id_start
    MIN_SPEED, MAX_SPEED = 0.8, 1.5
    SPEED_VARIABILITY = 0.1
    MIN_STANDING_FRAMES, MAX_STANDING_FRAMES = 5, 7
    STANDING_POSITION_NOISE = 0.02

    if not stations or len(stations) < 1:
        print("  ERROR: No stations available!")
        return traj_id

    print(f"  Generating {n_trajectories} trajectories from {len(stations)} stations...")
    
    # ------------------------------------------------------------------
    # Generate exactly n_trajectories successful paths
    # ------------------------------------------------------------------
    successful_trajectories = 0
    max_attempts = n_trajectories * 10  # Allow multiple retries per desired trajectory
    attempts = 0
    
    while successful_trajectories < n_trajectories and attempts < max_attempts:
        attempts += 1
        
        # Randomly pick start and goal station (can be the same)
        start_station = random.choice(stations)
        goal_station = random.choice(stations)
        
        sx, sy = start_station
        gx, gy = goal_station
        
        raw = theta_star((sx, sy), (gx, gy), grid, pads)
        if not raw:
            continue

        path = smooth_path_with_beziers(raw)
        if len(path) < 2:
            continue

        frame_id = 0
        trajectory_frames = []
        base_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)

        # --------------------- Distance & time calculation ---------------------
        distances = []
        cumulative_dist = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
            cumulative_dist.append(cumulative_dist[-1] + dist)

        cumulative_time = [0.0]
        for dist in distances:
            segment_speed = base_speed * np.random.uniform(1 - SPEED_VARIABILITY, 1 + SPEED_VARIABILITY)
            cumulative_time.append(cumulative_time[-1] + dist / segment_speed)

        total_time = cumulative_time[-1]

        # --------------------- Generate movement frames ---------------------
        current_time = 0.0
        while current_time <= total_time:
            x, y = interpolate_position_at_time(path, cumulative_time, current_time)
            trajectory_frames.append((x, y))
            current_time += frame_interval

        if trajectory_frames and trajectory_frames[-1] != path[-1]:
            trajectory_frames.append(path[-1])

        # --------------------- Add standing periods ---------------------
        if len(trajectory_frames) < min_points:
            frames_needed = min_points - len(trajectory_frames)
            start_standing = min(np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1), int(frames_needed * 0.6))
            end_standing = frames_needed - start_standing

            initial_frames = []
            start_x, start_y = trajectory_frames[0] if trajectory_frames else (sx, sy)
            for _ in range(start_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                initial_frames.append((start_x + nx, start_y + ny))

            final_frames = []
            end_x, end_y = trajectory_frames[-1] if trajectory_frames else (gx, gy)
            for _ in range(end_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                final_frames.append((end_x + nx, end_y + ny))

            trajectory_frames = initial_frames + trajectory_frames + final_frames

        # --------------------- Write to CSV ---------------------
        for x, y in trajectory_frames:
            pos_mask = 1 if np.random.random() > mask_prob else 0

            # Base row data
            row = [task_id, grid_id, traj_id, frame_id, 0, 1, x, y, 0, pos_mask]
            
            # Append ALL station coordinates to the end of the row
            row.extend(station_coords_flat)
            
            writer.writerow(row)
            frame_id += 1

        if draw and ax is not None:
            px, py = zip(*path)
            ax.plot(px, py, lw=1.2, alpha=0.7)

        successful_trajectories += 1
        traj_id += 1
    
    if successful_trajectories < n_trajectories:
        print(f"    ⚠️  Warning: Could only generate {successful_trajectories}/{n_trajectories} trajectories after {attempts} attempts")

    return traj_id

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    traj_id_counter = 0
    grid_id_counter = 0
    
    print(f"=== FULLY RANDOM MODE ===")
    print(f"Generating {NUM_RANDOM_MAPS} random maps with {STATIONS_PER_MAP} stations each")
    
    # 1. Prepare CSV Header dynamically based on STATIONS_PER_MAP
    csv_header = ["task_id", "grid_id", "traj_id", "frame_id", 
                  "agent_id", "agent_type", "x", "y", "z", "pos_mask"]
    
    # Add station columns: station_0_x, station_0_y, station_1_x, etc.
    for i in range(STATIONS_PER_MAP):
        csv_header.append(f"station_{i}_x")
        csv_header.append(f"station_{i}_y")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
        for map_num in range(NUM_RANDOM_MAPS):
            workspace_width_px = random.randint(*ORIG_GRID_WIDTH_RANGE)
            workspace_height_px = random.randint(*ORIG_GRID_HEIGHT_RANGE)
            
            if ORIGIN_MODE == "fixed":
                map_origin_x = FIXED_MAP_ORIGIN_X
                map_origin_y = FIXED_MAP_ORIGIN_Y
            elif ORIGIN_MODE == "random":
                map_origin_x = random.uniform(*X_MIN_RANGE)
                map_origin_y = random.uniform(*Y_MIN_RANGE)
            else:
                raise ValueError("Invalid ORIGIN_MODE")
            
            workspace = calculate_workspace_bounds(map_origin_x, map_origin_y, 
                                                  workspace_width_px, workspace_height_px)
            x_min, x_max = workspace['x_min'], workspace['x_max']
            y_min, y_max = workspace['y_min'], workspace['y_max']
            workspace_area = workspace['area']
            
            num_regular_obstacles = calculate_num_regular_obstacles(workspace_area)
            total_obstacles = num_regular_obstacles + NUM_SMALL_OBSTACLES + NUM_NARROW_OBSTACLES
            
            print(f"\n[Map {map_num + 1}] Area: {workspace_area:.2f}m², Obs: {total_obstacles}")
            
            ax = setup_plot(f"Map {map_num}", x_min, x_max, y_min, y_max) if DRAW else None
            
            for retry in range(10):
                try:
                    # Build Environment
                    obstacle_config = {
                        'regular': {'count': num_regular_obstacles, 'size_range': REGULAR_OBSTACLE_SIZE_RANGE},
                        'small': {'count': NUM_SMALL_OBSTACLES, 'size_range': SMALL_OBSTACLE_SIZE_RANGE, 'pad': 0.1},
                        'narrow': {'count': NUM_NARROW_OBSTACLES, 'width_range': NARROW_OBSTACLE_WIDTH_RANGE, 
                                   'height_range': NARROW_OBSTACLE_HEIGHT_RANGE, 'pad': 0.15}
                    }
                    
                    pads, (agv_c, mob_c), objects_no_pad = build_random_environment(
                        ax, x_min, x_max, y_min, y_max, total_obstacles, 
                        obstacle_config=obstacle_config, draw=DRAW)
                    
                    # NOTE: Grid is created here, but its is_free is not used in goal spawning logic.
                    grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)
                    
                    # --- STATION SPAWNING ---
                    # Spawn stations NEAR OBSTACLES (these serve as both starts and goals)
                    stations = spawn_goals_near_obstacles(
                        STATIONS_PER_MAP, 
                        grid, 
                        pads, 
                        ((x_min, x_max), (y_min, y_max)),
                        offset_dist=STATION_OFFSET_DISTANCE
                    )
                    
                    # Flatten Station Coordinates for CSV
                    # Format: [s0x, s0y, s1x, s1y, ... sNx, sNy]
                    station_coords_flat = []
                    for (sx, sy) in stations:
                        station_coords_flat.extend([sx, sy])
                    
                    # Ensure we have enough columns even if fewer stations spawned (pad with zeros)
                    while len(station_coords_flat) < STATIONS_PER_MAP * 2:
                        station_coords_flat.extend([0.0, 0.0])

                    # Generate PGM
                    task_id = f"random_{map_num}"
                    grid_id = f"random_{grid_id_counter}"
                    generate_occupancy_grid(grid, objects_no_pad, grid_id, 
                                          grid_size=GRID_SIZE, resolution=GRID_RESOLUTION,
                                          x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                                          map_origin_x=map_origin_x, map_origin_y=map_origin_y,
                                          output_dir=GRID_OUTPUT_DIR)
                    grid_id_counter += 1
                    
                    # Draw
                    if DRAW and ax:
                        if stations: ax.plot(*zip(*stations), 'r*', ms=8, label='stations')
                    
                    # Generate Trajectories
                    traj_id_counter = generate_paths_and_log(
                        ax, stations, grid, pads, writer,
                        task_id, grid_id, mob_c, agv_c,
                        station_coords_flat,
                        traj_id_counter, FRAME_INTERVAL, MIN_POINTS, DRAW,
                        mask_prob=MASK_PROBABILITY, n_trajectories=10)
                    
                    break # Success

                except NoFreeCellsError:
                    print(f"  Retry {retry+1}...")
                    if DRAW and ax: ax.cla()
            
            if DRAW and ax: finalize_plot(ax)

    print(f"\n✓ Generated {NUM_RANDOM_MAPS} maps, {traj_id_counter} trajectories")

if __name__ == '__main__':
    main()