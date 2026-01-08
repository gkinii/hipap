"""
trajectory_utils.py - Utility functions for trajectory generation
"""
import random
import heapq
import numpy as np
from utils import (
    spawn_goals,
    theta_star,
    smooth_path_with_beziers,
    create_padded,
    get_bounds,
)

# Constants for object sizes
MOBILE_WIDTH, MOBILE_HEIGHT = 1.75, 0.9
AGV_WIDTH, AGV_HEIGHT = 1.75, 3.0
TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT = 1.0, 0.5
BATTERY_WIDTH, BATTERY_HEIGHT = 1.2, 0.5
DYNAMIC_WIDTH, DYNAMIC_HEIGHT = 1.0, 1.0
DYNAMIC2_WIDTH, DYNAMIC2_HEIGHT = 2.0, 2.0

# Position and variation constants
DYDYNAMIC_CENTER = (-0.05, 1.25)
DYDYNAMIC2_CENTER = (0.5, 3.5)
TOOL_VAR_X, TOOL_VAR_Y = 1.5, 0.5
BATTERY_VAR_X = 1

# Rotation ranges
TOOL_ROT_RANGE = 90
AGV_ROT_RANGE = 10
MOBILE_ROT_RANGE = 10

"""
Updated build_environment function with origin offset support.
Add this to your trajectory_utils.py file, replacing the existing build_environment function.
"""

def build_environment(ax, agv_key, mobile_key, offset_x=0.0, offset_y=0.0, 
                     num_narrow_obstacles=7, draw=True):
    """
    Build structured environment with predefined objects.
    
    Parameters:
    - ax: Matplotlib axes for drawing
    - agv_key: AGV placement mode ("random" or "marriage_point_agv")
    - mobile_key: Mobile placement mode ("assembly" or other)
    - offset_x: X-axis offset to shift entire environment (default: 0.0)
    - offset_y: Y-axis offset to shift entire environment (default: 0.0)
    - num_narrow_obstacles: Number of narrow obstacles to place (default: 5)
    - draw: Whether to draw the objects
    
    Returns:
    - pads: List of padded collision data
    - centers: Tuple of (tool_center, agv_center, mobile_center, battery_center)
    - objects_no_pad: List of original collision data without padding
    """
    pads = []
    objects_no_pad = []

    def cp(ax, *args, **kwargs):
        return create_padded(ax, *args, draw=draw, **kwargs)
    
    # Helper to apply offset to positions
    def apply_offset(pos):
        return (pos[0] + offset_x, pos[1] + offset_y)

    # Dynamic objects (with offset applied)
    dynamic_center = apply_offset(DYDYNAMIC_CENTER)
    dynamic = cp(ax, dynamic_center, DYNAMIC_WIDTH, DYNAMIC_HEIGHT,
                 color="lightgrey", edgecolor="black",
                 jitter_x=(-0.5, 0.5), jitter_y=(-0.5, 0.5),
                 rotation_deg=random.uniform(-20, 20))
    pads.append(dynamic["collision_data"])
    objects_no_pad.append(dynamic["original_collision_data"])

    dynamic2_center = apply_offset(DYDYNAMIC2_CENTER)
    dynamic2 = cp(ax, dynamic2_center, DYNAMIC2_WIDTH, DYNAMIC2_HEIGHT,
                  color="lightgrey", edgecolor="black",
                  jitter_x=(-3, 2), jitter_y=(-0.5, 0.5),
                  rotation_deg=random.uniform(-20, 20))
    pads.append(dynamic2["collision_data"])
    objects_no_pad.append(dynamic2["original_collision_data"])

    # Desk (with offset applied)
    desk_center = apply_offset((2.8, -6.25))
    desk = cp(ax, desk_center, 2.5, 7.0, color="wheat", edgecolor="brown",
              jitter_x=(-1, 1), jitter_y=(-1, 4.5),
              rotation_deg=random.uniform(-10, 10))
    pads.append(desk["collision_data"])
    objects_no_pad.append(desk["original_collision_data"])

    # Battery (with offset applied)
    battery_center = apply_offset((-0.4, -3.7))
    battery = cp(ax, battery_center, BATTERY_WIDTH, BATTERY_HEIGHT,
                 color="lightblue", jitter_x=(0, BATTERY_VAR_X),
                 edgecolor="blue", jitter_y=(-1, 0),
                 rotation_deg=random.uniform(-20, 20))
    pads.append(battery["collision_data"])
    objects_no_pad.append(battery["original_collision_data"])

    # Tool (with offset applied)
    tool_center = apply_offset((-0.3, -1.35))
    tool = cp(ax, tool_center, TOOL_BASE_WIDTH, TOOL_BASE_HEIGHT,
              color="lightgreen", edgecolor="green",
              jitter_x=(0, TOOL_VAR_X),
              jitter_y=(-TOOL_VAR_Y, TOOL_VAR_Y * 0.1),
              rotation_deg=random.uniform(-TOOL_ROT_RANGE, TOOL_ROT_RANGE))
    pads.append(tool["collision_data"])
    objects_no_pad.append(tool["original_collision_data"])

    # AGV (with offset applied)
    agv_base_center = {"random": (-4.55, -2.75), 
                       "marriage_point_agv": (-4.55, -3.25)}[agv_key]
    agv_center = apply_offset(agv_base_center)
    agv_jitter_y = (-3, 5) if agv_key == "random" else (0., 1.)
    agv = cp(ax, agv_center, AGV_WIDTH, AGV_HEIGHT,
             color="lightcoral", edgecolor="red",
             jitter_y=agv_jitter_y, jitter_x=(0., 0.8),
             rotation_deg=random.uniform(-AGV_ROT_RANGE, AGV_ROT_RANGE))
    pads.append(agv["collision_data"])
    objects_no_pad.append(agv["original_collision_data"])

    # Mobile (with offset applied)
    if mobile_key == "assembly":
        # Position relative to battery (which already has offset applied)
        mob_pos = (battery["center"][0] - 0.6, battery["center"][1] - 1.395)
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 jitter_x=(-0.5, 0.3), jitter_y=(-0.2, 0.2),
                 rotation_deg=random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE))
    else:
        # Position relative to AGV (which already has offset applied)
        mob_pos = (agv["center"][0] + 0.33 * AGV_WIDTH, agv["center"][1])
        mob = cp(ax, mob_pos, MOBILE_WIDTH, MOBILE_HEIGHT,
                 color="plum", edgecolor="purple",
                 rotation_deg=90 + random.uniform(-MOBILE_ROT_RANGE, MOBILE_ROT_RANGE),
                 jitter_y=(-0.5, 0.5), jitter_x=(-0.1, 0.1))
    pads.append(mob["collision_data"])
    objects_no_pad.append(mob["original_collision_data"])

    # ═══════════════════════════════════════════════════════════════════════
    # NARROW OBSTACLES - Place anywhere except tool and battery areas
    # ═══════════════════════════════════════════════════════════════════════
    
    # Define narrow obstacle settings
    NARROW_WIDTH_RANGE = (0.1, 0.3)   # meters (thin width)
    NARROW_HEIGHT_RANGE = (1.5, 4.0)  # meters (tall height)
    NARROW_PAD = 0.15
    
    # Define exclusion zones (with offset already applied via centers)
    # Tool station exclusion zone (with jitter considered)
    tool_exclusion = {
        'x_min': tool["center"][0] - TOOL_VAR_X - 0.8,
        'x_max': tool["center"][0] + TOOL_VAR_X + 0.8,
        'y_min': tool["center"][1] - TOOL_VAR_Y - 0.8,
        'y_max': tool["center"][1] + TOOL_VAR_Y + 0.8
    }
    
    # Battery exclusion zone (with jitter considered)
    battery_exclusion = {
        'x_min': battery["center"][0] - BATTERY_VAR_X - 0.8,
        'x_max': battery["center"][0] + BATTERY_VAR_X + 0.8,
        'y_min': battery["center"][1] - 1.2,
        'y_max': battery["center"][1] + 1.2
    }
    
    # Define the overall spawn area (workspace bounds with offset)
    spawn_area = {
        'x_min': -6.4 + offset_x,
        'x_max': 6.4 + offset_x,
        'y_min': -6.4 + offset_y,
        'y_max': 6.4 + offset_y
    }
    
    # Helper function to check if point is in exclusion zone
    def in_exclusion_zone(x, y):
        # Check tool exclusion
        if (tool_exclusion['x_min'] <= x <= tool_exclusion['x_max'] and
            tool_exclusion['y_min'] <= y <= tool_exclusion['y_max']):
            return True
        # Check battery exclusion
        if (battery_exclusion['x_min'] <= x <= battery_exclusion['x_max'] and
            battery_exclusion['y_min'] <= y <= battery_exclusion['y_max']):
            return True
        return False
    
    # Color configurations for narrow obstacles
    narrow_colors = [
        {"color": "khaki", "edge": "olive"},
        {"color": "lightcyan", "edge": "teal"},
        {"color": "mistyrose", "edge": "maroon"},
    ]
    
    # Place narrow obstacles
    for i in range(num_narrow_obstacles):
        placed = False
        attempts = 0
        max_attempts = 100
        
        while not placed and attempts < max_attempts:
            # Random position within spawn area
            narrow_x = random.uniform(spawn_area['x_min'], spawn_area['x_max'])
            narrow_y = random.uniform(spawn_area['y_min'], spawn_area['y_max'])
            
            # Check if position is valid (not in exclusion zones)
            if not in_exclusion_zone(narrow_x, narrow_y):
                # Random narrow dimensions
                width = random.uniform(*NARROW_WIDTH_RANGE)
                height = random.uniform(*NARROW_HEIGHT_RANGE)
                rotation = random.uniform(-45, 45)
                
                colors = random.choice(narrow_colors)
                
                # Create narrow obstacle
                narrow = cp(ax, (narrow_x, narrow_y), width, height,
                           color=colors["color"], edgecolor=colors["edge"],
                           jitter_x=(0, 0), jitter_y=(0, 0),
                           rotation_deg=rotation, pad=NARROW_PAD)
                
                pads.append(narrow["collision_data"])
                objects_no_pad.append(narrow["original_collision_data"])
                placed = True
            
            attempts += 1
        
        if not placed:
            print(f"  Warning: Could not place narrow obstacle {i+1}/{num_narrow_obstacles}")

    centers = tool["center"], agv["center"], mob["center"], battery["center"], dynamic["center"], dynamic2["center"]

    return pads, centers, objects_no_pad

def build_random_environment(ax, x_min, x_max, y_min, y_max, num_obstacles=5, 
                            obstacle_config=None, draw=True):
    """
    Build completely random environment with obstacles at random positions.
    
    Parameters:
    - ax: Matplotlib axes for drawing
    - x_min, x_max, y_min, y_max: Workspace bounds
    - num_obstacles: Total number of obstacles (used if obstacle_config is None)
    - obstacle_config: Optional dict specifying obstacle groups with different sizes
                      Example: {
                          'regular': {
                              'count': 14, 
                              'size_range': (1.0, 3.0)
                          },
                          'small': {
                              'count': 6, 
                              'size_range': (0.3, 0.8), 
                              'pad': 0.1
                          },
                          'narrow': {
                              'count': 20,
                              'width_range': (0.1, 0.3),
                              'height_range': (1.5, 3.5),
                              'pad': 0.1
                          }
                      }
    - draw: Whether to draw the obstacles
    
    Returns:
    - pads: List of padded collision data
    - centers: Tuple of (agv_center, mobile_center)
    - objects_no_pad: List of original collision data without padding
    """
    pads = []
    objects_no_pad = []

    def cp(ax, *args, **kwargs):
        return create_padded(ax, *args, draw=draw, **kwargs)

    # Smaller workspace margins to allow objects near edges
    margin = 0.
    work_x_min, work_x_max = x_min + margin, x_max - margin
    work_y_min, work_y_max = y_min + margin, y_max - margin

    # Color configurations for visual variety
    color_configs = [
        {"color": "lightgrey", "edge": "black"},
        {"color": "wheat", "edge": "brown"},
        {"color": "lightblue", "edge": "blue"},
        {"color": "lightgreen", "edge": "green"},
        {"color": "lightcoral", "edge": "red"},
        {"color": "lightyellow", "edge": "orange"},
        {"color": "lavender", "edge": "purple"},
        {"color": "peachpuff", "edge": "darkorange"},
    ]

    # Track placed object centers for better distribution
    placed_centers = []
    min_spacing = 3  # Minimum distance between object centers

    # Helper function to place obstacles with separate width/height ranges
    def place_obstacle(width_range, height_range, pad_value=None, attempts=50):
        """
        Try to place an obstacle with specified width and height ranges.
        
        Parameters:
        - width_range: Tuple (min_width, max_width)
        - height_range: Tuple (min_height, max_height)
        - pad_value: Optional collision padding
        - attempts: Number of placement attempts
        """
        for attempt in range(attempts):
            # Random position
            center_x = random.uniform(work_x_min, work_x_max)
            center_y = random.uniform(work_y_min, work_y_max)
            
            # Check spacing from existing objects
            too_close = False
            for prev_x, prev_y in placed_centers:
                dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                if dist < min_spacing:
                    too_close = True
                    break
            
            if not too_close or attempt == attempts - 1:
                # Accept this position
                width = random.uniform(width_range[0], width_range[1])
                height = random.uniform(height_range[0], height_range[1])
                rotation = random.uniform(-45, 45)
                
                colors = random.choice(color_configs)
                
                # Create obstacle with specified padding if provided
                if pad_value is not None:
                    obstacle = create_padded(ax, (center_x, center_y), width, height,
                                           color=colors["color"], edgecolor=colors["edge"],
                                           jitter_x=(0, 0), jitter_y=(0, 0), 
                                           rotation_deg=rotation, pad=pad_value, draw=draw)
                else:
                    obstacle = cp(ax, (center_x, center_y), width, height,
                                 color=colors["color"], edgecolor=colors["edge"],
                                 jitter_x=(0, 0), jitter_y=(0, 0), rotation_deg=rotation)
                
                pads.append(obstacle["collision_data"])
                objects_no_pad.append(obstacle["original_collision_data"])
                placed_centers.append((center_x, center_y))
                return True
        return False

    # Place obstacles based on configuration
    if obstacle_config is not None:
        # Use mixed obstacle sizes
        for group_name, group_params in obstacle_config.items():
            count = group_params['count']
            pad_value = group_params.get('pad', None)
            
            # Check if width_range and height_range are specified separately
            if 'width_range' in group_params and 'height_range' in group_params:
                # Separate width and height ranges (for narrow obstacles)
                width_range = group_params['width_range']
                height_range = group_params['height_range']
            elif 'size_range' in group_params:
                # Same range for both width and height (for regular/small obstacles)
                width_range = group_params['size_range']
                height_range = group_params['size_range']
            else:
                raise ValueError(f"Obstacle group '{group_name}' must specify either "
                               "'size_range' or both 'width_range' and 'height_range'")
            
            for i in range(count):
                placed = place_obstacle(width_range, height_range, pad_value=pad_value)
                if not placed:
                    print(f"  Warning: Could not place {group_name} obstacle {i+1}/{count}")
    else:
        # Backward compatibility: use default size ranges
        print('WENT HERE')
        default_size_range = (0.7, 2)
        
        for i in range(num_obstacles):
            placed = place_obstacle(default_size_range, default_size_range)
            if not placed:
                print(f"  Warning: Could not place obstacle {i+1}/{num_obstacles}")

    # Random AGV position - allow anywhere in workspace
    for attempt in range(50):
        agv_x = random.uniform(work_x_min, work_x_max)
        agv_y = random.uniform(work_y_min, work_y_max)
        
        # Check spacing from obstacles
        too_close = False
        for prev_x, prev_y in placed_centers:
            dist = np.sqrt((agv_x - prev_x)**2 + (agv_y - prev_y)**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if not too_close or attempt == 49:
            break
    
    agv = cp(ax, (agv_x, agv_y), AGV_WIDTH, AGV_HEIGHT,
             color="lightcoral", edgecolor="red",
             jitter_x=(0, 0), jitter_y=(0, 0),
             rotation_deg=random.uniform(-30, 30))
    pads.append(agv["collision_data"])
    objects_no_pad.append(agv["original_collision_data"])
    placed_centers.append((agv_x, agv_y))

    # Random Mobile position - allow anywhere in workspace
    for attempt in range(50):
        mobile_x = random.uniform(work_x_min, work_x_max)
        mobile_y = random.uniform(work_y_min, work_y_max)
        
        # Check spacing from existing objects
        too_close = False
        for prev_x, prev_y in placed_centers:
            dist = np.sqrt((mobile_x - prev_x)**2 + (mobile_y - prev_y)**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if not too_close or attempt == 49:
            break
    
    mobile = cp(ax, (mobile_x, mobile_y), MOBILE_WIDTH, MOBILE_HEIGHT,
                color="plum", edgecolor="purple",
                jitter_x=(0, 0), jitter_y=(0, 0),
                rotation_deg=random.uniform(-30, 30))
    pads.append(mobile["collision_data"])
    objects_no_pad.append(mobile["original_collision_data"])

    centers = (agv["center"], mobile["center"])
    return pads, centers, objects_no_pad

def interpolate_position_at_time(path, cumulative_time, target_time):
    """Interpolate position along path at a specific time"""
    if target_time <= 0:
        return path[0]
    if target_time >= cumulative_time[-1]:
        return path[-1]

    for i in range(1, len(cumulative_time)):
        if cumulative_time[i] >= target_time:
            t0, t1 = cumulative_time[i-1], cumulative_time[i]
            alpha = (target_time - t0) / (t1 - t0)
            x = path[i-1][0] + alpha * (path[i][0] - path[i-1][0])
            y = path[i-1][1] + alpha * (path[i][1] - path[i-1][1])
            return (x, y)

    return path[-1]

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: STATION SPAWNING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def spawn_goals_near_obstacles(num_goals, grid, pads, bounds, offset_dist=0.6):
    """
    Spawn goals (stations) close to the edges of obstacles.
    
    Parameters:
    - pads: List of obstacle collision data (assumed [min_x, min_y, max_x, max_y])
    - offset_dist: How far from the obstacle edge to place the station (meters)
    """
    goals = []
    attempts = 0
    max_attempts = num_goals * 50
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # If no obstacles exist, fall back to random sampling
    if not pads:
        return spawn_goals(num_goals, grid, pads, bounds[0], bounds[1])

    while len(goals) < num_goals and attempts < max_attempts:
        attempts += 1
        
        # 1. Pick a random obstacle
        pad = random.choice(pads)
        # Assuming pad is [min_x, min_y, max_x, max_y]
        p_xmin, p_ymin, p_xmax, p_ymax = pad
        
        # 2. Pick a random side (0:Top, 1:Right, 2:Bottom, 3:Left)
        side = random.randint(0, 3)
        
        gx, gy = 0, 0
        
        if side == 0: # Top (y_max + offset)
            gx = random.uniform(p_xmin, p_xmax)
            gy = p_ymax + offset_dist
        elif side == 1: # Right (x_max + offset)
            gx = p_xmax + offset_dist
            gy = random.uniform(p_ymin, p_ymax)
        elif side == 2: # Bottom (y_min - offset)
            gx = random.uniform(p_xmin, p_xmax)
            gy = p_ymin - offset_dist
        elif side == 3: # Left (x_min - offset)
            gx = p_xmin - offset_dist
            gy = random.uniform(p_ymin, p_ymax)

        # 3. Validation
        # Check workspace bounds
        if not (x_min <= gx <= x_max and y_min <= gy <= y_max):
            continue

        # Check occupancy grid (is_free)
        if not grid.is_free((gx, gy)):
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

def generate_paths_and_log(ax, starts, stations, grid, pads, writer,
                           task_id, grid_id, mobile_center, agv_center,
                           station_coords_flat, # <--- NEW ARGUMENT
                           traj_id_start=0, frame_interval=0.25,
                           min_points=25, draw=True, include_robots=False,
                           tool_center=None, battery_center=None,
                           mask_prob=0.3, n_trajectories=20):   
    
    # Imports inside function to avoid circular deps if this was in a separate file
    from trajectory_utils import theta_star, smooth_path_with_beziers, interpolate_position_at_time
    
    traj_id = traj_id_start
    MIN_SPEED, MAX_SPEED = 0.8, 1.5
    SPEED_VARIABILITY = 0.1
    MIN_STANDING_FRAMES, MAX_STANDING_FRAMES = 5, 7
    STANDING_POSITION_NOISE = 0.02

    # ------------------------------------------------------------------
    # Build list of all possible (start, station) pairs
    # This allows Many-to-Many relationships randomly
    # ------------------------------------------------------------------
    all_pairs = [(s, g) for s in starts for g in stations]
    if not all_pairs:
        return traj_id

    # Sample random subset (Random start -> Random station)
    n_wanted = min(n_trajectories, len(all_pairs))
    selected_pairs = random.sample(all_pairs, n_wanted)

    for (sx, sy), (gx, gy) in selected_pairs:
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

        traj_id += 1

    return traj_id


def generate_task_paths_and_log(ax, starts, goals, grid, pads, writer,
                           task_id, grid_id, mobile_center, agv_center,
                           traj_id_start=0, frame_interval=0.25,
                           min_points=25, draw=True, include_robots=False,
                           tool_center=None, battery_center=None,
                           dyn_center=None, dyn2_center=None,
                           mask_prob=0.0,
                           n_trajectories=20):
    """
    Generate paths with realistic speed modeling and standing periods.
    Now generates only a random subset of start→goal combinations.

    Parameters:
    - n_trajectories : int
        How many random human trajectories to generate (default 50).
        Will generate fewer if there are not enough valid paths.
    - include_robots : bool
        If True, also log AGV and mobile robot positions (structured maps)
    - tool_center / battery_center / dyn_center / dyn2_center : (x, y) or None
        Station positions for CSV logging
    - mask_prob : float [0.0 – 1.0]
        Probability of masking a position
    """
    traj_id = traj_id_start

    MIN_SPEED, MAX_SPEED = 0.8, 1.5
    SPEED_VARIABILITY = 0.1
    MIN_STANDING_FRAMES, MAX_STANDING_FRAMES = 5, 7
    STANDING_POSITION_NOISE = 0.02

    # Extract station coordinates (default to 0,0 if not provided)
    station_1_x = tool_center[0] if tool_center else 0.0
    station_1_y = tool_center[1] if tool_center else 0.0
    station_2_x = battery_center[0] if battery_center else 0.0
    station_2_y = battery_center[1] if battery_center else 0.0
    station_3_x = dyn_center[0] if dyn_center else 0.0
    station_3_y = dyn_center[1] if dyn_center else 0.0
    station_4_x = dyn2_center[0] if dyn2_center else 0.0
    station_4_y = dyn2_center[1] if dyn2_center else 0.0
    station_5_x = agv_center[0] if agv_center else 0.0
    station_5_y = agv_center[1] if agv_center else 0.0

    # ------------------------------------------------------------------
    # Build list of all possible (start, goal) pairs
    # ------------------------------------------------------------------
    all_pairs = [(s, g) for s in starts for g in goals]
    if not all_pairs:
        return traj_id

    # Sample desired number of random unique pairs
    n_wanted = min(n_trajectories, len(all_pairs))
    selected_pairs = random.sample(all_pairs, n_wanted)

    # ------------------------------------------------------------------
    # Process each selected (start, goal) pair
    # ------------------------------------------------------------------
    for (sx, sy), (gx, gy) in selected_pairs:
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
            segment_speed = base_speed * np.random.uniform(1 - SPEED_VARIABILITY,
                                                          1 + SPEED_VARIABILITY)
            cumulative_time.append(cumulative_time[-1] + dist / segment_speed)

        total_time = cumulative_time[-1]

        # --------------------- Generate movement frames ---------------------
        current_time = 0.0
        while current_time <= total_time:
            x, y = interpolate_position_at_time(path, cumulative_time, current_time)
            trajectory_frames.append((x, y))
            current_time += frame_interval

        # Make sure goal is always included
        if trajectory_frames and trajectory_frames[-1] != path[-1]:
            trajectory_frames.append(path[-1])

        # --------------------- Add standing periods if needed ---------------------
        if len(trajectory_frames) < min_points:
            frames_needed = min_points - len(trajectory_frames)

            start_standing = min(np.random.randint(MIN_STANDING_FRAMES,
                                                   MAX_STANDING_FRAMES + 1),
                                 int(frames_needed * 0.6))
            end_standing = frames_needed - start_standing

            # Initial standing at start point
            initial_frames = []
            start_x, start_y = trajectory_frames[0] if trajectory_frames else (sx, sy)
            for _ in range(start_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                initial_frames.append((start_x + nx, start_y + ny))

            # Final standing at goal point
            final_frames = []
            end_x, end_y = trajectory_frames[-1] if trajectory_frames else (gx, gy)
            for _ in range(end_standing):
                nx = np.random.normal(0, STANDING_POSITION_NOISE)
                ny = np.random.normal(0, STANDING_POSITION_NOISE)
                final_frames.append((end_x + nx, end_y + ny))

            trajectory_frames = initial_frames + trajectory_frames + final_frames

        # --------------------- Write to CSV (with optional masking) ---------------------
        for x, y in trajectory_frames:
            pos_mask = 1 if np.random.random() > mask_prob else 0

            # Human trajectory (agent_type=0, is_human=1) with station positions
            writer.writerow([task_id, grid_id, traj_id, frame_id, 0, 1, x, y, 0, pos_mask,
                           station_1_x, station_1_y, station_2_x, station_2_y,
                           station_3_x, station_3_y, station_4_x, station_4_y,
                           station_5_x, station_5_y])

            # Optional static robots (only in structured environments)
            if include_robots:
                writer.writerow([task_id, grid_id, traj_id, frame_id, 1, 0,
                                agv_center[0], agv_center[1], 0, 1,
                                station_1_x, station_1_y, station_2_x, station_2_y,
                                station_3_x, station_3_y, station_4_x, station_4_y,
                                station_5_x, station_5_y])
                writer.writerow([task_id, grid_id, traj_id, frame_id, 2, 0,
                                mobile_center[0], mobile_center[1], 0, 1,
                                station_1_x, station_1_y, station_2_x, station_2_y,
                                station_3_x, station_3_y, station_4_x, station_4_y,
                                station_5_x, station_5_y])

            frame_id += 1

        # --------------------- Optional visualisation ---------------------
        if draw and ax is not None:
            px, py = zip(*path)
            ax.plot(px, py, lw=1.2, alpha=0.7)

        traj_id += 1

    return traj_id

def generate_large_trajectory(ax, grid, pads, writer, task_id, grid_id,
                              mobile_center, agv_center, traj_id,
                              num_waypoints=6, frame_interval=0.4, draw=True,
                              include_robots=True):
    """
    Generate large trajectory visiting multiple waypoints.
    
    Parameters:
    - include_robots: If True, also log AGV and mobile robot positions (for structured maps)
                     If False, only log human trajectories
    """
    MIN_SPEED, MAX_SPEED = 0.8, 1.5
    SPEED_VARIABILITY = 0.05
    MIN_STANDING_FRAMES, MAX_STANDING_FRAMES = 5, 10
    STANDING_POSITION_NOISE = 0.02

    # Generate waypoints from different areas
    waypoints = []
    tool_area = get_bounds((-0.3, -1.35), 1.05, 0.55, 0.0)
    assembly_area = get_bounds((-0.4, -3.15), 1.2, 0.5, 0.05)
    random_area = get_bounds((0.0, 0.0), 10.0, 12.0, 0)

    for _ in range(2):
        waypoints.extend(spawn_goals(1, grid, pads, 
                                     tool_area["x_bounds"], tool_area["y_bounds"]))
    
    for _ in range(2):
        waypoints.extend(spawn_goals(1, grid, pads,
                                     assembly_area["x_bounds"], assembly_area["y_bounds"]))
    
    remaining = num_waypoints - len(waypoints)
    for _ in range(remaining):
        waypoints.extend(spawn_goals(1, grid, pads,
                                     random_area["x_bounds"], random_area["y_bounds"]))

    random.shuffle(waypoints)
    print(f"  Generated {len(waypoints)} waypoints for trajectory {traj_id}")

    # Build complete trajectory
    all_frames = []
    frame_id = 0

    for i in range(len(waypoints) - 1):
        raw = theta_star(waypoints[i], waypoints[i + 1], grid, pads)
        if not raw:
            continue
        path = smooth_path_with_beziers(raw)
        if len(path) < 2:
            continue

        # Calculate times
        distances = []
        cumulative_dist = [0]
        for j in range(1, len(path)):
            dx = path[j][0] - path[j-1][0]
            dy = path[j][1] - path[j-1][1]
            distances.append(np.sqrt(dx**2 + dy**2))
            cumulative_dist.append(cumulative_dist[-1] + distances[-1])

        base_speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
        cumulative_time = [0]
        for dist in distances:
            segment_speed = base_speed * np.random.uniform(
                1 - SPEED_VARIABILITY, 1 + SPEED_VARIABILITY)
            cumulative_time.append(cumulative_time[-1] + dist / segment_speed)

        # Generate frames
        segment_frames = []
        current_time = 0
        while current_time <= cumulative_time[-1]:
            x, y = interpolate_position_at_time(path, cumulative_time, current_time)
            segment_frames.append((x, y))
            current_time += frame_interval

        if current_time - frame_interval < cumulative_time[-1]:
            segment_frames.append(path[-1])

        # Add standing at waypoint
        if i < len(waypoints) - 2:
            standing_count = np.random.randint(MIN_STANDING_FRAMES, MAX_STANDING_FRAMES + 1)
            end_x, end_y = segment_frames[-1] if segment_frames else waypoints[i + 1]
            for _ in range(standing_count):
                noise_x = np.random.normal(0, STANDING_POSITION_NOISE)
                noise_y = np.random.normal(0, STANDING_POSITION_NOISE)
                segment_frames.append((end_x + noise_x, end_y + noise_y))

        all_frames.extend(segment_frames)

        if draw and ax is not None:
            px, py = zip(*path)
            ax.plot(px, py, lw=1.2, alpha=0.7)

    # Write to CSV
    for x, y in all_frames:
        # Always write human agent
        writer.writerow([task_id, grid_id, traj_id, frame_id, 0, 1, x, y, 0])
        
        # Optionally write robot agents
        if include_robots:
            writer.writerow([task_id, grid_id, traj_id, frame_id, 1, 0,
                            agv_center[0], agv_center[1], 0])
            writer.writerow([task_id, grid_id, traj_id, frame_id, 2, 0,
                            mobile_center[0], mobile_center[1], 0])
        frame_id += 1

    # Draw waypoints
    if draw and ax is not None and waypoints:
        wx, wy = zip(*waypoints)
        ax.plot(wx, wy, 'ro', ms=8, label='waypoints')
        for i in range(len(waypoints) - 1):
            ax.annotate('', xy=waypoints[i+1], xytext=waypoints[i],
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))

    print(f"  Trajectory {traj_id}: {len(all_frames)} frames")
    return traj_id + 1

