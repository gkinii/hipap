"""
generate_task_library_trajectories.py - Generate task library trajectories
"""
import json
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Grid, spawn_goals, get_bounds, NoFreeCellsError
from generate_grid import generate_occupancy_grid
from trajectory_utils import (
    build_environment,
    generate_task_paths_and_log,
    AGV_WIDTH, AGV_HEIGHT
)
from utils import (
    partition_goals
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Generation Settings
NUM_PARTS = 400
GRID_OUTPUT_DIR = "256_task" 
CSV_PATH = f"{GRID_OUTPUT_DIR}/paths.csv"
TASK_LIBRARY_PATH = "task_library.json"

# General Settings
DRAW = False
FRAME_INTERVAL = 0.4
MIN_POINTS = 20
MASK_PROB = 0.3

# Grid Settings
GRID_SIZE = 320  # Total grid size in pixels (with padding)
GRID_RESOLUTION = 0.05  # Meters per pixel

# Workspace Settings (visible area before padding)
WORKSPACE_SIZE_RANGE = (240, 320)  # pixels - determines visible workspace area

# Map Origin Range (bottom-left corner position in world coordinates)
MAP_ORIGIN_X_RANGE = (-8, -8)  # meters - set to same value for fixed origin
MAP_ORIGIN_Y_RANGE = (-8, -8)  # meters - set to same value for fixed origin

# Small Random Obstacle Settings
NUM_SMALL_OBSTACLES = 30
SMALL_OBSTACLE_SIZE_RANGE = (0.15, 0.25)  # meters
SMALL_OBSTACLE_PAD = 0.1

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_workspace_bounds(map_origin_x, map_origin_y, 
                               workspace_width_px, workspace_height_px,
                               grid_size=GRID_SIZE, grid_resolution=GRID_RESOLUTION):
    """
    Calculate workspace bounds from map origin and workspace size.
    
    Parameters:
    - map_origin_x, map_origin_y: Bottom-left corner of the map in world coordinates
    - workspace_width_px, workspace_height_px: Visible workspace size in pixels
    - grid_size: Total grid size in pixels (with padding)
    - grid_resolution: Meters per pixel
    
    Returns:
    - Dictionary with workspace bounds and metadata
    """
    # Calculate padding
    pad_x_pixels = grid_size - workspace_width_px
    pad_y_pixels = grid_size - workspace_height_px
    
    # Workspace starts after left/bottom padding
    pad_left = pad_x_pixels // 2
    pad_bottom = pad_y_pixels // 2
    
    # Calculate workspace bounds in world coordinates
    x_min = map_origin_x + pad_left * grid_resolution
    y_min = map_origin_y + pad_bottom * grid_resolution
    
    workspace_width_m = workspace_width_px * grid_resolution
    workspace_height_m = workspace_height_px * grid_resolution
    
    x_max = x_min + workspace_width_m
    y_max = y_min + workspace_height_m
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': workspace_width_m,
        'height': workspace_height_m,
        'area': workspace_width_m * workspace_height_m,
        'workspace_width_px': workspace_width_px,
        'workspace_height_px': workspace_height_px,
        'pad_x_pixels': pad_x_pixels,
        'pad_y_pixels': pad_y_pixels,
        'map_origin_x': map_origin_x,
        'map_origin_y': map_origin_y
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SMALL OBSTACLE PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def place_small_obstacles(ax, x_min, x_max, y_min, y_max, existing_centers, 
                         num_obstacles=NUM_SMALL_OBSTACLES, draw=True):
    """
    Place small random obstacles in the workspace.
    Small obstacles can overlap with structured objects.
    
    Parameters:
    - ax: Matplotlib axes for drawing
    - x_min, x_max, y_min, y_max: Workspace bounds
    - existing_centers: List of (x, y) centers of existing objects (unused, kept for compatibility)
    - num_obstacles: Number of small obstacles to place
    - draw: Whether to draw the obstacles
    
    Returns:
    - pads: List of padded collision data
    - objects_no_pad: List of original collision data without padding
    """
    from utils import create_padded

    pads = []
    objects_no_pad = []
    
    # Smaller workspace margins for small objects
    margin = 0.3
    work_x_min, work_x_max = x_min + margin, x_max - margin
    work_y_min, work_y_max = y_min + margin, y_max - margin
    
    # Color configurations for visual variety
    color_configs = [
        {"color": "lightgrey", "edge": "darkgrey"},
        {"color": "lightblue", "edge": "blue"},
        {"color": "lightgreen", "edge": "green"},
        {"color": "lightyellow", "edge": "orange"},
        {"color": "lavender", "edge": "purple"},
        {"color": "peachpuff", "edge": "darkorange"},
    ]
    
    # Place obstacles without collision checking (overlaps allowed)
    for i in range(num_obstacles):
        # Random position
        center_x = random.uniform(work_x_min, work_x_max)
        center_y = random.uniform(work_y_min, work_y_max)
        
        # Random size and rotation
        width = random.uniform(*SMALL_OBSTACLE_SIZE_RANGE)
        height = random.uniform(*SMALL_OBSTACLE_SIZE_RANGE)
        rotation = random.uniform(-45, 45)
        
        colors = random.choice(color_configs)
        
        # Create obstacle with small padding
        obstacle = create_padded(ax, (center_x, center_y), width, height,
                               color=colors["color"], edgecolor=colors["edge"],
                               jitter_x=(0, 0), jitter_y=(0, 0), 
                               rotation_deg=rotation, pad=SMALL_OBSTACLE_PAD, 
                               draw=draw)
        
        pads.append(obstacle["collision_data"])
        objects_no_pad.append(obstacle["original_collision_data"])
    
    return pads, objects_no_pad

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def setup_plot(title, x_min, x_max, y_min, y_max):
    """Create and configure a plot"""
    fig, ax = plt.subplots(figsize=(9, 7))
    setup_axes(ax, title, x_min, x_max, y_min, y_max)
    return ax

def setup_axes(ax, title, x_min, x_max, y_min, y_max):
    """Configure plot axes"""
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def finalize_plot(ax):
    """Finalize and show plot"""
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Generate task library trajectories"""
    # Load task library
    tasks = json.load(open(Path(TASK_LIBRARY_PATH)))
    
    traj_id_counter = 0
    grid_id_counter = 0
    
    print(f"=== TASK LIBRARY MODE ===")
    print(f"Processing {len(tasks)} tasks with {NUM_PARTS} parts each")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} @ {GRID_RESOLUTION}m/pixel")
    print(f"Workspace size range: {WORKSPACE_SIZE_RANGE[0]}-{WORKSPACE_SIZE_RANGE[1]} pixels")
    print(f"Map origin range: x ∈ {MAP_ORIGIN_X_RANGE}, y ∈ {MAP_ORIGIN_Y_RANGE}")
    print(f"Small obstacles: {NUM_SMALL_OBSTACLES} per map\n")
    
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id",
                "agent_id", "agent_type", "x", "y", "z", "pos_mask",
                "station_1_x", "station_1_y", "station_2_x", "station_2_y",
                "station_3_x", "station_3_y", "station_4_x", "station_4_y",
                "station_5_x", "station_5_y"])
        
        for task in tasks:
            start_parts = partition_goals(task["start_pos"], NUM_PARTS)
            goal_parts = partition_goals(task["goal_pos"], NUM_PARTS)
            
            for stage in range(NUM_PARTS):
                # Randomize workspace size (determines visible area)
                workspace_width_px = random.randint(*WORKSPACE_SIZE_RANGE)
                workspace_height_px = random.randint(*WORKSPACE_SIZE_RANGE)
                
                # Randomize map origin (bottom-left corner)
                map_origin_x = random.uniform(*MAP_ORIGIN_X_RANGE)
                map_origin_y = random.uniform(*MAP_ORIGIN_Y_RANGE)
                
                # Calculate workspace bounds from origin and size
                workspace = calculate_workspace_bounds(map_origin_x, map_origin_y,
                                                      workspace_width_px, workspace_height_px)
                
                x_min = workspace['x_min']
                x_max = workspace['x_max']
                y_min = workspace['y_min']
                y_max = workspace['y_max']
                workspace_area = workspace['area']
                
                # For build_environment, calculate offset from default position
                # (structured objects are defined relative to origin (0,0))
                offset_x = (x_min + x_max) / 2  # Center of workspace
                offset_y = (y_min + y_max) / 2
                
                print(f"\n[Task {task['task_id']}-{stage+1}] Workspace: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
                print(f"  Area: {workspace_area:.2f}m² ({workspace_width_px}×{workspace_height_px}px)")
                print(f"  Map origin: ({map_origin_x:.2f}, {map_origin_y:.2f})")
                print(f"  Padding: {workspace['pad_x_pixels']}×{workspace['pad_y_pixels']}px")
                
                ax = setup_plot(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: "
                               f"{task['title']}", x_min, x_max, y_min, y_max) if DRAW else None
                
                while True:
                    try:
                        # Build structured environment with offset applied
                        pads, (tool_c, agv_c, mob_c, batt_c, dyn_c, dyn2_c), objects_no_pad = \
                            build_environment(ax, task["agv_pos"], task["mobile_pos"], 
                                            offset_x=offset_x, offset_y=offset_y, draw=DRAW)
                        
                        # Collect centers of existing structured objects
                        existing_centers = [tool_c, agv_c, mob_c, batt_c]
                        
                        # Add small random obstacles
                        small_pads, small_objects_no_pad = place_small_obstacles(
                            ax, x_min, x_max, y_min, y_max, existing_centers, 
                            num_obstacles=NUM_SMALL_OBSTACLES, draw=DRAW)
                        
                        # Combine structured and small obstacles
                        pads.extend(small_pads)
                        objects_no_pad.extend(small_objects_no_pad)
                        
                        grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)
                        
                        grid_id = f"{task['task_id']}_{stage+1}"
                        generate_occupancy_grid(grid, objects_no_pad, grid_id, 
                                              grid_size=GRID_SIZE,
                                              resolution=GRID_RESOLUTION,
                                              x_min=x_min, y_min=y_min,
                                              x_max=x_max, y_max=y_max,
                                              map_origin_x=map_origin_x,
                                              map_origin_y=map_origin_y,
                                              output_dir=GRID_OUTPUT_DIR)
                        grid_id_counter += 1
                        
                        # Define goal areas
                        workspace_center_x = (x_min + x_max) / 2
                        workspace_center_y = (y_min + y_max) / 2
                        workspace_width = x_max - x_min
                        workspace_height = y_max - y_min
                        
                        GOAL_AREA = {
                            "random": get_bounds((workspace_center_x, workspace_center_y), 
                                               workspace_width * 0.8, workspace_height * 0.8, 0),
                            "tool_station": get_bounds(tool_c, 1.05, 0.55, 0.0),
                            "battery_assembly": get_bounds((batt_c[0], batt_c[1]+0.45),
                                                          0.6, 0.5, 0.05),
                            # "agv_ph1": get_bounds((agv_c[0]+2, agv_c[1]+3.5), 1.5, 1.5, 0.5),
                            "agv_ph1": get_bounds((dyn_c[0], dyn_c[1]), 1, 1, 0.5),
                            "agv_ph2": get_bounds(agv_c, AGV_WIDTH+0.2, AGV_HEIGHT+0.2, 0.0),
                            "agv_ph3": get_bounds((dyn2_c[0], dyn2_c[1]), 1.5, 1.5, 0.5),

                        }
                        
                        # Spawn starts and goals
                        starts, goals = [], []
                        for goal_type, n in start_parts[stage]:
                            if n > 0:
                                area = GOAL_AREA[goal_type]
                                min_dist = 0.2 if goal_type != "random" else None
                                starts.extend(spawn_goals(n, grid, pads,
                                                        area["x_bounds"], area["y_bounds"],
                                                        min_distance=min_dist))

                        for goal_type, n in goal_parts[stage]:
                            if n > 0:
                                area = GOAL_AREA[goal_type]
                                min_dist = 0.2 if goal_type != "random" else None
                                goals.extend(spawn_goals(n, grid, pads,
                                                        area["x_bounds"], area["y_bounds"],
                                                        min_distance=min_dist))
                        
                        print(f"  {len(starts)}×{len(goals)} trajectories, "
                              f"{len(objects_no_pad)} total objects")
                        break
                        
                    except NoFreeCellsError as e:
                        print(f"  {e} → rebuilding environment and retrying...")
                        if DRAW and ax:
                            ax.cla()
                            setup_axes(ax, f"Task {task['task_id']} - Part {stage+1}",
                                     x_min, x_max, y_min, y_max)
                
                if DRAW and ax:
                    if starts:
                        ax.plot(*zip(*starts), 'bo', ms=4, label='start')
                    if goals:
                        ax.plot(*zip(*goals), 'rx', ms=5, label='goal')
                
                traj_id_counter = generate_task_paths_and_log(
                    ax, starts, goals, grid, pads, writer,
                    task['task_id'], grid_id, mob_c, agv_c,
                    traj_id_counter, FRAME_INTERVAL, MIN_POINTS, DRAW,
                    include_robots=True,
                    tool_center=tool_c, battery_center=batt_c, mask_prob=MASK_PROB,
                    dyn_center=dyn_c, dyn2_center=dyn2_c)
                
                if DRAW and ax:
                    finalize_plot(ax)
    
    print(f"\n✓ Generated {traj_id_counter} trajectories")
    print(f"  Saved to: {CSV_PATH}")
    print(f"  PGM files: {grid_id_counter}")


if __name__ == '__main__':
    main()