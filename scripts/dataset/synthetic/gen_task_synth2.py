"""
generate_task_library_trajectories.py - Generate task library trajectories
"""
import json
import csv
import random
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Grid, spawn_goals, get_bounds, NoFreeCellsError
from generate_grid import generate_occupancy_grid
from trajectory_utils import (
    build_environment,
    generate_paths_and_log,
    AGV_WIDTH, AGV_HEIGHT
)
from utils import (
    partition_goals
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
ORIGIN_MODE = "fixed"  # Set to "fixed" for constant origin, "random" for variable origin

# Generation Settings
NUM_PARTS = 40
CSV_PATH = "paths.csv"
TASK_LIBRARY_PATH = "task_library.json"

# General Settings
DRAW = False
FRAME_INTERVAL = 0.4
MIN_POINTS = 25

# Fixed Grid Settings (matching random script)
GRID_SIZE = 640
GRID_RESOLUTION = 0.05

# Randomization Ranges - adjusted to always contain structured objects
# To ensure all objects fit with jitter/rotation, use minimum dimensions
MIN_WORKSPACE_WIDTH = 12.0   # meters (to contain all objects with margin)
MIN_WORKSPACE_HEIGHT = 14.0  # meters (to contain all objects with margin)

ORIGIN_OFFSET_X_RANGE = (-2.0, 2.0)  # meters (offset from calculated origin)
ORIGIN_OFFSET_Y_RANGE = (-2.0, 2.0)  # meters (offset from calculated origin)

ORIG_GRID_WIDTH_RANGE = (400, 640)   # pixels (larger minimum to fit objects)
ORIG_GRID_HEIGHT_RANGE = (400, 640)  # pixels (larger minimum to fit objects)
X_MIN_RANGE = (-12.0, -6.0)          # meters
Y_MIN_RANGE = (-10.0, -6.0)          # meters

FIXED_MAP_ORIGIN_X = -14.4
FIXED_MAP_ORIGIN_Y = -5.2

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_workspace_bounds(x_min, y_min, orig_grid_width, orig_grid_height, 
                               grid_size=GRID_SIZE, grid_resolution=GRID_RESOLUTION):
    """
    Calculate workspace bounds based on randomized parameters.
    
    Parameters:
    - x_min, y_min: Bottom-left corner of workspace
    - orig_grid_width, orig_grid_height: Original workspace size in pixels
    - grid_size: Target grid size in pixels
    - grid_resolution: Meters per pixel
    
    Returns:
    - Dictionary with workspace bounds and metadata
    """
    ws_width = grid_size * grid_resolution
    
    pad_x_pixels = grid_size - orig_grid_width
    pad_y_pixels = grid_size - orig_grid_height
    
    pad_x_meters = pad_x_pixels * grid_resolution
    pad_y_meters = pad_y_pixels * grid_resolution
    
    x_max = x_min + ws_width - pad_x_meters
    y_max = y_min + ws_width - pad_y_meters
    
    # Calculate workspace area
    workspace_width = x_max - x_min
    workspace_height = y_max - y_min
    workspace_area = workspace_width * workspace_height
    
    # Validate minimum dimensions for structured objects
    if workspace_width < MIN_WORKSPACE_WIDTH:
        print(f"  Warning: Workspace width {workspace_width:.2f}m < minimum {MIN_WORKSPACE_WIDTH}m")
    if workspace_height < MIN_WORKSPACE_HEIGHT:
        print(f"  Warning: Workspace height {workspace_height:.2f}m < minimum {MIN_WORKSPACE_HEIGHT}m")
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': workspace_width,
        'height': workspace_height,
        'area': workspace_area,
        'orig_width': orig_grid_width,
        'orig_height': orig_grid_height,
        'pad_x_pixels': pad_x_pixels,
        'pad_y_pixels': pad_y_pixels
    }


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
    
    print(f"=== TASK LIBRARY MODE (WITH RANDOMIZED WORKSPACE) ===")
    print(f"Processing {len(tasks)} tasks with {NUM_PARTS} parts each")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} @ {GRID_RESOLUTION}m/pixel")
    print(f"Randomizing workspace dimensions and position")
    print(f"Minimum workspace: {MIN_WORKSPACE_WIDTH}m x {MIN_WORKSPACE_HEIGHT}m (to fit all objects)")
    print(f"All structured objects (tool, battery, AGV, mobile, desk, dynamic) always generated")
    
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id",
                        "agent_id", "agent_type", "x", "y", "z"])
        
        for task in tasks:
            start_parts = partition_goals(task["start_pos"], NUM_PARTS)
            goal_parts = partition_goals(task["goal_pos"], NUM_PARTS)
            
            for stage in range(NUM_PARTS):
                # Randomize workspace parameters for this stage
                x_min = random.uniform(*X_MIN_RANGE)
                y_min = random.uniform(*Y_MIN_RANGE)
                orig_grid_width = random.randint(*ORIG_GRID_WIDTH_RANGE)
                orig_grid_height = random.randint(*ORIG_GRID_HEIGHT_RANGE)
                
                # Calculate workspace bounds
                workspace = calculate_workspace_bounds(x_min, y_min, orig_grid_width, orig_grid_height)
                x_min, x_max = workspace['x_min'], workspace['x_max']
                y_min, y_max = workspace['y_min'], workspace['y_max']
                workspace_area = workspace['area']
                
                print(f"\n[Task {task['task_id']}-{stage+1}] Workspace: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
                print(f"  Area: {workspace_area:.2f}m², Original grid: {orig_grid_width}×{orig_grid_height}px, "
                      f"Padding: {workspace['pad_x_pixels']}×{workspace['pad_y_pixels']}px")
                
                ax = setup_plot(f"Task {task['task_id']} - Part {stage+1}/{NUM_PARTS}: "
                               f"{task['title']}", x_min, x_max, y_min, y_max) if DRAW else None
                
                while True:
                    try:
                        pads, (tool_c, agv_c, mob_c, batt_c), objects_no_pad = \
                            build_environment(ax, task["agv_pos"], task["mobile_pos"], draw=DRAW)
                        grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)
                        
                        grid_id = f"{task['task_id']}_{stage+1}"
                        # Determine origin based on mode
                        if ORIGIN_MODE == "fixed":
                            map_origin_x = FIXED_MAP_ORIGIN_X
                            map_origin_y = FIXED_MAP_ORIGIN_Y
                        elif ORIGIN_MODE == "random":
                            # Calculate the workspace origin (with padding)
                            padding_x_left = (GRID_SIZE - orig_grid_width) // 2
                            padding_y_bottom = (GRID_SIZE - orig_grid_height) // 2
                            base_origin_x = x_min - padding_x_left * GRID_RESOLUTION
                            base_origin_y = y_min - padding_y_bottom * GRID_RESOLUTION
                            
                            # Apply random offset
                            origin_offset_x = random.uniform(*ORIGIN_OFFSET_X_RANGE)
                            origin_offset_y = random.uniform(*ORIGIN_OFFSET_Y_RANGE)
                            map_origin_x = base_origin_x + origin_offset_x
                            map_origin_y = base_origin_y + origin_offset_y
                        else:
                            raise ValueError(f"Invalid ORIGIN_MODE: {ORIGIN_MODE}. Must be 'fixed' or 'random'")

                        generate_occupancy_grid(grid, objects_no_pad, grid_id, 
                                            grid_size=GRID_SIZE, 
                                            resolution=GRID_RESOLUTION,
                                            x_min=x_min, y_min=y_min, 
                                            x_max=x_max, y_max=y_max,
                                            map_origin_x=map_origin_x,
                                            map_origin_y=map_origin_y)
                        grid_id_counter += 1

                        # Define goal areas
                        GOAL_AREA = {
                            "random": get_bounds((0.0, 0.0), 10.0, 12.0, 0),
                            "tool_station": get_bounds(tool_c, 1.05, 0.55, 0.0),
                            "battery_assembly": get_bounds((batt_c[0], batt_c[1]+0.45),
                                                          batt_c[0]+0.6, 0.0, 0.05),
                            "agv_ph1": get_bounds((agv_c[0]+2, agv_c[1]+3.5), 1.5, 1.5, 0.5),
                            "agv_ph2": get_bounds(agv_c, AGV_WIDTH+0.2, AGV_HEIGHT+0.2, 0.0),
                        }
                        
                        # Spawn starts and goals
                        starts, goals = [], []
                        for goal_type, n in start_parts[stage]:
                            if n > 0:
                                area = GOAL_AREA[goal_type]
                                starts.extend(spawn_goals(n, grid, pads,
                                                         area["x_bounds"], area["y_bounds"]))
                        for goal_type, n in goal_parts[stage]:
                            if n > 0:
                                area = GOAL_AREA[goal_type]
                                goals.extend(spawn_goals(n, grid, pads,
                                                        area["x_bounds"], area["y_bounds"]))
                        
                        print(f"  {len(starts)}×{len(goals)} trajectories")
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
                
                traj_id_counter = generate_paths_and_log(
                    ax, starts, goals, grid, pads, writer,
                    task['task_id'], grid_id, mob_c, agv_c,
                    traj_id_counter, FRAME_INTERVAL, MIN_POINTS, DRAW)
                
                if DRAW and ax:
                    finalize_plot(ax)
    
    print(f"\n✓ Generated {traj_id_counter} trajectories")
    print(f"  Saved to: {CSV_PATH}")
    print(f"  PGM files: {grid_id_counter}")


if __name__ == '__main__':
    main()