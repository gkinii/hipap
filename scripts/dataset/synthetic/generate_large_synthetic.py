"""
generate_large_trajectories.py - Generate large multi-station trajectories with 640x640 maps
"""
import csv
import random
import matplotlib.pyplot as plt
from utils import Grid, NoFreeCellsError
from generate_grid import generate_occupancy_grid
from trajectory_utils import build_environment, generate_large_trajectory

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Generation Settings
NUM_LARGE_TRAJECTORIES = 1
NUM_WAYPOINTS = 10
CSV_PATH = "path_large.csv"

# General Settings
DRAW = False
FRAME_INTERVAL = 0.4

# Grid Settings
GRID_SIZE = 640
GRID_RESOLUTION = 0.05

# Workspace bounds (256x256 pixels at 0.05m/pixel = 12.8m x 12.8m)
# This gives 192 pixels of padding on each side in the 640x640 grid
x_min, y_min = -6.4, -6.4
x_max, y_max = 6.4, 6.4

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def setup_plot(title):
    """Create and configure a plot"""
    fig, ax = plt.subplots(figsize=(9, 7))
    setup_axes(ax, title)
    return ax

def setup_axes(ax, title):
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
    """Generate large multi-station trajectories with 640x640 maps"""
    traj_id_counter = 0
    grid_id_counter = 0
    
    print(f"=== LARGE TRAJECTORY MODE ===")
    print(f"Generating {NUM_LARGE_TRAJECTORIES} trajectories with {NUM_WAYPOINTS} waypoints")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} @ {GRID_RESOLUTION}m/pixel")
    print(f"Workspace: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
    
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id",
                        "agent_id", "agent_type", "x", "y", "z"])
        
        for traj_num in range(NUM_LARGE_TRAJECTORIES):
            ax = setup_plot(f"Large Trajectory {traj_num + 1}") if DRAW else None
            
            while True:
                try:
                    # Build environment (objects are centered in workspace)
                    agv_key = random.choice(["random", "marriage_point_agv"])
                    mobile_key = random.choice(["assembly", "other"])
                    pads, (tool_c, agv_c, mob_c, batt_c), objects_no_pad = \
                        build_environment(ax, agv_key, mobile_key, draw=DRAW)
                    
                    grid = Grid((x_min, x_max), (y_min, y_max), resolution=0.2)
                    
                    task_id = f"large_{traj_num}"
                    grid_id = f"large_{grid_id_counter}"
                    
                    # Generate 640x640 occupancy grid
                    generate_occupancy_grid(grid, objects_no_pad, grid_id, 
                                          grid_size=GRID_SIZE,
                                          resolution=GRID_RESOLUTION,
                                          x_min=x_min, y_min=y_min,
                                          x_max=x_max, y_max=y_max)
                    grid_id_counter += 1
                    
                    print(f"[Trajectory {traj_num + 1}] Task: {task_id}, Grid: {grid_id}")
                    
                    # Generate large trajectory with robots included
                    traj_id_counter = generate_large_trajectory(
                        ax, grid, pads, writer, task_id, grid_id,
                        mob_c, agv_c, traj_id_counter, NUM_WAYPOINTS,
                        FRAME_INTERVAL, draw=DRAW, include_robots=True)

                    break
                    
                except NoFreeCellsError as e:
                    print(f"  {e} → rebuilding environment and retrying...")
                    if DRAW and ax:
                        ax.cla()
                        setup_axes(ax, f"Large Trajectory {traj_num + 1}")
            
            if DRAW and ax:
                finalize_plot(ax)
    
    print(f"\n✓ Generated {NUM_LARGE_TRAJECTORIES} large trajectories")
    print(f"  Saved to: {CSV_PATH}")
    print(f"  PGM files: {grid_id_counter}")

if __name__ == '__main__':
    main()