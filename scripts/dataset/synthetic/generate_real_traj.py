import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Input CSV from pose estimation
CSV_INPUT_PATH = "/home/gkini/Human-Traj-Prediction/runs/predict/pose_fused2/com_series_world_frame.csv"

# Map configuration
YAML_FILE = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_cell/maps_1/map_real_0.yaml"  # Update to actual map.yaml path
SAVE_PATH = "trajectory_on_map.png"           # Output visualization
SHOW_GRID = True                              # Show coordinate grid lines
GRID_SPACING = 1.0                            # Spacing between grid lines in meters
FIGSIZE = (12, 10)                            # Figure size in inches
TARGET_SIZE = 256                             # Pad to this size (640x640)
PADDING_VALUE = 192                           # Value for padding pixels

# Camera position in world frame (from pose estimation script)
CAMERA_POS_WORLD = np.array([0.96618, 3.4708, 0.2825])

# Metadata for CSV output
TASK_ID = "real_0"
GRID_ID = "real_0"
TRAJ_ID = 0
AGENT_ID = 0
AGENT_TYPE = 1           # 1 = human
CSV_PATH = "path_real.csv"

# Person ID to extract
PERSON_ID = 1

# ═══════════════════════════════════════════════════════════════════════════════

def read_pgm(filename):
    """Read a PGM file and return as numpy array."""
    try:
        with open(filename, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            if header != 'P5':
                raise ValueError(f"Not a valid P5 PGM file: {header}")
            
            # Skip comments
            line = f.readline().decode('ascii').strip()
            while line.startswith('#'):
                line = f.readline().decode('ascii').strip()
            
            # Read width and height
            width, height = map(int, line.split())
            
            # Read max value
            max_val = int(f.readline().decode('ascii').strip())
            
            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape((height, width))
        
        return data
    except Exception as e:
        print(f"Error reading PGM {filename}: {e}")
        return None


def read_map_yaml(yaml_file):
    """Read map YAML file and return metadata."""
    try:
        with open(yaml_file, 'r') as f:
            map_data = yaml.safe_load(f)
        return {
            'image': map_data['image'],
            'resolution': map_data['resolution'],
            'origin': map_data['origin'],
            'negate': map_data.get('negate', 0),
            'occupied_thresh': map_data.get('occupied_thresh', 0.65),
            'free_thresh': map_data.get('free_thresh', 0.196)
        }
    except Exception as e:
        print(f"Error reading YAML {yaml_file}: {e}")
        return None


def pad_grid_to_target(original_grid, target_size, padding_value=128):
    """
    Pad grid to target size with even padding on all sides.
    
    Returns:
        - padded_grid: Padded occupancy grid
        - pad_x_left, pad_y_bottom: Padding amounts
    """
    orig_height, orig_width = original_grid.shape
    
    if orig_width > target_size or orig_height > target_size:
        print(f"Warning: Original map ({orig_width}x{orig_height}) is larger than target ({target_size}x{target_size})")
        return original_grid, 0, 0
    
    if orig_width == target_size and orig_height == target_size:
        return original_grid, 0, 0
    
    pad_x_total = target_size - orig_width
    pad_y_total = target_size - orig_height
    
    pad_x_left = pad_x_total // 2
    pad_x_right = pad_x_total - pad_x_left
    pad_y_bottom = pad_y_total // 2
    pad_y_top = pad_y_total - pad_y_bottom
    
    padded_grid = np.full((target_size, target_size), padding_value, dtype=np.uint8)
    
    start_row = pad_y_bottom
    end_row = start_row + orig_height
    start_col = pad_x_left
    end_col = start_col + orig_width
    
    padded_grid[start_row:end_row, start_col:end_col] = original_grid
    
    return padded_grid, pad_x_left, pad_y_bottom


def load_trajectory_from_csv(csv_path, person_id):
    """
    Load trajectory points for a specific person from the CSV.
    
    Returns:
        List of (x, y) tuples representing the trajectory
    """
    trajectory = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["person_id"]) == person_id:
                    x = float(row["X_m"])
                    y = float(row["Y_m"])
                    trajectory.append((x, y))
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    if not trajectory:
        print(f"No data found for person_id {person_id} in {csv_path}")
    return trajectory


def visualize_on_pgm(yaml_file, trajectory, camera_pos, save_path, show_grid=True, 
                     grid_spacing=1.0, figsize=(12, 10), target_size=640, 
                     padding_value=128):
    """
    Visualize trajectory and camera position on the PGM map using Matplotlib.
    
    Args:
        camera_pos: Camera position in world frame as [x, y, z] (only x,y used)
    
    Returns True if successful, False otherwise.
    """
    # Read YAML metadata
    map_info = read_map_yaml(yaml_file)
    if map_info is None:
        return False
    
    # Get PGM file path (relative to YAML file)
    yaml_path = Path(yaml_file)
    pgm_file = yaml_path.parent / map_info['image']
    
    # Read original occupancy grid
    original_grid = read_pgm(pgm_file)
    if original_grid is None:
        return False
    
    orig_height, orig_width = original_grid.shape
    
    # Pad to target size
    padded_grid, pad_x_left, pad_y_bottom = pad_grid_to_target(
        original_grid, target_size, padding_value)
    
    height, width = padded_grid.shape
    
    # Get map parameters
    resolution = map_info['resolution']
    orig_origin_x, orig_origin_y, orig_origin_theta = map_info['origin']
    
    # Calculate new origin (shifted by padding)
    origin_x = orig_origin_x - pad_x_left * resolution
    origin_y = orig_origin_y - pad_y_bottom * resolution
    
    # Calculate world coordinates for padded grid
    x_min = origin_x
    x_max = origin_x + width * resolution
    y_min = origin_y
    y_max = origin_y + height * resolution
    
    print(f"Map: {pgm_file.name}")
    print(f"Original size: {orig_width}×{orig_height} pixels")
    print(f"Padded size: {width}×{height} pixels")
    print(f"Padding: left={pad_x_left}px, bottom={pad_y_bottom}px")
    print(f"Resolution: {resolution} m/pixel")
    print(f"Original origin: ({orig_origin_x:.3f}, {orig_origin_y:.3f}, {orig_origin_theta:.3f})")
    print(f"Padded origin: ({origin_x:.3f}, {origin_y:.3f}, {orig_origin_theta:.3f})")
    print(f"World bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    
    # Camera position verification
    cam_x, cam_y = camera_pos[0], camera_pos[1]
    cam_u = int((cam_x - origin_x) / resolution)
    cam_v = height - 1 - int((cam_y - origin_y) / resolution)
    print(f"\nCamera position: ({cam_x:.4f}, {cam_y:.4f}) -> Pixel ({cam_u}, {cam_v})")
    if 0 <= cam_u < width and 0 <= cam_v < height:
        print("✓ Camera is within map bounds")
    else:
        print("⚠ WARNING: Camera is OUTSIDE map bounds!")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display padded occupancy grid with proper extent
    im = ax.imshow(padded_grid, 
                   cmap='gray',
                   extent=[x_min, x_max, y_min, y_max],
                   origin='upper',
                   interpolation='nearest')
    
    # Plot camera position (smaller marker)
    ax.plot(cam_x, cam_y, 'c^', markersize=8, markeredgewidth=1, 
            markeredgecolor='black', label=f'Camera', zorder=10)
    
    # Plot trajectory
    valid_points = 0
    if trajectory:
        x_coords, y_coords = zip(*trajectory)
        
        # Verify each point
        print("\n=== TRAJECTORY POINTS ===")
        for i, (x, y) in enumerate(trajectory):
            # Check if point is within map bounds
            u = int((x - origin_x) / resolution)
            v = height - 1 - int((y - origin_y) / resolution)
            in_bounds = 0 <= u < width and 0 <= v < height
            status = "✓" if in_bounds else "✗"
            print(f"{status} Frame {i}: World ({x:.4f}, {y:.4f}) -> Pixel ({u}, {v})")
            if in_bounds:
                valid_points += 1
        
        # Plot trajectory line
        ax.plot(x_coords, y_coords, 'g-', linewidth=1.5, label='Trajectory', 
                alpha=0.7, zorder=5)
        
        # Plot trajectory points (smaller)
        ax.scatter(x_coords, y_coords, c='lime', s=20, edgecolors='darkgreen', 
                   linewidths=0.5, zorder=6)
        
        # Start point (smaller)
        ax.scatter(x_coords[0], y_coords[0], c='yellow', s=60, marker='*', 
                   edgecolors='black', linewidths=1, label='Start', zorder=7)
        
        # End point (smaller)
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=50, marker='s', 
                   edgecolors='black', linewidths=1, label='End', zorder=7)
    
    # Add coordinate grid
    if show_grid:
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        x_ticks = np.arange(np.floor(x_min), np.ceil(x_max) + grid_spacing, grid_spacing)
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + grid_spacing, grid_spacing)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    
    # Draw rectangle showing original workspace bounds
    orig_x_min = orig_origin_x
    orig_x_max = orig_origin_x + orig_width * resolution
    orig_y_min = orig_origin_y
    orig_y_max = orig_origin_y + orig_height * resolution
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((orig_x_min, orig_y_min), 
                     orig_x_max - orig_x_min, 
                     orig_y_max - orig_y_min,
                     linewidth=1.5, edgecolor='blue', facecolor='none',
                     linestyle='--', label='Original workspace', alpha=0.6)
    ax.add_patch(rect)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title(f'Trajectory on Map: {pgm_file.name}', fontsize=12, pad=10)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Add colorbar with occupancy meanings (smaller)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy', fontsize=9)
    
    # Add origin markers (smaller)
    ax.plot(origin_x, origin_y, 'r+', markersize=8, markeredgewidth=1.5, 
            label=f'Padded origin', alpha=0.7)
    ax.plot(orig_origin_x, orig_origin_y, 'gx', markersize=7, markeredgewidth=1.5,
            label=f'Map origin', alpha=0.7)
    
    # Add legend (smaller, more compact)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, 
              markerscale=0.8, handlelength=1.5, handletextpad=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualized map saved to: {save_path}")
    print(f"Valid trajectory points: {valid_points}/{len(trajectory)}")
    plt.close()
    
    return valid_points > 0


def main():
    """Generate trajectory CSV from pose estimation data and visualize on PGM"""
    
    print(f"=== REAL TRAJECTORY FROM POSE ESTIMATION ===")
    print(f"Input CSV: {CSV_INPUT_PATH}")
    print(f"Person ID: {PERSON_ID}")
    print(f"Camera Position: ({CAMERA_POS_WORLD[0]:.4f}, {CAMERA_POS_WORLD[1]:.4f}, {CAMERA_POS_WORLD[2]:.4f})")
    print(f"YAML File: {YAML_FILE}")
    
    # Load trajectory
    trajectory = load_trajectory_from_csv(CSV_INPUT_PATH, PERSON_ID)
    
    if not trajectory:
        print("No trajectory data to process.")
        return
    
    print(f"Frames: {len(trajectory)}")
    
    # Write to CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["task_id", "grid_id", "traj_id", "frame_id", 
                        "agent_id", "agent_type", "x", "y", "z", "pos_mask"])
        
        # Write trajectory frames
        for frame_id, (x, y) in enumerate(trajectory):
            writer.writerow([
                TASK_ID, 
                GRID_ID, 
                TRAJ_ID, 
                frame_id, 
                AGENT_ID, 
                AGENT_TYPE, 
                f"{x:.6f}", 
                f"{y:.6f}", 
                0,
                1
            ])
    
    print(f"\n✓ Trajectory saved to: {CSV_PATH}")
    
    # Print first and last few frames for verification
    print("\nFirst 3 frames:")
    for i in range(min(3, len(trajectory))):
        print(f"  Frame {i}: ({trajectory[i][0]:.4f}, {trajectory[i][1]:.4f})")
    
    if len(trajectory) > 3:
        print("\nLast 3 frames:")
        for i in range(max(0, len(trajectory) - 3), len(trajectory)):
            print(f"  Frame {i}: ({trajectory[i][0]:.4f}, {trajectory[i][1]:.4f})")
    
    # Visualize on PGM
    visualize_on_pgm(
        YAML_FILE, 
        trajectory, 
        CAMERA_POS_WORLD,  # Pass camera position
        SAVE_PATH, 
        show_grid=SHOW_GRID, 
        grid_spacing=GRID_SPACING, 
        figsize=FIGSIZE, 
        target_size=TARGET_SIZE, 
        padding_value=PADDING_VALUE
    )


if __name__ == '__main__':
    main()