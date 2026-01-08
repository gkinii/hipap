"""
visualize_occupancy_grid.py - Visualize occupancy grid with coordinate overlay

Simply edit the constants below and run the script.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════════

YAML_FILE = "/home/gkini/Downloads/map_cleared.yaml"  # Path to map YAML file
SAVE_PATH = 'output2.png'                  # Set to "output.png" to save, or None to display
SHOW_GRID = True                  # Show coordinate grid lines
GRID_SPACING = 1.0                # Spacing between grid lines in meters
FIGSIZE = (12, 10)                # Figure size in inches (width, height)
# Padding settings
TARGET_SIZE = 640                 # Pad to this size (640x640)
PADDING_VALUE = 128 

def read_pgm(filename):
    """Read a PGM (Portable Gray Map) file and return as numpy array."""
    with open(filename, 'rb') as f:
        # Read header
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


def read_map_yaml(yaml_file):
    """Read map YAML file and return metadata."""
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
    
    # Calculate padding
    pad_x_total = target_size - orig_width
    pad_y_total = target_size - orig_height
    
    # Distribute padding evenly
    pad_x_left = pad_x_total // 2
    pad_x_right = pad_x_total - pad_x_left
    pad_y_bottom = pad_y_total // 2
    pad_y_top = pad_y_total - pad_y_bottom
    
    # Create padded grid
    padded_grid = np.full((target_size, target_size), padding_value, dtype=np.uint8)
    
    # Copy original grid into center
    start_row = pad_y_bottom
    end_row = start_row + orig_height
    start_col = pad_x_left
    end_col = start_col + orig_width
    
    padded_grid[start_row:end_row, start_col:end_col] = original_grid
    
    return padded_grid, pad_x_left, pad_y_bottom


def visualize_occupancy_grid(yaml_file, save_path=None, show_grid=True, 
                             grid_spacing=1.0, figsize=(12, 10),
                             target_size=640, padding_value=128):
    """
    Visualize occupancy grid with coordinate overlay, padded to target size.
    
    Parameters:
    - yaml_file: Path to map YAML file
    - save_path: Optional path to save the figure
    - show_grid: Whether to show coordinate grid lines
    - grid_spacing: Spacing between grid lines in meters
    - figsize: Figure size in inches
    - target_size: Target size to pad to
    - padding_value: Value for padding pixels
    """
    # Read YAML metadata
    map_info = read_map_yaml(yaml_file)
    
    # Get PGM file path (relative to YAML file)
    yaml_path = Path(yaml_file)
    pgm_file = yaml_path.parent / map_info['image']
    
    # Read original occupancy grid
    original_grid = read_pgm(pgm_file)
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
    
    print(f"Original map: {pgm_file.name}")
    print(f"Original size: {orig_width}×{orig_height} pixels")
    print(f"Padded size: {width}×{height} pixels")
    print(f"Padding: left={pad_x_left}px, bottom={pad_y_bottom}px")
    print(f"Resolution: {resolution} m/pixel")
    print(f"Original origin: ({orig_origin_x:.3f}, {orig_origin_y:.3f}, {orig_origin_theta:.3f})")
    print(f"Padded origin: ({origin_x:.3f}, {origin_y:.3f}, {orig_origin_theta:.3f})")
    print(f"World bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display padded occupancy grid with proper extent
    # Use origin='upper' (default) to match how the coordinate grid is set up
    im = ax.imshow(padded_grid, 
                   cmap='gray',
                   extent=[x_min, x_max, y_min, y_max],
                   origin='upper',
                   interpolation='nearest')
    
    # Add coordinate grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set major ticks at grid_spacing intervals
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
                     linewidth=2, edgecolor='blue', facecolor='none',
                     linestyle='--', label='Original workspace')
    ax.add_patch(rect)
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Padded Occupancy Grid: {pgm_file.name}\n'
                f'{width}×{height} pixels (padded from {orig_width}×{orig_height}) @ {resolution}m/px', 
                fontsize=14)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Add colorbar with occupancy meanings
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy Value', fontsize=10)
    cbar.ax.text(1.5, 255, 'Free', transform=cbar.ax.transData, 
                va='center', fontsize=9)
    cbar.ax.text(1.5, 192, 'Light Gray', transform=cbar.ax.transData, 
                va='center', fontsize=9)
    cbar.ax.text(1.5, 128, 'Unknown', transform=cbar.ax.transData, 
                va='center', fontsize=9)
    cbar.ax.text(1.5, 0, 'Occupied', transform=cbar.ax.transData, 
                va='center', fontsize=9)
    
    # Add origin markers
    ax.plot(origin_x, origin_y, 'r+', markersize=15, markeredgewidth=2, 
           label=f'Padded origin ({origin_x:.2f}, {origin_y:.2f})')
    ax.plot(orig_origin_x, orig_origin_y, 'gx', markersize=12, markeredgewidth=2,
           label=f'Original origin ({orig_origin_x:.2f}, {orig_origin_y:.2f})')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    else:
        plt.show()


def main():
    visualize_occupancy_grid(
        YAML_FILE,
        save_path=SAVE_PATH,
        show_grid=SHOW_GRID,
        grid_spacing=GRID_SPACING,
        figsize=FIGSIZE,
        target_size=TARGET_SIZE,
        padding_value=PADDING_VALUE
    )


if __name__ == '__main__':
    main()