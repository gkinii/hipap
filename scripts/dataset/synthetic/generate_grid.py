import numpy as np
from shapely.geometry import Polygon, Point
import random
from scipy import ndimage
import os
def generate_occupancy_grid(grid, objects_no_pad, grid_id, grid_size=256, resolution=0.05,
                            wall_thickness=2, x_min=-5.3, x_max=5.4, y_min=-6.3, y_max=6.3,
                            map_origin_x=-6.4, map_origin_y=-6.4, output_dir='256'):
    """
    Generate occupancy grid as a PGM file with proper workspace mapping.
    
    Parameters:
    - map_origin_x, map_origin_y: Origin values to write to YAML file
    """
  
    # Calculate workspace dimensions
    workspace_width = x_max - x_min
    workspace_height = y_max - y_min

    # Calculate workspace size in pixels based on resolution
    workspace_width_pixels = int(workspace_width / resolution)
    workspace_height_pixels = int(workspace_height / resolution)

    # Verify calculations
    print(f"Workspace dimensions: {workspace_width:.2f}m x {workspace_height:.2f}m")
    print(f"Workspace in pixels: {workspace_width_pixels} x {workspace_height_pixels}")

    # Calculate padding needed to reach target grid size
    padding_x_total = grid_size - workspace_width_pixels
    padding_y_total = grid_size - workspace_height_pixels

    # Distribute padding evenly (left/right for x, bottom/top for y)
    padding_x_left = padding_x_total // 2
    padding_x_right = padding_x_total - padding_x_left
    padding_y_bottom = padding_y_total // 2
    padding_y_top = padding_y_total - padding_y_bottom

    print(f"Padding: x_left={padding_x_left}, x_right={padding_x_right}, "
          f"y_bottom={padding_y_bottom}, y_top={padding_y_top}")

    # Create the full grid initialized with gray (unknown/unexplored)
    # Use 128 for gray in 0-255 range
    full_grid = np.full((grid_size, grid_size), 192, dtype=np.uint8)

    # Create workspace occupancy grid (initially all free = 255)
    workspace_grid = np.full((workspace_height_pixels, workspace_width_pixels), 255, dtype=np.uint8)

    # Function to convert environment coords to workspace grid indices
    def env_to_workspace_grid(x, y):
        """Convert environment coordinates to workspace grid pixel indices."""
        # Convert to pixel coordinates in workspace grid
        i = int((x - x_min) / resolution)
        j = int((y - y_min) / resolution)
        # Clamp to workspace grid bounds
        i = max(0, min(workspace_width_pixels - 1, i))
        j = max(0, min(workspace_height_pixels - 1, j))
        return i, j

    # Add walls (black borders) around the workspace edges
    # Top and bottom walls (horizontal)
    workspace_grid[0:wall_thickness, :] = 0  # Top wall
    workspace_grid[-wall_thickness:, :] = 0  # Bottom wall
    # Left and right walls (vertical)
    workspace_grid[:, 0:wall_thickness] = 0  # Left wall
    workspace_grid[:, -wall_thickness:] = 0  # Right wall

    # Mark occupied cells for each object in the workspace grid
    obstacle_outline_value = 0
    outline_thickness = 2  # Thickness of outline in pixels

    for obj in objects_no_pad:
        poly = Polygon(obj['corners'])
        
        # Get bounding box in workspace grid coordinates
        env_coords = obj['corners']
        grid_coords = [env_to_workspace_grid(x, y) for x, y in env_coords]
        
        i_min = max(0, min(i for i, j in grid_coords))
        i_max = min(workspace_width_pixels - 1, max(i for i, j in grid_coords))
        j_min = max(0, min(j for i, j in grid_coords))
        j_max = min(workspace_height_pixels - 1, max(j for i, j in grid_coords))

        # Create a temporary mask for this obstacle
        obstacle_mask = np.zeros((workspace_height_pixels, workspace_width_pixels), dtype=bool)

        # Mark all cells inside the polygon
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # Convert grid indices back to environment coordinates
                x = x_min + (i + 0.5) * resolution
                y = y_min + (j + 0.5) * resolution
                
                # Check if this point is inside the polygon
                if poly.contains(Point(x, y)):
                    # Flip j index so y=0 is at bottom
                    flipped_j = workspace_height_pixels - 1 - j
                    obstacle_mask[flipped_j, i] = True

        # --- MODIFICATION STARTS HERE ---
        # Randomly choose between 192 (light gray) and 255 (white)
        # random.choice selects one value from the list with equal probability
        current_interior_value = random.choice([192, 255, 128, 216])
        
        # Apply color to the mask
        workspace_grid[obstacle_mask] = current_interior_value
        # --- MODIFICATION ENDS HERE ---

        # Then mark the outline as black
        # Create erosion structure based on desired thickness
        structure_size = 2 * outline_thickness + 1
        erode_structure = np.ones((structure_size, structure_size))
        
        # Erode the mask to get interior only
        eroded = ndimage.binary_erosion(obstacle_mask, structure=erode_structure)
        
        # Outline is where original mask is True but eroded is False
        outline_mask = obstacle_mask & ~eroded
        workspace_grid[outline_mask] = obstacle_outline_value

    # Place the workspace grid into the full grid with padding
    start_row = padding_y_bottom
    end_row = start_row + workspace_height_pixels
    start_col = padding_x_left
    end_col = start_col + workspace_width_pixels

    # Copy workspace grid into the padded full grid
    full_grid[start_row:end_row, start_col:end_col] = workspace_grid

    # Write PGM file
    write_pgm(full_grid, f"grid_{grid_id}.pgm", grid_size, output_dir=output_dir)

    # Save map YAML for ROS/navigation
    save_map_yaml(grid_id, resolution, map_origin_x, map_origin_y, output_dir=output_dir, grid_dir=output_dir)

    return grid_id

   
METADATA_OUTPUT_DIR = "grid_metadata" 

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def write_pgm(occupancy_grid, filename, grid_size, output_dir='256'):
    """
    Write occupancy grid as PGM file.
    
    Parameters:
    - occupancy_grid: The grid data to save
    - filename: Name of the file (without path)
    - grid_size: Size of the grid
    - output_dir: Directory to save the file (default: GRID_OUTPUT_DIR)
    """
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    
    height, width = occupancy_grid.shape
    with open(filepath, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(occupancy_grid.tobytes())
    print(f"Saved occupancy grid to {filepath} ({grid_size}x{grid_size} PGM)")

def save_map_yaml(grid_id, resolution, origin_x, origin_y, 
                  output_dir='256', grid_dir='256'):
    """
    Save map YAML file compatible with ROS navigation stack.
    
    Parameters:
    - grid_id: Identifier for the map
    - resolution: Meters per pixel
    - origin_x, origin_y: Origin coordinates for the map
    - output_dir: Directory to save YAML file (default: YAML_OUTPUT_DIR)
    - grid_dir: Directory where PGM files are stored (for relative path in YAML)
    """
    ensure_directory_exists(output_dir)
    
    # Determine the relative path to the PGM file
    if grid_dir and output_dir:
        # Calculate relative path from YAML dir to grid dir
        grid_path = os.path.relpath(
            os.path.join(grid_dir, f"grid_{grid_id}.pgm"),
            output_dir
        )
    elif grid_dir:
        grid_path = os.path.join(grid_dir, f"grid_{grid_id}.pgm")
    else:
        grid_path = f"grid_{grid_id}.pgm"
    
    origin_z = 0.0
    yaml_content = f"""image: {grid_path}
mode: trinary
resolution: {resolution:.3f}
origin: [{origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    
    filename = f"map_{grid_id}.yaml"
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    
    with open(filepath, 'w') as f:
        f.write(yaml_content)
    print(f"Saved map YAML to {filepath}")
    print(f"  Origin: [{origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}]")
    print(f"  Image path: {grid_path}")

def save_grid_metadata(grid_id, x_min, x_max, y_min, y_max, resolution,
                       pad_left, pad_right, pad_bottom, pad_top, 
                       wall_thickness, grid_size, output_dir=METADATA_OUTPUT_DIR):
    """
    Save metadata about the grid.
    
    Parameters:
    - grid_id: Identifier for the grid
    - x_min, x_max, y_min, y_max: Workspace bounds
    - resolution: Grid resolution in meters per pixel
    - pad_left, pad_right, pad_bottom, pad_top: Padding in pixels
    - wall_thickness: Thickness of walls
    - grid_size: Total grid size
    - output_dir: Directory to save metadata (default: METADATA_OUTPUT_DIR)
    """
    import json
    
    ensure_directory_exists(output_dir)
    
    # Calculate the origin of the full grid in environment coordinates
    origin_x = x_min - pad_left * resolution
    origin_y = y_min - pad_bottom * resolution
    
    metadata = {
        'grid_id': grid_id,
        'workspace_bounds': {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        },
        'resolution': resolution,
        'grid_size': grid_size,
        'wall_thickness': wall_thickness,
        'origin': {
            'x': origin_x,
            'y': origin_y,
            'description': 'Position of full grid bottom-left corner (0,0) in environment coordinates'
        },
        'workspace_pixels': {
            'width': int((x_max - x_min) / resolution),
            'height': int((y_max - y_min) / resolution)
        },
        'padding': {
            'left': pad_left,
            'right': pad_right,
            'bottom': pad_bottom,
            'top': pad_top
        }
    }
    
    filename = f"grid_{grid_id}_metadata.json"
    filepath = os.path.join(output_dir, filename) if output_dir else filename
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved grid metadata to {filepath}")
    print(f"Full grid origin in environment coords: ({origin_x:.3f}, {origin_y:.3f})")

def coords_to_pixel(x, y, resolution=0.05, x_min=-5.3, x_max=5.4, y_min=-6.3, y_max=6.3, grid_size=256):
    """
    Convert world coordinates to pixel coordinates in the 256x256 grid.
    
    Parameters:
    - x, y: World coordinates
    - resolution: Meters per pixel (default 0.05)
    
    Returns:
    - pixel_x, pixel_y: Pixel coordinates in the 256x256 grid
    """
    # Workspace bounds
    x_min, x_max = x_min, x_max
    y_min, y_max = y_min, y_max
    
    # Calculate workspace dimensions in pixels
    workspace_width_pixels = int((x_max - x_min) / resolution)   
    workspace_height_pixels = int((y_max - y_min) / resolution)  

    # Calculate padding
    padding_x_left = (grid_size - workspace_width_pixels) // 2  
    padding_y_bottom = (grid_size - workspace_height_pixels) // 2  
    
    # Convert to workspace pixel coordinates
    workspace_pixel_x = (x - x_min) / resolution
    workspace_pixel_y = (y - y_min) / resolution
    
    # Add padding and flip y-axis (image coordinates have y=0 at top)
    pixel_x = workspace_pixel_x + padding_x_left
    pixel_y = grid_size - (workspace_pixel_y + padding_y_bottom)
    
    # Clamp to grid bounds
    pixel_x = max(0, min(255, int(pixel_x)))
    pixel_y = max(0, min(255, int(pixel_y)))
    
    return pixel_x, pixel_y


def pixel_to_coords(pixel_x, pixel_y, resolution=0.05, x_min=-5.3, x_max=5.4, y_min=-6.3, y_max=6.3, grid_size=256):
    """
    Convert pixel coordinates in the 256x256 grid to world coordinates.
    
    Parameters:
    - pixel_x, pixel_y: Pixel coordinates in the 256x256 grid
    - resolution: Meters per pixel (default 0.05)
    
    Returns:
    - x, y: World coordinates
    """
    # Workspace bounds
    x_min, x_max = x_min, x_max
    y_min, y_max = y_min, y_max
    
    # Calculate workspace dimensions in pixels
    workspace_width_pixels = int((x_max - x_min) / resolution)   # 212
    workspace_height_pixels = int((y_max - y_min) / resolution)  # 250
    
    # Calculate padding
    padding_x_left = (grid_size - workspace_width_pixels) // 2  # 22
    padding_y_bottom = (grid_size - workspace_height_pixels) // 2  # 3
    
    # Remove padding and unflip y-axis
    workspace_pixel_x = pixel_x - padding_x_left
    workspace_pixel_y = (grid_size - pixel_y) - padding_y_bottom
    
    # Convert to world coordinates
    x = x_min + workspace_pixel_x * resolution
    y = y_min + workspace_pixel_y * resolution
    
    return x, y


def visualize_pgm_with_grid(filename):
    """
    Load and visualize a PGM file with grid lines showing the workspace.
    
    Parameters:
    - filename: PGM file to visualize
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Read PGM file
    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode('ascii').strip()
        if header != 'P5':
            raise ValueError(f"Not a valid binary PGM file: {header}")
        
        # Skip comments
        line = f.readline().decode('ascii').strip()
        while line.startswith('#'):
            line = f.readline().decode('ascii').strip()
        
        # Read dimensions
        width, height = map(int, line.split())
        
        # Read max value
        max_val = int(f.readline().decode('ascii').strip())
        
        # Read data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        image = data.reshape((height, width))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    # Add workspace boundary rectangle
    padding_x_left = 22
    padding_y_top = 3  # In image coordinates (top of image)
    workspace_width = 212
    workspace_height = 250
    
    # Draw workspace boundary
    rect = patches.Rectangle((padding_x_left, padding_y_top), 
                            workspace_width, workspace_height,
                            linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Add labels
    ax.text(128, 10, 'Gray Padding (Unknown)', ha='center', color='red', fontsize=10)
    ax.text(128, 246, 'Gray Padding (Unknown)', ha='center', color='red', fontsize=10)
    ax.text(11, 128, 'Padding', ha='center', color='red', rotation=90, fontsize=10)
    ax.text(245, 128, 'Padding', ha='center', color='red', rotation=90, fontsize=10)
    
    # Add workspace walls label
    ax.text(128, padding_y_top + workspace_height/2, 
            'Workspace with Black Walls', ha='center', color='blue', fontsize=9)
    
    plt.title(f'Occupancy Grid from {filename}')
    plt.colorbar(im, label='0=Occupied/Walls (black), 128=Unknown (gray), 255=Free (white)')
    ax.set_xlabel('X pixels')
    ax.set_ylabel('Y pixels')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return image