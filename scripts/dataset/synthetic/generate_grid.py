import numpy as np
from shapely.geometry import Polygon, Point

def generate_occupancy_grid(grid, objects_no_pad, grid_id, grid_size=256, resolution=0.05, wall_thickness=2, x_min=-5.3, x_max=5.4, y_min=-6.3, y_max=6.3):
    """
    Generate occupancy grid as a 256x256 PGM file with proper workspace mapping.
    
    Parameters:
    - grid: Grid object with environment bounds and resolution
    - objects_no_pad: List of objects without padding
    - grid_id: Identifier for the grid file
    - grid_size: Target size of the square grid (default 256x256)
    - resolution: Meters per pixel (default 0.05)
    - wall_thickness: Thickness of workspace walls in pixels (default 2)
    
    Returns:
    - grid_id: The grid identifier used
    """
    # Workspace bounds (from your specification)
  
    
    # Calculate actual workspace dimensions
    workspace_width = x_max - x_min   # 10.6 meters
    workspace_height = y_max - y_min   # 12.5 meters
    
    # Calculate workspace size in pixels based on resolution
    workspace_width_pixels = int(workspace_width / resolution)   # 10.6 / 0.05 = 212 pixels
    workspace_height_pixels = int(workspace_height / resolution)  # 12.5 / 0.05 = 250 pixels
    
    # Verify calculations
    print(f"Workspace dimensions: {workspace_width:.2f}m x {workspace_height:.2f}m")
    print(f"Workspace in pixels: {workspace_width_pixels} x {workspace_height_pixels}")
    
    # Calculate padding needed to reach 256x256
    padding_x_total = grid_size - workspace_width_pixels  # 256 - 212 = 44
    padding_y_total = grid_size - workspace_height_pixels  # 256 - 250 = 6
    
    # Distribute padding evenly (left/right for x, bottom/top for y)
    padding_x_left = padding_x_total // 2    # 22
    padding_x_right = padding_x_total - padding_x_left  # 22
    padding_y_bottom = padding_y_total // 2  # 3
    padding_y_top = padding_y_total - padding_y_bottom  # 3
    
    print(f"Padding: x_left={padding_x_left}, x_right={padding_x_right}, "
          f"y_bottom={padding_y_bottom}, y_top={padding_y_top}")
    
    # Create the full 256x256 grid initialized with gray (unknown/unexplored)
    # Use 128 for gray in 0-255 range
    full_grid = np.full((grid_size, grid_size), 128, dtype=np.uint8)
    
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
    # wall_thickness is now a parameter (default 2 pixels)
    
    # Top and bottom walls (horizontal)
    workspace_grid[0:wall_thickness, :] = 0  # Top wall (in image coords)
    workspace_grid[-wall_thickness:, :] = 0  # Bottom wall (in image coords)
    
    # Left and right walls (vertical)
    workspace_grid[:, 0:wall_thickness] = 0  # Left wall
    workspace_grid[:, -wall_thickness:] = 0  # Right wall
    
    # Mark occupied cells for each object in the workspace grid
    for obj in objects_no_pad:
        poly = Polygon(obj['corners'])
        
        # Get bounding box in workspace grid coordinates
        env_coords = obj['corners']
        grid_coords = [env_to_workspace_grid(x, y) for x, y in env_coords]
        
        i_min = max(0, min(i for i, j in grid_coords))
        i_max = min(workspace_width_pixels - 1, max(i for i, j in grid_coords))
        j_min = max(0, min(j for i, j in grid_coords))
        j_max = min(workspace_height_pixels - 1, max(j for i, j in grid_coords))
        
        # Check each cell in the bounding box
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # Convert grid indices back to environment coordinates
                x = x_min + (i + 0.5) * resolution
                y = y_min + (j + 0.5) * resolution
                
                # Check if this point is inside the polygon
                if poly.contains(Point(x, y)):
                    # Mark as occupied (0 = black)
                    # Note: In image coordinates, row 0 is at top
                    # We want y=0 at bottom, so we flip the j index
                    workspace_grid[workspace_height_pixels - 1 - j, i] = 0
    
    # Place the workspace grid into the full 256x256 grid with padding
    # Calculate where to place the workspace grid in the full grid
    start_row = padding_y_bottom  # Start from bottom padding
    end_row = start_row + workspace_height_pixels
    start_col = padding_x_left  # Start from left padding
    end_col = start_col + workspace_width_pixels
    
    # Copy workspace grid into the padded full grid
    full_grid[start_row:end_row, start_col:end_col] = workspace_grid
    
    # Write PGM file
    write_pgm(full_grid, f"grid_{grid_id}.pgm")
    
    # Also save metadata for reference
    save_grid_metadata(grid_id, x_min, x_max, y_min, y_max, resolution, 
                      padding_x_left, padding_x_right, padding_y_bottom, padding_y_top,
                      wall_thickness)
    
    return grid_id


def write_pgm(occupancy_grid, filename):
    """
    Write occupancy grid as a PGM file.
    
    Parameters:
    - occupancy_grid: 2D numpy array (0=occupied, 128=unknown/padding, 255=free)
    - filename: Output PGM filename
    """
    height, width = occupancy_grid.shape
    
    # Write PGM file (P5 = binary format)
    with open(filename, 'wb') as f:
        # Header
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        # Binary data
        f.write(occupancy_grid.tobytes())
    
    print(f"Saved occupancy grid to {filename} (256x256 PGM)")


def save_grid_metadata(grid_id, x_min, x_max, y_min, y_max, resolution,
                       pad_left, pad_right, pad_bottom, pad_top, wall_thickness):
    """
    Save metadata about the grid for later reference.
    
    This helps with coordinate transformations when using the grid.
    """
    import json
    
    metadata = {
        'grid_id': grid_id,
        'workspace_bounds': {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        },
        'resolution': resolution,
        'grid_size': 256,
        'wall_thickness': wall_thickness,
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
    
    with open(f"grid_{grid_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved grid metadata to grid_{grid_id}_metadata.json")


def coords_to_pixel(x, y, resolution=0.05):
    """
    Convert world coordinates to pixel coordinates in the 256x256 grid.
    
    Parameters:
    - x, y: World coordinates
    - resolution: Meters per pixel (default 0.05)
    
    Returns:
    - pixel_x, pixel_y: Pixel coordinates in the 256x256 grid
    """
    # Workspace bounds
    x_min, x_max = -5.3, 5.4
    y_min, y_max = -6.3, 6.3
    
    # Calculate workspace dimensions in pixels
    workspace_width_pixels = int((x_max - x_min) / resolution)   # 212
    workspace_height_pixels = int((y_max - y_min) / resolution)  # 250
    
    # Calculate padding
    padding_x_left = (256 - workspace_width_pixels) // 2  # 22
    padding_y_bottom = (256 - workspace_height_pixels) // 2  # 3
    
    # Convert to workspace pixel coordinates
    workspace_pixel_x = (x - x_min) / resolution
    workspace_pixel_y = (y - y_min) / resolution
    
    # Add padding and flip y-axis (image coordinates have y=0 at top)
    pixel_x = workspace_pixel_x + padding_x_left
    pixel_y = 256 - (workspace_pixel_y + padding_y_bottom)
    
    # Clamp to grid bounds
    pixel_x = max(0, min(255, int(pixel_x)))
    pixel_y = max(0, min(255, int(pixel_y)))
    
    return pixel_x, pixel_y


def pixel_to_coords(pixel_x, pixel_y, resolution=0.05):
    """
    Convert pixel coordinates in the 256x256 grid to world coordinates.
    
    Parameters:
    - pixel_x, pixel_y: Pixel coordinates in the 256x256 grid
    - resolution: Meters per pixel (default 0.05)
    
    Returns:
    - x, y: World coordinates
    """
    # Workspace bounds
    x_min, x_max = -5.25, 5.35
    y_min, y_max = -3.5, 9.0
    
    # Calculate workspace dimensions in pixels
    workspace_width_pixels = int((x_max - x_min) / resolution)   # 212
    workspace_height_pixels = int((y_max - y_min) / resolution)  # 250
    
    # Calculate padding
    padding_x_left = (256 - workspace_width_pixels) // 2  # 22
    padding_y_bottom = (256 - workspace_height_pixels) // 2  # 3
    
    # Remove padding and unflip y-axis
    workspace_pixel_x = pixel_x - padding_x_left
    workspace_pixel_y = (256 - pixel_y) - padding_y_bottom
    
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