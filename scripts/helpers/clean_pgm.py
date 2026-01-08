import numpy as np
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Input files
INPUT_PGM = "map_cleared1.pgm"
MAP_YAML = "/home/gkini/Downloads/map.yaml"  # YAML file with map metadata (resolution, origin)

# Output PGM file
OUTPUT_PGM = "map_cleared.pgm"

# Circle center and radius in WORLD COORDINATES (meters)
CENTER_X = -0.4    # X coordinate in meters
CENTER_Y = -4.4    # Y coordinate in meters
RADIUS = 0.5       # Radius in meters

# Value for free/cleared cells (255 = white/free, 0 = black/occupied)
FREE_VALUE = 255

# ═══════════════════════════════════════════════════════════════════════════════

def read_yaml(yaml_file):
    """Read map YAML file and return metadata."""
    with open(yaml_file, 'r') as f:
        map_data = yaml.safe_load(f)
    return {
        'resolution': map_data['resolution'],
        'origin': map_data['origin']  # [x, y, theta]
    }


def read_pgm(filename):
    """Read a PGM file and return as numpy array."""
    with open(filename, 'rb') as f:
        # Read header
        header = f.readline().decode('ascii').strip()
        if header not in ['P5', 'P2']:
            raise ValueError(f"Not a valid PGM file. Header: {header}")
        
        # Skip comments
        line = f.readline().decode('ascii').strip()
        while line.startswith('#'):
            line = f.readline().decode('ascii').strip()
        
        # Read dimensions
        width, height = map(int, line.split())
        
        # Read max value
        max_val = int(f.readline().decode('ascii').strip())
        
        # Read image data
        if header == 'P5':  # Binary
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:  # ASCII (P2)
            data = np.array([int(x) for x in f.read().split()], dtype=np.uint8)
        
        data = data.reshape((height, width))
    
    return data, max_val


def write_pgm(filename, image, max_val=255):
    """Write a numpy array to a PGM file (binary format P5)."""
    height, width = image.shape
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(f"P5\n".encode('ascii'))
        f.write(f"{width} {height}\n".encode('ascii'))
        f.write(f"{max_val}\n".encode('ascii'))
        
        # Write image data
        f.write(image.tobytes())


def world_to_pixel(x_world, y_world, origin_x, origin_y, resolution, height):
    """
    Convert world coordinates (meters) to pixel coordinates.
    
    Args:
        x_world, y_world: World coordinates in meters
        origin_x, origin_y: Map origin in world frame
        resolution: Meters per pixel
        height: Image height in pixels
    
    Returns:
        pixel_x, pixel_y: Pixel coordinates (column, row)
    """
    # Convert to pixel coordinates
    pixel_x = int((x_world - origin_x) / resolution)
    pixel_y = height - 1 - int((y_world - origin_y) / resolution)
    
    return pixel_x, pixel_y


def clear_circular_area(image, center_x, center_y, radius, free_value=254):
    """
    Clear a circular area in the occupancy grid.
    
    Args:
        image: Input numpy array (height x width)
        center_x: X coordinate of circle center (pixel column)
        center_y: Y coordinate of circle center (pixel row)
        radius: Radius of circle in pixels
        free_value: Value to set for cleared cells (default 254 = free)
    
    Returns:
        Modified image with cleared circular area, number of cleared pixels
    """
    height, width = image.shape
    output = image.copy()
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Calculate distance from center for all pixels
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create circular mask
    mask = distances <= radius
    
    # Set pixels within circle to free value
    output[mask] = free_value
    
    # Count modified pixels
    num_cleared = np.sum(mask)
    
    return output, num_cleared


def main():
    print("=" * 70)
    print("PGM CIRCULAR AREA CLEARER (World Coordinates)")
    print("=" * 70)
    
    # Read map metadata
    print(f"\nReading map metadata: {MAP_YAML}")
    try:
        map_info = read_yaml(MAP_YAML)
        resolution = map_info['resolution']
        origin_x, origin_y, origin_theta = map_info['origin']
        print(f"  Resolution: {resolution} m/pixel")
        print(f"  Origin: ({origin_x}, {origin_y}, {origin_theta})")
    except FileNotFoundError:
        print(f"ERROR: YAML file not found: {MAP_YAML}")
        return
    except Exception as e:
        print(f"ERROR reading YAML: {e}")
        return
    
    # Read input PGM
    print(f"\nReading input: {INPUT_PGM}")
    try:
        image, max_val = read_pgm(INPUT_PGM)
        height, width = image.shape
        print(f"  Size: {width} x {height} pixels")
        print(f"  Max value: {max_val}")
        
        # Calculate world bounds
        x_min = origin_x
        x_max = origin_x + width * resolution
        y_min = origin_y
        y_max = origin_y + height * resolution
        print(f"  World bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    except FileNotFoundError:
        print(f"ERROR: File not found: {INPUT_PGM}")
        return
    except Exception as e:
        print(f"ERROR reading PGM: {e}")
        return
    
    # Convert world coordinates to pixel coordinates
    print(f"\nConverting world coordinates to pixels:")
    print(f"  World center: ({CENTER_X:.3f}, {CENTER_Y:.3f}) meters")
    print(f"  World radius: {RADIUS:.3f} meters")
    
    pixel_x, pixel_y = world_to_pixel(CENTER_X, CENTER_Y, origin_x, origin_y, 
                                       resolution, height)
    pixel_radius = RADIUS / resolution
    
    print(f"  Pixel center: ({pixel_x}, {pixel_y})")
    print(f"  Pixel radius: {pixel_radius:.1f} pixels")
    
    # Validate coordinates
    if not (0 <= pixel_x < width):
        print(f"ERROR: Center X is outside image bounds [0, {width-1}]")
        print(f"       World X={CENTER_X:.3f} is outside map range [{x_min:.3f}, {x_max:.3f}]")
        return
    if not (0 <= pixel_y < height):
        print(f"ERROR: Center Y is outside image bounds [0, {height-1}]")
        print(f"       World Y={CENTER_Y:.3f} is outside map range [{y_min:.3f}, {y_max:.3f}]")
        return
    
    # Clear circular area
    print(f"\nClearing circular area:")
    print(f"  Free value: {FREE_VALUE}")
    
    modified_image, num_cleared = clear_circular_area(
        image, pixel_x, pixel_y, pixel_radius, FREE_VALUE
    )
    
    area_m2 = np.pi * RADIUS**2
    print(f"  Cleared pixels: {num_cleared}")
    print(f"  Cleared area: {area_m2:.3f} m²")
    
    # Write output PGM
    print(f"\nWriting output: {OUTPUT_PGM}")
    write_pgm(OUTPUT_PGM, modified_image, max_val)
    
    print("\n✓ Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()