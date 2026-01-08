"""
pad_occupancy_grid.py - Pad an occupancy grid to 640x640 with gray borders

Takes an existing PGM/YAML map and pads it to target size with gray (unknown) areas.
"""

import numpy as np
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_YAML = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_320/maps/map_4.yaml"   # Input map YAML file
OUTPUT_PGM = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_320/maps/map_4.pgm"      # Output padded PGM file
OUTPUT_YAML = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_320/maps/map_4.yaml"    # Output padded YAML file

TARGET_SIZE = 320                   # Target grid size (square)
PADDING_VALUE = 192                 # Gray value for padding (unknown space)

# ═══════════════════════════════════════════════════════════════════════════════

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


def write_pgm(occupancy_grid, filename):
    """Write occupancy grid as PGM file."""
    height, width = occupancy_grid.shape
    
    # Create parent directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(occupancy_grid.tobytes())
    
    print(f"Saved padded PGM to: {filename}")


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
        'free_thresh': map_data.get('free_thresh', 0.196),
        'mode': map_data.get('mode', 'trinary')
    }

def write_map_yaml(yaml_file, pgm_file, resolution, origin, mode='trinary',
                   negate=0, occupied_thresh=0.65, free_thresh=0.196):
    """Write map YAML file."""
    yaml_content = f"""image: {pgm_file}\nmode: {mode}\nresolution: {resolution:.3f}\norigin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]\nnegate: {negate}\noccupied_thresh: {occupied_thresh}\nfree_thresh: {free_thresh}
    """
    
    # Create parent directory if it doesn't exist
    Path(yaml_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Saved padded YAML to: {yaml_file}")


def pad_occupancy_grid(input_yaml, output_pgm, output_yaml, 
                       target_size=640, padding_value=128):
    """
    Pad an occupancy grid to target size with gray borders.
    
    Parameters:
    - input_yaml: Input YAML file path
    - output_pgm: Output PGM file path
    - output_yaml: Output YAML file path
    - target_size: Target grid size (square)
    - padding_value: Value for padding pixels (128 = gray/unknown)
    """
    print(f"=== PADDING OCCUPANCY GRID ===")
    print(f"Input: {input_yaml}")
    print(f"Target size: {target_size}x{target_size}")
    
    # Read input YAML
    map_info = read_map_yaml(input_yaml)
    yaml_path = Path(input_yaml)
    input_pgm = yaml_path.parent / map_info['image']
    
    # Read input PGM
    original_grid = read_pgm(input_pgm)
    orig_height, orig_width = original_grid.shape
    
    print(f"\nOriginal size: {orig_width}x{orig_height}")
    print(f"Resolution: {map_info['resolution']} m/pixel")
    print(f"Original origin: {map_info['origin']}")
    
    # Check if padding is needed
    if orig_width > target_size or orig_height > target_size:
        print(f"\nError: Original map ({orig_width}x{orig_height}) is larger than target ({target_size}x{target_size})")
        return
    
    if orig_width == target_size and orig_height == target_size:
        print(f"\nMap is already {target_size}x{target_size}, no padding needed!")
        return
    
    # Calculate padding
    pad_x_total = target_size - orig_width
    pad_y_total = target_size - orig_height
    
    # Distribute padding evenly
    pad_x_left = pad_x_total // 2
    pad_x_right = pad_x_total - pad_x_left
    pad_y_bottom = pad_y_total // 2
    pad_y_top = pad_y_total - pad_y_bottom
    
    print(f"\nPadding:")
    print(f"  Left: {pad_x_left}px, Right: {pad_x_right}px")
    print(f"  Bottom: {pad_y_bottom}px, Top: {pad_y_top}px")
    
    # Create padded grid with gray background
    padded_grid = np.full((target_size, target_size), padding_value, dtype=np.uint8)
    
    # Copy original grid into center of padded grid
    start_row = pad_y_bottom
    end_row = start_row + orig_height
    start_col = pad_x_left
    end_col = start_col + orig_width
    
    padded_grid[start_row:end_row, start_col:end_col] = original_grid
    
    # Calculate new origin
    resolution = map_info['resolution']
    orig_origin_x, orig_origin_y, orig_origin_theta = map_info['origin']
    
    # The new origin shifts back by the padding amount
    new_origin_x = orig_origin_x - pad_x_left * resolution
    new_origin_y = orig_origin_y - pad_y_bottom * resolution
    new_origin = [new_origin_x, new_origin_y, orig_origin_theta]
    
    print(f"\nNew origin: [{new_origin_x:.3f}, {new_origin_y:.3f}, {orig_origin_theta:.3f}]")
    
    # Write padded PGM
    write_pgm(padded_grid, output_pgm)
    
    # Write new YAML
    write_map_yaml(
        output_yaml,
        Path(output_pgm).name,
        resolution,
        new_origin,
        mode=map_info['mode'],
        negate=map_info['negate'],
        occupied_thresh=map_info['occupied_thresh'],
        free_thresh=map_info['free_thresh']
    )
    
    print(f"\n✓ Padded map saved:")
    print(f"  PGM: {output_pgm}")
    print(f"  YAML: {output_yaml}")
    print(f"  Final size: {target_size}x{target_size}")


def main():
    pad_occupancy_grid(
        INPUT_YAML,
        OUTPUT_PGM,
        OUTPUT_YAML,
        target_size=TARGET_SIZE,
        padding_value=PADDING_VALUE
    )


if __name__ == '__main__':
    main()