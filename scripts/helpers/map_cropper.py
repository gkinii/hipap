#!/usr/bin/env python3
"""
PGM Map Cropper - Crops map.pgm files with configurable modes
Supports independent cropping modes for height and width: 'center', 'top', or 'bottom'
"""

import yaml
from PIL import Image
import os

# ==================== CONFIGURATION ====================
# Input files
PGM_PATH = '/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/new_cell/maps_1/grid_real_0.pgm'
YAML_PATH = '/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/new_cell/maps_1/map_real_0.yaml'

# Output files
OUTPUT_PGM = '/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/new_cell/maps_1/grid_real_0_cropped.pgm'
OUTPUT_YAML = '/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/new_cell/maps_1/map_real_0_cropped.yaml'

# Cropping modes
WIDTH_MODE = 'left'   # 'left', 'center', or 'right'
HEIGHT_MODE = 'top'     # 'top', 'center', or 'bottom'

# Amount to crop (remove) in pixels
WIDTH_CROP_SIZE = 144   # pixels to remove
HEIGHT_CROP_SIZE = 144  # pixels to remove  
# =======================================================
def crop_image(img, width_mode, height_mode, width_crop_size, height_crop_size):
    """
    Crop image based on specified modes
    
    Args:
        img: PIL Image object
        width_mode: 'left', 'center', or 'right' - indicates which side to remove FROM
        height_mode: 'top', 'center', or 'bottom' - indicates which side to remove FROM
        width_crop_size: Amount to crop (remove) in pixels
        height_crop_size: Amount to crop (remove) in pixels
    
    Returns:
        Cropped PIL Image, left offset, and bottom offset (for origin adjustment)
    
    Behavior:
        - height_mode='top': removes height_crop_size pixels FROM the top
        - height_mode='bottom': removes height_crop_size pixels FROM the bottom
        - height_mode='center': removes height_crop_size/2 pixels from both top and bottom
        - width_mode='left': removes width_crop_size pixels FROM the left
        - width_mode='right': removes width_crop_size pixels FROM the right
        - width_mode='center': removes width_crop_size/2 pixels from both left and right
    """
    orig_width, orig_height = img.size
    
    # Calculate width cropping coordinates
    if width_crop_size >= orig_width:
        raise ValueError(f"width_crop_size ({width_crop_size}) must be smaller than original width ({orig_width})")
    
    if width_mode == 'center':
        left = width_crop_size // 2
        right = orig_width - (width_crop_size - width_crop_size // 2)
    elif width_mode == 'left':
        # Remove FROM the left (keep right side)
        left = width_crop_size
        right = orig_width
    elif width_mode == 'right':
        # Remove FROM the right (keep left side)
        left = 0
        right = orig_width - width_crop_size
    else:
        raise ValueError(f"Invalid width_mode: {width_mode}. Must be 'left', 'center', or 'right'")
    
    # Calculate height cropping coordinates
    if height_crop_size >= orig_height:
        raise ValueError(f"height_crop_size ({height_crop_size}) must be smaller than original height ({orig_height})")
    
    if height_mode == 'center':
        top = height_crop_size // 2
        bottom = orig_height - (height_crop_size - height_crop_size // 2)
    elif height_mode == 'top':
        # Remove FROM the top (keep bottom)
        top = height_crop_size
        bottom = orig_height
    elif height_mode == 'bottom':
        # Remove FROM the bottom (keep top)
        top = 0
        bottom = orig_height - height_crop_size
    else:
        raise ValueError(f"Invalid height_mode: {height_mode}. Must be 'top', 'center', or 'bottom'")
    
    # Calculate how many pixels were removed from left and bottom (for origin adjustment)
    left_offset = left
    bottom_offset = orig_height - bottom
    
    # Crop the image
    cropped = img.crop((left, top, right, bottom))
    
    return cropped, left_offset, bottom_offset

def main():
    # Check if input files exist
    if not os.path.exists(PGM_PATH):
        raise FileNotFoundError(f"PGM file not found: {PGM_PATH}")
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML file not found: {YAML_PATH}")
    
    # Load the PGM image
    print(f"Loading {PGM_PATH}...")
    img = Image.open(PGM_PATH)
    print(f"Original size: {img.size[0]}x{img.size[1]}")
    
    # Crop the image
    print(f"Cropping with width_mode={WIDTH_MODE}, height_mode={HEIGHT_MODE}")
    print(f"Removing: {WIDTH_CROP_SIZE}px width, {HEIGHT_CROP_SIZE}px height")
    cropped_img, left_offset, bottom_offset = crop_image(
        img, 
        WIDTH_MODE, 
        HEIGHT_MODE, 
        WIDTH_CROP_SIZE, 
        HEIGHT_CROP_SIZE
    )
    print(f"Result size: {cropped_img.size[0]}x{cropped_img.size[1]}")
    print(f"Pixels removed from left: {left_offset}, from bottom: {bottom_offset}")
    
    # Save the cropped image
    cropped_img.save(OUTPUT_PGM)
    print(f"Saved cropped image to {OUTPUT_PGM}")
    
    # Update and save YAML
    with open(YAML_PATH, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    resolution = yaml_data.get('resolution', 0.05)  # default resolution
    
    if 'origin' in yaml_data:
        old_origin = yaml_data['origin'].copy()
        
        # Origin changes based on pixels removed from left and bottom
        x_adjustment = left_offset * resolution
        y_adjustment = bottom_offset * resolution
        
        yaml_data['origin'][0] += x_adjustment
        yaml_data['origin'][1] += y_adjustment
        
        print(f"\nOrigin transformation:")
        print(f"  Resolution: {resolution} m/pixel")
        print(f"  Old origin: [{old_origin[0]:.3f}, {old_origin[1]:.3f}, {old_origin[2]}]")
        print(f"  X: {old_origin[0]:.3f} + ({left_offset}px × {resolution}) = {yaml_data['origin'][0]:.3f}")
        print(f"  Y: {old_origin[1]:.3f} + ({bottom_offset}px × {resolution}) = {yaml_data['origin'][1]:.3f}")
        print(f"  New origin: [{yaml_data['origin'][0]:.3f}, {yaml_data['origin'][1]:.3f}, {yaml_data['origin'][2]}]")
    
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"Saved updated YAML to {OUTPUT_YAML}")
    
    print("\nCropping complete!")

if __name__ == '__main__':
    main()