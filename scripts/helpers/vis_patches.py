import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============ CONFIGURATION ============
IMAGE_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/maps_6/grid_random_0.pgm"  # Path to your PGM image
PATCH_SIZE = 8            # Size of each patch (8x8, 16x16, etc.)
OUTPUT_PATH = "output.png"  # Output path (set to None to not save)
LINE_WIDTH = 2            # Width of grid lines in pixels
# =======================================

def draw_grid_on_image(image_path, patch_size, output_path=None, line_width=2):
    """
    Draw a red grid overlay on a PGM image to visualize patch divisions.
    
    Args:
        image_path: Path to input PGM image
        patch_size: Size of each patch (e.g., 8 for 8x8 patches)
        output_path: Path to save output image (optional)
        line_width: Width of grid lines in pixels
    """
    # Load the PGM image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert grayscale to RGB for colored grid lines
    if len(img_array.shape) == 2:
        img_rgb = np.stack([img_array] * 3, axis=-1)
    else:
        img_rgb = img_array.copy()
    
    height, width = img_array.shape[:2]
    
    # Draw vertical lines
    for x in range(0, width, patch_size):
        x_start = max(0, x - line_width // 2)
        x_end = min(width, x + line_width // 2)
        img_rgb[:, x_start:x_end, 0] = 255  # Red channel
        img_rgb[:, x_start:x_end, 1] = 0    # Green channel
        img_rgb[:, x_start:x_end, 2] = 0    # Blue channel
    
    # Draw horizontal lines
    for y in range(0, height, patch_size):
        y_start = max(0, y - line_width // 2)
        y_end = min(height, y + line_width // 2)
        img_rgb[y_start:y_end, :, 0] = 255  # Red channel
        img_rgb[y_start:y_end, :, 1] = 0    # Green channel
        img_rgb[y_start:y_end, :, 2] = 0    # Blue channel
    
    # Display the result
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.title(f'Grid Overlay (patch_size={patch_size}x{patch_size})')
    plt.axis('off')
    
    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    plt.suptitle(f'Image: {width}x{height} | Patches: {num_patches_w}x{num_patches_h} = {num_patches_w * num_patches_h} total', 
                 fontsize=10, y=0.98)
    
    # Save if output path is provided
    if output_path:
        output_img = Image.fromarray(img_rgb.astype(np.uint8))
        output_img.save(output_path)
        print(f"Saved output to: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return img_rgb

if __name__ == "__main__":
    
    draw_grid_on_image(IMAGE_PATH, PATCH_SIZE, OUTPUT_PATH, LINE_WIDTH)