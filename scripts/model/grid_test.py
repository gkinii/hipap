import numpy as np

# Image parameters
img_size = 256
resolution = 0.05
origin_x, origin_y = -6.425, -6.425  # bottom-left world coordinates

# Create grid of map positions (optional, if you want full array of world coords)
x_coords = origin_x + np.arange(img_size) * resolution
y_coords = origin_y + np.arange(img_size) * resolution
grid_x, grid_y = np.meshgrid(x_coords, y_coords)

# World point to transform
world_x = 1.9480213268507125
world_y = -3.2209816040291046

# Convert world -> pixel indices
col = int((world_x - origin_x) / resolution)   # x → column
row = int((world_y - origin_y) / resolution)   # y → row

print(f"Pixel coordinates (row, col): ({row}, {col})")

# Make sure point lies within image bounds
if 0 <= row < img_size and 0 <= col < img_size:
    print("✅ Point lies inside the image.")
else:
    print("❌ Point is outside the image.")
