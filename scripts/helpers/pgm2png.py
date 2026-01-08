from PIL import Image
import os

# === Define paths here ===
input_path = "/home/gkini/Human-Traj-Prediction/scripts/dataset/synthetic/256/grid_random_0.pgm"
output_path = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/maps_png/grid_random_0.png"
# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Convert PGM → PNG
with Image.open(input_path) as img:
    img.convert("RGB").save(output_path)

print(f"✅ Converted: {input_path} → {output_path}")