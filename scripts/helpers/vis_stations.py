import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def load_ros_map(yaml_file):
    """Load ROS map and return image array, origin, resolution."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    image_path = config['image']
    if not os.path.isabs(image_path):
        yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
        image_path = os.path.join(yaml_dir, image_path)

    img = Image.open(image_path).convert("RGB")
    origin = config['origin'][:2]      # [ox, oy]
    resolution = config['resolution']

    return np.array(img), origin, resolution, img.size  # (H, W, 3), origin, res, (W, H)


# =============================================================================
#                               MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    # -------------------------- CONFIG --------------------------
    yaml_file = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_cell_new/maps_1/map_real_0.yaml"
    trajectory_csv = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_cell_new/paths_1.csv"

    stations = [
        {'x': -2.0,   'y': 4.5,   'color': '#FF3333', 'label': 'Tools'},
        {'x': -0.05,  'y': -0.8,  'color': '#3388FF', 'label': 'Assembly'},
    ]

    # -------------------------- LOAD MAP --------------------------
    map_array, origin, resolution, (img_width, img_height) = load_ros_map(yaml_file)
    ox, oy = origin

    # -------------------------- LOAD TRAJECTORY --------------------------
    df = pd.read_csv(trajectory_csv)
    target_x = df['x'].values
    target_y = df['y'].values

    # -------------------------- PLOT --------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Correct way to display ROS map:
    # - Use origin='lower' so (0,0) in pixel is bottom-left
    # - Set extent in world coordinates
    left   = ox
    right  = ox + img_width * resolution
    bottom = oy
    top    = oy + img_height * resolution

    ax.imshow(map_array, extent=[left, right, bottom, top], origin='upper')

    # Plot ground truth trajectory exactly as requested
    ax.plot(target_x, target_y, 'g-o',
            linewidth=2, markersize=6, alpha=0.85,
            markerfacecolor='limegreen', markeredgecolor='darkgreen',
            label='Ground Truth')

    # Optional: highlight start and end
    ax.plot(target_x[0], target_y[0], 'o', markersize=10,
            markerfacecolor='yellow', markeredgecolor='black', zorder=5)
    ax.plot(target_x[-1], target_y[-1], 'X', markersize=12,
            markerfacecolor='red', markeredgecolor='black', zorder=5)

    # Plot stations
    for station in stations:
        ax.plot(station['x'], station['y'], 'o', markersize=14,
                markerfacecolor=station['color'], markeredgecolor='black',
                markeredgewidth=2, zorder=6)
        if station['label']=='Tools':
            ax.text(station['x'] - 1, station['y'] - 0.5, station['label'],
                    fontsize=14, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=4))
        else:
            ax.text(station['x'] - 0.8, station['y'] + 0.8, station['label'],
                    fontsize=14, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=4))


            # Optional: highlight start and end
        ax.plot(target_x[0], target_y[0], 'o', markersize=10,
                markerfacecolor='yellow', markeredgecolor='black', zorder=5)
        ax.text(target_x[0] + 0.2, target_y[0] + 0.2, "Start",
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=4))

        ax.plot(target_x[-1], target_y[-1], 'X', markersize=12,
                markerfacecolor='red', markeredgecolor='black', zorder=5)
        ax.text(target_x[-1] + 0.2, target_y[-1] + 0.2, "End",
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=4))


    # Styling
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)
    ax.set_title("Human Ground Truth Trajectory", fontsize=18, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    output_path = "/home/gkini/human_trajectory_corrected.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correctly aligned map saved to:\n    {output_path}")

    plt.show()