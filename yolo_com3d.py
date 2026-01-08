#!/usr/bin/env python3
import os
import csv
import math
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ========================== USER CONFIG ==========================
COLOR_IMAGE_DIR = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/cell_real/images/infra"
DEPTH_IMAGE_DIR = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/cell_real/images/depth"
WEIGHTS         = "yolo11m-pose.pt"
IMG_SIZE        = 640
CONF_THRES      = 0.5
DEVICE          = "0"
OUTPUT_DIR      = "runs/predict/pose_fused2"

# Camera intrinsics - UPDATED
FX = 433.3813781738281
FY = 433.3813781738281
CX = 431.5794372558594
CY = 238.99366760253906

# Camera position in world frame (meters) - BASE position before offset
CAMERA_POS_WORLD_FRAME_BASE = np.array([0.96618, 3.4708, 0.2825])

# Position offset to apply to camera (in meters, world frame coordinates)
# Positive X = move EAST/RIGHT, Negative X = move WEST/LEFT
# Positive Y = move NORTH/FORWARD, Negative Y = move SOUTH/BACK  
# Positive Z = move UP, Negative Z = move DOWN
OFFSET_POSITION = np.array([0.0, 1.5, 0.0])  # Default: shift 0.5m back (negative Y)

# Quaternion format: 'xyzw' or 'wxyz'
QUATERNION_TYPE = 'wxyz'  # Change to 'wxyz' to switch format

# Confidence weighting power (higher = more influence from confident keypoints)
# 1.0 = linear weighting (default)
# 2.0 = square confidence (emphasizes high-confidence keypoints more)
# 0.5 = sqrt confidence (reduces confidence influence)
# 0.0 = ignore confidence (pure anatomical weights)
CONFIDENCE_POWER = 1.0

# Rotation offset to apply to camera orientation (in degrees)
# Positive = rotate LEFT, Negative = rotate RIGHT
# This rotates around the vertical axis (Z-axis by default)
OFFSET_ROTATION = -15  # Try: -10, -5, 0, 5, 10, 15, etc.
ROTATION_AXIS = 'z'    # 'x', 'y', or 'z' - which axis is vertical in your world frame

# Camera orientation as quaternion (BEFORE applying offset rotation)
# If QUATERNION_TYPE='xyzw': [x, y, z, w]
# If QUATERNION_TYPE='wxyz': [w, x, y, z]
CAMERA_QUAT_BASE = np.array([0.0, 0.0, -0.77312, 0.63426])
# ===============================================================

# COCO-17 keypoint order
COCO_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Approximate body segment mass weights for anatomically accurate COM
KEYPOINT_WEIGHTS = np.array([
    0.07,  # nose (head)
    0.07, 0.07,  # eyes (head)
    0.07, 0.07,  # ears (head)
    0.15, 0.15,  # shoulders (torso)
    0.03, 0.03,  # elbows (arms)
    0.015, 0.015,  # wrists (hands)
    0.12, 0.12,  # hips (pelvis) - most important for COM!
    0.06, 0.06,  # knees (legs)
    0.03, 0.03   # ankles (feet)
])

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: quaternion as [x, y, z, w] if QUATERNION_TYPE='xyzw'
           or [w, x, y, z] if QUATERNION_TYPE='wxyz'
    
    Returns:
        3x3 rotation matrix
    """
    if QUATERNION_TYPE == 'xyzw':
        x, y, z, w = q
    else:  # wxyz
        w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

def quaternion_inverse(q):
    """
    Compute quaternion inverse (conjugate for unit quaternions).
    
    Args:
        q: quaternion in format specified by QUATERNION_TYPE
    
    Returns:
        Inverse quaternion in same format
    """
    if QUATERNION_TYPE == 'xyzw':
        x, y, z, w = q
        return np.array([-x, -y, -z, w])
    else:  # wxyz
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q_result = q1 * q2
    
    Args:
        q1, q2: quaternions in format specified by QUATERNION_TYPE
    
    Returns:
        Product quaternion in same format
    """
    if QUATERNION_TYPE == 'wxyz':
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:  # xyzw
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    if QUATERNION_TYPE == 'wxyz':
        return np.array([w, x, y, z])
    else:
        return np.array([x, y, z, w])

def rotation_around_axis(axis, angle_degrees):
    """
    Create quaternion for rotation around an axis.
    
    Args:
        axis: 'x', 'y', or 'z'
        angle_degrees: rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Quaternion representing the rotation in format specified by QUATERNION_TYPE
    """
    angle_rad = np.deg2rad(angle_degrees)
    half_angle = angle_rad / 2.0
    
    cos_half = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    
    if axis.lower() == 'x':
        quat_wxyz = np.array([cos_half, sin_half, 0.0, 0.0])
    elif axis.lower() == 'y':
        quat_wxyz = np.array([cos_half, 0.0, sin_half, 0.0])
    elif axis.lower() == 'z':
        quat_wxyz = np.array([cos_half, 0.0, 0.0, sin_half])
    else:
        raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")
    
    if QUATERNION_TYPE == 'xyzw':
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return quat_wxyz

def apply_rotation_offset(base_quat, offset_degrees, axis='z'):
    """
    Apply rotation offset to base quaternion.
    
    Args:
        base_quat: Base camera quaternion
        offset_degrees: Rotation offset in degrees (positive = left, negative = right)
        axis: Axis to rotate around ('x', 'y', or 'z')
    
    Returns:
        New quaternion with offset applied
    """
    if abs(offset_degrees) < 1e-6:
        return base_quat
    
    # Create rotation quaternion
    offset_quat = rotation_around_axis(axis, offset_degrees)
    
    # Compose: new_orientation = offset_rotation * base_orientation
    new_quat = quaternion_multiply(offset_quat, base_quat)
    
    # Normalize
    norm = np.linalg.norm(new_quat)
    new_quat = new_quat / norm
    
    return new_quat

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_depth_as_meters(path):
    """Load depth image, return float32 meters and scale used."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image at {path}")
    if depth.dtype == np.uint16:
        scale = 0.001  # mm -> m
        depth_m = depth.astype(np.float32) * scale
    elif depth.dtype == np.float32 or depth.dtype == np.float64:
        scale = 1.0  # already meters
        depth_m = depth.astype(np.float32)
    else:
        if depth.max() > 100:
            scale = 0.001
            depth_m = depth.astype(np.float32) * scale
        else:
            scale = 1.0
            depth_m = depth.astype(np.float32)
    return depth_m, scale

def bilinear_depth(depth_m, x, y):
    """Sample depth with bilinear interpolation at float (x, y)."""
    h, w = depth_m.shape[:2]
    if x < 0 or y < 0 or x > (w - 1) or y > (h - 1):
        return 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0

    d00 = depth_m[y0, x0]
    d10 = depth_m[y0, x1]
    d01 = depth_m[y1, x0]
    d11 = depth_m[y1, x1]

    vals = np.array([d00, d10, d01, d11], dtype=np.float32)
    if np.all(vals <= 0):
        return 0.0

    weights = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], dtype=np.float32)
    mask = vals > 0
    if not np.any(mask):
        return 0.0
    wsum = weights[mask].sum()
    if wsum <= 1e-8:
        return 0.0
    return float((vals[mask] * weights[mask]).sum() / wsum)

def fallback_depth(depth_m, x, y, window=3):
    """Median of a small valid neighborhood if bilinear fails."""
    h, w = depth_m.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    r = window
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))

def depth_at(depth_m, x, y):
    d = bilinear_depth(depth_m, x, y)
    if d <= 0:
        d = fallback_depth(depth_m, x, y, window=2)
    return d

def deproject(x, y, Z, fx, fy, cx, cy):
    """Back-project pixel (x,y) with depth Z (meters)."""
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return X, Y, Z

def transform_to_world_frame(com_camera, camera_pos, camera_rotation):
    """
    Transform COM from camera frame to world frame using full rigid body transformation.
    
    Args:
        com_camera: 3D point in camera frame
        camera_pos: Camera position in world frame
        camera_rotation: 3x3 rotation matrix from camera to world
    
    Returns:
        3D point in world frame
    """
    if np.any(np.isnan(com_camera)):
        return np.array([np.nan, np.nan, np.nan])
    
    # Full rigid transformation: P_world = R @ P_camera + T
    return camera_rotation @ com_camera + camera_pos

def colorize_depth(depth_m, clip_min=0.2, clip_max=5.0):
    """Make depth visualization in 8-bit."""
    d = depth_m.copy()
    d = np.clip(d, clip_min, clip_max)
    d_norm = ((d - clip_min) / (clip_max - clip_min) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

def draw_com_on_depth(depth_vis, persons_com_2d):
    """Overlay center of mass on colorized depth."""
    out = depth_vis.copy()
    h, w = out.shape[:2]
    palette = [
        (255, 255, 255), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 128, 255)
    ]
    for pid, com2d in enumerate(persons_com_2d):
        if com2d is not None:
            x, y = com2d
            color = palette[pid % len(palette)]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (int(round(x)), int(round(y))), 5, color, -1, cv2.LINE_AA)
                cv2.putText(out, f"P{pid+1} COM", (int(round(x)) + 10, int(round(y))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def save_com_series_csv(out_csv_path, com_series):
    """Save series of COM positions to CSV."""
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "person_id", "X_m", "Y_m", "Z_m"]
        writer.writerow(header)
        for frame_idx, coms in enumerate(com_series):
            for pid, com in enumerate(coms, start=1):
                if not np.any(np.isnan(com)):
                    writer.writerow([frame_idx, pid, f"{com[0]:.6f}", f"{com[1]:.6f}", f"{com[2]:.6f}"])

def save_com_series_camera_frame_csv(out_csv_path, com_series_camera):
    """Save series of COM positions in camera frame to CSV."""
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "person_id", "X_cam_m", "Y_cam_m", "Z_cam_m"]
        writer.writerow(header)
        for frame_idx, coms in enumerate(com_series_camera):
            for pid, com in enumerate(coms, start=1):
                if not np.any(np.isnan(com)):
                    writer.writerow([frame_idx, pid, f"{com[0]:.6f}", f"{com[1]:.6f}", f"{com[2]:.6f}"])

def visualize_2d_com_series(com_series):
    """Display 2D visualization of COM trajectories in X-Y plane."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute dynamic axis limits
    all_coms = []
    for coms in com_series:
        valid_coms = [com for com in coms if not np.any(np.isnan(com))]
        all_coms.extend(valid_coms)
    all_points = np.stack(all_coms) if all_coms else np.empty((0, 3))
    if len(all_points) == 0:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
    else:
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        max_range = max(x_max - x_min, y_max - y_min) * 0.5
        margin = max_range * 0.2  # 20% margin
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin

    # Set axis labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Colors for different people
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    # Plot COM trajectories
    for pid in range(max(len(coms) for coms in com_series)):
        x_traj, y_traj = [], []
        for coms in com_series:
            if pid < len(coms) and not np.any(np.isnan(coms[pid])):
                x_traj.append(coms[pid][0])
                y_traj.append(coms[pid][1])
        if x_traj:
            color = colors[pid % len(colors)]
            ax.plot(x_traj, y_traj, c=color, linewidth=2, label=f'Person {pid+1} COM')
            ax.scatter(x_traj, y_traj, c=color, s=50)
            # Label the last point
            ax.text(x_traj[-1], y_traj[-1], f'P{pid+1}', color=color, fontsize=8)

    if all_coms:
        ax.legend()

    return fig

def main():
    ensure_dir(OUTPUT_DIR)

    # Print configuration
    print(f"{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Quaternion format: {QUATERNION_TYPE}")
    print(f"Base camera quaternion: {CAMERA_QUAT_BASE}")
    print(f"Rotation offset: {OFFSET_ROTATION}° around {ROTATION_AXIS.upper()}-axis")
    if abs(OFFSET_ROTATION) > 1e-6:
        direction = "LEFT" if OFFSET_ROTATION > 0 else "RIGHT"
        print(f"  → Rotating camera {abs(OFFSET_ROTATION):.1f}° {direction}")
    else:
        print(f"  → No rotation offset applied")
    
    # Apply rotation offset to get final camera quaternion
    CAMERA_QUAT = apply_rotation_offset(CAMERA_QUAT_BASE, OFFSET_ROTATION, ROTATION_AXIS)
    print(f"Final camera quaternion: {CAMERA_QUAT}")
    
    # Apply position offset to get final camera position
    print(f"\nBase camera position: [{CAMERA_POS_WORLD_FRAME_BASE[0]:.4f}, {CAMERA_POS_WORLD_FRAME_BASE[1]:.4f}, {CAMERA_POS_WORLD_FRAME_BASE[2]:.4f}]")
    print(f"Position offset: [{OFFSET_POSITION[0]:+.4f}, {OFFSET_POSITION[1]:+.4f}, {OFFSET_POSITION[2]:+.4f}]")
    CAMERA_POS_WORLD_FRAME = CAMERA_POS_WORLD_FRAME_BASE + OFFSET_POSITION
    print(f"Final camera position: [{CAMERA_POS_WORLD_FRAME[0]:.4f}, {CAMERA_POS_WORLD_FRAME[1]:.4f}, {CAMERA_POS_WORLD_FRAME[2]:.4f}]")
    
    print(f"\nConfidence weighting power: {CONFIDENCE_POWER}")
    print(f"\nCamera Intrinsics:")
    print(f"  FX = {FX:.6f}")
    print(f"  FY = {FY:.6f}")
    print(f"  CX = {CX:.6f}")
    print(f"  CY = {CY:.6f}")
    print(f"\nCamera Extrinsics (final):")
    print(f"  Position (world frame): [{CAMERA_POS_WORLD_FRAME[0]:.4f}, {CAMERA_POS_WORLD_FRAME[1]:.4f}, {CAMERA_POS_WORLD_FRAME[2]:.4f}]")
    
    # Convert quaternion to rotation matrix
    camera_rotation = quaternion_to_rotation_matrix(CAMERA_QUAT)
    print(f"\nCamera Rotation Matrix (camera → world):")
    print(f"  [{camera_rotation[0,0]:+.6f}, {camera_rotation[0,1]:+.6f}, {camera_rotation[0,2]:+.6f}]")
    print(f"  [{camera_rotation[1,0]:+.6f}, {camera_rotation[1,1]:+.6f}, {camera_rotation[1,2]:+.6f}]")
    print(f"  [{camera_rotation[2,0]:+.6f}, {camera_rotation[2,1]:+.6f}, {camera_rotation[2,2]:+.6f}]")
    print(f"{'='*70}\n")

    # Load image files
    color_files = sorted([f for f in os.listdir(COLOR_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([f for f in os.listdir(DEPTH_IMAGE_DIR) if f.endswith('.png')])
    
    if not color_files or not depth_files:
        print("No images found in the specified directories.")
        return

    # Ensure matching number of files
    if len(color_files) != len(depth_files):
        print(f"Warning: Mismatch in number of files (color: {len(color_files)}, depth: {len(depth_files)})")
        return

    model = YOLO(WEIGHTS)
    com_series = []
    com_series_camera = []  # Store camera frame coordinates for debugging

    for idx, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        color_path = os.path.join(COLOR_IMAGE_DIR, color_file)
        depth_path = os.path.join(DEPTH_IMAGE_DIR, depth_file)

        # Load images
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color is None:
            print(f"Could not read color image at {color_path}")
            continue
        depth_m, depth_scale = load_depth_as_meters(depth_path)

        h, w = color.shape[:2]
        depth_h, depth_w = depth_m.shape[:2]
        
        # Print detailed image information for first frame
        if idx == 0:
            print(f"\n{'='*70}")
            print(f"IMAGE DIMENSIONS CHECK (Frame 0)")
            print(f"{'='*70}")
            print(f"Color image:")
            print(f"  File: {color_file}")
            print(f"  Dimensions: {w} x {h} pixels (width x height)")
            print(f"  Shape: {color.shape}")
            print(f"\nDepth image:")
            print(f"  File: {depth_file}")
            print(f"  Dimensions: {depth_w} x {depth_h} pixels (width x height)")
            print(f"  Shape: {depth_m.shape}")
            print(f"  Scale factor: {depth_scale} (to convert to meters)")
            print(f"  Data type: {depth_m.dtype}")
            print(f"\nCamera Intrinsics Check:")
            print(f"  CX = {CX:.2f} (should be ~{w/2:.2f} for centered principal point)")
            print(f"  CY = {CY:.2f} (should be ~{h/2:.2f} for centered principal point)")
            print(f"  Principal point offset from center: ({CX - w/2:.2f}, {CY - h/2:.2f}) pixels")
            print(f"\nYOLO Input Size: {IMG_SIZE}")
            print(f"  Note: YOLO resizes input to {IMG_SIZE}x{IMG_SIZE} internally for detection")
            print(f"  Keypoints are returned in ORIGINAL image coordinates ({w}x{h})")
            print(f"{'='*70}\n")
        
        if depth_m.shape[:2] != (h, w):
            print(f"\n⚠️  WARNING - Frame {idx}: Dimension mismatch!")
            print(f"  Depth image size {depth_w}x{depth_h} != Color {w}x{h}")
            print(f"  This will cause misalignment between RGB keypoints and depth values!")
            continue

        # Run YOLO-pose
        results = model.predict(
            source=color_path,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=DEVICE,
            save=False,
            save_dir=OUTPUT_DIR,
            show=False
        )
        if not results:
            print(f"No results for {color_file}")
            com_series.append([])
            com_series_camera.append([])
            continue

        kps = results[0].keypoints
        persons_xy = [xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy) for xy in kps.xy]
        persons_conf = [c.cpu().numpy() if hasattr(c, "cpu") else np.asarray(c) for c in getattr(kps, "conf", [None]*len(persons_xy))]
        
        # Check keypoint coordinates for first frame
        if idx == 0 and len(persons_xy) > 0:
            print(f"KEYPOINT COORDINATES CHECK (Frame 0, Person 1):")
            print(f"  Image dimensions: {w}x{h}")
            xy_sample = persons_xy[0]
            conf_sample = persons_conf[0] if persons_conf else None
            print(f"  Sample keypoints (name, x, y, conf):")
            for j, (x, y) in enumerate(xy_sample[:5]):  # First 5 keypoints
                cval = float(conf_sample[j]) if conf_sample is not None else 1.0
                in_bounds = (0 <= x < w and 0 <= y < h)
                status = "✓" if in_bounds else "⚠️"
                print(f"    {status} {COCO_KPTS[j]:15s}: ({x:7.2f}, {y:7.2f}) conf={cval:.3f}")
            print(f"  All keypoints within image bounds: {all(0 <= x < w and 0 <= y < h for x, y in xy_sample)}")
            print()

        # Compute 3D keypoints and COM
        coms_3d = []
        coms_3d_camera = []  # Store camera frame for debugging
        coms_2d = []
        
        # For first frame, print detailed depth sampling info
        debug_depth = (idx == 0)
        
        for pid, xy in enumerate(persons_xy):
            conf = persons_conf[pid] if persons_conf else None
            pts3d = np.zeros((xy.shape[0], 3), dtype=np.float32)
            
            if debug_depth and pid == 0:
                print(f"DEPTH SAMPLING CHECK (Frame 0, Person 1):")
                print(f"  Keypoint -> Depth value -> 3D position")
            
            for j, (x, y) in enumerate(xy):
                cval = float(conf[j]) if conf is not None else 1.0
                if cval < 0.2:
                    pts3d[j] = (np.nan, np.nan, np.nan)
                    if debug_depth and pid == 0 and j < 5:
                        print(f"    {COCO_KPTS[j]:15s}: confidence too low ({cval:.3f})")
                    continue
                Z = depth_at(depth_m, float(x), float(y))
                if Z <= 0 or math.isfinite(Z) is False:
                    pts3d[j] = (np.nan, np.nan, np.nan)
                    if debug_depth and pid == 0 and j < 5:
                        print(f"    {COCO_KPTS[j]:15s}: invalid depth (Z={Z:.3f})")
                    continue
                X, Y, Z_final = deproject(float(x), float(y), Z, FX, FY, CX, CY)
                pts3d[j] = (X, Y, Z_final)
                
                if debug_depth and pid == 0 and j < 5:
                    print(f"    {COCO_KPTS[j]:15s}: pixel({x:6.1f},{y:6.1f}) -> depth={Z:.3f}m -> cam({X:+.3f},{Y:+.3f},{Z_final:+.3f})")
            
            if debug_depth and pid == 0:
                print()

            # Compute weighted COM in camera frame
            valid = ~np.any(np.isnan(pts3d), axis=1)
            if np.sum(valid) > 0:
                # Combine anatomical weights with confidence scores
                # Higher confidence keypoints will have more influence
                anatomical_weights = KEYPOINT_WEIGHTS[valid]
                confidence_scores = conf[valid] if conf is not None else np.ones(np.sum(valid))
                
                # Apply confidence power to control influence
                # CONFIDENCE_POWER = 0.0: ignore confidence (pure anatomy)
                # CONFIDENCE_POWER = 1.0: linear confidence weighting
                # CONFIDENCE_POWER > 1.0: emphasize high-confidence keypoints more
                if CONFIDENCE_POWER != 0.0:
                    confidence_weights = np.power(confidence_scores, CONFIDENCE_POWER)
                else:
                    confidence_weights = np.ones_like(confidence_scores)
                
                # Final weight = anatomical_weight * confidence_weight
                # This gives more weight to confident keypoints while respecting anatomy
                combined_weights = anatomical_weights * confidence_weights
                
                # Normalize weights to sum to 1 for proper averaging
                combined_weights = combined_weights / combined_weights.sum()
                
                com_3d_camera = np.average(pts3d[valid], axis=0, weights=combined_weights)
                
                # Also use confidence-weighted average for 2D visualization
                com_2d = np.average(xy[valid], axis=0, weights=combined_weights)
            else:
                com_3d_camera = np.array([np.nan, np.nan, np.nan])
                com_2d = None
            
            # Store camera frame COM
            coms_3d_camera.append(com_3d_camera.copy())
            
            # Transform COM to world frame (with rotation!)
            com_3d_world = transform_to_world_frame(com_3d_camera, CAMERA_POS_WORLD_FRAME, camera_rotation)
            coms_3d.append(com_3d_world)
            coms_2d.append(com_2d)

        com_series.append(coms_3d)
        com_series_camera.append(coms_3d_camera)  # Store camera frame

        # Print COM positions - BOTH camera frame and world frame
        print(f"\n{'='*70}")
        print(f"Frame {idx} ({color_file})")
        print(f"{'='*70}")
        for pid, (com_cam, com_world) in enumerate(zip(coms_3d_camera, coms_3d), start=1):
            if not np.any(np.isnan(com_world)):
                print(f"Person {pid}:")
                print(f"  Camera Frame: X={com_cam[0]:+.4f}m, Y={com_cam[1]:+.4f}m, Z={com_cam[2]:+.4f}m")
                print(f"  World Frame:  X={com_world[0]:+.4f}m, Y={com_world[1]:+.4f}m, Z={com_world[2]:+.4f}m")
                # Count valid keypoints
                pts3d_temp = np.zeros((persons_xy[pid-1].shape[0], 3), dtype=np.float32)
                conf_temp = persons_conf[pid-1] if persons_conf else None
                for j, (x, y) in enumerate(persons_xy[pid-1]):
                    cval = float(conf_temp[j]) if conf_temp is not None else 1.0
                    if cval < 0.2:
                        pts3d_temp[j] = (np.nan, np.nan, np.nan)
                        continue
                    Z = depth_at(depth_m, float(x), float(y))
                    if Z <= 0 or math.isfinite(Z) is False:
                        pts3d_temp[j] = (np.nan, np.nan, np.nan)
                        continue
                    X, Y, Z = deproject(float(x), float(y), Z, FX, FY, CX, CY)
                    pts3d_temp[j] = (X, Y, Z)
                valid_count = np.sum(~np.any(np.isnan(pts3d_temp), axis=1))
                print(f"  Valid keypoints: {valid_count}/17")
            else:
                print(f"Person {pid}: No valid keypoints")

        # Save depth visualization with COM
        depth_vis = colorize_depth(depth_m, clip_min=0.2, clip_max=5.0)
        overlay = draw_com_on_depth(depth_vis, coms_2d)
        overlay_path = os.path.join(OUTPUT_DIR, f"depth_with_com_{idx:04d}.png")
        cv2.imwrite(overlay_path, overlay)

    # Save COM series to CSV
    csv_path = os.path.join(OUTPUT_DIR, "com_series_world_frame.csv")
    save_com_series_csv(csv_path, com_series)
    print(f"\n✓ Saved COM series (world frame): {csv_path}")
    
    # Save camera frame CSV for debugging
    csv_camera_path = os.path.join(OUTPUT_DIR, "com_series_camera_frame.csv")
    save_com_series_camera_frame_csv(csv_camera_path, com_series_camera)
    print(f"✓ Saved COM series (camera frame): {csv_camera_path}")

    # Save text dump
    txt_path = os.path.join(OUTPUT_DIR, "com_series_world_frame.txt")
    with open(txt_path, "w") as f:
        f.write(f"Coordinates in world reference frame (meters)\n")
        f.write(f"Quaternion format: {QUATERNION_TYPE}\n")
        f.write(f"Base quaternion: {CAMERA_QUAT_BASE}\n")
        f.write(f"Rotation offset: {OFFSET_ROTATION}° around {ROTATION_AXIS.upper()}-axis\n")
        f.write(f"Final quaternion: {CAMERA_QUAT}\n")
        f.write(f"Base camera position: {CAMERA_POS_WORLD_FRAME_BASE}\n")
        f.write(f"Position offset: {OFFSET_POSITION}\n")
        f.write(f"Final camera position: {CAMERA_POS_WORLD_FRAME}\n")
        f.write(f"Confidence weighting power: {CONFIDENCE_POWER}\n")
        f.write(f"Using weighted anatomical COM calculation with confidence scores\n\n")
        for frame_idx, coms in enumerate(com_series):
            f.write(f"Frame {frame_idx}:\n")
            for pid, com in enumerate(coms, start=1):
                if not np.any(np.isnan(com)):
                    f.write(f"  Person {pid} COM (world): {com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}\n")
            f.write("\n")
    print(f"✓ Saved text dump (world frame): {txt_path}")
    
    # Save camera frame text dump
    txt_camera_path = os.path.join(OUTPUT_DIR, "com_series_camera_frame.txt")
    with open(txt_camera_path, "w") as f:
        f.write(f"Coordinates in camera reference frame (meters)\n")
        f.write(f"Quaternion format: {QUATERNION_TYPE}\n")
        f.write(f"Base quaternion: {CAMERA_QUAT_BASE}\n")
        f.write(f"Rotation offset: {OFFSET_ROTATION}° around {ROTATION_AXIS.upper()}-axis\n")
        f.write(f"Final quaternion: {CAMERA_QUAT}\n")
        f.write(f"Base camera position: {CAMERA_POS_WORLD_FRAME_BASE}\n")
        f.write(f"Position offset: {OFFSET_POSITION}\n")
        f.write(f"Final camera position: {CAMERA_POS_WORLD_FRAME}\n")
        f.write(f"Confidence weighting power: {CONFIDENCE_POWER}\n")
        f.write(f"Using weighted anatomical COM calculation with confidence scores\n\n")
        for frame_idx, coms in enumerate(com_series_camera):
            f.write(f"Frame {frame_idx}:\n")
            for pid, com in enumerate(coms, start=1):
                if not np.any(np.isnan(com)):
                    f.write(f"  Person {pid} COM (camera): {com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}\n")
            f.write("\n")
    print(f"✓ Saved text dump (camera frame): {txt_camera_path}")

    # Visualize 2D COM trajectories
    fig = visualize_2d_com_series(com_series)
    print("\nDisplaying 2D visualization of COM trajectories in X-Y plane (world frame).")
    print("Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()