#!/usr/bin/env python3
"""
Combine trajectory plot images into a single video.

Expected structure:
<root>/trajectory_plots/
  ├─ batch_0/
  │    ├─ trajectory_batch_0.png
  │    ├─ trajectory_batch_1.png
  │    └─ ...
  ├─ batch_1/
  │    ├─ trajectory_batch_0.png
  │    ├─ trajectory_batch_1.png
  │    └─ ...
  └─ ...

Usage:
python make_video.py \
  --root /path/to/trajectory_plots \
  --out out.mp4 \
  --fps 10

Notes:
- Sorts batches and images by their numeric suffixes.
- Accepts any image extension (png/jpg/jpeg/webp) by default.
"""

import argparse
import re
import sys
from pathlib import Path
import cv2

BATCH_DIR_RE = re.compile(r"batch[_\-]?(\d+)$", re.IGNORECASE)
IMG_FILE_RE  = re.compile(r"trajectory_batch[_\-]?(\d+)\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)

def natural_key_from_match(m):
    # convert "…_12" to 12 (int) for proper sorting
    return int(m.group(1))

def list_batches(root: Path):
    batches = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            m = BATCH_DIR_RE.search(p.name)
            if m:
                batches.append((natural_key_from_match(m), p))
    batches.sort(key=lambda x: x[0])
    return [p for _, p in batches]

def list_images_in_batch(batch_dir: Path):
    imgs = []
    for p in batch_dir.iterdir():
        if p.is_file():
            m = IMG_FILE_RE.search(p.name)
            if m:
                imgs.append((natural_key_from_match(m), p))
    imgs.sort(key=lambda x: x[0])
    return [p for _, p in imgs]

def probe_first_frame(all_images):
    for img_path in all_images:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            h, w = frame.shape[:2]
            return (w, h), frame
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Combine trajectory images into a video.")
    ap.add_argument("--root", default="/home/gkini/Human-Traj-Prediction/trajectory_plots", help="Path to trajectory_plots directory (parent of batch_*).")
    ap.add_argument("--out", default="trajectory2.mp4", help="Output video file (e.g., out.mp4)")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames per second.")
    ap.add_argument("--codec", default="mp4v", help="FourCC codec (e.g., mp4v, avc1, XVID).")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any image can't be read or size mismatches; otherwise skip.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Find and order batches
    batches = list_batches(root)
    if not batches:
        print(f"ERROR: No batch_* directories found under {root}", file=sys.stderr)
        sys.exit(1)

    # Collect images in global order
    all_images = []
    for b in batches:
        imgs = list_images_in_batch(b)
        if not imgs:
            print(f"WARNING: No images in {b} matching 'trajectory_batch_*.*' — skipping.", file=sys.stderr)
            continue
        all_images.extend(imgs)

    if not all_images:
        print("ERROR: No images found to compile.", file=sys.stderr)
        sys.exit(1)

    # Determine frame size from first readable image
    size, first_frame = probe_first_frame(all_images)
    if size is None:
        print("ERROR: Couldn't read any images.", file=sys.stderr)
        sys.exit(1)

    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print(f"ERROR: Failed to open VideoWriter for {args.out} with codec {args.codec}", file=sys.stderr)
        sys.exit(1)

    written = 0
    for img_path in all_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            msg = f"WARNING: Could not read {img_path} — skipping."
            if args.strict:
                print("ERROR: " + msg, file=sys.stderr)
                writer.release()
                sys.exit(1)
            else:
                print(msg, file=sys.stderr)
                continue

        fh, fw = frame.shape[:2]
        if (fw, fh) != (w, h):
            if args.strict:
                print(f"ERROR: Size mismatch in {img_path}: got {(fw, fh)}, expected {(w, h)}", file=sys.stderr)
                writer.release()
                sys.exit(1)
            # Auto-resize to the first frame size
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        written += 1
        if written % 50 == 0:
            print(f"Wrote {written} frames...")

    writer.release()
    print(f"Done. Wrote {written} frames to {args.out} at {args.fps} FPS.")

if __name__ == "__main__":
    main()
