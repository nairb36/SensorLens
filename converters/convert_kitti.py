#!/usr/bin/env python3
"""Convert a KITTI tracking sequence to the SensorLens universal format.

Expects the KITTI tracking dataset layout:
  {dataroot}/
    image_02/{sequence_id}/       # left color camera images
    image_03/{sequence_id}/       # right color camera images (optional)
    velodyne/{sequence_id}/       # LiDAR point clouds (.bin)
    label_02/{sequence_id}.txt    # tracking labels
    calib/{sequence_id}.txt       # calibration
"""

import argparse
from pathlib import Path

import numpy as np

from .common import (
    KITTI_CATEGORY_MAP,
    ensure_dirs,
    write_meta_json,
    write_frame_json,
    write_pointcloud,
    copy_camera_image,
    write_gt_json,
)


def _parse_calib(calib_path):
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array([float(v) for v in vals.split()])

    P2 = data["P2"].reshape(3, 4)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = data["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = data["Tr_velo_to_cam"].reshape(3, 4)

    # cam_rect_to_velo: inverse of (R0_rect @ Tr_velo_to_cam)
    cam_rect_to_velo = np.linalg.inv(R0_rect @ Tr_velo_to_cam)
    return P2, R0_rect, Tr_velo_to_cam, cam_rect_to_velo


def _parse_labels(label_path):
    """Parse KITTI tracking labels. Returns dict: frame_id -> list of objects."""
    frames = {}
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            # parts[3]: truncated, parts[4]: occluded, parts[5]: alpha
            # parts[6:10]: 2D bbox (left, top, right, bottom)
            h, w, l = float(parts[10]), float(parts[11]), float(parts[12])
            x, y, z = float(parts[13]), float(parts[14]), float(parts[15])
            rotation_y = float(parts[16])

            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append({
                "track_id": track_id,
                "type": obj_type,
                "h": h, "w": w, "l": l,
                "x": x, "y": y, "z": z,
                "rotation_y": rotation_y,
            })
    return frames


def _cam_rect_to_ego(x, y, z, h, rotation_y, cam_rect_to_velo):
    # KITTI location is bottom-center of box in camera rect frame.
    # Camera rect: X-right, Y-down, Z-forward.
    # Adjust to true 3D center: move up by h/2 (decrease y since y points down).
    center_cam = np.array([x, y - h / 2.0, z, 1.0])
    center_velo = cam_rect_to_velo @ center_cam

    # rotation_y is around camera Y-axis (pointing down).
    # In Velodyne/ego frame (X-fwd, Y-left, Z-up), yaw is around Z-axis.
    # Camera Z-forward maps to Velodyne X-forward.
    # Camera X-right maps to Velodyne -Y (since Y is left).
    # So: yaw_ego = -(rotation_y) after accounting for the 90-degree axis swap.
    # More precisely: rotation_y=0 means the object faces camera Z (= ego X-forward).
    # rotation_y rotates clockwise in cam frame = counterclockwise around ego Z.
    yaw_ego = -rotation_y

    return center_velo[:3], yaw_ego


def convert_sequence(dataroot, sequence_id, output_dir, gt_output=None,
                     max_frames=None, no_images=False):
    dataroot = Path(dataroot)
    seq = f"{sequence_id:04d}"
    print(f"Converting KITTI tracking sequence {seq}")

    calib_path = dataroot / "calib" / f"{seq}.txt"
    label_path = dataroot / "label_02" / f"{seq}.txt"
    velo_dir = dataroot / "velodyne" / seq
    img2_dir = dataroot / "image_02" / seq
    img3_dir = dataroot / "image_03" / seq

    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration not found: {calib_path}")

    P2, R0_rect, Tr_velo_to_cam, cam_rect_to_velo = _parse_calib(calib_path)

    labels = {}
    if label_path.is_file():
        labels = _parse_labels(label_path)

    # Discover frames from velodyne files
    if velo_dir.is_dir():
        frame_files = sorted(velo_dir.glob("*.bin"))
    else:
        frame_files = sorted(img2_dir.glob("*.png"))

    frame_ids = [int(f.stem) for f in frame_files]
    if max_frames:
        frame_ids = frame_ids[:max_frames]
    num_frames = len(frame_ids)

    scene_dir = ensure_dirs(output_dir)

    camera_names = ["left"]
    if img3_dir.is_dir():
        camera_names.append("right")

    gt_frames = []

    for frame_idx, kitti_frame_id in enumerate(frame_ids):
        # Point cloud
        velo_path = velo_dir / f"{kitti_frame_id:06d}.bin"
        if velo_path.is_file():
            points = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)
        else:
            points = np.empty((0, 4), dtype=np.float32)
        write_pointcloud(scene_dir, frame_idx, points)

        # Camera images
        cam_files = {}
        if not no_images:
            img2_path = img2_dir / f"{kitti_frame_id:06d}.png"
            if img2_path.is_file():
                copy_camera_image(scene_dir, frame_idx, "left", str(img2_path))
                cam_files["left"] = str(img2_path)
            if img3_dir.is_dir():
                img3_path = img3_dir / f"{kitti_frame_id:06d}.png"
                if img3_path.is_file():
                    copy_camera_image(scene_dir, frame_idx, "right", str(img3_path))
                    cam_files["right"] = str(img3_path)

        write_frame_json(scene_dir, frame_idx, timestamp=kitti_frame_id,
                         camera_files=cam_files)

        # GT labels
        detections = []
        for obj in labels.get(kitti_frame_id, []):
            cat = KITTI_CATEGORY_MAP.get(obj["type"])
            if cat is None:
                continue

            center_ego, yaw_ego = _cam_rect_to_ego(
                obj["x"], obj["y"], obj["z"], obj["h"],
                obj["rotation_y"], cam_rect_to_velo
            )
            # KITTI dimensions: h, w, l -> universal size: [w, l, h]
            detections.append({
                "instance_token": f"kitti_{seq}_{obj['track_id']}",
                "category_name": cat,
                "translation": center_ego.tolist(),
                "size": [obj["w"], obj["l"], obj["h"]],
                "yaw": round(float(yaw_ego), 6),
            })

        gt_frames.append({
            "frame_index": frame_idx,
            "timestamp": kitti_frame_id,
            "detections": detections,
        })

        if (frame_idx + 1) % 50 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    write_meta_json(scene_dir, "kitti", f"sequence_{seq}", num_frames,
                    camera_names,
                    category_mapping={k: v for k, v in KITTI_CATEGORY_MAP.items() if v})

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT written to {gt_output}")

    print(f"Sequence converted to {scene_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI tracking sequence to SensorLens format")
    parser.add_argument("--dataroot", required=True, help="Path to KITTI tracking data root")
    parser.add_argument("--sequence", type=int, default=0, help="Sequence number")
    parser.add_argument("--output", required=True, help="Output scene directory")
    parser.add_argument("--gt-output", help="Output path for GT JSON file")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    args = parser.parse_args()

    convert_sequence(args.dataroot, args.sequence, args.output,
                     gt_output=args.gt_output, max_frames=args.max_frames,
                     no_images=args.no_images)


if __name__ == "__main__":
    main()
