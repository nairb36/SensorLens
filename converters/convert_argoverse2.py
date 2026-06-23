#!/usr/bin/env python3
"""Convert an Argoverse 2 sensor log to the SensorLens universal format.

Requires: pip install av2

Expects the Argoverse 2 sensor dataset layout:
  {dataroot}/
    {split}/
      {log_id}/
        sensors/
          cameras/
            ring_front_center/
            ring_front_left/
            ...
          lidar/
            {timestamp_ns}.feather
        annotations.feather
        calibration/
        city_SE3_egovehicle.feather
"""

import argparse
from pathlib import Path

import numpy as np

from .common import (
    ARGOVERSE2_CATEGORY_MAP,
    ensure_dirs,
    write_meta_json,
    write_frame_json,
    write_pointcloud,
    copy_camera_image,
    write_gt_json,
    quaternion_to_yaw,
)

RING_CAMERAS = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]

STEREO_CAMERAS = [
    "stereo_front_left",
    "stereo_front_right",
]

CAMERA_NAME_MAP = {
    "ring_front_center": "front",
    "ring_front_left": "front_left",
    "ring_front_right": "front_right",
    "ring_side_left": "side_left",
    "ring_side_right": "side_right",
    "ring_rear_left": "rear_left",
    "ring_rear_right": "rear_right",
    "stereo_front_left": "stereo_left",
    "stereo_front_right": "stereo_right",
}


def _quat_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ])


def _nearest_ego_pose(ego_df, timestamp_ns, ego_timestamps):
    idx = np.argmin(np.abs(ego_timestamps - timestamp_ns))
    return ego_df.iloc[idx]


def _ego_row_to_dict(row):
    return {
        "translation": [float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])],
        "rotation": [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
    }


def _transform_ego_to_city(translation_ego, yaw_ego, ego_row):
    qw, qx, qy, qz = ego_row["qw"], ego_row["qx"], ego_row["qy"], ego_row["qz"]
    t_city = np.array([ego_row["tx_m"], ego_row["ty_m"], ego_row["tz_m"]])
    R = _quat_to_rotation_matrix(qw, qx, qy, qz)
    pos_city = R @ np.array(translation_ego) + t_city
    yaw_city = yaw_ego + quaternion_to_yaw([qw, qx, qy, qz])
    return pos_city.tolist(), float(yaw_city)


def convert_log(dataroot, split, log_id, output_dir, gt_output=None,
                max_frames=None, no_images=False, include_stereo=False):
    try:
        import pyarrow.feather as feather
        import pandas as pd
    except ImportError:
        raise ImportError("Argoverse 2 converter requires: pip install pyarrow pandas")

    log_dir = Path(dataroot) / split / log_id
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    print(f"Converting Argoverse 2 log: {log_id}")

    scene_dir = ensure_dirs(output_dir)

    lidar_dir = log_dir / "sensors" / "lidar"
    lidar_files = sorted(lidar_dir.glob("*.feather"))
    if max_frames:
        lidar_files = lidar_files[:max_frames]
    num_frames = len(lidar_files)

    # Load ego poses
    ego_path = log_dir / "city_SE3_egovehicle.feather"
    if not ego_path.is_file():
        raise FileNotFoundError(f"city_SE3_egovehicle.feather not found in {log_dir}")
    ego_df = pd.read_feather(str(ego_path)).sort_values("timestamp_ns").reset_index(drop=True)
    ego_timestamps = ego_df["timestamp_ns"].values.astype(np.int64)

    # Load annotations
    ann_path = log_dir / "annotations.feather"
    annotations = None
    if ann_path.is_file():
        annotations = pd.read_feather(str(ann_path))

    # Discover cameras
    cam_sources = RING_CAMERAS[:]
    if include_stereo:
        cam_sources.extend(STEREO_CAMERAS)
    available_cams = []
    cam_dirs = {}
    for cam in cam_sources:
        cam_dir = log_dir / "sensors" / "cameras" / cam
        if cam_dir.is_dir() and any(cam_dir.iterdir()):
            available_cams.append(cam)
            cam_dirs[cam] = cam_dir

    camera_names = [CAMERA_NAME_MAP[c] for c in available_cams]

    gt_frames = []

    for frame_idx, lidar_file in enumerate(lidar_files):
        timestamp_ns = int(lidar_file.stem)

        # Ego pose (nearest to this LiDAR timestamp)
        ego_row = _nearest_ego_pose(ego_df, timestamp_ns, ego_timestamps)
        ego_dict = _ego_row_to_dict(ego_row)

        # Point cloud (already in ego frame)
        lidar_df = pd.read_feather(str(lidar_file))
        x = lidar_df["x"].values.astype(np.float32)
        y = lidar_df["y"].values.astype(np.float32)
        z = lidar_df["z"].values.astype(np.float32)
        intensity = lidar_df.get("intensity")
        if intensity is not None:
            intensity = intensity.values.astype(np.float32)
        else:
            intensity = np.zeros_like(x)
        points = np.column_stack([x, y, z, intensity])
        write_pointcloud(scene_dir, frame_idx, points)

        # Camera images (find closest timestamp)
        cam_files = {}
        if not no_images:
            for cam in available_cams:
                universal_name = CAMERA_NAME_MAP[cam]
                cam_dir = cam_dirs[cam]
                cam_images = sorted(cam_dir.glob("*.jpg"))
                if not cam_images:
                    cam_images = sorted(cam_dir.glob("*.png"))
                if not cam_images:
                    continue
                cam_timestamps = [int(f.stem) for f in cam_images]
                closest_idx = np.argmin(np.abs(np.array(cam_timestamps) - timestamp_ns))
                src_path = cam_images[closest_idx]
                copy_camera_image(scene_dir, frame_idx, universal_name, str(src_path))
                cam_files[universal_name] = str(src_path)

        write_frame_json(scene_dir, frame_idx, timestamp=timestamp_ns,
                         ego_pose=ego_dict, camera_files=cam_files)

        # Annotations: stored in ego frame, transform to city/global frame
        detections = []
        if annotations is not None:
            frame_anns = annotations[annotations["timestamp_ns"] == timestamp_ns]
            for _, ann in frame_anns.iterrows():
                cat = ARGOVERSE2_CATEGORY_MAP.get(ann["category"])
                if cat is None:
                    continue
                yaw_ego = quaternion_to_yaw([ann["qw"], ann["qx"], ann["qy"], ann["qz"]])
                pos_ego = [float(ann["tx_m"]), float(ann["ty_m"]), float(ann["tz_m"])]
                pos_city, yaw_city = _transform_ego_to_city(pos_ego, yaw_ego, ego_row)
                detections.append({
                    "instance_token": ann["track_uuid"],
                    "category_name": cat,
                    "translation": [round(v, 6) for v in pos_city],
                    "size": [float(ann["width_m"]), float(ann["length_m"]), float(ann["height_m"])],
                    "yaw": round(yaw_city, 6),
                })

        gt_frames.append({
            "frame_index": frame_idx,
            "timestamp": timestamp_ns,
            "detections": detections,
        })

        if (frame_idx + 1) % 20 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    write_meta_json(scene_dir, "argoverse2", log_id, num_frames,
                    camera_names,
                    category_mapping={k: v for k, v in ARGOVERSE2_CATEGORY_MAP.items() if v})

    embedded_gt = str(Path(scene_dir) / "gt.json")
    write_gt_json(embedded_gt, gt_frames)
    print(f"GT embedded in scene: {embedded_gt}")

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT also written to {gt_output}")

    print(f"Log converted to {scene_dir}")


def convert_all(dataroot, split, output_dir, max_frames=None, no_images=False,
                include_stereo=False):
    split_dir = Path(dataroot) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    log_ids = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting all {len(log_ids)} Argoverse 2 logs from {split}")
    for log_id in log_ids:
        log_out = str(output_dir / log_id)
        convert_log(dataroot, split, log_id, log_out,
                    max_frames=max_frames, no_images=no_images,
                    include_stereo=include_stereo)
    print(f"\nAll {len(log_ids)} logs converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Argoverse 2 log to SensorLens format")
    parser.add_argument("--dataroot", required=True, help="Path to Argoverse 2 sensor root")
    parser.add_argument("--split", default="val", help="Split (train/val/test)")
    parser.add_argument("--output", required=True, help="Output directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--log-id", help="Single log UUID")
    group.add_argument("--all", action="store_true", help="Convert all logs in split")

    parser.add_argument("--gt-output", help="Extra copy of GT JSON (single log only)")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    parser.add_argument("--include-stereo", action="store_true", help="Include stereo cameras")
    args = parser.parse_args()

    if args.all:
        convert_all(args.dataroot, args.split, args.output,
                    max_frames=args.max_frames, no_images=args.no_images,
                    include_stereo=args.include_stereo)
    else:
        log_id = args.log_id
        if not log_id:
            split_dir = Path(args.dataroot) / args.split
            logs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            log_id = logs[0]
            print(f"No --log-id specified, using first log: {log_id}")
        convert_log(args.dataroot, args.split, log_id, args.output,
                    gt_output=args.gt_output, max_frames=args.max_frames,
                    no_images=args.no_images, include_stereo=args.include_stereo)


if __name__ == "__main__":
    main()
