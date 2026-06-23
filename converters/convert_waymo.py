#!/usr/bin/env python3
"""Convert a Waymo Open Dataset v2 segment to the SensorLens universal format.

Reads Parquet files directly — no TensorFlow or waymo-open-dataset package needed.
Requires: pip install pyarrow pandas

Expects the Waymo v2 dataset layout:
  {dataroot}/
    lidar_box/{segment_name}.parquet
    vehicle_pose/{segment_name}.parquet
    camera_image/{segment_name}.parquet
    lidar/{segment_name}.parquet
    lidar_calibration/{segment_name}.parquet
    lidar_pose/{segment_name}.parquet          (optional, for motion compensation)
"""

import argparse
from pathlib import Path

import numpy as np

from .common import (
    WAYMO_CATEGORY_MAP,
    ensure_dirs,
    write_meta_json,
    write_frame_json,
    write_pointcloud,
    write_gt_json,
)

CAMERA_NAMES = {
    1: "front",
    2: "front_left",
    3: "front_right",
    4: "side_left",
    5: "side_right",
}


def _mat4x4_to_ego_dict(mat):
    t = mat[:3, 3].tolist()
    R = mat[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return {
        "translation": t,
        "rotation": [float(w), float(x), float(y), float(z)],
    }


def _range_image_to_points(ri, beam_inclinations, extrinsic_translation):
    H, W = ri.shape[0], ri.shape[1]
    ranges = ri[:, :, 0]
    intensity = ri[:, :, 1]

    azimuth = np.linspace(np.pi, -np.pi, W, endpoint=False)
    # Waymo v2: beam inclinations stored bottom-to-top, range image rows top-to-bottom
    inc_reversed = beam_inclinations[::-1]
    inc_grid, az_grid = np.meshgrid(inc_reversed, azimuth, indexing='ij')

    cos_inc = np.cos(inc_grid)
    x = ranges * cos_inc * np.cos(az_grid)
    y = ranges * cos_inc * np.sin(az_grid)
    z = ranges * np.sin(inc_grid)

    mask = ranges > 0
    # Waymo v2 range images are pre-aligned to vehicle frame — only translation offset needed
    pts_vehicle = np.column_stack([x[mask], y[mask], z[mask]]) + extrinsic_translation
    int_valid = intensity[mask]

    return np.column_stack([pts_vehicle, int_valid]).astype(np.float32)


def convert_segment(dataroot, segment_name, output_dir, gt_output=None,
                    max_frames=None, no_images=False, top_lidar_only=False):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Waymo v2 converter requires: pip install pyarrow pandas")

    dataroot = Path(dataroot)
    print(f"Converting Waymo segment: {segment_name}")

    lidar_box_path = dataroot / "lidar_box" / f"{segment_name}.parquet"
    vehicle_pose_path = dataroot / "vehicle_pose" / f"{segment_name}.parquet"
    lidar_path = dataroot / "lidar" / f"{segment_name}.parquet"
    lidar_cal_path = dataroot / "lidar_calibration" / f"{segment_name}.parquet"
    cam_img_path = dataroot / "camera_image" / f"{segment_name}.parquet"

    for p, name in [(lidar_box_path, "lidar_box"), (vehicle_pose_path, "vehicle_pose"),
                    (lidar_path, "lidar"), (lidar_cal_path, "lidar_calibration")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    lb_df = pd.read_parquet(str(lidar_box_path))
    vp_df = pd.read_parquet(str(vehicle_pose_path))
    lidar_df = pd.read_parquet(str(lidar_path))
    cal_df = pd.read_parquet(str(lidar_cal_path))

    has_cameras = cam_img_path.is_file() and not no_images
    cam_df = pd.read_parquet(str(cam_img_path)) if has_cameras else None

    # Calibration: build per-laser extrinsics and beam inclinations
    laser_names = [1] if top_lidar_only else sorted(cal_df['key.laser_name'].unique())
    laser_cal = {}
    for ln in laser_names:
        row = cal_df[cal_df['key.laser_name'] == ln].iloc[0]
        extrinsic = np.array(row['[LiDARCalibrationComponent].extrinsic.transform']).reshape(4, 4)
        translation = extrinsic[:3, 3]
        beam_vals = row['[LiDARCalibrationComponent].beam_inclination.values']
        if beam_vals is not None and hasattr(beam_vals, '__len__'):
            beam_inc = np.array(beam_vals)
        else:
            beam_min = row['[LiDARCalibrationComponent].beam_inclination.min']
            beam_max = row['[LiDARCalibrationComponent].beam_inclination.max']
            lidar_frame_sample = lidar_df[lidar_df['key.laser_name'] == ln].iloc[0]
            H = lidar_frame_sample['[LiDARComponent].range_image_return1.shape'][0]
            beam_inc = np.linspace(beam_min, beam_max, H)
        laser_cal[ln] = (translation, beam_inc)

    timestamps = sorted(vp_df['key.frame_timestamp_micros'].unique())
    if max_frames:
        timestamps = timestamps[:max_frames]
    num_frames = len(timestamps)

    scene_dir = ensure_dirs(output_dir)
    camera_names = list(CAMERA_NAMES.values())

    gt_frames = []

    for frame_idx, ts in enumerate(timestamps):
        # Ego pose
        vp_row = vp_df[vp_df['key.frame_timestamp_micros'] == ts].iloc[0]
        pose_mat = np.array(vp_row['[VehiclePoseComponent].world_from_vehicle.transform']).reshape(4, 4)
        ego_dict = _mat4x4_to_ego_dict(pose_mat)

        # Point cloud: combine all lidars
        all_points = []
        for ln in laser_names:
            laser_frame = lidar_df[
                (lidar_df['key.frame_timestamp_micros'] == ts) &
                (lidar_df['key.laser_name'] == ln)
            ]
            if laser_frame.empty:
                continue
            row = laser_frame.iloc[0]
            shape = row['[LiDARComponent].range_image_return1.shape']
            values = row['[LiDARComponent].range_image_return1.values']
            ri = np.array(values).reshape(shape)
            translation, beam_inc = laser_cal[ln]
            pts = _range_image_to_points(ri, beam_inc, translation)
            if len(pts) > 0:
                all_points.append(pts)

        if all_points:
            combined = np.vstack(all_points)
        else:
            combined = np.empty((0, 4), dtype=np.float32)
        write_pointcloud(scene_dir, frame_idx, combined)

        # Camera images
        cam_files = {}
        if has_cameras:
            import cv2
            frame_cams = cam_df[cam_df['key.frame_timestamp_micros'] == ts]
            cam_dir = Path(scene_dir) / "cameras" / f"{frame_idx:06d}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for _, ci in frame_cams.iterrows():
                cam_name = CAMERA_NAMES.get(ci['key.camera_name'])
                if cam_name is None:
                    continue
                img_bytes = ci['[CameraImageComponent].image']
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    dst = cam_dir / f"{cam_name}.jpg"
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cam_files[cam_name] = str(dst)

        write_frame_json(scene_dir, frame_idx, timestamp=int(ts),
                         ego_pose=ego_dict, camera_files=cam_files)

        # Labels: vehicle frame → world frame
        frame_labels = lb_df[lb_df['key.frame_timestamp_micros'] == ts]
        detections = []
        for _, label in frame_labels.iterrows():
            cat = WAYMO_CATEGORY_MAP.get(label['[LiDARBoxComponent].type'])
            if cat is None:
                continue
            cx = label['[LiDARBoxComponent].box.center.x']
            cy = label['[LiDARBoxComponent].box.center.y']
            cz = label['[LiDARBoxComponent].box.center.z']
            heading = label['[LiDARBoxComponent].box.heading']

            pos_vehicle = np.array([cx, cy, cz, 1.0])
            pos_world = (pose_mat @ pos_vehicle)[:3]

            R = pose_mat[:3, :3]
            ego_yaw = np.arctan2(R[1, 0], R[0, 0])
            yaw_world = heading + ego_yaw

            detections.append({
                "instance_token": label['key.laser_object_id'],
                "category_name": cat,
                "translation": [round(float(v), 6) for v in pos_world],
                "size": [
                    float(label['[LiDARBoxComponent].box.size.y']),  # width
                    float(label['[LiDARBoxComponent].box.size.x']),  # length
                    float(label['[LiDARBoxComponent].box.size.z']),  # height
                ],
                "yaw": round(float(yaw_world), 6),
            })

        gt_frames.append({
            "frame_index": frame_idx,
            "timestamp": int(ts),
            "detections": detections,
        })

        if (frame_idx + 1) % 20 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    write_meta_json(scene_dir, "waymo", segment_name, num_frames,
                    camera_names,
                    category_mapping={str(k): v for k, v in WAYMO_CATEGORY_MAP.items() if v})

    embedded_gt = str(Path(scene_dir) / "gt.json")
    write_gt_json(embedded_gt, gt_frames)
    print(f"GT embedded in scene: {embedded_gt}")

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT also written to {gt_output}")

    print(f"Segment converted to {scene_dir}")


def convert_all(dataroot, output_dir, max_frames=None, no_images=False,
                top_lidar_only=False):
    dataroot = Path(dataroot)
    lidar_box_dir = dataroot / "lidar_box"
    if not lidar_box_dir.is_dir():
        raise FileNotFoundError(f"lidar_box directory not found: {lidar_box_dir}")
    segments = sorted([p.stem for p in lidar_box_dir.glob("*.parquet")])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting all {len(segments)} Waymo segments")
    for seg in segments:
        seg_out = str(output_dir / seg)
        convert_segment(dataroot, seg, seg_out,
                        max_frames=max_frames, no_images=no_images,
                        top_lidar_only=top_lidar_only)
    print(f"\nAll {len(segments)} segments converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo v2 segment(s) to SensorLens format")
    parser.add_argument("--dataroot", required=True,
                        help="Path to Waymo v2 data root (containing lidar_box/, vehicle_pose/, etc.)")
    parser.add_argument("--output", required=True, help="Output directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--segment", help="Single segment name")
    group.add_argument("--all", action="store_true", help="Convert all segments")

    parser.add_argument("--gt-output", help="Extra copy of GT JSON (single segment only)")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    parser.add_argument("--top-lidar-only", action="store_true",
                        help="Use only TOP lidar (faster, fewer points)")
    args = parser.parse_args()

    if args.all:
        convert_all(args.dataroot, args.output,
                    max_frames=args.max_frames, no_images=args.no_images,
                    top_lidar_only=args.top_lidar_only)
    else:
        segment = args.segment
        if not segment:
            lidar_box_dir = Path(args.dataroot) / "lidar_box"
            segments = sorted([p.stem for p in lidar_box_dir.glob("*.parquet")])
            segment = segments[0]
            print(f"No --segment specified, using first: {segment}")
        convert_segment(args.dataroot, segment, args.output,
                        gt_output=args.gt_output, max_frames=args.max_frames,
                        no_images=args.no_images, top_lidar_only=args.top_lidar_only)


if __name__ == "__main__":
    main()
