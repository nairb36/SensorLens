#!/usr/bin/env python3
"""Convert a Waymo Open Dataset segment to the SensorLens universal format.

Requires: pip install waymo-open-dataset-tf-2-11-0 (or compatible version)

Each .tfrecord file contains one segment (driving sequence).
"""

import argparse
import os
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


def convert_segment(tfrecord_path, output_dir, gt_output=None,
                    max_frames=None, no_images=False):
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.utils import frame_utils
    except ImportError:
        raise ImportError(
            "Waymo converter requires: pip install waymo-open-dataset-tf-2-11-0\n"
            "Also requires TensorFlow."
        )

    tfrecord_path = str(tfrecord_path)
    segment_name = Path(tfrecord_path).stem
    print(f"Converting Waymo segment: {segment_name}")

    scene_dir = ensure_dirs(output_dir)
    camera_names = list(CAMERA_NAMES.values())

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    gt_frames = []
    frame_idx = 0

    for raw_record in dataset:
        if max_frames and frame_idx >= max_frames:
            break

        frame = dataset_pb2.Frame()
        frame.ParseFromString(raw_record.numpy())

        # Point cloud from range images
        (range_images, camera_projections,
         seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        points_all, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            keep_polar_features=False,
        )
        # Concatenate all LiDAR returns (TOP is index 0, usually the main one)
        all_pts = np.concatenate(points_all, axis=0)
        # points_all gives [x, y, z] in vehicle frame; add intensity as 0
        if all_pts.shape[1] == 3:
            all_pts = np.column_stack([all_pts, np.zeros(len(all_pts), dtype=np.float32)])
        write_pointcloud(scene_dir, frame_idx, all_pts[:, :4])

        # Camera images
        cam_files = {}
        if not no_images:
            import cv2
            cam_dir = Path(scene_dir) / "cameras" / f"{frame_idx:06d}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for ci in frame.images:
                cam_name = CAMERA_NAMES.get(ci.name)
                if cam_name is None:
                    continue
                img_bytes = ci.image
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    dst = cam_dir / f"{cam_name}.jpg"
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cam_files[cam_name] = str(dst)

        # Ego pose (4x4 matrix -> translation + quaternion)
        pose_matrix = np.array(frame.pose.transform).reshape(4, 4)
        translation = pose_matrix[:3, 3].tolist()
        ego_pose = {"translation": translation, "rotation": [1, 0, 0, 0]}

        write_frame_json(scene_dir, frame_idx, timestamp=frame.timestamp_micros,
                         ego_pose=ego_pose, camera_files=cam_files)

        # Labels (already in vehicle/ego frame)
        detections = []
        for label in frame.laser_labels:
            cat = WAYMO_CATEGORY_MAP.get(label.type)
            if cat is None:
                continue
            box = label.box
            detections.append({
                "instance_token": label.id,
                "category_name": cat,
                "translation": [box.center_x, box.center_y, box.center_z],
                "size": [box.width, box.length, box.height],
                "yaw": round(box.heading, 6),
            })

        gt_frames.append({
            "frame_index": frame_idx,
            "timestamp": frame.timestamp_micros,
            "detections": detections,
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}")

    num_frames = frame_idx
    print(f"  Total: {num_frames} frames")

    write_meta_json(scene_dir, "waymo", segment_name, num_frames,
                    camera_names,
                    category_mapping={str(k): v for k, v in WAYMO_CATEGORY_MAP.items() if v})

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT written to {gt_output}")

    print(f"Segment converted to {scene_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Waymo segment to SensorLens format")
    parser.add_argument("--tfrecord", required=True, help="Path to .tfrecord file")
    parser.add_argument("--output", required=True, help="Output scene directory")
    parser.add_argument("--gt-output", help="Output path for GT JSON file")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    args = parser.parse_args()

    convert_segment(args.tfrecord, args.output,
                    gt_output=args.gt_output, max_frames=args.max_frames,
                    no_images=args.no_images)


if __name__ == "__main__":
    main()
