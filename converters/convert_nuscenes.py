#!/usr/bin/env python3
"""Convert a nuScenes scene to the SensorLens universal format."""

import argparse
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .common import (
    NUSCENES_CATEGORY_MAP,
    ensure_dirs,
    write_meta_json,
    write_frame_json,
    write_pointcloud,
    copy_camera_image,
    write_gt_json,
    quaternion_to_yaw,
)

CAMERAS = [
    ("CAM_FRONT_LEFT", "front_left"),
    ("CAM_FRONT", "front"),
    ("CAM_FRONT_RIGHT", "front_right"),
    ("CAM_BACK_LEFT", "back_left"),
    ("CAM_BACK", "back"),
    ("CAM_BACK_RIGHT", "back_right"),
]


def _lidar_to_ego(points, calibrated_sensor):
    rot = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
    trans = np.array(calibrated_sensor["translation"])
    pts = points.copy()
    pts[:, :3] = points[:, :3] @ rot.T + trans
    return pts


def _global_to_ego(translation, yaw, ego_pose):
    ego_trans = np.array(ego_pose["translation"])
    ego_rot = Quaternion(ego_pose["rotation"])
    pos = ego_rot.rotation_matrix.T @ (np.array(translation) - ego_trans)
    ego_yaw = ego_rot.yaw_pitch_roll[0]
    local_yaw = yaw - ego_yaw
    return pos, local_yaw


def convert_scene(dataroot, version, scene_index, output_dir,
                  gt_output=None, max_frames=None, no_images=False):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scene = nusc.scene[scene_index]
    scene_name = scene["name"]
    print(f"Converting {scene_name} ({scene['description']})")

    scene_dir = ensure_dirs(output_dir)
    camera_names = [uname for _, uname in CAMERAS]

    sample_token = scene["first_sample_token"]
    samples = []
    while sample_token:
        sample = nusc.get("sample", sample_token)
        samples.append(sample)
        sample_token = sample["next"]
    if max_frames:
        samples = samples[:max_frames]

    num_frames = len(samples)
    gt_frames = []

    for frame_idx, sample in enumerate(samples):
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])

        lidar_path = Path(nusc.dataroot) / lidar_sd["filename"]
        raw_points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        ego_points = _lidar_to_ego(raw_points[:, :4], cs)
        write_pointcloud(scene_dir, frame_idx, ego_points)

        cam_files = {}
        if not no_images:
            for nusc_cam, universal_cam in CAMERAS:
                cam_sd = nusc.get("sample_data", sample["data"][nusc_cam])
                cam_path = str(Path(nusc.dataroot) / cam_sd["filename"])
                copy_camera_image(scene_dir, frame_idx, universal_cam, cam_path)
                cam_files[universal_cam] = cam_path

        ego_pose_data = {
            "translation": ego_pose["translation"],
            "rotation": ego_pose["rotation"],
        }
        write_frame_json(scene_dir, frame_idx, sample["timestamp"],
                         ego_pose=ego_pose_data, camera_files=cam_files)

        anns = [nusc.get("sample_annotation", tok) for tok in sample["anns"]]
        detections = []
        for ann in anns:
            cat = ann["category_name"]
            universal_cat = NUSCENES_CATEGORY_MAP.get(cat)
            if universal_cat is None:
                continue
            yaw = quaternion_to_yaw(ann["rotation"])
            pos, local_yaw = _global_to_ego(ann["translation"], yaw, ego_pose)
            detections.append({
                "instance_token": ann["instance_token"],
                "category_name": universal_cat,
                "translation": pos.tolist(),
                "size": ann["size"],
                "yaw": round(local_yaw, 6),
            })
        gt_frames.append({
            "frame_index": frame_idx,
            "sample_token": sample["token"],
            "timestamp": sample["timestamp"],
            "detections": detections,
        })

        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    cat_mapping = {k: v for k, v in NUSCENES_CATEGORY_MAP.items() if v is not None}
    write_meta_json(scene_dir, "nuscenes", scene_name, num_frames,
                    camera_names, category_mapping=cat_mapping)

    embedded_gt = str(Path(scene_dir) / "gt.json")
    write_gt_json(embedded_gt, gt_frames)
    print(f"GT embedded in scene: {embedded_gt}")

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT also written to {gt_output}")

    print(f"Scene converted to {scene_dir}")


def convert_all(dataroot, version, output_dir, max_frames=None, no_images=False):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting all {len(nusc.scene)} scenes from {version}")
    for i, scene in enumerate(nusc.scene):
        scene_out = str(output_dir / scene["name"])
        convert_scene(dataroot, version, i, scene_out,
                      max_frames=max_frames, no_images=no_images)
    print(f"\nAll {len(nusc.scene)} scenes converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert nuScenes scene(s) to SensorLens format")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--output", required=True, help="Output directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene", type=int, help="Single scene index (0-based)")
    group.add_argument("--all", action="store_true", help="Convert all scenes")

    parser.add_argument("--gt-output", help="Extra copy of GT JSON (single scene only)")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames per scene")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    args = parser.parse_args()

    if args.all:
        convert_all(args.dataroot, args.version, args.output,
                    max_frames=args.max_frames, no_images=args.no_images)
    else:
        scene_idx = args.scene if args.scene is not None else 0
        convert_scene(args.dataroot, args.version, scene_idx, args.output,
                      gt_output=args.gt_output, max_frames=args.max_frames,
                      no_images=args.no_images)


if __name__ == "__main__":
    main()
