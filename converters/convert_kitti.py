#!/usr/bin/env python3
"""Convert a KITTI tracking sequence to the SensorLens universal format.

Expects the KITTI tracking dataset layout:
  {dataroot}/training/
    oxts/{sequence_id}.txt          # GPS/IMU poses (one line per frame)
    calib/{sequence_id}.txt         # calibration matrices
    label_02/{sequence_id}.txt      # tracking labels
    velodyne/{sequence_id}/         # LiDAR point clouds (.bin) [optional]
    image_02/{sequence_id}/         # left color camera images [optional]
    image_03/{sequence_id}/         # right color camera images [optional]
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


EARTH_RADIUS = 6378137.0  # WGS84 semi-major axis


def _parse_oxts(oxts_path):
    """Parse oxts file into list of 4x4 global pose matrices.

    Uses Mercator projection (same as KITTI devkit) to convert
    lat/lon/alt/roll/pitch/yaw into a local ENU frame with the
    first frame as origin.
    """
    poses = []
    origin_set = False
    scale = 1.0
    ox, oy, oz = 0.0, 0.0, 0.0

    with open(oxts_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                poses.append(np.eye(4))
                continue

            lat = float(parts[0]) * np.pi / 180.0
            lon = float(parts[1]) * np.pi / 180.0
            alt = float(parts[2])
            roll = float(parts[3])
            pitch = float(parts[4])
            yaw = float(parts[5])

            if not origin_set:
                scale = np.cos(lat)
                ox = scale * EARTH_RADIUS * lon
                oy = scale * EARTH_RADIUS * np.log(np.tan(np.pi / 4.0 + lat / 2.0))
                oz = alt
                origin_set = True

            tx = scale * EARTH_RADIUS * lon - ox
            ty = scale * EARTH_RADIUS * np.log(np.tan(np.pi / 4.0 + lat / 2.0)) - oy
            tz = alt - oz

            # Rotation from roll/pitch/yaw (ZYX convention)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)

    return poses


def _pose_to_ego_dict(T):
    """Convert 4x4 pose matrix to ego_pose dict with translation + quaternion."""
    t = T[:3, 3].tolist()
    R = T[:3, :3]
    # Rotation matrix to quaternion (w, x, y, z)
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


def _parse_calib(calib_path):
    """Parse KITTI calib file. Returns Tr_velo_to_cam, R0_rect, Tr_imu_to_velo, and
    the composite cam_rect_to_velo transform."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, vals = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, vals = parts[0], parts[1]
            data[key.strip()] = np.array([float(v) for v in vals.split()])

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = data["R_rect"].reshape(3, 3)

    Tr_velo_cam = np.eye(4)
    Tr_velo_cam[:3, :] = data["Tr_velo_cam"].reshape(3, 4)

    Tr_imu_velo = np.eye(4)
    Tr_imu_velo[:3, :] = data["Tr_imu_velo"].reshape(3, 4)

    cam_rect_to_velo = np.linalg.inv(R0_rect @ Tr_velo_cam)

    return Tr_velo_cam, R0_rect, Tr_imu_velo, cam_rect_to_velo


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


def _label_to_global(obj, cam_rect_to_velo, Tr_imu_velo, ego_pose):
    """Transform a single label from camera rect frame to global frame.

    Chain: cam_rect -> velodyne -> IMU -> global
    """
    # Bottom-center in cam rect → true center (camera Y points down)
    center_cam = np.array([obj["x"], obj["y"] - obj["h"] / 2.0, obj["z"], 1.0])

    # cam_rect → velodyne
    center_velo = cam_rect_to_velo @ center_cam

    # velodyne → IMU
    velo_to_imu = np.linalg.inv(Tr_imu_velo)
    center_imu = velo_to_imu @ center_velo

    # IMU → global
    center_global = ego_pose @ center_imu

    # Yaw: rotation_y is around camera Y-axis (down), measured from camera Z-axis (forward).
    # In velodyne frame: camera Z → velo X, camera X → velo -Y
    # So object heading in velo frame: yaw_velo = -(rotation_y - pi/2) ... but simpler:
    # rotation_y=0 means facing camera Z = velo X. In velo frame that's yaw=0.
    # rotation_y rotates CW in camera top-view = CW when looking down Z.
    # In velo frame (X-fwd, Y-left), CW from above = negative yaw.
    # But we also need to account for the rotation between velo and cam frames.
    # The clean way: transform a heading vector.
    heading_cam = np.array([np.sin(obj["rotation_y"]), 0, np.cos(obj["rotation_y"]), 0])
    heading_velo = cam_rect_to_velo @ heading_cam
    heading_imu = velo_to_imu @ heading_velo
    heading_global = ego_pose @ heading_imu
    yaw_global = float(np.arctan2(heading_global[1], heading_global[0]))

    return center_global[:3].tolist(), yaw_global


def convert_sequence(dataroot, sequence_id, output_dir, gt_output=None,
                     max_frames=None, no_images=False):
    dataroot = Path(dataroot) / "training"
    seq = f"{sequence_id:04d}"
    print(f"Converting KITTI tracking sequence {seq}")

    calib_path = dataroot / "calib" / f"{seq}.txt"
    oxts_path = dataroot / "oxts" / f"{seq}.txt"
    label_path = dataroot / "label_02" / f"{seq}.txt"
    velo_dir = dataroot / "velodyne" / seq
    img2_dir = dataroot / "image_02" / seq
    img3_dir = dataroot / "image_03" / seq

    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration not found: {calib_path}")
    if not oxts_path.is_file():
        raise FileNotFoundError(f"Oxts not found: {oxts_path}")

    Tr_velo_cam, R0_rect, Tr_imu_velo, cam_rect_to_velo = _parse_calib(calib_path)
    poses = _parse_oxts(oxts_path)

    labels = {}
    if label_path.is_file():
        labels = _parse_labels(label_path)

    # Determine number of frames from oxts (one line per frame)
    num_frames = len(poses)
    if max_frames:
        num_frames = min(num_frames, max_frames)

    scene_dir = ensure_dirs(output_dir)

    camera_names = []
    if img2_dir.is_dir():
        camera_names.append("left")
    if img3_dir.is_dir():
        camera_names.append("right")

    gt_frames = []

    for frame_idx in range(num_frames):
        ego_pose = poses[frame_idx]
        ego_dict = _pose_to_ego_dict(ego_pose)

        # Point cloud (velodyne frame → keep as-is, it's ego frame)
        velo_path = velo_dir / f"{frame_idx:06d}.bin"
        if velo_path.is_file():
            points = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)
            # Transform points from velodyne to global frame
            velo_to_imu = np.linalg.inv(Tr_imu_velo)
            pts_hom = np.column_stack([points[:, :3], np.ones(len(points))])
            pts_imu = (velo_to_imu @ pts_hom.T).T
            pts_global = (ego_pose @ pts_imu.T).T
            points_out = np.column_stack([pts_global[:, :3].astype(np.float32),
                                          points[:, 3:4]])
        else:
            points_out = np.empty((0, 4), dtype=np.float32)
        write_pointcloud(scene_dir, frame_idx, points_out)

        # Camera images
        cam_files = {}
        if not no_images:
            img2_path = img2_dir / f"{frame_idx:06d}.png"
            if img2_path.is_file():
                copy_camera_image(scene_dir, frame_idx, "left", str(img2_path))
                cam_files["left"] = str(img2_path)
            img3_path = img3_dir / f"{frame_idx:06d}.png"
            if img3_path.is_file():
                copy_camera_image(scene_dir, frame_idx, "right", str(img3_path))
                cam_files["right"] = str(img3_path)

        write_frame_json(scene_dir, frame_idx, timestamp=frame_idx,
                         ego_pose=ego_dict, camera_files=cam_files)

        # GT labels → global frame
        detections = []
        for obj in labels.get(frame_idx, []):
            cat = KITTI_CATEGORY_MAP.get(obj["type"])
            if cat is None:
                continue

            center_global, yaw_global = _label_to_global(
                obj, cam_rect_to_velo, Tr_imu_velo, ego_pose
            )
            # KITTI dimensions: h, w, l → universal size: [w, l, h]
            detections.append({
                "instance_token": f"kitti_{seq}_{obj['track_id']}",
                "category_name": cat,
                "translation": [round(v, 6) for v in center_global],
                "size": [obj["w"], obj["l"], obj["h"]],
                "yaw": round(yaw_global, 6),
            })

        gt_frames.append({
            "frame_index": frame_idx,
            "timestamp": frame_idx,
            "detections": detections,
        })

        if (frame_idx + 1) % 50 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    write_meta_json(scene_dir, "kitti", f"sequence_{seq}", num_frames,
                    camera_names,
                    category_mapping={k: v for k, v in KITTI_CATEGORY_MAP.items() if v})

    embedded_gt = str(Path(scene_dir) / "gt.json")
    write_gt_json(embedded_gt, gt_frames)
    print(f"GT embedded in scene: {embedded_gt}")

    if gt_output:
        write_gt_json(gt_output, gt_frames)
        print(f"GT also written to {gt_output}")

    print(f"Sequence converted to {scene_dir}")


def convert_all(dataroot, output_dir, max_frames=None, no_images=False):
    dataroot_training = Path(dataroot) / "training"
    oxts_dir = dataroot_training / "oxts"
    sequences = sorted([int(f.stem) for f in oxts_dir.glob("*.txt")])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting all {len(sequences)} KITTI tracking sequences")
    for seq_id in sequences:
        seq_out = str(output_dir / f"{seq_id:04d}")
        convert_sequence(dataroot, seq_id, seq_out,
                         max_frames=max_frames, no_images=no_images)
    print(f"\nAll {len(sequences)} sequences converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KITTI tracking sequence(s) to SensorLens format")
    parser.add_argument("--dataroot", required=True,
                        help="Path to KITTI tracking root (containing training/)")
    parser.add_argument("--output", required=True, help="Output directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sequence", type=int, help="Single sequence number (0-based)")
    group.add_argument("--all", action="store_true", help="Convert all sequences")

    parser.add_argument("--gt-output", help="Extra copy of GT JSON (single sequence only)")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames per sequence")
    parser.add_argument("--no-images", action="store_true", help="Skip camera images")
    args = parser.parse_args()

    if args.all:
        convert_all(args.dataroot, args.output,
                    max_frames=args.max_frames, no_images=args.no_images)
    else:
        seq_id = args.sequence if args.sequence is not None else 0
        convert_sequence(args.dataroot, seq_id, args.output,
                         gt_output=args.gt_output, max_frames=args.max_frames,
                         no_images=args.no_images)


if __name__ == "__main__":
    main()
