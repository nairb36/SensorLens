import json
import shutil
from pathlib import Path

import numpy as np


NUSCENES_CATEGORY_MAP = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.debris": "barrier",
    "movable_object.pushable_pullable": "barrier",
    "static_object.bicycle_rack": "barrier",
}

WAYMO_CATEGORY_MAP = {
    1: "car",        # TYPE_VEHICLE
    2: "pedestrian",  # TYPE_PEDESTRIAN
    3: "barrier",     # TYPE_SIGN
    4: "bicycle",     # TYPE_CYCLIST
}

KITTI_CATEGORY_MAP = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "bicycle",
    "Tram": "bus",
    "Misc": None,
    "DontCare": None,
}

ARGOVERSE2_CATEGORY_MAP = {
    "REGULAR_VEHICLE": "car",
    "LARGE_VEHICLE": "truck",
    "BUS": "bus",
    "BOX_TRUCK": "truck",
    "TRUCK": "truck",
    "VEHICULAR_TRAILER": "trailer",
    "TRUCK_CAB": "truck",
    "SCHOOL_BUS": "bus",
    "ARTICULATED_BUS": "bus",
    "MESSAGE_BOARD_TRAILER": "trailer",
    "PEDESTRIAN": "pedestrian",
    "STROLLER": "pedestrian",
    "WHEELCHAIR": "pedestrian",
    "OFFICIAL_SIGNALER": "pedestrian",
    "BICYCLE": "bicycle",
    "BICYCLIST": "bicycle",
    "MOTORCYCLE": "motorcycle",
    "MOTORCYCLIST": "motorcycle",
    "WHEELED_DEVICE": "bicycle",
    "WHEELED_RIDER": "bicycle",
    "DOG": "pedestrian",
    "BOLLARD": "barrier",
    "CONSTRUCTION_BARREL": "barrier",
    "CONSTRUCTION_CONE": "traffic_cone",
    "SIGN": "barrier",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": "barrier",
    "STOP_SIGN": "barrier",
    "RAILED_VEHICLE": "bus",
    "ANIMAL": None,
}


def quaternion_to_yaw(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def ensure_dirs(scene_dir):
    scene_dir = Path(scene_dir)
    (scene_dir / "frames").mkdir(parents=True, exist_ok=True)
    (scene_dir / "pointclouds").mkdir(parents=True, exist_ok=True)
    (scene_dir / "cameras").mkdir(parents=True, exist_ok=True)
    return scene_dir


def write_meta_json(scene_dir, source_dataset, scene_name, num_frames,
                    camera_names, category_mapping=None):
    meta = {
        "format_version": "1.0",
        "source_dataset": source_dataset,
        "scene_name": scene_name,
        "num_frames": num_frames,
        "sensors": {
            "lidar": {
                "points_format": "float32",
                "points_per_row": 4,
                "channels": ["x", "y", "z", "intensity"],
            },
            "cameras": [
                {"name": name, "order": i}
                for i, name in enumerate(camera_names)
            ],
        },
    }
    if category_mapping:
        meta["category_mapping"] = category_mapping
    with open(Path(scene_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def write_frame_json(scene_dir, frame_index, timestamp, ego_pose=None,
                     camera_files=None):
    pc_rel = f"pointclouds/{frame_index:06d}.bin"
    cam_rel = {}
    if camera_files:
        for cam_name in camera_files:
            cam_rel[cam_name] = f"cameras/{frame_index:06d}/{cam_name}.jpg"

    frame = {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "pointcloud_file": pc_rel,
        "camera_files": cam_rel,
    }
    if ego_pose:
        frame["ego_pose"] = ego_pose

    path = Path(scene_dir) / "frames" / f"{frame_index:06d}.json"
    with open(path, "w") as f:
        json.dump(frame, f, indent=2)


def write_pointcloud(scene_dir, frame_index, points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 2 and pts.shape[1] > 4:
        pts = pts[:, :4]
    elif pts.ndim == 2 and pts.shape[1] == 3:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    path = Path(scene_dir) / "pointclouds" / f"{frame_index:06d}.bin"
    pts.tofile(str(path))


def copy_camera_image(scene_dir, frame_index, camera_name, src_path):
    dst_dir = Path(scene_dir) / "cameras" / f"{frame_index:06d}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{camera_name}.jpg"
    src = Path(src_path)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(str(src), str(dst))
    else:
        import cv2
        img = cv2.imread(str(src))
        if img is not None:
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_gt_json(out_path, frames):
    with open(out_path, "w") as f:
        json.dump(frames, f, indent=2)


def write_tracker_json(out_path, frames):
    with open(out_path, "w") as f:
        json.dump(frames, f, indent=2)
