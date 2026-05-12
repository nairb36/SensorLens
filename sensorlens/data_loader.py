import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class NuScenesLoader:
    def __init__(self, dataroot: str, version: str = "v1.0-mini"):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    def get_sample(self, sample_token: str):
        return self.nusc.get("sample", sample_token)

    def get_ego_pose(self, sample_token: str) -> dict:
        sample = self.get_sample(sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        return self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

    def get_lidar_points_ego(self, sample_token: str) -> np.ndarray:
        sample = self.get_sample(sample_token)
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = Path(self.nusc.dataroot) / lidar_sd["filename"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
        pts = points[:, :4]  # x, y, z, intensity

        cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        rot = Quaternion(cs["rotation"]).rotation_matrix
        trans = np.array(cs["translation"])
        pts_ego = pts.copy()
        pts_ego[:, :3] = pts[:, :3] @ rot.T + trans
        return pts_ego

    def get_camera_paths(self, sample_token: str) -> dict[str, str]:
        sample = self.get_sample(sample_token)
        cam_order = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ]
        paths = {}
        for cam in cam_order:
            cam_data = self.nusc.get("sample_data", sample["data"][cam])
            paths[cam] = str(Path(self.nusc.dataroot) / cam_data["filename"])
        return paths


def global_to_ego(translation: list, yaw: float, ego_pose: dict) -> tuple[np.ndarray, float]:
    ego_trans = np.array(ego_pose["translation"])
    ego_rot = Quaternion(ego_pose["rotation"])
    pos = np.array(translation) - ego_trans
    pos = ego_rot.rotation_matrix.T @ pos
    ego_yaw = ego_rot.yaw_pitch_roll[0]
    local_yaw = yaw - ego_yaw
    return pos, local_yaw


def load_gt_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_tracker_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


CATEGORY_GROUPS = {
    "Pedestrians": [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
    ],
    "Cars": ["vehicle.car"],
    "Trucks & Buses": ["vehicle.truck", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.construction"],
    "Two-Wheelers": ["vehicle.motorcycle", "vehicle.bicycle"],
    "Static Objects": [
        "movable_object.barrier",
        "movable_object.trafficcone",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "static_object.bicycle_rack",
        "vehicle.trailer",
    ],
}

CATEGORY_TO_GROUP = {}
for group, cats in CATEGORY_GROUPS.items():
    for cat in cats:
        CATEGORY_TO_GROUP[cat] = group

DEFAULT_ON = {"Pedestrians", "Cars", "Trucks & Buses", "Two-Wheelers"}


def shorten_category(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2:
        return parts[1]
    return name
