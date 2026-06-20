import json
from pathlib import Path

import numpy as np


class UniversalLoader:
    """Loads scenes in the SensorLens universal format."""

    def __init__(self, scene_dir: str):
        self.scene_dir = Path(scene_dir)
        meta_path = self.scene_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta.json not found in {scene_dir}")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.num_frames = self.meta["num_frames"]
        self.camera_names = [
            c["name"] for c in
            sorted(self.meta["sensors"].get("cameras", []), key=lambda c: c["order"])
        ]
        self._frame_cache: dict[int, dict] = {}

    def get_frame_meta(self, frame_index: int) -> dict:
        if frame_index not in self._frame_cache:
            path = self.scene_dir / "frames" / f"{frame_index:06d}.json"
            with open(path) as f:
                self._frame_cache[frame_index] = json.load(f)
        return self._frame_cache[frame_index]

    def get_pointcloud(self, frame_index: int) -> np.ndarray:
        frame = self.get_frame_meta(frame_index)
        pc_path = self.scene_dir / frame["pointcloud_file"]
        if not pc_path.is_file():
            return np.empty((0, 4), dtype=np.float32)
        channels = self.meta["sensors"]["lidar"]["points_per_row"]
        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, channels)
        return points[:, :4]

    def get_camera_paths(self, frame_index: int) -> dict[str, str]:
        frame = self.get_frame_meta(frame_index)
        result = {}
        for cam_name, rel_path in frame.get("camera_files", {}).items():
            full = self.scene_dir / rel_path
            if full.is_file():
                result[cam_name] = str(full)
        return result

    def get_timestamp(self, frame_index: int) -> int:
        return self.get_frame_meta(frame_index).get("timestamp", 0)


def load_gt_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_tracker_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


CATEGORY_GROUPS = {
    "Pedestrians": ["pedestrian"],
    "Cars": ["car"],
    "Trucks & Buses": ["truck", "bus", "construction_vehicle"],
    "Two-Wheelers": ["motorcycle", "bicycle"],
    "Static Objects": ["barrier", "traffic_cone", "trailer"],
}

CATEGORY_TO_GROUP: dict[str, str] = {}
for _group, _cats in CATEGORY_GROUPS.items():
    for _cat in _cats:
        CATEGORY_TO_GROUP[_cat] = _group

DEFAULT_ON = {"Pedestrians", "Cars", "Trucks & Buses", "Two-Wheelers"}


def shorten_category(name: str) -> str:
    return name
