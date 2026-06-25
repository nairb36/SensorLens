#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from sensorlens.data_loader import UniversalLoader
from sensorlens.app import create_demo_app

data_root = Path(__file__).resolve().parent / "data"

SCENES = [
    ("nuscenes", "scene-0061", "scene_0061"),
    ("kitti", "KITTI 0000", "kitti_0000"),
    ("waymo", "Waymo 1005", "waymo_1005"),
    ("argoverse2", "Argoverse 02678d04", "argo_02678d04"),
]

demo_scenes = {}
for dataset, display_name, dir_name in SCENES:
    scene_dir = data_root / dir_name
    loader = UniversalLoader(str(scene_dir))
    gt_data = loader.load_gt() if loader.has_gt() else None
    demo_scenes[dir_name] = {
        "scene_loader": loader,
        "camera_names": loader.camera_names,
        "gt_data": gt_data,
        "tracker_data": None,
        "num_frames": loader.num_frames,
        "scene_mismatch": False,
        "app_mode": "visualization",
        "mot_accumulator": None,
        "mot_id_map": None,
        "mot_summary": None,
        "scene_name": display_name,
        "dataset": dataset,
    }
    print(f"Loaded {display_name}: {loader.num_frames} frames, {len(loader.camera_names)} cameras")

app = create_demo_app(demo_scenes)
server = app.server

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
