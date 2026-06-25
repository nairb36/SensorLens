#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from sensorlens.data_loader import UniversalLoader, load_tracker_json
from sensorlens.mot_evaluator import run_evaluation, compute_summary
from sensorlens.app import create_demo_app

data_root = Path(__file__).resolve().parent / "data"

SCENES = [
    ("nuscenes", "scene-0061", "scene_0061", "visualization"),
    ("nuscenes", "scene-0061", "scene_0061", "debug"),
    ("kitti", "KITTI 0000", "kitti_0000", "visualization"),
    ("waymo", "Waymo 1005", "waymo_1005", "visualization"),
    ("argoverse2", "Argoverse 02678d04", "argo_02678d04", "visualization"),
]

demo_scenes = {}
for dataset, display_name, dir_name, mode in SCENES:
    scene_dir = data_root / dir_name
    loader = UniversalLoader(str(scene_dir))
    gt_data = loader.load_gt() if loader.has_gt() else None

    tracker_data = None
    mot_acc = None
    mot_id_map = None
    mot_summary = None

    if mode == "debug":
        tracker_path = scene_dir / "tracker_output.json"
        if tracker_path.exists():
            tracker_data = load_tracker_json(str(tracker_path))
            if gt_data and tracker_data:
                mot_acc, mot_id_map = run_evaluation(gt_data, tracker_data, max_dist=2.0)
                mot_summary = compute_summary(mot_acc)
                print(f"  MOT evaluation: MOTA={mot_summary.get('mota', 'N/A')}")

    scene_key = f"{dir_name}_debug" if mode == "debug" else dir_name
    demo_scenes[scene_key] = {
        "scene_loader": loader,
        "camera_names": loader.camera_names,
        "gt_data": gt_data,
        "tracker_data": tracker_data,
        "num_frames": loader.num_frames,
        "scene_mismatch": False,
        "app_mode": mode,
        "mot_accumulator": mot_acc,
        "mot_id_map": mot_id_map,
        "mot_summary": mot_summary,
        "scene_name": display_name,
        "dataset": dataset,
    }
    print(f"Loaded {display_name} [{mode}]: {loader.num_frames} frames, {len(loader.camera_names)} cameras")

app = create_demo_app(demo_scenes)
server = app.server

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
