#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from sensorlens.data_loader import UniversalLoader
from sensorlens.app import create_demo_app

data_dir = Path(__file__).resolve().parent / "data" / "scene_0061"

scene_loader = UniversalLoader(str(data_dir))
gt_data = scene_loader.load_gt() if scene_loader.has_gt() else None

demo_state = {
    "scene_loader": scene_loader,
    "camera_names": scene_loader.camera_names,
    "gt_data": gt_data,
    "tracker_data": None,
    "num_frames": scene_loader.num_frames,
    "scene_mismatch": False,
    "app_mode": "visualization",
    "mot_accumulator": None,
    "mot_id_map": None,
    "mot_summary": None,
    "scene_name": "scene-0061",
}

print(f"Loaded {demo_state['num_frames']} frames, {len(demo_state['camera_names'])} cameras")

app = create_demo_app(demo_state)
server = app.server

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
