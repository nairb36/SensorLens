# SensorLens

Interactive 3D visualization tool for multi-object tracking on nuScenes data.

## Features

- **3D LiDAR point cloud** with height-based coloring (WebGL-accelerated)
- **3D wireframe bounding boxes** for ground-truth detections and tracker output
- **Identity-consistent coloring** across frames (by instance token / track ID)
- **6-camera synchronized view** (front-left, front, front-right, back-left, back, back-right)
- **Playback controls**: step forward/backward, play/pause, frame slider
- **Toggle overlays**: show/hide GT and tracker independently or together

## Usage

```bash
cd Project_SensorLens

python run.py \
  --gt ../Project_MOT/results/gt/scene_0000.json \
  --tracker ../Project_MOT/results/tracking/results_20260511_233838.json
```

Then open http://localhost:8050 in your browser.

### Arguments

| Argument     | Default                      | Description                     |
|-------------|------------------------------|---------------------------------|
| `--dataroot`| `/workspace/data/nuscenes`   | Path to nuScenes dataset        |
| `--version` | `v1.0-mini`                  | nuScenes version                |
| `--gt`      | —                            | Path to GT detections JSON      |
| `--tracker` | —                            | Path to tracker results JSON    |
| `--port`    | `8050`                       | Server port                     |
| `--host`    | `0.0.0.0`                    | Host to bind to                 |

At least one of `--gt` or `--tracker` must be provided.

## Controls

- **Prev / Next**: step one frame at a time
- **Play / Pause**: auto-advance at ~2 FPS
- **Frame slider**: jump to any frame
- **GT / Tracker buttons**: toggle each overlay on/off
- **3D view**: orbit (drag), zoom (scroll), pan (right-drag)

## Requirements

```
pip install -r requirements.txt
```

Requires the nuScenes mini dataset at the specified `--dataroot`.
