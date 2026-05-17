# SensorLens

Interactive 3D visualization tool for multi-object tracking on nuScenes data. Built for visual debugging and comparison of tracker output against ground-truth annotations.

![SensorLens Demo](sensorlens/assets/MOT_V1.gif)

## Features

### Browser-Based Configuration
- **No CLI data arguments** — launch with `python3 run.py` and configure everything in the browser
- **Dataset type selection** — NuScenes, Waymo Open Dataset (coming soon), or Custom (bounding boxes only)
- **File upload or path entry** — drag-and-drop JSON files or enter filesystem paths for GT/tracker data
- **Scene mismatch detection** — warns when GT and tracker files are from different scenes
- **Multi-scene workflow** — home button returns to config page to load a new scene without restarting

### 3D Scene View
- **LiDAR point cloud** rendered with height-based Turbo colormap (WebGL-accelerated via Plotly)
- **3D wireframe bounding boxes** for ground-truth detections and/or tracker output, displayed simultaneously with distinct identity-consistent coloring
- **3D ego vehicle model** loaded from OBJ/MTL with per-face material colors and a forward-direction indicator
- **Category filtering** — toggle visibility of pedestrians, cars, trucks/buses, two-wheelers, and static objects independently
- **Layer toggles** — show/hide GT and tracker overlays independently or together
- **2D/3D view toggle** — switch between free orbit (3D) and top-down turntable (2D) modes; zoom and pan persist between frames in both modes

### Custom Mode (No Dataset Required)
- **Bounding box only rendering** — visualize GT and tracker JSONs without a NuScenes/Waymo dataset
- **No LiDAR or camera data needed** — boxes are rendered on a clean dark background
- **Auto-centering** — scene is centered on the first frame's object centroid
- **Assumes ego-frame coordinates** — translations in the JSON are used directly (no global-to-ego transform)

### Camera Panoramas
- **Stitched front panorama** from CAM_FRONT_LEFT, CAM_FRONT, and CAM_FRONT_RIGHT (~180° FOV)
- **Stitched rear panorama** from CAM_BACK_LEFT, CAM_BACK, and CAM_BACK_RIGHT (~180° FOV)
- Cylindrical projection with precomputed remap tables using camera intrinsics and extrinsics from nuScenes calibration data
- Weighted blending in overlap regions for seamless transitions

### Playback
- **⏮ / ⏭** buttons for frame-by-frame stepping
- **▶ / ⏸** auto-advance at ~2 FPS (nuScenes keyframe rate)
- **Frame slider** to jump to any frame
- **Frame info bar** showing frame index, object count, sample token, and timestamp

## Installation

```bash
cd Project_SensorLens
pip install -r requirements.txt
```

### Dependencies

- `dash` / `plotly` — web UI and 3D rendering
- `numpy` — point cloud and geometry operations
- `opencv-python` — panorama stitching (cv2.remap)
- `nuscenes-devkit` — dataset access (samples, calibration, ego poses)
- `pyquaternion` — rotation handling

## Usage

```bash
python3 run.py
```

Then open http://localhost:8050 in your browser. The configuration page lets you:

1. Select dataset type (NuScenes / Custom)
2. Enter dataroot path and version (NuScenes only)
3. Upload or enter paths for GT and/or tracker JSON files
4. Click **Launch** to start visualization

### CLI Arguments

| Argument | Default  | Description       |
|----------|----------|-------------------|
| `--port` | `8050`   | Server port       |
| `--host` | `0.0.0.0`| Host to bind to  |

## Controls

| Action | Input |
|--------|-------|
| Orbit 3D view | Left-click drag (3D mode) |
| Rotate around Z only | Left-click drag (2D mode) |
| Zoom | Scroll wheel |
| Pan | Right-click drag |
| Toggle 2D/3D | 2D/3D button in controls bar |
| Step frame | ⏮ / ⏭ buttons |
| Auto-play | ▶ / ⏸ button |
| Jump to frame | Drag the frame slider |
| Toggle GT / Tracker | Layer checkboxes (top-left overlay) |
| Filter categories | Category checkboxes (top-left overlay) |
| Return to config | ⌂ home button (top-left) |

## Data Formats

### GT Detections JSON

Array of frames, each containing detections in global coordinates:

```json
[
  {
    "sample_token": "ca9a282c9e77460f...",
    "timestamp": 1532402927647951,
    "detections": [
      {
        "instance_token": "6dd2cbf4c24b4cae...",
        "category_name": "vehicle.car",
        "translation": [353.794, 1132.355, 0.602],
        "size": [2.011, 4.633, 1.573],
        "yaw": -0.4034
      }
    ]
  }
]
```

- `translation`: [x, y, z] in global frame (meters) — or ego frame for Custom mode
- `size`: [width, length, height] in meters
- `yaw`: rotation about z-axis (radians)
- `instance_token`: unique object identity across frames (drives consistent coloring)
- `category_name`: nuScenes category (e.g. `vehicle.car`, `human.pedestrian.adult`)

### Tracker Output JSON

```json
[
  {
    "frame_id": 0,
    "timestamp": 1532402927647951.0,
    "tracks": [
      {
        "id": 0,
        "category_name": "vehicle.car",
        "translation": [353.8, 1132.4, 0.6],
        "size": [2.011, 4.633, 1.573],
        "yaw": -0.4034
      }
    ]
  }
]
```

- `id`: integer track ID (drives consistent coloring across frames)

## Architecture

```
sensorlens/
  app.py             — Dash application layout, config page, and callbacks
  data_loader.py     — nuScenes data access, coordinate transforms, category mapping
  scene_builder.py   — 3D figure construction (point cloud, boxes, ego car model)
  image_stitcher.py  — Cylindrical panorama stitching with precomputed remap tables
  assets/
    style.css        — UI styling (config page, checkboxes, buttons)
    NormalCar2.obj   — 3D ego vehicle model (Blender export)
    NormalCar2.mtl   — Material definitions (7 materials)
run.py               — CLI entry point
```
