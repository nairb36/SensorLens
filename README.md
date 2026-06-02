# SensorLens

Interactive 3D visualization and debugging tool for multi-object tracking (MOT) on autonomous driving datasets. Think of it as an **IDE for your tracker** — just as a code IDE lets you set breakpoints and inspect variables, SensorLens lets you step through frames, inspect every bounding box, and see exactly where your tracker fails (ID switches, false positives, missed detections) with color-coded diagnostics.

Built for visual debugging and comparison of tracker output against ground-truth annotations.

## Two Modes

### Visualization Mode

Pure playback for inspecting scenes. Load ground-truth detections, tracker output, or both, and explore them frame-by-frame in an interactive 3D scene with LiDAR point clouds, camera panoramas, and bounding box overlays.

![Visualization Mode](sensorlens/assets/MOT_V1.gif)

### Debug Mode

**The core differentiator.** Debug mode runs a full CLEAR MOT evaluation (via `motmetrics`) when you launch, then overlays the results directly onto the 3D scene:

![Debug Mode](sensorlens/assets/SensorLens_debug_mode.gif)

- **Color-coded boxes** — every GT and tracker box is colored by its MOT event type:
  - Green = correct match
  - Red = ID switch (thicker wireframe)
  - Yellow = false positive
  - Blue = missed detection
- **Per-frame debug log** — a panel showing match counts, ID switch details (which GT matched which tracker ID), false positive IDs, and missed detection IDs
- **Tracking metrics dashboard** — collapsible panel with MOTA, MOTP, IDF1, Recall, Precision, ID Switches, Fragmentations, and more
- **Configurable evaluation** — set match distance threshold and filter which object categories to evaluate
- **Auto-play disabled** — frame stepping only, so you can inspect each frame carefully

This is analogous to running a debugger on your code: instead of just seeing "MOTA = 72%", you can step to the exact frame where an ID switch happened, see which objects were involved, and understand *why* it failed.

## Features

### Browser-Based Configuration
- **No CLI data arguments** — launch with `python3 run.py` and configure everything in the browser
- **Dataset type selection** — NuScenes, Waymo Open Dataset (coming soon), or Custom (bounding boxes only)
- **File upload or path entry** — drag-and-drop JSON files or enter filesystem paths for GT/tracker data
- **Scene mismatch detection** — warns when GT and tracker files are from different scenes
- **Multi-scene workflow** — home button returns to config page to load a new scene without restarting

### 3D Scene View
- **LiDAR point cloud** rendered with height-based Turbo colormap (WebGL-accelerated via Plotly), with toggle to switch to white point cloud
- **3D wireframe bounding boxes** for tracker output with identity-consistent coloring and per-box ID tags
- **Solid semi-transparent bounding boxes** for ground-truth detections with identity-consistent coloring
- **3D ego vehicle model** loaded from OBJ/MTL with per-face material colors and a forward-direction indicator
- **Category filtering** — toggle visibility of Pedestrians, Cars, Trucks/Buses, Two-Wheelers, and Static Objects independently
- **Layer toggles** — show/hide GT bounding boxes, GT centers, tracker bounding boxes, and tracker centers independently
- **2D/3D view toggle** — switch between free orbit (3D) and top-down turntable (2D) modes; zoom and pan persist between frames
- **Hover info** — hover over any box to see category, identity, and (for tracker boxes) age/hits/misses metadata

### Custom Mode (No Dataset Required)
- **Bounding box only rendering** — visualize GT and tracker JSONs without a NuScenes/Waymo dataset
- **No LiDAR or camera data needed** — boxes are rendered on a clean dark background
- **Auto-centering** — scene is centered on the first frame's object centroid
- **Assumes ego-frame coordinates** — translations in the JSON are used directly (no global-to-ego transform)

### Camera Panoramas (NuScenes)
- **Stitched front panorama** from CAM_FRONT_LEFT, CAM_FRONT, and CAM_FRONT_RIGHT (~180 deg FOV)
- **Stitched rear panorama** from CAM_BACK_LEFT, CAM_BACK, and CAM_BACK_RIGHT (~180 deg FOV)
- Cylindrical projection with precomputed remap tables using camera intrinsics and extrinsics
- Weighted blending in overlap regions for seamless transitions

### Playback
- **Prev / Next** buttons for frame-by-frame stepping
- **Play / Pause** auto-advance at ~2 FPS (nuScenes keyframe rate) — available in Visualization mode only
- **Frame slider** to jump to any frame
- **Frame info bar** showing frame index, object count, sample token, and timestamp

## Installation

### Local Install

```bash
cd Project_SensorLens
pip install -r requirements.txt
```

#### Dependencies

- `dash` / `plotly` — web UI and 3D rendering
- `numpy` — point cloud and geometry operations
- `opencv-python` — panorama stitching (cv2.remap)
- `nuscenes-devkit` — dataset access (samples, calibration, ego poses)
- `pyquaternion` — rotation handling
- `motmetrics` — CLEAR MOT evaluation for debug mode
- `Pillow` — image handling

### Docker

Build and run SensorLens as a Docker container — no local Python setup needed.

#### Using Docker Compose (recommended)

```bash
# Build and start the container
docker compose up --build

# Or run in detached mode
docker compose up --build -d
```

#### Using Docker directly

```bash
# Build the image
docker build -t sensorlens:latest .

# Run the container
docker run -p 8050:8050 sensorlens:latest
```

Then open http://localhost:8050 in your browser.

#### Mounting NuScenes data

To use NuScenes mode inside Docker, you need to mount your local dataset into the container:

1. Create a `.env` file in the project root:
   ```
   NUSCENES_PATH=/path/to/your/nuscenes/v1.0-mini
   ```

2. Uncomment the `volumes` section in `docker-compose.yml`:
   ```yaml
   volumes:
     - ${NUSCENES_PATH}:/data/nuscenes:ro
   ```

3. Run `docker compose up --build` and enter `/data/nuscenes` as the dataroot path in the browser config page.

Without mounting data, you can still use **Custom mode** by uploading JSON files through the browser.

> **Note:** The default `docker-compose.yml` targets `linux/arm64` (Apple Silicon). Remove or change the `platform` line for other architectures.

## Usage

```bash
python3 run.py
```

Then open http://localhost:8050 in your browser. The configuration page lets you:

1. Select mode — **Visualization** or **Debug**
2. Select dataset type (NuScenes / Custom)
3. Enter dataroot path and version (NuScenes only)
4. Upload or enter paths for GT and/or tracker JSON files
5. (Debug mode) Set match distance threshold and select evaluation categories
6. Click **Launch** to start

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
| Toggle point cloud color | Circle button in controls bar |
| Step frame | Prev / Next buttons |
| Auto-play | Play / Pause button (Visualization mode only) |
| Jump to frame | Drag the frame slider |
| Toggle GT / Tracker layers | Layer checkboxes (bbox, center) in overlay panel |
| Filter categories | Category checkboxes in overlay panel |
| Expand tracking metrics | Metrics button (Debug mode only) |
| Return to config | Home button (top-left) |

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
        "yaw": -0.4034,
        "age": 5,
        "hits": 5,
        "consecutive_misses": 0
      }
    ]
  }
]
```

- `id`: integer track ID (drives consistent coloring across frames)
- `age`, `hits`, `consecutive_misses`: optional tracker metadata shown on hover

## Architecture

```
sensorlens/
  app.py             -- Dash app: config page, viz/debug layouts, all callbacks
  data_loader.py     -- NuScenes data access, coordinate transforms, category mapping
  scene_builder.py   -- 3D figure construction (point cloud, boxes, ego car model)
  image_stitcher.py  -- Cylindrical panorama stitching with precomputed remap tables
  mot_evaluator.py   -- CLEAR MOT evaluation, per-frame event extraction, metrics
  assets/
    style.css        -- UI styling (config page, checkboxes, buttons)
    NormalCar2.obj   -- 3D ego vehicle model (Blender export)
    NormalCar2.mtl   -- Material definitions
run.py               -- CLI entry point
Dockerfile           -- Container image definition
docker-compose.yml   -- Docker Compose configuration
docker/
  constraints.txt    -- Pip version constraints for Docker builds
  patch_motmetrics.py -- Numpy compatibility patch for motmetrics
```
