# SensorLens

Interactive 3D visualization and debugging tool for multi-object tracking (MOT) on autonomous driving datasets. Think of it as an **IDE for your tracker** — just as a code IDE lets you set breakpoints and inspect variables, SensorLens lets you step through frames, inspect every bounding box, and see exactly where your tracker fails (ID switches, false positives, missed detections) with color-coded diagnostics.

Built for visual debugging and comparison of tracker output against ground-truth annotations.

## Two Modes

### Visualization Mode

Pure playback for inspecting scenes. Load a converted scene directory and explore frame-by-frame in an interactive 3D view with LiDAR point clouds, camera images, and bounding box overlays.

![Visualization Mode](sensorlens/assets/MOT_V1.gif)

### Debug Mode

**The core differentiator.** Debug mode runs a full CLEAR MOT evaluation (via `motmetrics`) when you launch, then overlays the results directly onto the 3D scene:

![Debug Mode](sensorlens/assets/SensorLens_debug_mode.gif)

- **Color-coded boxes** — every GT and tracker box is colored by its MOT event type:
  - Green = correct match
  - Red = ID switch (thicker wireframe)
  - Yellow = false positive
  - Blue = missed detection
- **Per-frame debug log** — match counts, ID switch details, false positive IDs, and missed detection IDs
- **Tracking metrics dashboard** — MOTA, MOTP, IDF1, Recall, Precision, ID Switches, Fragmentations
- **Configurable evaluation** — set match distance threshold and filter which object categories to evaluate
- **Auto-play disabled** — frame stepping only for careful inspection

This is analogous to running a debugger on your code: instead of just seeing "MOTA = 72%", you can step to the exact frame where an ID switch happened, see which objects were involved, and understand *why* it failed.

## Features

### Browser-Based Configuration
- **No CLI data arguments** — launch with `python3 run.py` and configure everything in the browser
- **Scene directory input** — point to any converted scene for full visualization (LiDAR + cameras + GT)
- **File upload or path entry** — drag-and-drop JSON files or enter filesystem paths for GT/tracker data
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

### Camera Views
- Individual camera images displayed in a grid layout
- Adapts to the number of cameras available (6 for nuScenes, 1 for KITTI, etc.)
- No cameras required — works with just point clouds or even boxes only

### Playback
- **Prev / Next** buttons for frame-by-frame stepping
- **Play / Pause** auto-advance at ~2 FPS — available in Visualization mode only
- **Frame slider** to jump to any frame
- **Frame info bar** showing frame index, object count, and timestamp

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

---

## Universal Scene Format

SensorLens defines a **dataset-agnostic universal scene format** that all supported datasets convert into. The idea: convert once from any dataset's native format into a common interface, then use the same visualization and debugging tools regardless of where the data came from.

This means:
- **Your tracker code doesn't need to know which dataset it's running on** — the input format is always the same
- **SensorLens has zero dataset-specific code** — it only reads the universal format
- **Adding a new dataset = writing one converter** — the viewer and evaluator work immediately

### Format Structure

A converted scene is a self-contained directory:

```
scene_name/
  meta.json              # scene metadata, sensor list, source info
  gt.json                # ground-truth annotations (embedded)
  frames/
    000000.json          # per-frame: timestamp, ego_pose, file refs
    000001.json
  pointclouds/
    000000.bin           # float32 Nx4 binary (x, y, z, intensity) in ego frame
  cameras/
    000000/
      front.jpg          # camera images (dataset-dependent names)
      front_left.jpg
```

### Coordinate Convention

- **Point clouds** are stored in **ego frame** (sensor-relative, centered on the vehicle)
- **Detections and tracks** are stored in **global/map frame**
- **Ego pose** (translation + quaternion) is stored per frame
- At render time, SensorLens transforms global coordinates to ego frame using the ego pose

This design ensures the MOT evaluator works correctly regardless of dataset (pairwise distances are frame-invariant), while the viewer always shows objects relative to the ego vehicle.

### Universal Category Taxonomy

All datasets map to a shared set of category names:

| Group | Categories |
|-------|-----------|
| Pedestrians | `pedestrian` |
| Cars | `car` |
| Trucks & Buses | `truck`, `bus`, `construction_vehicle` |
| Two-Wheelers | `motorcycle`, `bicycle` |
| Static Objects | `barrier`, `traffic_cone`, `trailer` |

## Supported Datasets

### nuScenes

Full support — LiDAR, 6 cameras (360 degree coverage), and complete 3D annotations.

```bash
# Convert all scenes from nuScenes mini
python3 -m converters.convert_nuscenes \
  --dataroot /path/to/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --all \
  --output /path/to/sensorlens_scenes/nuscenes/

# Convert a single scene
python3 -m converters.convert_nuscenes \
  --dataroot /path/to/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --scene 0 \
  --output /path/to/sensorlens_scenes/nuscenes/scene-0061
```

**Required data:** nuScenes dataset (any version — mini, trainval, test)

**What you get:** LiDAR point clouds, 6 camera views (front_left, front, front_right, back_left, back, back_right), full 360-degree annotated ground truth

### KITTI Tracking

Full support — LiDAR, left camera, 3D annotations in the camera field of view.

```bash
# Convert all 21 training sequences
python3 -m converters.convert_kitti \
  --dataroot /path/to/kitti \
  --all \
  --output /path/to/sensorlens_scenes/kitti/

# Convert a single sequence
python3 -m converters.convert_kitti \
  --dataroot /path/to/kitti \
  --sequence 0 \
  --output /path/to/sensorlens_scenes/kitti/0000
```

**Required data (from KITTI tracking benchmark):**
- Camera calibration matrices (`data_tracking_calib.zip`, 1 MB)
- Training labels (`data_tracking_label_2.zip`, 9 MB)
- GPS/IMU data (`data_tracking_oxts.zip`, 64 MB)

**Optional data (for full visualization):**
- Left color images (`data_tracking_image_2.zip`, 12 GB)
- Velodyne point clouds (`data_tracking_velodyne.zip`, 29 GB)

Place the extracted data at:
```
{dataroot}/training/
  calib/0000.txt ... 0020.txt
  label_02/0000.txt ... 0020.txt
  oxts/0000.txt ... 0020.txt
  velodyne/0000/000000.bin ...     (optional)
  image_02/0000/000000.png ...     (optional)
```

The converter works with just labels + calib + oxts (annotations only). When velodyne/image data is present, it automatically includes point clouds and camera images in the converted scenes — no code changes needed.

**Note:** KITTI only annotates objects visible in the front-facing camera (~90 degree FOV), so ground-truth boxes only appear in front of the ego vehicle.

### Waymo Open Dataset (v2)

Full support — TOP + 4 side LiDARs, 5 cameras, and 3D LiDAR box annotations. Uses the v2 Parquet format directly — no TensorFlow or `waymo-open-dataset` package required.

```bash
# Convert all segments in a download directory
python3 -m converters.convert_waymo \
  --dataroot /path/to/waymo_v2 \
  --all \
  --output /path/to/sensorlens_scenes/waymo/

# Convert a single segment
python3 -m converters.convert_waymo \
  --dataroot /path/to/waymo_v2 \
  --segment 10017090168044687777_6380_000_6400_000 \
  --output /path/to/sensorlens_scenes/waymo/10017090168044687777_6380_000_6400_000

# Fast conversion: TOP lidar only, skip camera images
python3 -m converters.convert_waymo \
  --dataroot /path/to/waymo_v2 \
  --segment 10017090168044687777_6380_000_6400_000 \
  --output /path/to/output \
  --top-lidar-only --no-images
```

**Required data (Waymo v2 Parquet components per segment):**
- `lidar_box` — 3D LiDAR box annotations
- `vehicle_pose` — ego vehicle pose (world_from_vehicle 4x4 matrix)
- `lidar` — range images for all 5 LiDARs
- `lidar_calibration` — per-LiDAR extrinsic and beam inclinations

**Optional data:**
- `camera_image` — 5 camera images (front, front_left, front_right, side_left, side_right)

Download individual components using `gsutil`:
```bash
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_2_0_1/training/lidar_box/10017090168044687777_6380_000_6400_000.parquet" \
  /path/to/waymo_v2/lidar_box/
# Repeat for vehicle_pose, lidar, lidar_calibration, camera_image
```

Place the downloaded Parquet files at:
```
{dataroot}/
  lidar_box/{segment_name}.parquet
  vehicle_pose/{segment_name}.parquet
  lidar/{segment_name}.parquet
  lidar_calibration/{segment_name}.parquet
  camera_image/{segment_name}.parquet        (optional)
```

**What you get:** combined point cloud from all 5 LiDARs (~170k points/frame), 5 camera views, full 3D annotated ground truth with tracked object identities

**Converter dependencies:** `pyarrow`, `pandas` (no TensorFlow, no Waymo SDK)

### Argoverse 2

Coming soon.

## Data Formats

### GT Detections JSON

Array of frames, each containing detections in **global frame**:

```json
[
  {
    "frame_index": 0,
    "timestamp": 1532402927647951,
    "detections": [
      {
        "instance_token": "6dd2cbf4c24b4cae...",
        "category_name": "car",
        "translation": [353.794, 1132.355, 0.602],
        "size": [2.011, 4.633, 1.573],
        "yaw": -0.4034
      }
    ]
  }
]
```

- `translation`: [x, y, z] in global/map frame (meters)
- `size`: [width, length, height] in meters
- `yaw`: rotation about z-axis (radians)
- `instance_token`: unique object identity across frames (drives consistent coloring)
- `category_name`: universal category (e.g. `car`, `pedestrian`, `truck`)

### Tracker Output JSON

```json
[
  {
    "frame_index": 0,
    "timestamp": 1532402927647951,
    "tracks": [
      {
        "id": 0,
        "category_name": "car",
        "translation": [353.8, 1132.4, 0.6],
        "size": [2.011, 4.633, 1.573],
        "yaw": -0.4034,
        "tracking_score": 0.95,
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

## Installation

### Local Install

```bash
cd Project_SensorLens
pip install -r requirements.txt
```

For running converters, install the converter dependencies:

```bash
pip install -r converters/requirements.txt
```

#### Core Dependencies

- `dash` / `plotly` — web UI and 3D rendering
- `numpy` — point cloud and geometry operations
- `opencv-python` — image handling
- `motmetrics` — CLEAR MOT evaluation for debug mode
- `Pillow` — image handling

#### Converter Dependencies (install as needed)

- `nuscenes-devkit` / `pyquaternion` — nuScenes converter

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

#### Mounting Data

Mount your datasets and converted scenes into the container:

```yaml
volumes:
  - /path/to/datasets:/data/datasets:ro
  - /path/to/sensorlens_scenes:/data/sensorlens_scenes
```

> **Note:** The default `docker-compose.yml` targets `linux/arm64` (Apple Silicon). Remove or change the `platform` line for other architectures.

## Usage

```bash
python3 run.py
```

Then open http://localhost:8050 in your browser. The configuration page lets you:

1. Select mode — **Visualization** or **Debug**
2. Enter the path to a converted scene directory (for LiDAR + cameras + embedded GT)
3. Optionally upload or enter paths for GT and/or tracker JSON files
4. (Debug mode) Set match distance threshold and select evaluation categories
5. Click **Launch** to start

### CLI Arguments

| Argument | Default  | Description       |
|----------|----------|-------------------|
| `--port` | `8050`   | Server port       |
| `--host` | `0.0.0.0`| Host to bind to  |

## Architecture

```
Project_SensorLens/
  sensorlens/
    app.py             -- Dash app: config page, viz/debug layouts, all callbacks
    data_loader.py     -- Universal scene loader, category mapping
    scene_builder.py   -- 3D figure construction (point cloud, boxes, ego car model)
    mot_evaluator.py   -- CLEAR MOT evaluation, per-frame event extraction, metrics
    assets/
      style.css        -- UI styling (config page, checkboxes, buttons)
      NormalCar2.obj   -- 3D ego vehicle model
      NormalCar2.mtl   -- Material definitions
  converters/
    common.py          -- Shared utilities: category maps, file writers, coordinate helpers
    convert_nuscenes.py -- nuScenes → universal format
    convert_kitti.py   -- KITTI tracking → universal format
    convert_waymo.py   -- Waymo Open Dataset v2 → universal format
    convert_argoverse2.py -- Argoverse 2 → universal format (coming soon)
    requirements.txt   -- Dataset SDK dependencies for converters
  run.py               -- CLI entry point
  Dockerfile           -- Container image definition
  docker-compose.yml   -- Docker Compose configuration
```
