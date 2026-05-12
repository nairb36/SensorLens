Build an interactive 3D + camera visualization tool for a 3D multi-object tracking pipeline on nuScenes data. This should be its own standalone Python project.

Layout
Two-panel layout:

Top panel (main): interactive 3D scene showing LiDAR point cloud + 3D wireframe bounding boxes. Defaults to a top-down (BEV) camera angle, but supports orbit, zoom, and pan
Bottom panel: 6 camera images in a single row: FRONT_LEFT | FRONT | FRONT_RIGHT | BACK_LEFT | BACK | BACK_RIGHT
What it does
Renders a nuScenes scene frame by frame. The 3D panel shows the full LiDAR point cloud with 3D wireframe cuboid bounding boxes overlaid. The user can rotate, zoom, and pan the 3D view freely (default view is top-down BEV). The user can toggle between two overlay modes:

Ground-truth detections — loaded from a GT JSON file
Tracker output — loaded from a tracker results JSON file
Both can optionally be shown simultaneously (with different colors) so the user can visually compare tracker output against ground truth.

The bottom panel shows synchronized camera images from all 6 cameras for the current frame.

Data formats
GT detections JSON (input)
A JSON array of frames. Each frame has detections in global coordinates:


[
  {
    "frame_id": 0,
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "timestamp": 1532402927647951,
    "detections": [
      {
        "instance_token": "6dd2cbf4c24b4caeb625035869bca7b5",
        "category_name": "vehicle.car",
        "translation": [353.794, 1132.355, 0.602],
        "size": [2.011, 4.633, 1.573],
        "rotation": [0.9797, 0.0, 0.0, -0.2003],
        "yaw": -0.4034
      }
    ]
  }
]
translation: [x, y, z] in global frame (meters)
size: [width, length, height] in meters
rotation: [w, x, y, z] quaternion
yaw: rotation about z-axis (radians), already extracted from the quaternion
instance_token: unique object identity across frames (use for consistent coloring in GT)
category_name: nuScenes category string (e.g. vehicle.car, human.pedestrian.adult)
Tracker output JSON (input)
A JSON array of frames. Each frame has tracked objects:


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
        "age": 3,
        "hits": 3,
        "consecutive_misses": 0
      }
    ]
  }
]
id: integer track ID (use for consistent coloring across frames)
age, hits, consecutive_misses: track lifecycle metadata — display as tooltip or label
3D panel requirements
Interactive 3D scene: orbit, zoom, pan. Default camera position is top-down (BEV) looking at the ego vehicle
LiDAR point cloud: load the LIDAR_TOP sweep for each sample. Color points by height (z-value) or intensity. Use sample_token from the GT JSON to look up the sample's LiDAR data via the nuScenes devkit
3D wireframe bounding boxes: full 3D cuboids (8 corners, 12 edges) using translation (center), size (width x length x height), and yaw for rotation about z-axis
Color by identity: GT objects colored by instance_token, tracker objects colored by id. Use a distinct color palette so the same object keeps the same color across frames
Labels: show category_name (shortened — e.g. "car" not "vehicle.car") and the identity (instance_token suffix or track id) near each box
Ego vehicle: mark the ego vehicle position at the origin. Use sample_token from the GT JSON to look up ego pose via the nuScenes devkit. Transform LiDAR points and bounding boxes into ego-centered coordinates
Frame info: display current frame ID, timestamp, and count of objects on screen
Camera panel requirements
6 camera images in a single row: FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT
Load from nuScenes: use sample_token to look up camera sample_data for each of the 6 cameras via the nuScenes devkit
Label each image with the camera name
Synchronized: all 6 images correspond to the same sample/frame as the 3D panel
Interaction
Step forward/backward through frames
Play/pause animation at ~2 FPS (nuScenes keyframe rate)
Toggle GT overlay on/off
Toggle tracker overlay on/off
Slider to jump to any frame
Inputs
The tool should accept paths via CLI arguments or a UI sidebar:

--dataroot : path to nuScenes dataset (default: /workspace/data/nuscenes)
--version : nuScenes version (default: v1.0-mini)
--gt : path to GT detections JSON file
--tracker : path to tracker results JSON file
Either or both of --gt/--tracker can be provided
nuScenes data location
The nuScenes mini split is at /workspace/data/nuscenes (version v1.0-mini). Camera images are under samples/CAM_*, LiDAR sweeps under samples/LIDAR_TOP.

Tech stack
Use whatever will produce the most polished interactive 3D result. Consider Plotly Dash with 3D scatter, Streamlit with a Three.js component, PyVista, or Open3D. Prioritize smooth 3D interaction with large point clouds and visual quality.

Project structure
This is a standalone repo, not part of the MOT pipeline. Keep it self-contained with its own requirements.txt and README with usage instructions.