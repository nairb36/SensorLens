import base64
import json
import logging
import re
from pathlib import Path

import numpy as np
import cv2
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update

logger = logging.getLogger(__name__)

from .data_loader import (
    NuScenesLoader,
    load_gt_json,
    load_tracker_json,
    global_to_ego,
    shorten_category,
    CATEGORY_GROUPS,
    CATEGORY_TO_GROUP,
    DEFAULT_ON,
)
from .scene_builder import build_3d_figure
from .image_stitcher import PanoramaStitcher, encode_panorama

FRONT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
REAR_CAMS = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

_server_state = {
    "nusc_loader": None,
    "gt_data": None,
    "tracker_data": None,
    "sample_tokens": None,
    "num_frames": 0,
    "front_stitcher": None,
    "rear_stitcher": None,
    "scene_mismatch": False,
}


def _extract_scene_prefix(name: str) -> str | None:
    stem = Path(name).stem
    m = re.match(r"(scene_\d+)", stem)
    return m.group(1) if m else None


def _config_layout():
    return html.Div(
        style={
            "backgroundColor": "#0f0f23",
            "minHeight": "100vh",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontFamily": "'Segoe UI', 'Roboto', sans-serif",
        },
        children=[
            html.Div(
                className="config-card",
                style={
                    "backgroundColor": "#16213e",
                    "borderRadius": "12px",
                    "padding": "36px 40px",
                    "width": "560px",
                    "maxWidth": "90vw",
                    "boxShadow": "0 8px 32px rgba(0,0,0,0.4)",
                    "border": "1px solid rgba(255,255,255,0.06)",
                },
                children=[
                    html.H1(
                        "SensorLens",
                        style={
                            "color": "#00d4ff",
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "margin": "0 0 4px 0",
                            "textAlign": "center",
                        },
                    ),
                    html.P(
                        "3D Multi-Object Tracking Visualizer",
                        style={
                            "color": "#666",
                            "fontSize": "13px",
                            "textAlign": "center",
                            "margin": "0 0 28px 0",
                        },
                    ),
                    # Dataset type
                    _config_label("Dataset Type"),
                    dcc.Dropdown(
                        id="config-dataset-type",
                        options=[{"label": "NuScenes", "value": "nuscenes"}],
                        value="nuscenes",
                        clearable=False,
                        style={"marginBottom": "16px"},
                    ),
                    # Dataroot
                    _config_label("Dataroot Path"),
                    dcc.Input(
                        id="config-dataroot",
                        type="text",
                        placeholder="/path/to/nuscenes",
                        className="config-input",
                        style=_input_style(),
                    ),
                    # Version
                    _config_label("Version"),
                    dcc.Dropdown(
                        id="config-version",
                        options=[
                            {"label": "v1.0-mini", "value": "v1.0-mini"},
                            {"label": "v1.0-trainval", "value": "v1.0-trainval"},
                            {"label": "v1.0-test", "value": "v1.0-test"},
                        ],
                        value="v1.0-mini",
                        clearable=False,
                        style={"marginBottom": "16px"},
                    ),
                    # GT file
                    _config_label("GT Detections (JSON)"),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "alignItems": "stretch", "marginBottom": "4px"},
                        children=[
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    dcc.Input(
                                        id="config-gt-path",
                                        type="text",
                                        placeholder="Path to GT JSON (optional)",
                                        className="config-input",
                                        style=_input_style(mb="0"),
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "color": "#555", "fontSize": "12px"},
                                children=["or"],
                            ),
                            dcc.Upload(
                                id="config-gt-upload",
                                children=html.Div("Upload", className="upload-btn"),
                                className="upload-dropzone",
                            ),
                        ],
                    ),
                    html.Div(id="config-gt-filename", style={"fontSize": "11px", "color": "#1abc9c", "marginBottom": "16px", "minHeight": "16px"}),
                    # Tracker file
                    _config_label("Tracker Results (JSON)"),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "alignItems": "stretch", "marginBottom": "4px"},
                        children=[
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    dcc.Input(
                                        id="config-trk-path",
                                        type="text",
                                        placeholder="Path to tracker JSON (optional)",
                                        className="config-input",
                                        style=_input_style(mb="0"),
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "color": "#555", "fontSize": "12px"},
                                children=["or"],
                            ),
                            dcc.Upload(
                                id="config-trk-upload",
                                children=html.Div("Upload", className="upload-btn"),
                                className="upload-dropzone",
                            ),
                        ],
                    ),
                    html.Div(id="config-trk-filename", style={"fontSize": "11px", "color": "#1abc9c", "marginBottom": "24px", "minHeight": "16px"}),
                    # Error message
                    html.Div(id="config-error-msg", style={
                        "color": "#ff6b6b",
                        "fontSize": "13px",
                        "marginBottom": "12px",
                        "minHeight": "20px",
                        "textAlign": "center",
                    }),
                    # Launch button
                    dcc.Loading(
                        type="dot",
                        color="#00d4ff",
                        children=[
                            html.Button(
                                "Launch",
                                id="btn-launch",
                                className="launch-btn",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _config_label(text):
    return html.Label(
        text,
        style={
            "color": "#aaa",
            "fontSize": "12px",
            "fontWeight": "600",
            "display": "block",
            "marginBottom": "4px",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        },
    )


def _input_style(mb="16px"):
    return {
        "width": "100%",
        "backgroundColor": "#0f0f23",
        "border": "1px solid rgba(255,255,255,0.1)",
        "borderRadius": "6px",
        "color": "#e0e0e0",
        "padding": "8px 12px",
        "fontSize": "13px",
        "marginBottom": mb,
        "boxSizing": "border-box",
    }


def _viz_layout():
    s = _server_state
    gt_data = s["gt_data"]
    tracker_data = s["tracker_data"]
    num_frames = s["num_frames"]
    scene_mismatch = s["scene_mismatch"]

    gt_viz_options = [
        {"label": html.Span("bbox", style={"color": "#999"}), "value": "bbox"},
        {"label": html.Span("center", style={"color": "#999"}), "value": "center"},
    ]
    trk_viz_options = [
        {"label": html.Span("bbox", style={"color": "#999"}), "value": "bbox"},
        {"label": html.Span("center", style={"color": "#999"}), "value": "center"},
    ]
    gt_viz_defaults = ["bbox"] if gt_data else []
    trk_viz_defaults = ["bbox"] if tracker_data else []

    cat_options = [
        {"label": html.Span(g, style={"color": "#999"}), "value": g}
        for g in CATEGORY_GROUPS
    ]
    cat_defaults = [g for g in CATEGORY_GROUPS if g in DEFAULT_ON]

    return html.Div(
        style={
            "backgroundColor": "#0f0f23",
            "color": "#e0e0e0",
            "fontFamily": "'Segoe UI', 'Roboto', sans-serif",
            "minHeight": "100vh",
            "padding": "12px",
        },
        children=[
            html.Div(
                "WARNING: GT and tracker files are from different scenes — results may not be comparable.",
                style={
                    "backgroundColor": "#8b0000",
                    "color": "#fff",
                    "padding": "8px 16px",
                    "borderRadius": "6px",
                    "marginBottom": "8px",
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "textAlign": "center",
                    "display": "block" if scene_mismatch else "none",
                },
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "8px",
                    "padding": "8px 16px",
                    "backgroundColor": "#16213e",
                    "borderRadius": "8px",
                },
                children=[
                    html.H1(
                        "SensorLens",
                        style={
                            "margin": "0",
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "color": "#00d4ff",
                        },
                    ),
                    html.Div(id="frame-info", style={"fontSize": "14px", "color": "#aaa"}),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "marginBottom": "8px",
                    "padding": "8px 16px",
                    "backgroundColor": "#16213e",
                    "borderRadius": "8px",
                },
                children=[
                    html.Button("Prev", id="btn-prev", style=_button_style()),
                    html.Button("Play", id="btn-play", style=_button_style("#1abc9c")),
                    html.Button("Next", id="btn-next", style=_button_style()),
                    html.Div(
                        style={"flex": "1", "margin": "0 12px"},
                        children=[
                            dcc.Slider(
                                id="frame-slider",
                                min=0,
                                max=num_frames - 1,
                                value=0,
                                step=1,
                                marks={
                                    0: "0",
                                    num_frames - 1: str(num_frames - 1),
                                },
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "height": "calc(100vh - 130px)",
                },
                children=[
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "borderRadius": "8px",
                            "overflow": "hidden",
                            "position": "relative",
                        },
                        children=[
                            dcc.Graph(
                                id="scene-3d",
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                },
                                style={"height": "100%"},
                            ),
                            html.Div(
                                style={
                                    "position": "absolute",
                                    "top": "10px",
                                    "left": "10px",
                                    "backgroundColor": "rgba(15, 15, 35, 0.85)",
                                    "borderRadius": "6px",
                                    "padding": "10px 14px",
                                    "zIndex": "10",
                                    "backdropFilter": "blur(4px)",
                                    "border": "1px solid rgba(255,255,255,0.06)",
                                },
                                children=[
                                    html.Div(
                                        "Detections",
                                        style={
                                            "fontSize": "10px",
                                            "color": "#666" if gt_data else "#444",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                            "marginBottom": "4px",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="check-gt-viz",
                                        options=gt_viz_options,
                                        value=gt_viz_defaults,
                                    ),
                                    html.Div(style={"height": "6px"}),
                                    html.Div(
                                        "Tracks",
                                        style={
                                            "fontSize": "10px",
                                            "color": "#666" if tracker_data else "#444",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                            "marginBottom": "4px",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="check-trk-viz",
                                        options=trk_viz_options,
                                        value=trk_viz_defaults,
                                    ),
                                    html.Hr(style={
                                        "border": "none",
                                        "borderTop": "1px solid rgba(255,255,255,0.08)",
                                        "margin": "8px 0",
                                    }),
                                    html.Div(
                                        "Categories",
                                        style={
                                            "fontSize": "10px",
                                            "color": "#666",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="check-categories",
                                        options=cat_options,
                                        value=cat_defaults,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                        children=[
                            html.Div(
                                style={
                                    "backgroundColor": "#16213e",
                                    "borderRadius": "4px",
                                    "padding": "4px",
                                },
                                children=[
                                    html.Div(
                                        "FRONT",
                                        style={
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                            "color": "#00d4ff",
                                            "marginBottom": "2px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.Img(
                                        id="pano-front",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "2px",
                                            "display": "block",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "backgroundColor": "#16213e",
                                    "borderRadius": "4px",
                                    "padding": "4px",
                                },
                                children=[
                                    html.Div(
                                        "REAR",
                                        style={
                                            "fontSize": "11px",
                                            "fontWeight": "600",
                                            "color": "#00d4ff",
                                            "marginBottom": "2px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    html.Img(
                                        id="pano-rear",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "2px",
                                            "display": "block",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="store-frame", data=0),
            dcc.Store(id="store-playing", data=False),
            dcc.Interval(id="play-interval", interval=500, disabled=True),
        ],
    )


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        dcc.Store(id="app-phase", data="config"),
        html.Div(id="page-content"),
    ])

    # -- Page switching --
    @app.callback(
        Output("page-content", "children"),
        Input("app-phase", "data"),
    )
    def switch_page(phase):
        if phase == "viz":
            return _viz_layout()
        return _config_layout()

    # -- Show uploaded filenames --
    @app.callback(
        Output("config-gt-filename", "children"),
        Input("config-gt-upload", "filename"),
        prevent_initial_call=True,
    )
    def show_gt_filename(filename):
        return f"Uploaded: {filename}" if filename else ""

    @app.callback(
        Output("config-trk-filename", "children"),
        Input("config-trk-upload", "filename"),
        prevent_initial_call=True,
    )
    def show_trk_filename(filename):
        return f"Uploaded: {filename}" if filename else ""

    # -- Launch --
    @app.callback(
        Output("app-phase", "data"),
        Output("config-error-msg", "children"),
        Input("btn-launch", "n_clicks"),
        State("config-dataroot", "value"),
        State("config-version", "value"),
        State("config-gt-upload", "contents"),
        State("config-gt-upload", "filename"),
        State("config-gt-path", "value"),
        State("config-trk-upload", "contents"),
        State("config-trk-upload", "filename"),
        State("config-trk-path", "value"),
        prevent_initial_call=True,
    )
    def launch(n_clicks, dataroot, version, gt_upload, gt_upload_name,
               gt_path, trk_upload, trk_upload_name, trk_path):
        if not dataroot or not dataroot.strip():
            return no_update, "Please enter a dataroot path."
        dataroot = dataroot.strip()
        if not Path(dataroot).is_dir():
            return no_update, f"Dataroot not found: {dataroot}"

        has_gt = bool(gt_upload) or bool(gt_path and gt_path.strip())
        has_trk = bool(trk_upload) or bool(trk_path and trk_path.strip())
        if not has_gt and not has_trk:
            return no_update, "Provide at least one of GT or Tracker file."

        # Load GT
        gt_data = None
        gt_name = None
        try:
            if gt_upload:
                gt_data = _parse_upload(gt_upload)
                gt_name = gt_upload_name
            elif gt_path and gt_path.strip():
                p = gt_path.strip()
                if not Path(p).is_file():
                    return no_update, f"GT file not found: {p}"
                gt_data = load_gt_json(p)
                gt_name = Path(p).name
        except Exception as e:
            return no_update, f"Error loading GT: {e}"

        # Load Tracker
        tracker_data = None
        trk_name = None
        try:
            if trk_upload:
                tracker_data = _parse_upload(trk_upload)
                trk_name = trk_upload_name
            elif trk_path and trk_path.strip():
                p = trk_path.strip()
                if not Path(p).is_file():
                    return no_update, f"Tracker file not found: {p}"
                tracker_data = load_tracker_json(p)
                trk_name = Path(p).name
        except Exception as e:
            return no_update, f"Error loading tracker: {e}"

        # Scene mismatch check
        scene_mismatch = False
        if gt_name and trk_name:
            gt_prefix = _extract_scene_prefix(gt_name)
            trk_prefix = _extract_scene_prefix(trk_name)
            if gt_prefix and trk_prefix and gt_prefix != trk_prefix:
                scene_mismatch = True
                logger.warning(
                    "Scene mismatch: GT=%s (%s) vs Tracker=%s (%s)",
                    gt_name, gt_prefix, trk_name, trk_prefix,
                )

        # Initialize NuScenes
        try:
            nusc_loader = NuScenesLoader(dataroot, version)
        except Exception as e:
            return no_update, f"Error loading NuScenes: {e}"

        # Determine frames
        if gt_data:
            num_frames = len(gt_data)
            sample_tokens = [f["sample_token"] for f in gt_data]
        elif tracker_data:
            num_frames = len(tracker_data)
            sample_tokens = None
        else:
            return no_update, "No data loaded."

        if sample_tokens is None:
            scene = nusc_loader.nusc.scene[0]
            sample_token = scene["first_sample_token"]
            sample_tokens = []
            for _ in range(num_frames):
                sample_tokens.append(sample_token)
                sample = nusc_loader.get_sample(sample_token)
                if sample["next"]:
                    sample_token = sample["next"]
                else:
                    break

        # Build stitchers
        cals = nusc_loader.get_camera_calibrations(sample_tokens[0])
        front_stitcher = PanoramaStitcher(
            [cals[c] for c in FRONT_CAMS], center_yaw=0.0,
        )
        rear_stitcher = PanoramaStitcher(
            [cals[c] for c in REAR_CAMS], center_yaw=np.pi, mirror=True,
        )

        # Populate server state
        _server_state.update({
            "nusc_loader": nusc_loader,
            "gt_data": gt_data,
            "tracker_data": tracker_data,
            "sample_tokens": sample_tokens,
            "num_frames": num_frames,
            "front_stitcher": front_stitcher,
            "rear_stitcher": rear_stitcher,
            "scene_mismatch": scene_mismatch,
        })

        return "viz", ""

    # -- Viz callbacks --
    @app.callback(
        Output("store-playing", "data"),
        Output("play-interval", "disabled"),
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
        State("store-playing", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, playing):
        new_playing = not playing
        return new_playing, not new_playing, "Pause" if new_playing else "Play"

    @app.callback(
        Output("store-frame", "data"),
        Output("frame-slider", "value"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("frame-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("store-frame", "data"),
        State("store-playing", "data"),
        prevent_initial_call=True,
    )
    def update_frame(prev_clicks, next_clicks, slider_val, n_intervals, current_frame, playing):
        if _server_state["nusc_loader"] is None:
            return no_update, no_update

        num_frames = _server_state["num_frames"]
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "btn-prev":
            new_frame = max(0, current_frame - 1)
        elif trigger_id == "btn-next":
            new_frame = min(num_frames - 1, current_frame + 1)
        elif trigger_id == "play-interval":
            if not playing:
                return no_update, no_update
            new_frame = current_frame + 1
            if new_frame >= num_frames:
                new_frame = 0
        elif trigger_id == "frame-slider":
            new_frame = slider_val
        else:
            return no_update, no_update

        return new_frame, new_frame

    @app.callback(
        Output("scene-3d", "figure"),
        Output("pano-front", "src"),
        Output("pano-rear", "src"),
        Output("frame-info", "children"),
        Input("store-frame", "data"),
        Input("check-gt-viz", "value"),
        Input("check-trk-viz", "value"),
        Input("check-categories", "value"),
    )
    def render_frame(frame_idx, gt_viz, trk_viz, active_categories):
        s = _server_state
        if s["nusc_loader"] is None:
            return no_update, no_update, no_update, no_update

        nusc_loader = s["nusc_loader"]
        gt_data = s["gt_data"]
        tracker_data = s["tracker_data"]
        sample_tokens = s["sample_tokens"]
        num_frames = s["num_frames"]
        front_stitcher = s["front_stitcher"]
        rear_stitcher = s["rear_stitcher"]

        if frame_idx is None or frame_idx < 0 or frame_idx >= len(sample_tokens):
            return no_update, no_update, no_update, no_update

        gt_viz = set(gt_viz or [])
        trk_viz = set(trk_viz or [])
        active_groups = set(active_categories or [])

        sample_token = sample_tokens[frame_idx]

        ego_pose = nusc_loader.get_ego_pose(sample_token)
        points = nusc_loader.get_lidar_points_ego(sample_token)

        def category_visible(cat_name):
            group = CATEGORY_TO_GROUP.get(cat_name)
            return group is not None and group in active_groups

        gt_boxes_ego = None
        tracker_boxes_ego = None
        total_objects = 0

        if gt_viz and gt_data and frame_idx < len(gt_data):
            frame = gt_data[frame_idx]
            gt_boxes_ego = []
            for det in frame.get("detections", []):
                if not category_visible(det["category_name"]):
                    continue
                pos, yaw = global_to_ego(det["translation"], det["yaw"], ego_pose)
                gt_boxes_ego.append({
                    "translation": pos.tolist(),
                    "size": det["size"],
                    "yaw": yaw,
                    "label": shorten_category(det["category_name"]),
                    "instance_token": det.get("instance_token", ""),
                })
            total_objects += len(gt_boxes_ego)

        if trk_viz and tracker_data and frame_idx < len(tracker_data):
            frame = tracker_data[frame_idx]
            tracker_boxes_ego = []
            for trk in frame.get("tracks", []):
                if not category_visible(trk["category_name"]):
                    continue
                pos, yaw = global_to_ego(trk["translation"], trk["yaw"], ego_pose)
                tracker_boxes_ego.append({
                    "translation": pos.tolist(),
                    "size": trk["size"],
                    "yaw": yaw,
                    "label": shorten_category(trk["category_name"]),
                    "id": trk.get("id", 0),
                    "age": trk.get("age", ""),
                    "hits": trk.get("hits", ""),
                    "misses": trk.get("consecutive_misses", ""),
                })
            total_objects += len(tracker_boxes_ego)

        fig = build_3d_figure(points, gt_boxes_ego, tracker_boxes_ego,
                              gt_viz=gt_viz, trk_viz=trk_viz)

        cam_paths = nusc_loader.get_camera_paths(sample_token)
        front_imgs = [cv2.imread(cam_paths[c]) for c in FRONT_CAMS]
        rear_imgs = [cv2.imread(cam_paths[c]) for c in REAR_CAMS]

        front_src = encode_panorama(front_stitcher.stitch(front_imgs))
        rear_src = encode_panorama(rear_stitcher.stitch(rear_imgs))

        timestamp = ""
        if gt_data and frame_idx < len(gt_data):
            timestamp = str(gt_data[frame_idx].get("timestamp", ""))
        elif tracker_data and frame_idx < len(tracker_data):
            timestamp = str(tracker_data[frame_idx].get("timestamp", ""))

        info_text = f"Frame {frame_idx}/{num_frames - 1}  |  Objects: {total_objects}  |  Token: {sample_token[:8]}...  |  TS: {timestamp}"

        return fig, front_src, rear_src, info_text

    return app


def _parse_upload(contents: str) -> list[dict]:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return json.loads(decoded)


def _button_style(bg="#2c3e50"):
    return {
        "backgroundColor": bg,
        "color": "white",
        "border": "none",
        "padding": "8px 16px",
        "borderRadius": "4px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "600",
    }
