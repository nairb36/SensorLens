import numpy as np
import cv2
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update

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


def create_app(
    dataroot: str,
    version: str,
    gt_path: str | None = None,
    tracker_path: str | None = None,
) -> Dash:
    nusc_loader = NuScenesLoader(dataroot, version)

    gt_data = load_gt_json(gt_path) if gt_path else None
    tracker_data = load_tracker_json(tracker_path) if tracker_path else None

    if gt_data:
        num_frames = len(gt_data)
        sample_tokens = [f["sample_token"] for f in gt_data]
    elif tracker_data:
        num_frames = len(tracker_data)
        sample_tokens = None
    else:
        raise ValueError("At least one of --gt or --tracker must be provided")

    FRONT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    REAR_CAMS = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

    if sample_tokens is None and gt_data is None:
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

    cals = nusc_loader.get_camera_calibrations(sample_tokens[0])
    front_stitcher = PanoramaStitcher(
        [cals[c] for c in FRONT_CAMS], center_yaw=0.0,
    )
    rear_stitcher = PanoramaStitcher(
        [cals[c] for c in REAR_CAMS], center_yaw=np.pi, mirror=True,
    )

    app = Dash(__name__)

    layer_options = [
        {
            "label": html.Span("Ground Truth", style={"color": "#999" if gt_data else "#555"}),
            "value": "gt",
            "disabled": not bool(gt_data),
        },
        {
            "label": html.Span("Tracker", style={"color": "#999" if tracker_data else "#555"}),
            "value": "tracker",
            "disabled": not bool(tracker_data),
        },
    ]
    layer_defaults = []
    if gt_data:
        layer_defaults.append("gt")
    if tracker_data:
        layer_defaults.append("tracker")

    cat_options = [
        {"label": html.Span(g, style={"color": "#999"}), "value": g}
        for g in CATEGORY_GROUPS
    ]
    cat_defaults = [g for g in CATEGORY_GROUPS if g in DEFAULT_ON]

    app.layout = html.Div(
        style={
            "backgroundColor": "#0f0f23",
            "color": "#e0e0e0",
            "fontFamily": "'Segoe UI', 'Roboto', sans-serif",
            "minHeight": "100vh",
            "padding": "12px",
        },
        children=[
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
                                        "Layers",
                                        style={
                                            "fontSize": "10px",
                                            "color": "#666",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="check-layers",
                                        options=layer_options,
                                        value=layer_defaults,
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
        Input("check-layers", "value"),
        Input("check-categories", "value"),
    )
    def render_frame(frame_idx, active_layers, active_categories):
        if frame_idx is None or frame_idx < 0 or frame_idx >= len(sample_tokens):
            return no_update, no_update, no_update, no_update

        gt_on = "gt" in (active_layers or [])
        tracker_on = "tracker" in (active_layers or [])
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

        if gt_on and gt_data and frame_idx < len(gt_data):
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

        if tracker_on and tracker_data and frame_idx < len(tracker_data):
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
                })
            total_objects += len(tracker_boxes_ego)

        fig = build_3d_figure(points, gt_boxes_ego, tracker_boxes_ego)

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
