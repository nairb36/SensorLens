import base64
import json
import logging
import re
from pathlib import Path

import numpy as np
import cv2
import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update

logger = logging.getLogger(__name__)

from .data_loader import (
    UniversalLoader,
    load_gt_json,
    load_tracker_json,
    shorten_category,
    CATEGORY_GROUPS,
    CATEGORY_TO_GROUP,
    DEFAULT_ON,
)
from .scene_builder import build_3d_figure
from .mot_evaluator import (
    run_evaluation,
    compute_summary,
    get_frame_events,
    get_box_error_types,
    ERROR_COLORS,
    SWITCH_LINE_WIDTH,
    METRIC_DISPLAY,
    format_metric,
)

MAX_CAMERA_SLOTS = 9

_server_state = {
    "scene_loader": None,
    "camera_names": [],
    "gt_data": None,
    "tracker_data": None,
    "num_frames": 0,
    "scene_mismatch": False,
    "app_mode": "visualization",
    "mot_accumulator": None,
    "mot_id_map": None,
    "mot_summary": None,
    "demo_mode": False,
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
                            "margin": "0 0 20px 0",
                        },
                    ),
                    # App mode selector
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "marginBottom": "24px",
                            "gap": "0",
                        },
                        children=[
                            html.Button(
                                "Visualization",
                                id="btn-mode-viz",
                                style={
                                    "backgroundColor": "#00d4ff",
                                    "color": "#0f0f23",
                                    "border": "1px solid #00d4ff",
                                    "padding": "8px 24px",
                                    "borderRadius": "6px 0 0 6px",
                                    "cursor": "pointer",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                },
                            ),
                            html.Button(
                                "Debug",
                                id="btn-mode-debug",
                                style={
                                    "backgroundColor": "transparent",
                                    "color": "#aaa",
                                    "border": "1px solid rgba(255,255,255,0.15)",
                                    "padding": "8px 24px",
                                    "borderRadius": "0 6px 6px 0",
                                    "cursor": "pointer",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                },
                            ),
                        ],
                    ),
                    dcc.Store(id="config-app-mode", data="visualization"),
                    # Debug mode options (hidden by default)
                    html.Div(
                        id="config-debug-fields",
                        style={"display": "none"},
                        children=[
                            _config_label("Match Distance Threshold (m)"),
                            dcc.Input(
                                id="config-max-dist",
                                type="number",
                                value=2.0,
                                placeholder="Default: 2.0",
                                className="config-input",
                                style=_input_style(),
                            ),
                            _config_label("Evaluate Categories"),
                            dcc.Checklist(
                                id="config-eval-categories",
                                options=[
                                    {"label": html.Span(
                                        f" {g}", style={"color": "#ccc", "fontSize": "13px"}
                                    ), "value": g}
                                    for g in CATEGORY_GROUPS
                                ],
                                value=[g for g in CATEGORY_GROUPS if g in DEFAULT_ON],
                                style={"marginBottom": "16px"},
                            ),
                        ],
                    ),
                    # Scene directory
                    _config_label("Scene Directory (optional)"),
                    dcc.Input(
                        id="config-scene-path",
                        type="text",
                        placeholder="Path to converted scene (for point cloud + cameras)",
                        className="config-input",
                        style=_input_style(),
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


def encode_image(path: str) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    max_w = 800
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _viz_layout():
    s = _server_state
    gt_data = s["gt_data"]
    tracker_data = s["tracker_data"]
    num_frames = s["num_frames"]
    scene_mismatch = s["scene_mismatch"]
    camera_names = s["camera_names"]
    has_scene = s["scene_loader"] is not None
    app_mode = s["app_mode"]
    is_debug = app_mode == "debug"
    demo_mode = s.get("demo_mode", False)

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

    # 3D panel
    scene_panel = html.Div(
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
    )

    # Right panel children: cameras + debug
    right_children = []

    # Camera grid
    if camera_names:
        cam_cells = []
        for i, cam_name in enumerate(camera_names):
            if i >= MAX_CAMERA_SLOTS:
                break
            cam_cells.append(
                html.Div(
                    className="camera-cell",
                    style={
                        "backgroundColor": "#16213e",
                        "borderRadius": "4px",
                        "padding": "4px",
                        "flex": "1 1 30%",
                        "minWidth": "0",
                    },
                    children=[
                        html.Div(
                            cam_name.upper().replace("_", " "),
                            style={
                                "fontSize": "10px",
                                "fontWeight": "600",
                                "color": "#00d4ff",
                                "marginBottom": "2px",
                                "textAlign": "center",
                                "letterSpacing": "0.5px",
                            },
                        ),
                        html.Img(
                            id=f"cam-{i}",
                            style={
                                "width": "100%",
                                "borderRadius": "2px",
                                "display": "block",
                            },
                        ),
                    ],
                )
            )
        right_children.append(
            html.Div(
                className="camera-grid",
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "4px",
                },
                children=cam_cells,
            )
        )

    # Debug panel
    if is_debug:
        debug_cat_options = [
            {"label": html.Span(g, style={"color": "#999"}), "value": g}
            for g in CATEGORY_GROUPS
        ]
        debug_cat_defaults = [g for g in CATEGORY_GROUPS if g in DEFAULT_ON]

        legend_items = [
            ("Match", ERROR_COLORS["match"]),
            ("ID Switch", ERROR_COLORS["switch"]),
            ("FP", ERROR_COLORS["fp"]),
            ("Missed", ERROR_COLORS["miss"]),
        ]

        right_children.append(
            html.Div(
                style={
                    "backgroundColor": "#16213e",
                    "borderRadius": "4px",
                    "padding": "8px 10px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                        children=[
                            html.Div(
                                "DEBUG LOG",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "600",
                                    "color": "#ff4444",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                },
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px"},
                                children=[
                                    html.Span(
                                        [html.Span("■ ", style={"color": c}), name],
                                        style={"fontSize": "10px", "color": "#888"},
                                    )
                                    for name, c in legend_items
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="debug-content",
                        style={
                            "maxHeight": "180px",
                            "overflowY": "auto",
                            "fontSize": "12px",
                            "color": "#ccc",
                        },
                    ),
                    html.Hr(style={
                        "border": "none",
                        "borderTop": "1px solid rgba(255,255,255,0.08)",
                        "margin": "8px 0",
                    }),
                    html.Div(
                        "Debug Categories",
                        style={
                            "fontSize": "10px",
                            "color": "#666",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                            "marginBottom": "4px",
                        },
                    ),
                    dcc.Checklist(
                        id="check-debug-categories",
                        options=debug_cat_options,
                        value=debug_cat_defaults,
                    ),
                ],
            ),
        )

    # Metrics panel (collapsible, debug mode only)
    metrics_section = html.Div(style={"display": "none"})
    if is_debug and s["mot_summary"]:
        summary = s["mot_summary"]
        metrics_rows = []
        for key, display_name, fmt in METRIC_DISPLAY:
            val = summary.get(key)
            if val is not None:
                metrics_rows.append(
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "padding": "3px 0"},
                        children=[
                            html.Span(display_name, style={"color": "#aaa", "fontSize": "12px"}),
                            html.Span(
                                format_metric(val, fmt),
                                style={"color": "#00d4ff", "fontSize": "12px", "fontWeight": "600"},
                            ),
                        ],
                    )
                )
        metrics_section = html.Div(
            style={"marginBottom": "8px"},
            children=[
                html.Button(
                    id="btn-metrics",
                    style={
                        "width": "100%",
                        "backgroundColor": "#1a2744",
                        "color": "#00d4ff",
                        "border": "1px solid rgba(0,212,255,0.2)",
                        "borderRadius": "8px",
                        "padding": "8px 16px",
                        "cursor": "pointer",
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "textAlign": "left",
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                    },
                    children=[
                        html.Span("Tracking Metrics"),
                        html.Span("▼", id="metrics-arrow", style={"fontSize": "10px"}),
                    ],
                ),
                html.Div(
                    id="metrics-panel",
                    style={
                        "display": "none",
                        "backgroundColor": "#1a2744",
                        "borderRadius": "0 0 8px 8px",
                        "padding": "8px 16px",
                        "borderTop": "1px solid rgba(0,212,255,0.15)",
                    },
                    children=metrics_rows,
                ),
            ],
        )

    # Hidden placeholders for components that may not exist
    hidden_placeholders = []
    if not is_debug:
        hidden_placeholders.extend([
            html.Div(id="debug-content", style={"display": "none"}),
            html.Div(id="check-debug-categories", style={"display": "none"}),
            html.Div(id="btn-metrics", style={"display": "none"}),
            html.Div(id="metrics-panel", style={"display": "none"}),
            html.Div(id="metrics-arrow", style={"display": "none"}),
        ])
    # Hidden placeholders for unused camera slots
    num_active_cams = min(len(camera_names), MAX_CAMERA_SLOTS)
    for i in range(num_active_cams, MAX_CAMERA_SLOTS):
        hidden_placeholders.append(
            html.Img(id=f"cam-{i}", style={"display": "none"})
        )

    show_right_panel = bool(camera_names) or is_debug
    right_panel = html.Div(
        style={
            "flex": "1",
            "minWidth": "0",
            "display": "flex" if show_right_panel else "none",
            "flexDirection": "column",
            "gap": "4px",
            "overflowY": "auto",
        },
        children=right_children,
    )

    main_children = [scene_panel, right_panel]

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
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "12px"},
                        children=[
                            html.Button(
                                "⌂",
                                id="btn-home",
                                style={
                                    "backgroundColor": "transparent",
                                    "border": "1px solid rgba(255,255,255,0.15)",
                                    "borderRadius": "6px",
                                    "color": "#00d4ff",
                                    "fontSize": "20px",
                                    "cursor": "pointer",
                                    "padding": "2px 10px",
                                    "lineHeight": "1",
                                },
                                title="Back to config",
                            ),
                            html.H1(
                                [
                                    "SensorLens",
                                    html.Span(
                                        " (Demo)",
                                        style={"fontSize": "14px", "color": "#888", "fontWeight": "400"},
                                    ) if demo_mode else None,
                                ],
                                style={
                                    "margin": "0",
                                    "fontSize": "24px",
                                    "fontWeight": "700",
                                    "color": "#00d4ff",
                                },
                            ),
                            html.Span(
                                "DEBUG" if is_debug else "VIZ",
                                style={
                                    "backgroundColor": "#ff4444" if is_debug else "#1abc9c",
                                    "color": "#fff",
                                    "fontSize": "10px",
                                    "fontWeight": "700",
                                    "padding": "2px 8px",
                                    "borderRadius": "4px",
                                    "letterSpacing": "1px",
                                },
                            ),
                        ],
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
                    html.Button("⏮", id="btn-prev", style=button_style()),
                    html.Button(
                        "▶", id="btn-play",
                        style={**button_style("#1abc9c"), "display": "none" if is_debug else "inline-block"},
                    ),
                    html.Button("⏭", id="btn-next", style=button_style()),
                    html.Button("3D", id="btn-view-mode", style=button_style("#8e44ad")),
                    html.Button("⬤", id="btn-pc-color", title="Toggle point cloud color",
                                style={**button_style("#555"), "fontSize": "10px"}),
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
            metrics_section,
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "height": "calc(100vh - 130px)" if not is_debug else "calc(100vh - 170px)",
                },
                children=main_children,
            ),
            dcc.Store(id="store-frame", data=0),
            dcc.Store(id="store-playing", data=False),
            dcc.Store(id="store-view-mode", data="3d"),
            dcc.Store(id="store-pc-color", data="color"),
            dcc.Interval(id="play-interval", interval=500, disabled=True),
            *hidden_placeholders,
        ],
    )


def compute_frame_data(frame_idx, gt_viz, trk_viz, active_categories,
                       view_mode, pc_color_mode, debug_categories,
                       is_debug, state):
    s = state
    if s["num_frames"] == 0:
        empty_cams = [""] * MAX_CAMERA_SLOTS
        return no_update, *empty_cams, no_update, no_update

    scene_loader = s["scene_loader"]
    gt_data = s["gt_data"]
    tracker_data = s["tracker_data"]
    num_frames = s["num_frames"]
    camera_names = s["camera_names"]
    mot_acc = s["mot_accumulator"]
    mot_id_map = s["mot_id_map"]

    if frame_idx is None or frame_idx < 0 or frame_idx >= num_frames:
        empty_cams = [""] * MAX_CAMERA_SLOTS
        return no_update, *empty_cams, no_update, no_update

    gt_viz = set(gt_viz or [])
    trk_viz = set(trk_viz or [])
    active_groups = set(active_categories or [])

    def category_visible(cat_name):
        group = CATEGORY_TO_GROUP.get(cat_name)
        return group is not None and group in active_groups

    gt_errors = {}
    trk_errors = {}
    if is_debug and mot_acc and mot_id_map is not None:
        gt_errors, trk_errors = get_box_error_types(mot_acc, frame_idx, mot_id_map)

    if scene_loader:
        points = scene_loader.get_pointcloud(frame_idx)
    else:
        points = np.empty((0, 3), dtype=np.float32)

    ego_trans = np.zeros(3)
    ego_yaw = 0.0
    ego_rot_inv = np.eye(3)
    if scene_loader:
        ego_pose = scene_loader.get_ego_pose(frame_idx)
        if ego_pose:
            ego_trans = np.array(ego_pose["translation"])
            r = ego_pose["rotation"]
            w, x, y, z = r[0], r[1], r[2], r[3]
            ego_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            rot = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
            ])
            ego_rot_inv = rot.T

    def global_to_ego(translation, yaw):
        pos = ego_rot_inv @ (np.array(translation) - ego_trans)
        return pos.tolist(), yaw - ego_yaw

    gt_boxes_ego = None
    tracker_boxes_ego = None
    total_objects = 0

    if gt_viz and gt_data and frame_idx < len(gt_data):
        frame = gt_data[frame_idx]
        gt_boxes_ego = []
        for det in frame.get("detections", []):
            if not category_visible(det["category_name"]):
                continue
            pos, local_yaw = global_to_ego(det["translation"], det["yaw"])
            box = {
                "translation": pos,
                "size": det["size"],
                "yaw": local_yaw,
                "label": shorten_category(det["category_name"]),
                "instance_token": det.get("instance_token", ""),
            }
            if is_debug:
                err = gt_errors.get(det.get("instance_token", ""), "miss")
                box["debug_color"] = ERROR_COLORS[err]
            gt_boxes_ego.append(box)
        total_objects += len(gt_boxes_ego)

    if trk_viz and tracker_data and frame_idx < len(tracker_data):
        frame = tracker_data[frame_idx]
        tracker_boxes_ego = []
        for trk in frame.get("tracks", []):
            if not category_visible(trk["category_name"]):
                continue
            pos, local_yaw = global_to_ego(trk["translation"], trk["yaw"])
            box = {
                "translation": pos,
                "size": trk["size"],
                "yaw": local_yaw,
                "label": shorten_category(trk["category_name"]),
                "id": trk.get("id", 0),
                "age": trk.get("age", ""),
                "hits": trk.get("hits", ""),
                "misses": trk.get("consecutive_misses", ""),
            }
            if is_debug:
                err = trk_errors.get(trk.get("id", 0), "fp")
                box["debug_color"] = ERROR_COLORS[err]
                if err == "switch":
                    box["debug_line_width"] = SWITCH_LINE_WIDTH
            tracker_boxes_ego.append(box)
        total_objects += len(tracker_boxes_ego)

    fig = build_3d_figure(points, gt_boxes_ego, tracker_boxes_ego,
                          gt_viz=gt_viz, trk_viz=trk_viz,
                          show_ego_car=(scene_loader is not None),
                          top_down=(view_mode == "2d"),
                          white_pc=(pc_color_mode == "white"))

    cam_srcs = [""] * MAX_CAMERA_SLOTS
    if scene_loader and camera_names:
        cam_paths = scene_loader.get_camera_paths(frame_idx)
        for i, cam_name in enumerate(camera_names):
            if i >= MAX_CAMERA_SLOTS:
                break
            path = cam_paths.get(cam_name)
            if path:
                cam_srcs[i] = encode_image(path)

    timestamp = ""
    if scene_loader:
        timestamp = str(scene_loader.get_timestamp(frame_idx))
    elif gt_data and frame_idx < len(gt_data):
        timestamp = str(gt_data[frame_idx].get("timestamp", ""))
    elif tracker_data and frame_idx < len(tracker_data):
        timestamp = str(tracker_data[frame_idx].get("timestamp", ""))

    info_text = f"Frame {frame_idx}/{num_frames - 1}  |  Objects: {total_objects}"
    if timestamp and timestamp != "0":
        info_text += f"  |  TS: {timestamp}"

    debug_content = []
    if is_debug and mot_acc and mot_id_map is not None:
        debug_content = build_debug_panel(
            mot_acc, frame_idx, gt_data, tracker_data,
            set(debug_categories or []), mot_id_map
        )

    return fig, *cam_srcs, info_text, debug_content


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

    # -- Home button: back to config --
    @app.callback(
        Output("app-phase", "data", allow_duplicate=True),
        Input("btn-home", "n_clicks"),
        prevent_initial_call=True,
    )
    def go_home(n_clicks):
        return "config"

    # -- App mode toggle buttons --
    @app.callback(
        Output("config-app-mode", "data"),
        Output("btn-mode-viz", "style"),
        Output("btn-mode-debug", "style"),
        Output("config-debug-fields", "style"),
        Input("btn-mode-viz", "n_clicks"),
        Input("btn-mode-debug", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_app_mode(viz_clicks, debug_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        active_style = {
            "backgroundColor": "#00d4ff",
            "color": "#0f0f23",
            "border": "1px solid #00d4ff",
            "padding": "8px 24px",
            "cursor": "pointer",
            "fontSize": "13px",
            "fontWeight": "600",
        }
        inactive_style = {
            "backgroundColor": "transparent",
            "color": "#aaa",
            "border": "1px solid rgba(255,255,255,0.15)",
            "padding": "8px 24px",
            "cursor": "pointer",
            "fontSize": "13px",
            "fontWeight": "600",
        }
        viz_left = {**active_style, "borderRadius": "6px 0 0 6px"}
        viz_left_off = {**inactive_style, "borderRadius": "6px 0 0 6px"}
        debug_right = {**active_style, "borderRadius": "0 6px 6px 0"}
        debug_right_off = {**inactive_style, "borderRadius": "0 6px 6px 0"}

        if trigger == "btn-mode-debug":
            return "debug", viz_left_off, debug_right, {"display": "block"}
        return "visualization", viz_left, debug_right_off, {"display": "none"}

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
        State("config-app-mode", "data"),
        State("config-max-dist", "value"),
        State("config-scene-path", "value"),
        State("config-gt-upload", "contents"),
        State("config-gt-upload", "filename"),
        State("config-gt-path", "value"),
        State("config-trk-upload", "contents"),
        State("config-trk-upload", "filename"),
        State("config-trk-path", "value"),
        State("config-eval-categories", "value"),
        prevent_initial_call=True,
    )
    def launch(n_clicks, app_mode, max_dist, scene_path,
               gt_upload, gt_upload_name, gt_path,
               trk_upload, trk_upload_name, trk_path,
               eval_categories):
        try:
            return _do_launch(app_mode, max_dist, scene_path,
                              gt_upload, gt_upload_name, gt_path,
                              trk_upload, trk_upload_name, trk_path,
                              eval_categories)
        except Exception as e:
            logger.exception("Launch failed")
            return no_update, f"Error: {e}"

    def _do_launch(app_mode, max_dist, scene_path,
                   gt_upload, gt_upload_name, gt_path,
                   trk_upload, trk_upload_name, trk_path,
                   eval_categories):
        # Load scene directory (optional)
        scene_loader = None
        camera_names = []
        if scene_path and scene_path.strip():
            scene_path = scene_path.strip()
            if not Path(scene_path).is_dir():
                return no_update, f"Scene directory not found: {scene_path}"
            try:
                scene_loader = UniversalLoader(scene_path)
                camera_names = scene_loader.camera_names
            except Exception as e:
                return no_update, f"Error loading scene: {e}"

        # Must have at least one data source
        has_gt = bool(gt_upload) or bool(gt_path and gt_path.strip())
        has_trk = bool(trk_upload) or bool(trk_path and trk_path.strip())
        has_embedded_gt = scene_loader is not None and scene_loader.has_gt()
        if not has_gt and not has_embedded_gt and not has_trk and not scene_loader:
            return no_update, "Provide a scene directory and/or GT/Tracker file."

        # Debug mode requires both GT and tracker
        if app_mode == "debug":
            if (not has_gt and not has_embedded_gt) or not has_trk:
                return no_update, "Debug mode requires both GT and Tracker files."

        # Load GT (user-provided overrides embedded)
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
            elif has_embedded_gt:
                gt_data = scene_loader.load_gt()
                gt_name = "gt.json (from scene)"
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

        # Determine frames
        if scene_loader:
            num_frames = scene_loader.num_frames
        elif gt_data:
            num_frames = len(gt_data)
        elif tracker_data:
            num_frames = len(tracker_data)
        else:
            return no_update, "No data loaded."

        # Run MOT evaluation for debug mode
        mot_acc = None
        mot_id_map = None
        mot_summary = None
        if app_mode == "debug" and gt_data and tracker_data:
            dist_threshold = max_dist if max_dist and max_dist > 0 else 2.0
            allowed_cats = None
            if eval_categories:
                allowed_cats = set()
                for group in eval_categories:
                    allowed_cats.update(CATEGORY_GROUPS.get(group, []))
            try:
                mot_acc, mot_id_map = run_evaluation(
                    gt_data, tracker_data,
                    max_dist=dist_threshold,
                    allowed_categories=allowed_cats,
                )
                mot_summary = compute_summary(mot_acc)
            except Exception as e:
                return no_update, f"Error running MOT evaluation: {e}"

        # Populate server state
        _server_state.update({
            "scene_loader": scene_loader,
            "camera_names": camera_names,
            "gt_data": gt_data,
            "tracker_data": tracker_data,
            "num_frames": num_frames,
            "scene_mismatch": scene_mismatch,
            "app_mode": app_mode,
            "mot_accumulator": mot_acc,
            "mot_id_map": mot_id_map,
            "mot_summary": mot_summary,
        })

        return "viz", ""

    # -- View mode toggle --
    @app.callback(
        Output("store-view-mode", "data"),
        Output("btn-view-mode", "children"),
        Input("btn-view-mode", "n_clicks"),
        State("store-view-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_view_mode(n_clicks, current_mode):
        new_mode = "2d" if current_mode == "3d" else "3d"
        return new_mode, new_mode.upper()

    # -- Point cloud color toggle --
    @app.callback(
        Output("store-pc-color", "data"),
        Output("btn-pc-color", "style"),
        Input("btn-pc-color", "n_clicks"),
        State("store-pc-color", "data"),
        prevent_initial_call=True,
    )
    def toggle_pc_color(n_clicks, current):
        if current == "color":
            return "white", {**button_style("#ccc"), "fontSize": "10px"}
        return "color", {**button_style("#555"), "fontSize": "10px"}

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
        return new_playing, not new_playing, "⏸" if new_playing else "▶"

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
        if _server_state["num_frames"] == 0:
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

    # Camera output list
    cam_outputs = [Output(f"cam-{i}", "src") for i in range(MAX_CAMERA_SLOTS)]

    @app.callback(
        Output("scene-3d", "figure"),
        *cam_outputs,
        Output("frame-info", "children"),
        Output("debug-content", "children"),
        Input("store-frame", "data"),
        Input("check-gt-viz", "value"),
        Input("check-trk-viz", "value"),
        Input("check-categories", "value"),
        Input("store-view-mode", "data"),
        Input("store-pc-color", "data"),
        Input("check-debug-categories", "value"),
    )
    def render_frame(frame_idx, gt_viz, trk_viz, active_categories, view_mode,
                     pc_color_mode, debug_categories):
        is_debug = _server_state["app_mode"] == "debug"
        return compute_frame_data(
            frame_idx, gt_viz, trk_viz, active_categories,
            view_mode, pc_color_mode, debug_categories,
            is_debug, _server_state
        )

    # -- Metrics panel toggle --
    @app.callback(
        Output("metrics-panel", "style"),
        Output("metrics-arrow", "children"),
        Input("btn-metrics", "n_clicks"),
        State("metrics-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_metrics(n_clicks, current_style):
        if not current_style:
            current_style = {}
        if current_style.get("display") == "none":
            return {**current_style, "display": "block"}, "▲"
        return {**current_style, "display": "none"}, "▼"

    return app


def build_debug_panel(acc, frame_idx, gt_data, tracker_data, active_debug_groups,
                      int_to_token):
    events = get_frame_events(acc, frame_idx, int_to_token)

    gt_categories = {}
    if gt_data and frame_idx < len(gt_data):
        for det in gt_data[frame_idx].get("detections", []):
            gt_categories[det["instance_token"]] = det["category_name"]

    trk_categories = {}
    if tracker_data and frame_idx < len(tracker_data):
        for trk in tracker_data[frame_idx].get("tracks", []):
            trk_categories[trk["id"]] = trk["category_name"]

    def debug_cat_visible(gt_id=None, trk_id=None):
        cat = None
        if gt_id is not None:
            cat = gt_categories.get(gt_id)
        if cat is None and trk_id is not None:
            cat = trk_categories.get(trk_id)
        if cat is None:
            return True
        group = CATEGORY_TO_GROUP.get(cat)
        return group is not None and group in active_debug_groups

    children = []
    line_style = {"padding": "2px 0", "borderBottom": "1px solid rgba(255,255,255,0.04)"}

    matches = [m for m in events["matches"]
               if debug_cat_visible(gt_id=m["gt_id"], trk_id=m["trk_id"])]
    children.append(html.Div(
        f"✓ {len(matches)} Matches",
        style={**line_style, "color": ERROR_COLORS["match"], "fontWeight": "600"},
    ))

    switches = [s for s in events["switches"]
                if debug_cat_visible(gt_id=s["gt_id"], trk_id=s["trk_id"])]
    if switches:
        switch_items = [
            html.Div(
                f"✓ GT ...{s['gt_id'][-6:]} ↔ T{s['trk_id']}",
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            )
            for s in switches
        ]
        children.append(html.Div([
            html.Div(
                f"⚡ {len(switches)} ID Switches",
                style={**line_style, "color": ERROR_COLORS["switch"], "fontWeight": "600"},
            ),
            *switch_items,
        ]))
    else:
        children.append(html.Div(
            "⚡ 0 ID Switches",
            style={**line_style, "color": "#666"},
        ))

    fps = [fp for fp in events["false_positives"]
           if debug_cat_visible(trk_id=fp["trk_id"])]
    if fps:
        fp_ids = ", ".join(f"T{fp['trk_id']}" for fp in fps)
        children.append(html.Div([
            html.Div(
                f"✗ {len(fps)} False Positives",
                style={**line_style, "color": ERROR_COLORS["fp"], "fontWeight": "600"},
            ),
            html.Div(
                fp_ids,
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            ),
        ]))
    else:
        children.append(html.Div(
            "✗ 0 False Positives",
            style={**line_style, "color": "#666"},
        ))

    misses = [m for m in events["misses"]
              if debug_cat_visible(gt_id=m["gt_id"])]
    if misses:
        miss_ids = ", ".join(f"...{m['gt_id'][-6:]}" for m in misses)
        children.append(html.Div([
            html.Div(
                f"○ {len(misses)} Missed Detections",
                style={**line_style, "color": ERROR_COLORS["miss"], "fontWeight": "600"},
            ),
            html.Div(
                miss_ids,
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            ),
        ]))
    else:
        children.append(html.Div(
            "○ 0 Missed Detections",
            style={**line_style, "color": "#666"},
        ))

    return children


def _parse_upload(contents: str) -> list[dict]:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return json.loads(decoded)


def button_style(bg="#2c3e50"):
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


def _demo_scene_card(scene_key, dataset, scene_name, num_frames, num_cameras, app_mode):
    has_cameras = num_cameras > 0
    is_debug = app_mode == "debug"
    badge_text = "DEBUG" if is_debug else "VIZ"
    badge_color = "#e74c3c" if is_debug else "#1abc9c"

    info_items = [
        html.Span(f"{num_frames} frames", style={"color": "#888", "fontSize": "12px"}),
    ]
    if has_cameras:
        info_items.append(html.Span(f"{num_cameras} cameras", style={"color": "#888", "fontSize": "12px"}))
    info_items.append(html.Span("LiDAR + GT", style={"color": "#888", "fontSize": "12px"}))
    if is_debug:
        info_items.append(html.Span("Precomputed tracker comparison", style={"color": "#e74c3c", "fontSize": "12px", "fontWeight": "600"}))

    return html.Button(
        id={"type": "btn-scene", "index": scene_key},
        style={
            "backgroundColor": "#0f0f23",
            "borderRadius": "8px",
            "padding": "16px 20px",
            "border": "1px solid rgba(255,255,255,0.06)",
            "textAlign": "left",
            "cursor": "pointer",
            "width": "100%",
            "transition": "border-color 0.2s",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                children=[
                    html.Span(dataset, style={"color": "#00d4ff", "fontSize": "15px", "fontWeight": "600"}),
                    html.Span(badge_text, style={
                        "backgroundColor": badge_color, "color": "#fff",
                        "fontSize": "10px", "fontWeight": "700",
                        "padding": "2px 8px", "borderRadius": "4px", "letterSpacing": "1px",
                    }),
                ],
            ),
            html.Div(scene_name, style={"color": "#aaa", "fontSize": "13px", "marginBottom": "6px"}),
            html.Div(style={"display": "flex", "gap": "12px"}, children=info_items),
        ],
    )


def _demo_landing_layout(demo_scenes):
    scene_cards = []
    for key, s in demo_scenes.items():
        scene_cards.append(_demo_scene_card(
            key, s["dataset"], s["scene_name"],
            s["num_frames"], len(s["camera_names"]),
            s["app_mode"],
        ))

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
                style={
                    "backgroundColor": "#16213e",
                    "borderRadius": "12px",
                    "padding": "36px 40px",
                    "width": "520px",
                    "maxWidth": "90vw",
                    "boxShadow": "0 8px 32px rgba(0,0,0,0.4)",
                    "border": "1px solid rgba(255,255,255,0.06)",
                    "textAlign": "center",
                },
                children=[
                    html.H1([
                        "SensorLens ",
                        html.Span("(Demo)", style={"fontSize": "18px", "color": "#888", "fontWeight": "400"}),
                    ], style={
                        "color": "#00d4ff", "fontSize": "32px",
                        "fontWeight": "700", "margin": "0 0 4px 0",
                    }),
                    html.P("3D Multi-Object Tracking Visualizer", style={
                        "color": "#666", "fontSize": "13px", "margin": "0 0 20px 0",
                    }),
                    html.Div(
                        style={
                            "backgroundColor": "rgba(255,165,0,0.1)",
                            "border": "1px solid rgba(255,165,0,0.3)",
                            "borderRadius": "6px",
                            "padding": "10px 14px",
                            "marginBottom": "8px",
                            "textAlign": "left",
                        },
                        children=[
                            html.Span("This is a limited demo with minimal functionality.", style={
                                "color": "#e0a030", "fontSize": "12px",
                            }),
                        ],
                    ),
                    html.Div(
                        style={
                            "backgroundColor": "rgba(0,212,255,0.06)",
                            "border": "1px solid rgba(0,212,255,0.2)",
                            "borderRadius": "6px",
                            "padding": "10px 14px",
                            "marginBottom": "16px",
                            "textAlign": "left",
                        },
                        children=[
                            html.Span("Run locally with Docker for full functionality.", style={
                                "color": "#5bb8d0", "fontSize": "12px",
                            }),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
                        children=scene_cards,
                    ),
                    html.Div(style={"marginTop": "20px"}, children=[
                        html.A(
                            children=[
                                html.Img(
                                    src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjODg4Ij48cGF0aCBkPSJNMTIgMEM1LjM3IDAgMCA1LjM3IDAgMTJjMCA1LjMgMy40MzggOS44IDguMjA1IDExLjM4NS42LjExMy44Mi0uMjU4LjgyLS41NzcgMC0uMjg1LS4wMS0xLjA0LS4wMTUtMi4wNC0zLjMzOC43MjQtNC4wNDItMS42MS00LjA0Mi0xLjYxLS41NDYtMS4zODUtMS4zMzUtMS43MjUtMS4zMzUtMS43MjUtMS4wODctLjc0NS4wODQtLjcyOS4wODQtLjcyOSAxLjIwNS4wODQgMS44MzggMS4yMzYgMS44MzggMS4yMzYgMS4wNyAxLjgzNSAyLjgwOSAxLjMwNSAzLjQ5NS45OTguMTA4LS43NzYuNDE3LTEuMzA1Ljc2LTEuNjA1LTIuNjY1LS4zLTUuNDY2LTEuMzMyLTUuNDY2LTUuOTMgMC0xLjMxLjQ2NS0yLjM4IDEuMjM1LTMuMjItLjEzNS0uMzAzLS41NC0xLjUyMy4xMDUtMy4xNzYgMCAwIDEuMDA1LS4zMjIgMy4zIDEuMjMuOTYtLjI2NyAxLjk4LS4zOTkgMy0uNDA1IDEuMDIuMDA2IDIuMDQuMTM4IDMgLjQwNSAyLjI4LTEuNTUyIDMuMjg1LTEuMjMgMy4yODUtMS4yMy42NDUgMS42NTMuMjQgMi44NzMuMTIgMy4xNzYuNzY1Ljg0IDEuMjMgMS45MSAxLjIzIDMuMjIgMCA0LjYxLTIuODA1IDUuNjI1LTUuNDc1IDUuOTIuNDIuMzYuODEgMS4wOTYuODEgMi4yMiAwIDEuNjA1LS4wMTUgMi44OTYtLjAxNSAzLjI4NiAwIC4zMTUuMjEuNjkuODI1LjU3QzIwLjU2NSAyMS43OTYgMjQgMTcuMyAyNCAxMmMwLTYuNjMtNS4zNy0xMi0xMi0xMnoiLz48L3N2Zz4=",
                                    style={"width": "20px", "height": "20px", "verticalAlign": "middle", "marginRight": "6px"},
                                ),
                                html.Span("GitHub", style={"verticalAlign": "middle"}),
                            ],
                            href="https://github.com/nairb36/SensorLens",
                            target="_blank",
                            style={
                                "color": "#888", "fontSize": "12px", "textDecoration": "none",
                                "display": "inline-flex", "alignItems": "center",
                                "border": "1px solid rgba(255,255,255,0.1)",
                                "borderRadius": "6px", "padding": "6px 14px",
                            },
                        ),
                    ]),
                ],
            ),
        ],
    )


def create_demo_app(demo_scenes: dict) -> Dash:
    first_key = next(iter(demo_scenes))
    _server_state.update(demo_scenes[first_key])
    _server_state["demo_mode"] = True

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        dcc.Store(id="app-phase", data="landing"),
        dcc.Store(id="store-scene", data=first_key, storage_type="session"),
        html.Div(id="page-content"),
    ])

    @app.callback(
        Output("page-content", "children"),
        Input("app-phase", "data"),
    )
    def switch_page(phase):
        if phase == "viz":
            return _viz_layout()
        return _demo_landing_layout(demo_scenes)

    @app.callback(
        Output("app-phase", "data"),
        Output("store-scene", "data"),
        Input({"type": "btn-scene", "index": dash.ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def scene_selected(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return no_update, no_update
        triggered = ctx.triggered[0]["prop_id"]
        scene_key = json.loads(triggered.split(".")[0])["index"]
        _server_state.update(demo_scenes[scene_key])
        _server_state["demo_mode"] = True
        return "viz", scene_key

    @app.callback(
        Output("app-phase", "data", allow_duplicate=True),
        Input("btn-home", "n_clicks"),
        prevent_initial_call=True,
    )
    def go_home(n_clicks):
        return "landing"

    @app.callback(
        Output("store-view-mode", "data"),
        Output("btn-view-mode", "children"),
        Input("btn-view-mode", "n_clicks"),
        State("store-view-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_view_mode(n_clicks, current_mode):
        new_mode = "2d" if current_mode == "3d" else "3d"
        return new_mode, new_mode.upper()

    @app.callback(
        Output("store-pc-color", "data"),
        Output("btn-pc-color", "style"),
        Input("btn-pc-color", "n_clicks"),
        State("store-pc-color", "data"),
        prevent_initial_call=True,
    )
    def toggle_pc_color(n_clicks, current):
        if current == "color":
            return "white", {**button_style("#ccc"), "fontSize": "10px"}
        return "color", {**button_style("#555"), "fontSize": "10px"}

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
        return new_playing, not new_playing, "⏸" if new_playing else "▶"

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
        num = _server_state["num_frames"]
        if num == 0:
            return no_update, no_update
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "btn-prev":
            new_frame = max(0, current_frame - 1)
        elif trigger_id == "btn-next":
            new_frame = min(num - 1, current_frame + 1)
        elif trigger_id == "play-interval":
            if not playing:
                return no_update, no_update
            new_frame = current_frame + 1
            if new_frame >= num:
                new_frame = 0
        elif trigger_id == "frame-slider":
            new_frame = slider_val
        else:
            return no_update, no_update
        return new_frame, new_frame

    cam_outputs = [Output(f"cam-{i}", "src") for i in range(MAX_CAMERA_SLOTS)]

    @app.callback(
        Output("scene-3d", "figure"),
        *cam_outputs,
        Output("frame-info", "children"),
        Output("debug-content", "children"),
        Input("store-frame", "data"),
        Input("check-gt-viz", "value"),
        Input("check-trk-viz", "value"),
        Input("check-categories", "value"),
        Input("store-view-mode", "data"),
        Input("store-pc-color", "data"),
        Input("check-debug-categories", "value"),
    )
    def render_frame(frame_idx, gt_viz, trk_viz, active_categories, view_mode,
                     pc_color_mode, debug_categories):
        is_debug = _server_state["app_mode"] == "debug"
        return compute_frame_data(
            frame_idx, gt_viz, trk_viz, active_categories,
            view_mode, pc_color_mode, debug_categories,
            is_debug, _server_state
        )

    @app.callback(
        Output("metrics-panel", "style"),
        Output("metrics-arrow", "children"),
        Input("btn-metrics", "n_clicks"),
        State("metrics-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_metrics(n_clicks, current_style):
        if not current_style:
            current_style = {}
        if current_style.get("display") == "none":
            return {**current_style, "display": "block"}, "▲"
        return {**current_style, "display": "none"}, "▼"

    return app
