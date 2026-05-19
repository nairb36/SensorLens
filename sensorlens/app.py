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

FRONT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
REAR_CAMS = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

_server_state = {
    "mode": None,  # "nuscenes", "waymo", "custom"
    "nusc_loader": None,
    "gt_data": None,
    "tracker_data": None,
    "sample_tokens": None,
    "num_frames": 0,
    "front_stitcher": None,
    "rear_stitcher": None,
    "scene_mismatch": False,
    "origin_offset": None,  # fixed centroid from frame 0 for custom mode
    "app_mode": "visualization",  # "visualization" or "debug"
    "mot_accumulator": None,
    "mot_id_map": None,
    "mot_summary": None,
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
                        ],
                    ),
                    # Dataset type
                    _config_label("Dataset Type"),
                    dcc.Dropdown(
                        id="config-dataset-type",
                        options=[
                            {"label": "NuScenes", "value": "nuscenes"},
                            {"label": "Waymo Open Dataset", "value": "waymo"},
                            {"label": "Custom (bounding boxes only)", "value": "custom"},
                        ],
                        value="nuscenes",
                        clearable=False,
                        style={"marginBottom": "16px"},
                    ),
                    # Dataset-specific fields (hidden/shown dynamically)
                    html.Div(
                        id="config-dataset-fields",
                        children=[
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
                        ],
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
    mode = s["mode"]
    app_mode = s["app_mode"]
    is_debug = app_mode == "debug"
    show_panos = mode != "custom"

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

    # Panorama panel (only for NuScenes/Waymo)
    pano_children = []
    if show_panos:
        pano_children.extend([
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
        ])

    # Debug panel (shown in debug mode)
    if is_debug:
        debug_cat_options = [
            {"label": html.Span(g, style={"color": "#999"}), "value": g}
            for g in CATEGORY_GROUPS
        ]
        debug_cat_defaults = [g for g in CATEGORY_GROUPS if g in DEFAULT_ON]

        # Color legend
        legend_items = [
            ("Match", ERROR_COLORS["match"]),
            ("ID Switch", ERROR_COLORS["switch"]),
            ("FP", ERROR_COLORS["fp"]),
            ("Missed", ERROR_COLORS["miss"]),
        ]

        pano_children.append(
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
                                        [html.Span("\u25a0 ", style={"color": c}), name],
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
                        html.Span("\u25bc", id="metrics-arrow", style={"fontSize": "10px"}),
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
    if not show_panos:
        hidden_placeholders.extend([
            html.Img(id="pano-front", style={"display": "none"}),
            html.Img(id="pano-rear", style={"display": "none"}),
        ])

    show_right_panel = show_panos or is_debug
    pano_panel = html.Div(
        style={
            "flex": "1",
            "minWidth": "0",
            "display": "flex" if show_right_panel else "none",
            "flexDirection": "column",
            "gap": "4px",
            "overflowY": "auto",
        },
        children=pano_children,
    )

    main_children = [scene_panel, pano_panel]

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
                                "\u2302",
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
                                "SensorLens",
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
                    html.Button("\u23ee", id="btn-prev", style=_button_style()),
                    html.Button(
                        "\u25b6", id="btn-play",
                        style={**_button_style("#1abc9c"), "display": "none" if is_debug else "inline-block"},
                    ),
                    html.Button("\u23ed", id="btn-next", style=_button_style()),
                    html.Button("3D", id="btn-view-mode", style=_button_style("#8e44ad")),
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
            dcc.Interval(id="play-interval", interval=500, disabled=True),
            *hidden_placeholders,
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

    # -- Show/hide dataset fields based on type --
    @app.callback(
        Output("config-dataset-fields", "style"),
        Input("config-dataset-type", "value"),
    )
    def toggle_dataset_fields(dataset_type):
        if dataset_type == "custom":
            return {"display": "none"}
        return {"display": "block"}

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
        State("config-dataset-type", "value"),
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
    def launch(n_clicks, app_mode, max_dist, dataset_type, dataroot, version,
               gt_upload, gt_upload_name, gt_path, trk_upload, trk_upload_name, trk_path):

        # Validate dataroot for non-custom modes
        if dataset_type != "custom":
            if not dataroot or not dataroot.strip():
                return no_update, "Please enter a dataroot path."
            dataroot = dataroot.strip()
            if not Path(dataroot).is_dir():
                return no_update, f"Dataroot not found: {dataroot}"

        # Must have at least one data file
        has_gt = bool(gt_upload) or bool(gt_path and gt_path.strip())
        has_trk = bool(trk_upload) or bool(trk_path and trk_path.strip())
        if not has_gt and not has_trk:
            return no_update, "Provide at least one of GT or Tracker file."

        # Debug mode requires both GT and tracker
        if app_mode == "debug" and (not has_gt or not has_trk):
            return no_update, "Debug mode requires both GT and Tracker files."

        # Debug mode only for NuScenes/Waymo (not custom)
        if app_mode == "debug" and dataset_type == "custom":
            return no_update, "Debug mode is not available for Custom datasets."

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

        # Determine frames
        if gt_data:
            num_frames = len(gt_data)
            sample_tokens = [f.get("sample_token", f"frame_{i}") for i, f in enumerate(gt_data)]
        elif tracker_data:
            num_frames = len(tracker_data)
            sample_tokens = [f.get("sample_token", f"frame_{i}") for i, f in enumerate(tracker_data)]
        else:
            return no_update, "No data loaded."

        # Mode-specific initialization
        nusc_loader = None
        front_stitcher = None
        rear_stitcher = None

        if dataset_type == "custom":
            # No NuScenes needed — boxes assumed in ego frame
            pass
        elif dataset_type == "waymo":
            return no_update, "Waymo Open Dataset support coming soon."
        else:
            # NuScenes
            try:
                nusc_loader = NuScenesLoader(dataroot, version)
            except Exception as e:
                return no_update, f"Error loading NuScenes: {e}"

            # If tracker-only with no sample_tokens from data, walk the scene
            if gt_data is None and tracker_data:
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

        # Compute fixed origin offset for custom mode (centroid of frame 0)
        origin_offset = np.zeros(3)
        if dataset_type == "custom":
            translations = []
            if gt_data:
                for det in gt_data[0].get("detections", []):
                    translations.append(det["translation"])
            if tracker_data:
                for trk in tracker_data[0].get("tracks", []):
                    translations.append(trk["translation"])
            if translations:
                origin_offset = np.mean(translations, axis=0)

        # Run MOT evaluation for debug mode
        mot_acc = None
        mot_id_map = None
        mot_summary = None
        if app_mode == "debug" and gt_data and tracker_data:
            dist_threshold = max_dist if max_dist and max_dist > 0 else 2.0
            try:
                mot_acc, mot_id_map = run_evaluation(gt_data, tracker_data, max_dist=dist_threshold)
                mot_summary = compute_summary(mot_acc)
            except Exception as e:
                return no_update, f"Error running MOT evaluation: {e}"

        # Populate server state
        _server_state.update({
            "mode": dataset_type,
            "nusc_loader": nusc_loader,
            "gt_data": gt_data,
            "tracker_data": tracker_data,
            "sample_tokens": sample_tokens,
            "num_frames": num_frames,
            "front_stitcher": front_stitcher,
            "rear_stitcher": rear_stitcher,
            "scene_mismatch": scene_mismatch,
            "origin_offset": origin_offset,
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
        return new_playing, not new_playing, "\u23f8" if new_playing else "\u25b6"

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

    @app.callback(
        Output("scene-3d", "figure"),
        Output("pano-front", "src"),
        Output("pano-rear", "src"),
        Output("frame-info", "children"),
        Output("debug-content", "children"),
        Input("store-frame", "data"),
        Input("check-gt-viz", "value"),
        Input("check-trk-viz", "value"),
        Input("check-categories", "value"),
        Input("store-view-mode", "data"),
        Input("check-debug-categories", "value"),
    )
    def render_frame(frame_idx, gt_viz, trk_viz, active_categories, view_mode,
                     debug_categories):
        s = _server_state
        if s["num_frames"] == 0:
            return no_update, no_update, no_update, no_update, no_update

        mode = s["mode"]
        nusc_loader = s["nusc_loader"]
        gt_data = s["gt_data"]
        tracker_data = s["tracker_data"]
        sample_tokens = s["sample_tokens"]
        num_frames = s["num_frames"]
        front_stitcher = s["front_stitcher"]
        rear_stitcher = s["rear_stitcher"]
        is_debug = s["app_mode"] == "debug"
        mot_acc = s["mot_accumulator"]
        mot_id_map = s["mot_id_map"]

        if frame_idx is None or frame_idx < 0 or frame_idx >= num_frames:
            return no_update, no_update, no_update, no_update, no_update

        gt_viz = set(gt_viz or [])
        trk_viz = set(trk_viz or [])
        active_groups = set(active_categories or [])

        def category_visible(cat_name):
            group = CATEGORY_TO_GROUP.get(cat_name)
            return group is not None and group in active_groups

        # Get debug error types for this frame
        gt_errors = {}
        trk_errors = {}
        if is_debug and mot_acc and mot_id_map is not None:
            gt_errors, trk_errors = get_box_error_types(mot_acc, frame_idx, mot_id_map)

        # Get point cloud and ego pose (NuScenes) or empty (custom)
        if mode == "custom":
            points = np.empty((0, 3), dtype=np.float32)
            origin_offset = s["origin_offset"]
        else:
            sample_token = sample_tokens[frame_idx]
            ego_pose = nusc_loader.get_ego_pose(sample_token)
            points = nusc_loader.get_lidar_points_ego(sample_token)

        gt_boxes_ego = None
        tracker_boxes_ego = None
        total_objects = 0

        if gt_viz and gt_data and frame_idx < len(gt_data):
            frame = gt_data[frame_idx]
            gt_boxes_ego = []
            for det in frame.get("detections", []):
                if not category_visible(det["category_name"]):
                    continue
                if mode == "custom":
                    pos = (np.array(det["translation"]) - origin_offset).tolist()
                    yaw = det["yaw"]
                else:
                    pos, yaw = global_to_ego(det["translation"], det["yaw"], ego_pose)
                    pos = pos.tolist()
                box = {
                    "translation": pos,
                    "size": det["size"],
                    "yaw": yaw,
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
                if mode == "custom":
                    pos = (np.array(trk["translation"]) - origin_offset).tolist()
                    yaw = trk["yaw"]
                else:
                    pos, yaw = global_to_ego(trk["translation"], trk["yaw"], ego_pose)
                    pos = pos.tolist()
                box = {
                    "translation": pos,
                    "size": trk["size"],
                    "yaw": yaw,
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
                              show_ego_car=(mode != "custom"),
                              top_down=(view_mode == "2d"))

        # Panoramas (NuScenes only)
        front_src = ""
        rear_src = ""
        if mode != "custom" and nusc_loader and front_stitcher and rear_stitcher:
            sample_token = sample_tokens[frame_idx]
            cam_paths = nusc_loader.get_camera_paths(sample_token)
            front_imgs = [cv2.imread(cam_paths[c]) for c in FRONT_CAMS]
            rear_imgs = [cv2.imread(cam_paths[c]) for c in REAR_CAMS]
            front_src = encode_panorama(front_stitcher.stitch(front_imgs))
            rear_src = encode_panorama(rear_stitcher.stitch(rear_imgs))

        # Frame info
        timestamp = ""
        if gt_data and frame_idx < len(gt_data):
            timestamp = str(gt_data[frame_idx].get("timestamp", ""))
        elif tracker_data and frame_idx < len(tracker_data):
            timestamp = str(tracker_data[frame_idx].get("timestamp", ""))

        token_str = sample_tokens[frame_idx][:8] if sample_tokens else ""
        info_text = f"Frame {frame_idx}/{num_frames - 1}  |  Objects: {total_objects}"
        if token_str and not token_str.startswith("frame_"):
            info_text += f"  |  Token: {token_str}..."
        if timestamp:
            info_text += f"  |  TS: {timestamp}"

        # Debug panel content
        debug_content = []
        if is_debug and mot_acc and mot_id_map is not None:
            debug_content = _build_debug_panel(
                mot_acc, frame_idx, gt_data, tracker_data,
                set(debug_categories or []), mot_id_map
            )

        return fig, front_src, rear_src, info_text, debug_content

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
            return {**current_style, "display": "block"}, "\u25b2"
        return {**current_style, "display": "none"}, "\u25bc"

    return app


def _build_debug_panel(acc, frame_idx, gt_data, tracker_data, active_debug_groups,
                       int_to_token):
    """Build the debug log panel content for a specific frame."""
    events = get_frame_events(acc, frame_idx, int_to_token)

    # Build category lookup for filtering debug output
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

    # Matches
    matches = [m for m in events["matches"]
               if debug_cat_visible(gt_id=m["gt_id"], trk_id=m["trk_id"])]
    children.append(html.Div(
        f"\u2713 {len(matches)} Matches",
        style={**line_style, "color": ERROR_COLORS["match"], "fontWeight": "600"},
    ))

    # ID Switches
    switches = [s for s in events["switches"]
                if debug_cat_visible(gt_id=s["gt_id"], trk_id=s["trk_id"])]
    if switches:
        switch_items = [
            html.Div(
                f"\u2713 GT ...{s['gt_id'][-6:]} \u2194 T{s['trk_id']}",
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            )
            for s in switches
        ]
        children.append(html.Div([
            html.Div(
                f"\u26a1 {len(switches)} ID Switches",
                style={**line_style, "color": ERROR_COLORS["switch"], "fontWeight": "600"},
            ),
            *switch_items,
        ]))
    else:
        children.append(html.Div(
            "\u26a1 0 ID Switches",
            style={**line_style, "color": "#666"},
        ))

    # False Positives
    fps = [fp for fp in events["false_positives"]
           if debug_cat_visible(trk_id=fp["trk_id"])]
    if fps:
        fp_ids = ", ".join(f"T{fp['trk_id']}" for fp in fps)
        children.append(html.Div([
            html.Div(
                f"\u2717 {len(fps)} False Positives",
                style={**line_style, "color": ERROR_COLORS["fp"], "fontWeight": "600"},
            ),
            html.Div(
                fp_ids,
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            ),
        ]))
    else:
        children.append(html.Div(
            "\u2717 0 False Positives",
            style={**line_style, "color": "#666"},
        ))

    # Misses
    misses = [m for m in events["misses"]
              if debug_cat_visible(gt_id=m["gt_id"])]
    if misses:
        miss_ids = ", ".join(f"...{m['gt_id'][-6:]}" for m in misses)
        children.append(html.Div([
            html.Div(
                f"\u25cb {len(misses)} Missed Detections",
                style={**line_style, "color": ERROR_COLORS["miss"], "fontWeight": "600"},
            ),
            html.Div(
                miss_ids,
                style={"paddingLeft": "12px", "color": "#ddd", "fontSize": "11px"},
            ),
        ]))
    else:
        children.append(html.Div(
            "\u25cb 0 Missed Detections",
            style={**line_style, "color": "#666"},
        ))

    return children


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
