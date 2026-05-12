import numpy as np
import plotly.graph_objects as go

IDENTITY_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
    "#1abc9c", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
]

GT_DEFAULT_COLOR = "#00ff88"
TRACKER_DEFAULT_COLOR = "#ff6644"


def get_identity_color(identity, palette=IDENTITY_COLORS) -> str:
    if isinstance(identity, int):
        return palette[identity % len(palette)]
    return palette[hash(identity) % len(palette)]


def build_box_corners(translation: np.ndarray, size: list, yaw: float) -> np.ndarray:
    w, l, h = size
    hw, hl, hh = w / 2, l / 2, h / 2

    corners = np.array([
        [-hw, -hl, -hh],
        [ hw, -hl, -hh],
        [ hw,  hl, -hh],
        [-hw,  hl, -hh],
        [-hw, -hl,  hh],
        [ hw, -hl,  hh],
        [ hw,  hl,  hh],
        [-hw,  hl,  hh],
    ])

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])

    corners = corners @ rot.T
    corners += translation
    return corners


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def build_box_traces(
    corners: np.ndarray, color: str, label: str, show_label: bool = True
) -> list[go.Scatter3d]:
    traces = []
    xs, ys, zs = [], [], []
    for i, j in BOX_EDGES:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        hoverinfo="text",
        hovertext=label,
        showlegend=False,
    ))

    if show_label:
        center = corners.mean(axis=0)
        top_z = corners[:, 2].max() + 0.3
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[top_z],
            mode="text",
            text=[label],
            textfont=dict(size=10, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    return traces


def build_point_cloud_trace(
    points: np.ndarray, max_points: int = 50000
) -> go.Scatter3d:
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    if z_max - z_min < 0.01:
        z_max = z_min + 1.0

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=1.2,
            color=z_vals,
            colorscale="Turbo",
            cmin=z_min,
            cmax=z_max,
            opacity=0.6,
        ),
        hoverinfo="skip",
        showlegend=False,
    )


def build_ego_marker() -> go.Scatter3d:
    size = 1.5
    xs = [0, size, 0, -size, 0, 0]
    ys = [size, 0, -size, 0, 0, 0]
    zs = [0, 0, 0, 0, -1, 1]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=6, color="white", symbol="diamond"),
        hoverinfo="text",
        hovertext="Ego Vehicle",
        showlegend=False,
    )


def build_3d_figure(
    points: np.ndarray,
    gt_boxes: list[dict] | None = None,
    tracker_boxes: list[dict] | None = None,
    x_range: float = 80.0,
    y_range: float = 80.0,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(build_point_cloud_trace(points))
    fig.add_trace(build_ego_marker())

    if gt_boxes:
        for box in gt_boxes:
            corners = build_box_corners(
                np.array(box["translation"]),
                box["size"],
                box["yaw"],
            )
            identity = box.get("instance_token", "")
            short_id = identity[-6:] if len(identity) > 6 else identity
            label = f'{box["label"]} ({short_id})'
            color = get_identity_color(identity)
            fig.add_traces(build_box_traces(corners, color, label))

    if tracker_boxes:
        for box in tracker_boxes:
            corners = build_box_corners(
                np.array(box["translation"]),
                box["size"],
                box["yaw"],
            )
            track_id = box.get("id", 0)
            label = f'{box["label"]} T{track_id}'
            color = get_identity_color(track_id)
            fig.add_traces(build_box_traces(corners, color, label))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-x_range, x_range], showbackground=False, showgrid=True, gridcolor="#333", title="", showticklabels=False),
            yaxis=dict(range=[-y_range, y_range], showbackground=False, showgrid=True, gridcolor="#333", title="", showticklabels=False),
            zaxis=dict(range=[-5, 10], showbackground=False, showgrid=False, title="", showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
            camera=dict(
                eye=dict(x=0, y=0, z=2.0),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            bgcolor="#1a1a2e",
        ),
        paper_bgcolor="#0f0f23",
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        uirevision="constant",
    )

    return fig
