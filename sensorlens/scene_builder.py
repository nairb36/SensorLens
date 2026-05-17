import functools
from pathlib import Path

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
        [-hl, -hw, -hh],
        [ hl, -hw, -hh],
        [ hl,  hw, -hh],
        [-hl,  hw, -hh],
        [-hl, -hw,  hh],
        [ hl, -hw,  hh],
        [ hl,  hw,  hh],
        [-hl,  hw,  hh],
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


BOX_FACES_I = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 2, 2]
BOX_FACES_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 3, 7]
BOX_FACES_K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 7, 6]


def _hex_to_rgba(hex_color: str, opacity: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


def build_wireframe_traces(
    corners: np.ndarray, color: str, label: str, tag: str = ""
) -> list[go.Scatter3d]:
    xs, ys, zs = [], [], []
    for i, j in BOX_EDGES:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    traces = [go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        hoverinfo="text",
        hovertext=label,
        showlegend=False,
    )]

    if tag:
        top_z = corners[:, 2].max() + 0.3
        center = corners.mean(axis=0)
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[top_z],
            mode="text",
            text=[tag],
            textfont=dict(size=9, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    return traces


def build_solid_box_traces(
    corners: np.ndarray, color: str, label: str
) -> list:
    fill_color = _hex_to_rgba(color, 0.25)
    return [go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=BOX_FACES_I, j=BOX_FACES_J, k=BOX_FACES_K,
        color=fill_color,
        flatshading=True,
        hoverinfo="text",
        hovertext=label,
        showlegend=False,
    )]


def build_center_trace(
    translation: np.ndarray, color: str, label: str
) -> go.Scatter3d:
    return go.Scatter3d(
        x=[translation[0]], y=[translation[1]], z=[translation[2]],
        mode="markers",
        marker=dict(size=6, color=color, symbol="circle"),
        hoverinfo="text",
        hovertext=label,
        showlegend=False,
    )


def build_point_cloud_trace(
    points: np.ndarray, max_points: int = 50000
) -> go.Scatter3d:
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    z_vals = points[:, 2]

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=1.2,
            color=z_vals,
            colorscale="Turbo",
            cmin=-3.0,
            cmax=8.0,
            opacity=0.6,
        ),
        hoverinfo="skip",
        showlegend=False,
    )


def _parse_mtl(path: Path) -> dict[str, str]:
    materials: dict[str, str] = {}
    current = None
    with open(path) as f:
        for line in f:
            if line.startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
            elif line.startswith("Kd ") and current:
                r, g, b = (float(x) for x in line.split()[1:4])
                # Blender linear → sRGB approximation for display
                sr, sg, sb = (int(min(x ** 0.45, 1.0) * 255) for x in (r, g, b))
                materials[current] = f"#{sr:02x}{sg:02x}{sb:02x}"
    return materials


@functools.lru_cache(maxsize=1)
def _load_ego_obj() -> tuple[np.ndarray, list[list[int]], list[str]]:
    assets = Path(__file__).parent / "assets"
    obj_path = assets / "NormalCar2.obj"
    mtl_path = assets / "NormalCar2.mtl"

    mtl_colors = _parse_mtl(mtl_path) if mtl_path.exists() else {}

    raw_verts: list[list[float]] = []
    faces: list[list[int]] = []
    face_colors: list[str] = []
    current_color = "#758ca3"

    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                raw_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("usemtl "):
                mat_name = line.split(None, 1)[1].strip()
                current_color = mtl_colors.get(mat_name, "#758ca3")
            elif line.startswith("f "):
                idxs = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                if len(idxs) == 3:
                    faces.append(idxs)
                    face_colors.append(current_color)
                else:
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
                        face_colors.append(current_color)

    v = np.array(raw_verts)
    # OBJ from Blender (X=right, Y=up, Z=back) → nuScenes (X=fwd, Y=left, Z=up)
    remapped = np.column_stack([v[:, 2], -v[:, 0], v[:, 1]])

    remapped[:, 0] -= remapped[:, 0].mean()
    remapped[:, 1] -= remapped[:, 1].mean()
    remapped[:, 2] -= remapped[:, 2].min()

    scale = 4.5 / np.ptp(remapped[:, 0])
    remapped *= scale

    remapped[:, 0] -= remapped[:, 0].mean()
    remapped[:, 1] -= remapped[:, 1].mean()

    return remapped, faces, face_colors


def build_ego_car() -> list:
    verts, faces, face_colors = _load_ego_obj()

    body_mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        facecolor=face_colors,
        opacity=1.0,
        hoverinfo="skip",
        showlegend=False,
        flatshading=True,
    )

    front_x = verts[:, 0].max()
    arrow_z = verts[:, 2].max() + 0.2
    arrow = [
        (front_x + 0.8, 0), (front_x + 0.1, -0.5),
        (front_x + 0.3, 0), (front_x + 0.1, 0.5),
        (front_x + 0.8, 0),
    ]
    ax, ay, az = [], [], []
    for k in range(len(arrow) - 1):
        ax.extend([arrow[k][0], arrow[k + 1][0], None])
        ay.extend([arrow[k][1], arrow[k + 1][1], None])
        az.extend([arrow_z, arrow_z, None])

    direction = go.Scatter3d(
        x=ax, y=ay, z=az,
        mode="lines",
        line=dict(color="rgba(0,212,255,0.7)", width=3),
        hoverinfo="skip",
        showlegend=False,
    )

    return [body_mesh, direction]


def build_3d_figure(
    points: np.ndarray,
    gt_boxes: list[dict] | None = None,
    tracker_boxes: list[dict] | None = None,
    x_range: float = 80.0,
    y_range: float = 80.0,
    gt_viz: set | None = None,
    trk_viz: set | None = None,
    show_ego_car: bool = True,
) -> go.Figure:
    gt_viz = gt_viz or set()
    trk_viz = trk_viz or set()
    fig = go.Figure()

    if len(points) > 0:
        fig.add_trace(build_point_cloud_trace(points))
    if show_ego_car:
        fig.add_traces(build_ego_car())

    if gt_boxes:
        for box in gt_boxes:
            translation = np.array(box["translation"])
            identity = box.get("instance_token", "")
            short_id = identity[-6:] if len(identity) > 6 else identity
            label = f'GT {box["label"]} ({short_id})'
            color = get_identity_color(identity)
            if "bbox" in gt_viz:
                corners = build_box_corners(translation, box["size"], box["yaw"])
                fig.add_traces(build_solid_box_traces(corners, color, label))
            if "center" in gt_viz:
                fig.add_trace(build_center_trace(translation, color, label))

    if tracker_boxes:
        for box in tracker_boxes:
            translation = np.array(box["translation"])
            track_id = box.get("id", 0)
            hover_lines = [f'T{track_id} {box["label"]}']
            if box.get("age") != "":
                hover_lines.append(f'age: {box["age"]}')
            if box.get("hits") != "":
                hover_lines.append(f'hits: {box["hits"]}')
            if box.get("misses") != "":
                hover_lines.append(f'misses: {box["misses"]}')
            label = "<br>".join(hover_lines)
            color = get_identity_color(track_id)
            if "bbox" in trk_viz:
                corners = build_box_corners(translation, box["size"], box["yaw"])
                tag = f'{box["label"]}·{track_id}'
                fig.add_traces(build_wireframe_traces(corners, color, label, tag=tag))
            if "center" in trk_viz:
                fig.add_trace(build_center_trace(translation, color, label))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-x_range, x_range], showbackground=False, showgrid=True, gridcolor="#333", title="", showticklabels=False),
            yaxis=dict(range=[-y_range, y_range], showbackground=False, showgrid=True, gridcolor="#333", title="", showticklabels=False),
            zaxis=dict(range=[-5, 10], showbackground=False, showgrid=False, title="", showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.15),
            camera=dict(
                eye=dict(x=0, y=0, z=2.0),
                up=dict(x=1, y=0, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            bgcolor="#1a1a2e",
        ),
        paper_bgcolor="#0f0f23",
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        uirevision="constant",
    )

    return fig
