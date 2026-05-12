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


def _quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def build_ego_car() -> list:
    hw = 0.95
    # y-forward cross-sections, each is a closed ring of (x, z) at a given y
    # Defines: undercarriage, rear bumper, rear body, rear window base,
    #          roof rear, roof front, windshield base, hood, front bumper, front tip
    sections = {
        "under_rear":    (-2.30, [(-hw, -0.30), ( hw, -0.30), ( hw, -0.30), (-hw, -0.30)]),
        "rear_bumper":   (-2.30, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.50), (-hw,  0.50)]),
        "rear_body":     (-1.80, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.70), (-hw,  0.70)]),
        "trunk_top":     (-1.40, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.70), (-hw,  0.70)]),
        "rear_win_base": (-1.00, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.72), (-hw,  0.72)]),
        "roof_rear":     (-0.60, [(-hw,  0.00), ( hw,  0.00), ( hw,  1.30), (-hw,  1.30)]),
        "roof_front":    ( 0.80, [(-hw,  0.00), ( hw,  0.00), ( hw,  1.30), (-hw,  1.30)]),
        "wind_base":     ( 1.40, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.72), (-hw,  0.72)]),
        "hood_rear":     ( 1.60, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.70), (-hw,  0.70)]),
        "hood_front":    ( 2.40, [(-hw,  0.00), ( hw,  0.00), ( hw,  0.68), (-hw,  0.68)]),
        "front_bumper":  ( 2.60, [(-0.85, 0.00), (0.85, 0.00), (0.85, 0.50), (-0.85, 0.50)]),
        "front_tip":     ( 2.80, [(-0.60, 0.10), (0.60, 0.10), (0.60, 0.40), (-0.60, 0.40)]),
    }

    order = [
        "rear_bumper", "rear_body", "trunk_top", "rear_win_base",
        "roof_rear", "roof_front", "wind_base", "hood_rear",
        "hood_front", "front_bumper", "front_tip",
    ]

    all_verts = []
    section_indices = {}
    for name in order:
        y_val, ring = sections[name]
        start = len(all_verts)
        for x, z in ring:
            all_verts.append((x, y_val, z))
        section_indices[name] = (start, len(ring))

    verts = np.array(all_verts)
    faces = []

    for idx in range(len(order) - 1):
        s1_start, s1_n = section_indices[order[idx]]
        s2_start, s2_n = section_indices[order[idx + 1]]
        n_pts = min(s1_n, s2_n)
        for j in range(n_pts):
            j_next = (j + 1) % n_pts
            a = s1_start + j
            b = s1_start + j_next
            c = s2_start + j_next
            d = s2_start + j
            faces.extend(_quad(a, b, c, d))

    first_start, first_n = section_indices[order[0]]
    for j in range(1, first_n - 1):
        faces.append((first_start, first_start + j, first_start + j + 1))
    last_start, last_n = section_indices[order[-1]]
    for j in range(1, last_n - 1):
        faces.append((last_start, last_start + j, last_start + j + 1))

    i_idx = [f[0] for f in faces]
    j_idx = [f[1] for f in faces]
    k_idx = [f[2] for f in faces]

    body_mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i_idx, j=j_idx, k=k_idx,
        color="#2a3a5c",
        opacity=0.6,
        hoverinfo="skip",
        showlegend=False,
        flatshading=True,
    )

    edge_pairs = []
    for idx in range(len(order)):
        s_start, s_n = section_indices[order[idx]]
        for j in range(s_n):
            j_next = (j + 1) % s_n
            edge_pairs.append((s_start + j, s_start + j_next))
    for idx in range(len(order) - 1):
        s1_start, s1_n = section_indices[order[idx]]
        s2_start, s2_n = section_indices[order[idx + 1]]
        n_pts = min(s1_n, s2_n)
        for j in range(n_pts):
            edge_pairs.append((s1_start + j, s2_start + j))

    xs, ys, zs = [], [], []
    for a, b in edge_pairs:
        xs.extend([verts[a, 0], verts[b, 0], None])
        ys.extend([verts[a, 1], verts[b, 1], None])
        zs.extend([verts[a, 2], verts[b, 2], None])

    wireframe = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(180,200,255,0.35)", width=1.5),
        hoverinfo="text",
        hovertext="Ego Vehicle",
        showlegend=False,
    )

    ax, ay, az = [], [], []
    arrow_z = 1.35
    arrow = [(0, 3.4), (-0.5, 2.7), (0, 2.9), (0.5, 2.7), (0, 3.4)]
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

    return [body_mesh, wireframe, direction]


def build_3d_figure(
    points: np.ndarray,
    gt_boxes: list[dict] | None = None,
    tracker_boxes: list[dict] | None = None,
    x_range: float = 80.0,
    y_range: float = 80.0,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(build_point_cloud_trace(points))
    fig.add_traces(build_ego_car())

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
        autosize=True,
        uirevision="constant",
    )

    return fig
