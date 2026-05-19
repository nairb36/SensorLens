import numpy as np
import motmetrics as mm


ERROR_COLORS = {
    "match": "#2ecc71",
    "switch": "#ff4444",
    "fp": "#f1c40f",
    "miss": "#3498db",
}

SWITCH_LINE_WIDTH = 6


def _build_id_map(gt_data, tracker_data, allowed_categories=None):
    """Build a mapping from string instance_tokens to integer IDs.

    motmetrics requires numeric IDs internally.
    Returns (token_to_int, int_to_token) dicts for GT IDs only.
    Tracker IDs are already integers.
    """
    token_to_int = {}
    int_to_token = {}
    counter = 0
    num_frames = min(len(gt_data), len(tracker_data))
    for i in range(num_frames):
        for det in gt_data[i].get("detections", []):
            if allowed_categories and det["category_name"] not in allowed_categories:
                continue
            tok = det["instance_token"]
            if tok not in token_to_int:
                token_to_int[tok] = counter
                int_to_token[counter] = tok
                counter += 1
    return token_to_int, int_to_token


def preprocess_frame(gt_frame, tracker_frame, token_to_int, allowed_categories=None):
    """Extract object IDs and 2D positions from a GT/tracker frame pair."""
    gt_ids = []
    gt_positions = []
    if gt_frame:
        for det in gt_frame.get("detections", []):
            if allowed_categories and det["category_name"] not in allowed_categories:
                continue
            tok = det["instance_token"]
            if tok not in token_to_int:
                continue
            gt_ids.append(token_to_int[tok])
            gt_positions.append(det["translation"][:2])

    trk_ids = []
    trk_positions = []
    if tracker_frame:
        for trk in tracker_frame.get("tracks", []):
            if allowed_categories and trk["category_name"] not in allowed_categories:
                continue
            trk_ids.append(trk["id"])
            trk_positions.append(trk["translation"][:2])

    gt_pos = np.array(gt_positions, dtype=float) if gt_positions else np.empty((0, 2))
    trk_pos = np.array(trk_positions, dtype=float) if trk_positions else np.empty((0, 2))

    return gt_ids, gt_pos, trk_ids, trk_pos


def run_evaluation(gt_data, tracker_data, max_dist=2.0, allowed_categories=None):
    """Run CLEAR MOT evaluation across all frames.

    Args:
        gt_data: list of GT frame dicts
        tracker_data: list of tracker frame dicts
        max_dist: matching distance threshold in meters
        allowed_categories: set of nuScenes category names to evaluate on,
                            or None for all categories

    Returns (accumulator, int_to_token) tuple.
    """
    token_to_int, int_to_token = _build_id_map(
        gt_data, tracker_data, allowed_categories
    )
    acc = mm.MOTAccumulator(auto_id=True)
    num_frames = min(len(gt_data), len(tracker_data))

    for i in range(num_frames):
        gt_ids, gt_pos, trk_ids, trk_pos = preprocess_frame(
            gt_data[i], tracker_data[i], token_to_int, allowed_categories
        )
        # Compute squared distances for threshold check, then sqrt for correct MOTP
        dists = mm.distances.norm2squared_matrix(
            gt_pos, trk_pos, max_d2=max_dist ** 2
        )
        dists = np.sqrt(dists)
        acc.update(gt_ids, trk_ids, dists)

    return acc, int_to_token


def compute_summary(acc):
    """Compute summary MOT metrics. Returns dict of metric_name -> value."""
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=mm.metrics.motchallenge_metrics, name="overall"
    )
    return summary.iloc[0].to_dict()


def get_frame_events(acc, frame_idx, int_to_token):
    """Get structured events for a specific frame.

    Returns dict with keys: matches, switches, false_positives, misses.
    GT IDs are mapped back to original instance_token strings.
    """
    events = acc.events
    result = {"matches": [], "switches": [], "false_positives": [], "misses": []}

    if frame_idx not in events.index.get_level_values(0):
        return result

    frame_ev = events.loc[frame_idx]

    for _, row in frame_ev.iterrows():
        ev_type = row["Type"]
        gt_id = int_to_token.get(row["OId"], row["OId"])
        trk_id = row["HId"]
        if ev_type == "MATCH":
            result["matches"].append(
                {"gt_id": gt_id, "trk_id": trk_id, "dist": row["D"]}
            )
        elif ev_type == "SWITCH":
            result["switches"].append(
                {"gt_id": gt_id, "trk_id": trk_id, "dist": row["D"]}
            )
        elif ev_type == "FP":
            result["false_positives"].append({"trk_id": trk_id})
        elif ev_type == "MISS":
            result["misses"].append({"gt_id": gt_id})

    return result


def get_box_error_types(acc, frame_idx, int_to_token):
    """Get error type for each GT and tracker box at a specific frame.

    Returns:
        gt_errors: dict mapping instance_token -> error_type
        trk_errors: dict mapping tracker_id -> error_type
    """
    events = get_frame_events(acc, frame_idx, int_to_token)

    gt_errors = {}
    trk_errors = {}

    for m in events["matches"]:
        gt_errors[m["gt_id"]] = "match"
        trk_errors[m["trk_id"]] = "match"

    for s in events["switches"]:
        gt_errors[s["gt_id"]] = "switch"
        trk_errors[s["trk_id"]] = "switch"

    for fp in events["false_positives"]:
        trk_errors[fp["trk_id"]] = "fp"

    for miss in events["misses"]:
        gt_errors[miss["gt_id"]] = "miss"

    return gt_errors, trk_errors


METRIC_DISPLAY = [
    ("mota", "MOTA", "pct"),
    ("motp", "MOTP", "dist"),
    ("idf1", "IDF1", "pct"),
    ("recall", "Recall", "pct"),
    ("precision", "Precision", "pct"),
    ("num_switches", "ID Switches", "int"),
    ("num_fragmentations", "Fragmentations", "int"),
    ("num_false_positives", "False Positives", "int"),
    ("num_misses", "Missed Detections", "int"),
    ("mostly_tracked", "Mostly Tracked", "int"),
    ("partially_tracked", "Partially Tracked", "int"),
    ("mostly_lost", "Mostly Lost", "int"),
]


def format_metric(value, fmt):
    """Format a metric value for display."""
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "dist":
        return f"{value:.3f}"
    return str(int(value))
