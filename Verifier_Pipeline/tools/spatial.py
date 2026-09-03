"""
Pure-Python geometry tools. No model calls, no GPU, deterministic.
Box format: [xmin, ymin, xmax, ymax].
"""


def _area(box):
    xmin, ymin, xmax, ymax = box
    return max(0, xmax - xmin) * max(0, ymax - ymin)


def _center(box):
    xmin, ymin, xmax, ymax = box
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


def _iou(box1, box2):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    inter = max(0, xmax - xmin) * max(0, ymax - ymin)
    union = _area(box1) + _area(box2) - inter
    return inter / union if union > 0 else 0.0


def check_spatial_relation(box1: list, box2: list, relation: str) -> bool:
    """
    relation in: overlapping, inside, left, right, above, below, near
    NOTE: 'in_front_of' and 'behind' are intentionally NOT handled here --
    see estimate_depth_order(). 2D box geometry cannot reliably resolve
    depth ordering; routing those to vlm_probe or the (unvalidated)
    depth heuristic is a deliberate choice, not an oversight.
    """
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    c1x, c1y = _center(box1)
    c2x, c2y = _center(box2)

    if relation == "overlapping":
        overlap_x = (x1min < x2max) and (x1max > x2min)
        overlap_y = (y1min < y2max) and (y1max > y2min)
        return overlap_x and overlap_y

    if relation == "inside":
        return x1min >= x2min and y1min >= y2min and x1max <= x2max and y1max <= y2max

    if relation == "left":
        return c1x < c2x

    if relation == "right":
        return c1x > c2x

    if relation == "above":
        return c1y < c2y

    if relation == "below":
        return c1y > c2y

    if relation == "near":
        # heuristic: centers within 1.5x the average box diagonal
        import math
        diag1 = math.hypot(x1max - x1min, y1max - y1min)
        diag2 = math.hypot(x2max - x2min, y2max - y2min)
        dist = math.hypot(c1x - c2x, c1y - c2y)
        return dist < 1.5 * ((diag1 + diag2) / 2)

    raise ValueError(
        f"Unsupported relation '{relation}'. Use vlm_probe for semantic "
        f"relations (wearing, holding, riding, etc.) or estimate_depth_order "
        f"for in_front_of/behind."
    )


def compare_size(box1: list, box2: list) -> str:
    """Returns 'box1' or 'box2' -- whichever has larger area. Pure math, no model call."""
    return "box1" if _area(box1) >= _area(box2) else "box2"


def resolve_left_right(boxes: list, reference_box: list = None) -> list:
    """
    Sorts candidate boxes left-to-right by x-center.
    If reference_box is given, sorts by x-position relative to it instead
    of absolute image left/right (use for "the X to the left of Y" claims).
    Returns boxes sorted ascending by x-center (leftmost first).
    """
    if reference_box is not None:
        ref_x, _ = _center(reference_box)
        return sorted(boxes, key=lambda b: _center(b)[0] - ref_x)
    return sorted(boxes, key=lambda b: _center(b)[0])


def estimate_depth_order(box1: list, box2: list) -> str:
    """
    UNVALIDATED HEURISTIC. Returns 'box1_in_front' or 'box2_in_front'.
    Proxy: larger box area + lower position (higher y) in frame = closer to camera.
    This is a rough guess, not ground truth -- has not been checked against
    real in_front_of/behind cases yet. Treat low-confidence; prefer vlm_probe
    for in_front_of/behind claims until this is validated on a real batch.
    """
    _, y1min, _, y1max = box1
    _, y2min, _, y2max = box2
    score1 = _area(box1) + (y1max - y1min) * 0.1 + y1max * 0.01
    score2 = _area(box2) + (y2max - y2min) * 0.1 + y2max * 0.01
    return "box1_in_front" if score1 >= score2 else "box2_in_front"