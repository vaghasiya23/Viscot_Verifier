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
    Evaluates 2D geometric relationship between box1 and box2.
    Accepts standard relations and common English aliases:
    - "inside", "in"
    - "overlapping", "on"
    - "left", "right", "above", "below", "near"
    """
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    c1x, c1y = _center(box1)
    c2x, c2y = _center(box2)

    r = relation.strip().lower().replace("_", " ")

    if r in ("inside", "in"):
        is_strictly_inside = (x1min >= x2min and y1min >= y2min and x1max <= x2max and y1max <= y2max)
        return is_strictly_inside or (_iou(box1, box2) > 0.25)

    if r in ("overlapping", "on"):
        overlap_x = (x1min < x2max) and (x1max > x2min)
        overlap_y = (y1min < y2max) and (y1max > y2min)
        return overlap_x and overlap_y

    if "left" in r:
        return c1x < c2x

    if "right" in r:
        return c1x > c2x

    if any(w in r for w in ["above", "over", "top"]):
        return c1y < c2y

    if any(w in r for w in ["below", "under", "beneath", "bottom"]):
        return c1y > c2y

    if any(w in r for w in ["near", "next to", "beside", "adjacent"]):
        # Distance between centers is within 1.5x of the larger dimension
        dx = abs(c1x - c2x)
        dy = abs(c1y - c2y)
        max_dim = max(x2max - x2min, y2max - y2min, x1max - x1min, y1max - y1min)
        return (dx**2 + dy**2)**0.5 < (max_dim * 1.5)

    return False


def compare_size(box1: list, box2: list) -> str:
    """Returns 'box1_larger', 'box2_larger', or 'similar'."""
    a1, a2 = _area(box1), _area(box2)
    if a1 > 1.3 * a2:
        return "box1_larger"
    if a2 > 1.3 * a1:
        return "box2_larger"
    return "similar"


def resolve_left_right(box1: list, box2: list) -> str:
    """Returns 'box1_left' or 'box2_left'."""
    c1x, _ = _center(box1)
    c2x, _ = _center(box2)
    return "box1_left" if c1x < c2x else "box2_left"


def estimate_depth_order(box1: list, box2: list) -> str:
    """
    Ground-plane heuristic: an object whose bottom edge (ymax) is lower
    in the 2D image plane is typically closer to the camera.
    """
    y1_bottom = box1[3]
    y2_bottom = box2[3]
    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    threshold = 0.05 * max(h1, h2)

    if y1_bottom - y2_bottom > threshold:
        return "box1_in_front"
    elif y2_bottom - y1_bottom > threshold:
        return "box2_in_front"
    else:
        return "ambiguous"
