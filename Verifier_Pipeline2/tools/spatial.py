"""
tools/spatial.py - Pure Python bounding box geometry and spatial relations.
"""
import math
from typing import List, Union


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes IoU between two [xmin, ymin, xmax, ymax] boxes."""
    inter_xmin = max(box_a[0], box_b[0])
    inter_ymin = max(box_a[1], box_b[1])
    inter_xmax = min(box_a[2], box_b[2])
    inter_ymax = min(box_a[3], box_b[3])

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def deduplicate_boxes(boxes: List[List[float]], iou_threshold: float = 0.6) -> List[List[float]]:
    """Suppresses duplicate boxes based on IoU threshold, preserving higher-confidence order."""
    kept = []
    for box in boxes:
        if not any(box_iou(box, prev) >= iou_threshold for prev in kept):
            kept.append(box)
    return kept


def box_distance(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Euclidean distance between box boundaries.
    If boxes overlap, returns negative IoU.
    """
    iou = box_iou(box_a, box_b)
    if iou > 0:
        return -iou

    # Distance between closest edges
    dx = max(0.0, max(box_a[0] - box_b[2], box_b[0] - box_a[2]))
    dy = max(0.0, max(box_a[1] - box_b[3], box_b[1] - box_a[3]))
    return math.sqrt(dx * dx + dy * dy)


def center_distance(box_a: List[float], box_b: List[float]) -> float:
    """Euclidean distance between center points of two boxes."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
