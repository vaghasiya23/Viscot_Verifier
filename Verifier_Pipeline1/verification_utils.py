"""Pure helpers shared by vision tools and regression tests."""
from filters.outcome_filter import are_answers_compatible


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    union = area_a+area_b-intersection
    return intersection/union if union else 0


def deduplicate_boxes(boxes, iou_threshold=0.6):
    """Input is confidence-ordered; suppress duplicates, preserve distinct objects."""
    kept = []
    for box in boxes:
        if not any(box_iou(box, previous) >= iou_threshold for previous in kept):
            kept.append(box)
    return kept


def answer_consensus(answers):
    if not answers:
        raise AssertionError('Uncertain: no visual answers')
    if not all(are_answers_compatible(answers[0], answer) for answer in answers):
        raise AssertionError(f'Uncertain: conflicting visual answers {answers!r}')
    return answers[0]


def select_position(image_path, boxes, axis, side):
    """Filter by image half, not arbitrary first detection or singleton extremum."""
    from PIL import Image
    if axis not in ('horizontal', 'vertical') or side not in ('left', 'right', 'top', 'bottom'):
        raise ValueError('Invalid position filter')
    if (axis == 'horizontal') != (side in ('left', 'right')):
        raise ValueError('Position axis/side mismatch')
    with Image.open(image_path) as image:
        extent = image.width if axis == 'horizontal' else image.height
    offset = 0 if axis == 'horizontal' else 1
    positive = side in ('right', 'bottom')
    return [b for b in boxes if (((b[offset]+b[offset+2])/2 > extent/2) if positive
                                else ((b[offset]+b[offset+2])/2 < extent/2))]


def annotated_regions(image_path, boxes):
    """Validate dataset-supplied localization supervision; does not verify identity."""
    import math
    from PIL import Image
    with Image.open(image_path) as image:
        width, height = image.size
    if not boxes:
        raise ValueError('No annotated regions')
    checked = []
    for box in boxes:
        if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            raise ValueError('Invalid annotated box')
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Annotated box outside image')
        checked.append(list(box))
    return deduplicate_boxes(checked)
