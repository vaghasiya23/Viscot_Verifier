"""
tools/image_patch.py - Unified Object-Oriented ImagePatch interface.
Inspired by ViperGPT, powered by local vision models.
"""
from __future__ import annotations
from typing import List, Optional
from PIL import Image

from tools.spatial import box_iou, box_distance, center_distance
from tools.detector import detect
from tools.probe import verify_property, visual_query, verify_relation
from tools.ocr import read_text


class ImagePatch:
    """
    An image or cropped bounding box region with visual and spatial reasoning capabilities.
    """

    def __init__(self, image_path: str, box: Optional[List[float]] = None):
        self.image_path = image_path

        with Image.open(image_path) as img:
            w, h = img.size
        self.image_size = (float(w), float(h))

        if box is None:
            self.box = [0.0, 0.0, float(w), float(h)]
        else:
            self.box = [float(b) for b in box]

        self.xmin = self.box[0]
        self.ymin = self.box[1]
        self.xmax = self.box[2]
        self.ymax = self.box[3]

        # Alias coordinate names for compatibility and natural prompting
        self.left = self.xmin
        self.top = self.ymin
        self.right = self.xmax
        self.bottom = self.ymax

        self.width = max(0.0, self.xmax - self.xmin)
        self.height = max(0.0, self.ymax - self.ymin)
        self.area = self.width * self.height

        self.horizontal_center = (self.xmin + self.xmax) / 2.0
        self.vertical_center = (self.ymin + self.ymax) / 2.0

    @property
    def is_full_image(self) -> bool:
        """Returns True if this patch represents the entire image."""
        w, h = self.image_size
        return self.xmin <= 0 and self.ymin <= 0 and self.xmax >= w and self.ymax >= h

    # ─── Core Visual Tools ──────────────────────────────────────────────

    def find(self, object_name: str, threshold: float = 0.1) -> List[ImagePatch]:
        """
        Detects instances of `object_name` within this patch.
        Returns a list of ImagePatch objects cropped to the detections.
        """
        crop_box = None if self.is_full_image else self.box
        raw_boxes = detect(self.image_path, object_name, threshold=threshold, crop_box=crop_box)
        return [ImagePatch(self.image_path, b) for b in raw_boxes]

    def exists(self, object_name: str, threshold: float = 0.1) -> bool:
        """Returns True if object_name is found in this patch."""
        return len(self.find(object_name, threshold=threshold)) > 0

    def count(self, object_name: str, threshold: float = 0.1) -> int:
        """Returns the number of instances of object_name detected in this patch."""
        return len(self.find(object_name, threshold=threshold))

    def verify_property(self, object_name: str, property_name: str) -> bool:
        """
        Verifies whether the object in this patch has the given property/attribute.
        Example: patch.verify_property("chair", "blue") -> True/False
        """
        return verify_property(self.image_path, self.box, object_name, property_name)

    def query(self, question: str) -> str:
        """
        Asks a visual question about this patch.
        Example: patch.query("What color is this shirt?") -> "blue"
        """
        return visual_query(self.image_path, self.box, question)

    def read_text(self) -> str:
        """Extracts text visible inside this patch via OCR."""
        return read_text(self.image_path, self.box)

    def crop(self, xmin: float, ymin: float, xmax: float, ymax: float) -> ImagePatch:
        """Returns a new ImagePatch cropped to [xmin, ymin, xmax, ymax]."""
        return ImagePatch(self.image_path, [xmin, ymin, xmax, ymax])

    # ─── Spatial Relations (positional — pure geometry) ─────────────────

    def iou(self, other: ImagePatch) -> float:
        """Computes IoU overlap with another patch."""
        return box_iou(self.box, other.box)

    def overlaps(self, other: ImagePatch) -> bool:
        """Returns True if this patch overlaps with another."""
        return self.iou(other) > 0.0

    def distance(self, other: ImagePatch) -> float:
        """Computes edge-to-edge distance (or negative IoU if overlapping)."""
        return box_distance(self.box, other.box)

    def center_distance(self, other: ImagePatch) -> float:
        """Computes Euclidean distance between patch centers."""
        return center_distance(self.box, other.box)

    def is_to_right_of(self, other: ImagePatch) -> bool:
        """Checks if this patch is located to the right of another patch."""
        return self.horizontal_center > other.horizontal_center

    def is_to_left_of(self, other: ImagePatch) -> bool:
        """Checks if this patch is located to the left of another patch."""
        return self.horizontal_center < other.horizontal_center

    def is_above(self, other: ImagePatch) -> bool:
        """Checks if this patch is vertically above another patch (y=0 is top)."""
        return self.vertical_center < other.vertical_center

    def is_below(self, other: ImagePatch) -> bool:
        """Checks if this patch is vertically below another patch (y=0 is top)."""
        return self.vertical_center > other.vertical_center

    def is_larger_than(self, other: ImagePatch) -> bool:
        """Checks if this patch's bounding box area is larger than another's."""
        return self.area > other.area

    def is_inside(self, other: ImagePatch, threshold: float = 0.75) -> bool:
        """
        Checks if this patch is mostly contained within another patch.
        Returns True if >= threshold fraction of this patch's area
        is inside the other patch.
        """
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        if self.area == 0:
            return False
        return (inter_area / self.area) >= threshold

    def closest(self, patches: List[ImagePatch]) -> Optional[ImagePatch]:
        """
        Returns the patch from `patches` that is closest to this patch.
        Returns None if the list is empty.
        """
        if not patches:
            return None
        return min(patches, key=lambda p: self.center_distance(p))

    # ─── Semantic Relation Verification (VLM-backed) ───────────────────

    def verify_relation(self, predicate: str, other: 'ImagePatch',
                         self_label: str, other_label: str) -> bool:
        """
        Verifies a semantic relationship between this patch and another.
        Draws RED box around self, BLUE box around other on the FULL image,
        then asks VLM: "Is the <self_label> in the RED box <predicate> the <other_label> in the BLUE box?"

        Example:
            cup.verify_relation("hanging on", cabinet, "cup", "cabinet") -> True/False
        """
        return verify_relation(
            self.image_path, self.box, other.box,
            predicate, self_label, other_label
        )

    # ─── Segmentation (SAM-backed) ─────────────────────────────────────

    def segment(self):
        """
        Returns a precise binary segmentation mask (H x W bool numpy array)
        for the object in this patch, using SAM with the bounding box as prompt.
        """
        from tools.segmenter import segment_box
        return segment_box(self.image_path, self.box)

    def precise_area(self) -> int:
        """
        Returns the precise pixel area of this object using SAM segmentation.
        More accurate than self.area (which is just bbox width * height).
        """
        from tools.segmenter import mask_pixel_area
        mask = self.segment()
        return mask_pixel_area(mask)

    def precise_iou(self, other: 'ImagePatch') -> float:
        """
        Computes pixel-level IoU between two objects using their SAM masks.
        More accurate than bbox IoU for irregularly shaped objects.
        """
        from tools.segmenter import mask_iou
        mask_a = self.segment()
        mask_b = other.segment()
        return mask_iou(mask_a, mask_b)

    # ─── Depth Estimation (Depth-Anything-backed) ──────────────────────

    def depth(self) -> float:
        """
        Returns the mean relative depth of this patch (float in [0, 1]).
        0 = nearest to camera, 1 = farthest from camera.
        """
        from tools.depth import estimate_depth
        return estimate_depth(self.image_path, self.box)

    def is_behind(self, other: 'ImagePatch') -> bool:
        """Checks if this patch is farther from the camera than another (higher depth)."""
        return self.depth() > other.depth()

    def is_in_front_of(self, other: 'ImagePatch') -> bool:
        """Checks if this patch is closer to the camera than another (lower depth)."""
        return self.depth() < other.depth()

    # ─── Representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ImagePatch([{self.xmin:.1f}, {self.ymin:.1f}, {self.xmax:.1f}, {self.ymax:.1f}])"
