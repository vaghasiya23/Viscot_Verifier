"""
Single source of truth for what tools exist in the sandbox.
Import this dict and pass it as the local_scope for exec() when running
generated traces -- keeps the tool surface consistent between trace
generation, filtering, and eventual deployment. Don't hand-assemble the
scope dict separately in each script; import it from here.
"""
from .detection import detect_and_crop
from .spatial import check_spatial_relation, compare_size, resolve_left_right, estimate_depth_order
from .probe import vlm_probe, MOCK_MODE as VLM_PROBE_MOCK_MODE
from .ocr import read_text_ocr
from .color import get_color

TOOL_REGISTRY = {
    "detect_and_crop": detect_and_crop,
    "check_spatial_relation": check_spatial_relation,
    "compare_size": compare_size,
    "resolve_left_right": resolve_left_right,
    "estimate_depth_order": estimate_depth_order,
    "vlm_probe": vlm_probe,
    "read_text_ocr": read_text_ocr,
    "get_color": get_color,
}

if VLM_PROBE_MOCK_MODE:
    import warnings
    warnings.warn(
        "[tools] vlm_probe is in MOCK_MODE -- traces generated now cannot "
        "be trusted as golden data for any claim that relies on vlm_probe.",
        stacklevel=2,
    )
else:
    print("[tools] All tools loaded with REAL backends. Traces can be trusted.")