"""
skeleton_gen.py

Turns a GQA-style `reasoning` program (select/relate/query/filter ops with
dependency indices) into:
  1. A deterministic SCAFFOLD: which entities must be detected, which
     relations must be checked, in what order, with variable bindings
     (no LLM involved -- this part can't drift or hallucinate).
  2. A PROMPT for the coder LLM that hands it the scaffold and asks it to
     write real claim sentences + tool-verification code grounded in that
     scaffold -- not to invent its own independent reasoning path.

This does NOT call any LLM itself. It only prepares deterministic
structure + prompt text. trace_gen.py (not written yet) does the actual
LLM call.
"""
import re

# --- predicate classification: spatial (pure geometry) vs semantic (needs vlm_probe) ---
# Extend this table as new predicates show up in real data. Anything not
# listed defaults to "semantic" (safer: routes to vlm_probe rather than
# silently mis-applying box math).
SPATIAL_PREDICATES = {
    "on", "in", "inside", "above", "below", "under", "beneath",
    "left", "right", "near", "overlapping",
}
DEPTH_PREDICATES = {"in_front_of", "behind", "in front of"}
# everything else (wearing, holding, riding, eating, feeding, carrying,
# watching, pulling, sitting, standing on, etc.) -> semantic / vlm_probe


def classify_predicate(pred: str) -> str:
    p = pred.strip().lower().replace(" ", "_")
    if p in SPATIAL_PREDICATES:
        return "spatial"
    if p in DEPTH_PREDICATES:
        return "depth"
    return "semantic"


def _parse_argument(argument: str):
    """
    GQA reasoning argument formats seen so far:
      select:  "children (827627)"          -> label="children", id=827627
      relate:  "_,on,o (827624)"             -> no new label, pred="on", role="o", id=827624
      relate:  "person,on,s (827627)"        -> new label="person", pred="on", role="s", id=827627
      query:   "name"                        -> no id
      filter:  "not (little) (827630)"  (approx; format varies -- treat leniently)
    Returns dict with whatever fields are parseable. Unparseable/unexpected
    formats get flagged with "_raw" and "_needs_review": True so they
    surface during the calibration pass rather than failing silently.
    """
    m = re.match(r"^(.*?)(?:\s*\((\d+)\))?$", argument.strip())
    body, entity_id = (m.group(1).strip(), m.group(2)) if m else (argument.strip(), None)

    if "," in body:
        parts = [x.strip() for x in body.split(",")]
        if len(parts) == 3:
            subj, pred, role = parts
            return {
                "kind": "relate",
                "new_label": None if subj == "_" else subj,
                "predicate": pred,
                "role": role,  # 's' = new entity is subject, 'o' = new entity is object
                "entity_id": entity_id,
            }
        return {"kind": "unknown", "_raw": argument, "_needs_review": True}

    if body in ("name", "color", "material", "shape", "type"):
        return {"kind": "query", "attribute": body}

    # plain "label" or "label (id)" -> select
    return {"kind": "select", "label": body, "entity_id": entity_id}


def build_scaffold(reasoning: list) -> list:
    """
    Deterministic pass: reasoning[] -> list of scaffold steps.
    Each scaffold step is a dict describing what tool call(s) this step
    needs and which prior-step variable(s) it depends on. No LLM.
    """
    scaffold = []
    for i, step in enumerate(reasoning):
        op = step["operation"]
        deps = step.get("dependencies", [])
        parsed = _parse_argument(step["argument"])
        var_name = f"e{i}"

        entry = {
            "step_index": i,
            "operation": op,
            "var_name": var_name,
            "dependencies": deps,
            "parsed": parsed,
            "needs_review": parsed.get("_needs_review", False),
        }

        if op == "select":
            entry["tool_call"] = f'{var_name} = detect_and_crop(img_path, "{parsed.get("label", parsed.get("_raw",""))}")'
            entry["claim_template"] = f'There is a {parsed.get("label","<?>")}.'

        elif op == "relate":
            pred = parsed.get("predicate", "<?>")
            family = classify_predicate(pred)
            dep_var = f"e{deps[0]}" if deps else "e?"
            new_label = parsed.get("new_label")

            if new_label:
                # relate introduces a NEW entity to detect, then check relation
                entry["tool_call"] = (
                    f'{var_name}_candidates = detect_and_crop(img_path, "{new_label}")\n'
                    f'# TODO(coder): for each candidate, check relation "{pred}" '
                    f'against {dep_var} using '
                    f'{"check_spatial_relation" if family=="spatial" else ("estimate_depth_order" if family=="depth" else "vlm_probe")}'
                )
                entry["claim_template"] = f'The {new_label} is {pred} the entity from step {deps}.'
            else:
                entry["tool_call"] = (
                    f'# TODO(coder): verify relation "{pred}" for {dep_var} using '
                    f'{"check_spatial_relation" if family=="spatial" else ("estimate_depth_order" if family=="depth" else "vlm_probe")}'
                )
                entry["claim_template"] = f'The entity from step {deps} is {pred} something.'

            entry["predicate_family"] = family

        elif op == "query":
            attr = parsed.get("attribute", "name")
            dep_var = f"e{deps[0]}" if deps else "e?"
            if attr == "name":
                entry["tool_call"] = f'# final answer = the object_query string used to detect {dep_var}'
            else:
                entry["tool_call"] = f'{var_name} = get_color(img_path, {dep_var}[0])  # if attr == "color", else vlm_probe'
            entry["claim_template"] = f'The {attr} of the entity from step {deps} answers the question.'

        elif op == "filter":
            entry["tool_call"] = f'# TODO(coder): re-query detect_and_crop with attribute folded into the label, or vlm_probe for state filters (old/open/little)'
            entry["claim_template"] = f'The entity from step {deps} is filtered by: {parsed.get("_raw", step["argument"])}.'

        else:
            entry["tool_call"] = f'# UNHANDLED OP "{op}" -- flag for review'
            entry["claim_template"] = f'(unhandled operation: {op})'
            entry["needs_review"] = True

        scaffold.append(entry)
    return scaffold


def build_generation_prompt(sample: dict, scaffold: list) -> str:
    """
    Builds the user-turn content handed to the coder LLM. The scaffold is
    presented as ground-truth structure the LLM must follow -- it should
    NOT invent different entities/relations, only translate the scaffold
    into real claim sentences (grounded in sample['thought'] for phrasing)
    and working tool-verification code.
    """
    scaffold_lines = []
    for s in scaffold:
        review_flag = "  [NEEDS REVIEW]" if s["needs_review"] else ""
        scaffold_lines.append(
            f'Step {s["step_index"]} ({s["operation"]}, deps={s["dependencies"]}):{review_flag}\n'
            f'  claim: {s["claim_template"]}\n'
            f'  tool_call_skeleton: {s["tool_call"]}'
        )
    scaffold_text = "\n".join(scaffold_lines)

    return f"""Image: {sample['image']}
Question: {sample['question']}
Ground-truth answer: {sample.get('answer', '<unknown>')}
Original CoT (for phrasing reference only, not the source of claims): {sample['thought']}

DETERMINISTIC SCAFFOLD (derived from the dataset's ground-truth reasoning
program -- these are the exact entities and relations that must be
verified, in this order. Do not add claims not listed here. Do not skip
any. Fill in each tool_call_skeleton with real, executable code):

{scaffold_text}

Write <thought>, <extract_claims> (one claim per scaffold step, phrased
naturally), <tool_verification> (```python ... ``` implementing every
tool_call_skeleton with real code, using stateful variables e0, e1, ...
as shown), and <final_verdict> (VALID/INVALID + which claim failed if any,
compare derived answer against the ground-truth answer above).
"""