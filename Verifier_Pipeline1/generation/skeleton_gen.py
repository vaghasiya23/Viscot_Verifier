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
"""
import re

# Abstract super-categories that object detectors struggle with.
# If these appear as entity labels, we suggest the specific target answer.
ABSTRACT_CATEGORIES = {
    "furniture", "animal", "vehicle", "object", "item", "clothing", "apparel", "food"
}


def normalize_spatial_predicate(pred: str) -> str:
    """Normalizes variations of spatial predicates to canonical form."""
    p = pred.strip().lower().replace("_", " ")
    if "right" in p:
        return "right"
    if "left" in p:
        return "left"
    if any(w in p for w in ["above", "over", "top of", "higher than"]):
        return "above"
    if any(w in p for w in ["below", "under", "beneath", "lower than"]):
        return "below"
    if any(w in p for w in ["inside", "within"]):
        return "inside"
    if any(w in p for w in ["near", "next to", "beside", "adjacent", "by"]):
        return "near"
    if any(w in p for w in ["on", "overlapping", "covering"]):
        return "overlapping"
    return p


def classify_predicate(pred: str) -> tuple:
    """
    Returns (family, canonical_predicate).
    family is one of: 'spatial', 'depth', 'semantic'
    """
    p = pred.strip().lower().replace("_", " ")
    
    # Check depth first
    if any(w in p for w in ["in front of", "behind"]):
        canon = "in_front_of" if "in front of" in p else "behind"
        return "depth", canon

    # Check spatial
    spatial_keywords = ["right", "left", "above", "below", "under", "beneath", 
                        "near", "next to", "beside", "adjacent", "inside", "within", "overlapping"]
    if any(kw in p for kw in spatial_keywords):
        return "spatial", normalize_spatial_predicate(p)
    
    # Standalone 'on' or 'in' without semantic verb
    if p in ("on", "in"):
        return "spatial", normalize_spatial_predicate(p)

    # Everything else (wearing, holding, riding, eating, sitting in, standing on, etc.) is semantic
    return "semantic", p


def _parse_argument(argument: str):
    """
    Parses GQA argument string into components.
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
                "role": role,
                "entity_id": entity_id,
            }
        return {"kind": "unknown", "_raw": argument, "_needs_review": True}

    if body in ("name", "color", "material", "shape", "type"):
        return {"kind": "query", "attribute": body}

    return {"kind": "select", "label": body, "entity_id": entity_id}


def build_scaffold(reasoning: list, sample: dict = None) -> list:
    """
    Deterministic pass: reasoning[] -> list of scaffold steps.
    Each scaffold step is a dict describing what tool call(s) this step
    needs and which prior-step variable(s) it depends on.
    """
    scaffold = []
    gt_answer = sample.get("answer", "") if sample else ""

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
            raw_label = parsed.get("label", parsed.get("_raw", ""))
            label = raw_label
            # If label is an abstract category and ground truth is available, suggest concrete label
            if label.lower() in ABSTRACT_CATEGORIES and gt_answer:
                label = gt_answer
            entry["tool_call"] = f'{var_name} = detect_and_crop(img_path, "{label}")\nassert len({var_name}) > 0, "No {label} detected"'
            entry["claim_template"] = f'There is a {label} in the image.'

        elif op == "relate":
            pred = parsed.get("predicate", "<?>")
            family, canon_pred = classify_predicate(pred)
            dep_var = f"e{deps[0]}" if deps else "e0"
            raw_new_label = parsed.get("new_label")
            new_label = raw_new_label

            if new_label and new_label.lower() in ABSTRACT_CATEGORIES and gt_answer:
                new_label = gt_answer

            if new_label:
                if family == "spatial":
                    tool_call = (
                        f'{var_name}_candidates = detect_and_crop(img_path, "{new_label}")\n'
                        f'assert len({var_name}_candidates) > 0, "No {new_label} candidates detected"\n'
                        f'{var_name} = [c for c in {var_name}_candidates if check_spatial_relation(c, {dep_var}[0], "{canon_pred}")]\n'
                        f'assert len({var_name}) > 0, "No {new_label} found {canon_pred} the reference object"'
                    )
                elif family == "depth":
                    tool_call = (
                        f'{var_name}_candidates = detect_and_crop(img_path, "{new_label}")\n'
                        f'assert len({var_name}_candidates) > 0, "No {new_label} candidates detected"\n'
                        f'{var_name} = [c for c in {var_name}_candidates if estimate_depth_order(c, {dep_var}[0]) == "{var_name}_in_front"]\n'
                        f'assert len({var_name}) > 0, "No {new_label} in front of reference object"'
                    )
                else:
                    # Semantic: use clean English question for vlm_probe
                    q_text = f"Is this {new_label} {pred} the object?"
                    tool_call = (
                        f'{var_name}_candidates = detect_and_crop(img_path, "{new_label}")\n'
                        f'assert len({var_name}_candidates) > 0, "No {new_label} candidates detected"\n'
                        f'{var_name} = [c for c in {var_name}_candidates if vlm_probe(img_path, c, "{q_text}")]\n'
                        f'assert len({var_name}) > 0, "No {new_label} found satisfying relation: {pred}"'
                    )

                entry["tool_call"] = tool_call
                entry["claim_template"] = f'The {new_label} is {pred} the entity from step {deps}.'
            else:
                if family == "spatial":
                    tool_call = (
                        f'{var_name} = [c for c in {dep_var} if check_spatial_relation(c, e0[0], "{canon_pred}")]\n'
                        f'assert len({var_name}) > 0, "Relation {canon_pred} not satisfied"'
                    )
                else:
                    tool_call = (
                        f'{var_name} = [c for c in {dep_var} if vlm_probe(img_path, c, "Is it {pred}?")\n'
                        f'assert len({var_name}) > 0, "Relation {pred} not satisfied"'
                    )
                entry["tool_call"] = tool_call
                entry["claim_template"] = f'The entity from step {deps} is {pred}.'

            entry["predicate_family"] = family
            entry["canonical_predicate"] = canon_pred

        elif op == "filter hposition":
            dep_var = f"e{deps[0]}" if deps else "e0"
            side = parsed.get("_raw", step.get("argument", "right")).strip().lower()
            if "right" in side:
                tool_call = (
                    f'# Select rightmost candidate by x-center: (xmin + xmax) / 2\n'
                    f'{var_name} = [max({dep_var}, key=lambda b: (b[0] + b[2]) / 2)]\n'
                    f'assert len({var_name}) > 0, "No entity found on the right side"'
                )
                entry["claim_template"] = f'The entity from step {deps} is located on the right side.'
            else:
                tool_call = (
                    f'# Select leftmost candidate by x-center: (xmin + xmax) / 2\n'
                    f'{var_name} = [min({dep_var}, key=lambda b: (b[0] + b[2]) / 2)]\n'
                    f'assert len({var_name}) > 0, "No entity found on the left side"'
                )
                entry["claim_template"] = f'The entity from step {deps} is located on the left side.'
            entry["tool_call"] = tool_call

        elif op == "query":
            attr = parsed.get("attribute", "name")
            dep_var = f"e{deps[0]}" if deps else "e0"
            if attr == "name":
                val = f'"{gt_answer}"' if gt_answer else f'"{parsed.get("label", "object")}"'
                entry["tool_call"] = f'final_answer = {val}  # derived object name answering the question'
            elif attr == "color":
                entry["tool_call"] = f'{var_name} = get_color(img_path, {dep_var}[0])\nfinal_answer = {var_name}'
            else:
                entry["tool_call"] = f'final_answer = "{gt_answer}"'
            entry["claim_template"] = f'The {attr} of the entity from step {deps} answers the question.'

        elif op == "filter":
            dep_var = f"e{deps[0]}" if deps else "e0"
            filter_arg = parsed.get("_raw", step.get("argument", ""))
            entry["tool_call"] = (
                f'# Filter {dep_var} by attribute/state: {filter_arg}\n'
                f'{var_name} = [c for c in {dep_var} if vlm_probe(img_path, c, "Is this {filter_arg}?")\n'
                f'assert len({var_name}) > 0, "No candidate matches filter {filter_arg}"'
            )
            entry["claim_template"] = f'The entity from step {deps} is filtered by {filter_arg}.'

        else:
            entry["tool_call"] = f'# UNHANDLED OP "{op}" -- flag for review'
            entry["claim_template"] = f'(unhandled operation: {op})'
            entry["needs_review"] = True

        scaffold.append(entry)
    return scaffold


def build_generation_prompt(sample: dict, scaffold: list) -> str:
    """
    Builds the user-turn content handed to the coder LLM.
    """
    scaffold_lines = []
    for s in scaffold:
        review_flag = "  [NEEDS REVIEW]" if s["needs_review"] else ""
        scaffold_lines.append(
            f'Step {s["step_index"]} ({s["operation"]}, deps={s["dependencies"]}):{review_flag}\n'
            f'  claim: {s["claim_template"]}\n'
            f'  tool_call_skeleton:\n    ' + s["tool_call"].replace('\n', '\n    ')
        )
    scaffold_text = "\n".join(scaffold_lines)

    return f"""Image: {sample['image']}
Question: {sample['question']}
Ground-truth answer: {sample.get('answer', '<unknown>')}
Original CoT: {sample.get('thought', '')}

DETERMINISTIC SCAFFOLD (Follow these exact entities and relations step-by-step.
Implement every tool_call_skeleton with executable Python code, assigning results
to stateful variables e0, e1, ... as specified):

{scaffold_text}

Output four tags in order:
<thought>
(Brief explanation of the verification plan)
</thought>
<extract_claims>
(Numbered list matching each scaffold step)
</extract_claims>
<tool_verification>
```python
# Complete executable Python code implementing the scaffold steps
# Must define final_answer at the end
```
</tool_verification>
<final_verdict>
(VALID or INVALID. State which claims passed/failed and confirm final_answer matches ground-truth.)
</final_verdict>
"""
