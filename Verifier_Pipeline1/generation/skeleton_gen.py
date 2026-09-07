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
    "furniture", "animal", "vehicle", "object", "item", "clothing", "apparel", "food", "fast food"
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
    if p == "in" or any(w in p for w in ["inside", "within"]):
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
    """Compile a conservative, auditable verification program.

    GQA role s means the NEW entity is the subject; o means it is the object.
    Unnamed relation entities are proposed visually, never from the target answer.
    """
    scaffold = []
    labels = {}
    # VisCoT GQA bboxs annotate the region relevant to the terminal query.
    # Use them as explicit SFT localization supervision, never as answer text.
    annotated_target = None
    regions = sample.get("bboxs", []) if sample else []
    if regions and reasoning and reasoning[-1]["operation"] == "query":
        last_deps = reasoning[-1].get("dependencies", [])
        cursor = last_deps[0] if len(last_deps) == 1 else None
        visited = set()
        while isinstance(cursor, int) and 0 <= cursor < len(reasoning)-1 and cursor not in visited:
            visited.add(cursor)
            node = reasoning[cursor]
            if node["operation"] in {"select", "relate"}:
                annotated_target = cursor
                break
            next_deps = node.get("dependencies", [])
            cursor = next_deps[0] if node["operation"].startswith("filter") and len(next_deps) == 1 else None
    for i, step in enumerate(reasoning):
        op, deps = step["operation"], step.get("dependencies", [])
        parsed = _parse_argument(step["argument"])
        var = f"e{i}"
        entry = dict(step_index=i, operation=op, var_name=var, dependencies=deps,
                     parsed=parsed, needs_review=False, tool_call="", claim_template="")
        def defer(reason):
            entry.update(needs_review=True, review_reason=reason,
                         tool_call=f"# REVIEW: {reason}", claim_template=reason)
        valid_dep = len(deps) == 1 and isinstance(deps[0], int) and 0 <= deps[0] < i
        dep = f"e{deps[0]}" if valid_dep else None
        if op == "select" and not deps and parsed.get("label"):
            label = parsed["label"]
            labels[i] = label
            source = f"annotated_regions(img_path, {regions!r})" if i == annotated_target else f"detect_and_crop(img_path, {label!r})"
            entry["tool_call"] = f'{var} = {source}\nassert len({var}) > 0, "Uncertain: no detection"'
            entry["claim_template"] = f"There is a {label} in the image."
        elif not valid_dep:
            defer("Unsupported or malformed dependencies")
        elif op == "relate":
            label, role = parsed.get("new_label"), parsed.get("role")
            pred = parsed.get("predicate", "")
            if role not in ("s", "o") or not pred:
                defer("Relation needs an explicit entity label and subject/object role")
            else:
                labels[i] = label or "related object"
                proposal = ""
                detection_label = repr(label)
                if i != annotated_target and (not label or label.lower() in ABSTRACT_CATEGORIES):
                    proposal = f'{var}_label = vlm_related_name(img_path, {dep}, {pred!r}, {role!r})\n'
                    detection_label = f"{var}_label"
                    label = label or "related object"
                label = label or "target object"
                reference_label = labels.get(deps[0], "reference entity")
                subject, obj = ("c", "reference") if role == "s" else ("reference", "c")
                subject_label, object_label = (label, reference_label) if role == "s" else (reference_label, label)
                # Bind BOTH detected boxes. on/in/depth require visual evidence,
                # not an overlap or ground-plane heuristic.
                relation = normalize_spatial_predicate(pred)
                geometric = pred.strip().lower() in {"left of", "to the left of", "right of", "to the right of", "above", "below"}
                if geometric:
                    check = f'check_spatial_relation({subject}, {obj}, {relation!r})'
                else:
                    check = f'vlm_relation(img_path, {subject}, {obj}, {pred!r}, {subject_label!r}, {object_label!r})'
                candidate_source = (f"annotated_regions(img_path, {regions!r})" if i == annotated_target else
                                    f"detect_and_crop(img_path, {detection_label})")
                entry["tool_call"] = (
                    proposal + f'{var}_candidates = {candidate_source}\n'
                    f'assert len({var}_candidates) > 0, "Uncertain: no relation candidates"\n'
                    f'{var} = [c for c in {var}_candidates if any({check} for reference in {dep})]\n'
                    f'assert len({var}) > 0, "Uncertain: relation not verified"'
                )
                entry["claim_template"] = f"The {subject_label} is {pred} the {object_label}."
                entry.update(subject_role=role, predicate_family="spatial" if geometric else "semantic")
        elif op in {"filter hposition", "filter vposition"}:
            side = step["argument"].strip().lower()
            axis = "horizontal" if op == "filter hposition" else "vertical"
            if side not in ({"left", "right"} if axis == "horizontal" else {"top", "bottom"}):
                defer("Unsupported position argument")
            else:
                labels[i] = labels.get(deps[0], "object")
                entry["tool_call"] = (f'{var} = select_position(img_path, {dep}, {axis!r}, {side!r})\n'
                                      f'assert len({var}) > 0, "Uncertain: no object on requested side"')
                entry["claim_template"] = f"The reference object is on the {side} side of the image."
        elif op == "filter" or op in {"filter pose", "filter material", "filter size", "filter tone",
                                      "filter activity", "filter shape", "filter height"}:
            labels[i] = labels.get(deps[0], "object")
            attribute = step["argument"].strip()
            negated = re.fullmatch(r"not\((.+)\)", attribute)
            if negated:
                attribute = "not " + negated.group(1)
            question = f"Is this object {attribute}?"
            entry["tool_call"] = (f'{var} = [c for c in {dep} if vlm_probe(img_path, c, {question!r})]\n'
                                  f'assert len({var}) > 0, "Uncertain: attribute not verified"')
            entry["claim_template"] = question
        elif op == "query" and parsed.get("attribute") in {"name", "color", "material", "shape", "type"}:
            attr = parsed["attribute"]
            entry["tool_call"] = (f'{var}_answers = [vlm_query(img_path, c, {attr!r}) for c in {dep}]\n'
                                  f'{var} = answer_consensus({var}_answers)\n'
                                  f'final_answer = {var}')
            entry["claim_template"] = f"Read the {attr} of the verified object from the image."
        else:
            defer(f"Unsupported operation: {op}")
        entry["localization_source"] = "viscot_annotation" if i == annotated_target else "model"
        scaffold.append(entry)
    if scaffold and scaffold[-1]["operation"] != "query":
        scaffold[-1].update(needs_review=True, review_reason="No terminal answer query")
    return scaffold


def canonical_code(scaffold):
    return "\n\n".join(step["tool_call"] for step in scaffold)


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
Copy every tool_call_skeleton exactly, in order, into one Python block.
Only comments and whitespace may differ. Do not add fallbacks, weaken checks,
change arguments, or assign the supplied answer to final_answer):

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
