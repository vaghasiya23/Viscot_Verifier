"""
Holds curated golden exemplars for stage B few-shot prompting (see the
master plan: stage A = bootstrap a small trusted set, stage B = use best
exemplars as few-shot to scale generation across the full dataset).

Empty for now -- populate this AFTER you've run stage A and manually
reviewed a handful of golden traces for quality. Don't fill this in with
unreviewed auto-generated traces; the whole point of few-shot exemplars
is that they're trusted enough to shape every downstream generation.
"""

# List of dicts: {"question":..., "scaffold_summary":..., "full_trace":...}
# Aim for 2-3 exemplars spanning different claim types (spatial, semantic
# relate, query-name, filter) once you have them.
FEWSHOT_EXAMPLES = []


def format_fewshot_block(examples: list = None) -> str:
    """Formats exemplars for injection into the coder prompt. Returns "" if none yet."""
    examples = examples if examples is not None else FEWSHOT_EXAMPLES
    if not examples:
        return ""
    blocks = [f"EXAMPLE {i+1}:\n{ex['full_trace']}" for i, ex in enumerate(examples)]
    return "\n\n".join(blocks)