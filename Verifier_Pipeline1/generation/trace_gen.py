"""
trace_gen.py

Calls the coder LLM (e.g. Qwen2.5-Coder-7B-Instruct) with the system
prompt + a sample's scaffold-grounded user prompt. This is stage A:
best-of-N sampling per sample, no claim-level branching yet (that's
mcts.py, stage B+).
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts.system_prompt import SYSTEM_PROMPT
from prompts.fewshot_examples import format_fewshot_block
from generation.skeleton_gen import build_scaffold, build_generation_prompt


class CoderModel:
    """Wraps the coder LLM. Loaded once, reused across all samples/attempts."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        print(f"[trace_gen] Loading coder model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate(self, user_content: str, temperature: float = 0.6, max_new_tokens: int = 1024) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True
        )
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def extract_code_block(trace: str) -> str:
    """Pulls the ```python ... ``` block out of a raw trace. Returns "" if none found."""
    m = re.search(r"```python(.*?)```", trace, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_verdict(trace: str) -> str:
    """Pulls <final_verdict>...</final_verdict> text out. Returns "" if missing."""
    m = re.search(r"<final_verdict>(.*?)</final_verdict>", trace, re.DOTALL)
    return m.group(1).strip() if m else ""


def generate_candidate(sample: dict, coder: CoderModel, temperature: float = 0.6) -> dict:
    """
    Builds the scaffold for `sample`, generates one trace attempt, and
    returns everything downstream filters need. Does NOT run/execute the
    code -- that's execution_filter.py's job.
    """
    scaffold = build_scaffold(sample["reasoning"], sample)
    fewshot = format_fewshot_block()
    user_content = build_generation_prompt(sample, scaffold)
    if fewshot:
        user_content = f"{fewshot}\n\n---\n\nNow verify this new sample:\n\n{user_content}"

    raw_trace = coder.generate(user_content, temperature=temperature)

    return {
        "sample": sample,
        "scaffold": scaffold,
        "raw_trace": raw_trace,
        "code": extract_code_block(raw_trace),
        "verdict_text": extract_verdict(raw_trace),
    }