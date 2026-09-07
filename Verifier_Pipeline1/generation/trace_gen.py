"""
trace_gen.py

Calls the coder LLM (e.g. Qwen2.5-Coder-7B-Instruct) with the system
prompt + a sample's scaffold-grounded user prompt. This is stage A:
best-of-N sampling per sample, no claim-level branching yet (that's
mcts.py, stage B+).
"""
import re
import os
import time

from prompts.system_prompt import SYSTEM_PROMPT
from prompts.fewshot_examples import format_fewshot_block
from generation.skeleton_gen import build_scaffold, build_generation_prompt


class CoderModel:
    """Wraps the coder LLM. Loaded once, reused across all samples/attempts."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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


class GeminiCoderModel:
    """Hosted coder; local vision tools still independently execute its code."""

    def __init__(self, model_name="gemini-3.5-flash-lite", min_interval_seconds=30,
                 max_retries=3, thinking_level="high"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to your Google AI Studio API key first.")
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("Install the Gemini SDK: python -m pip install -U google-genai") from None
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.min_interval = max(0, min_interval_seconds)
        self.max_retries = max_retries
        self.thinking_level = thinking_level
        self.last_request = None
        print(f"[trace_gen] Gemini coder: {model_name} (thinking={thinking_level})")

    def generate(self, user_content, temperature=1.0, max_new_tokens=16384):
        # Keep the shared coder interface; Gemini uses its default sampling settings.
        for attempt in range(self.max_retries + 1):
            if self.last_request is not None:
                time.sleep(max(0, self.min_interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.client.interactions.create(
                    model=self.model_name,
                    system_instruction=SYSTEM_PROMPT,
                    input=user_content,
                    store=False,
                    timeout=180,
                    generation_config={
                        "max_output_tokens": max_new_tokens,
                        "thinking_level": self.thinking_level,
                    },
                )
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if str(status) in {"429", "500", "502", "503", "504"} and attempt < self.max_retries:
                    delay = min(60, 15 * 2 ** attempt)
                    print(f"[trace_gen] Gemini HTTP {status}; retrying in {delay}s...", flush=True)
                    time.sleep(delay)
                    continue
                # Do not put credentials or request bodies into logs.
                raise RuntimeError(
                    f"Gemini request failed (HTTP {status or 'unknown'}, {type(exc).__name__}). "
                    "Check the API key, model access and free quota in AI Studio; "
                    "429 can mean the daily quota is exhausted."
                ) from None
            if response.status != "completed":
                raise RuntimeError(f"Gemini response was {response.status}; no complete trace received.")
            text = response.output_text
            if not text or not extract_code_block(text) or not extract_verdict(text):
                raise RuntimeError("Gemini returned an empty/incomplete trace; check token limit or safety blocking.")
            return text


def extract_code_block(trace: str) -> str:
    """Pulls the ```python ... ``` block out of a raw trace. Returns "" if none found."""
    m = re.search(r"```python(.*?)```", trace, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_verdict(trace: str) -> str:
    """Pulls <final_verdict>...</final_verdict> text out. Returns "" if missing."""
    m = re.search(r"<final_verdict>(.*?)</final_verdict>", trace, re.DOTALL)
    return m.group(1).strip() if m else ""


def generate_candidate(sample: dict, coder: CoderModel, temperature: float = 0.6, max_new_tokens: int = 1024) -> dict:
    """
    Builds the scaffold for `sample`, generates one trace attempt, and
    returns everything downstream filters need. Does NOT run/execute the
    code -- that's execution_filter.py's job.
    """
    scaffold = build_scaffold(sample["reasoning"], sample)
    fewshot = ""  # Legacy examples bypass evidence checks; use the exact scaffold.
    user_content = build_generation_prompt(sample, scaffold)
    if fewshot:
        user_content = f"{fewshot}\n\n---\n\nNow verify this new sample:\n\n{user_content}"

    raw_trace = coder.generate(user_content, temperature=temperature, max_new_tokens=max_new_tokens)

    return {
        "sample": sample,
        "scaffold": scaffold,
        "raw_trace": raw_trace,
        "code": extract_code_block(raw_trace),
        "verdict_text": extract_verdict(raw_trace),
    }