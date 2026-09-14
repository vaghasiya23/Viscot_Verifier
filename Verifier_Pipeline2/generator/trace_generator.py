"""
generator/trace_generator.py - Generation client for visual verification code.
"""
import os
import re
import time
from typing import Optional

from generator.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, format_user_prompt


def extract_code(raw_text: str) -> str:
    """Extracts Python code from Markdown fences."""
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, raw_text)
    if matches:
        return matches[0].strip()
    return raw_text.strip()


class GeminiTraceGenerator:
    """Uses Gemini API to generate executable Python verification code."""

    def __init__(self, model_name: str = "gemini-3.6-flash", api_key: Optional[str] = None, mock: bool = False):
        self.mock = mock or os.environ.get("MOCK_GENERATOR", "").lower() in ("1", "true")
        self.model_name = model_name

        if self.mock:
            self.client = None
            return

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            # Check if api_key.txt exists in pipeline root or parent
            key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_key.txt")
            if os.path.exists(key_path):
                with open(key_path, "r") as f:
                    key = f.read().strip()

        if not key:
            raise RuntimeError(
                "Gemini API key not found. Please export GEMINI_API_KEY or create api_key.txt in Verifier_Pipeline2/"
            )

        from google import genai
        self.client = genai.Client(api_key=key)

    def generate(self, question: str, thought: str, answer: Optional[str] = None, max_retries: int = 3) -> str:
        """Generates verification code for a given question and reasoning trace."""
        if self.mock:
            # Generate a minimal valid verification template for offline testing
            return (
                "def verify_reasoning(img) -> dict:\n"
                "    trace = []\n"
                f"    # Mock verification for question: {question}\n"
                "    trace.append({'step': 1, 'claim': 'mock verification', 'passed': True})\n"
                "    return {'verdict': 'VALID', 'failed_step': None, 'steps': trace}\n"
            )

        user_content = format_user_prompt(question, thought, answer)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}\n\n{user_content}"

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                )
                if response and response.text:
                    return extract_code(response.text)
                raise ValueError("Empty response from Gemini API")
            except Exception as e:
                print(f"[GeminiTraceGenerator] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)
        return ""
