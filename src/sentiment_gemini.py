"""
Simple sentiment scoring via Google Gemini.
Returns (score, summary) where score is in [-1, 1].
Requires `google-generativeai` and a valid API key.
"""
from __future__ import annotations

from typing import Tuple

import google.generativeai as genai  # type: ignore


PROMPT = """You are a quantitative sentiment rater. Read the following crypto-related headlines/notes.
Output a single JSON object with fields:
- "score": a float in [-1, 1] where -1 is very bearish, +1 is very bullish for the asset in question (if unclear, use 0)
- "summary": a concise one-sentence justification.

Text:
<<<
{TEXT}
>>>

Return only JSON, no extra commentary.
"""


def analyze_sentiment_gemini(text: str, api_key: str) -> Tuple[float, str]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = PROMPT.format(TEXT=text.strip()[:6000])
    resp = model.generate_content(prompt)
    content = resp.text or ""
    # Robustly extract JSON
    import json, re
    m = re.search(r"{.*}", content, re.S)
    if not m:
        raise ValueError("Gemini did not return JSON")
    obj = json.loads(m.group(0))
    score = float(obj.get("score", 0.0))
    score = max(-1.0, min(1.0, score))
    summary = str(obj.get("summary", "")).strip()
    return score, summary