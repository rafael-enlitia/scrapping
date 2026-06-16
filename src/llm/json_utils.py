"""Shared helpers to extract a single JSON object from LLM text output."""

from __future__ import annotations

import json
import re


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()
    return cleaned


def extract_json_object(raw: str) -> str:
    """Return one JSON object string from noisy LLM output (handles 'Extra data' cases)."""
    cleaned = strip_markdown_fences(raw)

    for candidate in _iter_json_object_candidates(cleaned):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "sentiment" in obj:
            return candidate

    for candidate in _iter_json_object_candidates(cleaned):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    return cleaned


def _iter_json_object_candidates(text: str):
    """Yield JSON object substrings from text."""
    seen: set[str] = set()

    def _offer(s: str):
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            return s
        return None

    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            decoder = json.JSONDecoder()
            _, end = decoder.raw_decode(text[start:])
            cand = _offer(text[start : start + end])
            if cand:
                yield cand
        except json.JSONDecodeError:
            pass
        idx = start + 1

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        cand = _offer(match.group(0))
        if cand:
            yield cand

    cand = _offer(text)
    if cand:
        yield cand
