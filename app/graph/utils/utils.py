import json
from typing import Any


def get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def extract_json_block(text: str) -> str:
    """
    文字列中から最初に現れる { ... } のブロックを抽出する
    ネストした JSON に対応
    """
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return ""


def parse_json(text: str) -> dict[str, Any]:
    try:
        text = extract_json_block(text)
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def parse_llm_response(result: Any) -> dict[str, Any]:
    return parse_json(get_content(result))
