# app/graph/nodes/decide_response_plan.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Action
from app.ports.llm import LLMPort


@dataclass(frozen=True)
class DecidePlanIn:
    joint_context: dict
    deep_decision: dict
    epistemic_state: dict
    predictions: dict
    metrics: dict


@dataclass(frozen=True)
class DecidePlanOut:
    status: str
    action: Action


def _get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


async def _decide_action(
    small_llm: LLMPort,
    inp: DecidePlanIn,
    fallback_action: Action,
) -> Action:
    prompt = (
        "Return JSON with keys: response_mode, questions_asked, confirm_questions, "
        "did_memory_search, did_web_search. "
        "response_mode is one of explain, ask, offer_options, summarize, repair, meta_frame."
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"joint_context: {inp.joint_context}\n"
                        f"deep_decision: {inp.deep_decision}\n"
                        f"predictions: {inp.predictions}\n"
                        f"metrics: {inp.metrics}"
                    ),
                },
            ]
        )
    except Exception:
        return fallback_action
    payload = _parse_json(_get_content(result))
    if not payload:
        return fallback_action
    action = dict(fallback_action)
    action["response_mode"] = payload.get("response_mode", action["response_mode"])
    action["questions_asked"] = _coerce_int(
        payload.get("questions_asked", action["questions_asked"]),
        action["questions_asked"],
    )
    action["confirm_questions"] = payload.get(
        "confirm_questions", action["confirm_questions"]
    )
    action["did_memory_search"] = bool(
        payload.get("did_memory_search", action["did_memory_search"])
    )
    action["did_web_search"] = bool(
        payload.get("did_web_search", action["did_web_search"])
    )
    return action


def make_decide_response_plan_node(deps: Deps):
    async def inner(inp: DecidePlanIn) -> DecidePlanOut:
        """
        何をするか:
        - joint_context/norms と、deep_decision（repair_plan含む）と、
          predictions/metrics を材料に response plan(action) を決定
          - response_mode（explain/ask/offer_options/summarize/repair/meta_frame）
          - 質問数・確認質問
          - memory/web を使う（計画/意図）かどうか
        """
        norms = inp.joint_context["norms"]
        fallback_action: Action = {
            "chosen_frame": inp.joint_context["frame"],
            "chosen_role_leader": inp.joint_context["roles"]["leader"],
            "response_mode": "explain",
            "questions_asked": 0,
            "question_budget": norms["question_budget"],
            "confirm_questions": [],
            "did_memory_search": False,
            "did_web_search": False,
            "used_levels": ["L0", "L1", "L2", "L3", "L4"],
            "used_depths": ["shallow"],
        }
        action = await _decide_action(deps.small_llm, inp, fallback_action)
        return DecidePlanOut(status="decide_response_plan:ok", action=action)

    async def node(state: AgentState) -> dict:
        out = await inner(
            DecidePlanIn(
                joint_context=state["joint_context"],
                deep_decision=state["deep_decision"],
                epistemic_state=state["epistemic_state"],
                predictions=state["predictions"],
                metrics=state["metrics"],
            )
        )
        return {"action": out.action}

    return node
