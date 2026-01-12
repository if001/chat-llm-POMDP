# app/graph/nodes/decide_response_plan.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Action
from app.graph.nodes.prompt_utils import (
    format_affective_state,
    format_deep_decision,
    format_epistemic_state,
    format_joint_context,
    format_metrics,
    format_predictions,
)
from app.ports.llm import LLMPort


@dataclass(frozen=True)
class DecidePlanIn:
    joint_context: dict
    deep_decision: dict
    epistemic_state: dict
    predictions: dict
    metrics: dict
    affective_state: dict


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


def _format_allowed_modes(modes: list[str]) -> str:
    if not modes:
        return "なし"
    return " / ".join(str(m) for m in modes)


async def _decide_action(
    small_llm: LLMPort,
    inp: DecidePlanIn,
    fallback_action: Action,
    allowed_modes: list[str],
) -> Action:
    prompt = (
        "あなたはresponse plan(action)の決定器。"
        "入力はjoint_context/deep_decision/predictions/metrics/epistemic_state/affective_state/allowed_modes。"
        "design_docの共同枠組みと予測階層に従い、適切な応答モードと質問方針を決める。"
        "出力はJSONのみ。"
        "出力フォーマット: {"
        '"response_mode": "allowed_modesのいずれか", '
        '"questions_asked": int, '
        '"confirm_questions": ["短い確認質問"], '
        '"did_memory_search": true|false, '
        '"did_web_search": true|false'
        "}。questions_askedはquestion_budget以下。"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力:\n"
                        "- joint_context: 現在の枠組み/役割/規範。response_modeや質問密度の制約。\n"
                        f"{format_joint_context(inp.joint_context)}\n"
                        "- deep_decision: deep_*の理由やrepair計画。応答方針に反映。\n"
                        f"{format_deep_decision(inp.deep_decision)}\n"
                        "- predictions: L0-L4の浅い予測。応答モードの選択に使う。\n"
                        f"{format_predictions(inp.predictions)}\n"
                        "- metrics: 予測誤差やリスク等。保守的/積極的の判断に使う。\n"
                        f"{format_metrics(inp.metrics)}\n"
                        "- epistemic_state: 不確実性/高ステークス。慎重さの調整に使う。\n"
                        f"{format_epistemic_state(inp.epistemic_state)}\n"
                        "- affective_state: 感情/関係指標。語調や質問量の調整に使う。\n"
                        f"{format_affective_state(inp.affective_state)}\n"
                        f"- allowed_modes: {_format_allowed_modes(allowed_modes)}"
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
    response_mode = payload.get("response_mode", action["response_mode"])
    if response_mode in allowed_modes:
        action["response_mode"] = response_mode
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
        frame = inp.joint_context["frame"]
        allowed_modes = {
            "explore": ["ask", "summarize", "clarify", "offer_hypotheses"],
            "decide": ["summarize", "offer_options", "compare", "ask"],
            "execute": ["explain_steps", "confirm", "check_progress"],
            "reflect": ["summarize", "mirror", "ask_open"],
            "vent": ["mirror", "acknowledge", "minimal_ask"],
        }.get(frame, ["ask", "summarize"])
        fallback_action: Action = {
            "chosen_frame": inp.joint_context["frame"],
            "chosen_role_leader": inp.joint_context["roles"]["leader"],
            "response_mode": allowed_modes[0],
            "questions_asked": 0,
            "question_budget": norms["question_budget"],
            "confirm_questions": [],
            "did_memory_search": False,
            "did_web_search": False,
            "used_levels": ["L0", "L1", "L2", "L3", "L4"],
            "used_depths": ["shallow"],
        }
        action = await _decide_action(
            deps.small_llm, inp, fallback_action, allowed_modes
        )
        return DecidePlanOut(status="decide_response_plan:ok", action=action)

    async def node(state: AgentState) -> dict:
        out = await inner(
            DecidePlanIn(
                joint_context=state["joint_context"],
                deep_decision=state["deep_decision"],
                epistemic_state=state["epistemic_state"],
                predictions=state["predictions"],
                metrics=state["metrics"],
                affective_state=state["affective_state"],
            )
        )
        return {"action": out.action}

    return node
