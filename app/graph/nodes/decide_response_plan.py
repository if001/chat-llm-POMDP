# app/graph/nodes/decide_response_plan.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import (
    AffectiveState,
    AgentState,
    Action,
    DeepDecision,
    EpistemicState,
    JointContext,
    Metrics,
    Predictions,
)
from app.graph.nodes.prompt_utils import (
    format_affective_state,
    format_deep_decision,
    format_epistemic_state,
    format_joint_context,
    format_metrics,
    format_predictions,
)
from app.ports.llm import LLMPort
from app.graph.utils.utils import (
    parse_json,
    coerce_int,
    get_content,
)


@dataclass(frozen=True)
class DecidePlanIn:
    joint_context: JointContext
    deep_decision: DeepDecision
    epistemic_state: EpistemicState
    predictions: Predictions
    metrics: Metrics
    affective_state: AffectiveState


@dataclass(frozen=True)
class DecidePlanOut:
    status: str
    action: Action


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
        "あなたはresponse plan(action)の決定器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "入力に基づいて適切な応答モードと質問方針を決定してください。\n"
        "【入力フィールド】\n"
        "- joint_context: 現在の枠組み/役割/規範。response_modeや質問密度の制約\n"
        "- deep_decision: deep_*の理由やrepair計画。応答方針に反映。\n"
        "- predictions: L0-L4の浅い予測。応答モードの選択\n"
        "- metrics: 予測誤差やリスク等。保守的/積極的の判断\n"
        "- epistemic_state: 不確実性/高ステークス。慎重さの調整\n"
        "- affective_state: 感情/関係指標。語調や質問量の調整\n"
        "- allowed_modes: 許可されたモード\n\n"
        "【出力フィールド】\n"
        "response_mode：このターンでアシスタントが採用する振る舞いの種類（frame に基づき許可された allowed_modes の中から選択）\n"
        "questions_asked：このターンで実際に投げる質問の数（norms.question_budget の遵守確認用）\n"
        "confirm_questions：理解確認や修復のために用いる短い確認質問のリスト（要約確認・意図確認など）\n"
        "did_memory_search：このターンで記憶検索（Chroma 等）を実行したかどうか\n"
        "did_web_search：このターンで外部Web検索（Firecrawl 等）を実行したかどうか\n"
        "【出力フォーマット】\n"
        "{\n"
        '"response_mode": "allowed_modesのいずれか", \n'
        '"questions_asked": int, \n'
        '"confirm_questions": ["短い確認質問"], \n'
        '"did_memory_search": true|false, \n'
        '"did_web_search": true|false\n'
        "}"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力に基づいて適切な応答モードと質問方針を決定してください。\n"
                        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
                        "- joint_context: 現在の枠組み/役割/規範。response_modeや質問密度の制約\n"
                        f"{format_joint_context(inp.joint_context)}\n"
                        "- deep_decision: deep_*の理由やrepair計画。応答方針に反映n"
                        f"{format_deep_decision(inp.deep_decision)}\n"
                        "- predictions: L0-L4の浅い予測。応答モードの選択\n"
                        f"{format_predictions(inp.predictions)}\n"
                        "- metrics: 予測誤差やリスク等。保守的/積極的の判断\n"
                        f"{format_metrics(inp.metrics)}\n"
                        "- epistemic_state: 不確実性/高ステークス。慎重さの調整\n"
                        f"{format_epistemic_state(inp.epistemic_state)}\n"
                        "- affective_state: 感情/関係指標。語調や質問量の調整\n"
                        f"{format_affective_state(inp.affective_state)}\n"
                        f"- allowed_modes: {_format_allowed_modes(allowed_modes)}\n"
                    ),
                },
            ]
        )
    except Exception:
        return fallback_action
    payload = parse_json(get_content(result))
    if not payload:
        return fallback_action
    action = dict(fallback_action)
    response_mode = payload.get("response_mode", action["response_mode"])
    if response_mode in allowed_modes:
        action["response_mode"] = response_mode
    action["questions_asked"] = coerce_int(
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
        try:
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
        except Exception as e:
            print("decide_response exception", e)
            fallback_action: Action = {
                "chosen_frame": state["joint_context"]["frame"],
                "chosen_role_leader": state["joint_context"]["roles"]["leader"],
                "response_mode": "ask",
                "questions_asked": 0,
                "question_budget": state["joint_context"]["norms"]["question_budget"],
                "confirm_questions": [],
                "did_memory_search": False,
                "did_web_search": False,
                "used_levels": ["L0", "L1", "L2", "L3", "L4"],
                "used_depths": ["shallow"],
            }
            return {"action": fallback_action}

    return node
