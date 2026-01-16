from __future__ import annotations

from dataclasses import dataclass

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
class DecideRepairPlanIn:
    joint_context: JointContext
    deep_decision: DeepDecision
    epistemic_state: EpistemicState
    predictions: Predictions
    metrics: Metrics
    affective_state: AffectiveState


@dataclass(frozen=True)
class DecideRepairPlanOut:
    status: str
    action: Action


def _format_allowed_modes(modes: list[str]) -> str:
    if not modes:
        return "なし"
    return " / ".join(str(m) for m in modes)


def _repair_allowed_modes(frame: str, reason: str) -> list[str]:
    base = {
        "explore": ["ask", "clarify", "summarize", "offer_hypotheses"],
        "decide": ["summarize", "offer_options", "compare", "ask"],
        "execute": ["explain_steps", "confirm", "check_progress"],
        "reflect": ["summarize", "mirror", "ask_open"],
        "vent": ["mirror", "acknowledge", "minimal_ask"],
    }.get(frame, ["ask", "summarize"])
    repair_modes = ["repair", "clarify", "ask", "summarize"]
    if reason == "frame_collapse":
        repair_modes.append("meta_frame")
    modes = list(dict.fromkeys(base + repair_modes))
    return modes


async def _decide_action(
    small_llm: LLMPort,
    inp: DecideRepairPlanIn,
    fallback_action: Action,
    allowed_modes: list[str],
) -> Action:
    prompt = (
        "あなたはrepair専用のresponse plan(action)決定器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "予測誤差・ズレ・破綻の修正を最優先に、応答モードと質問方針を決めてください。\n"
        "【入力フィールド】\n"
        "- joint_context: 現在の枠組み/役割/規範。response_modeや質問密度の制約\n"
        "- deep_decision: 修復理由やrepair_plan。修復の方針\n"
        "- predictions: L0-L4の浅い予測。誤差/ズレの補助\n"
        "- metrics: 予測誤差やリスク等。保守的/積極的の判断\n"
        "- epistemic_state: 不確実性/高ステークス。慎重さの調整\n"
        "- affective_state: 感情/関係指標。語調や質問量の調整\n"
        "- allowed_modes: 許可されたモード\n\n"
        "【出力フィールド】\n"
        "response_mode：このターンで採用する修復寄りの振る舞い（allowed_modesから選択）\n"
        "questions_asked：このターンで実際に投げる質問の数（norms.question_budgetの遵守確認用）\n"
        "confirm_questions：理解確認や修復のために用いる短い確認質問リスト\n"
        "did_memory_search：このターンで記憶検索を実行したかどうか\n"
        "did_web_search：このターンで外部Web検索を実行したかどうか\n"
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
                        "入力に基づいて修復寄りの応答モードと質問方針を決定してください。\n"
                        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
                        "- joint_context: 現在の枠組み/役割/規範。response_modeや質問密度の制約\n"
                        f"{format_joint_context(inp.joint_context)}\n"
                        "- deep_decision: 修復理由やrepair_plan。修復の方針\n"
                        f"{format_deep_decision(inp.deep_decision)}\n"
                        "- predictions: L0-L4の浅い予測。誤差/ズレの補助\n"
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


def make_decide_repair_plan_node(deps: Deps):
    async def inner(inp: DecideRepairPlanIn) -> DecideRepairPlanOut:
        norms = inp.joint_context["norms"]
        frame = inp.joint_context["frame"]
        reason = inp.deep_decision.get("reason", "")
        allowed_modes = _repair_allowed_modes(frame, reason)
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
            "used_depths": ["deep"],
        }
        action = await _decide_action(
            deps.small_llm, inp, fallback_action, allowed_modes
        )
        return DecideRepairPlanOut(status="decide_repair_plan:ok", action=action)

    async def node(state: AgentState) -> dict:
        out = await inner(
            DecideRepairPlanIn(
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
