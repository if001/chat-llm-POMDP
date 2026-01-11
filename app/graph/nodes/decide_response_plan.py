# app/graph/nodes/decide_response_plan.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, Action


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


def make_decide_response_plan_node():
    def inner(inp: DecidePlanIn) -> DecidePlanOut:
        """
        何をするか:
        - joint_context/norms と、deep_decision（repair_plan含む）と、
          predictions/metrics を材料に response plan(action) を決定
          - response_mode（explain/ask/offer_options/summarize/repair/meta_frame）
          - 質問数・確認質問
          - memory/web を使う（計画/意図）かどうか
        """
        reason = inp.deep_decision.get("reason", "")
        response_mode = "explain"
        if reason == "frame_collapse":
            response_mode = "meta_frame"
        elif reason == "meaning_mismatch":
            response_mode = "repair"

        repair_plan = inp.deep_decision.get("repair_plan", {})
        used_levels = list(inp.predictions.keys())
        used_depths = [
            p.get("depth", "shallow") for p in inp.predictions.values() if isinstance(p, dict)
        ]
        action: Action = {
            "chosen_frame": inp.joint_context["frame"],
            "chosen_role_leader": inp.joint_context["roles"]["leader"],
            "response_mode": response_mode,
            "questions_asked": 0,
            "question_budget": inp.joint_context["norms"]["question_budget"],
            "confirm_questions": list(repair_plan.get("questions", [])),
            "did_memory_search": reason == "persona_premise_mismatch",
            "did_web_search": reason == "need_evidence",
            "used_levels": used_levels,
            "used_depths": used_depths or ["shallow"],
        }
        return DecidePlanOut(status="decide_response_plan:ok", action=action)

    def node(state: AgentState) -> dict:
        out = inner(
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
