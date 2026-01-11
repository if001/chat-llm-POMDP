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
        norms = inp.joint_context["norms"]
        action: Action = {
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
        return DecidePlanOut(status="decide_response_plan:stub", action=action)

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
