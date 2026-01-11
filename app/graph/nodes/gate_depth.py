# app/graph/nodes/gate_depth.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, Metrics
from app.models.types import DeepDecision


@dataclass(frozen=True)
class GateDepthIn:
    metrics: Metrics
    predictions: dict
    epistemic_state: dict

    unresolved_points_count: int
    stance: float

    # 学習で更新されるゲート閾値など
    theta_deep: float
    deep_history: list[str]


@dataclass(frozen=True)
class GateDepthOut:
    status: str
    deep_decision: DeepDecision


def make_gate_depth_node():
    def inner(inp: GateDepthIn) -> GateDepthOut:
        """
        何をするか:
        - metrics(PE, uncertainties, high_stakes, cost_user 等)と theta_deep を用いて deep_score を計算
        - deep_reason を決定（meaning_mismatch / need_evidence / persona_premise_mismatch / frame_collapse）
        - deep_chain.plan を作成（最大2段などのポリシーに従う）
        """
        dd: DeepDecision = {
            "reason": "",
            "repair_plan": {"strategy": "", "questions": [], "optionality": False},
            "deep_chain": {"plan": [], "executed": [], "stop_reason": ""},
        }
        return GateDepthOut(status="gate_depth:stub", deep_decision=dd)

    def node(state: AgentState) -> dict:
        out = inner(
            GateDepthIn(
                metrics=state["metrics"],
                predictions=state["predictions"],
                epistemic_state=state["epistemic_state"],
                unresolved_points_count=len(state["unresolved_points"]),
                stance=state["affective_state"]["interpersonal_stance"],
                theta_deep=state["policy"]["theta_deep"],
                deep_history=state["policy"]["deep_history"],
            )
        )
        return {"deep_decision": out.deep_decision}

    return node
