# app/graph/nodes/gate_depth.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, Metrics
from app.models.types import DeepDecision
from app.graph.utils.write import a_stream_writer


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
        uncertainties = inp.epistemic_state.get("uncertainties", {})
        high_stakes = inp.epistemic_state.get("high_stakes", {}).get("value", 0.0)
        pe = inp.metrics.get("prediction_error", 0.0)
        cost_user = inp.metrics.get("cost_user", 0.0)
        deep_score = (
            1.0 * pe
            + 0.9 * uncertainties.get("semantic", 0.5)
            + 0.9 * uncertainties.get("epistemic", 0.5)
            + 1.1 * uncertainties.get("social", 0.5)
            + 0.8 * high_stakes
            - 0.6 * cost_user
        )

        reason = ""
        plan: list[str] = []
        if deep_score >= inp.theta_deep:
            if pe >= 0.6 or uncertainties.get("semantic", 0.0) >= 0.6:
                reason = "meaning_mismatch"
                plan = ["deep_repair"]
            elif uncertainties.get("social", 0.0) >= 0.6:
                reason = "persona_premise_mismatch"
                plan = ["deep_memory"]
            elif uncertainties.get("epistemic", 0.0) >= 0.6:
                reason = "need_evidence"
                plan = ["deep_web"]

        if pe >= 0.6 and inp.stance >= 0.7:
            reason = "frame_collapse"
            plan = ["deep_frame"]

        dd: DeepDecision = {
            "reason": reason,
            "repair_plan": {"strategy": "", "questions": [], "optionality": False},
            "deep_chain": {"plan": plan, "executed": [], "stop_reason": ""},
        }
        return GateDepthOut(status="gate_depth:ok", deep_decision=dd)

    @a_stream_writer("gate")
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
