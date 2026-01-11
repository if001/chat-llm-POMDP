# app/graph/nodes/compute_metrics.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, EpistemicUncertainty, Metrics


@dataclass(frozen=True)
class ComputeMetricsIn:
    # 観測と予測を結合してメトリクスを算出する
    observation: dict
    predictions: dict

    # 前後差分（ΔI等）
    uncertainties_prev: EpistemicUncertainty
    uncertainties_now: EpistemicUncertainty

    # unresolved差分（ΔG/Cost_user等）
    unresolved_prev_count: int
    unresolved_now_count: int
    resolved_count: int

    # cost/risk計算の材料（stubでも口を用意）
    prev_action: dict
    prev_response_meta: dict

    # 根拠利用（Risk_misinformationなど）
    sources_used: dict

    # 前回metrics（EMAなど将来拡張のため）
    metrics_prev: Metrics


@dataclass(frozen=True)
class ComputeMetricsOut:
    status: str
    metrics: Metrics


def make_compute_metrics_node():
    def inner(inp: ComputeMetricsIn) -> ComputeMetricsOut:
        """
        何をするか:
        - observation + predictions + 前後差分を用いて以下を計算
          - PE_t
          - ΔI, ΔG, ΔJ
          - Risk, Cost_user, Cost_agent
          - sources_used は deep_* の実行結果を反映
        - 特に ΔI は uncertainties_prev/now を用いて算出
        - ΔG は unresolved の増減・解消(resolved_count)を用いて算出
        """
        def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, val))

        events = inp.observation.get("events", {})
        pe = (
            0.4 * events.get("E_correct", 0)
            + 0.6 * events.get("E_refuse", 0)
            + 0.3 * events.get("E_clarify", 0)
            + 0.2 * events.get("E_miss", 0)
            + 0.4 * events.get("E_frame_break", 0)
            + 0.5 * events.get("E_overstep", 0)
        )
        pe = _clamp(pe)

        delta_i = (
            inp.uncertainties_prev.get("epistemic", 0.5)
            - inp.uncertainties_now.get("epistemic", 0.5)
        )
        delta_g = (
            inp.resolved_count
            - (inp.unresolved_now_count - inp.unresolved_prev_count)
        )
        ack_type = inp.observation.get("ack_type")
        if ack_type in {"explicit_yes", "implicit_yes"}:
            delta_g += 0.2
        elif ack_type == "no":
            delta_g -= 0.2

        reaction_type = inp.observation.get("reaction_type")
        delta_j = 0.0
        if reaction_type == "accept":
            delta_j += 0.2
        if reaction_type == "refuse":
            delta_j -= 0.3
        if reaction_type == "topic_shift" and events.get("E_frame_break", 0) == 1:
            delta_j -= 0.2

        risk = _clamp(
            0.5 * inp.uncertainties_now.get("social", 0.5)
            + 0.3 * events.get("E_overstep", 0)
            + 0.3 * events.get("E_refuse", 0)
        )
        cost_user = _clamp(0.2 * inp.unresolved_now_count)

        metrics: Metrics = {
            "prediction_error": pe,
            "delta_I": delta_i,
            "delta_G": delta_g,
            "delta_J": delta_j,
            "risk": risk,
            "cost_user": cost_user,
            "cost_agent": inp.metrics_prev.get("cost_agent", 0.0),
            "sources_used": dict(inp.sources_used),
        }
        return ComputeMetricsOut(status="compute_metrics:ok", metrics=metrics)

    def node(state: AgentState) -> dict:
        out = inner(
            ComputeMetricsIn(
                observation=state["observation"],
                predictions=state["predictions"],
                uncertainties_prev=state["last_turn"]["prev_uncertainties"],
                uncertainties_now=state["epistemic_state"]["uncertainties"],
                unresolved_prev_count=state["last_turn"]["prev_unresolved_count"],
                unresolved_now_count=len(state["unresolved_points"]),
                resolved_count=state["last_turn"]["resolved_count_last_turn"],
                prev_action=state["last_turn"]["prev_action"],
                prev_response_meta=state["last_turn"]["prev_response_meta"],
                sources_used=state["metrics"]["sources_used"],
                metrics_prev=state["metrics"],
            )
        )
        return {"metrics": out.metrics}

    return node
