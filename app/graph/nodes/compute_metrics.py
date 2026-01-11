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
        # stub: ひとまず前回値を踏襲
        m = dict(inp.metrics_prev)
        return ComputeMetricsOut(status="compute_metrics:stub", metrics=m)  # type: ignore[arg-type]

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
