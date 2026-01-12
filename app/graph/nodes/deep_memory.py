# app/graph/nodes/deep_memory.py
from __future__ import annotations

from dataclasses import dataclass

from app.core.deps import Deps
from app.models.state import AgentState
from app.models.types import DeepDecision


@dataclass(frozen=True)
class DeepMemoryIn:
    deep_decision: DeepDecision
    user_input: str
    user_model: dict
    common_ground: dict
    unresolved_points: list[dict]


@dataclass(frozen=True)
class DeepMemoryOut:
    status: str
    deep_decision: DeepDecision
    sources_used_memory: bool
    memory_snippets: list[dict]


def make_deep_memory_node(deps: Deps):
    def inner(inp: DeepMemoryIn) -> DeepMemoryOut:
        """
        何をするか:
        - 人物/前提ズレ（L3）に対して deep_memory を実行
          - Chroma に対して recall(query) を行い、過去の前提/嗜好/継続タスクを取得
        - 必要に応じて deep_repair と組み合わせるための材料を出す（ここでは snippets を返す口だけ）
        - deep_chain.executed に "deep_memory" を追加
        - sources_used.memory を True にする
        """
        dd = dict(inp.deep_decision)
        chain = dict(dd["deep_chain"])
        chain["executed"] = list(chain["executed"]) + ["deep_memory"]
        dd["deep_chain"] = chain
        return DeepMemoryOut(
            status="deep_memory:stub",
            deep_decision=dd,
            sources_used_memory=True,
            memory_snippets=[],
        )

    def node(state: AgentState) -> dict:
        out = inner(
            DeepMemoryIn(
                deep_decision=state["deep_decision"],
                user_input=state["user_input"],
                user_model=state["user_model"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
            )
        )

        # sources_used の更新（metricsに保持）
        metrics = dict(state["metrics"])
        sources = dict(metrics["sources_used"])
        sources["memory"] = sources["memory"] or out.sources_used_memory
        metrics["sources_used"] = sources

        return {
            "deep_decision": out.deep_decision,
            "metrics": metrics,
            "memory_snippets": out.memory_snippets,
        }

    return node
