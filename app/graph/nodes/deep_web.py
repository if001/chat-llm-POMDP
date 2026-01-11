# app/graph/nodes/deep_web.py
from __future__ import annotations

from dataclasses import dataclass

from app.core.deps import Deps
from app.models.state import AgentState
from app.models.types import DeepDecision


@dataclass(frozen=True)
class DeepWebIn:
    deep_decision: DeepDecision
    query: str


@dataclass(frozen=True)
class DeepWebOut:
    status: str
    deep_decision: DeepDecision
    sources_used_web: bool
    web_snippets: list[dict]


def make_deep_web_node(deps: Deps):
    def inner(inp: DeepWebIn) -> DeepWebOut:
        """
        何をするか:
        - 事実根拠が必要（L2）に対して deep_web を実行
          - Firecrawl で外部検索し根拠スニペットを取得
        - deep_chain.executed に "deep_web" を追加
        - sources_used.web を True にする
        """
        dd = dict(inp.deep_decision)
        chain = dict(dd["deep_chain"])
        chain["executed"] = list(chain["executed"]) + ["deep_web"]
        dd["deep_chain"] = chain
        return DeepWebOut(
            status="deep_web:stub",
            deep_decision=dd,
            sources_used_web=True,
            web_snippets=[],
        )

    def node(state: AgentState) -> dict:
        out = inner(
            DeepWebIn(deep_decision=state["deep_decision"], query=state["user_input"])
        )

        metrics = dict(state["metrics"])
        sources = dict(metrics["sources_used"])
        sources["web"] = sources["web"] or out.sources_used_web
        metrics["sources_used"] = sources

        return {"deep_decision": out.deep_decision, "metrics": metrics}

    return node
