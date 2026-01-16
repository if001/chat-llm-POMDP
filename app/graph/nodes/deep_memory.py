# app/graph/nodes/deep_memory.py
from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from app.core.deps import Deps
from app.graph.nodes.prompt_utils import (
    format_common_ground,
    format_unresolved_points,
    format_wm_messages,
)
from app.models.state import AgentState, AssumptionItem, UnresolvedItem, UserModel
from app.models.types import DeepDecision
from app.graph.utils.write import a_stream_writer
from app.graph.utils import utils

chroma_system_prompt = """あなたは Chroma（ベクトルデータベース）検索用の
クエリを生成する役割です。

以下を入力として受け取ります。
- ユーザーの最新の発話(user_input)
- 会話履歴(history)
- ユーザーとの会話で未解決な事項(unresolved_points)

あなたの目的は、
「Chroma に保存されている関連度の高い記憶・知識を
効率よく検索するためのクエリ」を作成することです。

### ルール

1. 最優先するのは「ユーザーの最新の入力」です。
   会話履歴は、意図が不明確な場合のみ補助的に使ってください。

2. クエリは以下の特徴を持たせてください。
   - 短い
   - 意味が明確
   - 抽象化しすぎない
   - 文章ではなく「検索向けの表現」

3. 次のことは禁止します。
   - 質問文のまま出力すること
   - 推測による情報補完
   - 説明文・理由・補足の出力
   - 回答の生成

4. 意図が曖昧な場合は、
   最小限の探索的クエリを1つだけ生成してください。
"""


@dataclass(frozen=True)
class DeepMemoryIn:
    deep_decision: DeepDecision
    user_input: str
    user_model: UserModel
    common_ground: dict[str, list[AssumptionItem]]
    unresolved_points: list[UnresolvedItem]
    wm_messages: list[dict]


@dataclass(frozen=True)
class DeepMemoryOut:
    status: str
    deep_decision: DeepDecision
    sources_used_memory: bool
    memory_snippets: list[dict]


def make_deep_memory_node(deps: Deps):
    async def inner(inp: DeepMemoryIn) -> DeepMemoryOut:
        """
        何をするか:
        - 人物/前提ズレ（L3）に対して deep_memory を実行
          - Chroma に対して recall(query) を行い、過去の前提/嗜好/継続タスクを取得
        - 必要に応じて deep_repair と組み合わせるための材料を出す（ここでは snippets を返す口だけ）
        - deep_chain.executed に "deep_memory" を追加
        - sources_used.memory を True にする
        """
        deep_decision: dict[str, Any] = dict(inp.deep_decision)
        deep_chain = deep_decision.get("deep_chain", {})

        try:
            result = await deps.small_llm.ainvoke(
                [
                    {"role": "system", "content": chroma_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Chroma（ベクトルデータベース）検索用のクエリを生成してください\n"
                            f"- user_input: {inp.user_input}\n\n"
                            f"- history: {format_wm_messages(inp.wm_messages)}\n\n"
                            "- common_ground: 共有前提の一覧。欠落候補の推定\n"
                            f"{format_common_ground(inp.common_ground)}\n\n"
                            "- unresolved_points: 未解決の論点。ギャップ候補の推定\n"
                            f"{format_unresolved_points(inp.unresolved_points)}\n\n"
                        ),
                    },
                ]
            )
            query = utils.get_content(result)
        except Exception:
            query = None

        if query is None:
            deep_chain.setdefault("stop_reason", "memory search error")
            deep_decision["deep_chain"] = deep_chain
            return DeepMemoryOut(
                status="deep_memory:ok",
                deep_decision=DeepDecision(**deep_decision),
                sources_used_memory=True,
                memory_snippets=[],
            )

        search_results = await deps.memory.recall(query)

        return DeepMemoryOut(
            status="deep_memory:ok",
            deep_decision=DeepDecision(**deep_decision),
            sources_used_memory=True,
            memory_snippets=search_results,
        )

    @a_stream_writer("deep_memory")
    async def node(state: AgentState) -> dict:
        out = await inner(
            DeepMemoryIn(
                deep_decision=state["deep_decision"],
                user_input=state["user_input"],
                user_model=state["user_model"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
                wm_messages=state["wm_messages"],
            )
        )
        metrics: dict[str, Any] = dict(state["metrics"])
        sources = dict(metrics["sources_used"])
        sources["memory"] = sources["memory"] or bool(out.sources_used_memory)
        metrics["sources_used"] = sources

        return {
            "deep_decision": out.deep_decision,
            "metrics": metrics,
            "memory_snippets": out.memory_snippets,
        }

    return node
