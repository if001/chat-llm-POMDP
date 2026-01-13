# app/graph/nodes/deep_web.py
from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from app.core.deps import Deps
from app.models.state import AgentState
from app.models.types import DeepDecision
from app.graph.utils.write import a_stream_writer
from app.graph.utils import utils
from app.graph.nodes.prompt_utils import format_wm_messages


@dataclass(frozen=True)
class DeepWebIn:
    deep_decision: DeepDecision
    query: str

    user_input: str
    wm_messages: list[dict[str, Any]]


@dataclass(frozen=True)
class DeepWebOut:
    status: str
    deep_decision: DeepDecision
    sources_used_web: bool
    web_snippets: list[dict]


system_prompt = """あなたは検索クエリ生成システムです。

あなたの任務は、以下の情報に基づいて最適なウェブ検索クエリを生成することです：
- ユーザーの最新の入力内容
- 提供された会話履歴

あなたの出力結果は直接ウェブ検索に使用されます。
ユーザーの質問に答えることではなく、検索の効率性と関連性に焦点を当てなければなりません。

以下のルールを厳守すること：

1. ユーザーの現在の情報ニーズを理解する
   - 最新のユーザーメッセージを優先する
   - 意図を明確化する場合に限り会話履歴を利用する

2. 以下の条件を満たす検索クエリを生成する
   - 簡潔かつ具体的であること
   - 一般的な検索用語を使用すること
   - 不要な丁寧語や説明を避けること
   - 推測や推論に基づく事実を避けること

3. 以下の行為は厳禁：
   - 質問に直接回答しない
   - 内容を要約しない
   - 説明や推論を追加しない
   - 追問を行わない

4. ユーザーの意図が不明確な場合：
   - 最小限の探索クエリを生成する
   - 不足する詳細を推測しない
"""


def make_deep_web_node(deps: Deps):
    async def inner(inp: DeepWebIn) -> DeepWebOut:
        """
        何をするか:
        - 事実根拠が必要（L2）に対して deep_web を実行
          - Firecrawl で外部検索し根拠スニペットを取得
        - deep_chain.executed に "deep_web" を追加
        - sources_used.web を True にする
        """
        deep_decision: dict[str, Any] = dict(inp.deep_decision)
        deep_chain = deep_decision.get("deep_chain", {})
        executed = list(deep_chain.get("executed", [])) + ["deep_web"]

        deep_chain["plan"] = []
        deep_chain["executed"] = executed

        prompt = (
            "ウェブ検索クエリを生成してください。\n"
            f"- user_input: {inp.user_input}\n"
            f"- history:"
            f"{format_wm_messages(inp.wm_messages, limit=6)}\n"
        )
        query = None
        try:
            result = await deps.llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            query = utils.get_content(result)
        except Exception:
            query = None
        if query is None:
            deep_chain.setdefault("stop_reason", "web search error")
            deep_decision["deep_chain"] = deep_chain
            return DeepWebOut(
                status="deep_web:ok",
                deep_decision=DeepDecision(**deep_decision),
                sources_used_web=True,
                web_snippets=[],
            )
        search_results = await deps.web.search(query)

        deep_chain.setdefault("stop_reason", "")
        deep_decision["deep_chain"] = deep_chain
        return DeepWebOut(
            status="deep_web:ok",
            deep_decision=DeepDecision(**deep_decision),
            sources_used_web=True,
            web_snippets=search_results,
        )

    @a_stream_writer("deep_web")
    async def node(state: AgentState) -> dict:
        out = await inner(
            DeepWebIn(
                deep_decision=state["deep_decision"],
                query=state["user_input"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
            )
        )
        metrics: dict[str, Any] = dict(state["metrics"])
        sources: dict[str, Any] = dict(metrics["sources_used"])
        sources["web"] = sources["web"] or out.sources_used_web
        metrics["sources_used"] = sources

        return {
            "deep_decision": out.deep_decision,
            "metrics": metrics,
            "web_snippets": out.web_snippets,
        }

    return node
