from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from app.core.deps import Deps
from app.models.state import AgentState
from app.graph.nodes.prompt_utils import (
    format_common_ground,
    format_unresolved_points,
    format_user_model,
    format_wm_messages,
)


@dataclass(frozen=True)
class DeepIntentIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict[str, Any]]
    user_model: dict
    common_ground: dict
    unresolved_points: list[dict]


@dataclass(frozen=True)
class DeepIntentOut:
    status: str
    intent_plan: dict[str, Any]
    sources_used_memory: bool
    sources_used_web: bool
    memory_snippets: list[dict]
    web_snippets: list[dict]


def _get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _normalize_intent_plan(payload: dict[str, Any], user_input: str) -> dict[str, Any]:
    plan = dict(payload)
    plan.setdefault("intent_summary", "")
    plan.setdefault("plan_steps", [])
    plan.setdefault("need_memory", False)
    plan.setdefault("need_web", False)
    plan.setdefault("memory_query", "")
    plan.setdefault("web_query", "")
    plan.setdefault("response_mode_hint", "")
    if not plan.get("memory_query"):
        plan["memory_query"] = user_input
    if not plan.get("web_query"):
        plan["web_query"] = user_input
    return plan


async def _infer_intent_plan(
    small_llm,
    user_input: str,
    wm_messages: list[dict[str, Any]],
    user_model: dict,
    common_ground: dict,
    unresolved_points: list[dict],
) -> dict[str, Any]:
    prompt = (
        "あなたは意図推定と行動計画の作成器。"
        "入力はユーザー発話/会話履歴/user_model/common_ground/unresolved_points。"
        "意図を推定し、必要に応じて外部知識取得(記憶検索/ウェブ検索)を選択する。"
        "出力はJSONのみ。"
        "出力フォーマット: {"
        '"intent_summary": "短い要約", '
        '"plan_steps": ["短い手順"], '
        '"need_memory": true|false, '
        '"memory_query": "検索用クエリ", '
        '"need_web": true|false, '
        '"web_query": "検索用クエリ", '
        '"response_mode_hint": "explain|ask|offer_options|summarize|repair|meta_frame|clarify|offer_hypotheses|compare|explain_steps|confirm|check_progress|mirror|ask_open|acknowledge|minimal_ask"'
        "}"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力:\n"
                        f"- user_input: {user_input}\n"
                        "- wm_messages: 直近の会話履歴(最大6件)。意図推定に使う。\n"
                        f"{format_wm_messages(wm_messages, limit=6)}\n"
                        "- user_model: 既知の嗜好/傾向。意図の補助に使う。\n"
                        f"{format_user_model(user_model)}\n"
                        "- common_ground: 共有前提。前提の確認に使う。\n"
                        f"{format_common_ground(common_ground)}\n"
                        "- unresolved_points: 未解決点。検索の必要性判断に使う。\n"
                        f"{format_unresolved_points(unresolved_points)}"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    return _parse_json(_get_content(result))


def make_deep_intent_plan_node(deps: Deps):
    async def inner(inp: DeepIntentIn) -> DeepIntentOut:
        payload = await _infer_intent_plan(
            deps.small_llm,
            inp.user_input,
            inp.wm_messages,
            inp.user_model,
            inp.common_ground,
            inp.unresolved_points,
        )
        plan = _normalize_intent_plan(payload, inp.user_input)

        memory_snippets: list[dict] = []
        web_snippets: list[dict] = []
        used_memory = bool(plan.get("need_memory"))
        used_web = bool(plan.get("need_web"))

        if used_memory:
            try:
                memory_snippets = await deps.memory.recall(str(plan.get("memory_query")))
            except Exception:
                memory_snippets = []
        if used_web:
            try:
                web_snippets = await deps.web.search(str(plan.get("web_query")))
            except Exception:
                web_snippets = []

        return DeepIntentOut(
            status="deep_intent_plan:ok",
            intent_plan=plan,
            sources_used_memory=used_memory,
            sources_used_web=used_web,
            memory_snippets=memory_snippets,
            web_snippets=web_snippets,
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
            DeepIntentIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                user_model=state["user_model"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
            )
        )
        metrics = dict(state["metrics"])
        sources = dict(metrics["sources_used"])
        sources["memory"] = sources["memory"] or out.sources_used_memory
        sources["web"] = sources["web"] or out.sources_used_web
        metrics["sources_used"] = sources

        memory_snippets = list(state["memory_snippets"]) + out.memory_snippets
        web_snippets = list(state["web_snippets"]) + out.web_snippets

        return {
            "intent_plan": out.intent_plan,
            "metrics": metrics,
            "memory_snippets": memory_snippets,
            "web_snippets": web_snippets,
        }

    return node
