# app/graph/nodes/respond.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Response
from app.ports.llm import LLMPort


@dataclass(frozen=True)
class RespondIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict]
    action: dict
    predictions: dict
    deep_decision: dict
    sources_used: dict
    joint_context: dict
    memory_snippets: list[dict]
    web_snippets: list[dict]


@dataclass(frozen=True)
class RespondOut:
    status: str
    response: Response


def _get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


async def _generate_response(
    llm: LLMPort,
    inp: RespondIn,
    label: str,
) -> str:
    norms = inp.joint_context.get("norms", {})
    response_mode = inp.action.get("response_mode")
    confirm_questions = inp.action.get("confirm_questions", [])
    repair_plan = inp.deep_decision.get("repair_plan", {})
    frame = inp.joint_context.get("frame")
    leader = inp.joint_context.get("roles", {}).get("leader")
    sources_context = []
    if inp.memory_snippets:
        sources_context.append(f"memory_snippets: {inp.memory_snippets}")
    if inp.web_snippets:
        sources_context.append(f"web_snippets: {inp.web_snippets}")
    sources_block = "\n".join(sources_context) if sources_context else "no external sources"
    prompt = (
        "You are a helpful assistant. "
        "Follow the response_mode, norms, and repair_plan. "
        "If memory/web snippets are provided, ground the response in them and cite them implicitly."
    )
    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"user_input: {inp.user_input}\n"
                        f"frame: {frame}\nleader: {leader}\n"
                        f"norms: {norms}\n"
                        f"response_mode: {response_mode}\n"
                        f"confirm_questions: {confirm_questions}\n"
                        f"repair_plan: {repair_plan}\n"
                        f"predictions: {inp.predictions}\n"
                        f"sources: {sources_block}"
                    ),
                },
            ]
        )
    except Exception:
        return f"{label}"
    return f"{_get_content(result)}{label}"


def make_respond_node(deps: Deps):
    async def inner(inp: RespondIn) -> RespondOut:
        """
        何をするか:
        - action.response_mode に従って最終応答文を生成（LLM）
        - deep_decision.repair_plan がある場合は質問/選択肢/言い換えを組み込む
        - sources_used に応じて末尾ラベル（参照: 会話のみ/記憶/Web検索/両方）を付与
        """
        if inp.sources_used.get("memory") and inp.sources_used.get("web"):
            label = "（参照: 記憶 + Web検索）"
        elif inp.sources_used.get("memory"):
            label = "（参照: 記憶）"
        elif inp.sources_used.get("web"):
            label = "（参照: Web検索）"
        else:
            label = "（参照: 会話のみ）"

        final_text = await _generate_response(deps.llm, inp, label)
        resp: Response = {
            "final_text": final_text,
            "meta": {"sources_label": label, "turn_id": inp.turn_id},
        }
        return RespondOut(status="respond:ok", response=resp)

    async def node(state: AgentState) -> dict:
        out = await inner(
            RespondIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                action=state["action"],
                predictions=state["predictions"],
                deep_decision=state["deep_decision"],
                sources_used=state["metrics"]["sources_used"],
                joint_context=state["joint_context"],
                memory_snippets=state["memory_snippets"],
                web_snippets=state["web_snippets"],
            )
        )
        return {"response": out.response}

    return node
