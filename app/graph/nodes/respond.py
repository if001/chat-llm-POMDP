# app/graph/nodes/respond.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Response
from app.graph.nodes.prompt_utils import (
    format_deep_decision,
    format_joint_context,
    format_predictions,
    format_snippets,
)
from app.ports.llm import LLMPort
from app.config.persona import PersonaConfig


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
    response_mode = inp.action.get("response_mode")
    confirm_questions = inp.action.get("confirm_questions", [])
    frame = inp.joint_context.get("frame")
    leader = inp.joint_context.get("roles", {}).get("leader")
    sources_block = "\n".join(
        [
            format_snippets(inp.memory_snippets, "memory_snippets"),
            format_snippets(inp.web_snippets, "web_snippets"),
        ]
    )
    prompt = (
        "ただし憶測で情報を追加しないこと。\n\n"
        "あなたは対話エージェントの応答生成器。"
        "入力はユーザー発話、joint_context、action、deep_decision、predictions、sources。"
        "design_docの共同枠組み・予測階層に従い、response_modeとnormsを守って応答する。"
        "memory/webの断片がある場合は内容に基づいて応答し、説明は具体的にする。"
        "出力は最終応答文のみ。参照ラベルや余計なメタ文は出力しない。"
    )

    try:
        result = await llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力:\n"
                        f"- user_input: {inp.user_input}\n"
                        "- joint_context: 枠組み/役割/規範。文体と応答方針の制約。\n"
                        f"{format_joint_context(inp.joint_context)}\n"
                        f"- frame: {frame}\n"
                        f"- leader: {leader}\n"
                        "- action.response_mode: 応答の型(質問/要約/比較など)。\n"
                        f"{response_mode}\n"
                        "- action.confirm_questions: 確認質問の候補。必要なら反映。\n"
                        f"{', '.join(str(q) for q in confirm_questions) or 'なし'}\n"
                        "- deep_decision: repair_planなど。修復方針に従う。\n"
                        f"{format_deep_decision(inp.deep_decision)}\n"
                        "- predictions: L0-L4の予測。語調/不確実性/枠組みの判断材料。\n"
                        f"{format_predictions(inp.predictions)}\n"
                        "- sources: memory/web断片(あれば根拠として反映)。\n"
                        f"{sources_block}"
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
