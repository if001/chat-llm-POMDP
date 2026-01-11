# app/graph/nodes/respond.py
from __future__ import annotations

from dataclasses import dataclass

from app.core.deps import Deps
from app.models.state import AgentState, Response


@dataclass(frozen=True)
class RespondIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict]
    action: dict
    predictions: dict
    deep_decision: dict
    sources_used: dict


@dataclass(frozen=True)
class RespondOut:
    status: str
    response: Response


def make_respond_node(deps: Deps):
    def inner(inp: RespondIn) -> RespondOut:
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

        resp: Response = {
            "final_text": "",
            "meta": {"sources_label": label, "turn_id": inp.turn_id},
        }
        return RespondOut(status="respond:stub", response=resp)

    def node(state: AgentState) -> dict:
        out = inner(
            RespondIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                action=state["action"],
                predictions=state["predictions"],
                deep_decision=state["deep_decision"],
                sources_used=state["metrics"]["sources_used"],
            )
        )
        return {"response": out.response}

    return node
