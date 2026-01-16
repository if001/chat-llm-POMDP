# app/graph/nodes/observe_reaction.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, Observation, ObservationEvents, UserModel
from app.graph.nodes.prompt_utils import format_action, format_user_model
from app.ports.llm import LLMPort
from app.graph.utils.write import a_stream_writer
from app.graph.utils import utils


@dataclass(frozen=True)
class ObserveReactionIn:
    turn_id: int
    user_input: str
    prev_assistant_text: str
    prev_action: dict
    user_model: UserModel


@dataclass(frozen=True)
class ObserveReactionOut:
    status: str
    observation: Observation


async def _classify_reaction(
    small_llm: LLMPort,
    user_input: str,
    prev_assistant_text: str,
    prev_action: dict,
    user_model: UserModel,
    fallback: Observation,
) -> Observation:
    prompt = (
        "あなたは前ターンへの反応分類器\n"
        "入力のユーザー発話や人格モデルに基づいて前ターンへの反応分類を推定してください。\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "【入力フィールド】\n"
        "- user_input: ユーザー発話\n"
        "- user_model: ユーザーの人物モデル\n"
        "- prev_assistant_text: 前ターンのassistant出力\n"
        "- prev_action: 前ターンのaction。\n\n"
        "【出力フィールド】\n"
        "reaction_type：ユーザーの全体的な反応タイプの要約（受容・確認要求・訂正・拒否・先送り・話題転換・混在）\n"
        "ack_type：アシスタントの理解や要約に対する同意の明示度（明示的肯定／暗黙的肯定／混在／否定／評価不能）\n"
        "events：対話上の重要イベントを0 or 1で示すフラグ集合（複数同時に立つことがある）\n"
        "events.E_correct：内容の誤りや前提をユーザーが明示的に訂正したか？\n"
        "events.E_refuse：提案・質問・進行方針をユーザーが拒否したか？\n"
        "events.E_clarify：意味や意図の確認を求められたか？\n"
        "events.E_miss：ユーザーの意図や要点を取り逃した兆候が出たか？\n"
        "events.E_frame_break：現在の会話枠組み（frame）が合っていないと示されたか？\n"
        "events.E_overstep：踏み込み過多・言い方の不適切さが示唆されたか？\n"
        "confidence：この行動観測（reaction_type / events 判定）の確信度（投票一致度など、0-1）\n\n"
        "【出力フォーマット】\n"
        "{\n"
        '"reaction_type": "accept|clarify|correct|refuse|defer|topic_shift|mixed", \n'
        '"ack_type": "explicit_yes|implicit_yes|mixed|no|none", \n'
        '"events": {\n'
        '"E_correct": 0|1, "E_refuse": 0|1, "E_clarify": 0|1, \n'
        '"E_miss": 0|1, "E_frame_break": 0|1, "E_overstep": 0|1\n'
        "}, \n"
        '"confidence": 0-1\n'
        "}"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力を用いて前ターンへの反応分類を推定してください。\n"
                        "【重要】前置きや装飾は不要で、必ずJSONのみを出力すること\n"
                        "- user_model:\n"
                        f"{format_user_model(user_model, 4)}\n\n"
                        f"- user_input: {user_input}\n\n"
                        "- prev_assistant_text: 直前のassistant発話。反応の対象。\n"
                        f"{prev_assistant_text}\n\n"
                        "- prev_action: 直前の応答計画。反応の妥当性判断に使う。\n"
                        f"{format_action(prev_action)}\n\n"
                    ),
                },
            ]
        )
    except Exception as e:
        print("observe error: ", e)
        return fallback

    payload = utils.parse_llm_response(result)
    if not payload:
        print("fallback...")
        return fallback

    events = payload.get("events", {})
    fallback_events = fallback["events"]
    merged_events: ObservationEvents = {
        "E_correct": utils.coerce_int(
            events.get("E_correct", fallback_events["E_correct"]), 0
        ),
        "E_refuse": utils.coerce_int(
            events.get("E_refuse", fallback_events["E_refuse"]), 0
        ),
        "E_clarify": utils.coerce_int(
            events.get("E_clarify", fallback_events["E_clarify"]), 0
        ),
        "E_miss": utils.coerce_int(events.get("E_miss", fallback_events["E_miss"]), 0),
        "E_frame_break": utils.coerce_int(
            events.get("E_frame_break", fallback_events["E_frame_break"]), 0
        ),
        "E_overstep": utils.coerce_int(
            events.get("E_overstep", fallback_events["E_overstep"]), 0
        ),
    }

    return {
        "reaction_type": payload.get("reaction_type", fallback["reaction_type"]),
        "ack_type": payload.get("ack_type", fallback["ack_type"]),
        "events": merged_events,
        "confidence": utils.coerce_float(
            payload.get("confidence", fallback["confidence"]), 0.0
        ),
    }


def make_observe_reaction_node(deps: Deps):
    async def inner(inp: ObserveReactionIn) -> ObserveReactionOut:
        """
        何をするか:
        - 前ターンのassistant出力(prev_assistant_text)に対する、今回user_inputの反応を分類
          - reaction_type / ack_type
          - events(E_correct/E_refuse/E_clarify/E_miss/E_frame_break/E_overstep)
        - small LLM vote + 構造特徴（質問未回答など）で confidence を算出
        """
        fallback: Observation = {
            "reaction_type": "mixed",
            "ack_type": "none",
            "events": {
                "E_correct": 0,
                "E_refuse": 0,
                "E_clarify": 0,
                "E_miss": 0,
                "E_frame_break": 0,
                "E_overstep": 0,
            },
            "confidence": 0.0,
        }
        obs = await _classify_reaction(
            deps.small_llm,
            inp.user_input,
            inp.prev_assistant_text,
            inp.prev_action,
            inp.user_model,
            fallback,
        )
        return ObserveReactionOut(status="observe_reaction:ok", observation=obs)

    @a_stream_writer("observe")
    async def node(state: AgentState) -> dict:
        try:
            out = await inner(
                ObserveReactionIn(
                    turn_id=state["turn_id"],
                    user_input=state["user_input"],
                    prev_assistant_text=state["last_turn"]["prev_assistant_text"],
                    prev_action=state["last_turn"]["prev_action"],
                    user_model=state["user_model"],
                )
            )
            return {"observation": out.observation}
        except Exception as e:
            print("observe exception", e)
            fallback: Observation = {
                "reaction_type": "mixed",
                "ack_type": "none",
                "events": {
                    "E_correct": 0,
                    "E_refuse": 0,
                    "E_clarify": 0,
                    "E_miss": 0,
                    "E_frame_break": 0,
                    "E_overstep": 0,
                },
                "confidence": 0.0,
            }
            return {"observation": fallback}

    return node
