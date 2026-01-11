# app/graph/nodes/predict_shallow.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, EpistemicUncertainty
from app.models.types import PredictionCommon


@dataclass(frozen=True)
class PredictShallowIn:
    turn_id: int
    wm_messages: list[dict[str, Any]]
    user_input: str

    # 保持情報
    joint_context: dict
    common_ground: dict
    unresolved_points: list[dict]
    user_model: dict

    # 観測/指標（観測を用いた予測、metricsに基づく補正）
    observation: dict
    metrics_prev: dict


@dataclass(frozen=True)
class PredictShallowOut:
    status: str
    predictions: dict[str, PredictionCommon]  # L0..L4
    uncertainties_now: EpistemicUncertainty
    # 将来拡張: affective_patch / unresolved_patch などを追加可能


def _get_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return getattr(result, "content", "")


def _run_async(coro):
    return asyncio.run(coro)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def make_predict_shallow_node(deps: Deps):
    def inner(inp: PredictShallowIn) -> PredictShallowOut:
        """
        何をするか:
        - L0〜L4 の shallow 予測を行う
          L0: 表層スタイル・応答圧
          L1: 発話行為・グラウンディング兆候
          L2: 局所意図・不確実性(semantic/epistemic/social)
          L3: 人物仮説・common ground ギャップ
          L4: 枠組みズレ兆候・再設計必要度
        - 入力として:
          - user_model / common_ground / unresolved_points / joint_context（保持情報）
          - observation（直近の反応）
          - metrics_prev（PE等により予測を補正）
        - 出力として:
          - predictions(L0..L4)
          - epistemic uncertainties の shallow 推定（uncertainties_now）
        """
        prompt = (
            "You are a classifier. Return JSON with keys: "
            "speech_act, local_intent, U_semantic, U_epistemic, U_social, "
            "style_fit, turn_pressure. Use numeric values 0..1 for U_*/style/pressure."
        )
        response = _run_async(
            deps.small_llm.ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": inp.user_input},
                ]
            )
        )
        parsed = _parse_json(_get_content(response))
        preds: dict[str, PredictionCommon] = {
            "L0": {
                "level": "L0",
                "depth": "shallow",
                "outputs": {
                    "style_fit": float(parsed.get("style_fit", 0.5)),
                    "turn_pressure": float(parsed.get("turn_pressure", 0.5)),
                    "features": {},
                },
                "confidence": 0.0,
                "evidence": {
                    "from_turns": [inp.turn_id],
                    "sources_used": {"memory": False, "web": False},
                },
                "timestamp_turn": inp.turn_id,
            },
            "L1": {
                "level": "L1",
                "depth": "shallow",
                "outputs": {
                    "speech_act": parsed.get("speech_act", "other"),
                    "grounding_need": 0.0,
                    "repair_need": 0.0,
                },
                "confidence": 0.0,
                "evidence": {
                    "from_turns": [inp.turn_id],
                    "sources_used": {"memory": False, "web": False},
                },
                "timestamp_turn": inp.turn_id,
            },
            "L2": {
                "level": "L2",
                "depth": "shallow",
                "outputs": {
                    "local_intent": parsed.get("local_intent", "unknown"),
                    "U_semantic": float(parsed.get("U_semantic", 0.5)),
                    "U_epistemic": float(parsed.get("U_epistemic", 0.5)),
                    "U_social": float(parsed.get("U_social", 0.5)),
                    "need_question_design": False,
                },
                "confidence": 0.0,
                "evidence": {
                    "from_turns": [inp.turn_id],
                    "sources_used": {"memory": False, "web": False},
                },
                "timestamp_turn": inp.turn_id,
            },
            "L3": {
                "level": "L3",
                "depth": "shallow",
                "outputs": {
                    "cg_gap_candidates": [],
                    "stance_update_signal": "",
                },
                "confidence": 0.0,
                "evidence": {
                    "from_turns": [inp.turn_id],
                    "sources_used": {"memory": False, "web": False},
                },
                "timestamp_turn": inp.turn_id,
            },
            "L4": {
                "level": "L4",
                "depth": "shallow",
                "outputs": {
                    "l4_trigger_score": 0.0,
                    "frame_hypothesis": inp.joint_context.get("frame"),
                },
                "confidence": 0.0,
                "evidence": {
                    "from_turns": [inp.turn_id],
                    "sources_used": {"memory": False, "web": False},
                },
                "timestamp_turn": inp.turn_id,
            },
        }

        uncertainties_now: EpistemicUncertainty = {
            "semantic": float(parsed.get("U_semantic", 0.5)),
            "epistemic": float(parsed.get("U_epistemic", 0.5)),
            "social": float(parsed.get("U_social", 0.5)),
            "confidence": 0.0,
        }
        return PredictShallowOut(
            status="predict_shallow:ok",
            predictions=preds,
            uncertainties_now=uncertainties_now,
        )

    def node(state: AgentState) -> dict:
        out = inner(
            PredictShallowIn(
                turn_id=state["turn_id"],
                wm_messages=state["wm_messages"],
                user_input=state["user_input"],
                joint_context=state["joint_context"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
                user_model=state["user_model"],
                observation=state["observation"],
                metrics_prev=state["metrics"],
            )
        )

        new_predictions = dict(state["predictions"])
        for k, v in out.predictions.items():
            new_predictions[k] = v  # type: ignore[assignment]

        new_epistemic = dict(state["epistemic_state"])
        new_epistemic["uncertainties"] = out.uncertainties_now

        return {
            "predictions": new_predictions,
            "epistemic_state": new_epistemic,
        }

    return node
