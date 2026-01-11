# app/graph/nodes/predict_shallow.py
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, EpistemicUncertainty
from app.models.types import PredictionCommon
from app.ports.llm import LLMPort


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


def _base_prediction(level: str, turn_id: int) -> PredictionCommon:
    return {
        "level": level,  # type: ignore
        "depth": "shallow",
        "outputs": {},
        "confidence": 0.0,
        "evidence": {
            "from_turns": [turn_id],
            "sources_used": {"memory": False, "web": False},
        },
        "timestamp_turn": turn_id,
    }


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


def _merge_outputs(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        merged[key] = value
    return merged


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


async def _run_small_llm_json(
    small_llm: LLMPort,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception:
        return fallback
    payload = _parse_json(_get_content(result))
    if not payload:
        return fallback
    return payload


async def _predict_l0(
    turn_id: int,
    user_input: str,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L0", turn_id)
    fallback_outputs = {
        "style_fit": 0.5,
        "turn_pressure": 0.5,
        "features": {
            "char_len": len(user_input),
            "question_mark_count": user_input.count("?"),
        },
    }
    payload = await _run_small_llm_json(
        small_llm,
        "Return JSON with keys: outputs (style_fit, turn_pressure, features) and confidence (0-1).",
        f"user_input: {user_input}",
        {"outputs": fallback_outputs, "confidence": 0.0},
    )
    outputs = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["outputs"] = outputs
    pred["confidence"] = _coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l1(
    turn_id: int,
    user_input: str,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L1", turn_id)
    fallback_outputs = {
        "speech_act": "other",
        "grounding_need": 0.5,
        "repair_need": 0.0,
    }
    payload = await _run_small_llm_json(
        small_llm,
        "Return JSON with keys: outputs (speech_act, grounding_need, repair_need) and confidence (0-1).",
        f"user_input: {user_input}",
        {"outputs": fallback_outputs, "confidence": 0.0},
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = _coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l2(
    turn_id: int,
    user_input: str,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L2", turn_id)
    fallback_outputs = {
        "local_intent": "unknown",
        "U_semantic": 0.5,
        "U_epistemic": 0.5,
        "U_social": 0.5,
        "need_question_design": False,
    }
    payload = await _run_small_llm_json(
        small_llm,
        "Return JSON with keys: outputs (local_intent, U_semantic, U_epistemic, U_social, need_question_design) and confidence (0-1).",
        f"user_input: {user_input}",
        {"outputs": fallback_outputs, "confidence": 0.0},
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = _coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l3(
    turn_id: int,
    user_input: str,
    common_ground: dict,
    unresolved_points: list[dict],
    observation: dict,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L3", turn_id)
    fallback_outputs = {
        "cg_gap_candidates": [],
        "stance_update_signal": "none",
    }
    payload = await _run_small_llm_json(
        small_llm,
        "Return JSON with keys: outputs (cg_gap_candidates, stance_update_signal) and confidence (0-1).",
        (
            "user_input: "
            f"{user_input}\ncommon_ground: {common_ground}\n"
            f"unresolved_points: {unresolved_points}\nobservation: {observation}"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = _coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l4(
    turn_id: int,
    user_input: str,
    joint_context: dict,
    metrics_prev: dict,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L4", turn_id)
    fallback_outputs = {
        "l4_trigger_score": 0.0,
        "frame_hypothesis": joint_context.get("frame", "explore"),
    }
    payload = await _run_small_llm_json(
        small_llm,
        "Return JSON with keys: outputs (l4_trigger_score, frame_hypothesis) and confidence (0-1).",
        f"user_input: {user_input}\njoint_context: {joint_context}\nmetrics_prev: {metrics_prev}",
        {"outputs": fallback_outputs, "confidence": 0.0},
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = _coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


def _uncertainties_from_l2(l2_prediction: PredictionCommon) -> EpistemicUncertainty:
    outputs = l2_prediction.get("outputs", {})
    return {
        "semantic": float(outputs.get("U_semantic", 0.5)),
        "epistemic": float(outputs.get("U_epistemic", 0.5)),
        "social": float(outputs.get("U_social", 0.5)),
        "confidence": float(l2_prediction.get("confidence", 0.0)),
    }


def make_predict_shallow_node(deps: Deps):
    async def inner(inp: PredictShallowIn) -> PredictShallowOut:
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
        l0, l1, l2, l3, l4 = await asyncio.gather(
            _predict_l0(inp.turn_id, inp.user_input, deps.small_llm),
            _predict_l1(inp.turn_id, inp.user_input, deps.small_llm),
            _predict_l2(inp.turn_id, inp.user_input, deps.small_llm),
            _predict_l3(
                inp.turn_id,
                inp.user_input,
                inp.common_ground,
                inp.unresolved_points,
                inp.observation,
                deps.small_llm,
            ),
            _predict_l4(
                inp.turn_id,
                inp.user_input,
                inp.joint_context,
                inp.metrics_prev,
                deps.small_llm,
            ),
        )
        preds: dict[str, PredictionCommon] = {
            "L0": l0,
            "L1": l1,
            "L2": l2,
            "L3": l3,
            "L4": l4,
        }

        uncertainties_now = _uncertainties_from_l2(preds["L2"])
        return PredictShallowOut(
            status="predict_shallow:ok",
            predictions=preds,
            uncertainties_now=uncertainties_now,
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
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
