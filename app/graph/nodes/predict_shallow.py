# app/graph/nodes/predict_shallow.py
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import (
    AgentState,
    EpistemicUncertainty,
    JointContext,
    Metrics,
    Observation,
    UserModel,
    AssumptionItem,
    UnresolvedItem,
)
from app.models.types import PredictionCommon
from app.graph.nodes.prompt_utils import (
    format_common_ground,
    format_joint_context,
    format_metrics,
    format_observation,
    format_unresolved_points,
)
from app.ports.llm import LLMPort
from app.graph.utils.write import a_stream_writer


@dataclass(frozen=True)
class PredictShallowIn:
    turn_id: int
    wm_messages: list[dict[str, Any]]
    user_input: str

    # 保持情報
    joint_context: JointContext
    common_ground: dict[str, list[AssumptionItem]]
    unresolved_points: list[UnresolvedItem]
    user_model: UserModel

    # 観測/指標（観測を用いた予測、metricsに基づく補正）
    observation: Observation
    metrics_prev: Metrics


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
        (
            "あなたはL0(表層・スタイル予測)の分類器。"
            "入力はこのターンのユーザー発話のみ。"
            "表層的特徴(語感/文長/圧力)の推定に使う。"
            "出力はJSONのみ。"
            "出力フォーマット: "
            "{"
            '"outputs": {"style_fit": 0-1, "turn_pressure": 0-1, '
            '"features": {"char_len": int, "question_mark_count": int}}, '
            '"confidence": 0-1'
            "}"
        ),
        (
            "入力:\n"
            f"- user_input: {user_input}\n"
            "用途: L0のstyle_fit/turn_pressure/表層特徴量の推定。"
        ),
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
        (
            "あなたはL1(発話行為/グラウンディング予測)の分類器。"
            "入力はこのターンのユーザー発話。"
            "発話行為とグラウンディング/repairの必要度を推定する。"
            "出力はJSONのみ。"
            "出力フォーマット: "
            "{"
            '"outputs": {"speech_act": "ask|answer|correct|vent|meta|other", '
            '"grounding_need": 0-1, "repair_need": 0-1}, '
            '"confidence": 0-1'
            "}"
        ),
        (
            "入力:\n"
            f"- user_input: {user_input}\n"
            "用途: L1のspeech_act/grounding_need/repair_need推定。"
        ),
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
        (
            "あなたはL2(局所意図/不確実性予測)の分類器。"
            "入力はこのターンのユーザー発話。"
            "局所意図と不確実性(U_semantic/U_epistemic/U_social)を推定する。"
            "出力はJSONのみ。"
            "出力フォーマット: "
            "{"
            '"outputs": {"local_intent": "短いラベル", '
            '"U_semantic": 0-1, "U_epistemic": 0-1, "U_social": 0-1, '
            '"need_question_design": true|false}, '
            '"confidence": 0-1'
            "}"
        ),
        (
            "入力:\n"
            f"- user_input: {user_input}\n"
            "用途: L2の意図/不確実性/質問設計必要性の推定。"
        ),
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
        (
            "あなたはL3(人物モデル/共通基盤の欠落予測)の分類器。"
            "入力はユーザー発話と共通基盤/未解決/観測。"
            "共通基盤の欠落候補やスタンス変化の兆候を推定する。"
            "出力はJSONのみ。"
            "出力フォーマット: "
            "{"
            '"outputs": {"cg_gap_candidates": ["短い候補"], '
            '"stance_update_signal": "none|shift|strengthen|soften|other"}, '
            '"confidence": 0-1'
            "}"
        ),
        (
            "入力:\n"
            f"- user_input: {user_input}\n"
            "- common_ground: 共有前提の一覧。欠落候補の推定に使う。\n"
            f"{format_common_ground(common_ground)}\n"
            "- unresolved_points: 未解決の論点。ギャップ候補の推定に使う。\n"
            f"{format_unresolved_points(unresolved_points)}\n"
            "- observation: 直近の反応分類。スタンス変化推定に使う。\n"
            f"{format_observation(observation)}"
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
        (
            "あなたはL4(枠組み再設計/長期価値予測)のトリガ判定。"
            "入力はユーザー発話とjoint_contextと前回metrics。"
            "枠組み崩壊や再交渉の必要度を推定する。"
            "出力はJSONのみ。"
            "出力フォーマット: "
            "{"
            '"outputs": {"l4_trigger_score": 0-1, '
            '"frame_hypothesis": "explore|decide|execute|reflect|vent"}, '
            '"confidence": 0-1'
            "}"
        ),
        (
            "入力:\n"
            f"- user_input: {user_input}\n"
            "- joint_context: 現在の枠組み/役割/規範。frame仮説の推定に使う。\n"
            f"{format_joint_context(joint_context)}\n"
            "- metrics_prev: 前回指標(PE,ΔI/ΔG/ΔJ等)。トリガ判定に使う。\n"
            f"{format_metrics(metrics_prev)}"
        ),
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

    @a_stream_writer("predict")
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
