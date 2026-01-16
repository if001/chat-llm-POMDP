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
    format_wm_messages,
    format_user_model,
)
from app.ports.llm import LLMPort
from app.graph.utils.write import a_stream_writer
from app.graph.utils.utils import coerce_float, parse_llm_response


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


def _merge_outputs(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        merged[key] = value
    return merged


async def _run_small_llm_json(
    small_llm: LLMPort,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
    layer: str,
) -> dict[str, Any]:
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        print("run small llm json error", e)
        return fallback
    payload = parse_llm_response(result)
    if not payload:
        print("predict fallback: ", layer)
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
            "あなたは表層・スタイル予測の分類器\n"
            "ユーザーの入力を用いて表層的特徴(語感/文長/圧力)の推定を行ってください。\n\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
            "【出力フォーマット】\n"
            "{\n"
            '"outputs": {"style_fit": 0-1, "turn_pressure": 0-1, \n'
            '"features": {"char_len": int, "question_mark_count": int}}, \n'
            '"confidence": 0-1\n'
            "}"
        ),
        (
            "ユーザーの入力を用いて表層的特徴(語感/文長/圧力)の推定を行ってください。\n"
            f"user_input: {user_input}\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
        "L0",
    )
    outputs = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["outputs"] = outputs
    pred["confidence"] = coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l1(
    turn_id: int,
    user_input: str,
    wm_messages: list[dict],
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L1", turn_id)
    fallback_outputs = {
        "speech_act": "other",
        "grounding_need": 0.2,
        "repair_need": 0.0,
    }
    payload = await _run_small_llm_json(
        small_llm,
        (
            "あなたは発話行為/グラウンディング予測の分類器。\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
            "ユーザー入力を用いて発話行為/グラウンディング予測の分類を行ってください。\n\n"
            "【出力フィールド】\n"
            "speech_act：ユーザー発話（または直近ターン）の主要な発話行為ラベル（質問・回答・訂正・吐露・メタ等）\n"
            "grounding_need：意味の共有・前提合わせ（言い換え／確認）が必要な度合い（0=不要、1=強く必要）\n"
            "repair_need：誤解やズレを解消するために修復手続き（確認質問・要約確認等）を起動すべき度合い（0-1）\n"
            "confidence：推定（speech_act/need値）の確信度（投票一致率など、0-1）\n\n"
            "【出力フォーマット】\n"
            "{\n"
            '"outputs": {"speech_act": "ask|answer|correct|vent|meta|other", \n'
            '"grounding_need": 0-1, "repair_need": 0-1}, \n'
            '"confidence": 0-1\n'
            "}"
        ),
        (
            "ユーザー入力を用いて発話行為/グラウンディング予測の分類を行ってください。\n"
            f"user_input: {user_input}\n\n"
            "- history:\n"
            f"{format_wm_messages(wm_messages, limit=8)}\n\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
        "L1",
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l2(
    turn_id: int,
    user_input: str,
    wm_messages: list[dict],
    user_model: UserModel,
    small_llm: LLMPort,
) -> PredictionCommon:
    pred = _base_prediction("L2", turn_id)
    fallback_outputs = {
        "local_intent": "unknown",
        "U_semantic": 0.2,
        "U_epistemic": 0.2,
        "U_social": 0.5,
        "need_question_design": False,
    }
    payload = await _run_small_llm_json(
        small_llm,
        (
            "あなたは局所意図/不確実性予測の分類器\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
            "ユーザー入力を用いて意図/不確実性/質問設計必要性の推定を行ってください。\n\n"
            "【出力フィールド】\n"
            "- local_intent：この発話でユーザーが達成したい局所目的の短いラベル（例：情報提供、依頼、意思決定、整理、苦情など）\n"
            "- U_semantic：用語・指示対象・意図など「意味」が曖昧で誤解しやすい度合い（0-1）\n"
            "- U_epistemic：事実・条件・根拠など「知識／情報」が不足している度合い（0-1）\n"
            "- U_social：踏み込み・言い方・関係性など「社会的リスク」が不確かな度合い（0-1）\n"
            "- need_question_design：不確実性を減らすために、質問の設計（優先順位付けや分岐）が必要かどうか\n"
            "- confidence：推定（intent/U値/need）の確信度（0-1）\n\n"
            "【出力フォーマット】\n"
            "{\n"
            '"outputs": {"local_intent": "短いラベル", \n'
            '"U_semantic": 0-1, "U_epistemic": 0-1, "U_social": 0-1, \n'
            '"need_question_design": true|false}, \n'
            '"confidence": 0-1\n'
            "}"
        ),
        (
            "ユーザー入力と履歴から意図/不確実性/質問設計必要性の推定を行ってください。\n"
            f"- user_input: {user_input}\n\n"
            "- history:\n"
            f"{format_wm_messages(wm_messages, limit=8)}\n\n"
            "- user_model:\n"
            f"{format_user_model(user_model)}\n\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
        "L2",
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l3(
    turn_id: int,
    user_input: str,
    wm_messages: list[dict],
    user_model: UserModel,
    common_ground: dict,
    unresolved_points: list[UnresolvedItem],
    observation: Observation,
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
            "あなたは人物モデル/共通基盤の欠落予測の分類器\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
            "ユーザー入力を用いて人物モデル/共通基盤の欠落予測の分類器してください。\n\n"
            "【入力フィールド】\n"
            "- user_input: ユーザー入力\n"
            "- history: 直近の会話履歴\n"
            "- common_ground: 共有前提の一覧。欠落候補の推定\n"
            "- unresolved_points: 未解決の論点。ギャップ候補の推定\n"
            "- observation: 直近の反応分類。スタンス変化推定\n\n"
            "【出力フィールド】\n"
            "- cg_gap_candidates：共有不足（common ground の穴）として疑わしい点の短い候補リスト（最大数件）\n"
            "- stance_update_signal：対人距離・警戒・丁寧さの調整方向（none=維持、shift=変化兆候、strengthen=警戒強め、soften=緩和など）\n"
            "- confidence：推定（cg_gap/stance信号）の確信度（0-1）\n\n"
            "【出力フォーマット】\n"
            "{\n"
            '"outputs": {"cg_gap_candidates": ["短い候補"], \n'
            '"stance_update_signal": "none|shift|strengthen|soften|other"}, \n'
            '"confidence": 0-1\n'
            "}"
        ),
        (
            "ユーザー入力を用いて人物モデル/共通基盤の欠落予測の分類器してください。\n"
            f"- user_input: {user_input}\n\n"
            "- history:\n"
            f"{format_wm_messages(wm_messages, limit=8)}\n\n"
            "- user_model:\n"
            f"{format_user_model(user_model)}\n\n"
            "- common_ground: 共有前提の一覧。欠落候補の推定に使う。\n"
            f"{format_common_ground(common_ground)}\n\n"
            "- unresolved_points: 未解決の論点。ギャップ候補の推定に使う。\n"
            f"{format_unresolved_points(unresolved_points)}\n\n"
            "- observation: 直近の反応分類。スタンス変化推定に使う。\n"
            f"{format_observation(observation)}\n\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
        "L3",
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = coerce_float(payload.get("confidence", 0.0), 0.0)
    return pred


async def _predict_l4(
    turn_id: int,
    user_input: str,
    wm_messages: list[dict],
    user_model: UserModel,
    joint_context: JointContext,
    metrics_prev: Metrics,
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
            "あなたは枠組み再設計/長期価値予測のトリガ判定器\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
            "ユーザー入力を用いて枠組み再設計/長期価値予測のトリガ判定を行ってください\n\n"
            "【入力フィールド】\n"
            "- joint_context: 現在の枠組み/役割/規範。frame仮説の推定に使う。\n"
            "- metrics_prev: 前回指標(PE,ΔI/ΔG/ΔJ等)。トリガ判定に使う。\n"
            "【出力フィールド】\n"
            "- l4_trigger_score：枠組みのズレや停滞が強く、メタ的な再調整（deep_frame）が必要な度合い（0-1）\n"
            "- frame_hypothesis：現在（または望ましい）会話枠組みの推定（explore/decide/execute/reflect/vent）\n"
            "- confidence：このL4推定（trigger/frame仮説）の確信度（0-1）\n\n"
            "【出力フォーマット】\n"
            "{\n"
            '"outputs": {"l4_trigger_score": 0-1, \n'
            '"frame_hypothesis": "explore|decide|execute|reflect|vent"}, \n'
            '"confidence": 0-1\n'
            "}"
        ),
        (
            "ユーザー入力を用いて枠組み再設計/長期価値予測のトリガ判定を行ってください\n"
            f"- user_input: {user_input}\n"
            "- history:\n"
            f"{format_wm_messages(wm_messages, limit=8)}\n\n"
            "- user_model:\n"
            f"{format_user_model(user_model)}\n\n"
            "- joint_context: 現在の枠組み/役割/規範。frame仮説の推定に使う。\n"
            f"{format_joint_context(joint_context)}\n"
            "- metrics_prev: 前回指標(PE,ΔI/ΔG/ΔJ等)。トリガ判定に使う。\n"
            f"{format_metrics(metrics_prev)}\n\n"
            "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        ),
        {"outputs": fallback_outputs, "confidence": 0.0},
        "L4",
    )
    pred["outputs"] = _merge_outputs(fallback_outputs, payload.get("outputs", {}))
    pred["confidence"] = coerce_float(payload.get("confidence", 0.0), 0.0)
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
            _predict_l1(inp.turn_id, inp.user_input, inp.wm_messages, deps.small_llm),
            _predict_l2(
                inp.turn_id,
                inp.user_input,
                inp.wm_messages,
                inp.user_model,
                deps.small_llm,
            ),
            _predict_l3(
                inp.turn_id,
                inp.user_input,
                inp.wm_messages,
                inp.user_model,
                inp.common_ground,
                inp.unresolved_points,
                inp.observation,
                deps.small_llm,
            ),
            _predict_l4(
                inp.turn_id,
                inp.user_input,
                inp.wm_messages,
                inp.user_model,
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
