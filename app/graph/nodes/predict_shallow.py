# app/graph/nodes/predict_shallow.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _predict_l0(inp: PredictShallowIn) -> PredictionCommon:
    pred = _base_prediction("L0", inp.turn_id)
    pred["outputs"] = {
        "style_fit": 0.5,
        "turn_pressure": 0.5,
        "features": {
            "char_len": len(inp.user_input),
            "question_mark_count": inp.user_input.count("?"),
        },
    }
    return pred


def _predict_l1(inp: PredictShallowIn) -> PredictionCommon:
    pred = _base_prediction("L1", inp.turn_id)
    pred["outputs"] = {
        "speech_act": "other",
        "grounding_need": 0.5,
        "repair_need": 0.0,
    }
    return pred


def _predict_l2(inp: PredictShallowIn) -> PredictionCommon:
    pred = _base_prediction("L2", inp.turn_id)
    pred["outputs"] = {
        "local_intent": "unknown",
        "U_semantic": 0.5,
        "U_epistemic": 0.5,
        "U_social": 0.5,
        "need_question_design": False,
    }
    return pred


def _predict_l3(inp: PredictShallowIn) -> PredictionCommon:
    pred = _base_prediction("L3", inp.turn_id)
    pred["outputs"] = {
        "cg_gap_candidates": [],
        "stance_update_signal": "none",
    }
    return pred


def _predict_l4(inp: PredictShallowIn) -> PredictionCommon:
    pred = _base_prediction("L4", inp.turn_id)
    pred["outputs"] = {
        "l4_trigger_score": 0.0,
        "frame_hypothesis": inp.joint_context.get("frame", "explore"),
    }
    return pred


def _uncertainties_from_l2(l2_prediction: PredictionCommon) -> EpistemicUncertainty:
    outputs = l2_prediction.get("outputs", {})
    return {
        "semantic": float(outputs.get("U_semantic", 0.5)),
        "epistemic": float(outputs.get("U_epistemic", 0.5)),
        "social": float(outputs.get("U_social", 0.5)),
        "confidence": float(l2_prediction.get("confidence", 0.0)),
    }


def make_predict_shallow_node():
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
        preds: dict[str, PredictionCommon] = {
            "L0": _predict_l0(inp),
            "L1": _predict_l1(inp),
            "L2": _predict_l2(inp),
            "L3": _predict_l3(inp),
            "L4": _predict_l4(inp),
        }

        uncertainties_now = _uncertainties_from_l2(preds["L2"])
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
