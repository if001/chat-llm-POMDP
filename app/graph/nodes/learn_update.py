# app/graph/nodes/learn_update.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, JointContext, PolicyState, UserAttribute, UserModel
from app.graph.nodes.prompt_utils import format_wm_messages
from app.ports.llm import LLMPort


@dataclass(frozen=True)
class LearnUpdateIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict]
    joint_context: JointContext
    user_model: UserModel

    # 観測/予測/指標を材料に、閾値やnormsを更新する
    observation: dict
    predictions: dict
    metrics: dict
    deep_decision: dict
    action: dict
    response: dict

    policy: PolicyState
    epistemic_uncertainties_now: dict
    unresolved_count_now: int


@dataclass(frozen=True)
class LearnUpdateOut:
    status: str
    joint_context: JointContext
    user_model: UserModel
    policy: PolicyState
    last_turn_patch: dict


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


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _update_attribute(
    current: UserAttribute | None,
    value: str,
    confidence: float,
    evidence: list[str],
    turn_id: int,
) -> UserAttribute:
    if current:
        new_conf = 0.8 * current["confidence"] + 0.2 * confidence
        merged_evidence = list(dict.fromkeys(current["evidence"] + evidence))
    else:
        new_conf = confidence
        merged_evidence = evidence
    return {
        "value": value,
        "confidence": new_conf,
        "evidence": merged_evidence,
        "last_updated_turn": turn_id,
    }


def _calc_value(metrics: dict) -> float:
    return (
        float(metrics.get("delta_I", 0.0))
        + float(metrics.get("delta_G", 0.0))
        + float(metrics.get("delta_J", 0.0))
        - 1.2 * float(metrics.get("risk", 0.0))
        - 0.8 * float(metrics.get("cost_user", 0.0))
    )


def _ema(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return value
    return (1.0 - alpha) * prev + alpha * value


def _infer_repair_type(action: dict, deep_decision: dict) -> str | None:
    response_mode = action.get("response_mode", "")
    reason = deep_decision.get("reason", "")
    if response_mode == "offer_options":
        return "offer_options"
    if response_mode in {"summarize"}:
        return "summarize_confirm"
    if response_mode in {"clarify", "ask", "ask_open", "confirm"}:
        return "intent_check"
    if response_mode in {"repair"}:
        return "rephrase"
    if response_mode == "meta_frame" or reason == "frame_collapse":
        return "meta_frame"
    return None


def _update_repair_stats(
    repair_stats: dict[str, dict[str, float]],
    repair_type: str,
    success: bool,
) -> dict[str, dict[str, float]]:
    updated = dict(repair_stats)
    stats = dict(updated.get(repair_type, {"alpha": 2.0, "beta": 2.0}))
    if success:
        stats["alpha"] = float(stats.get("alpha", 2.0)) + 1.0
    else:
        stats["beta"] = float(stats.get("beta", 2.0)) + 1.0
    updated[repair_type] = stats
    return updated


def _evaluate_repair_success(
    baseline: dict,
    metrics: dict,
    observation: dict,
) -> bool | None:
    base_pe = float(baseline.get("baseline_PE", 0.0))
    base_dg = float(baseline.get("baseline_delta_G", 0.0))
    pe = float(metrics.get("prediction_error", 0.0))
    delta_g = float(metrics.get("delta_G", 0.0))
    ack = observation.get("ack_type", "")
    reaction = observation.get("reaction_type", "")
    events = observation.get("events", {})

    if pe <= base_pe - 0.2:
        return True
    if ack in {"explicit_yes", "implicit_yes"} and delta_g > base_dg:
        return True
    if pe >= base_pe + 0.1:
        return False
    if reaction in {"refuse", "topic_shift"}:
        return False
    if events.get("E_frame_break", 0) == 1 or events.get("E_refuse", 0) == 1:
        return False
    return None


async def _extract_user_model_updates(
    small_llm: LLMPort,
    user_input: str,
    wm_messages: list[dict],
) -> dict[str, Any]:
    prompt = (
        "あなたはユーザー属性更新の抽出器。"
        "入力はユーザー発話と直近の会話履歴で、state.user_model更新に使う。"
        "明示的に根拠がある情報のみを抽出し、推測は入れない。"
        "出力はJSONのみ。"
        "出力フォーマット: {"
        '"basic": {"field": {"value": "文字列", "confidence": 0-1}}, '
        '"preferences": {"field": {"value": "文字列", "confidence": 0-1}}, '
        '"tendencies": {"field": {"value": "文字列", "confidence": 0-1}}, '
        '"topics": {"field": {"value": "文字列", "confidence": 0-1}}, '
        '"taboos": [{"value": "文字列", "confidence": 0-1}]'
        "}。必要なキーだけ出力。"
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
                        "- wm_messages: 直近の会話履歴(最大6件)。明示的根拠の確認に使う。\n"
                        f"{format_wm_messages(wm_messages, limit=6)}"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    return _parse_json(_get_content(result))


def _apply_user_model_updates(
    current: UserModel,
    updates: dict[str, Any],
    evidence_text: str,
    turn_id: int,
) -> UserModel:
    updated = dict(current)
    evidence = [evidence_text] if evidence_text else []
    for key in ["basic", "preferences", "tendencies", "topics"]:
        section_updates = updates.get(key, {})
        if not isinstance(section_updates, dict):
            continue
        section = dict(updated.get(key, {}))
        for field, entry in section_updates.items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            confidence = _coerce_float(entry.get("confidence", 0.6), 0.6)
            if confidence < 0.4:
                continue
            current_attr = section.get(field)
            section[field] = _update_attribute(
                current_attr,
                str(entry.get("value", "")),
                confidence,
                evidence,
                turn_id,
            )
        updated[key] = section
    taboos_updates = updates.get("taboos", [])
    if isinstance(taboos_updates, list):
        taboos = list(updated.get("taboos", []))
        for entry in taboos_updates:
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            confidence = _coerce_float(entry.get("confidence", 0.6), 0.6)
            if confidence < 0.4:
                continue
            taboos.append(
                _update_attribute(
                    None,
                    str(entry.get("value", "")),
                    confidence,
                    evidence,
                    turn_id,
                )
            )
        updated["taboos"] = taboos
    updated["last_updated_turn"] = turn_id
    return updated


def make_learn_update_node(deps: Deps):
    async def inner(inp: LearnUpdateIn) -> LearnUpdateOut:
        """
        何をするか:
        - オンライン適応（学習）を行う（重み学習ではなく状態更新）
          - theta_deep の更新（rolling_V/rolling_PE等のEMAを導入するならpolicyに保持）
          - norms（question_budget等）の更新
          - user_model（traits/topic_preferences/taboos等）の更新
          - deep_history の更新
        - 次ターンの観測/差分計算のために last_turn を更新する
          - prev_assistant_text / prev_action / prev_response_meta
          - prev_uncertainties / prev_unresolved_count
        """
        policy = dict(inp.policy)
        deep_chain = inp.deep_decision.get("deep_chain", {})
        executed = list(deep_chain.get("executed", []))
        policy["deep_history"] = list(policy.get("deep_history", [])) + executed

        alpha = 0.2
        prev_rolling = dict(policy.get("rolling", {}))
        rolling = dict(prev_rolling)
        rolling["PE_total"] = _ema(
            prev_rolling.get("PE_total"),
            float(inp.metrics.get("prediction_error", 0.0)),
            alpha,
        )
        rolling["V"] = _ema(prev_rolling.get("V"), _calc_value(inp.metrics), alpha)
        rolling["cost_user"] = _ema(
            prev_rolling.get("cost_user"),
            float(inp.metrics.get("cost_user", 0.0)),
            alpha,
        )
        rolling["delta_J"] = _ema(
            prev_rolling.get("delta_J"),
            float(inp.metrics.get("delta_J", 0.0)),
            alpha,
        )
        rolling["delta_G"] = _ema(
            prev_rolling.get("delta_G"),
            float(inp.metrics.get("delta_G", 0.0)),
            alpha,
        )
        policy["rolling"] = rolling

        theta = float(policy.get("theta_deep", 1.2))
        prev_pe = prev_rolling.get("PE_total")
        prev_v = prev_rolling.get("V")
        deep_used = bool(executed)
        if deep_used and prev_v is not None and rolling.get("V") is not None:
            if rolling["V"] < prev_v - 0.05:
                theta += 0.05
        if not deep_used and prev_pe is not None and rolling.get("PE_total") is not None:
            if rolling["PE_total"] > prev_pe + 0.05:
                theta -= 0.05
        policy["theta_deep"] = max(0.6, min(2.0, theta))

        repair_stats = dict(policy.get("repair_stats", {}))
        pending_evals = list(policy.get("pending_evals", []))
        updated_pending: list[dict[str, Any]] = []
        for pending in pending_evals:
            if pending.get("kind") != "repair":
                updated_pending.append(pending)
                continue
            repair_type = pending.get("repair_type")
            if not repair_type:
                continue
            ttl = int(pending.get("ttl", 1))
            outcome = _evaluate_repair_success(pending, inp.metrics, inp.observation)
            if outcome is True:
                repair_stats = _update_repair_stats(repair_stats, repair_type, True)
                continue
            if outcome is False:
                repair_stats = _update_repair_stats(repair_stats, repair_type, False)
                continue
            if ttl > 1:
                pending["ttl"] = ttl - 1
                updated_pending.append(pending)
        policy["repair_stats"] = repair_stats
        policy["pending_evals"] = updated_pending

        repair_type = _infer_repair_type(inp.action, inp.deep_decision)
        if repair_type:
            policy["pending_evals"] = list(policy.get("pending_evals", [])) + [
                {
                    "kind": "repair",
                    "turn_id": inp.turn_id,
                    "repair_type": repair_type,
                    "baseline_PE": float(inp.metrics.get("prediction_error", 0.0)),
                    "baseline_delta_G": float(inp.metrics.get("delta_G", 0.0)),
                    "ttl": 2,
                }
            ]

        updates = await _extract_user_model_updates(
            deps.small_llm, inp.user_input, inp.wm_messages
        )
        user_model = _apply_user_model_updates(
            inp.user_model, updates, inp.user_input, inp.turn_id
        )

        updates = await _extract_user_model_updates(
            deps.small_llm, inp.user_input, inp.wm_messages
        )
        user_model = _apply_user_model_updates(
            inp.user_model, updates, inp.user_input, inp.turn_id
        )

        last_turn_patch = {
            "prev_assistant_text": inp.response.get("final_text", ""),
            "prev_action": dict(inp.action),
            "prev_response_meta": dict(inp.response.get("meta", {})),
            "prev_uncertainties": dict(inp.epistemic_uncertainties_now),
            "prev_unresolved_count": int(inp.unresolved_count_now),
            "resolved_count_last_turn": 0,
        }
        return LearnUpdateOut(
            status="learn_update:ok",
            joint_context=inp.joint_context,
            user_model=user_model,
            policy=policy,  # type: ignore[arg-type]
            last_turn_patch=last_turn_patch,
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
            LearnUpdateIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                joint_context=state["joint_context"],
                user_model=state["user_model"],
                observation=state["observation"],
                predictions=state["predictions"],
                metrics=state["metrics"],
                deep_decision=state["deep_decision"],
                action=state["action"],
                response=state["response"],
                policy=state["policy"],
                epistemic_uncertainties_now=state["epistemic_state"]["uncertainties"],
                unresolved_count_now=len(state["unresolved_points"]),
            )
        )

        # last_turn を部分更新
        last_turn = dict(state["last_turn"])
        last_turn.update(out.last_turn_patch)

        return {
            "joint_context": out.joint_context,
            "user_model": out.user_model,
            "policy": out.policy,
            "last_turn": last_turn,
        }

    return node
