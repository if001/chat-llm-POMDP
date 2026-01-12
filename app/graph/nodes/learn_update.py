# app/graph/nodes/learn_update.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, JointContext, PolicyState, UserAttribute, UserModel
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


async def _extract_user_model_updates(
    small_llm: LLMPort,
    user_input: str,
    wm_messages: list[dict],
) -> dict[str, Any]:
    prompt = (
        "Extract user profile updates from the conversation. "
        "Return JSON with optional keys: "
        "basic (dict of field-> {value, confidence}), "
        "preferences (dict), tendencies (dict), topics (dict), "
        "taboos (list of {value, confidence}). "
        "Use confidence 0-1. Only include fields that are explicitly supported."
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"user_input: {user_input}\nwm_messages: {wm_messages[-6:]}",
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
