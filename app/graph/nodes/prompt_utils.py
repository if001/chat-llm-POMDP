from __future__ import annotations

from typing import Any
from app.models.state import (
    AffectiveState,
    AssumptionItem,
    JointContext,
    UnresolvedItem,
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_list(values: list[Any], empty: str = "なし") -> str:
    if not values:
        return empty
    return " / ".join(_as_text(v) for v in values)


def format_wm_messages(wm_messages: list[dict[str, Any]], limit: int = 6) -> str:
    if not wm_messages:
        return "なし"
    lines = []
    for idx, msg in enumerate(wm_messages[-limit:], 1):
        role = _as_text(msg.get("role", ""))
        content = _as_text(msg.get("content", ""))
        lines.append(f"{idx}. role={role} / content={content}")
    return "\n".join(lines)


def format_joint_context(joint_context: JointContext) -> str:
    norms = joint_context.get("norms", {})
    return "\n".join(
        [
            "norms:",
            f"- question_budget: {_as_text(norms.get('question_budget', ''))}",
            f"- max_response_length: {_as_text(norms.get('max_response_length', ''))}",
            f"- optionality_required: {_as_text(norms.get('optionality_required', ''))}",
            f"- summarize_before_advice: {_as_text(norms.get('summarize_before_advice', ''))}",
            f"- stance_sensitive: {_as_text(norms.get('stance_sensitive', ''))}",
        ]
    )


def format_common_ground(
    common_ground: dict[str, list[AssumptionItem]], limit: int = 5
) -> str:
    assumptions = common_ground.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = []
    lines = [f"assumptions_count: {len(assumptions)}"]
    for idx, item in enumerate(assumptions[:limit], 1):
        if not isinstance(item, dict):
            lines.append(f"{idx}. {_as_text(item)}")
            continue
        lines.append(
            " / ".join(
                [
                    f"{idx}. id={_as_text(item.get('id', ''))}",
                    f"proposition={_as_text(item.get('proposition', ''))}",
                    f"scope={_as_text(item.get('scope', ''))}",
                    f"confidence={_as_text(item.get('confidence', ''))}",
                    f"last_updated_turn={_as_text(item.get('last_updated_turn', ''))}",
                ]
            )
        )
    if len(assumptions) > limit:
        lines.append(f"... {len(assumptions) - limit} more")
    return "\n".join(lines)


def format_unresolved_points(
    unresolved_points: list[UnresolvedItem], limit: int = 5
) -> str:
    if not unresolved_points:
        return "なし"
    lines = [f"unresolved_count: {len(unresolved_points)}"]
    for idx, item in enumerate(unresolved_points[:limit], 1):
        if not isinstance(item, dict):
            lines.append(f"{idx}. {_as_text(item)}")
            continue
        lines.append(
            " / ".join(
                [
                    f"{idx}. id={_as_text(item.get('id', ''))}",
                    f"question={_as_text(item.get('question', ''))}",
                    f"kind={_as_text(item.get('kind', ''))}",
                    f"priority={_as_text(item.get('priority', ''))}",
                    f"asked={_as_text(item.get('asked', ''))}",
                    f"answered={_as_text(item.get('answered', ''))}",
                ]
            )
        )
    if len(unresolved_points) > limit:
        lines.append(f"... {len(unresolved_points) - limit} more")
    return "\n".join(lines)


def format_observation(observation: dict[str, Any]) -> str:
    events = observation.get("events", {})
    return "\n".join(
        [
            f"reaction_type: {_as_text(observation.get('reaction_type', ''))}",
            f"ack_type: {_as_text(observation.get('ack_type', ''))}",
            "events:",
            f"- E_correct: {_as_text(events.get('E_correct', ''))}",
            f"- E_refuse: {_as_text(events.get('E_refuse', ''))}",
            f"- E_clarify: {_as_text(events.get('E_clarify', ''))}",
            f"- E_miss: {_as_text(events.get('E_miss', ''))}",
            f"- E_frame_break: {_as_text(events.get('E_frame_break', ''))}",
            f"- E_overstep: {_as_text(events.get('E_overstep', ''))}",
            f"confidence: {_as_text(observation.get('confidence', ''))}",
        ]
    )


def format_metrics(metrics: dict[str, Any]) -> str:
    sources = metrics.get("sources_used", {})
    return "\n".join(
        [
            f"prediction_error: {_as_text(metrics.get('prediction_error', ''))}",
            f"delta_I: {_as_text(metrics.get('delta_I', ''))}",
            f"delta_G: {_as_text(metrics.get('delta_G', ''))}",
            f"delta_J: {_as_text(metrics.get('delta_J', ''))}",
            f"risk: {_as_text(metrics.get('risk', ''))}",
            f"cost_user: {_as_text(metrics.get('cost_user', ''))}",
            f"cost_agent: {_as_text(metrics.get('cost_agent', ''))}",
            f"sources_used.memory: {_as_text(sources.get('memory', ''))}",
            f"sources_used.web: {_as_text(sources.get('web', ''))}",
        ]
    )


def format_predictions(predictions: dict[str, Any]) -> str:
    def _get_outputs(level: str) -> tuple[dict[str, Any], dict[str, Any]]:
        pred = predictions.get(level, {})
        outputs = pred.get("outputs", {}) if isinstance(pred, dict) else {}
        return pred if isinstance(pred, dict) else {}, outputs if isinstance(
            outputs, dict
        ) else {}

    l0, l0_out = _get_outputs("L0")
    l1, l1_out = _get_outputs("L1")
    l2, l2_out = _get_outputs("L2")
    l3, l3_out = _get_outputs("L3")
    l4, l4_out = _get_outputs("L4")
    features = (
        l0_out.get("features", {})
        if isinstance(l0_out.get("features", {}), dict)
        else {}
    )
    cg_gaps = l3_out.get("cg_gap_candidates", [])
    return "\n".join(
        [
            "L0:",
            f"- style_fit: {_as_text(l0_out.get('style_fit', ''))}",
            f"- turn_pressure: {_as_text(l0_out.get('turn_pressure', ''))}",
            f"- features.char_len: {_as_text(features.get('char_len', ''))}",
            f"- features.question_mark_count: {_as_text(features.get('question_mark_count', ''))}",
            f"- confidence: {_as_text(l0.get('confidence', ''))}",
            "L1:",
            f"- speech_act: {_as_text(l1_out.get('speech_act', ''))}",
            f"- grounding_need: {_as_text(l1_out.get('grounding_need', ''))}",
            f"- repair_need: {_as_text(l1_out.get('repair_need', ''))}",
            f"- confidence: {_as_text(l1.get('confidence', ''))}",
            "L2:",
            f"- local_intent: {_as_text(l2_out.get('local_intent', ''))}",
            f"- U_semantic: {_as_text(l2_out.get('U_semantic', ''))}",
            f"- U_epistemic: {_as_text(l2_out.get('U_epistemic', ''))}",
            f"- U_social: {_as_text(l2_out.get('U_social', ''))}",
            f"- need_question_design: {_as_text(l2_out.get('need_question_design', ''))}",
            f"- confidence: {_as_text(l2.get('confidence', ''))}",
            "L3:",
            f"- cg_gap_candidates: {_format_list(cg_gaps)}",
            f"- stance_update_signal: {_as_text(l3_out.get('stance_update_signal', ''))}",
            f"- confidence: {_as_text(l3.get('confidence', ''))}",
            "L4:",
            f"- l4_trigger_score: {_as_text(l4_out.get('l4_trigger_score', ''))}",
            f"- frame_hypothesis: {_as_text(l4_out.get('frame_hypothesis', ''))}",
            f"- confidence: {_as_text(l4.get('confidence', ''))}",
        ]
    )


def format_deep_decision(deep_decision: dict[str, Any]) -> str:
    repair_plan = deep_decision.get("repair_plan", {})
    chain = deep_decision.get("deep_chain", {})
    return "\n".join(
        [
            f"reason: {_as_text(deep_decision.get('reason', ''))}",
            "repair_plan:",
            f"- strategy: {_as_text(repair_plan.get('strategy', ''))}",
            f"- questions: {_format_list(repair_plan.get('questions', []))}",
            f"- optionality: {_as_text(repair_plan.get('optionality', ''))}",
            "deep_chain:",
            f"- plan: {_format_list(chain.get('plan', []))}",
            f"- executed: {_format_list(chain.get('executed', []))}",
            f"- stop_reason: {_as_text(chain.get('stop_reason', ''))}",
        ]
    )


def format_affective_state(affective_state: AffectiveState) -> str:
    episode = affective_state.get("episode", {})
    mood = affective_state.get("mood", {})
    regulation = affective_state.get("regulation_bias", {})
    return "\n".join(
        [
            "episode:",
            f"- valence: {_as_text(episode.get('valence', ''))}",
            f"- arousal: {_as_text(episode.get('arousal', ''))}",
            f"- confidence: {_as_text(episode.get('confidence', ''))}",
            "mood:",
            f"- valence: {_as_text(mood.get('valence', ''))}",
            f"- arousal: {_as_text(mood.get('arousal', ''))}",
            f"interpersonal_stance: {_as_text(affective_state.get('interpersonal_stance', ''))}",
            "regulation_bias:",
            f"- mode: {_as_text(regulation.get('mode', ''))}",
            f"- confidence: {_as_text(regulation.get('confidence', ''))}",
        ]
    )


def format_epistemic_state(epistemic_state: dict[str, Any]) -> str:
    uncertainties = epistemic_state.get("uncertainties", {})
    high_stakes = epistemic_state.get("high_stakes", {})
    return "\n".join(
        [
            "uncertainties:",
            f"- semantic: {_as_text(uncertainties.get('semantic', ''))}",
            f"- epistemic: {_as_text(uncertainties.get('epistemic', ''))}",
            f"- social: {_as_text(uncertainties.get('social', ''))}",
            f"- confidence: {_as_text(uncertainties.get('confidence', ''))}",
            "high_stakes:",
            f"- value: {_as_text(high_stakes.get('value', ''))}",
            f"- categories: {_format_list(high_stakes.get('categories', []))}",
            f"- confidence: {_as_text(high_stakes.get('confidence', ''))}",
        ]
    )


def format_action(action: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"chosen_frame: {_as_text(action.get('chosen_frame', ''))}",
            f"chosen_role_leader: {_as_text(action.get('chosen_role_leader', ''))}",
            f"response_mode: {_as_text(action.get('response_mode', ''))}",
            f"questions_asked: {_as_text(action.get('questions_asked', ''))}",
            f"question_budget: {_as_text(action.get('question_budget', ''))}",
            f"confirm_questions: {_format_list(action.get('confirm_questions', []))}",
            f"did_memory_search: {_as_text(action.get('did_memory_search', ''))}",
            f"did_web_search: {_as_text(action.get('did_web_search', ''))}",
        ]
    )


def format_snippets(
    snippets: list[Any],
    label: str,
    limit: int = 3,
) -> str:
    if not snippets:
        return f"{label}: なし"
    lines = [f"{label}: {len(snippets)}件"]
    for idx, item in enumerate(snippets[:limit], 1):
        if isinstance(item, dict):
            text = _as_text(item.get("text") or item.get("content") or "")
            title = _as_text(item.get("title", ""))
            url = _as_text(item.get("url", ""))
            source = _as_text(item.get("source", ""))
            metadata = item.get("metadata", {})
            meta_keys = (
                ", ".join(_as_text(k) for k in metadata.keys())
                if isinstance(metadata, dict)
                else ""
            )
            lines.append(
                " / ".join(
                    [
                        f"{idx}. title={title}",
                        f"url={url}",
                        f"source={source}",
                        f"text={text}",
                        f"metadata_keys={meta_keys}",
                    ]
                )
            )
        else:
            lines.append(f"{idx}. {_as_text(item)}")
    if len(snippets) > limit:
        lines.append(f"... {len(snippets) - limit} more")
    return "\n".join(lines)
