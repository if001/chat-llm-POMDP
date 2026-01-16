from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from copy import deepcopy
import json

from app.core.deps import Deps
from app.models.state import (
    Action,
    AgentState,
    AssumptionItem,
    JointContext,
    Observation,
    UnresolvedItem,
)
from app.graph.nodes.prompt_utils import (
    format_common_ground,
    format_unresolved_points,
    format_wm_messages,
)
from app.ports.llm import LLMPort
from app.graph.utils.write import a_stream_writer
from app.graph.utils import utils


@dataclass(frozen=True)
class UpdateJointActionIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict[str, Any]]
    common_ground: dict[str, list[AssumptionItem]]
    unresolved_points: list[UnresolvedItem]
    joint_context: JointContext
    observation: Observation
    prev_action: Action
    action: Action


@dataclass(frozen=True)
class UpdateJointActionOut:
    status: str
    common_ground: dict
    unresolved_points: list[dict]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum())


def _is_similar(a: str, b: str) -> bool:
    na = _normalize(a)
    nb = _normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    overlap = sum(1 for ch in na if ch in nb)
    return overlap / max(len(na), 1) >= 0.6


async def _extract_candidates(
    small_llm: LLMPort,
    user_input: str,
    wm_messages: list[dict[str, Any]],
    common_ground: dict[str, list[AssumptionItem]],
    unresolved_points: list[UnresolvedItem],
) -> dict[str, Any]:
    prompt = (
        "あなたはcommon_ground/unresolved_pointsの抽出器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "【入力フィールド】\n"
        "- user_input: ユーザー入力\n"
        "- history: 直近の会話履歴\n"
        "- common_ground: 共有前提の一覧。欠落候補の推定\n"
        "- unresolved_points: 未解決の論点。ギャップ候補の推定\n\n"
        "【出力フィールド】\n"
        "common_ground：共有できている前提\n"
        "scope: 継続的（好み/習慣/恒常条件）なら global、今回だけ（期限/要件/状況）なら current_task\n"
        "unresolved_points：共有できていない穴（埋めるべき未確定点のリスト）\n\n"
        "【出力フォーマット】\n"
        "{\n"
        '"common_ground": [{"proposition": "短い命題", \n'
        '"scope": "global|current_task", "confidence": 0-1}], \n'
        '"unresolved_points": [{"question": "短い質問", \n'
        "}\n"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "common_ground/unresolved_pointsの抽出を開始してください。\n"
                        f"- user_input: {user_input}\n"
                        "- history: 直近の会話履歴(最大6件)。\n"
                        f"{format_wm_messages(wm_messages, limit=6)}\n"
                        "- common_ground: 既存の共有前提。重複回避用\n"
                        f"{format_common_ground(common_ground)}\n"
                        "- unresolved_points: 既存の未解決点。重複回避用\n"
                        f"{format_unresolved_points(unresolved_points)}"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    r = utils.parse_llm_response(result)
    if len(r) == 0:
        print("_extract_candidates( fallback")
    return r


async def _detect_resolved(
    small_llm: LLMPort,
    user_input: str,
    unresolved_points: list[UnresolvedItem],
    prev_action: Action,
    observation: Observation,
) -> dict[str, Any]:
    prompt = (
        "あなたは未解決点の解消判定器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "【入力フィールド】\n"
        "- user_input: ユーザー入力\n"
        "- unresolved_points: 未解決の論点。ギャップ候補の推定。asked=trueのものを中心に解消判定。\n"
        "- prev_action.confirm_questions: 以前の確認用の質問\n"
        "- ack_type: アシスタントの理解や要約に対するユーザーの同意の明示度（明示的肯定／暗黙的肯定／混在／否定／評価不能）\n\n"
        "【出力フィールド】\n"
        "- resolved_ids: 未解決のunresolved_pointsのid配列\n\n"
        "【出力フォーマット】\n"
        "{\n"
        '"resolved_ids": ["id1", "id2"]\n'
        "}\n"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "未解決点の解消判定を行ってください\n"
                        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
                        f"- user_input: {user_input}\n"
                        "- unresolved_points:\n"
                        f"{format_unresolved_points(unresolved_points)}\n"
                        f"- prev_action.confirm_questions: {prev_action.get('confirm_questions', [])}\n"
                        f"- ack_type: {observation.get('ack_type', '')}"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    r = utils.parse_llm_response(result)
    if len(r) == 0:
        print("_detect_resolved fallback")
    return r


async def _summarize_resolved(
    small_llm: LLMPort,
    user_input: str,
    resolved_questions: list[str],
) -> dict[str, Any]:
    prompt = (
        "あなたは要約器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
        "【入力フィールド】\n"
        "- user_input: ユーザー入力\n"
        "- resolved_questions: 解決済の質問\n\n"
        "【出力フィールド】\n"
        "assumptions: 解決済の質問の要約。解消なしなら空配列。\n"
        "scope: 継続的（好み/習慣/恒常条件）なら global、今回だけ（期限/要件/状況）なら current_task\n\n"
        "【出力フォーマット】\n"
        "{\n"
        '"assumptions": [{"proposition": "短い命題", \n'
        '"scope": "global|current_task", "confidence": 0-1}]\n'
        "}\n"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "解消済み未解決点を判定してください\n"
                        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
                        f"- user_input: {user_input}\n"
                        f"- resolved_questions: {resolved_questions}\n"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    r = utils.parse_llm_response(result)
    if len(r) == 0:
        print("_summarize_resolved fallback")
    return r


def _merge_common_ground(
    existing: list[dict],
    candidates: list[dict],
    turn_id: int,
) -> list[dict]:
    merged = deepcopy(existing)
    for cand in candidates:
        prop = str(cand.get("proposition", "")).strip()
        if not prop:
            continue
        scope = str(cand.get("scope", "current_task")).strip() or "current_task"
        confidence = _clamp(float(cand.get("confidence", 0.6)))
        matched = False
        for item in merged:
            if _is_similar(prop, str(item.get("proposition", ""))):
                item["confidence"] = _clamp(
                    max(float(item.get("confidence", 0.0)), confidence)
                )
                item["last_updated_turn"] = turn_id
                item.setdefault("evidence_ids", []).append(f"turn:{turn_id}")
                matched = True
                break
        if matched:
            continue
        merged.append(
            {
                "id": f"cg_{turn_id}_{len(merged)}",
                "proposition": prop,
                "scope": scope,
                "confidence": confidence,
                "evidence_ids": [f"turn:{turn_id}"],
                "last_updated_turn": turn_id,
            }
        )
    return merged


def _merge_unresolved_points(
    existing: list[dict],
    candidates: list[dict],
    turn_id: int,
) -> list[dict]:
    merged = deepcopy(existing)
    for cand in candidates:
        question = str(cand.get("question", "")).strip()
        if not question:
            continue
        kind = str(cand.get("kind", "semantic")).strip() or "semantic"
        priority = _clamp(float(cand.get("priority", 0.6)))
        matched = False
        for item in merged:
            if _is_similar(question, str(item.get("question", ""))):
                item["priority"] = max(float(item.get("priority", 0.0)), priority)
                item["last_touched_turn"] = turn_id
                matched = True
                break
        if matched:
            continue
        merged.append(
            {
                "id": f"u_{turn_id}_{len(merged)}",
                "question": question,
                "kind": kind,
                "priority": priority,
                "asked": False,
                "answered": False,
                "created_turn": turn_id,
                "last_touched_turn": turn_id,
            }
        )
    return merged


def _apply_asked_flags(
    unresolved_points: list[dict],
    asked_questions: list[str],
    turn_id: int,
) -> list[dict]:
    if not asked_questions:
        return unresolved_points
    updated = deepcopy(unresolved_points)
    for item in updated:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        for asked in asked_questions:
            if _is_similar(q, str(asked)):
                item["asked"] = True
                item["last_touched_turn"] = turn_id
                break
    return updated


def _apply_unresolved_priority_decay(
    unresolved_points: list[dict],
    current_frame: str,
) -> list[dict]:
    updated = deepcopy(unresolved_points)
    for item in updated:
        priority = float(item.get("priority", 0.5))
        if current_frame == "vent" and item.get("kind") == "epistemic":
            priority *= 0.9
        item["priority"] = _clamp(priority)
    return updated


def _apply_unanswered_boost(
    unresolved_points: list[dict],
) -> list[dict]:
    updated = deepcopy(unresolved_points)
    for item in updated:
        if item.get("asked") and not item.get("answered"):
            item["priority"] = _clamp(float(item.get("priority", 0.5)) + 0.05)
    return updated


def _promote_resolved(
    unresolved_points: list[dict],
    resolved_ids: set[str],
) -> tuple[list[dict], list[str]]:
    remaining = []
    resolved_questions = []
    for item in unresolved_points:
        if item.get("id") in resolved_ids:
            item["answered"] = True
            resolved_questions.append(str(item.get("question", "")))
            continue
        remaining.append(item)
    return remaining, resolved_questions


def _trim_common_ground(items: list[dict], max_items: int = 20) -> list[dict]:
    if len(items) <= max_items:
        return items
    return sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[
        :max_items
    ]


def _trim_unresolved(items: list[dict], max_items: int = 5) -> list[dict]:
    if len(items) <= max_items:
        return items
    return sorted(items, key=lambda x: float(x.get("priority", 0.0)), reverse=True)[
        :max_items
    ]


def make_update_joint_action_node(deps: Deps):
    async def inner(inp: UpdateJointActionIn) -> UpdateJointActionOut:
        """
        何をするか:
        - common_ground/unresolved_points を更新（抽出→マージ→解消→昇格）
        - 対話制御の中間表現として、次ターンの予測/repair/枠組み判断に使える形にする
        """
        common_ground = dict(inp.common_ground)
        unresolved_points = list(inp.unresolved_points)

        payload = await _extract_candidates(
            deps.small_llm,
            inp.user_input,
            inp.wm_messages,
            common_ground,
            unresolved_points,
        )

        candidates_common = payload.get("common_ground", [])
        candidates_unresolved = payload.get("unresolved_points", [])
        if not isinstance(candidates_common, list):
            candidates_common = []
        if not isinstance(candidates_unresolved, list):
            candidates_unresolved = []

        merged_common = _merge_common_ground(
            common_ground.get("assumptions", []), candidates_common, inp.turn_id
        )
        merged_unresolved = _merge_unresolved_points(
            unresolved_points, candidates_unresolved, inp.turn_id
        )

        prev_questions = inp.prev_action.get("confirm_questions", [])
        merged_unresolved = _apply_asked_flags(
            merged_unresolved, prev_questions, inp.turn_id
        )
        merged_unresolved = _apply_unresolved_priority_decay(
            merged_unresolved, str(inp.joint_context.get("frame", "explore"))
        )

        asked_unresolved = [
            item
            for item in merged_unresolved
            if item.get("asked") and not item.get("answered")
        ]
        resolved_ids: set[str] = set()
        if asked_unresolved:
            resolved_payload = await _detect_resolved(
                deps.small_llm,
                inp.user_input,
                asked_unresolved,
                inp.prev_action,
                inp.observation,
            )
            ids = resolved_payload.get("resolved_ids", [])
            if isinstance(ids, list):
                resolved_ids = {str(i) for i in ids}

        merged_unresolved, resolved_questions = _promote_resolved(
            merged_unresolved, resolved_ids
        )

        if resolved_questions:
            summarized = await _summarize_resolved(
                deps.small_llm, inp.user_input, resolved_questions
            )
            assumptions = summarized.get("assumptions", [])
            if isinstance(assumptions, list):
                merged_common = _merge_common_ground(
                    merged_common, assumptions, inp.turn_id
                )

        asked_questions = inp.action.get("confirm_questions", [])
        merged_unresolved = _apply_asked_flags(
            merged_unresolved, asked_questions, inp.turn_id
        )

        merged_unresolved = _apply_unanswered_boost(merged_unresolved)

        common_ground["assumptions"] = _trim_common_ground(merged_common)
        merged_unresolved = _trim_unresolved(merged_unresolved)

        return UpdateJointActionOut(
            status="update_joint_action:ok",
            common_ground=common_ground,
            unresolved_points=merged_unresolved,
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
            UpdateJointActionIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
                joint_context=state["joint_context"],
                observation=state["observation"],
                prev_action=state["last_turn"]["prev_action"],
                action=state["action"],
            )
        )
        return {
            "common_ground": out.common_ground,
            "unresolved_points": out.unresolved_points,
        }

    return node
