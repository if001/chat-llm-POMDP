# app/models/types.py
from __future__ import annotations
from typing import Literal, TypedDict


# === enums ===
Frame = Literal["explore", "decide", "execute", "reflect", "vent"]
Leader = Literal["user", "assistant", "joint"]
Depth = Literal["shallow", "deep"]

Level = Literal["L0", "L1", "L2", "L3", "L4"]

ReactionType = Literal[
    "accept", "clarify", "correct", "refuse", "defer", "topic_shift", "mixed"
]
AckType = Literal["explicit_yes", "implicit_yes", "mixed", "no", "none"]

ResponseMode = Literal[
    "explain", "ask", "offer_options", "summarize", "repair", "meta_frame"
]

DeepReason = Literal[
    "meaning_mismatch",  # 意味ズレ / 予測誤差 (L1+L2)
    "need_evidence",  # 事実根拠が必要 (L2)
    "persona_premise_mismatch",  # 人物/前提ズレ (L3)
    "frame_collapse",  # 枠組み崩壊 (L4)
]

DeepKind = Literal["deep_repair", "deep_web", "deep_memory", "deep_frame"]


# === small structs ===
class SourcesUsed(TypedDict):
    memory: bool
    web: bool


class PredictionCommon(TypedDict):
    level: Level
    depth: Depth    outputs: dict
    confidence: float
    evidence: dict
    timestamp_turn: int


class RepairPlan(TypedDict):
    strategy: str
    questions: list[str]
    optionality: bool


class DeepChain(TypedDict):
    plan: list[DeepKind]
    executed: list[DeepKind]
    stop_reason: str


class DeepDecision(TypedDict):
    reason: DeepReason | ""
    repair_plan: RepairPlan
    deep_chain: DeepChain


class ResponseMeta(TypedDict, total=False):
    sources_label: str
    used_levels: list[Level]
    used_depths: list[Depth]
