# app/models/state.py
from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.models.types import (
    AckType,
    Depth,
    Frame,
    Leader,
    ReactionType,
    ResponseMode,
    SourcesUsed,
)


# ====== Joint / Norms ======


class Norms(TypedDict):
    question_budget: int
    max_response_length: int
    optionality_required: bool
    summarize_before_advice: bool
    stance_sensitive: bool


class JointContext(TypedDict):
    frame: Frame
    roles: dict[str, Leader]  # {"leader": "..."}
    norms: Norms


# ====== Common Ground / Unresolved ======


class AssumptionItem(TypedDict):
    id: str
    proposition: str
    scope: Literal["global", "current_task"]
    confidence: float
    evidence_ids: list[str]
    last_updated_turn: int


class UnresolvedItem(TypedDict):
    id: str
    question: str
    kind: Literal["semantic", "epistemic", "social"]
    priority: float
    asked: bool
    answered: bool
    created_turn: int
    last_touched_turn: int


# ====== Affect ======


class AffectiveEpisode(TypedDict):
    valence: float  # 0..1
    arousal: float  # 0..1
    confidence: float


class AffectiveState(TypedDict):
    episode: AffectiveEpisode
    mood: dict[str, float]  # {"valence":..., "arousal":...}
    interpersonal_stance: float  # 0..1
    regulation_bias: dict[str, Any]  # {"mode": "calm|explore", "confidence": ...}


# ====== Epistemic ======


class EpistemicUncertainty(TypedDict):
    semantic: float  # 0..1
    epistemic: float  # 0..1
    social: float  # 0..1
    confidence: float


class HighStakes(TypedDict):
    value: float  # 0..1
    categories: list[str]
    confidence: float


class EpistemicState(TypedDict):
    uncertainties: EpistemicUncertainty
    high_stakes: HighStakes
    # NOTE: v1.2では predictive_hierarchy(L0-L4: depth/last_conf) もあったが、
    #       現段階の「入出力検証」では未使用のため省略。
    #       必要になればここへ戻す。


# ====== Observation ======


class ObservationEvents(TypedDict):
    E_correct: int
    E_refuse: int
    E_clarify: int
    E_miss: int
    E_frame_break: int
    E_overstep: int


class Observation(TypedDict):
    reaction_type: ReactionType
    ack_type: AckType
    events: ObservationEvents
    confidence: float


# ====== Metrics ======


class Metrics(TypedDict):
    prediction_error: float  # PE_t 0..1

    delta_I: float
    delta_G: float
    delta_J: float

    risk: float
    cost_user: float
    cost_agent: float

    sources_used: SourcesUsed


# ====== Predictions ======
# 予測の共通フォーマットは types.PredictionCommon を使う想定だが、
# state側では dict[str, Any] として受け、ノード側で厳密型を持つ方針にしている。


class Predictions(TypedDict):
    L0: dict[str, Any]
    L1: dict[str, Any]
    L2: dict[str, Any]
    L3: dict[str, Any]
    L4: dict[str, Any]


# ====== Deep Decision ======


class RepairPlan(TypedDict):
    strategy: str
    questions: list[str]
    optionality: bool


class DeepChain(TypedDict):
    plan: list[str]  # ["deep_repair", "deep_memory", ...]
    executed: list[str]  # 実行したもの
    stop_reason: str


class DeepDecision(TypedDict):
    reason: str  # meaning_mismatch / need_evidence / persona_premise_mismatch / frame_collapse / ""
    repair_plan: RepairPlan
    deep_chain: DeepChain


# ====== Action / Response ======


class Action(TypedDict):
    chosen_frame: Frame
    chosen_role_leader: Leader
    response_mode: ResponseMode

    questions_asked: int
    question_budget: int
    confirm_questions: list[str]

    # NOTE: did_* は「実際に使った」ではなく「計画/実行意図」を表す（重複を避けるため）
    did_memory_search: bool
    did_web_search: bool

    used_levels: list[str]  # ["L0", ...]
    used_depths: list[str]  # ["shallow", "deep"]


class Response(TypedDict):
    final_text: str
    meta: dict[str, Any]  # sources_label 等


# ====== User Model ======


class UserAttribute(TypedDict):
    value: str
    confidence: float
    evidence: list[str]
    last_updated_turn: int


class UserModel(TypedDict):
    basic: dict[str, UserAttribute]
    preferences: dict[str, UserAttribute]
    tendencies: dict[str, UserAttribute]
    topics: dict[str, UserAttribute]
    taboos: list[UserAttribute]
    last_updated_turn: int


# ====== Policy / Learning Params ======


class PolicyState(TypedDict):
    theta_deep: float
    deep_history: list[str]  # 直近のdeep実行ログ（種類や回数を保持）
    repair_stats: dict[str, dict[str, float]]
    rolling: dict[str, float]
    pending_evals: list[dict[str, Any]]
    # 将来: repair_success_stats, norms_update_stats, frame_prior などをここへ追加


# ====== Last Turn Snapshot ======
# 観測・メトリクスは「前ターンの出力と、今回の入力」の関係を見ることが多いので、
# 前ターン要約をstateに保持する（副作用ではなく、turn更新で生成されるデータ）。


class LastTurn(TypedDict):
    prev_assistant_text: str
    prev_action: dict[str, Any]  # 前ターン action のスナップショット
    prev_response_meta: dict[str, Any]  # 前ターン response.meta のスナップショット

    prev_uncertainties: EpistemicUncertainty
    prev_unresolved_count: int
    resolved_count_last_turn: int  # ΔG計算用の口（実装時に更新）


# ====== Agent State ======


class AgentState(TypedDict):
    # === working memory ===
    wm_messages: list[dict[str, Any]]
    turn_id: int
    user_input: str  # このターンの生入力（ingestでwm_messagesへ反映される）

    # === joint action constraints ===
    joint_context: JointContext

    # === shared understanding ===
    common_ground: dict[str, list[AssumptionItem]]
    unresolved_points: list[UnresolvedItem]

    # === affect / epistemic ===
    affective_state: AffectiveState
    epistemic_state: EpistemicState

    # === user model ===
    user_model: UserModel

    # === observation / metrics ===
    observation: Observation
    metrics: Metrics

    # === predictions / deep decision ===
    predictions: Predictions
    deep_decision: DeepDecision

    # === action / response ===
    action: Action
    response: Response
    memory_snippets: list[dict[str, Any]]
    web_snippets: list[dict[str, Any]]

    # === learning params ===
    policy: PolicyState

    # === last turn snapshot ===
    last_turn: LastTurn


def initial_state() -> AgentState:
    return {
        "wm_messages": [],
        "turn_id": 0,
        "user_input": "",
        "joint_context": {
            "frame": "explore",
            "roles": {"leader": "joint"},
            "norms": {
                "question_budget": 2,
                "max_response_length": 700,
                "optionality_required": True,
                "summarize_before_advice": False,
                "stance_sensitive": True,
            },
        },
        "common_ground": {"assumptions": []},
        "unresolved_points": [],
        "affective_state": {
            "episode": {"valence": 0.5, "arousal": 0.5, "confidence": 0.0},
            "mood": {"valence": 0.5, "arousal": 0.5},
            "interpersonal_stance": 0.5,
            "regulation_bias": {"mode": "explore", "confidence": 0.0},
        },
        "epistemic_state": {
            "uncertainties": {
                "semantic": 0.5,
                "epistemic": 0.5,
                "social": 0.5,
                "confidence": 0.0,
            },
            "high_stakes": {"value": 0.0, "categories": [], "confidence": 0.0},
        },
        "user_model": {
            "basic": {},
            "preferences": {},
            "tendencies": {},
            "topics": {},
            "taboos": [],
            "last_updated_turn": 0,
        },
        "observation": {
            "reaction_type": "mixed",
            "ack_type": "none",
            "events": {
                "E_correct": 0,
                "E_refuse": 0,
                "E_clarify": 0,
                "E_miss": 0,
                "E_frame_break": 0,
                "E_overstep": 0,
            },
            "confidence": 0.0,
        },
        "metrics": {
            "prediction_error": 0.0,
            "delta_I": 0.0,
            "delta_G": 0.0,
            "delta_J": 0.0,
            "risk": 0.0,
            "cost_user": 0.0,
            "cost_agent": 0.0,
            "sources_used": {"memory": False, "web": False},
        },
        "predictions": {"L0": {}, "L1": {}, "L2": {}, "L3": {}, "L4": {}},
        "deep_decision": {
            "reason": "",
            "repair_plan": {"strategy": "", "questions": [], "optionality": False},
            "deep_chain": {"plan": [], "executed": [], "stop_reason": ""},
        },
        "action": {
            "chosen_frame": "explore",
            "chosen_role_leader": "joint",
            "response_mode": "explain",
            "questions_asked": 0,
            "question_budget": 0,
            "confirm_questions": [],
            "did_memory_search": False,
            "did_web_search": False,
            "used_levels": ["L0", "L1", "L2", "L3", "L4"],
            "used_depths": ["shallow"],
        },
        "response": {"final_text": "", "meta": {}},
        "memory_snippets": [],
        "web_snippets": [],
        "policy": {
            "theta_deep": 1.2,
            "deep_history": [],
            "repair_stats": {
                "rephrase": {"alpha": 2.0, "beta": 2.0},
                "summarize_confirm": {"alpha": 2.0, "beta": 2.0},
                "offer_options": {"alpha": 2.0, "beta": 2.0},
                "intent_check": {"alpha": 2.0, "beta": 2.0},
                "meta_frame": {"alpha": 2.0, "beta": 2.0},
            },
            "rolling": {},
            "pending_evals": [],
        },
        "last_turn": {
            "prev_assistant_text": "",
            "prev_action": {},
            "prev_response_meta": {},
            "prev_uncertainties": {
                "semantic": 0.5,
                "epistemic": 0.5,
                "social": 0.5,
                "confidence": 0.0,
            },
            "prev_unresolved_count": 0,
            "resolved_count_last_turn": 0,
        },
    }
