# app/graph/nodes/deep_frame.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, JointContext
from app.models.types import DeepDecision
from app.graph.nodes.prompt_utils import (
    format_joint_context,
    format_metrics,
    format_observation,
)
from app.ports.llm import LLMPort


@dataclass(frozen=True)
class DeepFrameIn:
    deep_decision: DeepDecision
    joint_context: JointContext
    observation: dict
    metrics: dict


@dataclass(frozen=True)
class DeepFrameOut:
    status: str
    deep_decision: DeepDecision
    joint_context: JointContext


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


async def _suggest_frame_update(
    small_llm: LLMPort,
    joint_context: JointContext,
    observation: dict,
    metrics: dict,
) -> dict[str, Any]:
    prompt = (
        "あなたは枠組み再設計(L4)の提案器。"
        "入力はjoint_context/observation/metricsで、枠組みや運用normsの調整に使う。"
        "出力はJSONのみ。"
        "出力フォーマット: {"
        '"frame": "explore|decide|execute|reflect|vent", '
        '"question_budget": int, '
        '"summarize_before_advice": true|false, '
        '"max_response_length": int'
        "}"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "入力:\n"
                        "- joint_context: 現在の枠組み/役割/規範。再交渉の基準に使う。\n"
                        f"{format_joint_context(joint_context)}\n"
                        "- observation: 反応分類。枠組み崩壊や抵抗兆候の判定に使う。\n"
                        f"{format_observation(observation)}\n"
                        "- metrics: 直近指標(PE/ΔI/ΔG/ΔJ/risk等)。調整強度の判断に使う。\n"
                        f"{format_metrics(metrics)}"
                    ),
                },
            ]
        )
    except Exception:
        return {}
    return _parse_json(_get_content(result))


def make_deep_frame_node(deps: Deps):
    async def inner(inp: DeepFrameIn) -> DeepFrameOut:
        """
        何をするか:
        - 枠組み崩壊（L4）に対して deep_frame を実行
          - メタ交渉テンプレ（目的/進め方のズレ確認、選択肢提示、最小同意）
          - joint_context.frame / norms を更新
        - deep_chain.executed に "deep_frame" を追加
        """
        dd = dict(inp.deep_decision)
        joint_context = dict(inp.joint_context)
        norms = dict(joint_context.get("norms", {}))
        payload = await _suggest_frame_update(
            deps.small_llm, inp.joint_context, inp.observation, inp.metrics
        )
        if payload:
            frame = payload.get("frame")
            if frame:
                joint_context["frame"] = frame
            if "question_budget" in payload:
                norms["question_budget"] = payload["question_budget"]
            if "summarize_before_advice" in payload:
                norms["summarize_before_advice"] = payload["summarize_before_advice"]
            if "max_response_length" in payload:
                norms["max_response_length"] = payload["max_response_length"]
            joint_context["norms"] = norms
        chain = dict(dd["deep_chain"])
        chain["executed"] = list(chain["executed"]) + ["deep_frame"]
        dd["deep_chain"] = chain
        return DeepFrameOut(
            status="deep_frame:ok", deep_decision=dd, joint_context=joint_context
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
            DeepFrameIn(
                deep_decision=state["deep_decision"],
                joint_context=state["joint_context"],
                observation=state["observation"],
                metrics=state["metrics"],
            )
        )
        return {"deep_decision": out.deep_decision, "joint_context": out.joint_context}

    return node
