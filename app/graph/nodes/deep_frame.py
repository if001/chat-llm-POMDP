# app/graph/nodes/deep_frame.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState, JointContext, Metrics, Observation
from app.models.types import DeepDecision
from app.graph.nodes.prompt_utils import (
    format_joint_context,
    format_metrics,
    format_observation,
)
from app.ports.llm import LLMPort
from app.graph.utils import utils
from app.graph.utils.write import a_stream_writer


@dataclass(frozen=True)
class DeepFrameIn:
    deep_decision: DeepDecision
    joint_context: JointContext
    observation: Observation
    metrics: Metrics


@dataclass(frozen=True)
class DeepFrameOut:
    status: str
    deep_decision: DeepDecision
    joint_context: JointContext


async def _suggest_frame_update(
    small_llm: LLMPort,
    joint_context: JointContext,
    observation: Observation,
    metrics: Metrics,
) -> dict[str, Any]:
    prompt = (
        "あなたは枠組み再設計の提案器\n"
        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        "入力を用いて枠組み再設計の提案を行ってください。\n"
        "【入力フィールド】\n"
        "- joint_context: 現在の枠組み/役割/規範。再交渉の基準\n"
        "- observation: 反応分類。枠組み崩壊や抵抗兆候の判定\n"
        "- metrics: 直近指標(PE/ΔI/ΔG/ΔJ/risk等)。調整強度の判断\n\n"
        "【出力フィールド】\n"
        "frame：今後の対話で採用すべき会話の枠組み（何を共同でして進めるか）を指定\n"
        "question_budget：新しい枠組みにおいて、1ターンで許容される質問の最大数（探索度合いの調整）\n"
        "summarize_before_advice：助言や提案に入る前に、理解確認のための要約を必須とするかどうか\n"
        "max_response_length：新しい枠組みにおける1回の返答の最大長（情報量・認知負荷の制御）\n\n"
        "【重要】前置きや装飾は不要で、必ずJSONのみを出力すること\n"
        "【出力フォーマット】\n"
        "{\n"
        '"frame": "explore|decide|execute|reflect|vent", \n'
        '"question_budget": int, \n'
        '"summarize_before_advice": true|false, \n'
        '"max_response_length": int\n'
        "}"
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "枠組み再設計の提案を行ってください。\n"
                        "※重要 前置きや装飾は不要で、必ずJSONのみを出力すること\n\n"
                        "- joint_context: 現在の枠組み/役割/規範。再交渉の基準\n"
                        f"{format_joint_context(joint_context)}\n"
                        "- observation: 反応分類。枠組み崩壊や抵抗兆候の判定\n"
                        f"{format_observation(observation)}\n"
                        "- metrics: 直近指標(PE/ΔI/ΔG/ΔJ/risk等)。調整強度の判断\n"
                        f"{format_metrics(metrics)}\n"
                    ),
                },
            ]
        )
    except Exception:
        print("deep frame fallback")
        return {}
    return utils.parse_llm_response(result)


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

    @a_stream_writer("deep_frame")
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
