# app/graph/nodes/deep_frame.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, JointContext
from app.models.types import DeepDecision


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


def make_deep_frame_node():
    def inner(inp: DeepFrameIn) -> DeepFrameOut:
        """
        何をするか:
        - 枠組み崩壊（L4）に対して deep_frame を実行
          - メタ交渉テンプレ（目的/進め方のズレ確認、選択肢提示、最小同意）
          - joint_context.frame / norms を更新
        - deep_chain.executed に "deep_frame" を追加
        """
        deep_decision = dict(inp.deep_decision)
        deep_chain = dict(deep_decision.get("deep_chain", {}))
        executed = list(deep_chain.get("executed", [])) + ["deep_frame"]
        deep_chain.setdefault("plan", [])
        deep_chain["executed"] = executed
        deep_chain.setdefault("stop_reason", "")
        deep_decision["deep_chain"] = deep_chain
        return DeepFrameOut(
            status="deep_frame:ok",
            deep_decision=deep_decision,
            joint_context=inp.joint_context,
        )

    def node(state: AgentState) -> dict:
        out = inner(
            DeepFrameIn(
                deep_decision=state["deep_decision"],
                joint_context=state["joint_context"],
                observation=state["observation"],
                metrics=state["metrics"],
            )
        )
        return {"deep_decision": out.deep_decision, "joint_context": out.joint_context}

    return node
