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
        dd = dict(inp.deep_decision)
        chain = dict(dd["deep_chain"])
        chain["executed"] = list(chain["executed"]) + ["deep_frame"]
        dd["deep_chain"] = chain
        return DeepFrameOut(
            status="deep_frame:stub", deep_decision=dd, joint_context=inp.joint_context
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
