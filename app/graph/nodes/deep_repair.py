# app/graph/nodes/deep_repair.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState
from app.models.types import DeepDecision


@dataclass(frozen=True)
class DeepRepairIn:
    turn_id: int
    deep_decision: DeepDecision
    wm_messages: list[dict]
    user_input: str


@dataclass(frozen=True)
class DeepRepairOut:
    status: str
    deep_decision: DeepDecision
    wm_messages: list[dict]


def make_deep_repair_node():
    def inner(inp: DeepRepairIn) -> DeepRepairOut:
        """
        何をするか:
        - 意味ズレ/予測誤差（L1+L2）に対して deep_repair を実行
          - 確認質問の生成
          - 言い換え
          - 選択肢提示（optionality）
        - deep_decision.repair_plan を具体化
        - deep_chain.executed に "deep_repair" を追加
        """
        deep_decision = dict(inp.deep_decision)
        repair_plan = dict(deep_decision.get("repair_plan", {}))
        repair_plan.setdefault("strategy", "confirm")
        repair_plan.setdefault("questions", [])
        repair_plan.setdefault("optionality", False)

        deep_chain = dict(deep_decision.get("deep_chain", {}))
        executed = list(deep_chain.get("executed", [])) + ["deep_repair"]
        deep_chain.setdefault("plan", [])
        deep_chain["executed"] = executed
        deep_chain.setdefault("stop_reason", "")

        deep_decision["repair_plan"] = repair_plan
        deep_decision["deep_chain"] = deep_chain
        return DeepRepairOut(
            status="deep_repair:ok",
            deep_decision=deep_decision,
            wm_messages=inp.wm_messages,
        )

    def node(state: AgentState) -> dict:
        out = inner(
            DeepRepairIn(
                turn_id=state["turn_id"],
                deep_decision=state["deep_decision"],
                wm_messages=state["wm_messages"],
                user_input=state["user_input"],
            )
        )
        return {"deep_decision": out.deep_decision, "wm_messages": out.wm_messages}

    return node
