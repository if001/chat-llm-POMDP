# app/graph/nodes/deep_repair.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from app.core.deps import Deps
from app.models.state import AgentState
from app.models.types import DeepDecision
from app.ports.llm import LLMPort


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


async def _plan_repair(
    small_llm: LLMPort,
    user_input: str,
    wm_messages: list[dict],
) -> dict[str, Any]:
    prompt = (
        "Return JSON with keys: strategy, questions (list), optionality (bool). "
        "Keep questions short."
    )
    try:
        result = await small_llm.ainvoke(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"user_input: {user_input}\nwm_messages: {wm_messages}",
                },
            ]
        )
    except Exception:
        return {}
    return _parse_json(_get_content(result))


def make_deep_repair_node(deps: Deps):
    async def inner(inp: DeepRepairIn) -> DeepRepairOut:
        """
        何をするか:
        - 意味ズレ/予測誤差（L1+L2）に対して deep_repair を実行
          - 確認質問の生成
          - 言い換え
          - 選択肢提示（optionality）
        - deep_decision.repair_plan を具体化
        - deep_chain.executed に "deep_repair" を追加
        """
        dd = dict(inp.deep_decision)
        repair_plan = dict(dd.get("repair_plan", {}))
        payload = await _plan_repair(deps.small_llm, inp.user_input, inp.wm_messages)
        if payload:
            repair_plan["strategy"] = payload.get("strategy", repair_plan.get("strategy", ""))
            repair_plan["questions"] = payload.get("questions", repair_plan.get("questions", []))
            repair_plan["optionality"] = bool(
                payload.get("optionality", repair_plan.get("optionality", False))
            )
            dd["repair_plan"] = repair_plan
        chain = dict(dd["deep_chain"])
        chain["executed"] = list(chain["executed"]) + ["deep_repair"]
        dd["deep_chain"] = chain
        return DeepRepairOut(
            status="deep_repair:ok", deep_decision=dd, wm_messages=inp.wm_messages
        )

    async def node(state: AgentState) -> dict:
        out = await inner(
            DeepRepairIn(
                turn_id=state["turn_id"],
                deep_decision=state["deep_decision"],
                wm_messages=state["wm_messages"],
                user_input=state["user_input"],
            )
        )
        return {"deep_decision": out.deep_decision, "wm_messages": out.wm_messages}

    return node
