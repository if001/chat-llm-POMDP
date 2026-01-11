# app/graph/nodes/learn_update.py
from __future__ import annotations

from dataclasses import dataclass

from app.models.state import AgentState, JointContext, PolicyState


@dataclass(frozen=True)
class LearnUpdateIn:
    turn_id: int
    joint_context: JointContext
    user_model: dict

    # 観測/予測/指標を材料に、閾値やnormsを更新する
    observation: dict
    predictions: dict
    metrics: dict
    deep_decision: dict
    action: dict
    response: dict

    policy: PolicyState
    epistemic_uncertainties_now: dict
    unresolved_count_now: int


@dataclass(frozen=True)
class LearnUpdateOut:
    status: str
    joint_context: JointContext
    user_model: dict
    policy: PolicyState
    last_turn_patch: dict


def make_learn_update_node():
    def inner(inp: LearnUpdateIn) -> LearnUpdateOut:
        """
        何をするか:
        - オンライン適応（学習）を行う（重み学習ではなく状態更新）
          - theta_deep の更新（rolling_V/rolling_PE等のEMAを導入するならpolicyに保持）
          - norms（question_budget等）の更新
          - user_model（traits/topic_preferences/taboos等）の更新
          - deep_history の更新
        - 次ターンの観測/差分計算のために last_turn を更新する
          - prev_assistant_text / prev_action / prev_response_meta
          - prev_uncertainties / prev_unresolved_count
        """
        policy = dict(inp.policy)
        # stub: deep_history だけ追記する口
        executed = inp.deep_decision.get("deep_chain", {}).get("executed", [])
        policy["deep_history"] = list(policy.get("deep_history", [])) + list(executed)

        last_turn_patch = {
            "prev_assistant_text": inp.response.get("final_text", ""),
            "prev_action": dict(inp.action),
            "prev_response_meta": dict(inp.response.get("meta", {})),
            "prev_uncertainties": dict(inp.epistemic_uncertainties_now),
            "prev_unresolved_count": int(inp.unresolved_count_now),
            "resolved_count_last_turn": 0,
        }

        return LearnUpdateOut(
            status="learn_update:stub",
            joint_context=inp.joint_context,
            user_model=inp.user_model,
            policy=policy,  # type: ignore[arg-type]
            last_turn_patch=last_turn_patch,
        )

    def node(state: AgentState) -> dict:
        out = inner(
            LearnUpdateIn(
                turn_id=state["turn_id"],
                joint_context=state["joint_context"],
                user_model=state["user_model"],
                observation=state["observation"],
                predictions=state["predictions"],
                metrics=state["metrics"],
                deep_decision=state["deep_decision"],
                action=state["action"],
                response=state["response"],
                policy=state["policy"],
                epistemic_uncertainties_now=state["epistemic_state"]["uncertainties"],
                unresolved_count_now=len(state["unresolved_points"]),
            )
        )

        # last_turn を部分更新
        last_turn = dict(state["last_turn"])
        last_turn.update(out.last_turn_patch)

        return {
            "joint_context": out.joint_context,
            "user_model": out.user_model,
            "policy": out.policy,
            "last_turn": last_turn,
        }

    return node
