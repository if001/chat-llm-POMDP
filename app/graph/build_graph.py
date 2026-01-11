# app/graph/build_graph.py
from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END

from app.core.deps import Deps
from app.models.state import AgentState

from app.graph.nodes.ingest_turn import make_ingest_turn_node
from app.graph.nodes.predict_shallow import make_predict_shallow_node
from app.graph.nodes.observe_reaction import make_observe_reaction_node
from app.graph.nodes.compute_metrics import make_compute_metrics_node
from app.graph.nodes.gate_depth import make_gate_depth_node
from app.graph.nodes.deep_repair import make_deep_repair_node
from app.graph.nodes.deep_memory import make_deep_memory_node
from app.graph.nodes.deep_web import make_deep_web_node
from app.graph.nodes.deep_frame import make_deep_frame_node
from app.graph.nodes.decide_response_plan import make_decide_response_plan_node
from app.graph.nodes.respond import make_respond_node
from app.graph.nodes.learn_update import make_learn_update_node
from app.graph.nodes.persist_trace import make_persist_trace_node


def _route_after_gate(
    state: AgentState,
) -> Literal[
    "deep_repair", "deep_memory", "deep_web", "deep_frame", "decide_response_plan"
]:
    plan = state["deep_decision"]["deep_chain"]["plan"]
    if not plan:
        return "decide_response_plan"
    nxt = plan[0]
    if nxt == "deep_repair":
        return "deep_repair"
    if nxt == "deep_memory":
        return "deep_memory"
    if nxt == "deep_web":
        return "deep_web"
    if nxt == "deep_frame":
        return "deep_frame"
    return "decide_response_plan"


def build_graph(deps: Deps):
    g = StateGraph(AgentState)

    # nodes
    g.add_node("ingest_turn", make_ingest_turn_node())
    g.add_node("observe_reaction", make_observe_reaction_node(deps))
    g.add_node("predict_shallow", make_predict_shallow_node(deps))
    g.add_node("compute_metrics", make_compute_metrics_node())
    g.add_node("gate_depth", make_gate_depth_node())

    g.add_node("deep_repair", make_deep_repair_node())
    g.add_node("deep_memory", make_deep_memory_node(deps))
    g.add_node("deep_web", make_deep_web_node(deps))
    g.add_node("deep_frame", make_deep_frame_node())

    g.add_node("decide_response_plan", make_decide_response_plan_node())
    g.add_node("respond", make_respond_node(deps))
    g.add_node("learn_update", make_learn_update_node())
    g.add_node("persist_trace", make_persist_trace_node(deps))

    # edges (linear core)
    g.set_entry_point("ingest_turn")
    g.add_edge("ingest_turn", "observe_reaction")
    g.add_edge("observe_reaction", "predict_shallow")
    g.add_edge("predict_shallow", "compute_metrics")
    g.add_edge("compute_metrics", "gate_depth")

    # conditional: gate_depth -> deep_* or decide
    g.add_conditional_edges("gate_depth", _route_after_gate)

    # deep nodes converge to decide
    g.add_edge("deep_repair", "decide_response_plan")
    g.add_edge("deep_memory", "decide_response_plan")
    g.add_edge("deep_web", "decide_response_plan")
    g.add_edge("deep_frame", "decide_response_plan")

    # finalize
    g.add_edge("decide_response_plan", "respond")
    g.add_edge("respond", "learn_update")
    g.add_edge("learn_update", "persist_trace")
    g.add_edge("persist_trace", END)

    return g.compile()
