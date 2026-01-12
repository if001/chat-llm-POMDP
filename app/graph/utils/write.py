from langgraph.config import get_stream_writer
from app.models.state import AgentState


def write_status(text, *, node=""):
    writer = get_stream_writer()
    writer({"type": "status", "node": node, "text": text})


def write_thinking(text, *, node=""):
    writer = get_stream_writer()
    writer({"type": "thinking", "node": node, "text": text})


def write_token(text, *, node=""):
    writer = get_stream_writer()
    writer({"type": "token", "node": node, "text": text})


def stream_writer(func_name="node"):
    def wrapper(f):
        def inner(state: AgentState):
            print(f"start: {func_name}")
            state = f(state)
            print(f"done: {func_name}")
            # writer = get_stream_writer()
            # writer({"type": "status", "node": node, "text": text})
            return state

        return inner

    return wrapper


def a_stream_writer(func_name="node"):
    def wrapper(f):
        async def inner(state: AgentState):
            print(f"start: {func_name}")
            write_thinking(func_name)
            state = await f(state)
            print(f"end: {func_name}")
            return state

        return inner

    return wrapper
