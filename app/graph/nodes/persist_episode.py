from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from discord import ReactionType

from app.core.deps import Deps
from app.models import state
from app.models.state import AgentState


@dataclass(frozen=True)
class PersistEpisodeIn:
    user_input: str
    assistant_output: str
    messages: list[dict]
    reaction_type: ReactionType
    e_frame_break: int


@dataclass(frozen=True)
class PersistEpisodeOut:
    messages: list[dict]


SAVE_HISTORY_LIMIT = 20


def make_persist_episode(deps: Deps):
    async def inner(inp: PersistEpisodeIn) -> PersistEpisodeOut:
        new_messages = inp.messages
        new_messages.append(
            {"role": "user", "content": f"[{deps.clock.now_iso()}] {inp.user_input}"}
        )
        new_messages.append(
            {
                "role": "assistant",
                "content": f"[{deps.clock.now_iso()}] {inp.assistant_output}",
            }
        )
        if len(new_messages) >= SAVE_HISTORY_LIMIT:
            new_messages = new_messages[2:]

        if (
            inp.e_frame_break == 1
            or inp.reaction_type == "topic_shift"
            or len(inp.messages) > SAVE_HISTORY_LIMIT
        ):
            episode = await deps.memory.store_episode_from_history(inp.messages)
            print("save episode", episode)
            new_messages = []

        return PersistEpisodeOut(messages=new_messages)

    # @stream_writer("persist_episode")
    async def node(state: AgentState) -> dict:
        out = await inner(
            PersistEpisodeIn(
                user_input=state["user_input"],
                assistant_output=state["response"]["final_text"],
                messages=state["episode_memory_messages"],
                reaction_type=state["observation"]["reaction_type"],
                e_frame_break=state["observation"]["events"]["E_frame_break"],
            )
        )
        return {"episode_memory_messages": out.messages}

    return node
