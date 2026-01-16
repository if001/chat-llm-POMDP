# bot.py
import os
import re
import textwrap
from typing import Optional, Tuple
from urllib.parse import urlparse

import discord
from dotenv import load_dotenv

from app.config.settings import Settings
from app.core.deps import Deps
from app.adapters.ollama_llm import OllamaChatAdapter, OllamaEmbedder
from app.adapters.firecrawl_search import FirecrawlSearchAdapter
from app.adapters.chroma_memory import ChromaMemoryAdapter
from app.adapters.trace_json import JsonTraceAdapter
from app.adapters.clock import SystemClock
from app.models.state import initial_state
from app.graph.build_graph import build_graph

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


DISCORD_MESSAGE_LIMIT = 2000


def chunk_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Discord 2000 文字制限に合わせて分割（改行優先、最後は強制分割）"""
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    final = []
    for c in chunks:
        if len(c) <= limit:
            final.append(c)
        else:  # 非常に長い行を強制分割
            for i in range(0, len(c), limit):
                final.append(c[i : i + limit])
    return final


intents = discord.Intents.default()
intents.message_content = (
    True  # メッセージ本文の取得を許可（Bot設定画面でも有効化が必要）
)
intents.guilds = True
intents.messages = True


class AIChatBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = Settings()
        emb = OllamaEmbedder(model=s.embed_model, base_url=s.ollama_base_url)
        trace = JsonTraceAdapter(trace_dir=s.trace_dir)
        llm = OllamaChatAdapter(base_url=s.ollama_base_url, model=s.llm_model)
        small_llm = OllamaChatAdapter(
            base_url=s.ollama_base_url, model=s.small_llm_model
        )
        deps = Deps(
            llm=llm,
            small_llm=small_llm,
            memory=ChromaMemoryAdapter(
                persist_dir=s.chroma_persist_dir,
                collection_name=s.chroma_collection,
                embeder=emb,
                llm=llm,
                small_llm=small_llm,
            ),
            web=FirecrawlSearchAdapter(
                api_key=s.firecrawl_api_key, base_url=s.firecrawl_base_url
            ),
            trace=trace,
            clock=SystemClock(),
        )
        d = trace.load()
        self.graph = build_graph(deps)
        self.state = initial_state()
        self.state["wm_messages"] = d.get("wm_messages", [])
        _u = d.get("user_model")
        if _u is not None:
            self.state["user_model"] = _u
        print("prepare done...")

    async def on_ready(self):
        print(f"Logged in as {self.user} (id={self.user.id})")

    async def on_message(self, message: discord.Message):
        # 自分は無視
        if message.author.bot:
            return

        # メンションされていなければ無視
        if not self.user or self.user not in message.mentions:
            return

        # すでにスレッド内→親メッセージ側でスレッドにまとめたいので無視（必要なら変更OK）
        if isinstance(message.channel, discord.Thread):
            return

        ## はじめの改行はmenssionの@だとして取り除く
        user_input = ""
        part_message = message.content.split("\n", 1)
        if len(part_message) > 1:
            user_input = part_message[1]
        else:
            user_input = message.content

        self.state["user_input"] = user_input
        try:
            async with message.channel.typing():
                result = await self.graph.ainvoke(self.state)
                ans = result["response"]["final_text"]
                await message.channel.send(ans)
        except Exception as e:
            await message.reply(f"エラーが発生しました: {e}")


client = AIChatBot(intents=intents)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit(
            "DISCORD_BOT_TOKEN が未設定です（.env に DISCORD_BOT_TOKEN=...）。"
        )
    client.run(TOKEN)
