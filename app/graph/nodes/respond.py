# app/graph/nodes/respond.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo


from app.core.deps import Deps
from app.models.state import (
    AffectiveState,
    AgentState,
    JointContext,
    Response,
    Action,
    AssumptionItem,
    UnresolvedItem,
)
from app.graph.nodes.prompt_utils import (
    format_affective_state,
    format_common_ground,
    format_joint_context,
    format_unresolved_points,
    format_snippets,
)
from app.models.types import SourcesUsed
from app.ports.llm import LLMPort
from app.config.persona import PersonaConfig
from app.graph.utils.write import a_stream_writer, write_token
from app.graph.utils import utils


@dataclass(frozen=True)
class RespondIn:
    turn_id: int
    user_input: str
    wm_messages: list[dict]
    action: Action
    sources_used: SourcesUsed
    joint_context: JointContext
    memory_snippets: list[dict]
    web_snippets: list[dict]
    question_budget: int
    common_ground: dict[str, list[AssumptionItem]]
    unresolved_points: list[UnresolvedItem]
    affective_state: AffectiveState


@dataclass(frozen=True)
class RespondOut:
    status: str
    response: Response


def get_now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def gen_system_prompt(persona: PersonaConfig):
    _traits = "、".join(persona.traits)
    return f"""あなたは1対1テキスト対話エージェントの「最終応答生成器」です。名前は[{persona.name}]です。
以下の設定を必ず厳守してください。
{_traits}

目的は「共同行為（joint action）の制約のもとで、将来にわたる期待情報量・関係安定・共同性を最大化する」ことです。
あなたは、下記の入力に基づいて、返答を日本語で生成してください。

【入力の説明（状態 / state）】
1) joint_context：いまの対話の「共同作業の状態」
- frame：今この対話が「何を共同でしているか」を表す枠組み
- roles：主導権が誰にあるか（user / assistant / joint）
- norms：対話運用ルール（何を言うかではなく、どう進めるかの制約）

2) common_ground：共有できている前提（合意済みの命題のリスト）
- ここにある内容は、すでに共有できている前提として扱ってよい（ただしconfidenceが低いものは断定しない）

3) unresolved_points：共有できていない穴（埋めるべき未確定点のリスト）
- kind は semantic / epistemic / social のいずれか
  - semantic：意味の曖昧さ（用語・意図・指示対象の不明確さ）
  - epistemic：事実・根拠・条件が不足している
  - social：踏み込み、言い方、関係性の不確実さ

4) affective_state：感情・距離の信号（独立の目的ではなく、応答の制御入力）
- episode/mood：瞬間・持続の気分（valence/arousal）
- interpersonal_stance：距離や警戒（高いほど慎重）
- regulation_bias：落ち着かせたい（calm）/探索したい（explore）の傾向

5) response_plan：このターンの発話戦略
- response_mode：このターンの振る舞い（例：explain / ask / offer_options / summarize / repair / meta_frame）
- used_levels / used_depths：このターンで参照した予測レイヤーと深さ（内部ログ用の情報）
※あなたは内部用語をそのまま出力してはいけません。

6) sources_block：外部/記憶から得た情報
存在すれば応答の根拠として利用してください。

【あなたが守るべき生成ルール】
- 日本語で、自然で共同作業的（伴走的）な文体にする。
- frame に応じて最適化対象を切り替える：
  - explore：質問で前提を集める（ただしquestion_budget内）
  - decide：比較軸を整理し、選択肢を提示する（optionality_requiredなら必ず選択肢形式）
  - execute：具体的手順・次の一歩を提示する
  - reflect：要約・整理・問い返し（開かれた質問）
  - vent：共感・受容を優先し、助言は控えめ（必要なら「よければ」で提案）
- norms を厳守する：
  - 質問数は question_budget 以下
  - 返答は max_response_length を目安に簡潔に（超える場合は要点を優先）
  - summarize_before_advice=true のとき、提案前に短い要約確認を入れる
  - stance_sensitive=true のとき、踏み込みを抑え、断定を避け、選択肢を厚めにする
- unresolved_points がある場合、優先度が高いものから解消を試みる（質問は予算内）。
- 不確実な点は断定しない。必要なら確認質問・選択肢提示・保留を行う。
- 出力は「ユーザーへの返答テキストのみ」。JSONや内部状態の列挙は禁止。
"""


def gen_prompt(
    inp: RespondIn,
) -> str:
    response_mode = inp.action.get("response_mode")
    # confirm_questions = inp.action.get("confirm_questions", [])
    frame = inp.joint_context.get("frame")
    leader = inp.joint_context.get("roles", {}).get("leader")

    norms_text = format_joint_context(inp.joint_context)
    affective_text = format_affective_state(inp.affective_state)

    memory_text = (format_snippets(inp.memory_snippets, "memory_snippets"),)
    web_text = (format_snippets(inp.web_snippets, "web_snippets"),)

    common_ground_text = format_common_ground(inp.common_ground)
    unresolved_points_text = format_unresolved_points(inp.unresolved_points)

    return f"""【このターンのユーザー入力】
{inp.user_input}

【current_time】
{get_now()}

【joint_context】
- frame: {frame}
- roles.leader: {leader}
- norms:
    {norms_text}

【common_ground（共有前提）】
{common_ground_text}

【unresolved_points（未解決点）】
{unresolved_points_text}

【affective_state（感情・距離の信号）】
{affective_text}

【response_plan（このターンの発話戦略）】
- response_mode: {response_mode}

【knowledge_inputs】
{memory_text}

{web_text}

【出力指示】
- norms と frame を守って、ユーザーへの最終返答を作成してください。
- 質問は question_budget 以内。
- summarize_before_advice=true の場合は、提案前に短い要約確認を入れてください。
"""


def make_respond_node(deps: Deps):
    async def inner(inp: RespondIn, stream=True) -> RespondOut:
        """
        何をするか:
        - action.response_mode に従って最終応答文を生成（LLM）
        - deep_decision.repair_plan がある場合は質問/選択肢/言い換えを組み込む
        """
        persona = PersonaConfig.default()
        system_prompt = gen_system_prompt(persona)
        prompt = gen_prompt(inp)
        print(system_prompt)
        print("=" * 20)
        print(prompt)
        print("=" * 20)

        final_text = ""
        if stream:
            async for chunk in deps.llm.astream(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            ):
                text = getattr(chunk, "content", "") or ""
                if text:
                    final_text += text
                    write_token(text, node="respond")
        else:
            result = ""
            try:
                result = await deps.llm.ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as e:
                print("respond invoke error", e)
                result = "error"
            final_text = utils.get_content(result)
        resp: Response = {
            "final_text": final_text,
            "meta": {"turn_id": inp.turn_id},
        }
        return RespondOut(status="respond:ok", response=resp)

    @a_stream_writer("respond")
    async def node(state: AgentState) -> dict:
        out = await inner(
            RespondIn(
                turn_id=state["turn_id"],
                user_input=state["user_input"],
                wm_messages=state["wm_messages"],
                action=state["action"],
                sources_used=state["metrics"]["sources_used"],
                joint_context=state["joint_context"],
                memory_snippets=state["memory_snippets"],
                web_snippets=state["web_snippets"],
                question_budget=state["action"]["question_budget"],
                common_ground=state["common_ground"],
                unresolved_points=state["unresolved_points"],
                affective_state=state["affective_state"],
            )
        )
        return {"response": out.response}

    return node
