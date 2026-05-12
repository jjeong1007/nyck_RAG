"""Q&A chain: retrieve chunks from Pinecone and answer with Claude Haiku.

Two retrieval depths:

- ``qa`` — single embedding search, tight excerpts, strict extractive answers.
- ``synthesis`` — multi-query retrieval across many chunks, then grounded
  synthesis (personas, patterns, recommendations, brainstorms).

With ``routing="auto"`` (default), a planner step chooses ``qa`` vs
``synthesis``, optional ``source_type`` filters (transcripts vs Notion vs all),
and whether to apply creative brainstorming grounded in mission and tenets.

The chain is built once per process and reused by the FastAPI app.

Public API:
    - :class:`QAChain` — stateful holder so the FastAPI app reuses one instance.
    - :func:`answer_question` — convenience for tests / scripts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import anthropic
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_store_index,
    ensure_env_loaded,
)

QAMode = Literal["qa", "synthesis"]
RoutingMode = Literal["auto", "manual"]

# Ingest pipelines write these ``source_type`` metadata values on vectors.
INGEST_SOURCE_TYPES: Tuple[str, ...] = ("transcript", "notion", "discord", "local")

# ----- Model + retrieval constants ------------------------------------------

CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
CLAUDE_TEMPERATURE: float = 0.2
CLAUDE_MAX_TOKENS: int = 1024

QA_TOP_K: int = 5
SOURCE_PREVIEW_CHARS: int = 240
CHAT_HISTORY_CHAR_BUDGET: int = 12000

SYNTHESIS_TOP_K_PER_QUERY: int = 12
SYNTHESIS_MAX_SUBQUERIES: int = 7
SYNTHESIS_CONTEXT_CHAR_BUDGET: int = 90_000
SYNTHESIS_MAX_OUTPUT_TOKENS: int = 4096
SYNTHESIS_TEMPERATURE: float = 0.35

ENV_ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"
ENV_SYNTHESIS_TOP_K_PER_QUERY: str = "QA_SYNTHESIS_TOP_K_PER_QUERY"
ENV_SYNTHESIS_MAX_SUBQUERIES: str = "QA_SYNTHESIS_MAX_SUBQUERIES"
ENV_SYNTHESIS_CONTEXT_CHARS: str = "QA_SYNTHESIS_CONTEXT_CHARS"
ENV_SYNTHESIS_MAX_OUTPUT_TOKENS: str = "QA_SYNTHESIS_MAX_OUTPUT_TOKENS"

QUERY_EXPANSION_MAX_TOKENS: int = 400
QUERY_EXPANSION_TEMPERATURE: float = 0.2

ROUTER_MAX_TOKENS: int = 384
ROUTER_TEMPERATURE: float = 0.1

ROUTER_SYSTEM_PROMPT: str = (
    "You route internal employee questions to the right knowledge-base slices.\n\n"
    "Each chunk in the index has metadata ``source_type``, exactly one of:\n"
    "- transcript — customer or user interview / sales call transcripts\n"
    "- notion — internal Notion pages (wiki, missions, values, roadmaps, specs)\n"
    "- local — uploaded files (PDF/DOCX/MD in the company library)\n"
    "- discord — exported team Discord channels\n\n"
    "Choose ``mode``:\n"
    "- qa — narrow factual lookup (policies, dates, who owns X, single-doc facts).\n"
    "- synthesis — themes across many chunks: personas, JTBD, recommendations, "
    "prioritization, brainstorms, positioning, cross-call patterns.\n\n"
    "Choose ``source_types``:\n"
    "- Use null (omit or JSON null) to search **all** source types when the "
    "question is broad, operational, or clearly needs every category.\n"
    "- transcript — customer voice, personas from calls, quotes, objections, "
    "JTBD from users, \"what customers say\", usability feedback from calls.\n"
    "- notion + local together — company mission, values, product tenets, "
    "principles, strategy, positioning, requirements docs (prefer BOTH when "
    "the user asks for concepts grounded in \"our mission\", \"principles\", "
    "or \"how we build\").\n"
    "- discord — informal team discussion, past internal brainstorm threads, "
    "culture hints (combine with notion/local when useful).\n"
    "Use an array of one or more of the four strings when narrowing helps; "
    "never invent new source type names.\n\n"
    "Set ``creative_brainstorm`` true when the user wants **new** concepts, "
    "ideas, directions, campaigns, or features that must still be **anchored** "
    "in company principles from the KB — not generic AI fluff.\n\n"
    "Optional ``expansion_hint``: one short sentence telling the retriever what "
    "angles to search (e.g. \"Quotes on pricing pain\" or \"Mission and quality bar\").\n\n"
    "Output ONLY one JSON object with keys:\n"
    '- \"mode\": \"qa\" or \"synthesis\"\n'
    '- \"source_types\": JSON array of strings (each one of transcript, notion, '
    "discord, local) or JSON null to search all types\n"
    '- \"creative_brainstorm\": true or false\n'
    '- \"expansion_hint\": string (may be empty)\n'
    "Do not use markdown code fences."
)

QUERY_EXPANSION_SYSTEM: str = (
    "You write search queries for a semantic vector index over company "
    "knowledge: customer call transcripts, Notion pages, Discord exports, "
    "and internal files.\n\n"
    "Given the user's latest question and brief conversation context, output "
    "a JSON array of 5–8 short English search strings. Each string should "
    "retrieve chunks that mention concrete facts: roles, pains, workflows, "
    "tools, quotes, company sizes, industries, feature names, objections, "
    "success metrics.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON: one array of strings. No markdown fences, no "
    "commentary.\n"
    "- Strings must be clearly distinct (no near-duplicates).\n"
    "- Cover multiple angles of the question so retrieval is not narrowly "
    "matched to one phrasing."
)

SYNTHESIS_SYSTEM_PROMPT: str = (
    "You are an internal strategy assistant for a small company. Your job is "
    "to synthesize patterns, personas, recommendations, and summaries **using "
    "only** the knowledge-base excerpts supplied in the latest user message, "
    "plus conversation history for interpreting follow-ups.\n\n"
    "You MAY go beyond literal Q&A: infer personas, JTBD, recurring themes, "
    "and prioritized recommendations when the excerpts support them. Label "
    "clearly when something is a **pattern inferred across sources** versus a "
    "direct quote or explicit fact from one excerpt.\n\n"
    "Citation rules:\n"
    "- When you state a concrete fact or quote from the excerpts, cite inline "
    "with [file_name] exactly as shown in the excerpt headers.\n"
    "- For synthesized persona sections or cross-cutting patterns, add a short "
    'line such as "Grounded in:" listing the most relevant [file_name] '
    "sources that shaped that subsection.\n"
    "- Never invent file names. If evidence is thin, say so and suggest what "
    "data would help (e.g. more transcripts from segment X).\n\n"
    "Structure longer outputs with headings (e.g. Snapshot, Goals & pains, "
    "Buying triggers, Objections, Recommended next steps). Be substantive "
    "but scannable."
)

SYNTHESIS_CREATIVE_APPEND: str = (
    "\n\nCreative brainstorming (when applicable):\n"
    "- Offer several **distinct** concept directions; avoid generic filler that "
    "could apply to any company.\n"
    "- For each idea, tie explicitly to cited principles, mission language, or "
    "product tenets from the excerpts (by [file_name]).\n"
    "- Label ideas as **Grounded** when they closely follow stated principles, "
    "or **Stretch** when they extrapolate — still justify the leap using excerpts.\n"
    "- If excerpts lack strategic context, say so and propose what to ingest "
    "before brainstorming further."
)

SYNTHESIS_CREATIVE_TEMPERATURE: float = 0.42

QA_SYSTEM_PROMPT: str = (
    "You are the internal Q&A assistant for an 8-person company. "
    "Answer ONLY using the provided context excerpts from the company's "
    "knowledge base (Notion pages, internal docs, sales transcripts, Discord). "
    "Be concise and direct — 1–4 short paragraphs is plenty for most questions.\n\n"
    "Prior turns of the conversation may be included so you can interpret "
    "follow-ups (e.g. elliptical questions, pronouns). Use them only for "
    "intent and referents; every factual claim must still be grounded in the "
    "knowledge-base excerpts in the latest message. Do not treat earlier "
    "assistant replies as authoritative unless those facts appear in the "
    "current excerpts.\n\n"
    "Citation rules:\n"
    "- After any factual claim taken from a source, cite it inline like "
    "[file_name] (e.g. [onboarding.md] or [Acme call 2024-01-15]). "
    "If multiple sources support a sentence, cite each one.\n"
    "- Do not invent file names or sources. Only cite sources that appear in "
    "the provided context.\n\n"
    "When the context does not contain the answer, reply that you don't have "
    "that information in the knowledge base and suggest where the user might "
    "look (e.g. specific Notion area, a teammate). Do not guess."
)


# ----- Public types ----------------------------------------------------------


@dataclass
class QASource:
    """A retrieved chunk surfaced as a citation in the API response."""

    source_type: str
    file_name: str
    date: str
    score: float
    preview: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON responses."""
        return {
            "source_type": self.source_type,
            "file_name": self.file_name,
            "date": self.date,
            "score": self.score,
            "preview": self.preview,
        }


@dataclass
class QAResult:
    """Composite Q&A response: answer text + structured sources."""

    answer: str
    sources: List[QASource]
    retrieval_mode: QAMode = "qa"
    source_scope: Optional[List[str]] = None
    retrieval_routing: RoutingMode = "auto"
    creative_brainstorm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieval_meta": {
                "mode": self.retrieval_mode,
                "source_types": self.source_scope,
                "routing": self.retrieval_routing,
                "creative_brainstorm": self.creative_brainstorm,
            },
        }


# ----- Helpers ---------------------------------------------------------------


def _node_to_source(node: NodeWithScore) -> QASource:
    """Project a retrieved chunk onto the citation shape returned by the API."""
    metadata = node.metadata or {}
    text = (node.get_content() or "").strip().replace("\n", " ")
    if len(text) > SOURCE_PREVIEW_CHARS:
        text = text[: SOURCE_PREVIEW_CHARS - 1].rstrip() + "…"
    return QASource(
        source_type=str(metadata.get(METADATA_SOURCE_TYPE, "unknown")),
        file_name=str(metadata.get(METADATA_FILE_NAME, "unknown")),
        date=str(metadata.get(METADATA_DATE, "")),
        score=float(node.score) if node.score is not None else 0.0,
        preview=text,
    )


def _nodes_to_llm_context(nodes: Sequence[NodeWithScore]) -> str:
    """Full chunk text for the LLM (not the truncated API preview)."""
    if not nodes:
        return "(no related internal context found)"
    blocks: List[str] = []
    for i, node in enumerate(nodes, start=1):
        metadata = node.metadata or {}
        score_txt = f"{float(node.score):.3f}" if node.score is not None else "n/a"
        header = (
            f"[{i}] source_type={metadata.get(METADATA_SOURCE_TYPE, 'unknown')}, "
            f"file_name={metadata.get(METADATA_FILE_NAME, 'unknown')}, "
            f"date={metadata.get(METADATA_DATE, 'unknown')}, "
            f"score={score_txt}"
        )
        body = (node.get_content() or "").strip()
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _trim_chat_history(
    history: Sequence[Tuple[str, str]],
    *,
    max_chars: int = CHAT_HISTORY_CHAR_BUDGET,
) -> List[dict[str, str]]:
    """Keep the most recent turns that fit under ``max_chars`` (content only).

    Merges consecutive messages with the same role and drops empty strings.
    Strips leading ``assistant`` turns so the sequence can start with ``user``.
    """
    merged: List[dict[str, str]] = []
    for role, content in history:
        r = (role or "").strip().lower()
        if r not in ("user", "assistant"):
            continue
        text = (content or "").strip()
        if not text:
            continue
        if merged and merged[-1]["role"] == r:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + text
        else:
            merged.append({"role": r, "content": text})

    kept: List[dict[str, str]] = []
    total = 0
    for m in reversed(merged):
        piece = m["content"]
        if total + len(piece) > max_chars:
            break
        kept.append(m)
        total += len(piece)
    kept.reverse()

    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)

    return kept


def _final_user_payload(question: str, context_block: str) -> str:
    return (
        "The following excerpts are from the company knowledge base for this turn. "
        "Use only these (plus the conversation history above, for interpretation) "
        "to answer.\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"Latest user message:\n{question.strip()}"
    )


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return max(1, v)
    except ValueError:
        return default


def _metadata_filters_for_source_types(
    source_types: Optional[Sequence[str]],
) -> Optional[MetadataFilters]:
    if not source_types:
        return None
    cleaned = [s.strip() for s in source_types if (s or "").strip()]
    if not cleaned:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key=METADATA_SOURCE_TYPE,
                value=cleaned,
                operator=FilterOperator.IN,
            )
        ]
    )


def _node_stable_id(node: NodeWithScore) -> str:
    inner = getattr(node, "node", None)
    nid = getattr(inner, "node_id", None) if inner is not None else None
    if nid:
        return str(nid)
    meta = node.metadata or {}
    snippet = (node.get_content() or "")[:160]
    return f"{meta.get(METADATA_FILE_NAME, 'unknown')}::{hash(snippet)}"


def _dedupe_nodes_by_best_score(nodes: Sequence[NodeWithScore]) -> List[NodeWithScore]:
    best: dict[str, NodeWithScore] = {}
    for n in nodes:
        key = _node_stable_id(n)
        score_n = float(n.score if n.score is not None else 0.0)
        prev = best.get(key)
        if prev is None or score_n > float(prev.score if prev.score is not None else 0.0):
            best[key] = n
    return sorted(
        best.values(),
        key=lambda x: float(x.score if x.score is not None else 0.0),
        reverse=True,
    )


def _trim_nodes_to_context_budget(
    nodes: Sequence[NodeWithScore],
    max_chars: int,
) -> List[NodeWithScore]:
    """Keep highest-scoring nodes until serialized context fits ``max_chars``."""
    ordered = sorted(
        nodes,
        key=lambda n: float(n.score if n.score is not None else 0.0),
        reverse=True,
    )
    picked: List[NodeWithScore] = []
    for n in ordered:
        trial = picked + [n]
        ctx = _nodes_to_llm_context(trial)
        if len(ctx) <= max_chars:
            picked.append(n)
        elif not picked:
            picked.append(n)
            break
    return picked


def _conversation_hint_for_expansion(
    history: Sequence[Tuple[str, str]],
    *,
    max_chars: int = 3500,
) -> str:
    lines: List[str] = []
    total = 0
    for role, content in reversed(list(history)):
        if role not in ("user", "assistant"):
            continue
        piece = f"{role}: {(content or '').strip()}"
        if total + len(piece) > max_chars:
            break
        lines.append(piece)
        total += len(piece)
    lines.reverse()
    return "\n".join(lines) if lines else "(no prior turns)"


_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")


def _parse_expansion_json(raw: str) -> List[str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip()]
            return out
    except json.JSONDecodeError:
        pass
    match = _JSON_ARRAY_PATTERN.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _normalize_router_plan(
    obj: Optional[dict[str, Any]],
) -> tuple[QAMode, Optional[List[str]], bool, str]:
    allowed_modes = {"qa", "synthesis"}
    allowed_sources = set(INGEST_SOURCE_TYPES)
    if not obj:
        return "synthesis", None, False, ""

    mode_s = str(obj.get("mode", "synthesis")).strip().lower()
    mode: QAMode = mode_s if mode_s in allowed_modes else "synthesis"

    creative = bool(obj.get("creative_brainstorm", False))

    st_raw = obj.get("source_types")
    normalized_st: Optional[List[str]]
    if st_raw is None:
        normalized_st = None
    elif isinstance(st_raw, list):
        ordered: List[str] = []
        seen_s: set[str] = set()
        for x in st_raw:
            s = str(x).strip().lower()
            if s in allowed_sources and s not in seen_s:
                seen_s.add(s)
                ordered.append(s)
        normalized_st = ordered or None
    else:
        normalized_st = None

    hint = str(obj.get("expansion_hint") or "").strip()
    if len(hint) > 500:
        hint = hint[:500]

    if creative and mode == "qa":
        mode = "synthesis"

    return mode, normalized_st, creative, hint


def _route_retrieval_plan(
    client: anthropic.Anthropic,
    question: str,
    chat_history: Sequence[Tuple[str, str]],
) -> tuple[QAMode, Optional[List[str]], bool, str]:
    hint = _conversation_hint_for_expansion(chat_history)
    user_msg = (
        f"Prior conversation (most recent last, may be truncated):\n{hint}\n\n"
        f"Latest employee question:\n{question.strip()}"
    )
    message = client.messages.create(
        model=CLAUDE_HAIKU_MODEL,
        max_tokens=ROUTER_MAX_TOKENS,
        temperature=ROUTER_TEMPERATURE,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = "".join(
        getattr(block, "text", "") for block in (message.content or [])
    )
    obj = _parse_json_object(raw_text)
    return _normalize_router_plan(obj)


def _expand_retrieval_queries(
    client: anthropic.Anthropic,
    question: str,
    chat_history: Sequence[Tuple[str, str]],
    *,
    max_subqueries: int,
    expansion_hint: str = "",
) -> List[str]:
    hint = _conversation_hint_for_expansion(chat_history)
    user_msg = (
        f"Prior conversation (most recent last, may be truncated):\n{hint}\n\n"
        f"Latest question:\n{question.strip()}"
    )
    eh = (expansion_hint or "").strip()
    if eh:
        user_msg += f"\n\nRetrieval planner hint (bias sub-queries accordingly):\n{eh}"
    message = client.messages.create(
        model=CLAUDE_HAIKU_MODEL,
        max_tokens=QUERY_EXPANSION_MAX_TOKENS,
        temperature=QUERY_EXPANSION_TEMPERATURE,
        system=QUERY_EXPANSION_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = "".join(
        getattr(block, "text", "") for block in (message.content or [])
    )
    parsed = _parse_expansion_json(raw_text)
    seen: set[str] = set()
    merged: List[str] = []
    for q in [question.strip(), *parsed]:
        key = q.casefold()
        if key in seen or not q:
            continue
        seen.add(key)
        merged.append(q)
        if len(merged) >= max_subqueries:
            break
    return merged if merged else [question.strip()]


# ----- Stateful chain (built once, reused per request) ----------------------


class QAChain:
    """Reusable Q&A chain. Build once at API startup, then call :meth:`ask`."""

    def __init__(
        self,
        *,
        similarity_top_k: int = QA_TOP_K,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        ensure_env_loaded()
        embed_model = build_openai_embedding_model()
        vector_store = build_pinecone_vector_store()
        self._index = build_vector_store_index(
            embed_model=embed_model,
            vector_store=vector_store,
        )
        self._similarity_top_k = similarity_top_k
        api_key = anthropic_api_key or os.getenv(ENV_ANTHROPIC_API_KEY)
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or pass api_key=…."
            )
        self._anthropic = anthropic.Anthropic(api_key=api_key)

    def _retrieve_strict(
        self,
        query: str,
        filters: Optional[MetadataFilters],
    ) -> List[NodeWithScore]:
        r = self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters,
        )
        return list(r.retrieve(query))

    def _retrieve_synthesis(
        self,
        question: str,
        chat_history: Sequence[Tuple[str, str]],
        filters: Optional[MetadataFilters],
        *,
        expansion_hint: str = "",
    ) -> List[NodeWithScore]:
        per_q = _int_env(ENV_SYNTHESIS_TOP_K_PER_QUERY, SYNTHESIS_TOP_K_PER_QUERY)
        max_queries = _int_env(ENV_SYNTHESIS_MAX_SUBQUERIES, SYNTHESIS_MAX_SUBQUERIES)
        budget = _int_env(ENV_SYNTHESIS_CONTEXT_CHARS, SYNTHESIS_CONTEXT_CHAR_BUDGET)

        subqueries = _expand_retrieval_queries(
            self._anthropic,
            question,
            chat_history,
            max_subqueries=max_queries,
            expansion_hint=expansion_hint,
        )
        collected: List[NodeWithScore] = []
        for q in subqueries:
            r = self._index.as_retriever(similarity_top_k=per_q, filters=filters)
            collected.extend(r.retrieve(q))
        merged = _dedupe_nodes_by_best_score(collected)
        return _trim_nodes_to_context_budget(merged, budget)

    def _prepare_ask_llm(
        self,
        cleaned: str,
        hist_tuple: Tuple[Tuple[str, str], ...],
        routing: RoutingMode,
        mode: QAMode,
        source_types: Optional[Sequence[str]],
    ) -> Tuple[
        List[NodeWithScore],
        QAMode,
        Optional[List[str]],
        bool,
        str,
        int,
        float,
        List[dict[str, str]],
    ]:
        expansion_hint = ""
        creative = False
        if routing == "auto":
            effective_mode, scope_list, creative, expansion_hint = (
                _route_retrieval_plan(self._anthropic, cleaned, hist_tuple)
            )
        else:
            effective_mode = mode
            scope_list = (
                [s.strip() for s in source_types if (s or "").strip()]
                if source_types
                else None
            )
            if scope_list == []:
                scope_list = None

        filters = _metadata_filters_for_source_types(scope_list)

        if effective_mode == "synthesis":
            nodes = self._retrieve_synthesis(
                cleaned, hist_tuple, filters, expansion_hint=expansion_hint
            )
            system_prompt = SYNTHESIS_SYSTEM_PROMPT
            if creative:
                system_prompt += SYNTHESIS_CREATIVE_APPEND
            max_tokens = _int_env(
                ENV_SYNTHESIS_MAX_OUTPUT_TOKENS,
                SYNTHESIS_MAX_OUTPUT_TOKENS,
            )
            temperature = (
                SYNTHESIS_CREATIVE_TEMPERATURE
                if creative
                else SYNTHESIS_TEMPERATURE
            )
        else:
            nodes = self._retrieve_strict(cleaned, filters)
            system_prompt = QA_SYSTEM_PROMPT
            max_tokens = CLAUDE_MAX_TOKENS
            temperature = CLAUDE_TEMPERATURE

        context_block = _nodes_to_llm_context(nodes)
        history = _trim_chat_history(hist_tuple)
        api_messages: List[dict[str, str]] = list(history)
        api_messages.append(
            {"role": "user", "content": _final_user_payload(cleaned, context_block)}
        )
        return (
            nodes,
            effective_mode,
            scope_list,
            creative,
            system_prompt,
            max_tokens,
            temperature,
            api_messages,
        )

    def ask(
        self,
        question: str,
        *,
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
        routing: RoutingMode = "auto",
        mode: QAMode = "qa",
        source_types: Optional[Sequence[str]] = None,
    ) -> QAResult:
        """Answer ``question`` using retrieved company knowledge.

        Args:
            question: User message for this turn.
            chat_history: Optional prior ``(role, content)`` turns. Must not
                include the current ``question``.
            routing: ``auto`` runs a planner to choose retrieval depth and
                ``source_type`` filters; ``manual`` uses ``mode`` and
                ``source_types`` from this call.
            mode: Used when ``routing`` is ``manual`` only.
            source_types: Used when ``routing`` is ``manual`` only; restricts
                retrieval to these metadata ``source_type`` values.

        Returns:
            :class:`QAResult` with answer, citations, and ``retrieval_meta`` fields.

        Raises:
            ValueError: If ``question`` is empty.
            Exception: Network / API errors from OpenAI, Pinecone, or Anthropic
                are propagated; callers should wrap to return a clean HTTP error.
        """
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        hist_tuple = tuple(chat_history or ())
        (
            nodes,
            effective_mode,
            scope_list,
            creative,
            system_prompt,
            max_tokens,
            temperature,
            api_messages,
        ) = self._prepare_ask_llm(cleaned, hist_tuple, routing, mode, source_types)

        message = self._anthropic.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=api_messages,
        )

        raw_text = "".join(
            getattr(block, "text", "") for block in (message.content or [])
        )
        answer_text = raw_text.strip()
        sources = [_node_to_source(n) for n in nodes]
        return QAResult(
            answer=answer_text,
            sources=sources,
            retrieval_mode=effective_mode,
            source_scope=scope_list,
            retrieval_routing=routing,
            creative_brainstorm=creative,
        )

    def ask_stream_events(
        self,
        question: str,
        *,
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
        routing: RoutingMode = "auto",
        mode: QAMode = "qa",
        source_types: Optional[Sequence[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield SSE-friendly chunks: ``meta``, ``delta`` (text), then ``done``.

        Retrieval and routing run before the first yield; then Claude streams.
        """
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        hist_tuple = tuple(chat_history or ())
        (
            nodes,
            effective_mode,
            scope_list,
            creative,
            system_prompt,
            max_tokens,
            temperature,
            api_messages,
        ) = self._prepare_ask_llm(cleaned, hist_tuple, routing, mode, source_types)

        yield {
            "type": "meta",
            "sources": [_node_to_source(n).to_dict() for n in nodes],
            "retrieval_meta": {
                "mode": effective_mode,
                "source_types": scope_list,
                "routing": routing,
                "creative_brainstorm": creative,
            },
        }

        with self._anthropic.messages.stream(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=api_messages,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield {"type": "delta", "text": text}
        yield {"type": "done"}


# ----- Functional convenience -----------------------------------------------


_DEFAULT_CHAIN: Optional[QAChain] = None


def get_default_chain() -> QAChain:
    """Lazily build a process-wide :class:`QAChain` for scripts and the API."""
    global _DEFAULT_CHAIN
    if _DEFAULT_CHAIN is None:
        _DEFAULT_CHAIN = QAChain()
    return _DEFAULT_CHAIN


def answer_question(
    question: str,
    *,
    chat_history: Optional[Sequence[Tuple[str, str]]] = None,
    routing: RoutingMode = "auto",
    mode: QAMode = "qa",
    source_types: Optional[Sequence[str]] = None,
) -> QAResult:
    """Top-level helper used by tests and ad-hoc scripts.

    Args:
        question: User question for this turn.
        chat_history: Optional prior turns (see :meth:`QAChain.ask`).
        routing: ``auto`` or ``manual`` (see :meth:`QAChain.ask`).
        mode: Used when ``routing`` is ``manual``.
        source_types: Used when ``routing`` is ``manual``.

    Returns:
        :class:`QAResult` with answer and sources.
    """
    return get_default_chain().ask(
        question,
        chat_history=chat_history,
        routing=routing,
        mode=mode,
        source_types=source_types,
    )
