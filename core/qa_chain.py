"""Q&A chain: retrieve top‑K chunks from Pinecone and answer with Claude Haiku.

The chain is built once per process and reused by the FastAPI app. Each call to
:meth:`QAChain.ask` retrieves fresh chunks for the **current** question, then
calls Claude. Optional *chat history* (prior user/assistant turns) is sent so
follow-up questions resolve pronouns and context; factual answers must still
come from the retrieved excerpts supplied on that turn.

Public API:
    - :class:`QAChain` — stateful holder so the FastAPI app reuses one instance.
    - :func:`answer_question` — convenience for tests / scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import anthropic
from llama_index.core.schema import NodeWithScore

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_index_retriever,
    ensure_env_loaded,
)

# ----- Model + retrieval constants ------------------------------------------

CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
CLAUDE_TEMPERATURE: float = 0.2
CLAUDE_MAX_TOKENS: int = 1024

QA_TOP_K: int = 5
SOURCE_PREVIEW_CHARS: int = 240
CHAT_HISTORY_CHAR_BUDGET: int = 12000

ENV_ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
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
        self._retriever = build_vector_index_retriever(
            similarity_top_k=similarity_top_k,
            embed_model=embed_model,
            vector_store=vector_store,
        )
        api_key = anthropic_api_key or os.getenv(ENV_ANTHROPIC_API_KEY)
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or pass api_key=…."
            )
        self._anthropic = anthropic.Anthropic(api_key=api_key)

    def ask(
        self,
        question: str,
        *,
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> QAResult:
        """Answer ``question`` using top-K retrieved company knowledge.

        Args:
            question: User's natural language question for this turn (also used
                as the retrieval query).
            chat_history: Optional prior ``(role, content)`` turns, roles
                ``user`` or ``assistant``. Must not include the current
                ``question``; older turns may be truncated to save tokens.

        Returns:
            :class:`QAResult` with the answer and structured citations.

        Raises:
            ValueError: If ``question`` is empty.
            Exception: Network / API errors from OpenAI, Pinecone, or Anthropic
                are propagated; callers should wrap to return a clean HTTP error.
        """
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")

        nodes: List[NodeWithScore] = list(self._retriever.retrieve(cleaned))
        context_block = _nodes_to_llm_context(nodes)
        history = _trim_chat_history(tuple(chat_history or ()))

        api_messages: List[dict[str, str]] = list(history)
        api_messages.append(
            {"role": "user", "content": _final_user_payload(cleaned, context_block)}
        )

        message = self._anthropic.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=CLAUDE_TEMPERATURE,
            system=QA_SYSTEM_PROMPT,
            messages=api_messages,
        )

        raw_text = "".join(
            getattr(block, "text", "") for block in (message.content or [])
        )
        answer_text = raw_text.strip()
        sources = [_node_to_source(n) for n in nodes]
        return QAResult(answer=answer_text, sources=sources)


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
) -> QAResult:
    """Top-level helper used by tests and ad-hoc scripts.

    Args:
        question: User question for this turn.
        chat_history: Optional prior turns (see :meth:`QAChain.ask`).

    Returns:
        :class:`QAResult` with answer and sources.
    """
    return get_default_chain().ask(question, chat_history=chat_history)
