"""Q&A chain: retrieve top‑K chunks from Pinecone and answer with Claude Haiku.

The chain is the read-side counterpart of the ingest scripts. It is deliberately
stateless: each call to :func:`answer_question` builds (or reuses) a query engine
and returns ``(answer, sources)`` where ``sources`` is a flat list with
``source_type``, ``file_name``, ``date``, similarity ``score``, and a short
preview of the chunk text.

Public API:
    - :class:`QAChain` — stateful holder so the FastAPI app reuses one instance.
    - :func:`answer_question` — convenience for tests / scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic

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

ENV_ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"

QA_SYSTEM_PROMPT: str = (
    "You are the internal Q&A assistant for an 8-person company. "
    "Answer ONLY using the provided context excerpts from the company's "
    "knowledge base (Notion pages, internal docs, sales transcripts, Discord). "
    "Be concise and direct — 1–4 short paragraphs is plenty for most questions.\n\n"
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


def _build_claude_llm(*, api_key: Optional[str] = None) -> Anthropic:
    """Create the Claude Haiku LLM with our system prompt baked in.

    Args:
        api_key: Optional Anthropic key; falls back to env ``ANTHROPIC_API_KEY``.

    Raises:
        ValueError: If no API key is available.
    """
    ensure_env_loaded()
    resolved = api_key or os.getenv(ENV_ANTHROPIC_API_KEY)
    if not resolved:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or pass api_key=…."
        )
    return Anthropic(
        model=CLAUDE_HAIKU_MODEL,
        api_key=resolved,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=CLAUDE_TEMPERATURE,
        system_prompt=QA_SYSTEM_PROMPT,
    )


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
        self._llm = _build_claude_llm(api_key=anthropic_api_key)
        self._engine: RetrieverQueryEngine = RetrieverQueryEngine.from_args(
            retriever=self._retriever,
            llm=self._llm,
            response_mode=ResponseMode.COMPACT,
        )

    def ask(self, question: str) -> QAResult:
        """Answer ``question`` using top-K retrieved company knowledge.

        Args:
            question: User's natural language question.

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

        response = self._engine.query(cleaned)

        answer_text = str(response).strip()
        source_nodes: List[NodeWithScore] = []
        if isinstance(response, Response):
            source_nodes = list(response.source_nodes or [])
        else:  # pragma: no cover — non-Response synthesizer outputs
            maybe_nodes = getattr(response, "source_nodes", None)
            if maybe_nodes:
                source_nodes = list(maybe_nodes)

        sources = [_node_to_source(n) for n in source_nodes]
        return QAResult(answer=answer_text, sources=sources)


# ----- Functional convenience -----------------------------------------------


_DEFAULT_CHAIN: Optional[QAChain] = None


def get_default_chain() -> QAChain:
    """Lazily build a process-wide :class:`QAChain` for scripts and the API."""
    global _DEFAULT_CHAIN
    if _DEFAULT_CHAIN is None:
        _DEFAULT_CHAIN = QAChain()
    return _DEFAULT_CHAIN


def answer_question(question: str) -> QAResult:
    """Top-level helper used by tests and ad-hoc scripts.

    Args:
        question: User question.

    Returns:
        :class:`QAResult` with answer and sources.
    """
    return get_default_chain().ask(question)
