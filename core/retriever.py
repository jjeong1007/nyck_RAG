"""Shared OpenAI embeddings + Pinecone vector store wiring.

All ingest pipelines and retrieval chains should import from this module so
embedding dimensions, index names, and metadata field names stay consistent.

Expected chunk metadata keys (set during ingestion):

- ``source_type`` — e.g. ``local``, ``notion``, ``discord``, ``transcript``
- ``file_name`` — original file or logical name
- ``date`` — ISO date string when known, else empty
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# ----- Named defaults (override via environment where noted) ----------------

DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION: int = 1536
DEFAULT_PINECONE_INDEX_NAME: str = "company-rag"

ENV_OPENAI_EMBEDDING_MODEL: str = "OPENAI_EMBEDDING_MODEL"
ENV_OPENAI_EMBEDDING_DIMENSION: str = "OPENAI_EMBEDDING_DIMENSION"
ENV_PINECONE_API_KEY: str = "PINECONE_API_KEY"
ENV_PINECONE_INDEX_NAME: str = "PINECONE_INDEX_NAME"
ENV_PINECONE_NAMESPACE: str = "PINECONE_NAMESPACE"

# Metadata keys stored on each chunk in Pinecone / LlamaIndex nodes
METADATA_SOURCE_TYPE: str = "source_type"
METADATA_FILE_NAME: str = "file_name"
METADATA_DATE: str = "date"

def get_repo_root() -> Path:
    """Return the repository root directory (parent of ``core/``).

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def ensure_env_loaded() -> None:
    """Load ``.env`` from the repository root if present.

    Safe to call multiple times. Each call re-runs :func:`dotenv.load_dotenv`
    with ``override=False`` so explicit shell/Railway env always wins, but
    **new** keys added to ``.env`` after the process started are still picked
    up (unlike a one-time load that never refreshes).
    """
    load_dotenv(dotenv_path=get_repo_root() / ".env", override=False)


def build_openai_embedding_model(
    *,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OpenAIEmbedding:
    """Configure an OpenAI embedding model for document and query vectors.

    Reads ``OPENAI_API_KEY`` from the environment (or constructor) and optional
    ``OPENAI_EMBEDDING_MODEL`` / ``OPENAI_EMBEDDING_DIMENSION``.

    Args:
        api_key: Explicit API key; defaults to env ``OPENAI_API_KEY``.
        model_name: Embedding model id; defaults to env ``OPENAI_EMBEDDING_MODEL``
            or ``DEFAULT_EMBEDDING_MODEL``.
        dimensions: Output dimension (v3 models only). Defaults to env
            ``OPENAI_EMBEDDING_DIMENSION`` or ``DEFAULT_EMBEDDING_DIMENSION``.

    Returns:
        A configured :class:`OpenAIEmbedding` instance.

    Raises:
        ValueError: If the OpenAI API key cannot be resolved.
    """
    ensure_env_loaded()
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to .env or pass api_key=…."
        )

    resolved_model = (
        model_name
        or os.getenv(ENV_OPENAI_EMBEDDING_MODEL)
        or DEFAULT_EMBEDDING_MODEL
    )
    dim_raw = os.getenv(ENV_OPENAI_EMBEDDING_DIMENSION)
    resolved_dim = dimensions
    if resolved_dim is None and dim_raw:
        resolved_dim = int(dim_raw)
    if resolved_dim is None:
        resolved_dim = DEFAULT_EMBEDDING_DIMENSION

    return OpenAIEmbedding(
        model=resolved_model,
        dimensions=resolved_dim,
        api_key=resolved_key,
    )


def build_pinecone_vector_store(
    *,
    api_key: Optional[str] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> PineconeVectorStore:
    """Connect to the Pinecone index used for all company knowledge.

    Args:
        api_key: Pinecone API key; defaults to ``PINECONE_API_KEY``.
        index_name: Index name; defaults to ``PINECONE_INDEX_NAME`` or
            ``DEFAULT_PINECONE_INDEX_NAME``.
        namespace: Optional Pinecone namespace; defaults to ``PINECONE_NAMESPACE``.

    Returns:
        A :class:`PineconeVectorStore` backed by the serverless index.

    Raises:
        ValueError: If Pinecone credentials or index name are missing.
    """
    ensure_env_loaded()
    resolved_key = api_key or os.getenv(ENV_PINECONE_API_KEY)
    if not resolved_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Add it to .env or pass api_key=…."
        )

    resolved_index = (
        index_name
        or os.getenv(ENV_PINECONE_INDEX_NAME)
        or DEFAULT_PINECONE_INDEX_NAME
    )
    resolved_ns = (
        namespace if namespace is not None else os.getenv(ENV_PINECONE_NAMESPACE)
    )

    return PineconeVectorStore(
        api_key=resolved_key,
        index_name=resolved_index,
        namespace=resolved_ns,
    )


def build_vector_store_index(
    *,
    embed_model: Optional[BaseEmbedding] = None,
    vector_store: Optional[PineconeVectorStore] = None,
) -> VectorStoreIndex:
    """Construct a :class:`VectorStoreIndex` over the shared Pinecone store.

    Passing ``embed_model`` ensures inserts and hybrid flows use the same
    embeddings as retrieval. If omitted, a fresh model is built via
    :func:`build_openai_embedding_model`.

    Args:
        embed_model: Embedding model for this index; built if ``None``.
        vector_store: Pinecone backing store; built if ``None``.

    Returns:
        A :class:`VectorStoreIndex` connected to Pinecone.

    Raises:
        ValueError: If embedding or Pinecone configuration is invalid.
    """
    ensure_env_loaded()
    _embed = embed_model or build_openai_embedding_model()
    _store = vector_store or build_pinecone_vector_store()
    return VectorStoreIndex.from_vector_store(
        vector_store=_store,
        embed_model=_embed,
    )


def build_vector_index_retriever(
    *,
    similarity_top_k: int,
    embed_model: Optional[BaseEmbedding] = None,
    vector_store: Optional[PineconeVectorStore] = None,
) -> BaseRetriever:
    """Create a retriever over the Pinecone index with a fixed ``top_k``.

    Args:
        similarity_top_k: Number of vectors to retrieve per query.
        embed_model: Optional embedding model (see :func:`build_vector_store_index`).
        vector_store: Optional vector store (see :func:`build_vector_store_index`).

    Returns:
        A LlamaIndex :class:`BaseRetriever` (vector index retriever).

    Raises:
        ValueError: If ``similarity_top_k`` is not positive or config is invalid.
    """
    if similarity_top_k < 1:
        raise ValueError("similarity_top_k must be >= 1")
    index = build_vector_store_index(
        embed_model=embed_model,
        vector_store=vector_store,
    )
    return index.as_retriever(similarity_top_k=similarity_top_k)
