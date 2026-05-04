"""Load PDF, DOCX, TXT, and Markdown files from a directory into Pinecone.

Chunking matches internal-doc / Notion style: 800 tokens, 100 token overlap
via :class:`~llama_index.core.node_parser.SentenceSplitter`.

Run from the repo root::

    python -m ingest.ingest_local
    python -m ingest.ingest_local --path /absolute/path/to/folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_store_index,
    get_repo_root,
)

# Chunking for internal docs / Notion-style sources (token-based).
LOCAL_DOC_CHUNK_SIZE: int = 800
LOCAL_DOC_CHUNK_OVERLAP: int = 100

SOURCE_TYPE_LOCAL: str = "local"

# File types supported by :class:`SimpleDirectoryReader` + ``readers-file``.
LOCAL_FILE_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]


def _normalize_chunk_metadata(node: BaseNode) -> None:
    """Ensure each chunk has flat ``source_type``, ``file_name``, and ``date``."""
    meta = node.metadata
    file_name = meta.get(METADATA_FILE_NAME)
    if not file_name and meta.get("file_path"):
        file_name = Path(str(meta["file_path"])).name
    file_name = str(file_name) if file_name else "unknown"

    date_val = meta.get(METADATA_DATE)
    if not date_val:
        date_val = meta.get("last_modified_date") or meta.get("creation_date") or ""
    date_val = str(date_val) if date_val else ""

    node.metadata = {
        METADATA_SOURCE_TYPE: SOURCE_TYPE_LOCAL,
        METADATA_FILE_NAME: file_name,
        METADATA_DATE: date_val,
    }


def load_local_documents(data_dir: Path, *, recursive: bool = True) -> list[Document]:
    """Read all supported files under ``data_dir``.

    Args:
        data_dir: Directory to scan (must exist).
        recursive: Whether to walk subdirectories.

    Returns:
        Loaded LlamaIndex :class:`Document` instances (may be empty).

    Raises:
        ValueError: If ``data_dir`` is not a directory.
    """
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=recursive,
        required_exts=LOCAL_FILE_EXTENSIONS,
        raise_on_error=False,
        exclude_hidden=True,
    )
    return reader.load_data(show_progress=True)


def chunk_documents(documents: Sequence[Document]) -> list[BaseNode]:
    """Split documents into overlapping token windows.

    Args:
        documents: Source documents from :func:`load_local_documents`.

    Returns:
        Text nodes ready for embedding and upsert.
    """
    splitter = SentenceSplitter(
        chunk_size=LOCAL_DOC_CHUNK_SIZE,
        chunk_overlap=LOCAL_DOC_CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(list(documents))


def ingest_local(
    data_dir: Path,
    *,
    recursive: bool = True,
) -> tuple[int, int]:
    """Embed and upsert all eligible files from ``data_dir`` into Pinecone.

    Loads ``.env`` from the repository root before calling OpenAI / Pinecone.

    Args:
        data_dir: Folder containing PDFs, Word docs, and text/Markdown files.
        recursive: Pass-through to :func:`load_local_documents`.

    Returns:
        ``(num_source_documents, num_chunks_ingested)``. Both are zero if no
        files matched.

    Side effects:
        Calls OpenAI Embeddings API and writes vectors to Pinecone.

    Raises:
        ValueError: If configuration is invalid or ``data_dir`` is unusable.
        Exception: Network / API errors from OpenAI or Pinecone are propagated.
    """
    load_dotenv(dotenv_path=get_repo_root() / ".env", override=False)

    documents = load_local_documents(data_dir, recursive=recursive)
    if not documents:
        return (0, 0)

    nodes = chunk_documents(documents)
    for node in nodes:
        _normalize_chunk_metadata(node)

    embed_model = build_openai_embedding_model()
    vector_store = build_pinecone_vector_store()
    index = build_vector_store_index(
        embed_model=embed_model,
        vector_store=vector_store,
    )
    index.insert_nodes(nodes)

    return (len(documents), len(nodes))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest local PDF, DOCX, TXT, and Markdown into Pinecone."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory to scan (default: <repo>/data)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only read files in the top-level directory.",
    )
    args = parser.parse_args()

    root = get_repo_root()
    data_path = (args.path if args.path is not None else root / "data").resolve()

    print(f"→ Scanning {data_path} …")
    try:
        n_docs, n_chunks = ingest_local(
            data_path,
            recursive=not args.no_recursive,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - runtime/network
        print(f"❌ Ingestion failed: {exc}")
        sys.exit(2)

    if n_docs == 0:
        print(
            f"⚠️ No files found to ingest under:\n   {data_path}\n"
            f"   Supported extensions: {', '.join(LOCAL_FILE_EXTENSIONS)}\n"
            f"   Drop documents into that folder or use: "
            f"python -m ingest.ingest_local --path /path/to/docs"
        )
        sys.exit(0)

    print(
        f"✅ Ingested {n_docs} document(s) → {n_chunks} chunk(s) from {data_path}"
    )


if __name__ == "__main__":
    main()
