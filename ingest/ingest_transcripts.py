"""Ingest sales call transcripts (``.txt``) into Pinecone.

Filenames like ``2024-01-15_acme_call.txt`` contribute a ``date`` metadata
field. Chunking uses 1200 tokens / 150 overlap to keep conversational context.

Run::

    python -m ingest.ingest_transcripts
    python -m ingest.ingest_transcripts --path data/transcripts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

from dotenv import load_dotenv
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

TRANSCRIPT_CHUNK_SIZE: int = 1200
TRANSCRIPT_CHUNK_OVERLAP: int = 150

SOURCE_TYPE_TRANSCRIPT: str = "transcript"

# Date at start of filename: YYYY-MM-DD followed by _ or -
TRANSCRIPT_FILENAME_DATE: re.Pattern[str] = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?=[_-])"
)


def _date_from_filename(file_path: Path) -> str:
    """Extract ``YYYY-MM-DD`` from filenames such as ``2024-01-15_acme_call.txt``."""
    match = TRANSCRIPT_FILENAME_DATE.match(file_path.stem)
    return match.group(1) if match else ""


def _normalize_transcript_nodes(nodes: Sequence[BaseNode]) -> None:
    """Ensure standard metadata keys on each chunk."""
    for node in nodes:
        meta = dict(node.metadata)
        node.metadata = {
            METADATA_SOURCE_TYPE: SOURCE_TYPE_TRANSCRIPT,
            METADATA_FILE_NAME: str(meta.get(METADATA_FILE_NAME, "unknown")),
            METADATA_DATE: str(meta.get(METADATA_DATE, "")),
        }


def load_transcript_documents(transcripts_dir: Path) -> List[Document]:
    """Load every ``.txt`` file in the directory (non-recursive).

    Args:
        transcripts_dir: Folder containing ``.txt`` transcripts.

    Returns:
        One :class:`Document` per file.

    Raises:
        ValueError: If ``transcripts_dir`` is not a directory.
    """
    if not transcripts_dir.is_dir():
        raise ValueError(f"Not a directory: {transcripts_dir}")

    documents: List[Document] = []
    for path in sorted(transcripts_dir.glob("*.txt")):
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    METADATA_FILE_NAME: path.name,
                    METADATA_DATE: _date_from_filename(path),
                },
            )
        )
    return documents


def chunk_transcripts(documents: Sequence[Document]) -> List[BaseNode]:
    """Split transcripts with wider windows for dialogue context."""
    splitter = SentenceSplitter(
        chunk_size=TRANSCRIPT_CHUNK_SIZE,
        chunk_overlap=TRANSCRIPT_CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(list(documents))


def _upsert_nodes(nodes: List[BaseNode]) -> None:
    embed_model = build_openai_embedding_model()
    vector_store = build_pinecone_vector_store()
    index = build_vector_store_index(
        embed_model=embed_model,
        vector_store=vector_store,
    )
    index.insert_nodes(nodes)


def ingest_transcripts(
    transcripts_dir: Path,
    *,
    recursive: bool = False,
) -> tuple[int, int]:
    """Load transcripts, chunk, embed, and upsert.

    Args:
        transcripts_dir: Directory to scan for ``*.txt``.
        recursive: If True, include subfolders' ``*.txt`` (via ``rglob``).

    Returns:
        ``(num_files, num_chunks)``.

    Side effects:
        Calls OpenAI Embeddings + Pinecone.
    """
    load_dotenv(dotenv_path=get_repo_root() / ".env", override=False)

    if recursive:
        if not transcripts_dir.is_dir():
            raise ValueError(f"Not a directory: {transcripts_dir}")
        documents: List[Document] = []
        for path in sorted(transcripts_dir.rglob("*.txt")):
            if path.name.startswith("."):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            rel = path.relative_to(transcripts_dir)
            documents.append(
                Document(
                    text=text,
                    metadata={
                        METADATA_FILE_NAME: str(rel),
                        METADATA_DATE: _date_from_filename(path),
                    },
                )
            )
    else:
        documents = load_transcript_documents(transcripts_dir)

    if not documents:
        return (0, 0)
    nodes = chunk_transcripts(documents)
    _normalize_transcript_nodes(nodes)
    _upsert_nodes(nodes)
    return (len(documents), len(nodes))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest sales call transcript .txt files into Pinecone."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory of transcripts (default: <repo>/data/transcripts)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories for .txt files.",
    )
    args = parser.parse_args()

    root = get_repo_root()
    tdir = (args.path if args.path is not None else root / "data" / "transcripts")
    tdir = tdir.resolve()

    if not tdir.is_dir():
        print(
            f"⚠️ Directory does not exist:\n   {tdir}\n"
            f"   Create it and add .txt transcripts, e.g. "
            f"mkdir -p {tdir}"
        )
        sys.exit(0)

    print(f"→ Loading transcripts from {tdir} …")
    try:
        n_files, n_chunks = ingest_transcripts(tdir, recursive=args.recursive)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"❌ Transcript ingestion failed: {exc}")
        sys.exit(2)

    if n_files == 0:
        print(
            f"⚠️ No .txt transcripts found under {tdir}. "
            f"Add files or create the folder."
        )
        sys.exit(0)

    print(
        f"✅ Ingested {n_files} transcript(s) → {n_chunks} chunk(s) into Pinecone"
    )


if __name__ == "__main__":
    main()
