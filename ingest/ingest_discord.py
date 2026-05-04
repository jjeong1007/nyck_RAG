"""Ingest Discord exports (DiscordChatExporter JSON) into Pinecone.

Expects ``.json`` files (e.g. from
`DiscordChatExporter <https://github.com/Tyrrrz/DiscordChatExporter>`_).
Messages are flattened to a single timeline per file. Chunking: 600 tokens,
80 overlap.

Run::

    python -m ingest.ingest_discord
    python -m ingest.ingest_discord --path data/discord
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence

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

DISCORD_CHUNK_SIZE: int = 600
DISCORD_CHUNK_OVERLAP: int = 80

SOURCE_TYPE_DISCORD: str = "discord"


def _normalize_discord_nodes(nodes: Sequence[BaseNode]) -> None:
    for node in nodes:
        meta = dict(node.metadata)
        node.metadata = {
            METADATA_SOURCE_TYPE: SOURCE_TYPE_DISCORD,
            METADATA_FILE_NAME: str(meta.get(METADATA_FILE_NAME, "unknown")),
            METADATA_DATE: str(meta.get(METADATA_DATE, "")),
        }


def _author_name(msg: dict[str, Any]) -> str:
    author = msg.get("author")
    if isinstance(author, dict):
        return str(
            author.get("nickname")
            or author.get("displayName")
            or author.get("name")
            or "unknown"
        )
    return "unknown"


def _message_timestamp(msg: dict[str, Any]) -> str:
    ts = msg.get("timestamp") or msg.get("Timestamp") or ""
    if isinstance(ts, datetime):
        return ts.date().isoformat()
    if isinstance(ts, str) and ts:
        return ts[:10] if len(ts) >= 10 else ts
    return ""


def _format_export_messages(messages: List[dict[str, Any]]) -> tuple[str, str]:
    """Build transcript text and best-effort date (YYYY-MM-DD) from messages."""
    lines: List[str] = []
    min_date = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        ts = _message_timestamp(msg)
        day = ts[:10] if len(ts) >= 10 else ""
        if day and (not min_date or day < min_date):
            min_date = day
        author = _author_name(msg)
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"[{ts}] {author}: {content}")
    return ("\n".join(lines), min_date)


def document_from_discord_json(path: Path) -> Document | None:
    """Parse one DiscordChatExporter JSON file into a :class:`Document`.

    Args:
        path: Path to ``.json`` export.

    Returns:
        Document or ``None`` if the file has no usable message text.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    data: Any = json.loads(raw)

    messages: List[dict[str, Any]] = []
    if isinstance(data, dict):
        raw_msgs = data.get("messages")
        if isinstance(raw_msgs, list):
            messages = [m for m in raw_msgs if isinstance(m, dict)]
    elif isinstance(data, list):
        messages = [m for m in data if isinstance(m, dict)]

    text, date_guess = _format_export_messages(messages)
    if not text.strip():
        return None

    channel_name = ""
    if isinstance(data, dict):
        ch = data.get("channel")
        if isinstance(ch, dict):
            channel_name = str(ch.get("name") or "")

    label = f"{path.stem}" + (f" ({channel_name})" if channel_name else "")
    return Document(
        text=text,
        metadata={
            METADATA_FILE_NAME: f"{label}.json",
            METADATA_DATE: date_guess,
        },
    )


def load_discord_documents(discord_dir: Path, *, recursive: bool = False) -> List[Document]:
    """Load all ``.json`` exports under ``discord_dir``.

    Raises:
        ValueError: If ``discord_dir`` is not a directory.
    """
    if not discord_dir.is_dir():
        raise ValueError(f"Not a directory: {discord_dir}")

    globber = discord_dir.rglob if recursive else discord_dir.glob
    documents: List[Document] = []
    for path in sorted(globber("*.json")):
        if path.name.startswith("."):
            continue
        try:
            doc = document_from_discord_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"⚠️ Skipping {path.name}: {exc}")
            continue
        if doc:
            documents.append(doc)
    return documents


def chunk_discord_documents(documents: Sequence[Document]) -> List[BaseNode]:
    splitter = SentenceSplitter(
        chunk_size=DISCORD_CHUNK_SIZE,
        chunk_overlap=DISCORD_CHUNK_OVERLAP,
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


def ingest_discord(
    discord_dir: Path,
    *,
    recursive: bool = False,
) -> tuple[int, int]:
    load_dotenv(dotenv_path=get_repo_root() / ".env", override=False)
    documents = load_discord_documents(discord_dir, recursive=recursive)
    if not documents:
        return (0, 0)
    nodes = chunk_discord_documents(documents)
    _normalize_discord_nodes(nodes)
    _upsert_nodes(nodes)
    return (len(documents), len(nodes))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest DiscordChatExporter JSON into Pinecone."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Folder with .json exports (default: <repo>/data/discord)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include .json files in subfolders.",
    )
    args = parser.parse_args()

    root = get_repo_root()
    ddir = (args.path if args.path is not None else root / "data" / "discord")
    ddir = ddir.resolve()

    if not ddir.is_dir():
        print(
            f"⚠️ Directory does not exist:\n   {ddir}\n"
            f"   Create it and add DiscordChatExporter .json files, e.g. "
            f"mkdir -p {ddir}"
        )
        sys.exit(0)

    print(f"→ Loading Discord exports from {ddir} …")
    try:
        n_files, n_chunks = ingest_discord(ddir, recursive=args.recursive)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"❌ Discord ingestion failed: {exc}")
        sys.exit(2)

    if n_files == 0:
        print(
            f"⚠️ No Discord .json exports found under {ddir}. "
            f"Drop DiscordChatExporter output here."
        )
        sys.exit(0)

    print(
        f"✅ Ingested {n_files} export(s) → {n_chunks} chunk(s) into Pinecone"
    )


if __name__ == "__main__":
    main()
