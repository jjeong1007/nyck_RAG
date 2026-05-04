"""Ingest Notion pages (and all pages in given databases) into Pinecone.

Uses the same chunking as internal docs: 800 tokens, 100 token overlap.

The LlamaIndex reader expects ``NOTION_INTEGRATION_TOKEN``; this module also
accepts ``NOTION_TOKEN`` from the project ``.env`` file.

Examples::

    python -m ingest.ingest_notion --page-ids abc123...
    python -m ingest.ingest_notion --database-ids def456...
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document
from llama_index.readers.notion import NotionPageReader

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_store_index,
    get_repo_root,
)

NOTION_VERSION: str = "2022-06-28"
NOTION_PAGE_URL_TEMPLATE: str = "https://api.notion.com/v1/pages/{page_id}"

# Same as internal / Notion-style docs in the product spec.
NOTION_CHUNK_SIZE: int = 800
NOTION_CHUNK_OVERLAP: int = 100

SOURCE_TYPE_NOTION: str = "notion"


def _resolve_notion_token() -> str:
    """Return the integration token from ``NOTION_TOKEN`` or legacy env name."""
    token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_INTEGRATION_TOKEN")
    if not token:
        raise ValueError(
            "Set NOTION_TOKEN (or NOTION_INTEGRATION_TOKEN) in .env with your "
            "Notion integration secret."
        )
    return token


def _notion_headers(token: str) -> dict[str, str]:
    """Headers for Notion REST calls."""
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
    }


def fetch_page_title_and_date(token: str, page_id: str) -> Tuple[str, str]:
    """Fetch human-readable title and last-edited date for metadata.

    Args:
        token: Notion integration token.
        page_id: UUID of the page (with or without hyphens).

    Returns:
        ``(title, date)`` where ``date`` is ``YYYY-MM-DD`` or ``""`` on failure.

    Side effects:
        Performs one HTTP GET to the Notion API.
    """
    pid = page_id.strip()
    url = NOTION_PAGE_URL_TEMPLATE.format(page_id=pid)
    try:
        resp = requests.get(
            url, headers=_notion_headers(token), timeout=60
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception:
        return (f"notion-page-{page_id[:8]}", "")

    raw_date = data.get("last_edited_time") or ""
    date_str = raw_date[:10] if raw_date else ""

    title = "untitled"
    for _prop_name, prop in (data.get("properties") or {}).items():
        if prop.get("type") == "title":
            parts = [
                t.get("plain_text", "")
                for t in (prop.get("title") or [])
                if isinstance(t, dict)
            ]
            merged = "".join(parts).strip()
            if merged:
                title = merged
            break

    # Safe file_name for metadata (flat string).
    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)[
        :120
    ].strip()
    if not safe_title:
        safe_title = f"notion-page-{page_id[:8]}"
    return (safe_title, date_str)


def _normalize_notion_nodes(nodes: Sequence[BaseNode]) -> None:
    """Force flat metadata keys required by the RAG stack."""
    for node in nodes:
        meta = dict(node.metadata)
        node.metadata = {
            METADATA_SOURCE_TYPE: SOURCE_TYPE_NOTION,
            METADATA_FILE_NAME: str(meta.get(METADATA_FILE_NAME, "notion")),
            METADATA_DATE: str(meta.get(METADATA_DATE, "")),
        }


def load_notion_documents(
    *,
    token: str,
    page_ids: Optional[List[str]] = None,
    database_ids: Optional[List[str]] = None,
) -> List[Document]:
    """Load Notion pages as LlamaIndex :class:`Document` objects.

    Args:
        token: Notion integration token.
        page_ids: Explicit page IDs to load.
        database_ids: Databases whose rows (pages) should be loaded.

    Returns:
        One document per page (full page text).

    Raises:
        ValueError: If both ID lists are empty.
    """
    page_ids = list(page_ids or [])
    database_ids = list(database_ids or [])
    if not page_ids and not database_ids:
        raise ValueError("Provide at least one page ID or database ID.")

    reader = NotionPageReader(integration_token=token)
    docs = reader.load_data(
        page_ids=page_ids,
        database_ids=database_ids or None,
    )

    # Enrich with title + date (best-effort, one GET per page).
    total = len(docs)
    for i, doc in enumerate(docs, start=1):
        pid = str(doc.id_ or doc.metadata.get("page_id", ""))
        print(f"→ Notion metadata {i}/{total} (page {pid[:8]}…)")
        title, date_str = fetch_page_title_and_date(token, pid)
        meta = dict(doc.metadata)
        meta[METADATA_FILE_NAME] = f"{title}.md"
        meta[METADATA_DATE] = date_str
        doc.metadata = meta

    return [d for d in docs if (d.text or "").strip()]


def chunk_notion_documents(documents: Sequence[Document]) -> List[BaseNode]:
    """Split Notion pages into token windows."""
    splitter = SentenceSplitter(
        chunk_size=NOTION_CHUNK_SIZE,
        chunk_overlap=NOTION_CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(list(documents))


def _upsert_nodes(nodes: List[BaseNode]) -> None:
    """Embed nodes and write to Pinecone."""
    embed_model = build_openai_embedding_model()
    vector_store = build_pinecone_vector_store()
    index = build_vector_store_index(
        embed_model=embed_model,
        vector_store=vector_store,
    )
    index.insert_nodes(nodes)


def ingest_notion(
    *,
    page_ids: Optional[List[str]] = None,
    database_ids: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Load Notion content, chunk, embed, and upsert into Pinecone.

    Args:
        page_ids: Notion page UUIDs to ingest.
        database_ids: Notion database UUIDs; all pages in each DB are ingested.

    Returns:
        ``(num_pages, num_chunks)``.

    Side effects:
        Reads ``.env``, calls OpenAI + Notion + Pinecone.
    """
    load_dotenv(dotenv_path=get_repo_root() / ".env", override=False)
    token = _resolve_notion_token()
    documents = load_notion_documents(
        token=token,
        page_ids=page_ids,
        database_ids=database_ids,
    )
    if not documents:
        return (0, 0)
    nodes = chunk_notion_documents(documents)
    _normalize_notion_nodes(nodes)
    _upsert_nodes(nodes)
    return (len(documents), len(nodes))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest Notion pages / databases into Pinecone."
    )
    parser.add_argument(
        "--page-ids",
        nargs="*",
        default=[],
        metavar="ID",
        help="Notion page IDs to ingest (space-separated).",
    )
    parser.add_argument(
        "--database-ids",
        nargs="*",
        default=[],
        metavar="ID",
        help="Notion database IDs — ingests every page in each database.",
    )
    args = parser.parse_args()

    pages = [p.strip() for p in args.page_ids if p.strip()]
    dbs = [d.strip() for d in args.database_ids if d.strip()]

    if not pages and not dbs:
        parser.error("Provide --page-ids and/or --database-ids")

    try:
        n_pages, n_chunks = ingest_notion(page_ids=pages, database_ids=dbs)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        print(f"❌ Notion ingestion failed: {exc}")
        sys.exit(2)

    if n_pages == 0:
        print("⚠️ No Notion pages produced text (empty or inaccessible).")
        sys.exit(0)

    print(
        f"✅ Ingested {n_pages} Notion page(s) → {n_chunks} chunk(s) into Pinecone"
    )


if __name__ == "__main__":
    main()
