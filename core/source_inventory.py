"""Aggregate ingested sources from Pinecone by ``source_type`` (category).

Uses the Pinecone data plane ``list`` + ``fetch`` APIs to walk vector IDs in
the configured namespace and roll up ``file_name``, ``date``, and chunk
counts. Intended for a small-team KB; very large indexes may be slow — see
``SOURCE_INVENTORY_MAX_VECTORS``.

Also provides :func:`build_source_export` to merge all chunks for one
``(source_type, file_name)`` into a downloadable Markdown file (chunk order is
best-effort using LlamaIndex ``start_char_idx`` when present).

:func:`delete_source_vectors` walks the index the same way and removes every
vector that matches a logical source (all chunks for that file/category).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, DefaultDict, Optional, Tuple

from pinecone import Pinecone

from llama_index.core.vector_stores.utils import metadata_dict_to_node

from core.retriever import (
    DEFAULT_PINECONE_INDEX_NAME,
    ENV_PINECONE_API_KEY,
    ENV_PINECONE_INDEX_NAME,
    ENV_PINECONE_NAMESPACE,
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    ensure_env_loaded,
)

# Display titles for known ``source_type`` values from ingest pipelines.
CATEGORY_LABELS: dict[str, str] = {
    "local": "Documents",
    "notion": "Notion",
    "discord": "Discord",
    "transcript": "Sales call transcripts",
    "unknown": "Other",
}

LIST_PAGE_SIZE: int = 100
FETCH_BATCH_SIZE: int = 200
DELETE_VECTOR_BATCH: int = 1000

ENV_SOURCE_INVENTORY_MAX_VECTORS: str = "SOURCE_INVENTORY_MAX_VECTORS"


def _pinecone_index() -> Any:
    ensure_env_loaded()
    api_key = os.getenv(ENV_PINECONE_API_KEY)
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Add it to .env for source inventory."
        )
    index_name = os.getenv(ENV_PINECONE_INDEX_NAME) or DEFAULT_PINECONE_INDEX_NAME
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def _list_namespace_kwargs() -> dict[str, Any]:
    """Namespace args for list/fetch — match :func:`build_pinecone_vector_store`."""
    raw = os.getenv(ENV_PINECONE_NAMESPACE)
    if raw is None or str(raw).strip() == "":
        return {}
    return {"namespace": str(raw).strip()}


def _parse_vector_metadata(meta: Optional[dict[str, Any]]) -> Tuple[str, str, str]:
    """Return ``(source_type, file_name, date)`` from a Pinecone vector metadata dict."""
    if not meta:
        return ("unknown", "unknown", "")
    st = meta.get(METADATA_SOURCE_TYPE) or meta.get("source_type")
    fn = meta.get(METADATA_FILE_NAME) or meta.get("file_name")
    dt = meta.get(METADATA_DATE) or meta.get("date")
    return (
        str(st or "unknown"),
        str(fn or "unknown"),
        str(dt or ""),
    )


def _chunk_sort_key(meta: dict[str, Any], vector_id: str) -> tuple[Any, ...]:
    """Stable ordering for chunks of the same source (prefer document offset)."""
    raw = meta.get("_node_content")
    if isinstance(raw, str):
        try:
            node_dict = json.loads(raw)
            start = node_dict.get("start_char_idx")
            if start is not None:
                return (0, int(start), vector_id)
            md = node_dict.get("metadata") or {}
            if isinstance(md, dict) and md.get("start_char_idx") is not None:
                return (0, int(md["start_char_idx"]), vector_id)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return (1, vector_id)


def _chunk_text_from_metadata(meta: dict[str, Any]) -> str:
    """Recover chunk plain text from Pinecone/LlamaIndex metadata."""
    try:
        node = metadata_dict_to_node(meta)
        return (node.get_content() or "").strip()
    except Exception:
        return ""


def build_source_export(source_type: str, file_name: str) -> str:
    """Merge all Pinecone chunks for one logical source into one Markdown file body.

    Walks the **entire** index (ignores ``SOURCE_INVENTORY_MAX_VECTORS``) so the
    download is complete. Large indexes may take a long time or time out.

    Args:
        source_type: Ingest category (e.g. ``local``, ``notion``).
        file_name: Exact ``file_name`` metadata (e.g. ``Guide.md``).

    Returns:
        UTF-8 Markdown string with a short HTML comment header, then chunk texts
        separated by blank lines. Empty string if no matching chunks.

    Raises:
        ValueError: Missing Pinecone configuration or blank identifiers.
    """
    st_target = (source_type or "").strip()
    fn_target = (file_name or "").strip()
    if not st_target or not fn_target:
        raise ValueError("source_type and file_name must be non-empty.")
    if "\x00" in fn_target or "\n" in fn_target or "\r" in fn_target:
        raise ValueError("Invalid file_name.")

    index = _pinecone_index()
    ns_kw = _list_namespace_kwargs()
    list_kwargs: dict[str, Any] = {"limit": LIST_PAGE_SIZE, **ns_kw}

    collected: list[tuple[tuple[Any, ...], str]] = []

    for id_batch in index.list(**list_kwargs):
        if not id_batch:
            continue
        for i in range(0, len(id_batch), FETCH_BATCH_SIZE):
            slice_ids = id_batch[i : i + FETCH_BATCH_SIZE]
            fetched = index.fetch(ids=slice_ids, **ns_kw)
            vectors = getattr(fetched, "vectors", None) or {}
            for vid in slice_ids:
                vec = vectors.get(vid)
                meta_obj = getattr(vec, "metadata", None) if vec is not None else None
                if not isinstance(meta_obj, dict):
                    continue
                st, fn, _dt = _parse_vector_metadata(meta_obj)
                if st != st_target or fn != fn_target:
                    continue
                text = _chunk_text_from_metadata(meta_obj)
                if not text:
                    continue
                collected.append((_chunk_sort_key(meta_obj, vid), text))

    if not collected:
        return ""

    collected.sort(key=lambda x: x[0])
    parts = [c[1] for c in collected]
    header = (
        f"<!-- Nyck RAG export | source_type={st_target} | file_name={fn_target} "
        f"| chunks={len(parts)} -->\n\n"
    )
    return header + "\n\n".join(parts)


def delete_source_vectors(source_type: str, file_name: str) -> int:
    """Delete every Pinecone vector whose metadata matches one logical source.

    Uses the same full-index list/fetch walk as :func:`build_source_export`.
    Large indexes may take noticeable time.

    Args:
        source_type: Ingest category (e.g. ``local``, ``notion``).
        file_name: Exact ``file_name`` metadata for that source.

    Returns:
        Number of vectors removed (``0`` if none matched).

    Raises:
        ValueError: Missing Pinecone configuration, or blank/invalid identifiers.
    """
    st_target = (source_type or "").strip()
    fn_target = (file_name or "").strip()
    if not st_target or not fn_target:
        raise ValueError("source_type and file_name must be non-empty.")
    if "\x00" in fn_target or "\n" in fn_target or "\r" in fn_target:
        raise ValueError("Invalid file_name.")

    index = _pinecone_index()
    ns_kw = _list_namespace_kwargs()
    list_kwargs: dict[str, Any] = {"limit": LIST_PAGE_SIZE, **ns_kw}

    to_delete: list[str] = []

    for id_batch in index.list(**list_kwargs):
        if not id_batch:
            continue
        for i in range(0, len(id_batch), FETCH_BATCH_SIZE):
            slice_ids = id_batch[i : i + FETCH_BATCH_SIZE]
            fetched = index.fetch(ids=slice_ids, **ns_kw)
            vectors = getattr(fetched, "vectors", None) or {}
            for vid in slice_ids:
                vec = vectors.get(vid)
                meta_obj = getattr(vec, "metadata", None) if vec is not None else None
                if not isinstance(meta_obj, dict):
                    continue
                st, fn, _dt = _parse_vector_metadata(meta_obj)
                if st == st_target and fn == fn_target:
                    to_delete.append(vid)

    if not to_delete:
        return 0

    for i in range(0, len(to_delete), DELETE_VECTOR_BATCH):
        batch = to_delete[i : i + DELETE_VECTOR_BATCH]
        index.delete(ids=batch, **ns_kw)

    return len(to_delete)


def _safe_download_basename(file_name: str) -> str:
    """ASCII-ish filename for Content-Disposition (no path segments)."""
    base = os.path.basename(str(file_name).strip()) or "source"
    base = re.sub(r'[^\w.\- ()\[\]]+', "_", base, flags=re.UNICODE)
    base = base.strip(" .") or "source"
    if len(base) > 200:
        base = base[:200]
    if not base.lower().endswith((".md", ".txt", ".markdown")):
        base += ".md"
    return base


def source_export_attachment_filename(stored_file_name: str) -> str:
    """Safe ``filename=`` value for ``Content-Disposition`` (attachment)."""
    return _safe_download_basename(stored_file_name)


def _max_scan_limit() -> Optional[int]:
    raw = os.getenv(ENV_SOURCE_INVENTORY_MAX_VECTORS, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def build_source_inventory() -> dict[str, Any]:
    """Scan Pinecone and return sources grouped by category.

    Returns:
        A dict with:

        - ``categories`` — list of
          ``{category, label, total_chunks, sources: [{file_name, date, chunk_count}]}``
        - ``total_vectors_scanned`` — number of vectors visited
        - ``truncated`` — true if scanning stopped early due to
          ``SOURCE_INVENTORY_MAX_VECTORS``

    Side effects:
        Reads Pinecone (many list/fetch calls on large indexes).

    Raises:
        ValueError: Missing Pinecone configuration.
    """
    index = _pinecone_index()
    ns_kw = _list_namespace_kwargs()

    # (source_type, file_name) -> {chunk_count, date_max}
    rollup: DefaultDict[Tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"chunk_count": 0, "date_max": ""}
    )

    cap = _max_scan_limit()
    scanned = 0
    truncated = False

    list_kwargs: dict[str, Any] = {"limit": LIST_PAGE_SIZE, **ns_kw}

    for id_batch in index.list(**list_kwargs):
        if not id_batch:
            continue

        for i in range(0, len(id_batch), FETCH_BATCH_SIZE):
            if cap is not None and scanned >= cap:
                truncated = True
                break
            slice_ids = id_batch[i : i + FETCH_BATCH_SIZE]
            if cap is not None:
                remain = cap - scanned
                if len(slice_ids) > remain:
                    slice_ids = slice_ids[:remain]
                    truncated = True

            fetched = index.fetch(ids=slice_ids, **ns_kw)
            vectors = getattr(fetched, "vectors", None) or {}
            for vid in slice_ids:
                vec = vectors.get(vid)
                meta = getattr(vec, "metadata", None) if vec is not None else None
                if isinstance(meta, dict):
                    st, fn, dt = _parse_vector_metadata(meta)
                else:
                    st, fn, dt = ("unknown", "unknown", "")

                key = (st, fn)
                row = rollup[key]
                row["chunk_count"] += 1
                if dt and (not row["date_max"] or dt > row["date_max"]):
                    row["date_max"] = dt

            scanned += len(slice_ids)
            if truncated:
                break
        if truncated:
            break

    # Build categories
    by_type: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_totals: DefaultDict[str, int] = defaultdict(int)

    for (st, fn), data in rollup.items():
        chunk_totals[st] += int(data["chunk_count"])
        by_type[st].append(
            {
                "file_name": fn,
                "date": str(data["date_max"] or ""),
                "chunk_count": int(data["chunk_count"]),
            }
        )

    def type_sort_key(t: str) -> tuple[int, str]:
        order = list(CATEGORY_LABELS.keys())
        if t in order:
            return (order.index(t), t)
        return (len(order), t)

    categories: list[dict[str, Any]] = []
    for st in sorted(by_type.keys(), key=type_sort_key):
        rows = by_type[st]
        rows.sort(key=lambda r: (r["file_name"].lower(), r["date"]))
        label = CATEGORY_LABELS.get(st, st.replace("_", " ").title())
        categories.append(
            {
                "category": st,
                "label": label,
                "total_chunks": chunk_totals[st],
                "sources": rows,
            }
        )

    return {
        "categories": categories,
        "total_vectors_scanned": scanned,
        "truncated": truncated,
    }
