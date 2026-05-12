"""Notion ticket page template: read section structure from a Notion page.

Persisted locally (JSON) for API/config. Does **not** import ``ticket_chain`` —
``ticket_chain`` imports this module for prompts and Notion page bodies.

Requires ``NOTION_TOKEN`` and a template page shared with that integration.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import parse_qs, urlparse, unquote

import requests

from core.retriever import ensure_env_loaded, get_repo_root

ENV_TICKET_TEMPLATE_STORE: str = "TICKET_TEMPLATE_STORE"
DEFAULT_STORE_RELATIVE: str = "data/ticket_template.json"
ENV_NOTION_TOKEN: str = "NOTION_TOKEN"
# ``2025-09-03`` applies strict RFC ``uuid`` validation on paths; real ``notion.so``
# page ids often fail it. Use the same read version as ``ingest.ingest_notion``.
# Set ``NOTION_TEMPLATE_READ_API_VERSION`` to override (e.g. experiment with 2025).
ENV_NOTION_TEMPLATE_READ_VERSION: str = "NOTION_TEMPLATE_READ_API_VERSION"
NOTION_PAGE_URL: str = "https://api.notion.com/v1/pages/{page_id}"
BLOCKS_CHILDREN_URL: str = "https://api.notion.com/v1/blocks/{block_id}/children"
NOTION_TEXT_MAX: int = 1900

_UUID_DASHED = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_UUID_PLAIN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)


def template_store_path() -> Path:
    raw = (os.getenv(ENV_TICKET_TEMPLATE_STORE) or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = get_repo_root() / p
        return p
    return get_repo_root() / DEFAULT_STORE_RELATIVE


def _notion_read_version() -> str:
    ensure_env_loaded()
    v = (os.getenv(ENV_NOTION_TEMPLATE_READ_VERSION) or "").strip()
    return v or "2022-06-28"


def _headers(token: str, *, json_body: bool = False) -> dict[str, str]:
    """Headers for Notion REST. Avoid ``Content-Type`` on GET — some proxies reject it."""
    h: dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": _notion_read_version(),
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _notion_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            code = body.get("code") or ""
            msg = body.get("message") or ""
            if code and msg:
                return f"{code}: {msg}"
            return str(msg or code or body)
    except (ValueError, json.JSONDecodeError):
        pass
    text = (resp.text or "").strip()
    return text[:800] if text else resp.reason or "Unknown error"


def resolve_notion_token() -> str:
    token = (
        os.getenv(ENV_NOTION_TOKEN) or os.getenv("NOTION_INTEGRATION_TOKEN") or ""
    ).strip()
    if not token:
        raise ValueError(
            "Set NOTION_TOKEN in .env so the integration can read the template page."
        )
    return token


def extract_page_id(url_or_id: str) -> str:
    """Return a dashed lower-case UUID from a Notion URL or raw id.

    Database/board URLs often put a *collection view* id in ``?v=...``. That id is
    not valid for ``/blocks/{{id}}/children``, so we prefer UUIDs from the **path**
    (and explicit ``?p=``) over arbitrary query-string matches.
    """
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Template URL or page id is empty.")

    def _to_dashed(hex32: str) -> str:
        h = hex32.lower()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    def _from_last_path_segment(path_and_maybe: str) -> Optional[str]:
        """Prefer the page id at the end of the last ``/`` segment (Notion ``…/Title-hex32``)."""
        segments = [
            unquote(p)
            for p in path_and_maybe.strip("/").split("/")
            if p
        ]
        if not segments:
            return None
        last = segments[-1].strip()
        if not last:
            return None
        # ``PageName-e33d09f17af22807d8b25cc9f3255023`` — trailing 32 hex
        suf = re.search(r"([0-9a-f]{32})\s*$", last, re.IGNORECASE)
        if suf:
            return _to_dashed(suf.group(1).lower())
        # Entire segment is a dashed id
        if re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            last,
            re.IGNORECASE,
        ):
            return last.lower()
        return None

    if "://" in s:
        parsed = urlparse(s)
        qs = parse_qs(parsed.query)
        if "p" in qs and qs["p"]:
            raw_p = (qs["p"][0] or "").strip()
            dm = _UUID_DASHED.search(raw_p)
            if dm:
                return dm.group(1).lower()
            plain_p = re.sub(r"-", "", raw_p)
            pm = _UUID_PLAIN.search(plain_p)
            if pm and len(pm.group(0)) == 32:
                return _to_dashed(pm.group(0))
        path_part = (parsed.path or "").strip()
        path_and_maybe = path_part.split("#")[0]
        seg_id = _from_last_path_segment(path_and_maybe)
        if seg_id:
            return seg_id
    else:
        path_and_maybe = s.split("?")[0].split("#")[0]

    m = _UUID_DASHED.search(path_and_maybe)
    if m:
        return m.group(1).lower()
    compact = re.sub(r"-", "", path_and_maybe)
    m2 = _UUID_PLAIN.search(compact)
    if m2:
        h = m2.group(0).lower()
        if len(h) == 32:
            return _to_dashed(h)

    m = _UUID_DASHED.search(s)
    if m:
        return m.group(1).lower()
    compact = re.sub(r"-", "", s)
    m2 = _UUID_PLAIN.search(compact)
    if m2:
        h = m2.group(0).lower()
        if len(h) == 32:
            return _to_dashed(h)
    raise ValueError(
        "Could not find a Notion page UUID in the input. "
        "Use a normal **page** link (not only a database board ``?v=`` link)."
    )


def _rich_text_to_plain(rich: Any) -> str:
    if not isinstance(rich, list):
        return ""
    parts: List[str] = []
    for seg in rich:
        if not isinstance(seg, dict):
            continue
        if seg.get("type") == "text":
            t = (seg.get("text") or {}).get("content")
            if isinstance(t, str):
                parts.append(t)
        pt = seg.get("plain_text")
        if isinstance(pt, str) and seg.get("type") != "text":
            parts.append(pt)
    return "".join(parts).strip()


def _block_plain_text(block: dict[str, Any]) -> str:
    t = block.get("type")
    if not isinstance(t, str):
        return ""
    if t in ("child_page", "unsupported", "divider", "table_of_contents"):
        return ""
    inner = block.get(t) if isinstance(block.get(t), dict) else {}
    if "rich_text" in inner:
        return _rich_text_to_plain(inner.get("rich_text"))
    return ""


def fetch_block_children(token: str, block_id: str) -> List[dict[str, Any]]:
    """All direct children of a block (or page root), paginated."""
    out: List[dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: dict[str, str] = {"page_size": "100"}
        if cursor:
            params["start_cursor"] = cursor
        r = requests.get(
            BLOCKS_CHILDREN_URL.format(block_id=block_id),
            headers=_headers(token),
            params=params,
            timeout=120,
        )
        if not r.ok:
            raise ValueError(
                f"Notion API error ({r.status_code}) retrieving block children: "
                f"{_notion_error_message(r)}"
            )
        data = r.json()
        for b in data.get("results") or []:
            if isinstance(b, dict):
                out.append(b)
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break
    return out


def parse_sections_from_blocks(blocks: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """One section per top-level heading_1 / heading_2; last section not fillable."""
    sections: List[dict[str, Any]] = []
    for b in blocks:
        t = b.get("type")
        if t not in ("heading_1", "heading_2"):
            continue
        level = 1 if t == "heading_1" else 2
        title = _block_plain_text(b)
        if not title:
            continue
        sections.append({"title": title, "heading_level": level, "fillable": True})
    if sections:
        sections[-1]["fillable"] = False
    for i, s in enumerate(sections):
        s["index"] = i
    return sections


def load_template() -> Optional[dict[str, Any]]:
    path = template_store_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    secs = data.get("sections")
    if not isinstance(secs, list) or not secs:
        return None
    return data


def clear_template() -> None:
    template_store_path().unlink(missing_ok=True)


def verify_notion_page_accessible(token: str, page_id: str) -> None:
    """Confirm ``page_id`` is a **page** the integration can read (not e.g. ``?v=`` id)."""
    url = NOTION_PAGE_URL.format(page_id=page_id)
    r = requests.get(url, headers=_headers(token), timeout=60)
    if not r.ok:
        raise ValueError(
            f"Notion API error ({r.status_code}) opening page {page_id[:8]}… — "
            f"{_notion_error_message(r)}. "
            "Use a **page** URL from the address bar or Share → Copy link, "
            "and give your **integration** access to that page."
        )


def set_template_from_page_url(url_or_id: str) -> dict[str, Any]:
    """Fetch top-level headings from the page, persist template JSON."""
    token = resolve_notion_token()
    page_id = extract_page_id(url_or_id)
    verify_notion_page_accessible(token, page_id)
    blocks = fetch_block_children(token, page_id)
    sections = parse_sections_from_blocks(blocks)
    if not sections:
        raise ValueError(
            "No heading_1 or heading_2 blocks found on the template page. "
            "Add section headings in Notion, then try again."
        )
    payload: dict[str, Any] = {
        "page_id": page_id,
        "source_input": (url_or_id or "").strip()[:500],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }
    path = template_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def patch_template_section_fillable(index: int, fillable: bool) -> dict[str, Any]:
    tmpl = load_template()
    if not tmpl:
        raise ValueError("No ticket template is configured.")
    found = False
    for sec in tmpl.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        if int(sec.get("index", -1)) == index:
            sec["fillable"] = bool(fillable)
            found = True
            break
    if not found:
        raise ValueError(f"No section with index {index}.")
    tmpl["updated_at"] = datetime.now(timezone.utc).isoformat()
    template_store_path().write_text(
        json.dumps(tmpl, indent=2), encoding="utf-8"
    )
    return tmpl


def _split_paragraphs(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    out = [p.strip().replace("\n", " ") for p in parts if p.strip()]
    return out if out else [text]


def _paragraph_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": text[:NOTION_TEXT_MAX]}}
            ]
        },
    }


def build_notion_children_from_template(
    template: dict[str, Any],
    template_sections: dict[str, str],
) -> List[dict[str, Any]]:
    """Section headings plus paragraph bodies for fillable sections."""
    children: List[dict[str, Any]] = []
    for sec in template.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        idx = int(sec.get("index", -1))
        title = str(sec.get("title") or "").strip() or f"Section {idx}"
        level = int(sec.get("heading_level") or 2)
        htype = "heading_1" if level == 1 else "heading_2"
        children.append(
            {
                "object": "block",
                "type": htype,
                htype: {
                    "rich_text": [
                        {"type": "text", "text": {"content": title[:400]}}
                    ]
                },
            }
        )
        key = str(idx)
        if sec.get("fillable"):
            body = (template_sections.get(key) or "").strip()
            if not body:
                body = "—"
            for para in _split_paragraphs(body):
                children.append(_paragraph_block(para))
    return children


def ticket_template_prompt_suffix() -> str:
    """Extra system-prompt rules when a template is on disk."""
    tmpl = load_template()
    if not tmpl:
        return ""
    sections = tmpl.get("sections") or []
    fillable_keys = [
        str(s["index"])
        for s in sections
        if isinstance(s, dict) and s.get("fillable")
    ]
    lines: List[str] = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        flag = "fillable" if s.get("fillable") else "not fillable — omit key"
        lines.append(
            f'- Index {s.get("index")}: "{s.get("title", "")}" ({flag})'
        )
    return (
        "ACTIVE NOTION PAGE TEMPLATE\n"
        "The new ticket's Notion page body will use these sections in order. "
        'You MUST include top-level key "template_sections": an object whose '
        "keys are string indices "
        f"{json.dumps(fillable_keys)} for fillable sections only. "
        "Each value is Markdown or plain text for the body under that heading "
        "(do not repeat the heading title). For non-fillable sections, omit "
        "the key entirely.\n"
        "Sections:\n" + "\n".join(lines)
    )
