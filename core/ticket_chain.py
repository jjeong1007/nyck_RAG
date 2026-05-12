"""Ticket chain: turn a free-form description into a structured Notion ticket.

Pipeline:
    1. Retrieve top-6 chunks from Pinecone (similar past tickets / docs / calls).
    2. Call Claude Haiku via the **direct Anthropic SDK** (not LlamaIndex) with
       a JSON-only system prompt and parse the response robustly.
    3. Optionally create a page in the Notion database referenced by
       ``NOTION_TICKET_DB_ID`` (API ``2025-09-03``; multi-source DBs need
       ``NOTION_TICKET_DATA_SOURCE_ID`` when there is more than one source).
    4. Use ``NOTION_TICKET_FORMAT=roadmap`` when the target DB uses a Product
       Roadmap schema (Category / Ovr Status / numeric Prio).

Public API:
    - :class:`TicketChain` — reusable engine for the FastAPI app.
    - :func:`generate_ticket` — convenience wrapper.
    - :func:`push_ticket_to_notion` — standalone helper to write to Notion.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import anthropic
from notion_client import Client as NotionClient
from notion_client.errors import APIResponseError

from core.qa_chain import _trim_chat_history

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_index_retriever,
    ensure_env_loaded,
)
from core.notion_ticket_template import (
    build_notion_children_from_template,
    load_template,
    ticket_template_prompt_suffix,
)
from ingest.ingest_notion import SOURCE_TYPE_NOTION_TICKET_EXAMPLE
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

# ----- Constants -------------------------------------------------------------


CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
CLAUDE_TEMPERATURE: float = 0.2
CLAUDE_MAX_TOKENS: int = 2048

TICKET_TOP_K: int = 6
TICKET_EXAMPLE_TOP_K: int = 3
CONTEXT_PREVIEW_CHARS: int = 800

ENV_ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"
ENV_NOTION_TOKEN: str = "NOTION_TOKEN"
ENV_NOTION_TICKET_DB_ID: str = "NOTION_TICKET_DB_ID"
# Optional: required when the ticket database has more than one “data source”.
ENV_NOTION_TICKET_DATA_SOURCE_ID: str = "NOTION_TICKET_DATA_SOURCE_ID"

# Notion multi-source databases require API 2025-09-03 + data_source_id parent.
# https://developers.notion.com/guides/get-started/upgrade-guide-2025-09-03
NOTION_API_VERSION: str = "2025-09-03"

# Allowed enums – validated server-side and used to coerce model output.
TICKET_TYPES: tuple[str, ...] = ("feature", "bug", "improvement", "research")
TICKET_PRIORITIES: tuple[str, ...] = ("urgent", "high", "medium", "low")
TICKET_STATUS_DEFAULT: str = "Not started"
EFFORT_OPTIONS: tuple[str, ...] = (
    "small (< 1 day)",
    "medium (1-3 days)",
    "large (3+ days)",
)

# ``classic``: Type/Priority/Status selects + Estimated Effort + Tags (README defaults).
# ``roadmap``: Name (title) + Category (select) + Ovr Status (status) + Prio (number),
# plus optional sprint/epic/email columns when NOTION_ROADMAP_DEFAULT_* is set.
ENV_NOTION_TICKET_FORMAT: str = "NOTION_TICKET_FORMAT"
FORMAT_CLASSIC: str = "classic"
FORMAT_ROADMAP: str = "roadmap"

# Roadmap layout — column names (override if your Notion headers differ slightly).
ENV_ROADMAP_PROP_NAME: str = "NOTION_ROADMAP_PROP_NAME"
ENV_ROADMAP_PROP_CATEGORY: str = "NOTION_ROADMAP_PROP_CATEGORY"
ENV_ROADMAP_PROP_OVR_STATUS: str = "NOTION_ROADMAP_PROP_OVR_STATUS"
# If set, this **exact** Notion status option is sent for ``Ovr Status`` (roadmap mode).
# Use when your DB uses e.g. ``To be described`` instead of the model default ``Not started``.
ENV_ROADMAP_OVR_STATUS_OPTION: str = "NOTION_ROADMAP_OVR_STATUS_OPTION"
ENV_ROADMAP_PROP_PRIO: str = "NOTION_ROADMAP_PROP_PRIO"
ENV_ROADMAP_PROP_CURRENT_SPRINT: str = "NOTION_ROADMAP_PROP_CURRENT_SPRINT"
ENV_ROADMAP_PROP_ORIGINAL_SPRINT: str = "NOTION_ROADMAP_PROP_ORIGINAL_SPRINT"
ENV_ROADMAP_PROP_EPIC: str = "NOTION_ROADMAP_PROP_EPIC"
ENV_ROADMAP_PROP_EPIC_TIMING: str = "NOTION_ROADMAP_PROP_EPIC_TIMING"
ENV_ROADMAP_PROP_INCLUDE_EMAIL: str = "NOTION_ROADMAP_PROP_INCLUDE_EMAIL"
ENV_ROADMAP_PROP_EMAIL_SENT: str = "NOTION_ROADMAP_PROP_EMAIL_SENT"
ENV_ROADMAP_PROP_ENG_DUE: str = "NOTION_ROADMAP_PROP_ENG_DUE"
ENV_ROADMAP_PROP_CUST_RELEASE: str = "NOTION_ROADMAP_PROP_CUST_RELEASE"
ENV_ROADMAP_PROP_DEV_OWNER: str = "NOTION_ROADMAP_PROP_DEV_OWNER"

ENV_ROADMAP_DEFAULT_CURRENT_SPRINT: str = "NOTION_ROADMAP_DEFAULT_CURRENT_SPRINT"
ENV_ROADMAP_DEFAULT_ORIGINAL_SPRINT: str = "NOTION_ROADMAP_DEFAULT_ORIGINAL_SPRINT"
ENV_ROADMAP_DEFAULT_EPIC: str = "NOTION_ROADMAP_DEFAULT_EPIC"
ENV_ROADMAP_DEFAULT_EPIC_TIMING: str = "NOTION_ROADMAP_DEFAULT_EPIC_TIMING"
ENV_ROADMAP_DEFAULT_INCLUDE_IN_EMAIL: str = "NOTION_ROADMAP_DEFAULT_INCLUDE_IN_EMAIL"
ENV_ROADMAP_DEFAULT_EMAIL_SENT: str = "NOTION_ROADMAP_DEFAULT_EMAIL_SENT"
ENV_ROADMAP_DEFAULT_ENG_DUE: str = "NOTION_ROADMAP_DEFAULT_ENG_DUE"
ENV_ROADMAP_DEFAULT_CUST_RELEASE: str = "NOTION_ROADMAP_DEFAULT_CUST_RELEASE"
# Comma-separated Notion user UUIDs for the ``people`` property (optional).
ENV_ROADMAP_DEV_OWNER_IDS: str = "NOTION_ROADMAP_DEV_OWNER_IDS"

ROADMAP_DEFAULT_NAME: str = "Name"
ROADMAP_DEFAULT_CATEGORY: str = "Category"
ROADMAP_DEFAULT_OVR_STATUS: str = "Ovr Status"
ROADMAP_DEFAULT_PRIO: str = "Prio"
ROADMAP_DEFAULT_CURRENT_SPRINT: str = "Current Sprint"
ROADMAP_DEFAULT_ORIGINAL_SPRINT: str = "Original Sprint"
ROADMAP_DEFAULT_EPIC: str = "Epic"
ROADMAP_DEFAULT_EPIC_TIMING: str = "Epic timing"
ROADMAP_DEFAULT_INCLUDE_EMAIL: str = "Include in Email"
ROADMAP_DEFAULT_EMAIL_SENT: str = "Email sent in"
ROADMAP_DEFAULT_ENG_DUE: str = "Eng. Due Date"
ROADMAP_DEFAULT_CUST_RELEASE: str = "Cust. Release Date"
ROADMAP_DEFAULT_DEV_OWNER: str = "Dev Owner"

# Map ticket.priority strings → numeric ``Prio`` (lower = higher urgency).
ROADMAP_PRIO_BY_LEVEL: dict[str, int] = {
    "urgent": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
}

# JSON map of display name (or email) → Notion user id for ``Dev Owner`` (people).
ENV_ROADMAP_PEOPLE_MAP: str = "NOTION_ROADMAP_PEOPLE_MAP"
# Comma-separated sprint names; injected into the model prompt so it picks exact options.
ENV_ROADMAP_SPRINT_OPTIONS: str = "NOTION_ROADMAP_SPRINT_OPTIONS"

# Optional: map to your Notion DB column names if they differ from the defaults.
ENV_NOTION_PROP_NAME: str = "NOTION_PROP_NAME"
ENV_NOTION_PROP_TYPE: str = "NOTION_PROP_TYPE"
ENV_NOTION_PROP_PRIORITY: str = "NOTION_PROP_PRIORITY"
ENV_NOTION_PROP_STATUS: str = "NOTION_PROP_STATUS"
ENV_NOTION_PROP_EFFORT: str = "NOTION_PROP_EFFORT"
ENV_NOTION_PROP_TAGS: str = "NOTION_PROP_TAGS"

# Notion property names — must match the database schema (see README).
PROP_NAME: str = "Name"
PROP_TYPE: str = "Type"
PROP_PRIORITY: str = "Priority"
PROP_STATUS: str = "Status"
PROP_EFFORT: str = "Estimated Effort"
PROP_TAGS: str = "Tags"

NOTION_PARAGRAPH_MAX: int = 1900  # Notion limit is 2000 rich-text chars per block


TICKET_SYSTEM_PROMPT: str = (
    "You generate product tickets for an 8-person company's Notion database. "
    "Prior conversation turns may include earlier descriptions and previously "
    "generated ticket JSON. Use them only to interpret follow-ups (e.g. "
    "clarifications, extra acceptance criteria). Each reply must still be ONE "
    "new JSON object matching the schema below for the latest teammate "
    "description and the retrieved context in the final message — do not "
    "paste old JSON wholesale unless the user clearly asks to revise the same "
    "ticket without new information.\n\n"
    "Output ONLY a single JSON object — no prose, no Markdown fences, no "
    "explanation. The JSON must match this exact schema:\n"
    "{\n"
    '  "title": string (max 80 chars, action-oriented),\n'
    '  "type": one of "feature" | "bug" | "improvement" | "research",\n'
    '  "priority": one of "urgent" | "high" | "medium" | "low",\n'
    '  "status": "Not started",\n'
    '  "description": 2-3 sentences,\n'
    '  "acceptance_criteria": array of 2-5 short strings,\n'
    '  "related_context": brief note on which past tickets/docs informed this,\n'
    '  "estimated_effort": one of "small (< 1 day)" | "medium (1-3 days)" | '
    '"large (3+ days)",\n'
    '  "tags": array of 1-5 short lowercase strings,\n'
    '  "current_sprint": string (see user message; use \"\" if unknown),\n'
    '  "dev_owner": string (see user message; use \"\" if unknown)\n'
    "}\n"
    "Use the provided context to inform priority, effort, related_context, "
    "current_sprint, and dev_owner when evidence exists. "
    "If the context is unrelated, set related_context to "
    "\"No directly related context found.\""
)


TICKET_USER_MESSAGE_SUFFIX: str = (
    "\n\nRules for optional fields:\n"
    "- current_sprint: If CONSTRAINTS list allowed sprint names below, you MUST "
    "set current_sprint to one of those strings exactly, or \"\" if unclear.\n"
    "- dev_owner: If CONSTRAINTS list allowed owner keys below, you MUST set "
    "dev_owner to one of those keys exactly (spelling/capitalization), or \"\" "
    "if unclear. Never invent a name that is not listed.\n"
)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Cached Notion ticket data-source schema (properties + select/status options).
_SCHEMA_CACHE: Optional[Tuple[float, dict[str, Any]]] = None
SCHEMA_CACHE_TTL_SEC: float = 300.0


# ----- Public types ----------------------------------------------------------


@dataclass
class TicketContext:
    """Retrieved chunk surfaced as supporting context for ticket generation."""

    source_type: str
    file_name: str
    date: str
    score: float
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "file_name": self.file_name,
            "date": self.date,
            "score": self.score,
            "preview": self.preview,
        }


@dataclass
class Ticket:
    """Validated ticket payload returned to the API client."""

    title: str
    type: str
    priority: str
    status: str
    description: str
    acceptance_criteria: List[str]
    related_context: str
    estimated_effort: str
    tags: List[str] = field(default_factory=list)
    current_sprint: str = ""
    dev_owner: str = ""
    epic: str = ""
    epic_timing: str = ""
    original_sprint: str = ""
    include_in_email: str = ""
    email_sent_in: str = ""
    eng_due_date: str = ""
    cust_release_date: str = ""
    uses_template: bool = False
    template_sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "description": self.description,
            "acceptance_criteria": list(self.acceptance_criteria),
            "related_context": self.related_context,
            "estimated_effort": self.estimated_effort,
            "tags": list(self.tags),
            "current_sprint": self.current_sprint,
            "dev_owner": self.dev_owner,
            "epic": self.epic,
            "epic_timing": self.epic_timing,
            "original_sprint": self.original_sprint,
            "include_in_email": self.include_in_email,
            "email_sent_in": self.email_sent_in,
            "eng_due_date": self.eng_due_date,
            "cust_release_date": self.cust_release_date,
            "uses_template": self.uses_template,
            "template_sections": dict(self.template_sections),
        }

    def to_markdown(self) -> str:
        """Render the ticket as Markdown for copy/paste in Linear, Slack, etc."""
        tmpl = load_template()
        if self.uses_template and tmpl and tmpl.get("sections"):
            meta_bits_tpl: List[str] = [
                f"**Category / type:** {self.type}",
                f"**Priority:** {self.priority}",
                f"**Status:** {self.status}",
                f"**Effort:** {self.estimated_effort}",
            ]
            if self.current_sprint.strip():
                meta_bits_tpl.append(f"**Current sprint:** {self.current_sprint}")
            if self.original_sprint.strip():
                meta_bits_tpl.append(f"**Original sprint:** {self.original_sprint}")
            if self.epic.strip():
                meta_bits_tpl.append(f"**Epic:** {self.epic}")
            if self.epic_timing.strip():
                meta_bits_tpl.append(f"**Epic timing:** {self.epic_timing}")
            if self.dev_owner.strip():
                meta_bits_tpl.append(f"**Dev owner:** {self.dev_owner}")
            if self.include_in_email.strip():
                meta_bits_tpl.append(f"**Include in email:** {self.include_in_email}")
            if self.email_sent_in.strip():
                meta_bits_tpl.append(f"**Email sent:** {self.email_sent_in}")
            if self.eng_due_date.strip():
                meta_bits_tpl.append(f"**Eng. due:** {self.eng_due_date}")
            if self.cust_release_date.strip():
                meta_bits_tpl.append(f"**Cust. release:** {self.cust_release_date}")
            meta_line_tpl = "  \n".join(meta_bits_tpl)
            body_parts: List[str] = [
                f"# {self.title}\n\n",
                f"{meta_line_tpl}\n\n",
            ]
            for sec in tmpl.get("sections") or []:
                if not isinstance(sec, dict):
                    continue
                title = str(sec.get("title") or "").strip()
                idx = str(sec.get("index", ""))
                level = int(sec.get("heading_level") or 2)
                prefix = "##" if level == 2 else "#"
                body_parts.append(f"{prefix} {title}\n\n")
                if sec.get("fillable"):
                    content = (self.template_sections.get(idx) or "").strip()
                    body_parts.append((content or "_—_") + "\n\n")
            tags_tpl = ", ".join(self.tags) if self.tags else "_none_"
            body_parts.append(
                f"**Related context:** {self.related_context}\n\n"
                f"**Tags:** {tags_tpl}\n"
            )
            return "".join(body_parts)
        bullets = "\n".join(f"- {c}" for c in self.acceptance_criteria) or "- _none_"
        tags = ", ".join(self.tags) if self.tags else "_none_"
        meta_bits: List[str] = [
            f"**Category / type:** {self.type}",
            f"**Priority:** {self.priority}",
            f"**Status:** {self.status}",
            f"**Effort:** {self.estimated_effort}",
        ]
        if self.current_sprint.strip():
            meta_bits.append(f"**Current sprint:** {self.current_sprint}")
        if self.original_sprint.strip():
            meta_bits.append(f"**Original sprint:** {self.original_sprint}")
        if self.epic.strip():
            meta_bits.append(f"**Epic:** {self.epic}")
        if self.epic_timing.strip():
            meta_bits.append(f"**Epic timing:** {self.epic_timing}")
        if self.dev_owner.strip():
            meta_bits.append(f"**Dev owner:** {self.dev_owner}")
        if self.include_in_email.strip():
            meta_bits.append(f"**Include in email:** {self.include_in_email}")
        if self.email_sent_in.strip():
            meta_bits.append(f"**Email sent:** {self.email_sent_in}")
        if self.eng_due_date.strip():
            meta_bits.append(f"**Eng. due:** {self.eng_due_date}")
        if self.cust_release_date.strip():
            meta_bits.append(f"**Cust. release:** {self.cust_release_date}")
        meta_line = "  \n".join(meta_bits)
        return (
            f"# {self.title}\n\n"
            f"{meta_line}\n\n"
            f"{self.description}\n\n"
            f"## Acceptance Criteria\n{bullets}\n\n"
            f"**Related context:** {self.related_context}\n\n"
            f"**Tags:** {tags}\n"
        )


# ----- JSON parsing helpers --------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fences(raw: str) -> str:
    """Return the largest plausible JSON blob from ``raw``.

    Tolerates accidental Markdown code fences, leading prose, or trailing
    chatter the model may emit despite the JSON-only instruction.
    """
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Slice from the first ``{`` to the matching closing ``}``.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _coerce_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    return str(value).strip()


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_coerce_str(item) for item in value if _coerce_str(item)]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _coerce_enum(value: Any, allowed: Iterable[str], *, fallback: str) -> str:
    text = _coerce_str(value).lower()
    for option in allowed:
        if text == option.lower():
            return option
    # Loose match on prefix (e.g. "small" -> "small (< 1 day)").
    for option in allowed:
        if text and option.lower().startswith(text):
            return option
    return fallback


def _match_notion_option(
    value: Any, options: Sequence[str], *, fallback: str
) -> str:
    """Pick the first Notion option that matches ``value`` (case / fuzzy)."""
    raw = _coerce_str(value)
    if not raw:
        return fallback
    opts = [o for o in options if isinstance(o, str) and o.strip()]
    if not opts:
        return raw if raw else fallback
    if raw in opts:
        return raw
    rl = raw.casefold()
    for o in opts:
        if o.casefold() == rl:
            return o
    for o in opts:
        ol = o.casefold()
        if rl in ol or ol in rl:
            return o
    return fallback


def _roadmap_column_ticket_key(column_name: str) -> Optional[str]:
    """Map a Notion roadmap column label to a :class:`Ticket` field name."""
    pairs: List[Tuple[str, Optional[str]]] = [
        (
            _notion_prop_column(ENV_ROADMAP_PROP_CATEGORY, ROADMAP_DEFAULT_CATEGORY),
            "type",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_OVR_STATUS, ROADMAP_DEFAULT_OVR_STATUS
            ),
            "status",
        ),
        (
            _notion_prop_column(ENV_ROADMAP_PROP_PRIO, ROADMAP_DEFAULT_PRIO),
            None,
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_CURRENT_SPRINT, ROADMAP_DEFAULT_CURRENT_SPRINT
            ),
            "current_sprint",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_ORIGINAL_SPRINT, ROADMAP_DEFAULT_ORIGINAL_SPRINT
            ),
            "original_sprint",
        ),
        (
            _notion_prop_column(ENV_ROADMAP_PROP_EPIC, ROADMAP_DEFAULT_EPIC),
            "epic",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_EPIC_TIMING, ROADMAP_DEFAULT_EPIC_TIMING
            ),
            "epic_timing",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_INCLUDE_EMAIL, ROADMAP_DEFAULT_INCLUDE_EMAIL
            ),
            "include_in_email",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_EMAIL_SENT, ROADMAP_DEFAULT_EMAIL_SENT
            ),
            "email_sent_in",
        ),
        (
            _notion_prop_column(ENV_ROADMAP_PROP_ENG_DUE, ROADMAP_DEFAULT_ENG_DUE),
            "eng_due_date",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_CUST_RELEASE, ROADMAP_DEFAULT_CUST_RELEASE
            ),
            "cust_release_date",
        ),
        (
            _notion_prop_column(
                ENV_ROADMAP_PROP_DEV_OWNER, ROADMAP_DEFAULT_DEV_OWNER
            ),
            "dev_owner",
        ),
    ]
    needle = column_name.strip().casefold()
    for col, key in pairs:
        if col.strip().casefold() == needle:
            return key
    return None


def _classic_column_ticket_key(column_name: str) -> Optional[str]:
    """Map a classic-schema column to a ticket field."""
    pairs: List[Tuple[str, str]] = [
        (_notion_prop_column(ENV_NOTION_PROP_NAME, PROP_NAME), "title"),
        (_notion_prop_column(ENV_NOTION_PROP_TYPE, PROP_TYPE), "type"),
        (_notion_prop_column(ENV_NOTION_PROP_PRIORITY, PROP_PRIORITY), "priority"),
        (_notion_prop_column(ENV_NOTION_PROP_STATUS, PROP_STATUS), "status"),
        (_notion_prop_column(ENV_NOTION_PROP_EFFORT, PROP_EFFORT), "estimated_effort"),
        (_notion_prop_column(ENV_NOTION_PROP_TAGS, PROP_TAGS), "tags"),
    ]
    needle = column_name.strip().casefold()
    for col, key in pairs:
        if col.strip().casefold() == needle:
            return key if key != "title" else None
    return None


def _people_option_labels_from_env() -> List[str]:
    """Distinct people labels for the Dev Owner pill, sourced from ``NOTION_ROADMAP_PEOPLE_MAP``.

    Each Notion user id appears once. When multiple keys map to the same id,
    we prefer the longest label (full name) and fall back to the shortest.
    """
    raw = (os.getenv(ENV_ROADMAP_PEOPLE_MAP) or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    def _label_rank(label: str) -> tuple[int, int]:
        clean = label.strip()
        upper_count = sum(1 for c in clean if c.isupper())
        return (len(clean), upper_count)

    by_uid: dict[str, str] = {}
    for label, uid in data.items():
        if not isinstance(label, str) or not isinstance(uid, str):
            continue
        if not label.strip() or not uid.strip():
            continue
        current = by_uid.get(uid)
        if current is None or _label_rank(label) > _label_rank(current):
            by_uid[uid] = label.strip()
    return sorted(by_uid.values(), key=str.casefold)


def _property_options_list(meta: dict[str, Any], notion_type: str) -> List[str]:
    if notion_type == "select":
        return [
            o.get("name")
            for o in (meta.get("select") or {}).get("options", [])
            if o.get("name")
        ]
    if notion_type == "status":
        return [
            o.get("name")
            for o in (meta.get("status") or {}).get("options", [])
            if o.get("name")
        ]
    if notion_type == "multi_select":
        return [
            o.get("name")
            for o in (meta.get("multi_select") or {}).get("options", [])
            if o.get("name")
        ]
    return []


def _notion_rest_get(token: str, path: str) -> dict[str, Any]:
    """Authenticated GET against the Notion REST API (no SDK)."""
    url = "https://api.notion.com" + path
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"Notion GET {path} → {exc.code}: {detail[:240]}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Notion GET {path} failed: {exc.reason}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Notion GET {path} returned non-JSON body.") from exc


def _data_source_id_for_schema(token: str, database_id: str) -> str:
    env_ds = (os.getenv(ENV_NOTION_TICKET_DATA_SOURCE_ID) or "").strip()
    if env_ds:
        return env_ds
    db_obj = _notion_rest_get(token, f"/v1/databases/{database_id}")
    sources: List[dict[str, Any]] = db_obj.get("data_sources") or []
    if len(sources) == 1:
        sid = sources[0].get("id")
        if isinstance(sid, str) and sid:
            return sid
    raise ValueError(
        "Ticket database has no single data source; set NOTION_TICKET_DATA_SOURCE_ID."
    )


def _fetch_ticket_notion_schema_uncached() -> dict[str, Any]:
    """Load property names and select/status options from the ticket Notion data source."""
    ensure_env_loaded()
    fmt = _ticket_format()
    token = (os.getenv(ENV_NOTION_TOKEN) or "").strip()
    db_id = (os.getenv(ENV_NOTION_TICKET_DB_ID) or "").strip()
    out: dict[str, Any] = {
        "format": fmt,
        "database_title": "",
        "properties": [],
        "error": None,
    }
    if not token or not db_id:
        out["error"] = "not_configured"
        return out
    try:
        ds_id = _data_source_id_for_schema(token, db_id)
        ds = _notion_rest_get(token, f"/v1/data_sources/{ds_id}")
    except ValueError as exc:
        out["error"] = str(exc)
        return out
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
        return out
    out["database_title"] = "".join(
        t.get("plain_text", "") for t in (ds.get("title") or [])
    )
    header_order = [
        "type",
        "status",
        "priority",
        "current_sprint",
        "original_sprint",
        "epic",
        "epic_timing",
        "dev_owner",
        "include_in_email",
        "email_sent_in",
        "eng_due_date",
        "cust_release_date",
        "estimated_effort",
        "tags",
    ]

    people_options = _people_option_labels_from_env()
    props_raw = ds.get("properties") or {}
    rows: List[dict[str, Any]] = []
    for pname, pmeta in props_raw.items():
        if not isinstance(pmeta, dict):
            continue
        nt = str(pmeta.get("type") or "")
        opts = _property_options_list(pmeta, nt)
        if nt == "people" and not opts and people_options:
            opts = list(people_options)
        key: Optional[str] = ""
        if nt == "title":
            key = None
        elif fmt == FORMAT_ROADMAP:
            key = _roadmap_column_ticket_key(pname)
        else:
            key = _classic_column_ticket_key(pname)
        show_header = (
            key is not None
            and key != ""
            and nt
            in (
                "select",
                "status",
                "multi_select",
                "number",
                "date",
                "people",
                "rich_text",
            )
        )
        rows.append(
            {
                "name": pname,
                "notion_type": nt,
                "options": opts,
                "key": key if key else "",
                "show_in_header": bool(show_header),
            }
        )

    if fmt == FORMAT_ROADMAP:
        have = {r["key"] for r in rows if r.get("key")}
        if "priority" not in have:
            rows.append(
                {
                    "name": "Priority",
                    "notion_type": "internal",
                    "options": list(TICKET_PRIORITIES),
                    "key": "priority",
                    "show_in_header": True,
                }
            )

    def _sort_key(r: dict[str, Any]) -> Tuple[int, str]:
        k = r.get("key") or ""
        if k in header_order:
            return (header_order.index(k), r.get("name", ""))
        return (len(header_order), r.get("name", ""))

    rows.sort(key=_sort_key)
    out["properties"] = rows
    return out


def get_cached_ticket_notion_schema() -> dict[str, Any]:
    """Return schema dict, cached a few minutes (errors are not cached)."""
    global _SCHEMA_CACHE
    now = time.monotonic()
    if _SCHEMA_CACHE is not None:
        ts, payload = _SCHEMA_CACHE
        if now - ts < SCHEMA_CACHE_TTL_SEC and not payload.get("error"):
            return payload
    payload = _fetch_ticket_notion_schema_uncached()
    if not payload.get("error"):
        _SCHEMA_CACHE = (now, payload)
    else:
        _SCHEMA_CACHE = None
    return payload


def get_ticket_notion_schema() -> dict[str, Any]:
    """Public alias for API handlers."""
    return get_cached_ticket_notion_schema()


def build_ticket_system_prompt(schema: Optional[dict[str, Any]] = None) -> str:
    """System prompt: static classic text or roadmap/options from Notion schema."""
    schema = schema or {}
    props = schema.get("properties") or []
    by_key = {str(p.get("key")): p for p in props if p.get("key")}
    fmt = schema.get("format") or _ticket_format()

    if fmt != FORMAT_ROADMAP or not props or schema.get("error"):
        out = TICKET_SYSTEM_PROMPT
    else:
        def _opts(k: str) -> List[str]:
            o = by_key.get(k, {}).get("options") or []
            return [str(x) for x in o if str(x).strip()]

        cat_opts = _opts("type")
        st_opts = _opts("status")
        if not cat_opts:
            cat_opts = list(TICKET_TYPES)
        if not st_opts:
            st_opts = [TICKET_STATUS_DEFAULT]

        optional_fields: List[str] = []
        opt_keys = [
            ("current_sprint", "current_sprint"),
            ("original_sprint", "original_sprint"),
            ("epic", "epic"),
            ("epic_timing", "epic_timing"),
            ("dev_owner", "dev_owner"),
            ("include_in_email", "include_in_email"),
            ("email_sent_in", "email_sent_in"),
            ("eng_due_date", "eng_due_date"),
            ("cust_release_date", "cust_release_date"),
        ]
        for json_key, field_name in opt_keys:
            if json_key not in by_key:
                continue
            o = _opts(json_key)
            if o:
                optional_fields.append(
                    f'  "{field_name}": string (one of {json.dumps(o)} or "")'
                )
            elif field_name in ("eng_due_date", "cust_release_date"):
                optional_fields.append(
                    f'  "{field_name}": string (YYYY-MM-DD or "" if unknown)'
                )
            else:
                optional_fields.append(
                    f'  "{field_name}": string (use \"\" if unknown)'
                )

        opt_block = ""
        if optional_fields:
            opt_block = ",\n" + ",\n".join(optional_fields)

        tags_line = ""
        has_tags = "tags" in by_key
        if has_tags:
            tags_opts = _opts("tags")
            if tags_opts:
                tags_line = (
                    f',\n  "tags": array of strings, each one of '
                    f"{json.dumps(tags_opts)}"
                )
            else:
                tags_line = ',\n  "tags": array of 1-5 short lowercase strings'

        schema_body = (
            "{\n"
            f'  "title": string (max 80 chars, action-oriented),\n'
            f'  "type": one of {json.dumps(cat_opts)}'
            " — maps to Notion **Category**,\n"
            f'  "priority": one of "urgent" | "high" | "medium" | "low"'
            " — maps to numeric **Prio** in Notion,\n"
            f'  "status": one of {json.dumps(st_opts)}'
            " — maps to Notion **Ovr Status**,\n"
            f'  "description": 2-3 sentences,\n'
            f'  "acceptance_criteria": array of 2-5 short strings,\n'
            f'  "related_context": brief note on which past tickets/docs informed this,\n'
            f'  "estimated_effort": one of "small (< 1 day)" | "medium (1-3 days)" | '
            f'"large (3+ days)"'
            f"{tags_line}{opt_block}\n"
            "}\n"
        )
        tags_rule = ""
        if not has_tags:
            tags_rule = (
                " Do NOT include a 'tags' field — the target Notion database has "
                "no tags column."
            )
        out = (
            "You generate product tickets for the company's Notion **Product Roadmap** "
            "database. "
            "Prior conversation turns may include earlier descriptions and previously "
            "generated ticket JSON. Use them only to interpret follow-ups. Each reply "
            "must still be ONE new JSON object matching the schema below — no prose, no "
            "Markdown fences.\n\n"
            "Output ONLY a single JSON object. The JSON must match this exact schema:\n"
            f"{schema_body}"
            "Use the provided context to inform fields when evidence exists. "
            "For select/status fields, values must match one of the listed options "
            "exactly (spelling and capitalization)."
            f"{tags_rule} "
            "If the context is unrelated, set related_context to "
            '"No directly related context found."'
        )

    suf = ticket_template_prompt_suffix()
    return out + ("\n\n" + suf if suf else "")


def _parse_template_sections_from_model(
    data: dict[str, Any], tmpl: dict[str, Any]
) -> dict[str, str]:
    """Pull ``template_sections`` for fillable indices; default empty strings."""
    raw = data.get("template_sections")
    fillable = {
        str(s["index"])
        for s in (tmpl.get("sections") or [])
        if isinstance(s, dict) and s.get("fillable")
    }
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            sk = str(k).strip()
            if sk in fillable:
                out[sk] = _coerce_str(v)[:12000]
    for fk in fillable:
        out.setdefault(fk, "")
    return out


def parse_ticket_json(
    raw: str, schema: Optional[dict[str, Any]] = None
) -> Ticket:
    """Parse and validate Claude's JSON response into a :class:`Ticket`.

    Args:
        raw: The raw text response from Claude.

    Returns:
        A :class:`Ticket` with normalized enums and clipped lengths.

    Raises:
        ValueError: If ``raw`` cannot be coerced into the expected JSON object.
    """
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Model JSON must be an object at the top level.")

    title = _coerce_str(data.get("title"), default="Untitled ticket")[:80]
    description = _coerce_str(
        data.get("description"), default="(no description provided)"
    )
    related = _coerce_str(
        data.get("related_context"),
        default="No directly related context found.",
    )

    schema = schema or {}
    by_key = {
        str(p.get("key")): p
        for p in (schema.get("properties") or [])
        if p.get("key")
    }

    def _opt_list(k: str) -> List[str]:
        raw_o = list((by_key.get(k) or {}).get("options") or [])
        return [str(x) for x in raw_o if str(x).strip()]

    fmt_effective = str(schema.get("format") or _ticket_format())
    roadmap_schema = (
        fmt_effective == FORMAT_ROADMAP
        and not schema.get("error")
        and bool(by_key)
    )

    if roadmap_schema:
        cat_opts = _opt_list("type")
        st_opts = _opt_list("status")
        sp_opts = _opt_list("current_sprint")
        o_sp_opts = _opt_list("original_sprint")
        epic_opts = _opt_list("epic")
        et_opts = _opt_list("epic_timing")
        inc_opts = _opt_list("include_in_email")
        em_opts = _opt_list("email_sent_in")

        cat_fb = cat_opts[0] if cat_opts else TICKET_TYPES[0]
        type_val = _match_notion_option(
            data.get("type"), cat_opts, fallback=cat_fb
        )

        st_env = (os.getenv(ENV_ROADMAP_OVR_STATUS_OPTION) or "").strip()
        st_fb = st_env or (st_opts[0] if st_opts else TICKET_STATUS_DEFAULT)
        status_val = _match_notion_option(
            data.get("status"), st_opts, fallback=st_fb
        )

        priority_val = _coerce_enum(
            data.get("priority"), TICKET_PRIORITIES, fallback="medium"
        )

        cur_sp = _coerce_str(data.get("current_sprint"))[:120]
        if sp_opts:
            cur_sp = _match_notion_option(cur_sp, sp_opts, fallback="")

        orig_sp = _coerce_str(data.get("original_sprint"))[:120]
        if o_sp_opts:
            orig_sp = _match_notion_option(orig_sp, o_sp_opts, fallback="")

        epic_v = _coerce_str(data.get("epic"))
        if epic_opts:
            epic_v = _match_notion_option(epic_v, epic_opts, fallback="")

        et_v = _coerce_str(data.get("epic_timing"))
        if et_opts:
            et_v = _match_notion_option(et_v, et_opts, fallback="")

        inc_v = _coerce_str(data.get("include_in_email"))
        if inc_opts:
            inc_v = _match_notion_option(inc_v, inc_opts, fallback="")

        em_v = _coerce_str(data.get("email_sent_in"))
        if em_opts:
            em_v = _match_notion_option(em_v, em_opts, fallback="")

        eng_due = _coerce_str(data.get("eng_due_date"))[:32]
        cust_rel = _coerce_str(data.get("cust_release_date"))[:32]
        dev_o = _coerce_str(data.get("dev_owner"))[:200]
        tags_opts = _opt_list("tags")
        if "tags" in by_key:
            tags_raw = _coerce_str_list(data.get("tags"))
            if tags_opts:
                allowed_cf = {t.casefold(): t for t in tags_opts}
                tags_clean = []
                for t in tags_raw:
                    m = allowed_cf.get(t.casefold())
                    if m and m not in tags_clean:
                        tags_clean.append(m)
                tags_value = tags_clean
            else:
                tags_value = tags_raw
        else:
            tags_value = []
    else:
        type_opts = _opt_list("type")
        pri_opts = _opt_list("priority")
        st_opts = _opt_list("status")

        type_allowed: Sequence[str] = type_opts if type_opts else TICKET_TYPES
        type_val = _match_notion_option(
            data.get("type"), list(type_allowed), fallback=TICKET_TYPES[0]
        )

        if pri_opts:
            priority_val = _match_notion_option(
                data.get("priority"), pri_opts, fallback="medium"
            )
        else:
            priority_val = _coerce_enum(
                data.get("priority"), TICKET_PRIORITIES, fallback="medium"
            )

        if st_opts:
            status_val = _match_notion_option(
                data.get("status"), st_opts, fallback=st_opts[0]
            )
        else:
            status_val = (
                _coerce_str(data.get("status"), default=TICKET_STATUS_DEFAULT)
                or TICKET_STATUS_DEFAULT
            )

        cur_sp = _coerce_str(data.get("current_sprint"))[:120]
        orig_sp = _coerce_str(data.get("original_sprint"))[:120]
        epic_v = _coerce_str(data.get("epic"))
        et_v = _coerce_str(data.get("epic_timing"))
        inc_v = _coerce_str(data.get("include_in_email"))
        em_v = _coerce_str(data.get("email_sent_in"))
        eng_due = _coerce_str(data.get("eng_due_date"))[:32]
        cust_rel = _coerce_str(data.get("cust_release_date"))[:32]
        dev_o = _coerce_str(data.get("dev_owner"))[:200]
        tags_value = _coerce_str_list(data.get("tags"))

    tmpl = load_template()
    uses_template = bool(tmpl and tmpl.get("sections"))
    template_sections: dict[str, str] = {}
    if uses_template and tmpl:
        template_sections = _parse_template_sections_from_model(data, tmpl)

    return Ticket(
        title=title or "Untitled ticket",
        type=type_val,
        priority=priority_val,
        status=status_val,
        description=description,
        acceptance_criteria=_coerce_str_list(data.get("acceptance_criteria")),
        related_context=related,
        estimated_effort=_coerce_enum(
            data.get("estimated_effort"),
            EFFORT_OPTIONS,
            fallback="medium (1-3 days)",
        ),
        tags=tags_value,
        current_sprint=cur_sp,
        dev_owner=dev_o,
        epic=epic_v or "",
        epic_timing=et_v or "",
        original_sprint=orig_sp or "",
        include_in_email=inc_v or "",
        email_sent_in=em_v or "",
        eng_due_date=eng_due or "",
        cust_release_date=cust_rel or "",
        uses_template=uses_template,
        template_sections=template_sections,
    )


def _resolve_dev_owner_to_notion_user_ids(label: str) -> List[str]:
    """Map ``dev_owner`` from the model to Notion ``people`` user UUIDs.

    Accepts a raw Notion user id, or a key from ``NOTION_ROADMAP_PEOPLE_MAP``
    JSON (exact or case-insensitive key match).
    """
    label = (label or "").strip()
    if not label:
        return []
    if _UUID_RE.match(label):
        return [label]
    raw = (os.getenv(ENV_ROADMAP_PEOPLE_MAP) or "").strip()
    if not raw:
        return []
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(m, dict):
        return []
    val = m.get(label)
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    lf = label.casefold()
    for key, uid in m.items():
        if isinstance(key, str) and key.casefold() == lf:
            if isinstance(uid, str) and uid.strip():
                return [uid.strip()]
            return []
    return []


def _ticket_generation_constraints_block(
    schema: Optional[dict[str, Any]] = None,
) -> str:
    """Optional CONSTRAINTS block listing allowed sprint names and owner keys."""
    ensure_env_loaded()
    lines: List[str] = []
    schema = schema or {}
    by_key = {
        str(p.get("key")): p
        for p in (schema.get("properties") or [])
        if p.get("key")
    }
    sprint_from_schema = [
        str(x)
        for x in (by_key.get("current_sprint") or {}).get("options") or []
        if str(x).strip()
    ]
    if sprint_from_schema:
        lines.append(
            f"Allowed current_sprint values: {json.dumps(sprint_from_schema)}"
        )
    else:
        sprints_raw = (os.getenv(ENV_ROADMAP_SPRINT_OPTIONS) or "").strip()
        if sprints_raw:
            parts = [p.strip() for p in sprints_raw.split(",") if p.strip()]
            if parts:
                lines.append(f"Allowed current_sprint values: {json.dumps(parts)}")
    people_raw = (os.getenv(ENV_ROADMAP_PEOPLE_MAP) or "").strip()
    if people_raw:
        try:
            m = json.loads(people_raw)
            if isinstance(m, dict) and m:
                keys = sorted(str(k) for k in m.keys() if str(k).strip())
                if keys:
                    lines.append(f"Allowed dev_owner keys: {json.dumps(keys)}")
        except json.JSONDecodeError:
            lines.append(
                "(WARNING: NOTION_ROADMAP_PEOPLE_MAP is invalid JSON — dev_owner "
                "cannot be resolved to Notion users.)"
            )
    if not lines:
        return ""
    return "CONSTRAINTS:\n" + "\n".join(lines) + "\n"


# ----- Retrieval -------------------------------------------------------------


def _node_to_context(node: Any) -> TicketContext:
    """Project a LlamaIndex retrieved node into a :class:`TicketContext`."""
    metadata = getattr(node, "metadata", None) or {}
    text = (node.get_content() or "").strip().replace("\n", " ")
    if len(text) > CONTEXT_PREVIEW_CHARS:
        text = text[: CONTEXT_PREVIEW_CHARS - 1].rstrip() + "…"
    score = getattr(node, "score", None)
    return TicketContext(
        source_type=str(metadata.get(METADATA_SOURCE_TYPE, "unknown")),
        file_name=str(metadata.get(METADATA_FILE_NAME, "unknown")),
        date=str(metadata.get(METADATA_DATE, "")),
        score=float(score) if score is not None else 0.0,
        preview=text,
    )


def _format_context_for_prompt(contexts: List[TicketContext]) -> str:
    """Render retrieved chunks for inclusion in the user prompt."""
    if not contexts:
        return "(no related internal context found)"
    blocks: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        header = (
            f"[{i}] source_type={ctx.source_type}, file_name={ctx.file_name}, "
            f"date={ctx.date or 'unknown'}, score={ctx.score:.3f}"
        )
        blocks.append(f"{header}\n{ctx.preview}")
    return "\n\n".join(blocks)


# ----- Notion write-back -----------------------------------------------------


def _build_paragraph_block(text: str) -> dict[str, Any]:
    """Notion paragraph block (plain rich-text)."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": text[:NOTION_PARAGRAPH_MAX]}}
            ]
        },
    }


def _build_heading_block(text: str) -> dict[str, Any]:
    """Notion heading_2 block."""
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def _build_bullet_block(text: str) -> dict[str, Any]:
    """Notion bulleted list item block."""
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {"type": "text", "text": {"content": text[:NOTION_PARAGRAPH_MAX]}}
            ]
        },
    }


def _notion_prop_column(env_key: str, default: str) -> str:
    """Resolve a Notion column name from env, falling back to ``default``."""
    raw = (os.getenv(env_key) or "").strip()
    return raw if raw else default


def _ticket_format() -> str:
    """``classic`` (default) or ``roadmap`` for Product Roadmap–style schemas."""
    raw = (os.getenv(ENV_NOTION_TICKET_FORMAT) or FORMAT_CLASSIC).strip().lower()
    return raw if raw in (FORMAT_CLASSIC, FORMAT_ROADMAP) else FORMAT_CLASSIC


def _roadmap_priority_number(priority: str) -> int:
    """Coerce ticket priority to a numeric ``Prio`` for the roadmap database."""
    key = (priority or "").strip().lower()
    return ROADMAP_PRIO_BY_LEVEL.get(key, ROADMAP_PRIO_BY_LEVEL["medium"])


def _optional_select(col: str, value: str) -> Optional[dict[str, Any]]:
    """Return a Notion select payload, or ``None`` if ``value`` is empty."""
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    return {col: {"select": {"name": cleaned}}}


def _optional_date(col: str, iso_date: str) -> Optional[dict[str, Any]]:
    """Return a Notion date payload (``YYYY-MM-DD``), or ``None`` if invalid/empty."""
    cleaned = (iso_date or "").strip()
    if not cleaned:
        return None
    return {col: {"date": {"start": cleaned}}}


def _optional_people(col: str, comma_ids: str) -> Optional[dict[str, Any]]:
    """Map comma-separated Notion user IDs to a ``people`` property."""
    ids = [x.strip() for x in (comma_ids or "").split(",") if x.strip()]
    if not ids:
        return None
    return {col: {"people": [{"id": i} for i in ids]}}


def _ticket_to_notion_properties_classic(ticket: Ticket) -> dict[str, Any]:
    """Map a ticket to Notion using the original README schema."""
    name_col = _notion_prop_column(ENV_NOTION_PROP_NAME, PROP_NAME)
    type_col = _notion_prop_column(ENV_NOTION_PROP_TYPE, PROP_TYPE)
    pri_col = _notion_prop_column(ENV_NOTION_PROP_PRIORITY, PROP_PRIORITY)
    st_col = _notion_prop_column(ENV_NOTION_PROP_STATUS, PROP_STATUS)
    eff_col = _notion_prop_column(ENV_NOTION_PROP_EFFORT, PROP_EFFORT)
    tags_col = _notion_prop_column(ENV_NOTION_PROP_TAGS, PROP_TAGS)
    return {
        name_col: {
            "title": [{"type": "text", "text": {"content": ticket.title[:200]}}]
        },
        type_col: {"select": {"name": ticket.type}},
        pri_col: {"select": {"name": ticket.priority}},
        st_col: {"select": {"name": ticket.status}},
        eff_col: {
            "rich_text": [
                {"type": "text", "text": {"content": ticket.estimated_effort}}
            ]
        },
        tags_col: {
            "multi_select": [{"name": t} for t in ticket.tags if t][:25]
        },
    }


def _roadmap_db_property_names() -> Optional[set[str]]:
    """Property names on the ticket Notion data source, for push validation.

    When the cache resolves, optional roadmap fields are skipped unless the
    column exists — preventing ``… is not a property that exists`` when env
    defaults or the model mention columns your DB does not have.

    Returns:
        Non-empty set of names, or ``None`` if schema is unavailable (no filtering).
    """
    schema = get_cached_ticket_notion_schema()
    if schema.get("error"):
        return None
    rows = schema.get("properties") or []
    names = {str(r.get("name")) for r in rows if r.get("name")}
    return names if names else None


def _ticket_to_notion_properties_roadmap(ticket: Ticket) -> dict[str, Any]:
    """Map a ticket to a roadmap DB: title, category, status, number prio, optional defaults.

    Expects a **title** property for the ticket name, **select** for Category,
    **status** for ``Ovr Status``, and **number** for ``Prio``. Other columns
    are sent only when non-empty (ticket or ``NOTION_ROADMAP_DEFAULT_*``) **and**
    that column exists on the data source (when schema cache is available).
    """
    allowed = _roadmap_db_property_names()

    def _merge_select(col: str, value: str) -> None:
        if allowed is not None and col not in allowed:
            return
        merged = _optional_select(col, value)
        if merged:
            props.update(merged)

    def _merge_date(col: str, iso_date: str) -> None:
        if allowed is not None and col not in allowed:
            return
        merged = _optional_date(col, iso_date)
        if merged:
            props.update(merged)

    name_col = _notion_prop_column(ENV_ROADMAP_PROP_NAME, ROADMAP_DEFAULT_NAME)
    cat_col = _notion_prop_column(ENV_ROADMAP_PROP_CATEGORY, ROADMAP_DEFAULT_CATEGORY)
    st_col = _notion_prop_column(
        ENV_ROADMAP_PROP_OVR_STATUS, ROADMAP_DEFAULT_OVR_STATUS
    )
    prio_col = _notion_prop_column(ENV_ROADMAP_PROP_PRIO, ROADMAP_DEFAULT_PRIO)

    status_name = (os.getenv(ENV_ROADMAP_OVR_STATUS_OPTION) or "").strip()
    if not status_name:
        status_name = ticket.status

    props: dict[str, Any] = {
        name_col: {
            "title": [{"type": "text", "text": {"content": ticket.title[:200]}}]
        },
        cat_col: {"select": {"name": ticket.type}},
        st_col: {"status": {"name": status_name}},
        prio_col: {"number": _roadmap_priority_number(ticket.priority)},
    }

    spr_cur = _notion_prop_column(
        ENV_ROADMAP_PROP_CURRENT_SPRINT, ROADMAP_DEFAULT_CURRENT_SPRINT
    )
    sprint_val = (ticket.current_sprint or "").strip() or (
        os.getenv(ENV_ROADMAP_DEFAULT_CURRENT_SPRINT, "") or ""
    ).strip()
    _merge_select(spr_cur, sprint_val)

    spr_orig = _notion_prop_column(
        ENV_ROADMAP_PROP_ORIGINAL_SPRINT, ROADMAP_DEFAULT_ORIGINAL_SPRINT
    )
    _merge_select(
        spr_orig,
        (ticket.original_sprint or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_ORIGINAL_SPRINT, "") or "").strip(),
    )

    epic_col = _notion_prop_column(ENV_ROADMAP_PROP_EPIC, ROADMAP_DEFAULT_EPIC)
    _merge_select(
        epic_col,
        (ticket.epic or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_EPIC, "") or "").strip(),
    )

    epic_tim_col = _notion_prop_column(
        ENV_ROADMAP_PROP_EPIC_TIMING, ROADMAP_DEFAULT_EPIC_TIMING
    )
    _merge_select(
        epic_tim_col,
        (ticket.epic_timing or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_EPIC_TIMING, "") or "").strip(),
    )

    incl_col = _notion_prop_column(
        ENV_ROADMAP_PROP_INCLUDE_EMAIL, ROADMAP_DEFAULT_INCLUDE_EMAIL
    )
    _merge_select(
        incl_col,
        (ticket.include_in_email or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_INCLUDE_IN_EMAIL, "") or "").strip(),
    )

    sent_col = _notion_prop_column(
        ENV_ROADMAP_PROP_EMAIL_SENT, ROADMAP_DEFAULT_EMAIL_SENT
    )
    _merge_select(
        sent_col,
        (ticket.email_sent_in or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_EMAIL_SENT, "") or "").strip(),
    )

    due_col = _notion_prop_column(ENV_ROADMAP_PROP_ENG_DUE, ROADMAP_DEFAULT_ENG_DUE)
    _merge_date(
        due_col,
        (ticket.eng_due_date or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_ENG_DUE, "") or "").strip(),
    )

    cust_col = _notion_prop_column(
        ENV_ROADMAP_PROP_CUST_RELEASE, ROADMAP_DEFAULT_CUST_RELEASE
    )
    _merge_date(
        cust_col,
        (ticket.cust_release_date or "").strip()
        or (os.getenv(ENV_ROADMAP_DEFAULT_CUST_RELEASE, "") or "").strip(),
    )

    own_col = _notion_prop_column(
        ENV_ROADMAP_PROP_DEV_OWNER, ROADMAP_DEFAULT_DEV_OWNER
    )
    owner_ids = _resolve_dev_owner_to_notion_user_ids(ticket.dev_owner)
    if not owner_ids:
        owner_ids = [
            x.strip()
            for x in (os.getenv(ENV_ROADMAP_DEV_OWNER_IDS, "") or "").split(",")
            if x.strip()
        ]
    if owner_ids and (allowed is None or own_col in allowed):
        props[own_col] = {"people": [{"id": uid} for uid in owner_ids]}

    return props


def _ticket_to_notion_properties(ticket: Ticket) -> dict[str, Any]:
    """Map a :class:`Ticket` onto Notion property values."""
    if _ticket_format() == FORMAT_ROADMAP:
        return _ticket_to_notion_properties_roadmap(ticket)
    return _ticket_to_notion_properties_classic(ticket)


def _ticket_to_notion_children(ticket: Ticket) -> List[dict[str, Any]]:
    """Page body: template sections, or description + AC + related context."""
    tmpl = load_template()
    if ticket.uses_template and tmpl and tmpl.get("sections"):
        return build_notion_children_from_template(tmpl, ticket.template_sections)
    children: List[dict[str, Any]] = [
        _build_paragraph_block(ticket.description),
        _build_heading_block("Acceptance Criteria"),
    ]
    if ticket.acceptance_criteria:
        children.extend(_build_bullet_block(c) for c in ticket.acceptance_criteria)
    else:
        children.append(_build_bullet_block("(none provided)"))
    children.append(
        _build_paragraph_block(f"Related context: {ticket.related_context}")
    )
    return children


def _resolve_notion_page_parent(
    client: NotionClient, database_id: str
) -> dict[str, str]:
    """Build the ``parent`` object for ``pages.create``.

    Multi-source Notion databases require
    ``{"type": "data_source_id", "data_source_id": ...}`` under API
    ``2025-09-03``. Single-source databases return one entry in
    ``data_sources``; if there are several, set
    ``NOTION_TICKET_DATA_SOURCE_ID`` in the environment.

    Args:
        client: Notion client configured with ``NOTION_API_VERSION``.
        database_id: UUID of the ticket database from ``NOTION_TICKET_DB_ID``.

    Returns:
        A ``parent`` dict suitable for :meth:`notion_client.Client.pages.create`.

    Raises:
        ValueError: If the data source cannot be determined uniquely.
    """
    env_ds = (os.getenv(ENV_NOTION_TICKET_DATA_SOURCE_ID) or "").strip()
    if env_ds:
        return {"type": "data_source_id", "data_source_id": env_ds}

    try:
        db_obj = client.databases.retrieve(database_id=database_id)
    except APIResponseError as exc:
        raise ValueError(
            f"Could not retrieve Notion database {database_id}: {exc}"
        ) from exc

    sources: List[dict[str, Any]] = db_obj.get("data_sources") or []
    if len(sources) == 1:
        sid = sources[0].get("id")
        if isinstance(sid, str) and sid:
            return {"type": "data_source_id", "data_source_id": sid}
    if len(sources) > 1:
        parts = [
            f"{s.get('name', '(unnamed)')} → {s.get('id')}" for s in sources
        ]
        raise ValueError(
            f"This Notion database has multiple data sources. Set "
            f"{ENV_NOTION_TICKET_DATA_SOURCE_ID} in .env to the UUID for the "
            f"source where tickets should be created. Notion: open the database → "
            f"⋯ → Manage data sources → Copy data source ID. "
            f"Choices: {'; '.join(parts)}"
        )

    # Unusual empty response — try legacy database parent (older single-source DBs).
    return {"database_id": database_id}


def push_ticket_to_notion(
    ticket: Ticket,
    *,
    notion_token: Optional[str] = None,
    database_id: Optional[str] = None,
) -> str:
    """Create a new Notion page for ``ticket`` in the configured database.

    Uses Notion API version ``2025-09-03`` so that **multi-source** databases
    work: the parent is a ``data_source_id`` derived from the database or
    from ``NOTION_TICKET_DATA_SOURCE_ID`` when multiple sources exist.

    Args:
        ticket: A validated :class:`Ticket`.
        notion_token: Override for the Notion integration secret.
        database_id: Override for ``NOTION_TICKET_DB_ID``.

    Returns:
        The URL of the newly created Notion page.

    Raises:
        ValueError: If credentials or the database ID are missing.
        RuntimeError: If Notion rejects the create call.

    Side effects:
        Performs an authenticated ``POST /v1/pages`` to Notion.
    """
    ensure_env_loaded()
    token = notion_token or os.getenv(ENV_NOTION_TOKEN)
    if not token:
        raise ValueError(
            "NOTION_TOKEN is not set. Add it to .env or pass notion_token=…."
        )
    db_id = database_id or os.getenv(ENV_NOTION_TICKET_DB_ID)
    if not db_id:
        raise ValueError(
            "NOTION_TICKET_DB_ID is not set. Add it to .env or pass database_id=…."
        )

    client = NotionClient(auth=token, notion_version=NOTION_API_VERSION)
    parent = _resolve_notion_page_parent(client, db_id)
    try:
        response = client.pages.create(
            parent=parent,
            properties=_ticket_to_notion_properties(ticket),
            children=_ticket_to_notion_children(ticket),
        )
    except APIResponseError as exc:
        raise RuntimeError(f"Notion API rejected ticket: {exc}") from exc

    if isinstance(response, dict):
        url = response.get("url") or ""
        if isinstance(url, str) and url:
            return url
    raise RuntimeError("Notion did not return a page URL for the new ticket.")


# ----- Stateful chain --------------------------------------------------------


@dataclass
class TicketGeneration:
    """Composite output of :meth:`TicketChain.generate`."""

    ticket: Ticket
    sources: List[TicketContext]
    notion_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
            "notion_url": self.notion_url,
        }


class TicketChain:
    """Reusable ticket generator. Build once, call :meth:`generate` per request."""

    def __init__(
        self,
        *,
        similarity_top_k: int = TICKET_TOP_K,
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
        ex_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key=METADATA_SOURCE_TYPE,
                    value=SOURCE_TYPE_NOTION_TICKET_EXAMPLE,
                    operator=FilterOperator.EQ,
                )
            ]
        )
        self._example_retriever = build_vector_index_retriever(
            similarity_top_k=TICKET_EXAMPLE_TOP_K,
            embed_model=embed_model,
            vector_store=vector_store,
            filters=ex_filters,
        )
        api_key = anthropic_api_key or os.getenv(ENV_ANTHROPIC_API_KEY)
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or pass api_key=…."
            )
        self._anthropic = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        description: str,
        *,
        push_to_notion: bool = False,
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> TicketGeneration:
        """Build a structured ticket and optionally push it to Notion.

        Args:
            description: Free-form description of the feature, bug, or research item.
            push_to_notion: If True, also create a page in the ticket database.
            chat_history: Optional prior user/assistant turns (content only). Must
                not include the current ``description``. Retrieval still uses only
                ``description`` as the query.

        Returns:
            :class:`TicketGeneration` with the ticket, supporting sources, and
            (when applicable) the new Notion page URL.

        Raises:
            ValueError: If ``description`` is empty or required env config is missing.
            Exception: API errors from OpenAI, Pinecone, Anthropic, or Notion.
        """
        cleaned = (description or "").strip()
        if not cleaned:
            raise ValueError("description must be a non-empty string")

        tmpl_active = load_template()
        max_tokens = (
            max(CLAUDE_MAX_TOKENS, 4096) if tmpl_active else CLAUDE_MAX_TOKENS
        )

        retrieved_nodes = self._retriever.retrieve(cleaned)
        example_nodes = self._example_retriever.retrieve(cleaned)
        contexts = [_node_to_context(n) for n in retrieved_nodes]
        ex_ctx = [_node_to_context(n) for n in example_nodes]
        prompt_context = _format_context_for_prompt(contexts)
        ex_block = _format_context_for_prompt(ex_ctx)
        if ex_block.strip() and ex_block != "(no related internal context found)":
            prompt_context = (
                prompt_context
                + "\n\n---\n\nSimilar **past ticket examples** (match tone/structure):\n"
                + ex_block
            )

        schema = get_cached_ticket_notion_schema()
        constraints = _ticket_generation_constraints_block(schema)
        user_message = (
            f"Relevant internal context (top {len(contexts)} chunks):\n"
            f"{prompt_context}\n\n"
            f"---\n\n"
            f"Latest description from teammate:\n{cleaned}\n\n"
        )
        if constraints:
            user_message += constraints + "\n"
        user_message += TICKET_USER_MESSAGE_SUFFIX + "Return the ticket JSON now."

        history = _trim_chat_history(tuple(chat_history or ()))
        api_messages: List[dict[str, str]] = list(history)
        api_messages.append({"role": "user", "content": user_message})

        system_prompt = build_ticket_system_prompt(schema)
        message = self._anthropic.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=max_tokens,
            temperature=CLAUDE_TEMPERATURE,
            system=system_prompt,
            messages=api_messages,
        )

        raw_text = "".join(
            getattr(block, "text", "") for block in (message.content or [])
        )
        ticket = parse_ticket_json(raw_text, schema)

        notion_url: Optional[str] = None
        if push_to_notion:
            notion_url = push_ticket_to_notion(ticket)

        return TicketGeneration(
            ticket=ticket, sources=contexts, notion_url=notion_url
        )


# ----- Functional convenience -----------------------------------------------


_DEFAULT_CHAIN: Optional[TicketChain] = None


def get_default_chain() -> TicketChain:
    """Lazily build a process-wide :class:`TicketChain`."""
    global _DEFAULT_CHAIN
    if _DEFAULT_CHAIN is None:
        _DEFAULT_CHAIN = TicketChain()
    return _DEFAULT_CHAIN


def generate_ticket(
    description: str,
    *,
    push_to_notion: bool = False,
    chat_history: Optional[Sequence[Tuple[str, str]]] = None,
) -> TicketGeneration:
    """Top-level helper for tests and ad-hoc scripts.

    Args:
        description: Free-form ticket description.
        push_to_notion: If True, create a Notion page after generation.
        chat_history: Optional prior turns (see :meth:`TicketChain.generate`).

    Returns:
        :class:`TicketGeneration` instance.
    """
    return get_default_chain().generate(
        description,
        push_to_notion=push_to_notion,
        chat_history=chat_history,
    )
