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
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import anthropic
from notion_client import Client as NotionClient
from notion_client.errors import APIResponseError

from core.retriever import (
    METADATA_DATE,
    METADATA_FILE_NAME,
    METADATA_SOURCE_TYPE,
    build_openai_embedding_model,
    build_pinecone_vector_store,
    build_vector_index_retriever,
    ensure_env_loaded,
)

# ----- Constants -------------------------------------------------------------

CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
CLAUDE_TEMPERATURE: float = 0.2
CLAUDE_MAX_TOKENS: int = 1024

TICKET_TOP_K: int = 6
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
        }

    def to_markdown(self) -> str:
        """Render the ticket as Markdown for copy/paste in Linear, Slack, etc."""
        bullets = "\n".join(f"- {c}" for c in self.acceptance_criteria) or "- _none_"
        tags = ", ".join(self.tags) if self.tags else "_none_"
        extra = ""
        if self.current_sprint.strip():
            extra += f"\n**Current sprint:** {self.current_sprint}\n"
        if self.dev_owner.strip():
            extra += f"**Dev owner:** {self.dev_owner}\n"
        return (
            f"# {self.title}\n\n"
            f"**Type:** {self.type}  •  **Priority:** {self.priority}  •  "
            f"**Status:** {self.status}  •  **Effort:** {self.estimated_effort}\n"
            f"{extra}\n"
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


def parse_ticket_json(raw: str) -> Ticket:
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

    return Ticket(
        title=title or "Untitled ticket",
        type=_coerce_enum(
            data.get("type"), TICKET_TYPES, fallback=TICKET_TYPES[0]
        ),
        priority=_coerce_enum(
            data.get("priority"), TICKET_PRIORITIES, fallback="medium"
        ),
        status=_coerce_str(data.get("status"), default=TICKET_STATUS_DEFAULT)
        or TICKET_STATUS_DEFAULT,
        description=description,
        acceptance_criteria=_coerce_str_list(data.get("acceptance_criteria")),
        related_context=related,
        estimated_effort=_coerce_enum(
            data.get("estimated_effort"),
            EFFORT_OPTIONS,
            fallback="medium (1-3 days)",
        ),
        tags=_coerce_str_list(data.get("tags")),
        current_sprint=_coerce_str(data.get("current_sprint"))[:120],
        dev_owner=_coerce_str(data.get("dev_owner"))[:200],
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


def _ticket_generation_constraints_block() -> str:
    """Optional CONSTRAINTS block listing allowed sprint names and owner keys."""
    ensure_env_loaded()
    lines: List[str] = []
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


def _ticket_to_notion_properties_roadmap(ticket: Ticket) -> dict[str, Any]:
    """Map a ticket to a roadmap DB: title, category, status, number prio, optional defaults.

    Expects a **title** property for the ticket name, **select** for Category,
    **status** for ``Ovr Status``, and **number** for ``Prio``. Other columns
    are only sent when the matching ``NOTION_ROADMAP_DEFAULT_*`` env var is set.
    """
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
    merged = _optional_select(spr_cur, sprint_val)
    if merged:
        props.update(merged)

    spr_orig = _notion_prop_column(
        ENV_ROADMAP_PROP_ORIGINAL_SPRINT, ROADMAP_DEFAULT_ORIGINAL_SPRINT
    )
    merged = _optional_select(
        spr_orig, os.getenv(ENV_ROADMAP_DEFAULT_ORIGINAL_SPRINT, "") or ""
    )
    if merged:
        props.update(merged)

    epic_col = _notion_prop_column(ENV_ROADMAP_PROP_EPIC, ROADMAP_DEFAULT_EPIC)
    merged = _optional_select(epic_col, os.getenv(ENV_ROADMAP_DEFAULT_EPIC, "") or "")
    if merged:
        props.update(merged)

    epic_tim_col = _notion_prop_column(
        ENV_ROADMAP_PROP_EPIC_TIMING, ROADMAP_DEFAULT_EPIC_TIMING
    )
    merged = _optional_select(
        epic_tim_col, os.getenv(ENV_ROADMAP_DEFAULT_EPIC_TIMING, "") or ""
    )
    if merged:
        props.update(merged)

    incl_col = _notion_prop_column(
        ENV_ROADMAP_PROP_INCLUDE_EMAIL, ROADMAP_DEFAULT_INCLUDE_EMAIL
    )
    merged = _optional_select(
        incl_col, os.getenv(ENV_ROADMAP_DEFAULT_INCLUDE_IN_EMAIL, "") or ""
    )
    if merged:
        props.update(merged)

    sent_col = _notion_prop_column(
        ENV_ROADMAP_PROP_EMAIL_SENT, ROADMAP_DEFAULT_EMAIL_SENT
    )
    merged = _optional_select(
        sent_col, os.getenv(ENV_ROADMAP_DEFAULT_EMAIL_SENT, "") or ""
    )
    if merged:
        props.update(merged)

    due_col = _notion_prop_column(ENV_ROADMAP_PROP_ENG_DUE, ROADMAP_DEFAULT_ENG_DUE)
    merged = _optional_date(due_col, os.getenv(ENV_ROADMAP_DEFAULT_ENG_DUE, "") or "")
    if merged:
        props.update(merged)

    cust_col = _notion_prop_column(
        ENV_ROADMAP_PROP_CUST_RELEASE, ROADMAP_DEFAULT_CUST_RELEASE
    )
    merged = _optional_date(
        cust_col, os.getenv(ENV_ROADMAP_DEFAULT_CUST_RELEASE, "") or ""
    )
    if merged:
        props.update(merged)

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
    if owner_ids:
        props[own_col] = {"people": [{"id": uid} for uid in owner_ids]}

    return props


def _ticket_to_notion_properties(ticket: Ticket) -> dict[str, Any]:
    """Map a :class:`Ticket` onto Notion property values."""
    if _ticket_format() == FORMAT_ROADMAP:
        return _ticket_to_notion_properties_roadmap(ticket)
    return _ticket_to_notion_properties_classic(ticket)


def _ticket_to_notion_children(ticket: Ticket) -> List[dict[str, Any]]:
    """Page body: description paragraph, AC heading + bullets, related-context note."""
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
    ) -> TicketGeneration:
        """Build a structured ticket and optionally push it to Notion.

        Args:
            description: Free-form description of the feature, bug, or research item.
            push_to_notion: If True, also create a page in the ticket database.

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

        retrieved_nodes = self._retriever.retrieve(cleaned)
        contexts = [_node_to_context(n) for n in retrieved_nodes]
        prompt_context = _format_context_for_prompt(contexts)

        constraints = _ticket_generation_constraints_block()
        user_message = (
            f"Description from teammate:\n{cleaned}\n\n"
            f"Relevant internal context (top {len(contexts)} chunks):\n"
            f"{prompt_context}\n\n"
        )
        if constraints:
            user_message += constraints + "\n"
        user_message += TICKET_USER_MESSAGE_SUFFIX + "Return the ticket JSON now."

        message = self._anthropic.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=CLAUDE_TEMPERATURE,
            system=TICKET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = "".join(
            getattr(block, "text", "") for block in (message.content or [])
        )
        ticket = parse_ticket_json(raw_text)

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
    description: str, *, push_to_notion: bool = False
) -> TicketGeneration:
    """Top-level helper for tests and ad-hoc scripts.

    Args:
        description: Free-form ticket description.
        push_to_notion: If True, create a Notion page after generation.

    Returns:
        :class:`TicketGeneration` instance.
    """
    return get_default_chain().generate(
        description, push_to_notion=push_to_notion
    )
