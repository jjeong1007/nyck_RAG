# Company RAG

A lightweight Retrieval-Augmented Generation system for an 8-person company.
It does two things:

1. **Q&A Assistant** — ask questions in plain English about the company,
   product, customers, or operations. Answers are grounded in your internal
   knowledge base and come back with source citations.
2. **Notion Ticket Generator** — describe a feature, bug, or piece of
   research in a sentence or two and get back a fully structured product
   ticket. Push it directly to your Notion ticket database, or copy the
   formatted markdown into Linear / GitHub / wherever.

Built on **LlamaIndex + Pinecone + Claude Haiku + OpenAI embeddings**, served
by a small **FastAPI** backend with a single-page **vanilla HTML/JS**
frontend. Designed to deploy to **Railway** in one click.

---

## Prerequisites

- **Python 3.11 or newer** (3.11–3.13 are supported; the locked versions in
  `requirements.txt` are tested on 3.13 as well as 3.11)
- **API keys** for:
  - [Anthropic](https://console.anthropic.com/) — for Claude Haiku generation
  - [OpenAI](https://platform.openai.com/api-keys) — for `text-embedding-3-small`
  - [Pinecone](https://app.pinecone.io/) — free tier serverless is fine
  - [Notion](https://www.notion.so/my-integrations) — internal integration
    token (only needed if you want to ingest Notion pages or push tickets)

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-fork-url> company-rag
cd company-rag

# 2. Create a virtualenv (use python3.11 if you have it; otherwise python3)
python3.11 -m venv .venv   # or: python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# ...then fill in the keys in .env

# 5. Create the Pinecone index (one-time)
python scripts/setup_pinecone.py
```

---

## Ingesting data

All ingestion scripts read from `.env` automatically. Run any combination
of these depending on which sources you have.

### Local files (PDF, DOCX, TXT, MD)

Drop files into `data/` and run:

```bash
python -m ingest.ingest_local
```

You can also point at a different folder:

```bash
python -m ingest.ingest_local --path /absolute/path/to/folder
```

### Notion pages and databases

Set `NOTION_TOKEN` in `.env` (the LlamaIndex reader also accepts the legacy
name `NOTION_INTEGRATION_TOKEN`). Share each page or database with your
Notion integration. Then:

```bash
# Ingest specific pages (each page includes nested blocks in the editor)
python -m ingest.ingest_notion --page-ids <page_id_1> <page_id_2>

# Ingest every row-page from a database
python -m ingest.ingest_notion --database-ids <database_id>
```

### Discord chat exports

Export your Discord channels as JSON using
[DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter), drop
the `.json` files into `data/discord/`, then:

```bash
python -m ingest.ingest_discord
python -m ingest.ingest_discord --path data/discord
```

### Sales call transcripts

Drop `.txt` transcripts into `data/transcripts/`. Filenames in the form
`YYYY-MM-DD_<account>_call.txt` (e.g. `2024-01-15_acme_call.txt`) will have
their date parsed into metadata automatically.

```bash
python -m ingest.ingest_transcripts
python -m ingest.ingest_transcripts --path data/transcripts
```

---

## Running locally

```bash
uvicorn api.main:app --reload
```

Then open <http://localhost:8000/ui/> in your browser. The `/qa` and
`/ticket` endpoints are at the same host.

Health check: <http://localhost:8000/health>

---

## Deploying to Railway

1. Push this repo to GitHub.
2. Create a new Railway project from the repo. Railway auto-detects the
   `Procfile`.
3. In the Railway project **Variables** tab, add every key from
   `.env.example`.
4. Railway will build, install `requirements.txt`, and run the `Procfile`'s
   `web` process. Done.
5. Re-run ingestion any time your knowledge base changes — either locally
   (writes go to the same Pinecone index) or via a Railway one-off command.

---

## Notion database setup

Before you can push tickets via `/ticket`, pick **one** layout:

### A. Classic schema (default — `NOTION_TICKET_FORMAT` unset or `classic`)

Your Notion ticket database **must** have these properties (case-sensitive):

| Property name        | Type           | Notes                                                   |
| -------------------- | -------------- | ------------------------------------------------------- |
| `Name`               | Title          | The default title column. Maps to `title`.             |
| `Type`               | Select         | Options: `feature`, `bug`, `improvement`, `research`.   |
| `Priority`           | Select         | Options: `urgent`, `high`, `medium`, `low`.             |
| `Status`             | Select         | Must include `Not started` as an option.                |
| `Estimated Effort`   | Rich text      | Free-form (`small (< 1 day)`, `medium (1-3 days)`, …). |
| `Tags`               | Multi-select   | Auto-populated from the model's tag list.               |

Rename-only tweaks without changing types: set `NOTION_PROP_NAME`, `NOTION_PROP_TYPE`, etc. in `.env` (see `.env.example`).

### B. Product Roadmap schema (`NOTION_TICKET_FORMAT=roadmap`)

For databases like **Name** (title), **Category** (select), **Ovr Status** (Notion **status**), **Prio** (number), and optional sprint/email columns:

| Property (default name) | Notion type    | Source from the model |
| ------------------------| -------------- | ---------------------- |
| `Name`                  | Title          | `title` |
| `Category`              | Select         | `type` — add options `feature`, `bug`, `improvement`, `research` in Notion. |
| `Ovr Status`            | Status         | Uses model `status` unless you set `NOTION_ROADMAP_OVR_STATUS_OPTION` to an **exact** existing option (e.g. `To be described`). |
| `Prio`                  | Number         | `priority`: `urgent→1`, `high→2`, `medium→3`, `low→4`. |

**Current Sprint** — the model outputs JSON `current_sprint` when the description or RAG context implies it. For **exact** Notion select matching, set `NOTION_ROADMAP_SPRINT_OPTIONS` to comma-separated sprint names (e.g. `Sprint 12,Sprint 13`). If the model leaves it blank, `NOTION_ROADMAP_DEFAULT_CURRENT_SPRINT` is still used as a fallback.

**Dev Owner** — Notion’s **people** field needs user UUIDs. Set `NOTION_ROADMAP_PEOPLE_MAP` to JSON like `{"Jane Doe":"<notion-user-uuid>"}`; the model fills `dev_owner` with one of those keys (or you can output a raw UUID). If still empty, `NOTION_ROADMAP_DEV_OWNER_IDS` (comma-separated UUIDs) is used as a fallback.

Other optional columns are **only sent** when you set the matching `NOTION_ROADMAP_DEFAULT_*` value in `.env` (e.g. Epic). Date envs use `YYYY-MM-DD` for **Eng. Due Date** / **Cust. Release Date**.

Column renames: override with `NOTION_ROADMAP_PROP_NAME`, `NOTION_ROADMAP_PROP_CATEGORY`, etc. See `.env.example`.

The ticket page body will contain:

- A paragraph with the description.
- A heading **Acceptance Criteria** followed by a bulleted list.
- A short paragraph noting which past tickets/docs informed the ticket.

Get the database ID by opening the database as a full page in Notion and
copying the 32-character string from the URL. Add it as
`NOTION_TICKET_DB_ID` in `.env`. Don't forget to **Share** the database
with your Notion integration, otherwise writes will 404.

**Multi-source roadmap databases** (Notion’s newer model that combines several
sources in one view) need the Public API `2025-09-03` flow: the app resolves
a *data source* ID from your database. If your database has **more than one**
data source, set `NOTION_TICKET_DATA_SOURCE_ID` in `.env` to the UUID from
Notion → open the database → **⋯** → **Manage data sources** → **Copy data
source ID** for the source where new tickets should appear. See
[Notion’s upgrade guide](https://developers.notion.com/guides/get-started/upgrade-guide-2025-09-03).

---

## Cost estimate (8-person team)

Rough monthly costs assuming moderate usage (~500 Q&A queries +
~50 ticket generations + a one-time ingestion of ~5k pages):

| Item                                              | Estimated cost      |
| ------------------------------------------------- | ------------------- |
| OpenAI `text-embedding-3-small` (one-time + delta) | $1 – $3             |
| Anthropic Claude Haiku (Q&A + tickets)            | $10 – $25           |
| Pinecone serverless (free tier covers <1M vectors) | $0                  |
| Railway hobby plan                                | $5                  |
| **Total**                                         | **~$25 – $50/mo**   |

Costs scale roughly linearly with query volume; embedding costs are
dominated by the initial ingest and stay near-zero afterwards.

---

## Project layout

```
company-rag/
├── api/main.py             # FastAPI app (/qa, /ticket, /health, /ui)
├── core/
│   ├── retriever.py        # shared Pinecone + LlamaIndex setup
│   ├── qa_chain.py         # Q&A with source citations
│   └── ticket_chain.py     # ticket generation + Notion write-back
├── ingest/
│   ├── ingest_local.py
│   ├── ingest_notion.py
│   ├── ingest_discord.py
│   └── ingest_transcripts.py
├── scripts/setup_pinecone.py
├── ui/index.html           # vanilla HTML/CSS/JS frontend
├── data/                   # local files for ingestion (gitignored)
├── requirements.txt
├── Procfile
└── .env.example
```

---

## What this intentionally doesn't do (v1)

- No authentication — assume private Railway URL.
- No chat history / threading — each request is stateless.
- No streaming responses — basic request/response only.
- No tests — coming in v2.
- No LangChain — standardized on LlamaIndex.
- Always uses Claude Haiku and `text-embedding-3-small` to keep costs predictable.
