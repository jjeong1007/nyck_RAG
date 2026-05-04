"""FastAPI app exposing Q&A, ticket generation, and ingestion over HTTP.

Endpoints:
    - ``GET  /health``  → ``{"status": "ok"}``
    - ``POST /qa``      → ``{answer, sources}`` (optional ``messages`` = prior chat)
    - ``POST /ticket``  → ``{ticket, sources, notion_url}``
    - ``POST /ingest/local`` (multipart) → upload + ingest PDF/DOCX/TXT/MD
    - ``POST /ingest/transcripts`` (multipart) → upload + ingest .txt transcripts
    - ``POST /ingest/discord`` (multipart) → upload + ingest DiscordChatExporter JSON
    - ``POST /ingest/notion`` → ingest Notion pages / databases by ID
    - ``GET  /sources``  → Pinecone metadata rollup by ``source_type`` / file
    - ``GET  /sources/export`` → merged chunk text for one source (Markdown download)
    - ``GET  /``         → redirects to ``/ui/``
    - Static frontend mounted at ``/ui`` (vanilla HTML/JS).

Errors are always returned as structured JSON; the API never crashes the
process on a request. The Q&A and ticket chains are built lazily on first use
so a missing API key (e.g. Notion-only access) doesn't prevent startup.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from core.qa_chain import QAChain, QAResult
from core.source_inventory import (
    build_source_export,
    build_source_inventory,
    source_export_attachment_filename,
)
from core.ticket_chain import (
    Ticket,
    TicketChain,
    TicketContext,
    TicketGeneration,
    push_ticket_to_notion,
)
from ingest.ingest_discord import ingest_discord
from ingest.ingest_local import LOCAL_FILE_EXTENSIONS, ingest_local
from ingest.ingest_notion import ingest_notion
from ingest.ingest_transcripts import ingest_transcripts

logger = logging.getLogger("company_rag")
logging.basicConfig(level=logging.INFO)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
UI_DIR: Path = REPO_ROOT / "ui"

CORS_ALLOW_ORIGINS_ENV: str = "CORS_ALLOW_ORIGINS"
CORS_DEFAULT_ALLOW_ORIGINS: list[str] = ["*"]


# ----- Startup / shutdown ----------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ``.env`` once at startup; chains are built lazily on first call."""
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    logger.info("Company RAG API starting up.")
    yield
    logger.info("Company RAG API shutting down.")


# ----- App + middleware ------------------------------------------------------


app = FastAPI(
    title="Company RAG",
    description=(
        "Internal Q&A assistant and Notion ticket generator backed by "
        "LlamaIndex + Pinecone + Claude Haiku."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


def _cors_allow_origins() -> list[str]:
    """Read CORS allow-list from env, falling back to wildcard for dev."""
    raw = os.getenv(CORS_ALLOW_ORIGINS_ENV, "").strip()
    if not raw:
        return CORS_DEFAULT_ALLOW_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Static UI -------------------------------------------------------------


if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Send users to the bundled frontend by default."""
    target = "/ui/" if UI_DIR.is_dir() else "/health"
    return RedirectResponse(url=target)


# ----- Pydantic schemas ------------------------------------------------------


class QAChatMessageModel(BaseModel):
    """One prior turn in the Q&A thread (not including the current ``question``)."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message content must be non-empty")
        return cleaned


class QARequest(BaseModel):
    """Request body for ``POST /qa``."""

    question: str = Field(min_length=1, max_length=4000)
    messages: List[QAChatMessageModel] = Field(
        default_factory=list,
        max_length=40,
        description="Prior user/assistant turns, in order. Omit the current question.",
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-empty")
        return cleaned


class QASourceModel(BaseModel):
    source_type: str
    file_name: str
    date: str
    score: float
    preview: str


class QAResponse(BaseModel):
    answer: str
    sources: List[QASourceModel]


class TicketRequest(BaseModel):
    """Request body for ``POST /ticket``."""

    description: str = Field(min_length=1, max_length=4000)
    push_to_notion: bool = False
    messages: List[QAChatMessageModel] = Field(
        default_factory=list,
        max_length=40,
        description="Prior user/assistant turns, in order. Omit the current description.",
    )

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("description must be non-empty")
        return cleaned


class TicketModel(BaseModel):
    title: str
    type: str
    priority: str
    status: str
    description: str
    acceptance_criteria: List[str]
    related_context: str
    estimated_effort: str
    tags: List[str] = Field(default_factory=list)
    current_sprint: str = ""
    dev_owner: str = ""


class TicketSourceModel(BaseModel):
    source_type: str
    file_name: str
    date: str
    score: float
    preview: str


class TicketResponse(BaseModel):
    ticket: TicketModel
    sources: List[TicketSourceModel]
    notion_url: Optional[str] = None
    markdown: str


class SourceRowModel(BaseModel):
    """One logical source (file / page) within a category."""

    file_name: str
    date: str = ""
    chunk_count: int


class SourceCategoryModel(BaseModel):
    """Sources grouped by ingest category (``source_type`` metadata)."""

    category: str
    label: str
    total_chunks: int
    sources: List[SourceRowModel]


class SourcesInventoryResponse(BaseModel):
    """Response for ``GET /sources`` — inventory derived from Pinecone metadata."""

    categories: List[SourceCategoryModel]
    total_vectors_scanned: int
    truncated: bool = False


# ----- Chain accessors (lazy singletons) ------------------------------------


_qa_chain: Optional[QAChain] = None
_ticket_chain: Optional[TicketChain] = None


def _get_qa_chain() -> QAChain:
    """Build (or return) the process-wide :class:`QAChain`."""
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = QAChain()
    return _qa_chain


def _get_ticket_chain() -> TicketChain:
    """Build (or return) the process-wide :class:`TicketChain`."""
    global _ticket_chain
    if _ticket_chain is None:
        _ticket_chain = TicketChain()
    return _ticket_chain


# ----- Conversion helpers ---------------------------------------------------


def _qa_result_to_response(result: QAResult) -> QAResponse:
    return QAResponse(
        answer=result.answer,
        sources=[QASourceModel(**s.to_dict()) for s in result.sources],
    )


def _ticket_to_model(ticket: Ticket) -> TicketModel:
    return TicketModel(**ticket.to_dict())


def _ticket_sources_to_models(sources: List[TicketContext]) -> List[TicketSourceModel]:
    return [TicketSourceModel(**s.to_dict()) for s in sources]


def _ticket_generation_to_response(gen: TicketGeneration) -> TicketResponse:
    return TicketResponse(
        ticket=_ticket_to_model(gen.ticket),
        sources=_ticket_sources_to_models(gen.sources),
        notion_url=gen.notion_url,
        markdown=gen.ticket.to_markdown(),
    )


# ----- Routes ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check used by Railway and the UI."""
    return {"status": "ok"}


@app.get("/sources", response_model=SourcesInventoryResponse)
async def sources_inventory() -> SourcesInventoryResponse:
    """List ingested sources grouped by category (Pinecone ``source_type``).

    Scans vector metadata in the configured index/namespace; can be slow on
    very large indexes. Optional env ``SOURCE_INVENTORY_MAX_VECTORS`` caps work.
    """
    try:
        payload = build_source_inventory()
        return SourcesInventoryResponse.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Source inventory failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not read Pinecone inventory: {exc}",
        ) from exc


@app.get("/sources/export")
async def export_merged_source(
    source_type: str = Query(..., min_length=1, max_length=128),
    file_name: str = Query(..., min_length=1, max_length=2048),
) -> Response:
    """Download merged chunk text for one ingested source as a Markdown file.

    Scans the **full** Pinecone index (not capped by ``SOURCE_INVENTORY_MAX_VECTORS``)
    to assemble every chunk whose metadata matches the given ``source_type`` and
    exact ``file_name``.
    """
    fn = file_name.strip()
    if ".." in fn or "\x00" in fn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file_name.",
        )
    try:
        body = build_source_export(source_type.strip(), fn)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Source export failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not export source: {exc}",
        ) from exc

    if not body.strip():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No chunks found for this source_type and file_name.",
        )

    attach_name = source_export_attachment_filename(fn)
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{attach_name}"',
        },
    )


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest) -> QAResponse:
    """Answer a question using the company knowledge base.

    Returns the model's answer plus structured source citations.
    """
    try:
        chain = _get_qa_chain()
        history = [(m.role, m.content) for m in req.messages]
        result = chain.ask(req.question, chat_history=history)
        return _qa_result_to_response(result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Q&A request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Q&A backend error: {exc}",
        ) from exc


@app.post("/ticket", response_model=TicketResponse)
async def ticket(req: TicketRequest) -> TicketResponse:
    """Generate a structured Notion ticket from a free-form description.

    If ``push_to_notion`` is True, also creates the Notion page and returns
    its URL on ``notion_url``.
    """
    try:
        chain = _get_ticket_chain()
        history = [(m.role, m.content) for m in req.messages]
        gen = chain.generate(
            req.description,
            push_to_notion=req.push_to_notion,
            chat_history=history,
        )
        return _ticket_generation_to_response(gen)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Ticket generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ticket generation error: {exc}",
        ) from exc


class PushRequest(BaseModel):
    """Request body for ``POST /ticket/push`` — push an already-generated ticket."""

    ticket: TicketModel


class PushResponse(BaseModel):
    notion_url: str


@app.post("/ticket/push", response_model=PushResponse)
async def push_ticket(req: PushRequest) -> PushResponse:
    """Push an existing ticket payload to Notion.

    Useful when the UI generated the ticket without ``push_to_notion`` and
    the user later clicks "Push to Notion" without regenerating.
    """
    try:
        ticket_obj = Ticket(
            title=req.ticket.title,
            type=req.ticket.type,
            priority=req.ticket.priority,
            status=req.ticket.status,
            description=req.ticket.description,
            acceptance_criteria=list(req.ticket.acceptance_criteria),
            related_context=req.ticket.related_context,
            estimated_effort=req.ticket.estimated_effort,
            tags=list(req.ticket.tags),
            current_sprint=req.ticket.current_sprint or "",
            dev_owner=req.ticket.dev_owner or "",
        )
        url = push_ticket_to_notion(ticket_obj)
        return PushResponse(notion_url=url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Notion push failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Notion error: {exc}",
        ) from exc


# ----- Ingestion -------------------------------------------------------------


MAX_INGEST_FILES: int = 50
MAX_INGEST_BYTES: int = 50 * 1024 * 1024  # 50 MiB per file


class IngestResponse(BaseModel):
    """Result of an ingest run."""

    documents: int
    chunks: int
    accepted_files: List[str] = Field(default_factory=list)
    skipped_files: List[str] = Field(default_factory=list)


class NotionIngestRequest(BaseModel):
    """Request body for ``POST /ingest/notion``."""

    page_ids: List[str] = Field(default_factory=list)
    database_ids: List[str] = Field(default_factory=list)

    @field_validator("page_ids", "database_ids")
    @classmethod
    def _strip_ids(cls, value: List[str]) -> List[str]:
        cleaned = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return cleaned


def _save_uploads_to_dir(
    files: List[UploadFile], target_dir: Path, allowed_suffixes: List[str]
) -> tuple[List[str], List[str]]:
    """Stream uploaded files to ``target_dir``; return ``(accepted, skipped)``.

    Args:
        files: FastAPI ``UploadFile`` instances from the multipart request.
        target_dir: Existing directory to drop files into.
        allowed_suffixes: Lowercase file extensions to accept (e.g. ``[".pdf"]``).

    Returns:
        Two lists of original filenames: those written to disk, and those
        rejected (wrong extension, empty, or > ``MAX_INGEST_BYTES``).

    Raises:
        ValueError: On too many files in one request.
    """
    if len(files) > MAX_INGEST_FILES:
        raise ValueError(
            f"Too many files in one request (got {len(files)}, max "
            f"{MAX_INGEST_FILES})."
        )

    accepted: List[str] = []
    skipped: List[str] = []
    suffixes = {s.lower() for s in allowed_suffixes}

    for upload in files:
        original = upload.filename or "upload"
        safe = Path(original).name
        if not safe or safe in {".", ".."}:
            skipped.append(original)
            continue
        if Path(safe).suffix.lower() not in suffixes:
            skipped.append(original)
            continue

        dest = target_dir / safe
        size = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = upload.file.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_INGEST_BYTES:
                        out.close()
                        dest.unlink(missing_ok=True)
                        skipped.append(original)
                        size = -1
                        break
                    out.write(chunk)
        finally:
            upload.file.close()

        if size > 0:
            accepted.append(safe)

    return accepted, skipped


@app.post("/ingest/local", response_model=IngestResponse)
async def ingest_local_endpoint(
    files: List[UploadFile] = File(...),
    recursive: bool = Form(True),
) -> IngestResponse:
    """Upload and ingest PDF / DOCX / TXT / Markdown files."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )
    with tempfile.TemporaryDirectory(prefix="ingest-local-") as tmp:
        target = Path(tmp)
        try:
            accepted, skipped = _save_uploads_to_dir(
                files, target, LOCAL_FILE_EXTENSIONS
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        if not accepted:
            return IngestResponse(
                documents=0, chunks=0, accepted_files=[], skipped_files=skipped
            )
        try:
            n_docs, n_chunks = ingest_local(target, recursive=recursive)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Local ingestion failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Local ingestion error: {exc}",
            ) from exc
        return IngestResponse(
            documents=n_docs,
            chunks=n_chunks,
            accepted_files=accepted,
            skipped_files=skipped,
        )


@app.post("/ingest/transcripts", response_model=IngestResponse)
async def ingest_transcripts_endpoint(
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """Upload and ingest sales call transcript ``.txt`` files."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )
    with tempfile.TemporaryDirectory(prefix="ingest-transcripts-") as tmp:
        target = Path(tmp)
        try:
            accepted, skipped = _save_uploads_to_dir(files, target, [".txt"])
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        if not accepted:
            return IngestResponse(
                documents=0, chunks=0, accepted_files=[], skipped_files=skipped
            )
        try:
            n_files, n_chunks = ingest_transcripts(target, recursive=False)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Transcript ingestion failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Transcript ingestion error: {exc}",
            ) from exc
        return IngestResponse(
            documents=n_files,
            chunks=n_chunks,
            accepted_files=accepted,
            skipped_files=skipped,
        )


@app.post("/ingest/discord", response_model=IngestResponse)
async def ingest_discord_endpoint(
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """Upload and ingest DiscordChatExporter ``.json`` exports."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )
    with tempfile.TemporaryDirectory(prefix="ingest-discord-") as tmp:
        target = Path(tmp)
        try:
            accepted, skipped = _save_uploads_to_dir(files, target, [".json"])
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        if not accepted:
            return IngestResponse(
                documents=0, chunks=0, accepted_files=[], skipped_files=skipped
            )
        try:
            n_files, n_chunks = ingest_discord(target, recursive=False)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Discord ingestion failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Discord ingestion error: {exc}",
            ) from exc
        return IngestResponse(
            documents=n_files,
            chunks=n_chunks,
            accepted_files=accepted,
            skipped_files=skipped,
        )


@app.post("/ingest/notion", response_model=IngestResponse)
async def ingest_notion_endpoint(req: NotionIngestRequest) -> IngestResponse:
    """Ingest Notion pages and databases by ID. Each ID must already be shared
    with the integration listed in ``NOTION_TOKEN``."""
    if not req.page_ids and not req.database_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one page_id or database_id.",
        )
    try:
        n_pages, n_chunks = ingest_notion(
            page_ids=req.page_ids or None,
            database_ids=req.database_ids or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Notion ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Notion ingestion error: {exc}",
        ) from exc
    return IngestResponse(
        documents=n_pages,
        chunks=n_chunks,
        accepted_files=[*req.page_ids, *req.database_ids],
        skipped_files=[],
    )


# ----- Global error handler --------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler so the API always returns structured JSON."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {exc}"},
    )
