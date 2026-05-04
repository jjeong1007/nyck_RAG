"""One-time Pinecone index creation for the company RAG system.

Run this once per environment (local + each deployed environment) before
ingesting any data:

    python scripts/setup_pinecone.py

If the index already exists with the correct dimension, the script is a
no-op and exits 0. If it exists with a *different* dimension, the script
exits non-zero and refuses to silently delete data — drop the index
manually in the Pinecone console first if you need to recreate it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# ----- Constants -------------------------------------------------------------

# `text-embedding-3-small` -> 1536-dim vectors. If you switch embedding model
# you MUST change this AND recreate the index.
EMBEDDING_DIMENSION: int = 1536

# Cosine is the standard distance for OpenAI-style normalized embeddings.
DISTANCE_METRIC: str = "cosine"

# Pinecone free tier serverless region (only us-east-1 on AWS is in the free
# plan as of writing — change here if your account requires a different one).
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"

# How long to wait for the index to become ready after creation.
INDEX_READY_TIMEOUT_S: int = 120
INDEX_READY_POLL_INTERVAL_S: float = 2.0


# ----- Helpers ---------------------------------------------------------------


def _load_env() -> tuple[str, str]:
    """Load env vars and return (api_key, index_name).

    Raises:
        SystemExit: if any required env var is missing.
    """
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "company-rag")

    if not api_key:
        print("❌ PINECONE_API_KEY is not set. Add it to .env and retry.")
        sys.exit(1)

    return api_key, index_name


def _index_is_ready(desc: object) -> bool:
    """Return True if Pinecone reports the index as ready.

    Pinecone v7+ returns an ``IndexModel`` whose ``status`` object exposes
    ``ready: bool`` (not a dict).
    """
    status = getattr(desc, "status", None)
    if status is None:
        return False
    return getattr(status, "ready", False) is True


def _wait_for_ready(pc: Pinecone, index_name: str) -> None:
    """Block until the named Pinecone index reports ready=True.

    Pinecone's serverless indexes are usually ready within a few seconds
    but we poll defensively up to ``INDEX_READY_TIMEOUT_S``.

    Raises:
        TimeoutError: if the index does not become ready in time.
    """
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        desc = pc.describe_index(index_name)
        if _index_is_ready(desc):
            return
        time.sleep(INDEX_READY_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Pinecone index '{index_name}' did not become ready within "
        f"{INDEX_READY_TIMEOUT_S}s."
    )


# ----- Main ------------------------------------------------------------------


def setup_index() -> None:
    """Create the Pinecone index if it doesn't already exist.

    Idempotent: safe to run repeatedly. If an index with the configured
    name already exists with the expected dimension, this is a no-op.

    Side effects:
        Creates a serverless Pinecone index in the configured cloud/region.
        Prints progress to stdout. Exits the process non-zero on fatal
        config / dimension-mismatch errors.
    """
    api_key, index_name = _load_env()

    print(f"→ Connecting to Pinecone…")
    pc = Pinecone(api_key=api_key)

    existing_names = set(pc.list_indexes().names())

    if index_name in existing_names:
        desc = pc.describe_index(index_name)
        existing_dim = getattr(desc, "dimension", None)
        if existing_dim != EMBEDDING_DIMENSION:
            print(
                f"❌ Index '{index_name}' already exists with dimension "
                f"{existing_dim}, but this project expects "
                f"{EMBEDDING_DIMENSION}. Delete the index in the Pinecone "
                f"console and rerun this script."
            )
            sys.exit(2)
        print(
            f"✅ Pinecone index '{index_name}' already exists "
            f"(dim={existing_dim}, metric={DISTANCE_METRIC}). Nothing to do."
        )
        return

    print(
        f"→ Creating serverless index '{index_name}' "
        f"(dim={EMBEDDING_DIMENSION}, metric={DISTANCE_METRIC}, "
        f"{PINECONE_CLOUD}/{PINECONE_REGION})…"
    )
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=DISTANCE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    print("→ Waiting for index to become ready…")
    _wait_for_ready(pc, index_name)

    print(f"✅ Pinecone index '{index_name}' is ready for ingestion.")


if __name__ == "__main__":
    try:
        setup_index()
    except TimeoutError as exc:
        print(f"❌ {exc}")
        sys.exit(3)
    except Exception as exc:  # pragma: no cover -- top-level safety net
        print(f"❌ Unexpected error while setting up Pinecone: {exc}")
        sys.exit(4)
