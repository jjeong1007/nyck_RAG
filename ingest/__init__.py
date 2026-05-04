"""Ingestion pipelines for the company RAG system.

Each module exposes a `run()` function that ingests one source type
(local files, Notion, Discord exports, sales transcripts) into the
shared Pinecone index defined in `core.retriever`.
"""
