"""One-shot ingestion: chunk the BrightBox KB docs, embed them locally, and
store them in a persistent Chroma collection.

Chunking strategy (deliberate, not fixed-token-window):
  - Document 1 (company overview) and Document 3 (escalation policy) are
    chunked per paragraph -- each paragraph is already a self-contained rule.
  - Document 2 (Shipping/Returns/Billing FAQ) is chunked per Q&A pair, since
    that's the natural retrieval unit; a fixed-size window risks splitting a
    question from its answer.

Run whenever kb/*.txt changes:
    .venv/Scripts/python.exe scripts/ingest_kb.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from app import config, rag  # noqa: E402
from app.models import Chunk  # noqa: E402

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _split_paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _paragraph_chunks(path: Path, doc: str) -> list[Chunk]:
    return [
        Chunk(id=f"{doc}-{i}", text=paragraph, doc=doc, metadata={"doc": doc})
        for i, paragraph in enumerate(_split_paragraphs(path))
    ]


def _qa_chunks(path: Path, doc: str) -> list[Chunk]:
    chunks = []
    for i, block in enumerate(_split_paragraphs(path)):
        match = re.match(r"Q:\s*(.+)", block)
        question = match.group(1).strip() if match else ""
        chunks.append(Chunk(id=f"{doc}-{i}", text=block, doc=doc, metadata={"doc": doc, "question": question}))
    return chunks


def build_chunks() -> list[Chunk]:
    return [
        *_paragraph_chunks(KB_DIR / "doc1_company_overview.txt", "company_overview"),
        *_qa_chunks(KB_DIR / "doc2_shipping_returns_billing.txt", "shipping_returns_billing"),
        *_paragraph_chunks(KB_DIR / "doc3_escalation_policy.txt", "escalation_policy"),
    ]


def main() -> None:
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from {KB_DIR}")
    for chunk in chunks:
        print(f"  [{chunk.id}] {chunk.text[:70]!r}...")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- fine on first run
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = rag.embed([chunk.text for chunk in chunks])
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"\nStored {collection.count()} chunks in '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
