"""Local retrieval over the BrightBox knowledge base.

Embeddings are computed in-process with sentence-transformers (no external
API call) and stored/queried via a persistent local ChromaDB collection --
keeping the whole RAG path self-hosted, per the task's "local vector
database" requirement, and fast enough (~20-80ms) to run unconditionally on
every turn instead of behind an LLM tool-call round trip.
"""

from __future__ import annotations

import re

import chromadb
from sentence_transformers import SentenceTransformer

from app import config
from app.models import Relevance, RetrievedChunk

# Conjunctions that usually join two separate questions in one breath.
_QUESTION_SPLIT = re.compile(r"\band also\b|\band\b|\balso\b|[?;]", re.IGNORECASE)

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts locally. Normalized so cosine similarity == dot product."""
    return _get_model().encode(list(texts), normalize_embeddings=True).tolist()


def retrieve(query: str, k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """Return the top-k most similar chunks, best first.

    Chroma is configured for cosine space, where it returns *distance*
    (0 = identical, 2 = opposite). We convert to similarity (1 - distance,
    higher = better) so callers compare against the SIMILARITY_* bands with
    the intuitive "higher is more relevant" direction.
    """
    collection = _get_collection()
    query_embedding = embed([query])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=k)

    texts = result["documents"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    return [
        RetrievedChunk(text=text, score=1.0 - distance, doc=meta.get("doc", ""))
        for text, distance, meta in zip(texts, distances, metadatas)
    ]


def classify(top_score: float | None) -> Relevance:
    """Bucket the best chunk's similarity into the two-tier relevance gate."""
    if top_score is None or top_score < config.SIMILARITY_WEAK:
        return Relevance.NONE
    if top_score < config.SIMILARITY_STRONG:
        return Relevance.WEAK
    return Relevance.STRONG


def _split_questions(text: str) -> list[str]:
    """Break a compound utterance into candidate sub-questions on conjunctions."""
    parts = [p.strip(" ,.?;") for p in _QUESTION_SPLIT.split(text)]
    return [p for p in parts if len(p.split()) >= 2]


def retrieve_multi(query: str, k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """Retrieve for the whole utterance AND each sub-question, merged by best score.

    A single combined embedding of "what's the weather and the box pricing"
    gets diluted by the off-topic half, which can bury the pricing chunk. Also
    embedding each detected sub-question and taking the union guarantees every
    part's best chunks are on the table, so the LLM can answer both. For a
    normal single question this is just the plain retrieval (no sub-parts).
    """
    subs = _split_questions(query)
    queries = [query, *subs] if len(subs) > 1 else [query]
    best: dict[str, RetrievedChunk] = {}
    for q in queries:
        for chunk in retrieve(q, k=k):
            if chunk.text not in best or chunk.score > best[chunk.text].score:
                best[chunk.text] = chunk
    # One extra chunk vs single-question retrieval, to give the LLM room for two answers.
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[: k + 1]


def warmup() -> None:
    """Load the embedding model + vector store and run one real query now.

    The sentence-transformers model otherwise loads lazily on the first
    retrieval -- i.e. during the caller's first turn -- adding a ~several-second
    cold-start that feels like the agent freezing right after "hello". Calling
    this at server startup pays that cost once, before any call.
    """
    retrieve("warmup")
