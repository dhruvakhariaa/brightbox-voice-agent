"""Sanity check for the RAG relevance gate -- run after scripts/ingest_kb.py.

Not a pytest suite (nothing to mock, no CI here); this is a manual
verification script whose whole purpose is to eyeball similarity scores and
confirm the two-tier gate buckets queries correctly before the LLM and Twilio
layers are ever involved. Getting this wrong would silently make the agent
hallucinate from irrelevant chunks, or claim it knows nothing about valid
questions.

The gate has three outcomes (see app/rag.classify):
  STRONG -> answer from the chunks
  WEAK   -> hand chunks to the LLM flagged as maybe-irrelevant; it decides
  NONE   -> no usable match; the LLM redirects (off-topic) or offers a human
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, rag  # noqa: E402
from app.models import Relevance  # noqa: E402

# (query, expected relevance) -- documents intent and flags regressions.
KB_QUERIES = [
    ("How long does shipping take?", Relevance.STRONG),
    ("What plans do you offer and how much do they cost?", Relevance.STRONG),
    ("Can I get a refund instead of a replacement?", Relevance.STRONG),
    ("When does my card get charged?", Relevance.STRONG),
    ("Do you ship to Mexico?", Relevance.STRONG),  # answerable: "US and Canada only"
]

# Off-topic / not about BrightBox -> expect NONE -> the agent should REDIRECT,
# not offer a human agent (a teammate can't answer these either).
OFF_TOPIC_QUERIES = [
    "What's your CEO's favorite color?",
    "Can you help me file my taxes?",
    "What's the weather like today?",
]


def _check(query: str, expected: Relevance | None) -> bool:
    results = rag.retrieve(query)
    top = results[0] if results else None
    relevance = rag.classify(top.score if top else None)
    score = f"{top.score:.3f}" if top else "n/a"
    ok = expected is None or relevance is expected
    flag = "OK " if ok else "!! "
    print(f"[{flag}] relevance={relevance.value:<6} score={score}  q: {query}")
    if relevance is Relevance.NONE:
        # Make it unambiguous that this chunk is REJECTED, not used as the answer.
        print(f"        (best chunk rejected as irrelevant: {top.text[:70]!r}...)")
    else:
        print(f"        -> using: {top.text[:70]!r}...")
    return ok


def main() -> int:
    print(f"Bands: STRONG >= {config.SIMILARITY_STRONG}, WEAK >= {config.SIMILARITY_WEAK}, else NONE\n")
    all_ok = True

    print("=== In-KB questions (expect STRONG) ===")
    for query, expected in KB_QUERIES:
        all_ok &= _check(query, expected)

    print("\n=== Off-topic questions (expect NONE -> redirect, not human handoff) ===")
    for query in OFF_TOPIC_QUERIES:
        all_ok &= _check(query, Relevance.NONE)

    print("\nRESULT:", "all buckets as expected" if all_ok else "!! unexpected bucket(s) -- retune bands")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
