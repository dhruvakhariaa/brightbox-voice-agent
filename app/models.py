"""Shared data structures for the voice agent.

Each dataclass/enum here backs a real decision point elsewhere in the code
(retrieval results, provider selection, per-call state). See SETUP.md for
why each one exists and what was deliberately left out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrievable unit produced by the KB ingestion script."""

    id: str
    text: str
    doc: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned from a similarity search.

    `score` is a cosine *similarity* (higher = more relevant), not a raw
    distance -- see app/rag.py for why that convention was chosen.
    """

    text: str
    score: float
    doc: str


class Relevance(str, Enum):
    """How well the best-retrieved chunk matches the caller's question.

    A two-tier gate rather than one hard cutoff: a single threshold wrongly
    rejects borderline-but-valid questions (the best real query in this KB
    scores only 0.42). WEAK is a gray zone where the chunks are handed to the
    LLM but flagged as maybe-irrelevant, so the model makes the final call
    instead of the retrieval score alone.
    """

    STRONG = "strong"  # confident match -> answer from the chunks
    WEAK = "weak"      # possible match -> let the LLM judge, then answer or fall back
    NONE = "none"      # no usable match -> redirect (off-topic) or offer a human


class LLMProvider(str, Enum):
    # Only the two providers the task brief permits. Grok/xAI is deliberately
    # NOT here -- see SETUP.md ("Why not Grok").
    OPENAI = "openai"
    GEMINI = "gemini"


class STTProvider(str, Enum):
    SARVAM = "sarvam"
    DEEPGRAM = "deepgram"


class TTSProvider(str, Enum):
    SARVAM = "sarvam"
    CARTESIA = "cartesia"


@dataclass(slots=True)
class CallSession:
    """Mutable per-call state that isn't already implicit in the LLM's own
    conversation history (the caller's name, prior answers, etc. all live in
    context messages instead -- see SETUP.md).

    Each field here is read by something concrete:
      - `ended`      guards the idle-timeout and error handlers so they don't
                     fire a second goodbye after the call is already closing.
      - `escalated`  set by the transfer_to_human tool; surfaced in logs and
                     is where a real handoff/analytics hook would read from.
      - `error_count` drives graceful degradation in the resilience handler
                     (recover vs. give up) -- see app/bot.py.
    """

    call_sid: str
    stream_sid: str
    started_at: float = field(default_factory=time.monotonic)
    turn_count: int = 0
    ended: bool = False
    escalated: bool = False
    error_count: int = 0
