"""The RAG + rapport heart of the pipeline.

Sits between the user-side LLM context aggregator and the LLM service.
On every finalized user turn (an `LLMContextFrame` flowing downstream) it:

  1. Short-circuits near-empty transcripts (STT noise/silence misfires)
     straight to a "didn't catch that" reply -- skipping the LLM entirely.
  2. Otherwise, runs a fast local KB lookup and appends the result to the
     LLM context as a system note (or an explicit "no match" note, so the
     LLM knows to use the fallback/escalation line instead of guessing).
  3. On a *confident* match only, speaks a short neutral filler before
     forwarding the context frame, so it reaches the TTS service ahead of
     the (slower) LLM call and masks the lookup latency. On a no-match turn
     the reply is short and fast (a fallback/handoff), so no filler is
     needed and one would sound odd. See app/fillers.py.

Call-ending and human handoff are handled by the LLM's `end_call` /
`transfer_to_human` tools (app/tools.py), not here -- this processor only
shapes what the LLM sees on a normal answering turn.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app import config, rag
from app.fillers import random_filler
from app.models import CallSession, Relevance
from app.prompts import (
    NO_MATCH_CONTEXT,
    REPEAT_LINE,
    build_context_block,
    build_weak_context_block,
)


class RAGProcessor(FrameProcessor):
    def __init__(self, session: CallSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame) or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        user_text = _last_user_text(frame.context.messages)
        if user_text is None or len(user_text.strip()) < config.MIN_TRANSCRIPT_CHARS:
            logger.debug("RAGProcessor: near-empty transcript, asking caller to repeat")
            await self.push_frame(TTSSpeakFrame(REPEAT_LINE, append_to_context=True), direction)
            return

        self._session.turn_count += 1
        results = rag.retrieve_multi(user_text)  # handles compound (multi-part) questions
        top = results[0] if results else None
        relevance = rag.classify(top.score if top else None)

        if relevance is Relevance.STRONG:
            note = build_context_block(results)
        elif relevance is Relevance.WEAK:
            note = build_weak_context_block(results)
        else:
            note = NO_MATCH_CONTEXT

        top_score = f"{top.score:.3f}" if top else "n/a"
        logger.debug(
            f"RAGProcessor: turn={self._session.turn_count} top_score={top_score} relevance={relevance.value}"
        )
        frame.context.add_message({"role": "system", "content": note})

        # Filler only on a confident lookup (a "let me check" makes no sense when
        # we're about to redirect or hand off), and pushed *before* the context
        # frame so it reaches TTS ahead of the slower LLM call.
        if relevance is Relevance.STRONG:
            await self.push_frame(TTSSpeakFrame(random_filler()), direction)
        await self.push_frame(frame, direction)


def _last_user_text(messages: list[dict]) -> str | None:
    """Pull the text of the most recent user message out of the context.

    Content is usually a plain string; guard for the list-of-parts shape
    (used for multimodal messages) since LLMContext's type allows it even
    though this text-only pipeline never produces it.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return " ".join(part for part in parts if part)
    return None
