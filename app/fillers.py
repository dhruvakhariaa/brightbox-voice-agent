"""Latency-masking + rapport: a short acknowledgment spoken the instant a
user's turn ends, running concurrently with RAG retrieval and the LLM call.

This is spoken through the normal TTS service rather than pre-synthesized
and cached as raw audio. A cached-audio version would shave the filler's own
TTS startup latency (~150-250ms) but requires pushing raw PCM frames past
the BotSpeaking/interruption bracket frames that TTSSpeakFrame handles for
free -- not worth the risk for a phrase that's already shorter than, and
running in parallel with, the RAG+LLM work it's masking. See SETUP.md.
"""

from __future__ import annotations

import random

from app.prompts import ACKNOWLEDGE_FILLERS


def random_filler() -> str:
    return random.choice(ACKNOWLEDGE_FILLERS)
