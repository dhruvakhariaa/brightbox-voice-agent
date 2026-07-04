"""Environment configuration and provider selection.

All tunables live here so nothing else in the codebase reads `os.environ`
directly -- makes it obvious what's configurable and keeps provider
swapping (Sarvam <-> Deepgram/Cartesia) to a single place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from app.models import LLMProvider, STTProvider, TTSProvider

load_dotenv()

# --- LLM (the task brief permits OpenAI or Gemini only -- pick one here) ---
# Gemini via Google AI Studio has a genuinely free tier (Flash models); OpenAI
# is paid but pennies for a test. Swap with one env var, like STT/TTS below.
LLM_PROVIDER = LLMProvider(os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Use a Google AI Studio key (aistudio.google.com/apikey) -- NOT Vertex AI.
# gemini-2.5-flash is the current free-tier Flash model (~1.2s latency, good
# instruction-following). Note: gemini-2.0-flash's free tier was zeroed out for
# newer AI Studio projects (429 "limit: 0"), so 2.5-flash is the safe default.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- STT/TTS provider selection (swap without touching pipeline code) ---
STT_PROVIDER = STTProvider(os.getenv("STT_PROVIDER", STTProvider.SARVAM.value))
TTS_PROVIDER = TTSProvider(os.getenv("TTS_PROVIDER", TTSProvider.SARVAM.value))

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_VOICE = os.getenv("SARVAM_TTS_VOICE", "anushka")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")

# --- Twilio ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SAMPLE_RATE = 8000  # fixed by Twilio Media Streams (8kHz mulaw)

# Used only by scripts/place_call.py to have the agent call you (outbound), so
# you don't pay ISD to dial a US trial number from India. All E.164 (+countrycode).
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")  # your Twilio number
MY_NUMBER = os.getenv("MY_NUMBER", "")                    # your verified phone to receive the call
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")        # e.g. https://abcd.ngrok-free.dev

# --- Server ---
PORT = int(os.getenv("PORT", "8000"))

# --- RAG ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
COLLECTION_NAME = "brightbox_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 3
# Two-tier relevance gate on the top chunk's cosine similarity (0..1, higher =
# more similar), instead of one brittle cutoff. Tuned from measured scores
# (out-of-KB questions top out ~0.22; the weakest valid query is ~0.42 -- see
# tests/test_rag.py / SETUP.md):
#   score >= STRONG        -> confident: answer from the chunks (+ play filler)
#   WEAK <= score < STRONG -> gray zone: hand chunks to the LLM but flagged as
#                             maybe-irrelevant, so it answers only if they fit
#   score <  WEAK          -> no match: redirect (off-topic) or offer a human
SIMILARITY_STRONG = float(os.getenv("SIMILARITY_STRONG", "0.35"))
SIMILARITY_WEAK = float(os.getenv("SIMILARITY_WEAK", "0.28"))

# --- Call behavior ---
MIN_TRANSCRIPT_CHARS = 2  # shorter finalized transcripts are treated as STT noise
IDLE_TIMEOUT_SECS = float(os.getenv("IDLE_TIMEOUT_SECS", "18"))

# --- Audio / turn-taking (tuned for a phone call, not a headset) ---
# Twilio delivers 8 kHz mu-law. Sarvam STT recognizes noticeably better at
# 16 kHz, so we run the STT leg at 16 kHz (Pipecat upsamples for it); the rest
# of the pipeline stays at Twilio's native 8 kHz.
STT_SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "16000"))
# How long the caller may pause before the agent treats their turn as finished.
# Pipecat's default (0.2s) is headset-tuned and chops phone speech into fragments
# -- which feels like "the agent keeps mishearing / cutting me off". 0.8s fits a
# real phone conversation. Raise it if you're still getting cut off mid-sentence.
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))
# TTS speaking speed. 1.0 is Sarvam's default; 0.9 is ~10% slower for crisper,
# clearer, more natural-sounding speech on a phone line (Sarvam bulbul range 0.3-3.0).
TTS_PACE = float(os.getenv("TTS_PACE", "0.9"))
# Real-time enhancement (automatic gain control) of the caller's audio before
# STT. Off by default: it's an audio pre-processor that sits ahead of the VAD,
# so while debugging turn-detection it's best left out as a variable. Enable it
# once the STT/turn-taking is solid. See app/audio_enhance.py.
AUDIO_ENHANCE = os.getenv("AUDIO_ENHANCE", "false").strip().lower() in ("1", "true", "yes", "on")

# --- Resilience ---
# After this many pipeline errors in a single call, stop trying to recover and
# end gracefully with a spoken apology rather than trapping the caller in dead
# air. See the on_pipeline_error handler in app/bot.py.
MAX_ERRORS_BEFORE_GIVEUP = int(os.getenv("MAX_ERRORS_BEFORE_GIVEUP", "3"))
# OpenAI transient-failure retry (network blips, brief rate limits) before the
# failure ever surfaces as a spoken error to the caller.
LLM_RETRY_ON_TIMEOUT = True
LLM_RETRY_TIMEOUT_SECS = 5.0
