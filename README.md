# BrightBox Voice AI Agent

A voice agent that holds a real phone conversation for a fictional subscription-box company
("BrightBox"), answers questions from a local knowledge base, gracefully handles what it
doesn't know, escalates to a human when policy requires it, and ends the call cleanly.

Built for the Good4Scale take-home: **Twilio → speech-to-text → LLM (with RAG over a local
vector DB) → text-to-speech → back to the caller.**

## Demo

📹 **[Demo video — click here]https://drive.google.com/file/d/1UglKjmxfbF4-lp4FYd0NlMpbctR1jrYc/view?usp=drive_link**

<!-- Replace the link above with your uploaded recording (YouTube/Loom/Drive), or drag an
     mp4 into a GitHub issue/release and paste the URL. A 2–3 min real phone call showing:
     the greeting, 3+ KB answers, a compound question, an out-of-KB fallback, an escalation,
     and a clean hangup. -->

## Architecture

```
Caller ⇄ Twilio Voice (Media Streams, 8kHz)
          → FastAPI  (POST /voice → TwiML <Connect><Stream>;  WS /ws → pipeline)
          → Pipecat pipeline:
                STT (Sarvam Saarika / Deepgram)
              → Silero VAD (turn-taking + barge-in)
              → RAG lookup (Chroma + local sentence-transformers, two-tier relevance gate)
              → LLM (OpenAI gpt-4o-mini / Google gemini-2.0-flash) + end_call/transfer_to_human tools
              → TTS (Sarvam Bulbul / Cartesia), numbers spoken as words
          → back to the caller
```

A **cascaded** pipeline (not speech-to-speech), matching the brief's explicit shape. Retrieval
runs on every turn *before* the LLM (local embedding + Chroma query ≈ 20–80 ms, far cheaper
than an agentic tool-call round trip); a two-tier similarity gate decides whether the model
gets real KB context, a "weak match — judge for yourself" note, or a "nothing matched" signal
that drives the fallback deterministically instead of letting it guess.

## Tools chosen, and why

- **Pipecat** — first-class Twilio Media Streams support (`FastAPIWebsocketTransport` +
  `TwilioFrameSerializer`), no SFU/SIP infra to run (unlike LiveKit), and it doesn't hide the
  pipeline behind a hosted platform (unlike Vapi) — which matters for a task evaluating
  pipeline design.
- **LLM: OpenAI `gpt-4o-mini` or Google `gemini-2.0-flash`** — swappable via one env var
  (`LLM_PROVIDER`). Both are low-latency and follow the escalation policy reliably; Gemini's
  AI Studio free tier makes a zero-cost run possible. These are the only two the brief permits
  (a Grok/xAI key was deliberately left unused for that reason).
- **Sarvam AI (Saarika STT / Bulbul TTS)** — low-latency streaming, strong on Indian-accented
  English/Hindi (relevant for an India-based team). Swappable to **Deepgram/Cartesia** via one
  env var. Sarvam's own LLM is *not* used — the brief restricts the LLM to Gemini/OpenAI;
  Sarvam is only in the STT/TTS legs, which the brief leaves open.
- **ChromaDB + local `sentence-transformers` embeddings** — fully self-hosted per the brief's
  "local vector DB" requirement, no external embedding API in the loop.

## What it handles in a call

- **Multi-turn KB Q&A** — context persists across turns.
- **Two questions at once** — compound utterances ("what's the weather *and* the box pricing?")
  are split and retrieved per-part, and the model is instructed to address every part (answer
  what it can, redirect/hand off the rest) in one reply.
- **Graceful fallbacks, differentiated** — off-topic questions get a warm redirect (no pointless
  human offer); in-scope-but-unlookupable questions, policy exceptions, or complaints get an
  empathy line + human handoff per the KB's escalation policy.
- **LLM function-calling for call control** — `end_call` (model authors its own goodbye) and
  `transfer_to_human` (after the caller confirms), not brittle keyword matching. Both drain the
  spoken line, then hang up via `TwilioFrameSerializer`'s `auto_hang_up`.
- **Provider-failure resilience** — an `on_pipeline_error` handler speaks a recovery line and
  continues on a transient blip, and degrades to a graceful spoken apology + clean hangup on
  repeated/fatal errors, so the caller is never left on a dead line.
- **Natural, human feel** — a casual persona (Ishita), numbers spoken as words ("nineteen
  dollars a month", not "$19/month"), a ~10%-slower/clearer voice, barge-in, "didn't catch
  that" recovery, name personalization, and LLM-connection warmup so the *first* turn is snappy.
- **Observability** — per-call config + per-service latency metrics, and a full transcript
  logged at the end of every call.

## Setup & Run

**Prerequisites:** Python 3.11 (not 3.12+ — Pipecat's audio deps need `audioop`),
[`uv`](https://docs.astral.sh/uv/), [`ngrok`](https://ngrok.com/), and accounts for an LLM
(Gemini *or* OpenAI), STT/TTS (Sarvam *or* Deepgram+Cartesia), and Twilio.

```bash
# 1. Environment + dependencies
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt

# 2. Configure keys
cp .env.example .env      # then fill it in (see key sources below)

# 3. Build the local knowledge base (chunk → embed → store in Chroma)
.venv/Scripts/python.exe scripts/ingest_kb.py
.venv/Scripts/python.exe tests/test_rag.py      # optional: sanity-check retrieval

# 4. Run the server + a public tunnel (two terminals)
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
ngrok http 8000
```

**Where to get each key** (all have free tiers): Gemini → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(free, *not* Vertex AI); Sarvam → [dashboard.sarvam.ai](https://dashboard.sarvam.ai);
Deepgram → [console.deepgram.com](https://console.deepgram.com); Twilio →
[twilio.com/try-twilio](https://www.twilio.com/try-twilio) (verify your own number under
Verified Caller IDs). Pick providers with `LLM_PROVIDER` / `STT_PROVIDER` / `TTS_PROVIDER`
in `.env`.

**Place the call — two options:**
- **Inbound:** in the Twilio Console, point your number's voice webhook at
  `https://<ngrok-subdomain>.ngrok-free.dev/voice` (HTTP POST) and call it.
- **Outbound (free from India — avoids ISD on a US trial number):** have the agent call you:
  ```bash
  .venv/Scripts/python.exe scripts/place_call.py --base-url https://<ngrok-subdomain>.ngrok-free.dev
  ```

## What I'd change for production

- A real warm transfer (Twilio `<Dial>`) to a live support queue — `transfer_to_human`
  currently announces the handoff and ends, since there's no real human line in this
  fictional scenario; the hook is already there.
- Per-word ASR confidence thresholds (Deepgram/Sarvam expose these) instead of the
  empty-transcript proxy for "didn't catch that."
- Persisted conversation logging/analytics (turn counts, escalation rate, retrieval misses)
  to monitor KB-coverage gaps over time.
- Load testing for concurrent calls, and a health check that actively pings the providers.
