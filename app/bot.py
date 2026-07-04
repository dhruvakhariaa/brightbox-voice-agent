"""Assembles and runs the Pipecat pipeline for a single phone call.

STT/TTS provider construction is factored into two small functions keyed off
`config.STT_PROVIDER`/`TTS_PROVIDER` so swapping Sarvam <-> Deepgram/Cartesia
is a single env var, not an `if` scattered through the pipeline definition.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndWorkerFrame, ErrorFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport

from app import config
from app.models import CallSession, LLMProvider, STTProvider, TTSProvider
from app.prompts import (
    ERROR_RECOVERY_LINE,
    GREETING,
    IDLE_TIMEOUT_LINE,
    SYSTEM_PROMPT,
    TECH_DIFFICULTY_LINE,
)
from app.rag_processor import RAGProcessor
from app.text_filters import SpeechNormalizerFilter
from app.tools import TOOLS, register_call_control_tools


def _build_stt():
    if config.STT_PROVIDER == STTProvider.SARVAM:
        from pipecat.services.sarvam.stt import SarvamSTTService
        from pipecat.transcriptions.language import Language

        return SarvamSTTService(
            api_key=config.SARVAM_API_KEY,
            sample_rate=config.STT_SAMPLE_RATE,  # upsample 8k telephony -> 16k for better recognition
            settings=SarvamSTTService.Settings(model=config.SARVAM_STT_MODEL, language=Language.EN_IN),
        )

    from pipecat.services.deepgram.stt import DeepgramSTTService

    # Deepgram handles 8 kHz mu-law telephony natively -- no upsampling needed.
    return DeepgramSTTService(api_key=config.DEEPGRAM_API_KEY)


def _build_tts():
    if config.TTS_PROVIDER == TTSProvider.SARVAM:
        from pipecat.services.sarvam.tts import SarvamTTSService
        from pipecat.transcriptions.language import Language

        return SarvamTTSService(
            api_key=config.SARVAM_API_KEY,
            text_filters=[SpeechNormalizerFilter()],  # speak numbers as words
            settings=SarvamTTSService.Settings(
                model=config.SARVAM_TTS_MODEL,
                voice=config.SARVAM_TTS_VOICE,
                language=Language.EN_IN,
                pace=config.TTS_PACE,  # ~10% slower for clearer, more natural speech
            ),
        )

    from pipecat.services.cartesia.tts import CartesiaTTSService

    return CartesiaTTSService(
        api_key=config.CARTESIA_API_KEY,
        text_filters=[SpeechNormalizerFilter()],
        settings=CartesiaTTSService.Settings(voice=config.CARTESIA_VOICE_ID),
    )


def _build_llm() -> LLMService:
    # Both providers are LLMService subclasses that speak the universal
    # LLMContext + ToolsSchema, so the pipeline wiring and the call-control
    # tools are identical regardless of which one is selected.
    if config.LLM_PROVIDER == LLMProvider.GEMINI:
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=config.GEMINI_API_KEY,
            settings=GoogleLLMService.Settings(model=config.GEMINI_MODEL),
        )

    # retry_on_timeout absorbs brief network blips / rate limits inside the
    # service before they ever surface to the caller as a spoken error.
    return OpenAILLMService(
        api_key=config.OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(model=config.OPENAI_MODEL),
        retry_on_timeout=config.LLM_RETRY_ON_TIMEOUT,
        retry_timeout_secs=config.LLM_RETRY_TIMEOUT_SECS,
    )


async def _warm_llm_connection(llm: LLMService, label: str) -> None:
    """Fire a throwaway 1-token inference so the LLM's HTTP connection is hot.

    The first real call otherwise pays ~1-3s of TCP/TLS/SDK connection setup
    (measured) that every later turn reuses -- which is exactly why the *first*
    turn feels laggy while the rest feel natural. Running this on the actual
    per-call client while the greeting plays moves that cost off the first turn.
    Best-effort: any failure just leaves today's behavior.
    """
    try:
        client = getattr(llm, "_client", None)
        if client is None:
            return
        if config.LLM_PROVIDER == LLMProvider.GEMINI:
            await client.aio.models.generate_content(model=config.GEMINI_MODEL, contents="hi")
        else:
            await client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
        logger.debug(f"{label}: LLM connection warmed")
    except Exception as exc:  # noqa: BLE001 -- warmup is optional
        logger.debug(f"{label}: LLM warmup skipped ({type(exc).__name__})")


async def warm_llm_startup() -> None:
    """Process-level LLM warmup at server startup (primes SDK/DNS/TLS state so
    the very first call after boot isn't the slow one). Called from the app
    lifespan alongside rag.warmup()."""
    await _warm_llm_connection(_build_llm(), "startup")


async def run_bot(transport: BaseTransport, session: CallSession) -> None:
    """Build the pipeline for one call and run it to completion."""
    stt = _build_stt()
    tts = _build_tts()
    llm = _build_llm()

    logger.info(
        f"Call {session.call_sid} config | LLM={config.LLM_PROVIDER.value} "
        f"STT={config.STT_PROVIDER.value}@{config.STT_SAMPLE_RATE}Hz TTS={config.TTS_PROVIDER.value}"
        f"(pace={config.TTS_PACE}) audio_enhance={config.AUDIO_ENHANCE} vad_stop={config.VAD_STOP_SECS}s"
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}], tools=TOOLS)
    register_call_control_tools(llm, session)
    context_aggregator = LLMContextAggregatorPair(context)
    rag_processor = RAGProcessor(session)
    # stop_secs raised from Pipecat's headset default so the caller can pause
    # mid-sentence on a phone line without the agent grabbing the turn early.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS)))

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            context_aggregator.user(),
            rag_processor,
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=config.TWILIO_SAMPLE_RATE,
            audio_out_sample_rate=config.TWILIO_SAMPLE_RATE,
            # Observability: log per-service TTFB (STT/LLM/TTS latency) and token
            # usage. report_only_initial_ttfb keeps it to one line per response.
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
        idle_timeout_secs=config.IDLE_TIMEOUT_SECS,
        # Graceful goodbye instead of an abrupt cancel -- see the
        # on_idle_timeout handler below.
        cancel_on_idle_timeout=False,
    )

    @worker.event_handler("on_idle_timeout")
    async def _on_idle(_worker: PipelineWorker) -> None:
        if session.ended:
            return
        logger.info(f"Call {session.call_sid}: idle timeout after {config.IDLE_TIMEOUT_SECS}s, ending gracefully")
        session.ended = True
        await _worker.queue_frames(
            [TTSSpeakFrame(IDLE_TIMEOUT_LINE, append_to_context=True), EndWorkerFrame()]
        )

    @worker.event_handler("on_pipeline_error")
    async def _on_error(_worker: PipelineWorker, frame: ErrorFrame) -> None:
        """Graceful degradation instead of dead air.

        A single transient blip (a momentary STT/TTS/LLM hiccup) gets a spoken
        apology and the call continues. A fatal error, or repeated errors in
        one call, means we can't reliably serve the caller -- so we bow out
        with an apology and hang up cleanly rather than leaving them on a dead
        line. `session.ended` guards against stacking multiple goodbyes if
        errors arrive in a burst.
        """
        if session.ended:
            return
        session.error_count += 1
        give_up = frame.fatal or session.error_count >= config.MAX_ERRORS_BEFORE_GIVEUP
        logger.warning(
            f"Call {session.call_sid}: pipeline error #{session.error_count} "
            f"(fatal={frame.fatal}, give_up={give_up}): {frame.error}"
        )
        if give_up:
            session.ended = True
            await _worker.queue_frames(
                [TTSSpeakFrame(TECH_DIFFICULTY_LINE, append_to_context=True), EndWorkerFrame()]
            )
        else:
            await _worker.queue_frames([TTSSpeakFrame(ERROR_RECOVERY_LINE, append_to_context=True)])

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client) -> None:
        logger.info(f"Call {session.call_sid}: connected, greeting caller")
        # Warm the LLM connection concurrently with the greeting so the caller's
        # first turn doesn't pay the cold-connection latency.
        asyncio.create_task(_warm_llm_connection(llm, f"Call {session.call_sid}"))
        await worker.queue_frames([TTSSpeakFrame(GREETING, append_to_context=True)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client) -> None:
        logger.info(
            f"Call {session.call_sid}: disconnected after {session.turn_count} turn(s) "
            f"(escalated={session.escalated}, errors={session.error_count})"
        )

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

    _log_call_summary(session, context)


def _log_call_summary(session: CallSession, context: LLMContext) -> None:
    """One reviewable record per call: duration, outcome, and the transcript.

    Reads the conversation straight out of the LLM context (the source of
    truth), skipping the system prompt and per-turn RAG notes.
    """
    duration = time.monotonic() - session.started_at
    logger.info(
        f"Call {session.call_sid} ended | duration={duration:.1f}s turns={session.turn_count} "
        f"escalated={session.escalated} errors={session.error_count}"
    )
    lines = []
    for message in context.messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user" and content:
            lines.append(f"    caller: {content}")
        elif role == "assistant" and content:  # skip tool-call-only turns (no spoken text)
            lines.append(f"    Ishita: {content}")
    if lines:
        logger.info(f"Call {session.call_sid} transcript:\n" + "\n".join(lines))
