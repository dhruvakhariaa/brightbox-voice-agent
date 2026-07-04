"""FastAPI app: Twilio Voice webhook + Media Streams WebSocket endpoint.

Hand-rolled instead of using Pipecat's bundled runner/CLI helper so the
whole Twilio integration reads top-to-bottom in one file: return TwiML that
opens a bidirectional stream, read the `start` event for the call/stream
IDs, build the transport, hand off to the pipeline in app/bot.py.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from loguru import logger
from twilio.twiml.voice_response import Connect, VoiceResponse

from app import config, rag
from app.audio_enhance import CallerAudioEnhancer
from app.bot import run_bot, warm_llm_startup
from app.models import CallSession


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the embedding model + vector store AND the LLM connection before any
    # call, so the first caller's first turn doesn't eat the model cold-start or
    # the LLM's cold-connection latency.
    logger.info("Warming up RAG (embedding model + vector store)...")
    rag.warmup()
    await warm_llm_startup()
    logger.info("RAG + LLM ready; accepting calls.")
    yield


app = FastAPI(title="BrightBox Voice Agent", lifespan=lifespan)


@app.post("/voice")
async def voice(request: Request) -> PlainTextResponse:
    """Twilio Voice webhook. Returns TwiML opening a bidirectional Media
    Stream back to this server -- <Connect><Stream>, not <Start><Stream>,
    since the latter is receive-only and the bot would never be heard.

    The host is read from the incoming request rather than hardcoded, so an
    ngrok restart only needs a Twilio Console webhook update, not a code change.
    """
    host = request.headers["host"]
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/ws")
    response.append(connect)
    return PlainTextResponse(content=str(response), media_type="text/xml")


@app.websocket("/ws")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    # Twilio sends "connected" then "start" as the first two text frames.
    # The start event carries the callSid/streamSid the Twilio serializer
    # needs (for audio framing and, via auto_hang_up, REST-API teardown).
    # Consuming them here means the transport's own receive loop picks up
    # cleanly from the "media" events that follow.
    await websocket.receive_text()  # "connected" event, unused
    start_event = json.loads(await websocket.receive_text())
    stream_sid = start_event["start"]["streamSid"]
    call_sid = start_event["start"]["callSid"]
    logger.info(f"Call {call_sid}: media stream started (stream_sid={stream_sid})")

    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=config.TWILIO_ACCOUNT_SID,
        auth_token=config.TWILIO_AUTH_TOKEN,
        # auto_hang_up defaults to True: when the pipeline ends (EndFrame),
        # the serializer calls the Twilio REST API to terminate the call --
        # no manual hangup code needed here.
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            # Real-time gain control on the caller's audio before VAD/STT.
            audio_in_filter=CallerAudioEnhancer() if config.AUDIO_ENHANCE else None,
        ),
    )

    session = CallSession(call_sid=call_sid, stream_sid=stream_sid)
    try:
        await run_bot(transport, session)
    except WebSocketDisconnect:
        logger.info(f"Call {call_sid}: websocket disconnected")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
