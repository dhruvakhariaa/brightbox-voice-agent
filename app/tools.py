"""LLM function-calling tools for call control.

The model decides *when* to end the call or hand off to a human and authors
the exact line the caller hears; these handlers make that decision an actual,
deterministic action on the live Twilio call (speak the line, mark session
state, drain, hang up). This replaces the earlier keyword-substring matching,
which couldn't tell "goodbye" the closing from "goodbye" mid-sentence and had
no way to act on "yes, please connect me."

Handlers close over the call's `CallSession` (one `llm` service is built per
call in app/bot.py, so the closure is naturally per-call and thread-safe)
rather than reaching for globals.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndWorkerFrame, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

from app.models import CallSession
from app.prompts import GOODBYE_LINE, HANDOFF_LINE

Handler = Callable[[FunctionCallParams], Awaitable[None]]

END_CALL = FunctionSchema(
    name="end_call",
    description=(
        "End the phone call. Use only when the caller has said goodbye, said they have no more "
        "questions, or asked to end the call -- never mid-answer or on a brief pause."
    ),
    properties={
        "farewell": {
            "type": "string",
            "description": (
                "A short, warm goodbye to speak before hanging up, using the caller's name if "
                "you know it. This is the last thing they hear, so it must stand alone as a "
                "complete farewell."
            ),
        }
    },
    required=["farewell"],
)

TRANSFER_TO_HUMAN = FunctionSchema(
    name="transfer_to_human",
    description=(
        "Hand the call off to a human support agent. Use only after the caller has explicitly "
        "confirmed they want to be connected -- for order/account-specific issues, policy "
        "exceptions, or complaints that published policy can't resolve."
    ),
    properties={
        "handoff_message": {
            "type": "string",
            "description": "A short line to say as you connect them, e.g. 'Okay, connecting you now -- one moment.'",
        },
        "reason": {
            "type": "string",
            "description": "A brief internal note on why the handoff is needed (for logging; not spoken).",
        },
    },
    required=["handoff_message", "reason"],
)

TOOLS = ToolsSchema(standard_tools=[END_CALL, TRANSFER_TO_HUMAN])


async def _speak_then_end(params: FunctionCallParams, line: str) -> None:
    """Complete the tool call without re-invoking the LLM, then speak a final
    line and gracefully end the call.

    `run_llm=False` stops the model from generating another turn after the
    tool result. `EndWorkerFrame` (pushed downstream, after the TTS frame)
    drains the goodbye audio before the pipeline closes; the closing EndFrame
    then reaches TwilioFrameSerializer, whose auto_hang_up tears down the PSTN
    leg via Twilio's REST API. See SETUP.md for the drain-ordering caveat.
    """
    await params.result_callback(
        {"status": "ok"}, properties=FunctionCallResultProperties(run_llm=False)
    )
    await params.llm.push_frame(TTSSpeakFrame(line))
    await params.llm.push_frame(EndWorkerFrame())


def build_end_call_handler(session: CallSession) -> Handler:
    async def handler(params: FunctionCallParams) -> None:
        session.ended = True
        farewell = (params.arguments.get("farewell") or GOODBYE_LINE).strip()
        logger.info(f"Call {session.call_sid}: end_call tool invoked after {session.turn_count} turn(s)")
        await _speak_then_end(params, farewell)

    return handler


def build_transfer_handler(session: CallSession) -> Handler:
    async def handler(params: FunctionCallParams) -> None:
        session.escalated = True
        session.ended = True
        reason = params.arguments.get("reason", "(unspecified)")
        message = (params.arguments.get("handoff_message") or HANDOFF_LINE).strip()
        logger.info(f"Call {session.call_sid}: transfer_to_human invoked -- reason: {reason}")
        # No real support queue exists in this fictional scenario, so we announce
        # the handoff and end gracefully. Production would <Dial> a live agent /
        # SIP queue here instead of ending -- see README.
        await _speak_then_end(params, message)

    return handler


def register_call_control_tools(llm, session: CallSession) -> None:
    """Wire both tool handlers onto the LLM service for this call."""
    llm.register_function(END_CALL.name, build_end_call_handler(session))
    llm.register_function(TRANSFER_TO_HUMAN.name, build_transfer_handler(session))
