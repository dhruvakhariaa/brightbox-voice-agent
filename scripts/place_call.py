"""Have the Twilio agent call YOU (outbound), instead of you dialing it.

Why this exists: a Twilio trial usually gives you a US number. Dialing that US
number from an Indian mobile is an expensive ISD call. Flip the direction --
have Twilio place the call *to* your phone. Incoming calls in India are free for
the recipient (Twilio pays the outbound leg from your trial credit), so you test
the exact same agent for zero cost and no ISD top-up. The brief allows "make OR
receive a call", so an agent-initiated call is fully compliant.

The agent runs identically: Twilio fetches the same /voice webhook, gets the
<Connect><Stream> TwiML, and streams audio through the same pipeline.

Prerequisites:
  1. The server is running and reachable at --base-url (your ngrok https URL).
  2. --to is a Twilio-VERIFIED number (trial accounts can only call verified
     numbers -- verify yours in the Console first).
  3. Calls to India are enabled: Console -> Voice -> Settings -> Geo Permissions.

Usage:
  python scripts/place_call.py --base-url https://abcd.ngrok-free.dev
  python scripts/place_call.py --base-url https://abcd.ngrok-free.dev --to +9199XXXXXXXX
  python scripts/place_call.py --base-url https://abcd.ngrok-free.dev --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Have the Twilio agent call your phone.")
    parser.add_argument(
        "--base-url", default=config.PUBLIC_BASE_URL,
        help="Public https base URL of the running server (your ngrok URL). Env: PUBLIC_BASE_URL",
    )
    parser.add_argument(
        "--to", default=config.MY_NUMBER,
        help="Number to call, E.164 e.g. +9199... (must be Twilio-verified). Env: MY_NUMBER",
    )
    parser.add_argument(
        "--from", dest="from_", default=config.TWILIO_FROM_NUMBER,
        help="Your Twilio number, E.164. Env: TWILIO_FROM_NUMBER",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print, don't call.")
    args = parser.parse_args()

    required = {
        "TWILIO_ACCOUNT_SID": config.TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN": config.TWILIO_AUTH_TOKEN,
        "--from / TWILIO_FROM_NUMBER": args.from_,
        "--to / MY_NUMBER": args.to,
        "--base-url / PUBLIC_BASE_URL": args.base_url,
    }
    missing = [name for name, val in required.items() if not val]
    if missing:
        print("Missing required values: " + ", ".join(missing), file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    if not base.startswith("https://"):
        print(f"--base-url must be https (Twilio won't fetch http): {base!r}", file=sys.stderr)
        return 2
    webhook = f"{base}/voice"

    # Catch the most common credential mistake before we even call Twilio: an
    # Account SID must start with "AC". Anything else -> guaranteed 20003 auth error.
    if not config.TWILIO_ACCOUNT_SID.startswith("AC"):
        print(
            f"Warning: TWILIO_ACCOUNT_SID starts with {config.TWILIO_ACCOUNT_SID[:2]!r}, not 'AC'. "
            "Twilio Account SIDs always start with 'AC' -- this will fail authentication.",
            file=sys.stderr,
        )

    print(f"Placing call: {args.from_}  ->  {args.to}")
    print(f"  voice webhook: {webhook}")
    if args.dry_run:
        print("[dry-run] not placing the call.")
        return 0

    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client

    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    try:
        call = client.calls.create(to=args.to, from_=args.from_, url=webhook)
    except TwilioRestException as exc:
        print(f"\nTwilio error {exc.code}: {exc.msg}", file=sys.stderr)
        if exc.code == 20003:
            # Auth failure -- rejected before any trial/number/geo checks run.
            print(
                "\nThis is an AUTHENTICATION failure, so the trial/number/geo settings are\n"
                "irrelevant -- the request was rejected before those were ever checked. Fix\n"
                "your credentials in .env:\n"
                "  - TWILIO_ACCOUNT_SID must start with 'AC' (34 chars). Copy it fresh from the\n"
                "    Twilio Console home page > Account Info.\n"
                "  - TWILIO_AUTH_TOKEN must be that same account's current Auth Token.",
                file=sys.stderr,
            )
        else:
            print(
                "\nCommon causes on a trial account:\n"
                "  - The 'to' number isn't verified -> Console > Phone Numbers > Verified Caller IDs.\n"
                "  - Calls to India are disabled    -> Console > Voice > Settings > Geo Permissions.\n"
                "  - --base-url is stale/unreachable -> confirm the server + ngrok are up.",
                file=sys.stderr,
            )
        return 1

    print(f"\nCall queued (SID {call.sid}). Your phone should ring shortly.")
    print("Note: a trial account plays a short 'trial' preamble before the agent speaks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
