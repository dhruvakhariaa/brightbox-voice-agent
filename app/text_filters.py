"""TTS text normalization: turn digits and symbols into spoken words.

TTS engines often read numbers awkwardly on a phone line ("$19" as "dollar
one nine", "3-5" as "three dash five", "25th" as "twenty-five th"). This
filter rewrites the text the model produced into how a person would *say* it
before it reaches the TTS, so prices, dates, ranges, and counts all sound
natural. It runs on sentence-aggregated text inside the TTS service (via
`text_filters=`), and the system prompt also asks the model to spell numbers
out -- belt and suspenders.
"""

from __future__ import annotations

import re

from loguru import logger
from num2words import num2words
from pipecat.utils.text.base_text_filter import BaseTextFilter

# Order matters: patterns that consume digits (currency, ranges, ordinals,
# percent, #-numbers) must run before the bare-integer sweep.
_CURRENCY = re.compile(r"\$\s?(\d+)(?:\.(\d{1,2}))?")
_PER_UNIT = re.compile(r"\s*/\s*(month|mo|year|yr|week|wk|day)\b", re.IGNORECASE)
_HASH_NUM = re.compile(r"#\s?(\d+)")
_PERCENT = re.compile(r"(\d+)\s?%")
_RANGE = re.compile(r"\b(\d+)\s?[-–—]\s?(\d+)\b")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_INTEGER = re.compile(r"\b\d+\b")

_UNIT_WORD = {"mo": "month", "yr": "year", "wk": "week"}


def _cardinal(n: str | int) -> str:
    return num2words(int(n))


def normalize_for_speech(text: str) -> str:
    """Rewrite numbers/symbols in `text` into their spoken-word form."""

    def _currency(m: re.Match) -> str:
        dollars = _cardinal(m.group(1))
        unit = "dollar" if m.group(1) == "1" else "dollars"
        if m.group(2):
            cents = _cardinal(m.group(2).ljust(2, "0"))
            return f"{dollars} {unit} and {cents} cents"
        return f"{dollars} {unit}"

    text = _CURRENCY.sub(_currency, text)
    text = _PER_UNIT.sub(lambda m: f" a {_UNIT_WORD.get(m.group(1).lower(), m.group(1).lower())}", text)
    text = _HASH_NUM.sub(lambda m: f"number {_cardinal(m.group(1))}", text)
    text = _PERCENT.sub(lambda m: f"{_cardinal(m.group(1))} percent", text)
    text = _RANGE.sub(lambda m: f"{_cardinal(m.group(1))} to {_cardinal(m.group(2))}", text)
    text = _ORDINAL.sub(lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = _INTEGER.sub(lambda m: _cardinal(m.group(0)), text)

    text = text.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s{2,}", " ", text).strip()


class SpeechNormalizerFilter(BaseTextFilter):
    """Pipecat text filter that speaks numbers as words. Only `filter` needs
    implementing; the interruption/settings hooks are no-ops (stateless)."""

    async def filter(self, text: str) -> str:
        normalized = normalize_for_speech(text)
        if normalized != text:
            logger.debug(f"SpeechNormalizer: {text!r} -> {normalized!r}")
        return normalized

    async def update_settings(self, settings) -> None:
        pass

    async def handle_interruption(self) -> None:
        pass

    async def reset_interruption(self) -> None:
        pass
