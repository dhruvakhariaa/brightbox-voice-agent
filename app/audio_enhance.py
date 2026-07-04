"""Real-time enhancement of the caller's audio before it hits VAD/STT.

Telephony audio is 8 kHz and often quiet or uneven in level, which can make
STT mis-hear soft speech. This filter runs on every incoming chunk (as the
transport's `audio_in_filter`) and applies gentle automatic gain control.

IMPORTANT design note: the gain is a single, slowly-adapting value applied
uniformly to every frame -- NOT a per-frame normalization. Per-frame
normalization would flatten every 20 ms chunk to the same level, erasing the
speech-vs-silence energy contrast that the downstream VAD relies on to find
turn boundaries -- which shows up as erratic start/stop detection and split
turns. A smoothed, held-during-silence gain preserves that contrast.

Dependency-light on purpose (numpy only). RNNoise/Krisp-style denoisers were
rejected: native-lib/PyAV breakage or paid keys, and they're wideband-trained
-- risky for an 8 kHz phone leg. Off by default; enable with AUDIO_ENHANCE.
"""

from __future__ import annotations

import numpy as np
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

_INT16_MAX = 32767


class CallerAudioEnhancer(BaseAudioFilter):
    def __init__(
        self,
        target_rms: float = 2500.0,   # ~ -22 dBFS: a healthy speech level for STT
        max_gain: float = 4.0,        # gentle cap
        noise_floor_rms: float = 200.0,  # below this = silence; hold gain, don't adapt
        adapt: float = 0.08,          # how fast the gain drifts toward target per frame
    ) -> None:
        self._target_rms = target_rms
        self._max_gain = max_gain
        self._noise_floor_rms = noise_floor_rms
        self._adapt = adapt
        self._gain = 1.0  # persistent, slowly-adapting gain (not per-frame)
        self._enabled = True

    async def start(self, sample_rate: int) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or not audio:
            return audio

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return audio

        rms = float(np.sqrt(np.mean(np.square(samples))))
        # Only adapt the gain during clear speech; hold it during silence so the
        # speech/silence contrast the VAD needs is preserved.
        if rms >= self._noise_floor_rms:
            desired = min(max(self._target_rms / rms, 1.0), self._max_gain)
            self._gain += self._adapt * (desired - self._gain)
            self._gain = min(max(self._gain, 1.0), self._max_gain)

        if self._gain <= 1.01:
            return audio  # effectively unity -- skip the multiply/copy

        boosted = np.clip(samples * self._gain, -_INT16_MAX, _INT16_MAX).astype(np.int16)
        return boosted.tobytes()
