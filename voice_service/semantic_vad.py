"""Streaming Silero VAD and CPU Smart Turn endpointing for 16 kHz mono PCM16."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import math

from .input import load_call_vad

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
START_THRESHOLD = 0.5
STOP_THRESHOLD = 0.35
START_FRAMES = 3
PAUSE_FRAMES = math.ceil(200 / FRAME_MS)
PRE_ROLL_FRAMES = math.ceil(300 / FRAME_MS) + START_FRAMES
TRAILING_BYTES = SAMPLE_RATE * 2 * 120 // 1000
SMART_TURN_SAMPLES = 8 * SAMPLE_RATE


class SmartTurn:
    def __init__(self, model_path):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.extractor = WhisperFeatureExtractor(chunk_length=8)
        self(b"")  # Check the model and preprocessing together during startup.

    def __call__(self, pcm):
        import numpy as np

        audio = np.frombuffer(pcm[-2 * SMART_TURN_SAMPLES:], dtype="<i2").astype(np.float32) / 32768
        audio = np.pad(audio, (SMART_TURN_SAMPLES - audio.size, 0))
        features = self.extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np", do_normalize=True,
            padding="max_length", max_length=SMART_TURN_SAMPLES, truncation=True,
        ).input_features
        probability = float(self.session.run(None, {"input_features": features})[0].item())
        if not 0 <= probability <= 1:
            raise ValueError("Invalid Smart Turn completion probability")
        return probability


@dataclass(frozen=True)
class TurnEvent:
    reason: str
    pcm: bytes = b""
    probability: float | None = None


class SemanticVAD:
    """One call's audio, pause decisions, and Silero recurrent state."""

    def __init__(self, smart_turn, max_seconds, *, silence_ms=1500, silero=None):
        self.smart_turn = smart_turn
        self.silero = silero if silero is not None else load_call_vad()
        self.max_bytes = int(max_seconds * SAMPLE_RATE) * 2
        self.silence_frames = math.ceil(silence_ms / FRAME_MS)
        self.remainder = b""
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        self.onset_frames = 0
        self.audio = bytearray()
        self.quiet_frames = 0
        self.pause_sequence = 0
        self.score_task = None
        self.scored_pause = 0
        self.wait_for_quiet = False

    def _frames(self, pcm):
        import numpy as np
        import torch

        frames = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(-1, FRAME_SAMPLES) / 32768
        return [self.silero(frame, SAMPLE_RATE).item() for frame in torch.from_numpy(frames)]

    def _finish(self, reason, probability=None):
        trailing = self.quiet_frames * FRAME_BYTES
        keep = min(len(self.audio), len(self.audio) - trailing + TRAILING_BYTES, self.max_bytes)
        event = TurnEvent(reason, bytes(self.audio[:keep]), probability)
        self.wait_for_quiet = reason == "recording_limit" and self.quiet_frames < PAUSE_FRAMES
        self.audio.clear()
        self.quiet_frames = 0
        self.onset_frames = 0
        return event

    def _accept_frame(self, frame, speech_probability):
        if self.wait_for_quiet:
            if speech_probability < STOP_THRESHOLD:
                self.wait_for_quiet = False
            return None
        if not self.audio:
            self.pre_roll.append(frame)
            self.onset_frames = self.onset_frames + 1 if speech_probability >= START_THRESHOLD else 0
            if self.onset_frames < START_FRAMES:
                return None
            self.audio.extend(b"".join(self.pre_roll))
            self.pre_roll.clear()
            return TurnEvent("speech_start")

        self.audio.extend(frame)
        if speech_probability >= START_THRESHOLD:
            self.quiet_frames = 0
        elif speech_probability < STOP_THRESHOLD or self.quiet_frames:
            self.quiet_frames += 1
            if self.quiet_frames == PAUSE_FRAMES:
                self.pause_sequence += 1
        if self.quiet_frames >= self.silence_frames:
            return self._finish("silence_timeout")
        if len(self.audio) >= self.max_bytes:
            return self._finish("recording_limit")
        return None

    async def append(self, pcm):
        data = self.remainder + pcm
        end = len(data) // FRAME_BYTES * FRAME_BYTES
        self.remainder = data[end:]
        if not end:
            return []
        events = []
        probabilities = await asyncio.to_thread(self._frames, data[:end])
        for offset, probability in zip(range(0, end, FRAME_BYTES), probabilities):
            event = self._accept_frame(data[offset:offset + FRAME_BYTES], probability)
            if event is not None:
                events.append(event)

        # Apply scores only after consuming new audio. A resumed voice or a newer
        # pause invalidates the earlier score even if the CPU worker just finished.
        paused = self.quiet_frames >= PAUSE_FRAMES
        if self.score_task is not None and self.score_task.done():
            try:
                probability = self.score_task.result()
            except Exception:
                LOG.exception("Smart Turn inference failed; waiting for the silence timeout")
            else:
                if paused and self.scored_pause == self.pause_sequence and probability > 0.5:
                    events.append(self._finish("semantic_complete", probability))
                    paused = False
            self.score_task = None

        if paused and self.score_task is None and self.scored_pause != self.pause_sequence:
            self.scored_pause = self.pause_sequence
            audio = bytes(self.audio[-2 * SMART_TURN_SAMPLES:])
            self.score_task = asyncio.create_task(asyncio.to_thread(self.smart_turn, audio))
        return events

    async def close(self):
        if self.score_task is not None:
            await asyncio.gather(self.score_task, return_exceptions=True)
