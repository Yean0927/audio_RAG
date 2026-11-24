"""Prepare the 16 kHz mono PCM used by duplex call inference."""

import threading


def load_call_vad():
    import torch

    # Silero sets PyTorch's process-wide CPU thread count at import time.
    threads = torch.get_num_threads()
    try:
        from silero_vad import load_silero_vad
    finally:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
    return load_silero_vad(onnx=True)


class CallSpeechFilter:
    def __init__(self, sample_rate):
        self.model = load_call_vad()
        from silero_vad import get_speech_timestamps

        self.sample_rate = sample_rate
        self.detect = get_speech_timestamps
        self.lock = threading.Lock()

    def __call__(self, pcm):
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        # The ONNX wrapper keeps recurrent state, which must stay within one clip.
        with self.lock:
            speech = self.detect(samples, self.model, sampling_rate=self.sample_rate,
                                 min_speech_duration_ms=96, min_silence_duration_ms=180,
                                 speech_pad_ms=0)
        # Very faint background sounds can receive a high VAD score in isolation.
        speech = [region for region in speech
                  if np.mean(samples[region["start"]:region["end"]] ** 2) >= 0.002 ** 2]
        if not speech:
            return b""

        # Keep short pauses and quiet syllables between detected regions.
        start = max(0, speech[0]["start"] - self.sample_rate * 300 // 1000)
        end = speech[-1]["end"] + self.sample_rate * 120 // 1000
        clip = samples[start:end]
        rms = np.sqrt(np.mean(clip ** 2))
        peak = np.max(np.abs(clip))
        gain = min(4.0, 0.10 / rms, 0.90 / peak)
        return np.rint(clip * (32768 * gain)).astype("<i2").tobytes()
