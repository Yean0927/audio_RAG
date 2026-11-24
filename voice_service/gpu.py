"""Tensor operations shared by the voice service and the GPU benchmark."""
from __future__ import annotations

import re
from queue import SimpleQueue
import time


def make_gpu_encoder(backend, input_rate):
    import torch
    import torch.nn.functional as F
    from torchaudio.transforms import Resample

    device = backend.speech_device
    model = backend.whisper
    extractor = backend.feature_extractor
    hop = extractor.hop_length
    stride = hop * model.conv1.stride[0] * model.conv2.stride[0] * (model.config.pooling_kernel_size or 1)
    rate = extractor.sampling_rate
    resample = Resample(input_rate, rate).to(device) if input_rate != rate else None
    window = torch.hann_window(extractor.n_fft, dtype=torch.float32).to(device)
    filters = torch.as_tensor(extractor.mel_filters.T, dtype=torch.float32, device=device)

    def encode(pcm):
        pcm = resample(pcm) if resample is not None else pcm
        segments = pcm.split(30 * rate, dim=-1)
        lengths = [part.shape[-1] for part in segments]
        padded = (lengths[0] + stride - 1) // stride * stride
        pcm = torch.cat([F.pad(part, (0, padded - length), value=extractor.padding_value)
                         for part, length in zip(segments, lengths)])
        stft = torch.stft(pcm, extractor.n_fft, hop, window=window, return_complex=True)
        log_mel = (filters @ stft[..., :-1].abs().square()).clamp_min(1e-10).log10()
        log_mel = (torch.maximum(log_mel, log_mel.amax(dim=(-2, -1), keepdim=True) - 8) + 4) / 4
        frames = torch.arange(padded // hop, device=device) * hop
        mask = torch.stack([frames < length for length in lengths]).long()
        ids = model(input_features=log_mel, attention_mask=mask).quantized_token_ids
        return torch.cat([row[:(length + stride - 1) // stride] for row, length in zip(ids, lengths)]).unsqueeze(0)

    return encode


def audio_token_range(tokenizer, decoder):
    tokens = {int(match[1]): token_id for token, token_id in tokenizer.get_added_vocab().items()
              if (match := re.fullmatch(r"<\|audio_(\d+)\|>", token))}
    offset = tokens.get(0)
    size = len(tokens)
    if (offset is None or size > decoder.flow.input_embedding.num_embeddings
            or any(tokens.get(i) != offset + i for i in range(size))):
        raise ValueError("GLM 语音 token 的 ID 范围与解码器不匹配。")
    return offset, size


def audio_prompt_slice(tokenizer, input_ids, count):
    marker = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
    end_marker = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    ids = input_ids[0].tolist()
    start = len(ids) - ids[::-1].index(marker)
    end = start + count
    if end >= len(ids) or ids[end] != end_marker:
        raise ValueError("GLM 提示词中的语音 token 区间无效。")
    return slice(start, end)


class GpuGeneratedTokens:
    """Share small GPU token batches with the audio worker without a Transformers streamer."""

    def __init__(self, profile):
        self.queue = SimpleQueue()
        self.profile = profile
        self.latest = None
        self.count = 0
        self.timestamps = []

    def observe(self, input_ids):
        self.latest = input_ids
        self.count += 1
        if self.profile:
            self.timestamps.append(time.perf_counter())
        if self.count == 4:
            self.flush()

    def flush(self):
        if self.count:
            import torch

            batch = self.latest[0, -self.count:].clone()
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(batch.device))
            self.queue.put((batch, ready, self.timestamps))
            self.count = 0
            self.timestamps = []
        self.latest = None
