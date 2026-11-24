"""GLM-4-Voice loading, prompt construction and streaming inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
from queue import SimpleQueue
import threading
import time
import uuid

from . import ROOT
from .backend import (
    LOG, STREAM_BLOCK_SIZES, TURN_CHECK_MAX_TOKENS, TurnDecision, VoiceBackendBase, VoiceReply, answer_instruction, check_cancelled,
    greeting_instruction, history_reference, pcm16_bytes, turn_decision, turn_instruction, waveform_wav,
)
from .gpu import GpuGeneratedTokens, audio_prompt_slice, audio_token_range, make_gpu_encoder

GLM_TOKENIZER_REPO = "zai-org/glm-4-voice-tokenizer"
INTERLEAVED_REPLY = "Respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens. "
SPEECH_INSTRUCTION = (
    "User will provide you with a speech instruction. Do it step by step. "
    "First, think about the instruction and respond in a interleaved manner, "
    "with 13 text token followed by 26 audio tokens. "
)


class GeneratedTokens:
    """Forward token batches from the generation thread to the decoder."""

    def __init__(self, profile=None):
        self.queue = SimpleQueue()
        self.received_prompt = False
        self.profile = profile

    def put(self, value):
        if self.received_prompt:
            self.push(value.flatten().tolist())
        self.received_prompt = True

    def push(self, ids):
        self.queue.put((ids, time.perf_counter() if self.profile else None))

    def end(self):
        pass  # The worker signals completion even if generate() raises.


def make_prompt(history, question, context, language, persona="", *, clarify=False, exact_faq=False, courtesy=False):
    instruction = answer_instruction(context, language, history, clarify=clarify, exact_faq=exact_faq, courtesy=courtesy)
    if courtesy:
        persona = ""
    prompt = f"<|system|>\n{SPEECH_INSTRUCTION}{instruction}{persona}"
    for turn in history:
        prompt += (f"<|user|>\n{history_reference(turn)}{turn['question_tokens']}"
                   f"<|assistant|>streaming_transcription\n{turn['assistant_tokens']}")
    current = question
    if context:
        reference_label, question_label = (
            ("参考资料", "本轮语音提问") if language == "zh" else ("Reference material", "Current spoken question")
        )
        current = f"[{reference_label}]\n{context}\n[/{reference_label}]\n\n{question_label}\n{question}"
    return f"{prompt}<|user|>\n{current}<|assistant|>streaming_transcription\n"


def clean_assistant(raw):
    for marker in ("<|user|>", "<|endoftext|>"):
        raw = raw.split(marker, 1)[0]
    return raw.strip()


def unspoken_text_tail(tokenizer, generated, audio_offset, audio_end):
    last_audio = next((i for i in range(len(generated) - 1, -1, -1)
                       if audio_offset <= generated[i] < audio_end), -1)
    tail = tokenizer.decode(generated[last_audio + 1:], skip_special_tokens=True).strip()
    return tail if any(char.isalnum() for char in tail) else ""


def resolve_glm_tokenizer_path(tokenizer_path, model_path):
    if tokenizer_path != GLM_TOKENIZER_REPO:
        return tokenizer_path

    from huggingface_hub import try_to_load_from_cache

    name = Path(GLM_TOKENIZER_REPO).name
    legacy_repo = f"THUDM/{name}"
    folders = [ROOT / name, ROOT / legacy_repo]
    model_dir = Path(model_path).expanduser()
    if model_dir.is_dir():
        folders.insert(0, model_dir.resolve().parent / name)

    for repo in (GLM_TOKENIZER_REPO, legacy_repo):
        cached = try_to_load_from_cache(repo, "config.json", cache_dir=os.getenv("TRANSFORMERS_CACHE"))
        if isinstance(cached, str):
            folders.append(Path(cached).parent)

    for folder in folders:
        if (folder / "config.json").is_file() and (folder / "preprocessor_config.json").is_file():
            for filename in ("model.safetensors", "pytorch_model.bin"):
                weight = folder / filename
                # Ignore empty downloads and Git LFS pointer files.
                if weight.is_file() and weight.stat().st_size >= 1024:
                    return str(folder)
    return tokenizer_path


class VoiceBackend(VoiceBackendBase):
    """GLM-4-Voice with split-device encoding and streaming audio."""

    supports_concurrent_inference = True
    vllm = None

    def load(self):
        """Caller holds the lock, including during lazy model initialization."""
        if self.ready:
            return
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder
        from spoken_chatbot.model import AudioDecoder

        decoder_dir = Path(self.args.flow_path).resolve()
        missing = [name for name in ("config.yaml", "flow.pt", "hift.pt") if not (decoder_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"语音 Decoder 缺少文件：{', '.join(missing)}。请先下载到 {decoder_dir}，或指定 --flow-path。"
            )
        self.set_devices()

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        vllm = None
        model = None
        if getattr(self.args, "vllm_base_url", None):
            from transformers import AutoConfig, GenerationConfig
            from .vllm import VllmTokenGenerator

            vllm = VllmTokenGenerator(self.args.vllm_base_url, self.args.vllm_model)
            config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)
            max_model_len = vllm.model_context_length()
            self.vllm_context_length = min(getattr(config, "seq_length", 8192), max_model_len or 8192, 8192)
            try:
                generation_config = GenerationConfig.from_pretrained(self.args.model_path)
            except OSError:
                generation_config = GenerationConfig.from_model_config(config)
            eos = generation_config.eos_token_id
            if eos is None:
                eos = config.eos_token_id
            self.vllm_stop_ids = [eos] if isinstance(eos, int) else list(eos or ())
            vocab_size = getattr(config, "padded_vocab_size", None) or config.vocab_size
            LOG.info("Using vLLM model %s at %s (context %s)",
                     vllm.model, vllm.base_url, self.vllm_context_length)
        else:
            model = self.load_transformers_model()
        tokenizer_path = resolve_glm_tokenizer_path(self.args.tokenizer_path, self.args.model_path)
        LOG.info("Loading speech tokenizer from %s and decoder on %s", tokenizer_path, self.speech_device)
        whisper = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(self.speech_device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
        decoder = AudioDecoder(
            config_path=str(decoder_dir / "config.yaml"),
            flow_ckpt_path=str(decoder_dir / "flow.pt"),
            hift_ckpt_path=str(decoder_dir / "hift.pt"),
            device=self.speech_device,
        )
        decoder.flow.inference_steps = getattr(self.args, "flow_steps", 10)
        decoder.flow.eval()
        decoder.hift.eval()
        self.audio_vocab = audio_token_range(tokenizer, decoder)
        if vllm is not None:
            offset, size = self.audio_vocab
            self.vllm_text_ids = [*range(offset), *range(offset + size, vocab_size)]
        self.tokenizer, self.model, self.vllm = tokenizer, model, vllm
        self.whisper, self.feature_extractor, self.decoder = whisper, feature_extractor, decoder
        self.gpu_encoder = lru_cache(maxsize=4)(lambda rate: make_gpu_encoder(self, rate))
        self.ready = True
        LOG.info("Voice models ready")

    def load_transformers_model(self):
        import torch
        from transformers import AutoModel, BitsAndBytesConfig

        LOG.info("Loading language model (%s) on %s", self.args.quantization, self.llm_device)
        kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.llm_device.index},
            low_cpu_mem_usage=True,
        )
        attention = getattr(self.args, "attn_implementation", "auto")
        if attention != "auto":
            kwargs["attn_implementation"] = attention
        if self.args.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModel.from_pretrained(self.args.model_path, **kwargs).eval()
        LOG.info("GLM attention implementation: %s", model.config._attn_implementation)
        return model

    def decode_speech_tokens(self, tokens):
        import torch

        with torch.inference_mode():
            waveform, _ = self.decoder.token2wav(
                torch.as_tensor(tokens, dtype=torch.long, device=self.speech_device).unsqueeze(0),
                uuid=uuid.uuid4().hex, finalize=True,
            )
            return waveform.detach().float().reshape(1, -1)

    def decode_input_wav(self, tokens):
        return waveform_wav(self.decode_speech_tokens(tokens), self.output_sample_rate)

    def place_glm_inputs(self, inputs, audio_ids=None, *, non_blocking=False):
        if audio_ids is not None:
            audio_slice = audio_prompt_slice(self.tokenizer, inputs.input_ids, audio_ids.shape[-1])
        if self.vllm is None:
            inputs = inputs.to(self.llm_device)
        if audio_ids is not None:
            offset, _ = self.audio_vocab
            inputs.input_ids[:, audio_slice] = audio_ids.to(
                inputs.input_ids.device, non_blocking=non_blocking and self.vllm is None,
            ) + offset
        return inputs

    def judge_turn(self, pcm, rate, language, cancel_event, *, previous_reply="", include_previous_reply=True):
        """Check whether the caller's audio warrants a reply and retain its speech tokens."""
        import torch

        check_cancelled(cancel_event)
        instruction = turn_instruction(language, previous_reply if include_previous_reply else "")
        with torch.inference_mode():
            waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(self.speech_device, dtype=torch.float32)
            input_ids = self.gpu_encoder(rate)(waveform.unsqueeze(0).div_(32768))
            count = input_ids.shape[-1]
            if not count:
                return TurnDecision("discard"), None
            placeholder = "<|begin_of_audio|>" + "<|audio_0|>" * count + "<|end_of_audio|>"
            prompt = f"<|system|>\n{instruction}<|user|>\n{placeholder}<|assistant|>streaming_transcription\n"
            inputs = self.place_glm_inputs(self.tokenizer([prompt], return_tensors="pt"), input_ids)
            check_cancelled(cancel_event)
            if self.vllm is not None:
                generated = self.vllm.generate(
                    inputs.input_ids[0].tolist(), TURN_CHECK_MAX_TOKENS, cancel_event=cancel_event,
                    temperature=0.0, allowed_token_ids=self.vllm_text_ids, stop_token_ids=self.vllm_stop_ids,
                )
            else:
                from transformers import StoppingCriteriaList

                offset, size = self.audio_vocab
                stopping = StoppingCriteriaList([
                    lambda ids, scores, **kwargs: cancel_event is not None and cancel_event.is_set(),
                ])
                response = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=TURN_CHECK_MAX_TOKENS,
                    suppress_tokens=list(range(offset, offset + size)), stopping_criteria=stopping,
                )
                generated = response[0, inputs.input_ids.shape[1]:].tolist()
            check_cancelled(cancel_event)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
        decision = turn_decision(text, self.faq_requests, previous_reply=previous_reply)
        return decision, input_ids if decision.action not in ("discard", "echo") else None

    def synthesize_readback(self, text, language, cancel_event=None):
        """Use GLM to read supplied text verbatim."""
        import torch

        instruction = greeting_instruction(language)
        prompt = (f"<|system|>\nUser will provide you with a text instruction. {INTERLEAVED_REPLY}"
                  f"<|user|>\n{instruction}\n{text}<|assistant|>streaming_transcription\n")
        inputs = self.place_glm_inputs(self.tokenizer([prompt], return_tensors="pt"))
        offset, size = self.audio_vocab
        end = offset + size
        expected = "".join(char for char in text.casefold() if char.isalnum())
        generation = {}
        if self.vllm is None and cancel_event is not None:
            from transformers import StoppingCriteriaList

            def stop_if_interrupted(input_ids, scores, **kwargs):
                return cancel_event.is_set()

            generation["stopping_criteria"] = StoppingCriteriaList([stop_if_interrupted])
        with torch.inference_mode():
            # Try greedy generation first. A second attempt uses the conversation sampling settings.
            for options in ({"do_sample": False}, {"do_sample": True, "temperature": 0.2, "top_p": 0.8}):
                check_cancelled(cancel_event)
                if self.vllm is not None:
                    generated = self.vllm.generate(
                        inputs.input_ids[0].tolist(), 512, cancel_event=cancel_event,
                        temperature=options.get("temperature", 0.0), top_p=options.get("top_p", 1.0),
                        stop_token_ids=self.vllm_stop_ids,
                    )
                else:
                    response = self.model.generate(**inputs, max_new_tokens=512, **generation, **options)
                    generated = response[0, inputs.input_ids.shape[1]:].tolist()
                check_cancelled(cancel_event)
                audio = [token - offset for token in generated if offset <= token < end]
                transcript = self.tokenizer.decode([token for token in generated if not offset <= token < end],
                                                   skip_special_tokens=True)
                spoken = "".join(char for char in transcript.casefold() if char.isalnum())
                if audio and spoken == expected and not unspoken_text_tail(self.tokenizer, generated, offset, end):
                    waveform = self.decode_speech_tokens(audio)
                    check_cancelled(cancel_event)
                    return waveform
        return None

    def synthesize_greeting(self, greeting, language):
        waveform = self.synthesize_readback(greeting, language)
        if waveform is None:
            raise RuntimeError("模型未能完整朗读配置中的开场白。")
        return waveform_wav(waveform, self.output_sample_rate)

    def generate_audio_stream(self, generate, audio_offset, cancel_event, on_audio, profile=None,
                              block_sizes=STREAM_BLOCK_SIZES, audio_vocab=None):
        import torch

        gpu_tokens = self.vllm is None and audio_vocab is not None
        tokens = GpuGeneratedTokens(profile) if gpu_tokens else GeneratedTokens(profile)
        transfer_stream = (torch.cuda.Stream(device=self.llm_device)
                           if gpu_tokens and self.llm_device != self.speech_device else None)
        stream_id = uuid.uuid4().hex
        overlap = self.decoder.token_overlap_len
        pending = torch.empty(0, dtype=torch.long, device=self.speech_device)
        waveforms = []
        token_times = []
        first_token = first_audio_token = True
        packet = 0
        prompt_tokens = torch.empty((1, 0), dtype=torch.long, device=self.speech_device)
        prompt_mel = torch.empty((1, 0, 80), device=self.speech_device)
        stop = threading.Event()

        def run_generation():
            try:
                with torch.inference_mode():
                    if not gpu_tokens:
                        return generate(streamer=tokens, stop_stream=stop)
                    with torch.cuda.device(self.llm_device):
                        response = generate(token_sink=tokens, stop_stream=stop)
                        tokens.flush()
                    return response
            finally:
                finished = time.perf_counter() if profile else None
                if profile:
                    profile("generate_end", at=finished)
                tokens.queue.put((None, finished))

        def decode(count, ready_at, final=False):
            nonlocal pending, prompt_tokens, prompt_mel, packet
            check_cancelled(cancel_event)
            chunk = len(waveforms) + 1
            if profile:
                profile("chunk_tokens_ready", at=ready_at, chunk=chunk, tokens=count, final=final)
                profile("chunk_decode_start", chunk=chunk)
            block = pending[:count].unsqueeze(0)
            speech, mel = self.decoder.token2wav(
                block, uuid=stream_id, prompt_token=prompt_tokens, prompt_feat=prompt_mel, finalize=final,
            )
            check_cancelled(cancel_event)
            if not final:
                # The decoder withholds the overlap and prepends its vocoder cache on later blocks.
                advance = count - overlap
                cached = self.decoder.mel_cache_len if waveforms else 0
                prompt_tokens = torch.cat((prompt_tokens, block[:, :advance]), dim=1)
                prompt_mel = torch.cat((prompt_mel, mel[:, :, cached:].transpose(1, 2)), dim=1)
                pending = pending[advance:]
                del token_times[:advance]
            waveform = speech.detach().float().reshape(1, -1)
            # Send only the final PCM bytes to the host for WebSocket delivery.
            pcm = pcm16_bytes(waveform) if waveform.numel() else None
            if profile:
                profile("chunk_decoded", chunk=chunk)
            waveforms.append(waveform)
            if pcm is not None:
                if profile:
                    packet += 1
                    profile("chunk_pcm_ready", chunk=chunk, packet=packet)
                on_audio(pcm)
            check_cancelled(cancel_event)

        with ThreadPoolExecutor(max_workers=1) as executor:
            generation = executor.submit(run_generation)
            try:
                while True:
                    item = tokens.queue.get()
                    batch, produced_at = item[:2]
                    if batch is None:
                        break
                    check_cancelled(cancel_event)
                    if gpu_tokens:
                        ready, timestamps = produced_at, item[2]
                        speech_stream = torch.cuda.current_stream(self.speech_device)
                        if transfer_stream is None:
                            ready.wait(speech_stream)
                        else:
                            ready.wait(transfer_stream)
                            with torch.cuda.stream(transfer_stream):
                                batch.record_stream(transfer_stream)
                                batch = batch.to(self.speech_device, non_blocking=True)
                            speech_stream.wait_stream(transfer_stream)
                            batch.record_stream(speech_stream)
                        valid = (batch >= audio_offset) & (batch < audio_offset + audio_vocab)
                        if profile:
                            positions = valid.nonzero(as_tuple=True)[0]
                            audio = batch.index_select(0, positions) - audio_offset
                            audio_times = [timestamps[index] for index in positions.tolist()]
                            produced_at = timestamps[0]
                        else:
                            audio = batch.masked_select(valid) - audio_offset
                    else:
                        audio = torch.tensor(
                            [token - audio_offset for token in batch
                             if token >= audio_offset and (audio_vocab is None or token < audio_offset + audio_vocab)],
                            dtype=torch.long, device=self.speech_device,
                        )
                        if profile:
                            audio_times = [produced_at] * audio.numel()
                    if audio.numel():
                        pending = torch.cat((pending, audio))
                    if profile:
                        if first_token:
                            profile("first_token", at=produced_at)
                            first_token = False
                        if first_audio_token and audio_times:
                            profile("first_audio_token", at=audio_times[0])
                            first_audio_token = False
                        token_times.extend(audio_times)
                    count = block_sizes[min(len(waveforms), len(block_sizes) - 1)]
                    while len(pending) >= count:
                        decode(count, token_times[count - 1] if profile else None)
                        count = block_sizes[min(len(waveforms), len(block_sizes) - 1)]
                response = generation.result()
                check_cancelled(cancel_event)
                if pending.numel():
                    decode(len(pending), produced_at, final=True)
            finally:
                stop.set()
                self.decoder.hift_cache_dict.pop(stream_id, None)
                self.decoder.mel_overlap_dict.pop(stream_id, None)
        return response, torch.cat(waveforms, dim=1) if waveforms else None

    def prepare_input(self, question_path, input_ids, profile):
        import torch
        import torchaudio

        if profile:
            profile("input_tokenizer_start")
        if input_ids is None:
            pcm, rate = torchaudio.load(str(question_path))
            input_ids = self.gpu_encoder(rate)(pcm.to(self.speech_device))
        input_count = input_ids.shape[-1]
        if not input_count:
            raise ValueError("没有提取到语音内容，请重新录一段清晰的声音。")
        if profile:
            torch.cuda.current_stream(self.speech_device).synchronize()
            profile("input_tokenizer_ready", tokens=input_count)
        return input_ids

    def generate_reply(self, question_path, input_ids, history, context, language,
                       cancel_event, on_audio, profile, stream_block_sizes, *, clarify=False, exact_faq=False,
                       courtesy=False):
        import torch

        input_count = input_ids.shape[-1]
        placeholder = "<|begin_of_audio|>" + "<|audio_0|>" * input_count + "<|end_of_audio|>"
        previous = history
        limit = (self.vllm_context_length if self.vllm is not None else
                 min(getattr(self.model.config, "seq_length", 8192), 8192))
        while True:
            prompt = make_prompt(previous, placeholder, context, language, getattr(self.args, "system_prompt", ""),
                                 clarify=clarify, exact_faq=exact_faq, courtesy=courtesy)
            if profile:
                profile("prompt_built")
                profile("glm_tokenizer_start")
            inputs = self.tokenizer([prompt], return_tensors="pt")
            if profile:
                profile("glm_tokenizer_ready", tokens=inputs.input_ids.shape[1])
            if inputs.input_ids.shape[1] + self.args.max_new_tokens <= limit:
                break
            if not previous:
                raise ValueError("当前问题和参考资料过长，请缩短录音或知识库片段。")
            previous.pop(0)
        audio_offset, audio_vocab = self.audio_vocab
        inputs = self.place_glm_inputs(inputs, input_ids, non_blocking=True)
        if profile:
            if self.vllm is None:
                profile("prompt_on_device")
            profile("prompt_ready", tokens=inputs.input_ids.shape[1])

        def generate(stop_stream=None, token_sink=None, **options):
            if self.vllm is not None:
                prompt_ids = inputs.input_ids[0].tolist()
                streamer = options.get("streamer")
                if profile:
                    profile("generate_start")
                generated = self.vllm.generate(
                    prompt_ids, self.args.max_new_tokens, cancel_event=cancel_event, stop_stream=stop_stream,
                    on_tokens=streamer.push if streamer is not None else None,
                    temperature=0.2, top_p=0.8, stop_token_ids=self.vllm_stop_ids,
                )
                return torch.tensor([prompt_ids + generated], dtype=torch.long)
            if cancel_event is not None or stop_stream is not None or token_sink is not None:
                from transformers import StoppingCriteria, StoppingCriteriaList

                class ObserveGeneration(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        interrupted = ((cancel_event is not None and cancel_event.is_set())
                                       or (stop_stream is not None and stop_stream.is_set()))
                        if token_sink is not None and not interrupted:
                            token_sink.observe(input_ids)
                        return interrupted

                options["stopping_criteria"] = StoppingCriteriaList([ObserveGeneration()])
            if profile:
                from transformers import LogitsProcessor, LogitsProcessorList

                class MarkPrefill(LogitsProcessor):
                    marked = False

                    def __call__(self, input_ids, scores):
                        if not self.marked:
                            profile("glm_prefill_ready")
                            self.marked = True
                        return scores

                options["logits_processor"] = LogitsProcessorList([MarkPrefill()])
            check_cancelled(cancel_event)
            if profile:
                profile("generate_start")
            return self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens,
                do_sample=True, temperature=0.2, top_p=0.8, **options,
            )

        waveform = None
        if on_audio is None:
            response = generate()
        else:
            response, waveform = self.generate_audio_stream(
                generate, audio_offset, cancel_event, on_audio, profile, stream_block_sizes, audio_vocab,
            )
        check_cancelled(cancel_event)
        streamed = waveform is not None
        # Slice by input length instead of searching for a prompt marker in history.
        result_ids = response[0, inputs.input_ids.shape[1]:]
        if waveform is None:
            is_audio = (result_ids >= audio_offset) & (result_ids < audio_offset + audio_vocab)
            reply_audio = result_ids.masked_select(is_audio) - audio_offset
            if reply_audio.numel():
                if profile:
                    profile("full_audio_decoder_start", tokens=reply_audio.numel())
                waveform = self.decode_speech_tokens(reply_audio)
                if profile:
                    profile("full_audio_decoder_ready")
        generated = result_ids.tolist()
        audio_count = sum(audio_offset <= i < audio_offset + audio_vocab for i in generated)
        text_ids = [i for i in generated if not audio_offset <= i < audio_offset + audio_vocab]
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        # GLM emits text before its matching audio. A limit can end the turn during that text block.
        tail = unspoken_text_tail(self.tokenizer, generated, audio_offset, audio_offset + audio_vocab)
        if tail:
            suffix = self.synthesize_readback(tail, language, cancel_event)
            check_cancelled(cancel_event)
            if suffix is None or not suffix.numel():
                LOG.warning("Could not synthesize the remaining text of a GLM reply (%d characters)", len(tail))
            else:
                if streamed:
                    on_audio(pcm16_bytes(suffix))
                waveform = suffix if waveform is None else torch.cat((waveform, suffix), dim=1)
        raw = clean_assistant(self.tokenizer.decode(generated, skip_special_tokens=False))
        if profile:
            profile("reply_tokens_decoded", text_tokens=len(text_ids), audio_tokens=audio_count)
        question = "<|begin_of_audio|>" + "".join(f"<|audio_{i}|>" for i in input_ids[0].tolist()) + "<|end_of_audio|>"
        return VoiceReply(text, waveform, question_tokens=question, assistant_tokens=raw)
