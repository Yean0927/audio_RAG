"""Browser microphone chat for speech-to-speech models, with optional SONAR retrieval.

Run from the server's project directory: python voice_web_demo.py
Models load on the first request or when the load button is pressed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import os
from pathlib import Path
from queue import SimpleQueue
import threading
import time
import uuid

from voice_backend import (
    BACKENDS, ROOT, STREAM_BLOCK_SIZES, TURN_CHECK_MAX_TOKENS, InferenceCancelled as InferenceCancelled,
    VoiceBackendBase, VoiceReply, answer_instruction, check_cancelled, greeting_instruction, pcm16_bytes,
    turn_decision, turn_instruction, waveform_wav,
)
from voice_gpu import GpuGeneratedTokens, audio_prompt_slice, audio_token_range, make_gpu_encoder

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

LOG = logging.getLogger("voice-chat")
INTERLEAVED_REPLY = "Respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens. "
AUDIO_FIRST_REPLY = (
    "Generate the complete spoken response first using audio tokens. After the audio is complete, "
    "output the matching complete text response using text tokens. Do not alternate between text and audio tokens. "
)
AUDIO_FIRST = "User will provide you with a speech instruction. " + AUDIO_FIRST_REPLY


class GeneratedTokens:
    """Forward newly generated IDs from the Transformers generation thread."""

    def __init__(self, profile=None):
        self.queue = SimpleQueue()
        self.received_prompt = False
        self.profile = profile

    def put(self, value):
        if self.received_prompt:
            ids = value.flatten().tolist()
            self.queue.put((ids, time.perf_counter() if self.profile else None))
        self.received_prompt = True

    def end(self):
        pass  # The worker signals completion even if generate() raises.


def make_prompt(history, question, context, language, persona=""):
    instruction = answer_instruction(context, language)
    prompt = f"<|system|>\n{AUDIO_FIRST}{instruction}{persona}"
    for turn in history:
        prompt += f"<|user|>\n{turn['question_tokens']}<|assistant|>streaming_transcription\n{turn['assistant_tokens']}"
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


class VoiceBackend(VoiceBackendBase):
    """GLM-4-Voice with split-device encoding and streaming audio."""

    def load(self):
        """Caller holds the lock, including during lazy model initialization."""
        if self.ready:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, WhisperFeatureExtractor
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder
        from spoken_chatbot.model import AudioDecoder

        decoder_dir = Path(self.args.flow_path).resolve()
        missing = [name for name in ("config.yaml", "flow.pt", "hift.pt") if not (decoder_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"语音 Decoder 缺少文件：{', '.join(missing)}。请先下载到 {decoder_dir}，或指定 --flow-path。"
            )
        self.set_devices()

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
        # Keep partially loaded objects local so failed initialization can be retried.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.args.model_path, **kwargs).eval()
        LOG.info("GLM attention implementation: %s", model.config._attn_implementation)
        LOG.info("Loading speech tokenizer and decoder on %s", self.speech_device)
        whisper = WhisperVQEncoder.from_pretrained(self.args.tokenizer_path).eval().to(self.speech_device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.args.tokenizer_path)
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
        self.tokenizer, self.model = tokenizer, model
        self.whisper, self.feature_extractor, self.decoder = whisper, feature_extractor, decoder
        self.gpu_encoder = lru_cache(maxsize=4)(lambda rate: make_gpu_encoder(self, rate))
        self.ready = True
        LOG.info("Voice models ready")

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

    def judge_turn(self, pcm, rate, language, cancel_event):
        """Classify the caller's audio at a possible pause and retain its speech tokens."""
        import torch
        from transformers import StoppingCriteriaList

        check_cancelled(cancel_event)
        instruction = turn_instruction(language)
        with torch.inference_mode():
            waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(self.speech_device, dtype=torch.float32)
            input_ids = self.gpu_encoder(rate)(waveform.unsqueeze(0).div_(32768))
            count = input_ids.shape[-1]
            if not count:
                return "discard", None
            placeholder = "<|begin_of_audio|>" + "<|audio_0|>" * count + "<|end_of_audio|>"
            prompt = f"<|system|>\n{instruction}<|user|>\n{placeholder}<|assistant|>streaming_transcription\n"
            inputs = self.tokenizer([prompt], return_tensors="pt")
            audio_slice = audio_prompt_slice(self.tokenizer, inputs.input_ids, count)
            inputs = inputs.to(self.llm_device)
            offset, size = self.audio_vocab
            inputs.input_ids[:, audio_slice] = input_ids.to(self.llm_device) + offset
            stopping = StoppingCriteriaList([
                lambda ids, scores, **kwargs: cancel_event is not None and cancel_event.is_set(),
            ])
            check_cancelled(cancel_event)
            response = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=TURN_CHECK_MAX_TOKENS,
                suppress_tokens=list(range(offset, offset + size)), stopping_criteria=stopping,
            )
            check_cancelled(cancel_event)
            text = self.tokenizer.decode(response[0, inputs.input_ids.shape[1]:].tolist(), skip_special_tokens=True)
        decision = turn_decision(text)
        return decision, input_ids if decision != "discard" else None

    def synthesize_readback(self, text, language, cancel_event=None):
        """Use GLM to read supplied text verbatim."""
        import torch

        instruction = greeting_instruction(language)
        prompt = (f"<|system|>\nUser will provide you with a text instruction. {INTERLEAVED_REPLY}"
                  f"<|user|>\n{instruction}\n{text}<|assistant|>streaming_transcription\n")
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.llm_device)
        offset, size = self.audio_vocab
        end = offset + size
        expected = "".join(char for char in text.casefold() if char.isalnum())
        generation = {}
        if cancel_event is not None:
            from transformers import StoppingCriteriaList

            def stop_if_interrupted(input_ids, scores, **kwargs):
                return cancel_event.is_set()

            generation["stopping_criteria"] = StoppingCriteriaList([stop_if_interrupted])
        with torch.inference_mode():
            # Try greedy generation first. A second attempt uses the conversation sampling settings.
            for options in ({"do_sample": False}, {"do_sample": True, "temperature": 0.2, "top_p": 0.8}):
                check_cancelled(cancel_event)
                response = self.model.generate(**inputs, max_new_tokens=512, **generation, **options)
                check_cancelled(cancel_event)
                generated = response[0, inputs.input_ids.shape[1]:].tolist()
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

        gpu_tokens = audio_vocab is not None
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
                        audio = torch.tensor([token - audio_offset for token in batch if token >= audio_offset],
                                             dtype=torch.long, device=self.speech_device)
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
                    if len(pending) >= count:
                        decode(count, token_times[count - 1] if profile else None)
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
                       cancel_event, on_audio, profile, stream_block_sizes):
        import torch

        input_count = input_ids.shape[-1]
        placeholder = "<|begin_of_audio|>" + "<|audio_0|>" * input_count + "<|end_of_audio|>"
        previous = history
        limit = min(getattr(self.model.config, "seq_length", 8192), 8192)
        while True:
            prompt = make_prompt(previous, placeholder, context, language, getattr(self.args, "system_prompt", ""))
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
        audio_slice = audio_prompt_slice(self.tokenizer, inputs.input_ids, input_count)
        inputs = inputs.to(self.llm_device)
        inputs.input_ids[:, audio_slice] = input_ids.to(self.llm_device, non_blocking=True) + audio_offset
        if profile:
            profile("prompt_on_device")
            profile("prompt_ready", tokens=inputs.input_ids.shape[1])

        def generate(stop_stream=None, token_sink=None, **options):
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


def create_backend(args):
    name = getattr(args, "backend", "glm")
    if name == "glm":
        return VoiceBackend(args)
    from voice_audio_models import KimiAudioBackend, QwenOmniBackend
    return {"kimi": KimiAudioBackend, "omni": QwenOmniBackend}[name](args)


def build_ui(args, backend=None):
    import gradio as gr
    backend = backend or create_backend(args)

    def load_models():
        try:
            with backend.lock:
                backend.load()
            return "语音模型已就绪，可以录音提问。"
        except Exception as error:
            LOG.exception("Model initialization failed")
            raise gr.Error(str(error)) from error

    def answer(*values):
        try:
            with backend.lock:
                return backend.answer(*values)
        except Exception as error:
            LOG.exception("Voice request failed")
            raise gr.Error(str(error)) from error

    model_name = BACKENDS[getattr(args, "backend", "glm")][0]
    with gr.Blocks(title=f"{model_name} 语音对话", analytics_enabled=False) as app:
        gr.Markdown(f"# {model_name} 语音对话\n录音结束后点击「发送语音」，查看文字回答并播放回复。支持连续多轮提问。")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav", label="录音或上传音频")
                with gr.Row():
                    send = gr.Button("发送语音", variant="primary")
                    clear = gr.Button("开始新对话")
                language = gr.Radio(choices=[("中文", "zh"), ("English", "en")], value="zh", label="对话语言")
                use_rag = gr.Checkbox(label="使用知识库回答", value=False)
                with gr.Accordion("知识库（可选）", open=False):
                    knowledge_file = gr.File(file_types=[".txt", ".md"], type="filepath", label="上传 UTF-8 文本文件")
                    knowledge = gr.Textbox(lines=6, label="或直接粘贴知识库内容", placeholder="开启知识库后，上传的文件优先于此处文本。")
                load = gr.Button("提前加载模型")
                status = gr.Textbox(value="等待录音。首次加载模型可能需要下载权重，请同时查看服务器终端。", label="状态", interactive=False)
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="对话记录", height=400)
                reply_audio = gr.Audio(label="本轮语音回答", type="filepath", interactive=False)
                with gr.Accordion("本轮检索到的参考内容", open=False):
                    context = gr.Textbox(lines=8, label="参考内容", interactive=False)
        send.click(
            answer, [audio, use_rag, knowledge, knowledge_file, language, state],
            [chat, reply_audio, context, status, state], concurrency_limit=1, concurrency_id="voice-model",
            api_name="chat",
        )
        load.click(load_models, outputs=status, concurrency_limit=1, concurrency_id="voice-model", api_name="load_models")
        clear.click(
            lambda: ([], None, "", "已开始新对话。", []),
            outputs=[chat, reply_audio, context, status, state],
            concurrency_limit=1, concurrency_id="voice-model",
            api_name="new_chat",
        )
    return app


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--backend", choices=BACKENDS, default="glm", help="语音模型后端")
    parser.add_argument("--model-path", help="本地模型目录或 Hugging Face ID，默认随 --backend 选择")
    parser.add_argument("--tokenizer-path", help="GLM 独立语音分词器目录或 Hugging Face ID")
    parser.add_argument("--flow-path", help="GLM 独立语音 Decoder 目录")
    parser.add_argument("--flow-steps", type=int,
                        help="语音 Flow 的 Euler 采样步数（4–10，默认 10）")
    parser.add_argument("--quantization", choices=["4bit", "bf16"], help="默认 GLM 4bit，Kimi/Omni bf16")
    parser.add_argument("--omni-speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"],
                        default="auto", help="GLM 注意力实现；auto 保持 Transformers 默认选择")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--speech-device", help="GLM 编解码及 SONAR 设备，默认 GLM cuda:1，Kimi/Omni 跟随 --llm-device")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument("--max-audio-seconds", type=int, default=30)
    parser.add_argument("--retrieval-metric", choices=("euclidean", "cosine"), default="euclidean",
                        help="SONAR 检索使用平方欧氏距离或余弦相似度（默认 euclidean）")
    parser.add_argument("--compare-retrieval", action="store_true",
                        help="对同一轮录音记录两种检索排序，回答使用 --retrieval-metric 指定的结果")
    parser.add_argument("--output-dir", default=str(ROOT / "answer_data/web"))
    return parser


def validate_args(args, parser):
    if args.backend != "glm":
        if args.tokenizer_path or args.flow_path or args.flow_steps is not None:
            parser.error("--tokenizer-path / --flow-path / --flow-steps 仅适用于 GLM；Kimi/Omni 从自身模型目录加载组件")
        if args.quantization == "4bit":
            parser.error("Kimi/Omni 当前支持 --quantization bf16")
        if args.backend == "kimi" and args.attn_implementation != "auto":
            parser.error("Kimi-Audio 由官方运行时选择注意力实现，请使用 --attn-implementation auto")
    args.model_path = args.model_path or BACKENDS[args.backend][1]
    args.tokenizer_path = args.tokenizer_path or "zai-org/glm-4-voice-tokenizer"
    args.flow_path = args.flow_path or str(ROOT / "glm-4-voice-decoder")
    args.flow_steps = 10 if args.flow_steps is None else args.flow_steps
    args.quantization = args.quantization or ("4bit" if args.backend == "glm" else "bf16")
    args.speech_device = args.speech_device or ("cuda:1" if args.backend == "glm" else args.llm_device)
    if not 4 <= args.flow_steps <= 10:
        parser.error("--flow-steps must be 4–10")
    if not 64 <= args.max_new_tokens <= 2048 or not 0 <= args.history_turns <= 10:
        parser.error("--max-new-tokens must be 64–2048; --history-turns must be 0–10")
    if not 1 <= args.max_audio_seconds <= 60:
        parser.error("--max-audio-seconds must be 1–60")
    return args


def parse_args(argv=None):
    parser = create_parser()
    return validate_args(parser.parse_args(argv), parser)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)  # Decoder YAML and Matcha imports contain repository-relative paths.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    app = build_ui(args)
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True,
        allowed_paths=[str(Path(args.output_dir).resolve())],
    )


if __name__ == "__main__":
    main()
