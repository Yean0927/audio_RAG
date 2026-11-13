"""Adapters for models that own their audio encoding and waveform generation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import wave

from voice_backend import (
    LOG, TURN_CHECK_MAX_TOKENS, VoiceBackendBase, VoiceReply, answer_instruction, check_cancelled,
    greeting_instruction, pcm16_bytes, turn_decision, turn_instruction, waveform_wav,
)


QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


class AudioModelBackend(VoiceBackendBase):
    output_sample_rate = 24000

    def generate_reply(self, question_path, input_ids, history, context, language,
                       cancel_event, on_audio, profile, stream_block_sizes):
        # Paths and assistant text are portable between model families and old sessions.
        messages = []
        for turn in history:
            if turn.get("question_path") and Path(turn["question_path"]).is_file() and turn.get("text"):
                messages.extend([
                    {"role": "user", "audio": turn["question_path"], "text": ""},
                    {"role": "assistant", "text": turn["text"]},
                ])
        instruction = answer_instruction(context, language) + getattr(self.args, "system_prompt", "")
        if context:
            instruction += f"\n[Reference material]\n{context}\n[/Reference material]\n"
        messages.append({"role": "user", "text": instruction, "audio": str(question_path)})
        reply = self.infer(messages, True, self.args.max_new_tokens, cancel_event, profile)
        check_cancelled(cancel_event)
        if on_audio is not None and reply.waveform is not None and reply.waveform.numel():
            pcm = pcm16_bytes(reply.waveform.reshape(1, -1))
            if profile:
                profile("chunk_pcm_ready", chunk=1, packet=1)
            on_audio(pcm)
        return reply

    def judge_turn(self, pcm, rate, language, cancel_event):
        check_cancelled(cancel_event)
        with tempfile.TemporaryDirectory(prefix="voice-turn-") as directory:
            path = Path(directory) / "probe.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(pcm)
            reply = self.infer([
                {"role": "user", "text": turn_instruction(language), "audio": str(path)},
            ], False, TURN_CHECK_MAX_TOKENS, cancel_event)
        check_cancelled(cancel_event)
        return turn_decision(reply.text), None

    def synthesize_greeting(self, greeting, language):
        reply = self.infer([
            {"role": "user", "text": greeting_instruction(language) + "\n" + greeting},
        ], True, 512, None)
        expected = "".join(char for char in greeting.casefold() if char.isalnum())
        spoken = "".join(char for char in reply.text.casefold() if char.isalnum())
        if reply.waveform is None or not reply.waveform.numel() or spoken != expected:
            raise RuntimeError("模型未能完整朗读配置中的开场白。")
        return waveform_wav(reply.waveform, self.output_sample_rate)


class KimiAudioBackend(AudioModelBackend):
    def load(self):
        if self.ready:
            return
        import torch
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError as error:
            raise RuntimeError("Kimi-Audio 依赖加载失败，请按 VOICE_CHAT_zh.md 安装 Kimi 推理环境。") from error

        self.set_devices()
        # The official runtime allocates all audio and LLM components on current_device().
        with torch.cuda.device(self.llm_device):
            runtime = KimiAudio(model_path=self.args.model_path, load_detokenizer=True)
            runtime.alm.eval()
        self.runtime = runtime
        self.model = runtime.alm
        self.tokenizer = runtime.prompt_manager.text_tokenizer
        self.audio_tokenizer = runtime.prompt_manager.audio_tokenizer
        self.audio_encoder = runtime.prompt_manager.whisper_model
        self.decoder = runtime.detokenizer
        self.ready = True
        LOG.info("Kimi-Audio LLM, tokenizers, Whisper encoder and detokenizer ready on %s", self.llm_device)

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None):
        import torch

        check_cancelled(cancel_event)
        chats = []
        for message in messages:
            if message.get("text"):
                chats.append({"role": message["role"], "message_type": "text", "content": message["text"]})
            if message.get("audio"):
                chats.append({"role": message["role"], "message_type": "audio", "content": message["audio"]})
        output_type = "both" if speech else "text"
        runtime = self.runtime
        with torch.inference_mode(), torch.cuda.device(self.llm_device):
            if profile:
                profile("input_tokenizer_start")
            # Reference text consumes positions too. Use the model's declared context
            # capacity and the caller's generation bound for retrieval-augmented prompts.
            limit = runtime.alm.config.max_position_embeddings
            while True:
                check_cancelled(cancel_event)
                prompt = runtime.prompt_manager.get_prompt(chats, output_type=output_type)
                if len(prompt.audio_token_ids) + max_new_tokens <= limit:
                    break
                # Remove a complete oldest user/assistant pair, preserving this request.
                assistant = next((i for i, msg in enumerate(chats) if msg["role"] == "assistant"), None)
                if assistant is None:
                    break
                chats = chats[assistant + 1:]
            budget = min(max_new_tokens, limit - len(prompt.audio_token_ids))
            if budget <= runtime.kimia_text_audiodelaytokens:
                raise ValueError("Kimi-Audio 当前问题与参考资料超出上下文预算，请缩短录音或参考资料。")
            audio_ids, text_ids, continuous_mask, _, _ = prompt.to_tensor()
            if profile:
                profile("input_tokenizer_ready", tokens=audio_ids.shape[-1])
                profile("prompt_ready", tokens=audio_ids.shape[-1])
                profile("generate_start")
            check_cancelled(cancel_event)
            # The official public generate() overrides max_new_tokens for audio output.
            # Use its generation loop so the service's configured token bound is honored.
            forward = runtime.alm.forward

            def cancellable_forward(*args, **kwargs):
                check_cancelled(cancel_event)
                result = forward(*args, **kwargs)
                check_cancelled(cancel_event)
                return result

            # Kimi calls alm.forward directly and has no Transformers stopping_criteria.
            # The backend lock serializes calls. Always restore this instance's method.
            runtime.alm.forward = cancellable_forward
            try:
                audio_tokens, text_tokens = runtime._generate_loop(
                    audio_input_ids=audio_ids.to(self.llm_device),
                    text_input_ids=text_ids.to(self.llm_device),
                    is_continuous_mask=continuous_mask.to(self.llm_device),
                    continous_feature=[feature.to(self.llm_device) for feature in prompt.continuous_feature],
                    max_new_tokens=budget, output_type=output_type,
                    audio_temperature=0.8, audio_top_k=10, text_temperature=0.0, text_top_k=5,
                )
            finally:
                runtime.alm.forward = forward
            check_cancelled(cancel_event)
            if profile:
                profile("generate_end")
            offset = runtime.kimia_token_offset
            text = runtime.detokenize_text([token for token in text_tokens if token < offset]).strip()
            audio_tokens = [token - offset for token in audio_tokens if token >= offset]
            waveform = None
            if speech and audio_tokens:
                if profile:
                    profile("full_audio_decoder_start", tokens=len(audio_tokens))
                waveform = runtime.detokenize_audio(torch.tensor([audio_tokens], dtype=torch.long))
                if profile:
                    profile("full_audio_decoder_ready")
            # These IDs belong to Kimi's input tokenizer and remain local to this adapter.
            input_tokens = []
            if messages[-1].get("audio"):
                start = len(prompt.audio_token_ids) - 1 - prompt.audio_token_ids[::-1].index(runtime.extra_tokens.media_begin)
                end = prompt.audio_token_ids.index(runtime.extra_tokens.media_end, start + 1)
                input_tokens = [token - offset for token in prompt.audio_token_ids[start + 1:end]]
            check_cancelled(cancel_event)
            return VoiceReply(text, waveform, input_tokens=input_tokens)

    def decode_input_wav(self, tokens):
        import torch

        with torch.inference_mode(), torch.cuda.device(self.llm_device):
            waveform = self.runtime.detokenize_audio(torch.tensor([tokens], dtype=torch.long))
        return waveform_wav(waveform, self.output_sample_rate)


class QwenOmniBackend(AudioModelBackend):
    def load(self):
        if self.ready:
            return
        import torch
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as error:
            raise RuntimeError("Qwen2.5-Omni 需要对应 Transformers 版本，请按 VOICE_CHAT_zh.md 安装 Omni 推理环境。") from error

        self.set_devices()
        options = {"torch_dtype": torch.bfloat16, "device_map": {"": self.llm_device.index}}
        if self.args.attn_implementation != "auto":
            options["attn_implementation"] = self.args.attn_implementation
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.args.model_path, **options).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(self.args.model_path)
        self.model, self.processor = model, processor
        self.tokenizer = processor.tokenizer
        self.ready = True
        LOG.info("Qwen2.5-Omni processor, Thinker, Talker and Token2Wav ready on %s", self.llm_device)

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None):
        import librosa
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        check_cancelled(cancel_event)
        chats = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]}]
        audios = []
        for message in messages:
            check_cancelled(cancel_event)
            content = []
            if message.get("audio"):
                content.append({"type": "audio", "audio": message["audio"]})
                audios.append(librosa.load(message["audio"], sr=self.processor.feature_extractor.sampling_rate)[0])
            if message.get("text"):
                content.append({"type": "text", "text": message["text"]})
            chats.append({"role": message["role"], "content": content})
        if profile:
            profile("input_tokenizer_start")
        limit = self.model.config.thinker_config.text_config.max_position_embeddings
        while True:
            check_cancelled(cancel_event)
            text = self.processor.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=audios or None, padding=True, return_tensors="pt")
            if inputs.input_ids.shape[-1] + max_new_tokens <= limit:
                break
            if len(chats) <= 2:
                raise ValueError("Omni 当前问题与参考资料超出上下文预算，请缩短录音或参考资料。")
            if any(item["type"] == "audio" for item in chats[1]["content"]):
                audios.pop(0)
            del chats[1:3]
        inputs = inputs.to(self.llm_device).to(self.model.dtype)
        if profile:
            profile("input_tokenizer_ready", tokens=inputs.input_ids.shape[-1])
            profile("prompt_ready", tokens=inputs.input_ids.shape[-1])

        class CancelGeneration(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                check_cancelled(cancel_event)
                return False

        stopping = StoppingCriteriaList([CancelGeneration()])
        check_cancelled(cancel_event)
        with torch.inference_mode(), torch.cuda.device(self.llm_device):
            if profile:
                profile("generate_start")
            output = self.model.generate(
                **inputs, return_audio=speech, speaker=self.args.omni_speaker,
                thinker_max_new_tokens=max_new_tokens, thinker_do_sample=False,
                thinker_stopping_criteria=stopping, talker_stopping_criteria=stopping,
            )
            check_cancelled(cancel_event)
            if profile:
                profile("generate_end")
            ids, waveform = output if speech else (output, None)
            answer_ids = ids[:, inputs.input_ids.shape[-1]:]
            text = self.processor.batch_decode(answer_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()
        return VoiceReply(text, waveform)
