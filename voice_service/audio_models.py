"""Adapters for models that own their audio encoding and waveform generation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import wave

from .backend import (
    LOG, TURN_CHECK_MAX_TOKENS, VoiceBackendBase, VoiceReply, answer_instruction, check_cancelled,
    greeting_instruction, history_reference, pcm16_bytes, turn_decision, turn_instruction, waveform_wav,
)
from .links import spoken_link_references


QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


class AudioModelBackend(VoiceBackendBase):
    output_sample_rate = 24000
    normalize_spoken_links = False

    def generate_reply(self, question_path, input_ids, history, context, language,
                       cancel_event, on_audio, profile, stream_block_sizes, *, clarify=False, exact_faq=False,
                       courtesy=False):
        # Paths and assistant text are portable between model families and old sessions.
        messages = []
        for turn in history:
            if turn.get("question_path") and Path(turn["question_path"]).is_file() and turn.get("text"):
                messages.extend([
                    {"role": "user", "audio": turn["question_path"], "text": history_reference(turn)},
                    {"role": "assistant", "text": turn["text"]},
                ])
        instruction = answer_instruction(context, language, history, clarify=clarify, exact_faq=exact_faq,
                                         courtesy=courtesy)
        if not courtesy:
            instruction += getattr(self.args, "system_prompt", "")
        if self.normalize_spoken_links and not (clarify or courtesy):
            instruction += (
                "\n参考资料中有和本轮问题相关的网址时，先说用途或操作步骤，再用‘网址是’引出并放慢语速，"
                "把【网址读法】里的字母、数字和符号按顺序完整念一遍。用户追问前面的网址时也这样念。"
                "念网址时只念标记里的内容，按逗号停顿，原样保留大小写提示、端口、路径和参数。"
                "网址不受回答句数限制。用户询问网址但资料没有提供时，说明无法确认网址。"
                if language == "zh" else
                "\nWhen the references include a URL relevant to the current question, give its purpose or steps first. "
                "Say 'the web address is' and slowly read the contents of [URL reading] aloud once in order. "
                "Do the same if the caller asks for an earlier address. Pause at commas, preserve capitalization cues, "
                "ports, paths and parameters, and omit the enclosing label. Read the full address regardless of answer length. "
                "If the caller asks for an address missing from the references, say you cannot confirm it."
            )
        if context:
            instruction += f"\n[Reference material]\n{context}\n[/Reference material]\n"
        messages.append({"role": "user", "text": instruction, "audio": str(question_path)})
        if self.normalize_spoken_links:
            for message in messages:
                message["text"] = spoken_link_references(message["text"], language)
        return self.infer_reply(messages, cancel_event, on_audio, profile)

    def infer_reply(self, messages, cancel_event, on_audio, profile):
        reply = self.infer(messages, True, self.args.max_new_tokens, cancel_event, profile)
        check_cancelled(cancel_event)
        if on_audio is not None and reply.waveform is not None and reply.waveform.numel():
            pcm = pcm16_bytes(reply.waveform.reshape(1, -1))
            if profile:
                profile("chunk_pcm_ready", chunk=1, packet=1)
            on_audio(pcm)
        return reply

    def judge_turn(self, pcm, rate, language, cancel_event, *, previous_reply="", include_previous_reply=True):
        check_cancelled(cancel_event)
        with tempfile.TemporaryDirectory(prefix="voice-turn-") as directory:
            path = Path(directory) / "probe.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(pcm)
            instruction = turn_instruction(language, previous_reply if include_previous_reply else "")
            reply = self.infer([{"role": "user", "text": instruction, "audio": str(path)}],
                               False, TURN_CHECK_MAX_TOKENS, cancel_event)
        check_cancelled(cancel_event)
        return turn_decision(reply.text, self.faq_requests, previous_reply=previous_reply), None

    def synthesize_greeting(self, greeting, language):
        reply = self.infer([
            {"role": "user", "text": greeting_instruction(language) + "\n" + greeting},
        ], True, 512, None)
        expected = "".join(char for char in greeting.casefold() if char.isalnum())
        spoken = "".join(char for char in reply.text.casefold() if char.isalnum())
        if reply.waveform is None or not reply.waveform.numel() or spoken != expected:
            raise RuntimeError("模型未能完整朗读配置中的开场白。")
        return waveform_wav(reply.waveform, self.output_sample_rate)


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
