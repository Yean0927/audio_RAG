"""Backend contracts using CPU tensors and mocked model runtimes."""
from contextlib import nullcontext, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio

from voice_audio_models import AudioModelBackend, KimiAudioBackend, QwenOmniBackend, QWEN_SYSTEM
from voice_backend import InferenceCancelled, VoiceReply, pcm16_bytes
from voice_call_server import parse_args, prepare_decoder
from voice_web_demo import VoiceBackend, create_backend
from test_voice_call import recording


class BackendTests(unittest.TestCase):
    def test_glm_stream_reuses_the_last_block_size_and_flushes_remaining_tokens(self):
        backend = VoiceBackend(parse_args([]))
        backend.speech_device = torch.device("cpu")
        decoder = backend.decoder = SimpleNamespace(
            token_overlap_len=5, mel_cache_len=1, hift_cache_dict={}, mel_overlap_dict={},
        )

        def decode(block, *, finalize, **kwargs):
            playable = block if finalize else block[:, :-decoder.token_overlap_len]
            return playable.float() / 1000, torch.zeros(1, 80, 2)

        decoder.token2wav = Mock(side_effect=decode)
        offset = 1000
        audio = list(range(offset, offset + 367))

        def generate(*, streamer, stop_stream):
            streamer.put(torch.tensor([1]))
            for token in audio:
                streamer.put(torch.tensor([token]))
            return torch.tensor([[1, *audio]])

        chunks = []
        _, waveform = backend.generate_audio_stream(
            generate, offset, None, chunks.append, block_sizes=(25, 50, 100),
        )
        calls = decoder.token2wav.call_args_list
        self.assertEqual([call.args[0].size(1) for call in calls], [25, 50, 100, 100, 100, 17])
        self.assertTrue(calls[-1].kwargs["finalize"])
        torch.testing.assert_close(waveform, torch.arange(367).reshape(1, -1).float() / 1000)
        self.assertEqual(b"".join(chunks), pcm16_bytes(waveform))

    def test_glm_stream_appends_unspoken_text_without_replaying_the_prefix(self):
        backend = VoiceBackend(parse_args([]))
        backend.llm_device = torch.device("cpu")
        backend.audio_vocab = (1000, 100)
        backend.model = SimpleNamespace(config=SimpleNamespace(seq_length=8192))
        inputs = Mock(input_ids=torch.tensor([[1, 2, 1000, 3, 4]]))
        inputs.to.return_value = inputs
        backend.tokenizer = Mock(return_value=inputs)
        backend.tokenizer.convert_tokens_to_ids.side_effect = {
            "<|begin_of_audio|>": 2, "<|end_of_audio|>": 3,
        }.__getitem__
        backend.tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            {10: "第一句。", 11: "后半段。"}.get(token, "") for token in ids
        )
        prefix = torch.tensor([[0.1, 0.2]])
        suffix = torch.tensor([[0.3, 0.4]])
        backend.synthesize_readback = Mock(return_value=suffix)

        def stream(generate, offset, cancel, on_audio, *args):
            on_audio(pcm16_bytes(prefix))
            return torch.tensor([[1, 2, 1005, 3, 4, 10, 1002, 1003, 11]]), prefix

        backend.generate_audio_stream = stream
        chunks = []
        reply = backend.generate_reply(None, torch.tensor([[5]]), [], "", "zh", None, chunks.append, None, (25,))
        self.assertEqual(reply.text, "第一句。后半段。")
        backend.synthesize_readback.assert_called_once_with("后半段。", "zh", None)
        self.assertEqual(chunks, [pcm16_bytes(prefix), pcm16_bytes(suffix)])
        torch.testing.assert_close(reply.waveform, torch.cat((prefix, suffix), dim=1))

    def test_backend_defaults_and_factory(self):
        for name, model, implementation in (
            ("glm", "zai-org/glm-4-voice-9b", VoiceBackend),
            ("kimi", "moonshotai/Kimi-Audio-7B-Instruct", KimiAudioBackend),
            ("omni", "Qwen/Qwen2.5-Omni-7B", QwenOmniBackend),
        ):
            with self.subTest(backend=name):
                args = parse_args(["--backend", name])
                self.assertEqual(args.model_path, model)
                self.assertEqual(args.quantization, "4bit" if name == "glm" else "bf16")
                self.assertEqual(args.speech_device, "cuda:1" if name == "glm" else "cuda:0")
                self.assertEqual(args.max_new_tokens, 1024 if name == "glm" else 512)
                self.assertIsInstance(create_backend(args), implementation)
        args = parse_args(["--backend", "kimi", "--model-path", "/models/kimi", "--llm-device", "cuda:2"])
        self.assertEqual((args.model_path, args.speech_device), ("/models/kimi", "cuda:2"))

    def test_incompatible_component_options_fail_before_loading(self):
        for options in (
            ["--backend", "kimi", "--flow-path", "/glm"],
            ["--backend", "omni", "--tokenizer-path", "/glm"],
            ["--backend", "kimi", "--quantization", "4bit"],
            ["--backend", "kimi", "--attn-implementation", "sdpa"],
        ):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(options)

    def test_integrated_models_skip_glm_decoder_preparation(self):
        with patch("huggingface_hub.snapshot_download") as download:
            for backend in ("kimi", "omni"):
                prepare_decoder(parse_args(["--backend", backend, "--download-models"]))
        download.assert_not_called()

    def test_kimi_loads_all_components_and_retries_failed_initialization(self):
        backend = KimiAudioBackend(parse_args(["--backend", "kimi"]))
        backend.llm_device = torch.device("cuda:2")
        backend.set_devices = Mock()
        runtime = SimpleNamespace(
            alm=Mock(), detokenizer=object(),
            prompt_manager=SimpleNamespace(text_tokenizer=object(), audio_tokenizer=object(), whisper_model=object()),
        )
        factory = Mock(side_effect=[RuntimeError("load failed"), runtime])
        with patch.dict(sys.modules, {"kimia_infer.api.kimia": SimpleNamespace(KimiAudio=factory)}), \
                patch("torch.cuda.device", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                backend.load()
            self.assertFalse(backend.ready)
            self.assertFalse(hasattr(backend, "runtime"))
            backend.load()
            backend.load()
        self.assertEqual(factory.call_count, 2)
        factory.assert_called_with(model_path=backend.args.model_path, load_detokenizer=True)
        self.assertIs(backend.audio_tokenizer, runtime.prompt_manager.audio_tokenizer)
        self.assertIs(backend.audio_encoder, runtime.prompt_manager.whisper_model)
        self.assertIs(backend.decoder, runtime.detokenizer)
        self.assertIs(backend.tokenizer, runtime.prompt_manager.text_tokenizer)

    def test_omni_loads_its_processor_and_full_generation_model(self):
        backend = QwenOmniBackend(parse_args(["--backend", "omni", "--attn-implementation", "sdpa"]))
        backend.llm_device = torch.device("cuda:2")
        backend.set_devices = Mock()
        model = Mock()
        model.eval.return_value = model
        factory = SimpleNamespace(from_pretrained=Mock(return_value=model))
        processor = Mock()
        processor_factory = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        module = SimpleNamespace(Qwen2_5OmniForConditionalGeneration=factory,
                                 Qwen2_5OmniProcessor=processor_factory)
        with patch.dict(sys.modules, {"transformers": module}):
            backend.load()
            backend.load()
        factory.from_pretrained.assert_called_once_with(
            backend.args.model_path, torch_dtype=torch.bfloat16, device_map={"": 2}, attn_implementation="sdpa",
        )
        processor_factory.from_pretrained.assert_called_once_with(backend.args.model_path)
        self.assertIs(backend.model, model)
        self.assertIs(backend.processor, processor)

    def test_shared_answer_preserves_rag_history_and_native_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args([
                "--backend", "kimi", "--output-dir", directory,
                "--retrieval-metric", "cosine", "--compare-retrieval",
            ])
            backend = AudioModelBackend(args)
            backend.load = Mock()
            backend.infer = Mock(return_value=VoiceReply("测试回复", torch.ones(1, 2400) / 10,
                                                        input_tokens=[12, 13]))
            backend.prepare_knowledge = Mock()
            backend.embedding = SimpleNamespace(embed_query_tensor=Mock(return_value=torch.tensor([1.0, 0.0])))
            backend.chunks = ["欧氏首选", "其他条目", "余弦首选", "无关条目"]
            backend.vectors = torch.tensor([[1.0, 0.5], [0.0, 1.0], [3.0, 0.0], [-1.0, 0.0]])
            retrieval = {
                "selected_metric": "cosine",
                "cosine": [{"rank": 1, "score": 1.0, "passage": "余弦首选"}],
                "euclidean": [{"rank": 1, "score": 0.25, "passage": "欧氏首选"}],
            }
            source = Path(directory) / "input.wav"
            source.write_bytes(recording())
            old_turn = {"question_path": str(source), "text": "上轮回复", "question_tokens": "GLM-only"}
            publish, audio = Mock(), Mock()
            result = backend.answer(str(source), True, "知识库", None, "zh", [old_turn],
                                    on_audio=audio, on_retrieval=publish)
            _, wav, context, _, history = result
            self.assertEqual(context, "余弦首选")
            self.assertEqual(Path(wav).with_name("context.txt").read_text(encoding="utf-8"), context)
            self.assertEqual(torchaudio.info(wav).sample_rate, 24000)
            self.assertEqual(history[0], old_turn)
            self.assertEqual(history[-1]["input_tokens"], [12, 13])
            self.assertEqual(history[-1]["question_tokens"], "")
            self.assertEqual(len(audio.call_args.args[0]), 4800)
            publish.assert_called_once_with(retrieval)
            messages = backend.infer.call_args.args[0]
            self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
            self.assertIn(context, messages[-1]["text"])
            for passage in ("欧氏首选", "其他条目", "无关条目"):
                self.assertNotIn(passage, messages[-1]["text"])
            self.assertNotIn("GLM-only", str(messages))
            turn = json.loads(Path(wav).with_name("turn.json").read_text())
            self.assertEqual(turn["retrieval"], retrieval)

    def test_cancelled_native_reply_emits_no_audio_or_history(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = AudioModelBackend(parse_args(["--backend", "kimi", "--output-dir", directory]))
            backend.load = Mock()
            event, audio = threading.Event(), Mock()

            def infer(*args):
                event.set()
                return VoiceReply("取消的回复", torch.ones(1, 20))

            backend.infer = infer
            source = Path(directory) / "input.wav"
            source.write_bytes(recording())
            with self.assertRaises(InferenceCancelled):
                backend.answer(str(source), False, "", None, "zh", [], event, audio)
            audio.assert_not_called()
            self.assertFalse(list(Path(directory).rglob("turn.json")))

    def kimi(self, prompt_length=5):
        backend = KimiAudioBackend(parse_args(["--backend", "kimi"]))
        backend.llm_device = torch.device("cpu")
        ids = [500, 1012, 1013, 501] + [99] * (prompt_length - 4)
        prompt = SimpleNamespace(
            audio_token_ids=ids, continuous_feature=[torch.ones(1, 2, 3)],
            to_tensor=lambda: (torch.tensor([ids]), torch.zeros(1, len(ids), dtype=torch.long),
                               torch.zeros(1, len(ids), dtype=torch.bool), None, None),
        )
        runtime = SimpleNamespace(
            alm=SimpleNamespace(forward=Mock(return_value="logits"),
                                config=SimpleNamespace(max_position_embeddings=8192)),
            prompt_manager=SimpleNamespace(get_prompt=Mock(return_value=prompt)),
            kimia_token_offset=1000, kimia_text_audiodelaytokens=6,
            extra_tokens=SimpleNamespace(media_begin=500, media_end=501),
            _generate_loop=Mock(return_value=([1014, 1015, 501], [11, 12, 1001])),
            detokenize_text=Mock(return_value="Kimi 回复"),
            detokenize_audio=Mock(return_value=torch.ones(1, 20)),
        )
        backend.runtime = runtime
        return backend, runtime

    def test_kimi_uses_dual_stream_input_and_native_decoder(self):
        backend, runtime = self.kimi()
        with patch("torch.cuda.device", return_value=nullcontext()):
            reply = backend.infer([{"role": "user", "text": "资料", "audio": "question.wav"}], True, 64, None)
        self.assertEqual(reply.input_tokens, [12, 13])
        self.assertEqual(reply.text, "Kimi 回复")
        runtime.detokenize_text.assert_called_once_with([11, 12])
        self.assertEqual(runtime.detokenize_audio.call_args.args[0].tolist(), [[14, 15]])
        options = runtime._generate_loop.call_args.kwargs
        self.assertEqual(options["max_new_tokens"], 64)
        self.assertEqual(options["output_type"], "both")
        self.assertEqual(options["continous_feature"][0].shape, (1, 2, 3))
        self.assertEqual([chat["message_type"] for chat in runtime.prompt_manager.get_prompt.call_args.args[0]],
                         ["text", "audio"])

    def test_kimi_caps_audio_budget_and_text_mode_skips_decoder(self):
        backend, runtime = self.kimi(prompt_length=8142)
        with patch("torch.cuda.device", return_value=nullcontext()):
            backend.infer([{"role": "user", "text": "hello"}], True, 768, None)
            self.assertEqual(runtime._generate_loop.call_args.kwargs["max_new_tokens"], 50)
            runtime.detokenize_audio.reset_mock()
            reply = backend.infer([{"role": "user", "text": "hello"}], False, 8, None)
        self.assertIsNone(reply.waveform)
        runtime.detokenize_audio.assert_not_called()

    def test_kimi_cancellation_restores_forward_for_retry(self):
        backend, runtime = self.kimi()
        event = threading.Event()
        forward = runtime.alm.forward

        def generate(**kwargs):
            event.set()
            runtime.alm.forward()

        runtime._generate_loop.side_effect = generate
        with patch("torch.cuda.device", return_value=nullcontext()), self.assertRaises(InferenceCancelled):
            backend.infer([{"role": "user", "text": "hello"}], True, 64, event)
        self.assertIs(runtime.alm.forward, forward)
        runtime.detokenize_audio.assert_not_called()
        event.clear()
        runtime._generate_loop.side_effect = None
        with patch("torch.cuda.device", return_value=nullcontext()):
            self.assertEqual(backend.infer([{"role": "user", "text": "hello"}], False, 8, event).text, "Kimi 回复")

    def test_turn_judgment_uses_native_text_generation_and_removes_probe_file(self):
        backend = AudioModelBackend(parse_args(["--backend", "kimi"]))
        backend.infer = Mock(return_value=VoiceReply("1: 好", None))
        self.assertEqual(backend.judge_turn(recording()[44:], 16000, "zh", None), ("reply", None))
        self.assertFalse(Path(backend.infer.call_args.args[0][0]["audio"]).exists())
        self.assertFalse(backend.infer.call_args.args[1])
        for label in ("  ", "3", "1", "2:", "1: [cough]", "2：（打喷嚏）", "1: …", "ambiguous"):
            backend.infer.return_value = VoiceReply(label, None)
            self.assertEqual(backend.judge_turn(recording()[44:], 16000, "zh", None), ("discard", None))
        backend.infer.return_value = VoiceReply("2：I wanted to", None)
        self.assertEqual(backend.judge_turn(recording()[44:], 16000, "en", None), ("wait", None))

    def test_glm_turn_judgment_requires_spoken_text_and_reuses_audio_tokens(self):
        from transformers import BatchEncoding

        backend = VoiceBackend(parse_args([]))
        backend.speech_device = backend.llm_device = torch.device("cpu")
        backend.audio_vocab = (1000, 10)
        tokens = torch.tensor([[5]])
        backend.gpu_encoder = Mock(return_value=Mock(return_value=tokens))
        backend.tokenizer = Mock(side_effect=lambda *args, **kwargs: BatchEncoding({
            "input_ids": torch.tensor([[1, 2, 1000, 3, 4]]),
        }))
        backend.tokenizer.convert_tokens_to_ids.side_effect = {"<|begin_of_audio|>": 2, "<|end_of_audio|>": 3}.__getitem__
        backend.model = SimpleNamespace(generate=Mock(return_value=torch.tensor([[1, 2, 1005, 3, 4, 11]])))

        backend.tokenizer.decode.return_value = "1"
        self.assertEqual(backend.judge_turn(recording()[44:], 16000, "zh", None), ("discard", None))
        backend.tokenizer.decode.return_value = "2: 我想咨询"
        decision, input_ids = backend.judge_turn(recording()[44:], 16000, "zh", None)
        self.assertEqual(decision, "wait")
        self.assertIs(input_ids, tokens)
        self.assertEqual(backend.tokenizer.decode.call_args.args[0], [11])
        options = backend.model.generate.call_args.kwargs
        self.assertEqual(options["input_ids"][0, 2].item(), 1005)
        self.assertEqual(options["suppress_tokens"], list(range(1000, 1010)))

    def test_omni_uses_own_processor_and_decodes_only_new_text(self):
        from transformers import BatchFeature

        backend = QwenOmniBackend(parse_args(["--backend", "omni"]))
        backend.llm_device = torch.device("cpu")
        config = SimpleNamespace(thinker_config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100)))
        backend.model = SimpleNamespace(dtype=torch.bfloat16, config=config,
                                        generate=Mock(return_value=(torch.tensor([[1, 2, 3, 4, 5]]), torch.ones(1, 10))))
        backend.processor = Mock(return_value=BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])}))
        backend.processor.batch_decode.return_value = ["Omni 回复"]
        with patch("torch.cuda.device", return_value=nullcontext()):
            reply = backend.infer([{"role": "user", "text": "你好"}], True, 64, None)
        self.assertEqual(reply.text, "Omni 回复")
        self.assertEqual(backend.processor.batch_decode.call_args.args[0].tolist(), [[4, 5]])
        options = backend.model.generate.call_args.kwargs
        self.assertTrue(options["return_audio"])
        self.assertEqual(options["speaker"], "Chelsie")
        self.assertEqual(options["thinker_max_new_tokens"], 64)
        chats = backend.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(chats[0]["content"][0]["text"], QWEN_SYSTEM)


if __name__ == "__main__":
    unittest.main()
