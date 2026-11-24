"""Backend contracts using CPU tensors and mocked model runtimes."""
import asyncio
import base64
from contextlib import nullcontext, redirect_stderr
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import torch
import torchaudio

from voice_service.api import MingAPIBackend, MiniCPMOAPIBackend, QwenOmniAPIBackend
from voice_service.audio_models import AudioModelBackend, QwenOmniBackend, QWEN_SYSTEM
from voice_service.backend import InferenceCancelled, VoiceReply, pcm16_bytes
from voice_call_server import parse_args, prepare_decoder
from voice_web_demo import VoiceBackend, create_backend
from test_voice_call import recording


class SpeechStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self.chunks

    async def aclose(self):
        self.closed = True
        await self.chunks.aclose()


def speech_event(event_type, **fields):
    return f"data: {json.dumps({'type': event_type, **fields})}\r\n\r\n".encode()


class BackendTests(unittest.TestCase):
    def test_vllm_reply_stream_filters_ids_and_decodes_multiple_blocks(self):
        backend = VoiceBackend(parse_args(["--vllm-base-url", "http://127.0.0.1:18000/v1"]))
        backend.vllm = Mock()
        backend.vllm_context_length = 8192
        backend.vllm_stop_ids = [4]
        backend.audio_vocab = (1000, 100)
        backend.speech_device = torch.device("cpu")
        backend.tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[1, 2, 1000, 3]])))
        backend.tokenizer.convert_tokens_to_ids.side_effect = {
            "<|begin_of_audio|>": 2, "<|end_of_audio|>": 3,
        }.__getitem__
        backend.tokenizer.decode.side_effect = lambda ids, **kwargs: "你好" if 10 in ids else ""
        backend.decoder = SimpleNamespace(token_overlap_len=5, mel_cache_len=1,
                                          hift_cache_dict={}, mel_overlap_dict={})

        def decode(block, *, finalize, **kwargs):
            playable = block if finalize else block[:, :-5]
            return playable.float() / 1000, torch.zeros(1, 80, 2)

        backend.decoder.token2wav = Mock(side_effect=decode)

        def generate(prompt_ids, max_tokens, *, on_tokens, **kwargs):
            self.assertEqual(prompt_ids, [1, 2, 1007, 3])
            self.assertEqual(kwargs["stop_token_ids"], [4])
            ids = [10, 1100, *range(1000, 1066)]
            on_tokens(ids)
            return ids

        backend.vllm.generate.side_effect = generate
        chunks = []
        reply = backend.generate_reply(None, torch.tensor([[7]]), [], "", "zh", None, chunks.append, None, (25,))
        self.assertEqual(reply.text, "你好")
        calls = backend.decoder.token2wav.call_args_list
        self.assertEqual([call.args[0].size(1) for call in calls], [25, 25, 25, 6])
        torch.testing.assert_close(reply.waveform, torch.arange(66).reshape(1, -1).float() / 1000)
        self.assertEqual(b"".join(chunks), pcm16_bytes(reply.waveform))

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
            ("omni-local", "Qwen/Qwen2.5-Omni-7B", QwenOmniBackend),
        ):
            with self.subTest(backend=name):
                args = parse_args(["--backend", name])
                self.assertEqual(args.model_path, model)
                self.assertEqual(args.quantization, "4bit" if name == "glm" else "bf16")
                self.assertEqual(args.speech_device, "cuda:1" if name == "glm" else "cuda:0")
                self.assertEqual(args.max_new_tokens, 1024 if name == "glm" else 512)
                self.assertIsInstance(create_backend(args), implementation)
        args = parse_args(["--backend", "omni-local", "--model-path", "/models/omni", "--llm-device", "cuda:2"])
        self.assertEqual((args.model_path, args.speech_device), ("/models/omni", "cuda:2"))

    def test_local_backends_reject_unavailable_cuda_devices(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            for options in (
                ["--backend", "omni-local", "--llm-device", "cpu"],
                ["--backend", "omni-local", "--speech-device", "cuda:1"],
                ["--backend", "glm", "--speech-device", "cpu"],
            ):
                with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, "不可用"):
                    create_backend(parse_args(options)).set_devices()

    def test_incompatible_component_options_fail_before_loading(self):
        for options in (
            ["--backend", "omni-local", "--flow-path", "/glm"],
            ["--backend", "omni", "--tokenizer-path", "/glm"],
            ["--backend", "omni-local", "--quantization", "4bit"],
        ):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(options)

    def test_hosted_backends_send_audio_and_reference_and_save_speech(self):
        for name, implementation, model, token_field, audio_count in (
            ("ming", MingAPIBackend, "Ming-3.0-audio-flash", "max_completion_tokens", 1),
            ("omni", QwenOmniAPIBackend, "Qwen/Qwen3-Omni-30B-A3B-Instruct", "max_tokens", 2),
        ):
            with self.subTest(backend=name), tempfile.TemporaryDirectory() as directory:
                args = parse_args(["--backend", name, "--output-dir", directory])
                backend = create_backend(args)
                self.assertIsInstance(backend, implementation)
                source = Path(directory) / "input.wav"
                source.write_bytes(recording())
                reply_text = "请在七天内申请。"
                context = "退货申请限七天。"
                backend.retrieve = Mock(return_value={"selected_metric": "euclidean", "euclidean": [
                    {"rank": 1, "score": 0.1, "passage": context},
                ]})
                first_pcm = b"\xe8\x03" * 120
                second_pcm = b"\x18\xfc" * 120
                audio = Mock()

                async def chunks():
                    if name == "ming":
                        first_event = speech_event("speech.audio.delta", audio=base64.b64encode(first_pcm).decode())
                        yield first_event[:13]
                        yield first_event[13:]
                        audio.assert_called_once_with(first_pcm)
                        yield speech_event("speech.audio.delta", audio=base64.b64encode(second_pcm).decode())
                        yield speech_event("speech.audio.done")
                    else:
                        yield first_pcm[:1]
                        audio.assert_not_called()
                        yield first_pcm[1:]
                        audio.assert_called_once_with(first_pcm)
                        yield second_pcm

                stream = SpeechStream(chunks())

                def respond(request):
                    self.assertEqual(request.headers["Authorization"], "Bearer test-key")
                    payload = json.loads(request.content)
                    if request.url.path.endswith("/chat/completions"):
                        self.assertEqual(payload["model"], model)
                        self.assertEqual(payload[token_field], 512)
                        messages = payload["messages"]
                        self.assertIn("上轮答复", json.dumps(messages, ensure_ascii=False))
                        parts = messages[-1]["content"]
                        self.assertIn(context, parts[1]["text"])
                        prefix, encoded = parts[0]["audio_url"]["url"].split(",", 1)
                        self.assertEqual(prefix, "data:audio/wav;base64")
                        self.assertEqual(base64.b64decode(encoded)[:4], b"RIFF")
                        self.assertEqual(json.dumps(messages).count("data:audio/wav;base64"), audio_count)
                        return httpx.Response(200, json={"choices": [{"message": {"content": reply_text}}]})
                    self.assertTrue(request.url.path.endswith("/audio/speech"))
                    self.assertEqual(payload["model"], backend.tts_model)
                    self.assertEqual(payload["input"], reply_text)
                    self.assertEqual(payload["sample_rate"], 24000)
                    if payload["response_format"] == "wav":
                        self.assertNotIn("stream", payload)
                        self.assertNotIn("stream_format", payload)
                        return httpx.Response(200, content=recording(rate=24000))
                    self.assertEqual(payload["response_format"], "pcm")
                    if name == "ming":
                        self.assertEqual(payload["stream_format"], "sse")
                        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)
                    self.assertIs(payload["stream"], True)
                    return httpx.Response(200, headers={"Content-Type": "application/octet-stream"}, stream=stream)

                client_class = httpx.AsyncClient
                history = [{"question_path": str(source), "text": "上轮答复"}]
                with patch.dict(os.environ, {backend.key_env: "test-key"}), \
                        patch("voice_service.api.httpx.AsyncClient", side_effect=lambda **options: client_class(
                            transport=httpx.MockTransport(respond), **options)):
                    _, wav, selected, _, turns = backend.answer(source, True, "测试资料", None, "zh", history,
                                                                 on_audio=audio)
                    greeting = backend.synthesize_greeting(reply_text, "zh")
                self.assertEqual(backend.speech_device.type, "cpu")
                self.assertEqual(selected, context)
                self.assertEqual(turns[-1]["text"], reply_text)
                self.assertTrue(stream.closed)
                self.assertEqual([call.args for call in audio.call_args_list], [(first_pcm,), (second_pcm,)])
                saved, rate = torchaudio.load(wav)
                self.assertEqual(rate, 24000)
                self.assertEqual(saved.mul(32768).numpy().astype("<i2").tobytes(), first_pcm + second_pcm)
                with wave.open(io.BytesIO(greeting), "rb") as saved:
                    self.assertEqual(saved.getframerate(), 24000)

    def test_hosted_backend_cancels_pending_request_before_tts(self):
        backend = create_backend(parse_args(["--backend", "ming"]))
        cancel_event = threading.Event()
        cancelled = threading.Event()

        async def respond(request):
            self.assertTrue(request.url.path.endswith("/chat/completions"))
            cancel_event.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.dict(os.environ, {backend.key_env: "test-key"}), \
                patch("voice_service.api.httpx.AsyncClient", return_value=client):
            backend.load()
            with self.assertRaises(InferenceCancelled):
                backend.infer([{"role": "user", "text": "你好"}], True, 64, cancel_event)
        self.assertTrue(cancelled.is_set())

    def test_minicpmo_sends_faq_and_float_audio_and_streams_speech(self):
        from websockets.legacy.server import serve

        answer = "请在七天内申请。"
        context = "## 问题：如何退货\n答案：" + answer
        retrieval = {"selected_metric": "cosine", "cosine": [{"rank": 1, "score": 1, "passage": context}]}
        chunks = (struct.pack("<2f", 0.25, -0.5), struct.pack("<2f", 0.5, -0.25))
        streamed = []
        first_audio = threading.Event()

        def on_audio(pcm):
            streamed.append(pcm)
            first_audio.set()

        async def mock_model(socket):
            async def send(event_type, **fields):
                await socket.send(json.dumps({"type": event_type, **fields}))

            self.assertEqual(socket.path, "/v1/realtime?mode=chat&model=MiniCPM-o-4.5-Realtime")
            self.assertEqual(socket.request_headers["Authorization"], "Bearer test-key")
            await send("session.queued", position=1)
            await send("session.queue_done")
            self.assertEqual(json.loads(await socket.recv()), {"type": "session.init", "payload": {}})
            await send("session.created")
            request = json.loads(await socket.recv())
            self.assertEqual(request["type"], "input.append")
            payload = request["input"]
            self.assertTrue(payload["streaming"])
            self.assertTrue(payload["tts"]["enabled"])
            self.assertEqual(payload["generation"]["max_new_tokens"], 512)
            history, prior_reply, question = payload["messages"]
            self.assertEqual(prior_reply, {"role": "assistant", "content": "上轮答复"})
            for message in (history, question):
                audio = message["content"][0]
                self.assertEqual(audio["type"], "audio")
                pcm = base64.b64decode(audio["data"], validate=True)
                self.assertEqual(len(pcm), 9600 * 4)
                self.assertAlmostEqual(struct.unpack_from("<f", pcm, 400)[0], 1000 / 32768, delta=3e-5)
            self.assertIn(context, question["content"][1]["text"])

            await send("response.output.delta", kind="text", text=answer[:3])
            for chunk in chunks:
                await send("response.output.delta", kind="audio", audio=base64.b64encode(chunk).decode("ascii"))
            self.assertTrue(await asyncio.to_thread(first_audio.wait, 3))
            await send("response.done", text=answer, reason="turn_end")
            self.assertEqual(json.loads(await socket.recv()), {"type": "session.close", "reason": "turn_done"})

        async def run(backend, source):
            async with serve(mock_model, "127.0.0.1", 0) as server:
                backend.base_url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
                history = [{"question_path": str(source), "text": "上轮答复"}]
                return await asyncio.wait_for(asyncio.to_thread(
                    backend.answer, source, True, "测试资料", None, "zh", history, on_audio=on_audio,
                ), timeout=10)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MAP_API_KEY": "test-key"}):
            backend = create_backend(parse_args(["--backend", "minicpmo", "--output-dir", directory]))
            self.assertIsInstance(backend, MiniCPMOAPIBackend)
            source = Path(directory) / "input.wav"
            source.write_bytes(recording(rate=24000))
            backend.retrieve = Mock(return_value=retrieval)
            _, wav, selected, _, turns = asyncio.run(run(backend, source))
            self.assertEqual(backend.speech_device.type, "cpu")
            self.assertEqual(selected, context)
            self.assertEqual(turns[-1]["text"], answer)
            self.assertEqual(turns[-1]["retrieval"], retrieval)
            self.assertEqual(streamed, [struct.pack("<2h", 8192, -16384), struct.pack("<2h", 16384, -8192)])
            saved, rate = torchaudio.load(wav)
            self.assertEqual(rate, 24000)
            torch.testing.assert_close(saved, torch.tensor([[0.25, -0.5, 0.5, -0.25]]), rtol=0, atol=1 / 32768)

    def test_hosted_backend_cancels_stalled_tts_after_first_chunk(self):
        backend = create_backend(parse_args(["--backend", "omni"]))
        cancel_event = threading.Event()
        cancelled = threading.Event()
        first_pcm = b"\xe8\x03" * 120
        audio = Mock()

        async def chunks():
            yield first_pcm
            audio.assert_called_once_with(first_pcm)
            cancel_event.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        stream = SpeechStream(chunks())
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
        with patch.dict(os.environ, {backend.key_env: "test-key"}), \
                patch("voice_service.api.httpx.AsyncClient", return_value=client):
            backend.load()
            with self.assertRaises(InferenceCancelled):
                backend.synthesize_stream("你好", cancel_event, audio, None)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(stream.closed)
        audio.assert_called_once_with(first_pcm)

    def test_hosted_backends_reject_incomplete_speech_streams(self):
        for name, body, message in (
            ("ming", speech_event("speech.audio.delta", audio="6AM="), "SSE 音频流提前断开"),
            ("omni", b"\xe8\x03\x01", "PCM 样本不完整"),
        ):
            with self.subTest(backend=name):
                backend = create_backend(parse_args(["--backend", name]))
                client = httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=body)))
                audio = Mock()
                with patch.dict(os.environ, {backend.key_env: "test-key"}), \
                        patch("voice_service.api.httpx.AsyncClient", return_value=client):
                    backend.load()
                    with self.assertRaisesRegex(RuntimeError, message):
                        backend.synthesize_stream("你好", None, audio, None)
                audio.assert_called_once_with(b"\xe8\x03")

    def test_integrated_models_skip_glm_decoder_preparation(self):
        with patch("huggingface_hub.snapshot_download") as download:
            for backend in ("omni-local", "ming", "omni"):
                prepare_decoder(parse_args(["--backend", backend, "--download-models"]))
        download.assert_not_called()

    def test_omni_loads_its_processor_and_full_generation_model(self):
        backend = QwenOmniBackend(parse_args(["--backend", "omni-local", "--attn-implementation", "sdpa"]))
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
                "--backend", "omni-local", "--output-dir", directory,
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
            backend = AudioModelBackend(parse_args(["--backend", "omni-local", "--output-dir", directory]))
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

    def test_turn_judgment_uses_native_text_generation_and_removes_probe_file(self):
        backend = AudioModelBackend(parse_args(["--backend", "omni-local"]))
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

        backend = QwenOmniBackend(parse_args(["--backend", "omni-local"]))
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
