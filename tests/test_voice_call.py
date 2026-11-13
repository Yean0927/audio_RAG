"""CPU-only protocol tests: no CUDA, network downloads, or model weights."""
import base64
import io
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
import wave
import os
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from voice_call_server import create_app, parse_args, validate_wav, load_config, prepare_decoder
from voice_web_demo import InferenceCancelled, check_cancelled, make_prompt


def recording(seconds=0.6, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<h", 1000) * int(seconds * rate))
    return output.getvalue()


class FakeBackend:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.block_first = False
        self.calls = []
        self.inspected = []
        self.block_sizes = None
        self.greetings = []
        self.knowledge_prepared = []

    def load(self):
        pass

    def prepare_knowledge(self, content, language):
        self.knowledge_prepared.append((content, language))

    def synthesize_greeting(self, greeting, language):
        self.greetings.append((greeting, language))
        return recording(rate=22050)

    def answer(self, source, use_rag, knowledge, knowledge_file, language, history, cancel_event=None,
               on_audio=None, profile=None, stream_block_sizes=None):
        self.calls.append((len(history), use_rag, knowledge))
        self.block_sizes = stream_block_sizes
        if profile:
            profile("chunk_pcm_ready", chunk=1, packet=1)
        if on_audio is not None:
            on_audio(b"\x01\x00")
        if self.block_first and len(self.calls) == 1:
            self.started.set()
            if not cancel_event.wait(4):
                raise RuntimeError("Test did not interrupt the first request")
            self.cancelled.set()
            raise InferenceCancelled()
        check_cancelled(cancel_event)
        if profile:
            profile("chunk_pcm_ready", chunk=2, packet=2)
        if on_audio is not None:
            on_audio(b"\x02\x00")
        path = self.folder / "reply.wav"
        path.write_bytes(recording())
        record = {"text": f"客服回复 {len(history) + 1}",
                  "question_tokens": "<|begin_of_audio|><|audio_42|><|audio_71|><|end_of_audio|>",
                  "assistant_tokens": "reply"}
        return [], str(path), knowledge, "done", list(history) + [record]

    def decode_input_wav(self, tokens):
        self.inspected.append(tokens)
        return recording(rate=22050)


class CallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = parse_args(["--output-dir", self.temp.name])
        self.backend = FakeBackend(self.temp.name)
        self.config = {"title": "测试客服", "greeting": "您好", "language": "zh", "system_prompt": "客服设定",
                       "knowledge_text": "售后服务时间为九点到十八点", "silence_ms": 850, "min_speech_ms": 260}

    def client(self):
        return TestClient(create_app(self.args, self.config, self.backend))

    def wait_ready(self, client):
        for _ in range(100):
            if client.get("/api/health").json()["state"] == "ready":
                return
            time.sleep(0.01)
        self.fail("Backend did not become ready")

    def send_turn(self, socket, turn):
        socket.send_json({"type": "utterance", "id": turn})
        socket.send_bytes(recording())

    def reply(self, socket):
        for _ in range(8):
            message = socket.receive_json()
            if message["type"] == "reply":
                return message
            self.assertNotEqual(message["type"], "error", message)
        self.fail("No reply")

    def test_model_greeting_is_prepared_once_and_cached_across_restarts(self):
        def connect():
            with self.client() as client:
                self.wait_ready(client)
                with client.websocket_connect("/api/call?stream=1") as socket:
                    hello = socket.receive_json()
                    self.assertEqual(hello["greeting"], self.config["greeting"])
                    self.assertEqual(base64.b64decode(hello["greeting_audio"]), recording(rate=22050))
                    socket.send_json({"type": "hangup"})

        connect()
        connect()
        self.assertEqual(self.backend.greetings, [("您好", "zh")])
        directory = Path(self.temp.name) / "greetings"
        path, = directory.glob("*.wav")
        self.assertEqual(path.read_bytes(), recording(rate=22050))
        path.write_bytes(b"truncated")
        connect()
        self.assertEqual(self.backend.greetings, [("您好", "zh"), ("您好", "zh")])
        self.config["greeting"] = "欢迎来电"
        connect()
        self.assertEqual(self.backend.greetings[-1], ("欢迎来电", "zh"))
        self.assertEqual(len(list(directory.iterdir())), 2)

    def test_auto_turns_history_knowledge_and_fresh_call(self):
        with self.client() as client:
            self.wait_ready(client)
            self.assertEqual(client.get("/").status_code, 200)
            for name in ("call.js", "style.css", "capture-worklet.js", "audio-engine.mjs"):
                self.assertEqual(client.get("/assets/" + name).status_code, 200)
            with client.websocket_connect("/api/call") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                self.assertFalse(client.get("/api/health").json()["available"])
                self.send_turn(socket, 1)
                first = self.reply(socket)
                self.assertEqual(first["text"], "客服回复 1")
                self.assertTrue(first["audio"])
                self.assertEqual(first["input_tokens"], [42, 71])
                self.assertEqual(self.backend.inspected, [])
                socket.send_json({"type": "inspect_input", "id": 1})
                preview = socket.receive_json()
                self.assertEqual((preview["type"], preview["id"]), ("input_audio", 1))
                self.assertEqual(base64.b64decode(preview["audio"]), recording(rate=22050))
                self.assertEqual(self.backend.inspected, [[42, 71]])
                self.send_turn(socket, 2)
                self.assertEqual(self.reply(socket)["text"], "客服回复 2")
                socket.send_json({"type": "hangup"})
            self.assertEqual(self.backend.calls, [(0, True, self.config["knowledge_text"]), (1, True, self.config["knowledge_text"])])
            with client.websocket_connect("/api/call") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                self.send_turn(socket, 1)
                self.assertEqual(self.reply(socket)["text"], "客服回复 1")

    def test_interrupt_cancels_and_discards_old_reply(self):
        self.backend.block_first = True
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call") as socket:
                socket.receive_json()
                self.send_turn(socket, 1)
                self.assertEqual(socket.receive_json(), {"type": "thinking", "id": 1})
                self.assertTrue(self.backend.started.wait(2))
                socket.send_json({"type": "interrupt"})
                self.send_turn(socket, 2)
                reply = self.reply(socket)
                self.assertEqual(reply["id"], 2)
                self.assertTrue(self.backend.cancelled.wait(1))
                self.assertEqual(self.backend.calls[-1][0], 0)

    def test_stream_plays_before_completion_and_interrupt_discards_remaining_audio(self):
        self.backend.block_first = True
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call?stream=1") as socket:
                self.assertFalse(socket.receive_json()["auto_listen"])
                self.send_turn(socket, 1)
                self.assertEqual(socket.receive_json(), {"type": "thinking", "id": 1})
                first = socket.receive_json()
                self.assertEqual((first["type"], first["id"]), ("audio_chunk", 1))
                self.assertEqual(base64.b64decode(first["pcm"]), b"\x01\x00")
                self.assertTrue(self.backend.started.wait(2))
                socket.send_json({"type": "interrupt"})
                self.send_turn(socket, 2)
                self.assertEqual(socket.receive_json(), {"type": "thinking", "id": 2})
                for pcm in (b"\x01\x00", b"\x02\x00"):
                    chunk = socket.receive_json()
                    self.assertEqual((chunk["type"], chunk["id"]), ("audio_chunk", 2))
                    self.assertEqual(base64.b64decode(chunk["pcm"]), pcm)
                reply = socket.receive_json()
                self.assertEqual((reply["type"], reply["id"], reply["audio"]), ("reply", 2, None))
                self.assertNotIn("latency_profile", reply)
                self.assertEqual(reply["input_tokens"], [42, 71])
                self.assertTrue(self.backend.cancelled.wait(1))
                self.assertEqual(self.backend.calls[-1][0], 0)

    def test_latency_profile_tracks_each_audio_packet(self):
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call?stream=1&profile=1") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                self.send_turn(socket, 1)
                reply = self.reply(socket)
        events = reply["latency_profile"]["events"]
        self.assertEqual(events[0]["stage"], "wav_saved")
        self.assertEqual(events[-1]["stage"], "reply_ready")
        for packet in (1, 2):
            self.assertEqual([event["stage"] for event in events if event.get("packet") == packet],
                             ["chunk_pcm_ready", "chunk_dequeued", "chunk_send_start", "chunk_send_end"])

    def test_block_size_override_is_scoped_to_profiled_calls(self):
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call?stream=1&profile=1&block_sizes=25,32,100") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                self.send_turn(socket, 1)
                self.assertEqual(self.reply(socket)["latency_profile"]["block_sizes"], [25, 32, 100])
                self.assertEqual(self.backend.block_sizes, (25, 32, 100))
            with client.websocket_connect("/api/call?stream=1&block_sizes=25,32,100") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                self.send_turn(socket, 1)
                self.assertNotIn("latency_profile", self.reply(socket))
                self.assertIsNone(self.backend.block_sizes)

    def test_hangup_cancels_worker_and_releases_call(self):
        self.backend.block_first = True
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call") as socket:
                socket.receive_json()
                self.send_turn(socket, 1)
                self.assertEqual(socket.receive_json()["type"], "thinking")
                self.assertTrue(self.backend.started.wait(2))
                socket.send_json({"type": "hangup"})
            self.assertTrue(self.backend.cancelled.wait(2))
            self.assertTrue(client.get("/api/health").json()["available"])

    def test_one_call_at_a_time_and_origin_rejection(self):
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call") as first:
                first.receive_json()
                with client.websocket_connect("/api/call") as second:
                    self.assertEqual(second.receive_json()["type"], "error")
            from starlette.websockets import WebSocketDisconnect
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/api/call", headers={"origin": "https://unrelated.example"}):
                    pass

    def test_two_replicas_infer_concurrently_and_release_independently(self):
        args = parse_args(["--output-dir", self.temp.name, "--replica-gpus", "0,1"])
        first_backend, second_backend = self.backend, FakeBackend(self.temp.name)
        first_backend.block_first = second_backend.block_first = True
        with patch("voice_call_server.build_backends", return_value=[first_backend, second_backend]):
            app = create_app(args, self.config)
        with TestClient(app) as client:
            self.wait_ready(client)
            self.assertEqual(client.get("/api/health").json()["max_calls"], 2)
            for worker in (first_backend, second_backend):
                self.assertEqual(worker.knowledge_prepared, [(self.config["knowledge_text"], "zh")])
            with client.websocket_connect("/api/call") as first:
                self.assertEqual(first.receive_json()["type"], "ready")
                self.send_turn(first, 1)
                self.assertEqual(first.receive_json()["type"], "thinking")
                self.assertTrue(first_backend.started.wait(2))
                with client.websocket_connect("/api/call") as second:
                    self.assertEqual(second.receive_json()["type"], "ready")
                    self.send_turn(second, 1)
                    self.assertEqual(second.receive_json()["type"], "thinking")
                    self.assertTrue(second_backend.started.wait(2))
                    self.assertEqual(client.get("/api/health").json()["available_slots"], 0)
                    with client.websocket_connect("/api/call") as third:
                        self.assertEqual(third.receive_json()["type"], "error")
                    second.send_json({"type": "hangup"})
                self.assertTrue(second_backend.cancelled.wait(2))
                self.assertFalse(first_backend.cancelled.is_set())
                with client.websocket_connect("/api/call") as replacement:
                    self.assertEqual(replacement.receive_json()["type"], "ready")
                    self.send_turn(replacement, 1)
                    self.assertEqual(self.reply(replacement)["text"], "客服回复 1")
                self.assertEqual(len(first_backend.calls), 1)
                self.assertEqual(len(second_backend.calls), 2)
                first.send_json({"type": "hangup"})

    def test_reject_audio_without_metadata(self):
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call") as socket:
                socket.receive_json()
                socket.send_bytes(recording())
                self.assertEqual(socket.receive_json()["type"], "error")

    def test_pcm_input_waits_for_end_and_saves_the_original_samples(self):
        pcm = struct.pack("<9600h", *range(9600))
        incoming = Path(self.temp.name) / "incoming"
        with self.client() as client:
            self.wait_ready(client)
            with client.websocket_connect("/api/call?stream=1") as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_json({"type": "utterance_start", "id": 1})
                socket.send_bytes(pcm[:1024])
                socket.send_json({"type": "utterance_cancel", "id": 1})
                socket.send_json({"type": "utterance_start", "id": 2})
                socket.send_bytes(pcm[:2560])
                socket.send_bytes(pcm[2560:5120])
                socket.send_json({"type": "ping"})
                self.assertEqual(socket.receive_json(), {"type": "pong"})
                self.assertFalse(self.backend.calls)
                self.assertFalse(list(incoming.rglob("*.wav")))

                socket.send_bytes(pcm[5120:])
                socket.send_json({"type": "utterance_end", "id": 2})
                self.assertEqual(socket.receive_json(), {"type": "thinking", "id": 2})
                self.assertEqual(self.reply(socket)["id"], 2)
                path, = incoming.rglob("*.wav")
                self.assertEqual(path.name, "2.wav")
                with wave.open(str(path), "rb") as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                    self.assertEqual(audio.readframes(audio.getnframes()), pcm)

                socket.send_json({"type": "utterance_start", "id": 3})
                socket.send_bytes(pcm[:1024])
                socket.send_json({"type": "hangup"})
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(list(incoming.rglob("*.wav")), [path])

    def test_pcm_input_rejects_bad_alignment_wrong_turn_and_oversize(self):
        self.args.max_audio_seconds = 1
        invalid = (
            ([b"\x00"], {"type": "utterance_end", "id": 1}),
            ([b"\x00\x00"], {"type": "utterance_end", "id": 2}),
            ([bytes(16000), bytes(16002)], {"type": "utterance_end", "id": 1}),
        )
        with self.client() as client:
            self.wait_ready(client)
            for chunks, end in invalid:
                with self.subTest(chunks=list(map(len, chunks)), end=end):
                    with client.websocket_connect("/api/call") as socket:
                        self.assertEqual(socket.receive_json()["type"], "ready")
                        socket.send_json({"type": "utterance_start", "id": 1})
                        for pcm in chunks:
                            socket.send_bytes(pcm)
                        socket.send_json(end)
                        self.assertEqual(socket.receive_json()["type"], "error")
        self.assertFalse(self.backend.calls)
        self.assertFalse(list((Path(self.temp.name) / "incoming").rglob("*.wav")))

    def test_wav_validation_and_customer_prompt(self):
        validate_wav(recording(), 30)
        for value in (b"not-wav", recording(1, 8000), recording(1)[:-20]):
            with self.assertRaises(ValueError):
                validate_wav(value, 30)
        with self.assertRaises(ValueError):
            validate_wav(recording(2), 1)
        prompt = make_prompt([], "question", "reference", "zh", "门店客服设定")
        self.assertIn("门店客服设定", prompt)
        self.assertIn("reference", prompt)
        self.assertIn("question", prompt)

    def test_missing_decoder_fails_before_model_loading(self):
        self.args.flow_path = self.temp.name
        with self.assertRaisesRegex(RuntimeError, "Decoder"):
            prepare_decoder(self.args)

    def test_download_flag_disables_offline_for_process_and_checks_files(self):
        self.args.flow_path = self.temp.name
        self.args.download_models = True

        def download(repo, local_dir, allow_patterns, max_workers):
            self.assertEqual(repo, "zai-org/glm-4-voice-decoder")
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                self.assertEqual(os.environ[key], "0")
            for name in allow_patterns:
                (Path(local_dir) / name).write_bytes(b"test weight bytes " * 10)

        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), patch.dict(
            sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}
        ):
            prepare_decoder(self.args)
        self.assertTrue((Path(self.temp.name) / "flow.pt").is_file())


if __name__ == "__main__":
    unittest.main()
