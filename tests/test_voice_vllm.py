"""Test raw-token streaming and the vLLM call entry without model weights."""
import asyncio
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_service.backend import InferenceCancelled
from voice_call_server import create_app, parse_args
from voice_service.vllm import VllmTokenGenerator


BASE_URL = "http://127.0.0.1:18000/v1"


class TokenStream(httpx.AsyncByteStream):
    def __init__(self, stall=False):
        self.stall = stall
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"prompt_token_ids":[1,2,3],"token_ids":[10,11]}]}\n\n'
        if self.stall:
            await asyncio.Event().wait()
        yield b'data: {"choices":[{"token_ids":[12],"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'

    async def aclose(self):
        self.closed = True


class VllmTests(unittest.TestCase):
    def test_completion_stream_sends_and_receives_raw_ids(self):
        stream = TokenStream()
        batches = []

        def serve(request):
            self.assertEqual(request.url.path, "/v1/completions")
            self.assertEqual(json.loads(request.content), {
                "model": "glm-voice", "prompt": [1, 2, 3], "max_tokens": 3,
                "stream": True, "return_token_ids": True, "add_special_tokens": False,
                "temperature": 0.2,
            })
            return httpx.Response(200, stream=stream)

        client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
        generator = VllmTokenGenerator(BASE_URL, "glm-voice")
        with patch("voice_service.vllm.httpx.AsyncClient", return_value=client):
            self.assertEqual(generator.generate([1, 2, 3], 3, on_tokens=batches.append, temperature=0.2),
                             [10, 11, 12])
        self.assertEqual(batches, [[10, 11], [12]])
        self.assertTrue(stream.closed)

    def test_cancellation_closes_an_idle_stream(self):
        stream = TokenStream(stall=True)
        event = threading.Event()
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
        )
        generator = VllmTokenGenerator(BASE_URL, "glm-voice")
        with patch("voice_service.vllm.httpx.AsyncClient", return_value=client), self.assertRaises(InferenceCancelled):
            generator.generate([1, 2, 3], 3, cancel_event=event, on_tokens=lambda ids: event.set())
        self.assertTrue(stream.closed)

    def test_nested_vad_entry_only_serves_the_vllm_backend(self):
        for args, expected_status in (
            (parse_args([]), 404),
            (parse_args(["--vllm-base-url", BASE_URL]), 200),
        ):
            with self.subTest(vllm=args.vllm_base_url):
                client = TestClient(create_app(args, {}, backend=Mock()))
                self.addCleanup(client.close)
                self.assertEqual(client.get("/omni_vad").status_code, 200)
                self.assertEqual(client.get("/omni_vad/vllm").status_code, expected_status)
                if expected_status == 200:
                    self.assertEqual(client.get("/omni_vad/style.css").status_code, 200)


if __name__ == "__main__":
    unittest.main()
