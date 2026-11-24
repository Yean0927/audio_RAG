"""Stream raw GLM token IDs from a separate vLLM completion server."""
from __future__ import annotations

import asyncio
import json

import httpx

from .backend import check_cancelled


class VllmTokenGenerator:
    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def model_context_length(self):
        return self.model_card().get("max_model_len")

    def model_card(self):
        try:
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get(f"{self.base_url}/models")
                response.raise_for_status()
                cards = response.json()["data"]
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise RuntimeError(f"无法读取 vLLM 模型列表：{self.base_url}/models（{error}）") from error
        for card in cards:
            if card.get("id") == self.model:
                return card
        raise RuntimeError(f"vLLM 服务未提供模型 {self.model}，请检查 --vllm-model。")

    def generate(self, prompt_ids, max_tokens, *, cancel_event=None, stop_stream=None,
                 on_tokens=None, **sampling):
        check_cancelled(cancel_event)
        check_cancelled(stop_stream)
        request = {
            "model": self.model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "stream": True, "return_token_ids": True, "add_special_tokens": False,
            **sampling,
        }
        try:
            return asyncio.run(self._generate(request, cancel_event, stop_stream, on_tokens))
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"vLLM 推理请求失败（HTTP {error.response.status_code}）") from error
        except (httpx.RequestError, ValueError) as error:
            raise RuntimeError(f"vLLM 流式推理失败：{error}") from error

    async def _generate(self, request, cancel_event, stop_stream, on_tokens):
        async def watch_cancellation():
            while True:
                check_cancelled(cancel_event)
                check_cancelled(stop_stream)
                await asyncio.sleep(0.05)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False) as client:
            generation = asyncio.create_task(self._read_tokens(client, request, on_tokens))
            cancellation = asyncio.create_task(watch_cancellation())
            try:
                await asyncio.wait((generation, cancellation), return_when=asyncio.FIRST_COMPLETED)
                if cancellation.done():
                    cancellation.result()
                check_cancelled(cancel_event)
                check_cancelled(stop_stream)
                return generation.result()
            finally:
                generation.cancel()
                cancellation.cancel()
                await asyncio.gather(generation, cancellation, return_exceptions=True)

    async def _read_tokens(self, client, request, on_tokens):
        generated = []
        async with client.stream("POST", f"{self.base_url}/completions", json=request) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return generated
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"vLLM 推理失败：{chunk['error']}")
                for choice in chunk.get("choices", ()):
                    ids = choice.get("token_ids")
                    if ids:
                        generated.extend(ids)
                        if on_tokens is not None:
                            on_tokens(ids)
        raise RuntimeError("vLLM 的 token 流提前断开。")
