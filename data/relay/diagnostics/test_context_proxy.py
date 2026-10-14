"""Lightweight HTTP integration tests for context_proxy, using a fake vLLM."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import context_proxy as cp


class FakeVLLM:
    def __init__(self):
        self.token_count = 60_000
        self.max_model_len = 65_536
        self.tokenize_status = 200
        self.chat_status = 200
        self.usage_prompt_tokens = 60_000
        self.chat_response: dict[str, Any] = {
            "id": "chatcmpl-fake",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": self.usage_prompt_tokens, "completion_tokens": 1, "total_tokens": self.usage_prompt_tokens + 1},
        }
        self.tokenize_requests: list[dict[str, Any]] = []
        self.chat_requests: list[dict[str, Any]] = []
        self.authorization: list[str | None] = []
        self.app = web.Application()
        self.app.router.add_get("/models", self.models)
        self.app.router.add_get("/v1/models", self.models)
        self.app.router.add_post("/tokenize", self.tokenize)
        self.app.router.add_post("/v1/chat/completions", self.chat)

    async def models(self, request: web.Request) -> web.Response:
        self.authorization.append(request.headers.get("Authorization"))
        return web.json_response({"object": "list", "data": [{"id": "fake-model"}]})

    async def tokenize(self, request: web.Request) -> web.Response:
        self.authorization.append(request.headers.get("Authorization"))
        self.tokenize_requests.append(await request.json())
        if self.tokenize_status != 200:
            return web.json_response({"error": {"message": "fake tokenize failure"}}, status=self.tokenize_status)
        return web.json_response({"count": self.token_count, "max_model_len": self.max_model_len, "tokens": []})

    async def chat(self, request: web.Request) -> web.Response:
        self.authorization.append(request.headers.get("Authorization"))
        self.chat_requests.append(await request.json())
        response = json.loads(json.dumps(self.chat_response))
        usage = response.get("usage")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = self.usage_prompt_tokens
            usage["total_tokens"] = self.usage_prompt_tokens + usage.get("completion_tokens", 0)
        return web.json_response(response, status=self.chat_status)


class ContextProxyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "proxy.jsonl"
        self.fake = FakeVLLM()
        self.upstream = TestServer(self.fake.app)
        await self.upstream.start_server()
        self.upstream_url = str(self.upstream.make_url("/")).rstrip("/")
        self.proxy_client: TestClient | None = None

    async def asyncTearDown(self) -> None:
        if self.proxy_client is not None:
            await self.proxy_client.close()
        await self.upstream.close()
        self.temp_dir.cleanup()

    async def start_proxy(self, mode: str) -> TestClient:
        config = cp.ProxyConfig(
            upstream=self.upstream_url,
            mode=mode,
            log_path=str(self.log_path),
            context_length=65_536,
        )
        self.proxy_client = TestClient(TestServer(cp.ContextProxy(config).make_app()))
        await self.proxy_client.start_server()
        return self.proxy_client

    def read_log(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    @staticmethod
    def payload() -> dict[str, Any]:
        return {
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Run this command."},
                {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"echo hi\"}"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "hi"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}],
            "tool_choice": "auto",
            "chat_template": "{{ messages }}",
            "chat_template_kwargs": {"enable_thinking": False, "custom": {"keep": True}},
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
            "media_io_kwargs": {"image": {"max_pixels": 4}},
            "mm_processor_kwargs": {"crop": False},
            "max_tokens": 16_384,
            "temperature": 0.0,
        }

    async def test_adaptive_caps_and_tokenizes_matching_fields(self) -> None:
        client = await self.start_proxy("adaptive")
        original = self.payload()
        response = await client.post(
            "/v1/chat/completions",
            json=original,
            headers={"Authorization": "Bearer test-secret", "X-Request-ID": "request-7"},
        )
        self.assertEqual(response.status, 200)
        forwarded = self.fake.chat_requests[-1]
        expected_forwarded = dict(original, max_tokens=5_536)
        self.assertEqual(forwarded, expected_forwarded)

        expected_tokenize = {key: original[key] for key in cp.TOKENIZE_FIELDS if key in original}
        self.assertEqual(self.fake.tokenize_requests[-1], expected_tokenize)
        self.assertEqual(self.fake.authorization, ["Bearer test-secret", "Bearer test-secret"])

        record = self.read_log()[-1]
        self.assertEqual(record["prompt_tokens"], 60_000)
        self.assertEqual(record["tokenize_max_model_len"], 65_536)
        self.assertEqual(record["max_tokens_requested"], 16_384)
        self.assertEqual(record["max_tokens_forwarded"], 5_536)
        self.assertEqual(record["usage_prompt_tokens"], 60_000)
        self.assertEqual(record["decision"], "capped")
        self.assertNotIn("test-secret", self.log_path.read_text())
        self.assertNotIn("Run this command", self.log_path.read_text())

    async def test_fixed_mode_preserves_payload_even_near_context_limit(self) -> None:
        self.fake.token_count = 65_000
        self.fake.usage_prompt_tokens = 65_000
        client = await self.start_proxy("fixed")
        original = self.payload()
        response = await client.post("/v1/chat/completions", json=original)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.fake.chat_requests[-1], original)
        record = self.read_log()[-1]
        self.assertEqual(record["max_tokens_forwarded"], 16_384)
        self.assertEqual(record["decision"], "forwarded")

    async def test_adaptive_rejects_when_prompt_leaves_no_room(self) -> None:
        self.fake.token_count = 65_536
        client = await self.start_proxy("adaptive")
        response = await client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status, 400)
        self.assertEqual(self.fake.chat_requests, [])
        self.assertEqual(self.read_log()[-1]["decision"], "no_context_room")

    async def test_adaptive_fails_closed_when_tokenizer_errors(self) -> None:
        self.fake.tokenize_status = 503
        client = await self.start_proxy("adaptive")
        response = await client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status, 502)
        self.assertEqual(self.fake.chat_requests, [])
        self.assertEqual(self.read_log()[-1]["decision"], "tokenize_failed")

    async def test_fixed_mode_forwards_upstream_chat_error(self) -> None:
        self.fake.chat_status = 503
        self.fake.chat_response = {"error": {"message": "upstream unavailable"}}
        client = await self.start_proxy("fixed")
        response = await client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status, 503)
        self.assertEqual(await response.json(), self.fake.chat_response)
        self.assertEqual(self.read_log()[-1]["upstream_status"], 503)

    async def test_successful_response_with_prompt_count_mismatch_is_fatal(self) -> None:
        self.fake.token_count = 60_000
        self.fake.usage_prompt_tokens = 59_999
        client = await self.start_proxy("fixed")
        response = await client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status, 502)
        error = await response.json()
        self.assertEqual(error["error"]["type"], "context_proxy_parity_error")
        self.assertEqual(error["error"]["tokenize_prompt_tokens"], 60_000)
        self.assertEqual(error["error"]["usage_prompt_tokens"], 59_999)
        record = self.read_log()[-1]
        self.assertEqual(record["decision"], "fatal_usage_prompt_mismatch")
        self.assertTrue(record["fatal_mismatch"])

    async def test_fixed_mode_continues_if_internal_tokenize_fails(self) -> None:
        self.fake.tokenize_status = 503
        client = await self.start_proxy("fixed")
        response = await client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status, 200)
        self.assertEqual(len(self.fake.chat_requests), 1)
        self.assertEqual(self.read_log()[-1]["tokenize_status"], 503)

    async def test_streaming_is_rejected_before_upstream_calls(self) -> None:
        client = await self.start_proxy("adaptive")
        payload = self.payload()
        payload["stream"] = True
        response = await client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status, 400)
        self.assertEqual(self.fake.tokenize_requests, [])
        self.assertEqual(self.fake.chat_requests, [])

    async def test_models_route_is_forwarded(self) -> None:
        client = await self.start_proxy("fixed")
        response = await client.get("/models", headers={"Authorization": "Bearer models-secret"})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["data"][0]["id"], "fake-model")
        self.assertEqual(self.fake.authorization[-1], "Bearer models-secret")
        self.assertFalse(self.log_path.exists())

    async def test_v1_models_route_is_forwarded(self) -> None:
        client = await self.start_proxy("fixed")
        response = await client.get("/v1/models")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["data"][0]["id"], "fake-model")


if __name__ == "__main__":
    unittest.main()
