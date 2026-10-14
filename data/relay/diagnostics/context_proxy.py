#!/usr/bin/env python3
"""Local diagnostic proxy with an optional prompt-aware max-token cap.

The proxy tokenizes the exact chat messages, tools, and template-related fields
with the upstream vLLM ``/tokenize`` endpoint. In ``fixed`` mode it forwards the
chat request unchanged. In ``adaptive`` mode it sets the requested output limit
to ``min(requested, context_length - prompt_tokens)`` and rejects requests with
no remaining context. Streaming is rejected because this diagnostic proxy needs
the complete response to record usage.

Example:
    python context_proxy.py --upstream http://127.0.0.1:8000 --port 18124 \
        --mode adaptive --log /tmp/context-proxy.jsonl

Only request counts, limits, statuses, and returned usage are logged. Request
content and headers are never written to the JSONL log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

# These are the fields accepted by vLLM's TokenizeChatRequest. Keeping this
# allowlist in sync with that protocol lets /tokenize render the same chat
# template, tool definitions, and multimodal/template options as the completion.
TOKENIZE_FIELDS = (
    "model",
    "messages",
    "add_generation_prompt",
    "return_token_strs",
    "continue_final_message",
    "add_special_tokens",
    "chat_template",
    "chat_template_kwargs",
    "media_io_kwargs",
    "mm_processor_kwargs",
    "tools",
)
UPSTREAM_SESSION = web.AppKey("upstream_session", ClientSession)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass(frozen=True)
class ProxyConfig:
    upstream: str
    mode: str
    log_path: str
    context_length: int = 65_536
    timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "adaptive"}:
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        parsed = urlsplit(self.upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("upstream must be an http(s) URL")
        if parsed.path not in {"", "/", "/v1"}:
            raise ValueError("upstream must be a root URL, optionally ending in /v1")

    @property
    def upstream_root(self) -> str:
        return self.upstream.rstrip("/").removesuffix("/v1")


class ContextProxy:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self._log_lock = asyncio.Lock()

    async def startup(self, app: web.Application) -> None:
        app[UPSTREAM_SESSION] = ClientSession(
            connector=TCPConnector(limit=128),
            timeout=ClientTimeout(total=self.config.timeout_seconds),
        )

    async def cleanup(self, app: web.Application) -> None:
        await app[UPSTREAM_SESSION].close()

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.on_startup.append(self.startup)
        app.on_cleanup.append(self.cleanup)
        app.router.add_get("/models", self.handle_models)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        return app

    @staticmethod
    def _forward_headers(headers: Any) -> dict[str, str]:
        # Preserve end-to-end headers, including Authorization/api-key, without
        # printing them. aiohttp supplies Host and content framing itself.
        connection_tokens = {
            token.strip().lower()
            for value in headers.getall("Connection", [])
            for token in value.split(",")
            if token.strip()
        }
        blocked = HOP_BY_HOP_HEADERS | connection_tokens
        return {key: value for key, value in headers.items() if key.lower() not in blocked}

    @staticmethod
    def _response_headers(headers: Any) -> dict[str, str]:
        # The client library recalculates message framing and content encoding.
        blocked = HOP_BY_HOP_HEADERS | {"content-encoding"}
        return {key: value for key, value in headers.items() if key.lower() not in blocked}

    async def _request_upstream(
        self,
        request: web.Request,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        url = self.config.upstream_root + path
        async with request.app[UPSTREAM_SESSION].request(
            request.method if path in {"/models", "/v1/models"} else "POST",
            url,
            params=request.query,
            data=body,
            headers=headers if headers is not None else self._forward_headers(request.headers),
        ) as response:
            response_body = await response.read()
            return response.status, self._response_headers(response.headers), response_body

    async def handle_models(self, request: web.Request) -> web.Response:
        status, headers, body = await self._request_upstream(request, request.path)
        return web.Response(status=status, headers=headers, body=body)

    @staticmethod
    def _requested_output_limit(payload: dict[str, Any]) -> tuple[str | None, int | None]:
        present = [key for key in ("max_tokens", "max_completion_tokens") if key in payload]
        if len(present) > 1:
            raise web.HTTPBadRequest(text="Set only one of max_tokens or max_completion_tokens")
        if not present:
            return None, None
        key = present[0]
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise web.HTTPBadRequest(text=f"{key} must be a positive integer in adaptive mode")
        return key, value

    async def _write_log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        async with self._log_lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        path = Path(self.config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)

    async def handle_chat(self, request: web.Request) -> web.Response:
        if request.content_type != "application/json":
            raise web.HTTPBadRequest(text="Expected application/json")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text="Invalid JSON request body") from None
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="Expected a JSON object")
        if payload.get("stream"):
            raise web.HTTPBadRequest(text="Streaming is not supported by context_proxy")

        if self.config.mode == "adaptive":
            limit_field, requested_limit = self._requested_output_limit(payload)
        else:
            fields = [key for key in ("max_tokens", "max_completion_tokens") if key in payload]
            limit_field = fields[0] if len(fields) == 1 else None
            raw_limit = payload.get(limit_field) if limit_field else None
            requested_limit = (
                raw_limit
                if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) and raw_limit > 0
                else None
            )

        record: dict[str, Any] = {
            "time": round(time.time(), 3),
            "mode": self.config.mode,
            "model": payload.get("model"),
            "context_length": self.config.context_length,
            "max_tokens_field": limit_field,
            "max_tokens_requested": requested_limit,
            "max_tokens_forwarded": requested_limit,
            "prompt_tokens": None,
            "tokenize_max_model_len": None,
            "tokenize_status": None,
            "usage_prompt_tokens": None,
            "upstream_status": None,
            "decision": "forwarded",
        }

        tokenize_payload = {key: payload[key] for key in TOKENIZE_FIELDS if key in payload}
        token_status: int | None = None
        prompt_tokens: int | None = None
        tokenize_max_model_len: int | None = None
        try:
            token_status, _, token_body = await self._request_upstream(
                request,
                "/tokenize",
                body=json.dumps(tokenize_payload, separators=(",", ":")).encode("utf-8"),
                headers={
                    **self._forward_headers(request.headers),
                    "Content-Type": "application/json",
                },
            )
            record["tokenize_status"] = token_status
            if 200 <= token_status < 300:
                token_result = json.loads(token_body)
                if isinstance(token_result, dict):
                    count = token_result.get("count")
                    max_model_len = token_result.get("max_model_len")
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        prompt_tokens = count
                    if isinstance(max_model_len, int) and not isinstance(max_model_len, bool):
                        tokenize_max_model_len = max_model_len
        except (json.JSONDecodeError, UnicodeDecodeError, ClientError, asyncio.TimeoutError):
            # Fixed mode is a pass-through diagnostic; adaptive mode fails closed
            # because it cannot establish a safe output budget without a count.
            record["tokenize_status"] = "error"

        record["prompt_tokens"] = prompt_tokens
        record["tokenize_max_model_len"] = tokenize_max_model_len

        if self.config.mode == "adaptive":
            if limit_field is None or requested_limit is None:
                record["decision"] = "missing_output_limit"
                await self._write_log(record)
                raise web.HTTPBadRequest(text="Adaptive mode requires max_tokens or max_completion_tokens")
            if token_status is None or not (200 <= token_status < 300) or prompt_tokens is None:
                record["decision"] = "tokenize_failed"
                await self._write_log(record)
                return web.json_response(
                    {"error": {"message": "Upstream /tokenize failed; adaptive request was not sent", "type": "context_proxy_error"}},
                    status=502,
                )
            remaining = self.config.context_length - prompt_tokens
            if remaining <= 0:
                record["decision"] = "no_context_room"
                record["max_tokens_forwarded"] = 0
                await self._write_log(record)
                return web.json_response(
                    {"error": {"message": "Prompt leaves no room for a completion", "type": "context_length_exceeded"}},
                    status=400,
                )
            forwarded_limit = min(requested_limit, remaining)
            payload = dict(payload)
            payload[limit_field] = forwarded_limit
            record["max_tokens_forwarded"] = forwarded_limit
            record["decision"] = "capped" if forwarded_limit < requested_limit else "forwarded"

        chat_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        try:
            status, headers, response_body = await self._request_upstream(
                request,
                "/v1/chat/completions",
                body=chat_body,
                headers={**self._forward_headers(request.headers), "Content-Type": "application/json"},
            )
        except (ClientError, asyncio.TimeoutError):
            record["decision"] = "upstream_connection_error"
            record["upstream_status"] = "connection_error"
            await self._write_log(record)
            return web.json_response(
                {"error": {"message": "Unable to reach upstream chat endpoint", "type": "context_proxy_error"}},
                status=502,
            )

        record["upstream_status"] = status
        usage_prompt_tokens: int | None = None
        try:
            response_json = json.loads(response_body)
            usage = response_json.get("usage") or {}
            usage_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if isinstance(usage_prompt_tokens, int) and not isinstance(usage_prompt_tokens, bool):
                record["usage_prompt_tokens"] = usage_prompt_tokens
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass
        if 200 <= status < 300 and prompt_tokens is not None:
            if (
                not isinstance(usage_prompt_tokens, int)
                or isinstance(usage_prompt_tokens, bool)
                or usage_prompt_tokens != prompt_tokens
            ):
                record["decision"] = "fatal_usage_prompt_mismatch"
                record["fatal_mismatch"] = True
                await self._write_log(record)
                return web.json_response(
                    {
                        "error": {
                            "message": "Upstream usage.prompt_tokens does not match /tokenize count",
                            "type": "context_proxy_parity_error",
                            "tokenize_prompt_tokens": prompt_tokens,
                            "usage_prompt_tokens": usage_prompt_tokens,
                        }
                    },
                    status=502,
                )
        await self._write_log(record)
        return web.Response(status=status, headers=headers, body=response_body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True, help="Upstream vLLM root URL, with optional trailing /v1")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; defaults to localhost")
    parser.add_argument("--port", type=int, required=True, help="Local listen port")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--context-length", type=int, default=65_536)
    parser.add_argument("--log", required=True, help="Append-only JSONL diagnostics path")
    args = parser.parse_args()

    config = ProxyConfig(
        upstream=args.upstream,
        mode=args.mode,
        log_path=args.log,
        context_length=args.context_length,
    )
    web.run_app(ContextProxy(config).make_app(), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
