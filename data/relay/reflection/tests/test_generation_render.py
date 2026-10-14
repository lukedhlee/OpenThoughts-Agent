import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from data.relay.reflection.generation import generate_reflections, normalize_message
from data.relay.reflection.render import load_renderer, render_reflection

from .test_core import source


def config(url):
    return {
        "arm": "roft",
        "student": {"model": "grug", "weights_path": "/base"},
        "reflection": {
            "per_trajectory": 2,
            "temperature": 0.9,
            "top_p": 1.0,
            "max_tokens": 8192,
        },
        "runtime": {
            "timeout_sec": 3,
            "endpoints": {"student": {"url": url, "served_model": "student"}},
        },
    }


def test_http_generation_and_resume(tmp_path):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "student", "root": "/base"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(request)
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "Check logs first."},
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        c = config(f"http://127.0.0.1:{server.server_port}/v1")
        rows, bad = generate_reflections([source()], c, tmp_path)
        assert len(rows) == 2 and not bad and len(seen) == 2
        assert seen[0]["skip_special_tokens"] is False
        assert (
            seen[0]["max_tokens"] == 8192
            and "bad action" in seen[0]["messages"][0]["content"]
        )
        generate_reflections([source()], c, tmp_path)
        assert (
            len(seen) == 2
        )  # Successful generations reused exactly; no repeated requests.
        c["reflection"]["temperature"] = 0.8
        with pytest.raises(ValueError):
            generate_reflections([source()], c, tmp_path)
        with pytest.raises(ValueError, match="weights do not match"):
            generate_reflections([source()], c, tmp_path, expected_model="/other")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_nonstop_and_empty_rejected(tmp_path):
    def client(*args):
        return {
            "choices": [
                {"finish_reason": "length", "message": {"content": "truncated"}}
            ]
        }

    rows, bad = generate_reflections(
        [source()], config("unused"), tmp_path, client=client
    )
    assert not rows and bad == {"non_stop_completion": 2}
    assert normalize_message({"content": "<think> </think>lesson<|im_end|>"}) == {
        "role": "assistant",
        "content": "lesson",
        "reasoning_content": "",
    }
    assert (
        normalize_message({"content": "<|start_think|>reason<|end_think|>lesson"})[
            "reasoning_content"
        ]
        == "reason"
    )
    # Qwen's thought arrives in `reasoning` from this vLLM build and must not be dropped.
    assert normalize_message({"content": "\n\nlesson", "reasoning": "thought"}) == {
        "role": "assistant",
        "content": "lesson",
        "reasoning_content": "thought",
    }


def test_concurrent_generation_keeps_order(tmp_path):
    barrier = threading.Barrier(2, timeout=5)

    def client(url, payload, timeout, key):
        barrier.wait()  # both samples must be in flight at once
        return {"choices": [{"finish_reason": "stop", "message": {"content": "lesson"}}]}

    c = config("unused")
    c["reflection"]["concurrency"] = 2
    rows, bad = generate_reflections([source()], c, tmp_path, client=client)
    assert not bad and [r["sample"] for r in rows] == [0, 1]


@pytest.fixture
def native():
    p = os.environ.get("REFLECTION_TEST_TOKENIZER")
    if not p:
        pytest.skip(
            "set REFLECTION_TEST_TOKENIZER to the pinned native tokenizer directory"
        )
    return load_renderer(p)


@pytest.mark.parametrize("thinking", ["", "Evidence supports checking logs."])
def test_native_masks(native, thinking):
    tok, fmt, sha = native
    record = {
        "sid": "r",
        "source_id": "s",
        "task": "train",
        "prompt": "Recorded actions: broke the build.\nFeedback: failed.",
        "message": {
            "role": "assistant",
            "content": "Check the logs before editing.",
            "reasoning_content": thinking,
        },
        "trajectory_author": "student",
        "reflection_author": "student",
        "policy_version": 0,
    }
    row = render_reflection(record, tok, fmt, 65536, sha)
    boundary = row["prompt_tokens"]
    assert all(x == 0 for x in row["loss"][:boundary])
    assert all(x == 1 for x in row["loss"][boundary:])
    assert row["ids"][-1] == 128009 and row["loss"][-1] == 1
    target = tok.decode(row["ids"][boundary:], skip_special_tokens=False)
    assert "Check the logs" in target
    assert ("<|start_think|>" in target) == bool(thinking)
    with pytest.raises(ValueError, match="sequence_overflow"):
        render_reflection(record, tok, fmt, 8, sha)
    # A completion that writes a turn boundary inside its answer is rejected, not trained; a repeated think close is kept.
    broken = dict(record, message=dict(record["message"], content="Look.<|eot_id|><|start_header_id|>user<|end_header_id|>"))
    with pytest.raises(ValueError, match="control_token_in_completion"):
        render_reflection(broken, tok, fmt, 65536, sha)
    twice = dict(record, message=dict(record["message"], content="More thought.<|end_think|>Check the logs."))
    row = render_reflection(twice, tok, fmt, 65536, sha)
    assert tok.decode(row["ids"][row["prompt_tokens"]:], skip_special_tokens=False).count("<|end_think|>") == 1 + bool(thinking)
