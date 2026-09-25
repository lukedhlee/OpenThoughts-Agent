#!/usr/bin/env python3
"""Scripted OpenAI-compatible server for a zero-GPU dry run of mini-swe-agent-host on Daytona.

Each episode (keyed by the ``session_id`` harbor sends in every request body) gets the same
scripted replies in order, so the whole Harbor path (job, sandbox, setup hook, agent loop,
exec, verifier, cleanup) runs without a model. The default script checks the sandbox
(workdir, python), a command that outlives the 30 s mini_textbased timeout, that its
process is gone afterwards, and the submit sentinel. Every request is appended to
``--log`` as one JSON line (session_id, number of messages, roles) for the relay check.

    python scripted_openai.py --port 18765 --log requests.jsonl [--quick]
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def bash(command: str, thought: str = "THOUGHT: next step.") -> str:
    return f"{thought}\n\n```mswea_bash_command\n{command}\n```"


SCRIPT = [
    "<think>I could start with\n```mswea_bash_command\nrm -rf /app\n```\nbut I should look "
    "first.</think>" + bash("pwd; echo PATH=$PATH; which python python3; uname -m; echo PAGER=$PAGER"),
    bash("sleep 40; touch /tmp/survived_timeout", "THOUGHT: this outlives the 30 s timeout."),
    bash("sleep 12; ls /tmp/survived_timeout 2>&1; echo live_sleeps=$(pgrep -fc 'sleep 40')"),
    bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
]


QUICK_SCRIPT = [SCRIPT[0], SCRIPT[-1]]


class Handler(BaseHTTPRequestHandler):
    script = SCRIPT
    turns: dict[str, int] = {}
    lock = threading.Lock()
    log_path = ""

    def log_message(self, *args):
        pass

    def _send(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "scripted", "object": "model", "max_model_len": 65536}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        session = request.get("session_id") or (request.get("extra_body") or {}).get("session_id") or "none"
        with self.lock:
            turn = self.turns.get(session, 0)
            self.turns[session] = turn + 1
            with open(self.log_path, "a") as log:
                log.write(json.dumps({"session_id": session, "turn": turn, "n_messages": len(request["messages"]),
                                      "roles": [m["role"] for m in request["messages"]]}) + "\n")
        content = self.script[min(turn, len(self.script) - 1)]
        self._send({
            "id": f"scripted-{session}-{turn}", "object": "chat.completion", "model": "scripted",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--log", required=True)
    parser.add_argument("--quick", action="store_true", help="only the sandbox check and the submit")
    args = parser.parse_args()
    Handler.log_path = args.log
    Handler.script = QUICK_SCRIPT if args.quick else SCRIPT
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
