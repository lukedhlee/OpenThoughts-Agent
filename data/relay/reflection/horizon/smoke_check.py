"""Check a finished reflection run's artifacts against the HANDOFF smoke gates; prints a JSON report, exits 1 on a failure.

    python -m data.relay.reflection.horizon.smoke_check --config <config.json>

Gates per update: sources fresh, from the current policy version and the configured author; refresh confirmed the
current weights; rows mask the prompt and supervise every reflection token through the native end-of-turn; one optimizer
step with a finite loss; native metadata and online_state.json at the update's step with one fingerprint and distinct
source digests; the HF export carries the base's tokenizer files and template byte for byte, qk_mult 1.75, finite
tensors that moved from the previous weights, and router biases equal to the base's up to a per-layer constant (frozen).
Plus the post-check (the last export served as root and completed a request) and GPU samples and bytes per update.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from data.relay.reflection.core import load_config, read_jsonl

EOT = 128009


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def du(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def tensors(model_dir, wanted):
    """Selected tensors (as float32 torch tensors) from a sharded safetensors export."""
    import torch  # noqa: F401
    from safetensors import safe_open

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in wanted:
        with safe_open(str(Path(model_dir) / index[name]), framework="pt") as f:
            out[name] = f.get_tensor(name).float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    c = load_config(a.config)
    root = Path(c["runtime"]["output_dir"]) / c["name"]
    base = Path(c["backend"]["base_model"])
    state = json.loads((root / "state.json").read_text())
    report, failures = {"state": state, "updates": []}, []

    def gate(ok, what):
        if not ok:
            failures.append(what)

    gate(state["completed"] and state["iteration"] == c["training"]["updates"], "run not complete")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    biases = sorted(k for k in index if "router" in k and "bias" in k)
    probe = sorted(k for k in index if k.endswith("self_attn.q_proj.weight"))[:1] or sorted(index)[:1]
    previous = tensors(base, biases + probe)
    base_bias = {k: previous[k] for k in biases}
    model_before, tasks_seen, digests, fingerprints = str(base), set(), [], set()
    heldout = set(Path(c["data"]["heldout_tasks"]).read_text().split())
    for i in range(1, c["training"]["updates"] + 1):
        d = root / f"update-{i:04d}"
        u = {"update": i}
        sources = read_jsonl(d / "sources.jsonl")
        tasks = {s["task"] for s in sources}
        gate(len(sources) == c["training"]["source_attempts"], f"u{i} source count")
        gate(all(s["policy_version"] == i - 1 and s["author"] == "student" for s in sources), f"u{i} source provenance")
        gate(not tasks & tasks_seen and not tasks & heldout, f"u{i} sources not fresh or heldout")
        tasks_seen |= tasks
        u["sources"] = {"n": len(sources), "verdicts": [s["feedback"]["verdict"] for s in sources],
                        "terminations": [s["termination"] for s in sources]}
        if c["arm"] == "roft":
            refresh = json.loads((d / "refresh.response.json").read_text())
            roots = [m.get("root") for m in refresh["server_models"]["data"]]
            gate(refresh["model"] == model_before and refresh["weights_version"] == i - 1, f"u{i} refresh confirmation")
            gate([str(Path(r).resolve()) for r in roots] == [str(Path(model_before).resolve())], f"u{i} served root")
            u["served_root"] = roots
        u["census"] = json.loads((d / "generation_census.json").read_text())
        rows = read_jsonl(d / "rows.jsonl")
        boundary_text = []
        for r in rows:
            b = r["prompt_tokens"]
            gate(not any(r["loss"][:b]) and all(r["loss"][b:]) and r["ids"][-1] == EOT, f"u{i} row mask {r['sid']}")
            gate(r["policy_version"] == i - 1, f"u{i} row policy version")
            boundary_text.append([tok.decode(r["ids"][b - 4 : b], skip_special_tokens=False),
                                  tok.decode(r["ids"][b : b + 6], skip_special_tokens=False),
                                  tok.decode(r["ids"][-3:], skip_special_tokens=False)])
        u["rows"] = {"n": len(rows), "trained_tokens": [r["trained_tokens"] for r in rows],
                     "n_tokens": [r["n_tokens"] for r in rows], "boundaries": boundary_text}
        resp = json.loads((d / "train.response.json").read_text())
        gate(math.isfinite(resp["loss"]) and resp["optimizer_steps_before"] == i - 1
             and resp["optimizer_steps_after"] == i and resp["weights_version"] == i, f"u{i} train response")
        native, hf = Path(resp["optimizer_after"]), Path(resp["model_after"])
        meta = json.loads((native / "metadata.json").read_text())
        online = json.loads((native / "online_state.json").read_text())
        gate(meta.get("step") == i and online["iteration"] == i, f"u{i} native step")
        gate(len(online["source_digests"]) == i and len(set(online["source_digests"])) == i, f"u{i} source digests")
        fingerprints.add(online["fingerprint"])
        digests.append(resp["source_digest"])
        cfg = json.loads((hf / "config.json").read_text())
        gate(cfg.get("qk_mult") == 1.75, f"u{i} qk_mult")
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            gate((hf / name).is_file() and sha(hf / name) == sha(base / name), f"u{i} {name} differs from base")
        now = tensors(hf, biases + probe)
        gate(all(bool(t.isfinite().all()) for t in now.values()), f"u{i} nonfinite export tensors")
        moved = {k: float((now[k] - previous[k]).abs().max()) for k in probe}
        gate(all(v > 0 for v in moved.values()), f"u{i} weights did not move")
        drift = max((float(((now[k] - now[k].mean()) - (base_bias[k] - base_bias[k].mean())).abs().max())
                     for k in biases), default=None)
        gate(drift is not None and drift < 1e-2, f"u{i} router bias not frozen (drift {drift})")
        u["train"] = {"loss": resp["loss"], "supervised_tokens": resp["supervised_tokens"],
                      "native_step": meta.get("step"), "probe_max_abs_change": moved, "router_bias_max_drift": drift,
                      "router_bias_tensors": len(biases), "native_bytes": du(native), "hf_bytes": du(hf)}
        gpu = d / "training" / "gpu_samples.log"
        if gpu.exists():
            util, mem = [], []
            for line in gpu.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                try:
                    util.append(float(parts[2].split()[0]))
                    mem.append(float(parts[3].split()[0]))
                except (IndexError, ValueError):
                    continue
            u["gpu"] = {"samples": len(util), "max_util_pct": max(util, default=None),
                        "mean_util_pct": sum(util) / len(util) if util else None, "max_mem_mib": max(mem, default=None)}
        gate(gpu.exists() and u.get("gpu", {}).get("samples"), f"u{i} no GPU samples")
        previous, model_before = now, str(hf)
        report["updates"].append(u)
    gate(len(fingerprints) == 1 and len(set(digests)) == len(digests), "fingerprint or digest reuse across updates")
    post = root / "post_check"
    if (post / "refresh.response.json").exists():
        refresh = json.loads((post / "refresh.response.json").read_text())
        roots = [m.get("root") for m in refresh["server_models"]["data"]]
        completion = json.loads((post / "completion.json").read_text())
        text = completion["choices"][0]["message"].get("content") or ""
        gate([str(Path(r).resolve()) for r in roots] == [str(Path(model_before).resolve())], "post-check served root")
        gate(bool(text.strip()), "post-check empty completion")
        report["post_check"] = {"root": roots, "completion_head": text[:300], "usage": completion.get("usage")}
    else:
        failures.append("no post-check")
    report["failures"] = failures
    print(json.dumps(report, indent=1))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
