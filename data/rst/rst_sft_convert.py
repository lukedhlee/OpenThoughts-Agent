#!/usr/bin/env python3
"""Build the Recursive-Task-Synthesis (RST) SFT slice from the GLM-5.3 rollout bundle.

Source: open-athena/recursive-task-synthesis-glm-5.3-rollouts (Terminus-2 JSON traces of GLM 5.3 on 17,206
gold-validated synthetic terminal tasks, 8 attempts each). The conversation of a trial is rebuilt from the last
episode's ``agent/episode-N/debug.json``: ``request.messages`` (user/assistant history, assistant turns carry
``reasoning_content``) plus the response message. Assistant turns become ``<think>..</think>\\n\\n<json>`` and then
``kimi_sft_convert.rewrite_assistant`` puts them on Snowball's special think tokens, as every other slice.

Selection (2026-09-20 plan, runs 3 and 4):
  * held-out tasks = whole families (``task_group_id`` hash), never trained on; written as ``rst-heldout-tasks.jsonl``;
  * ``rst-all``: turn_count >= 5, ONE rollout per task, a reward-1 rollout when the task has one, else a random
    other (reward 0, or a null-reward AgentTimeoutError/RuntimeError trace that still carries >= 5 real turns);
  * ``rst-succ``: the reward-1 rows of ``rst-all`` (a strict subset, so run 4 minus run 3 = the failed rollouts);
  * ``rst-succ8``: every reward-1 rollout with >= 5 turns (up to 8 per task), for a later rollouts-per-task arm.
Rows over the 32,768-token packing length are dropped. Output schema = ota_sft_convert.py's, slice "rst".

    python rst_sft_convert.py --raw <bundle dir> --heldout rst_heldout_tasks.json --tokenizer-dir <S3 export> \\
        --out <dir> --jupiter-root /e/data1/mmlaion/lee27/snowball-sft/data/rst_sft_v1
"""
from __future__ import annotations
import argparse, collections, hashlib, json, random, re, statistics as st, sys, tarfile
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "swesmith")]  # kimi_sft_convert.py lives in data/swesmith (copied beside this on Jupiter)
from kimi_sft_convert import PACK_LEN, rewrite_assistant  # noqa: E402

SCHEMA = pa.schema([("task", pa.string()), ("base_task", pa.string()), ("row_id", pa.string()), ("slice", pa.string()),
    ("kind", pa.string()), ("trace_source", pa.string()), ("result", pa.string()),
    ("conversations", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))), ("model", pa.string()),
    ("agent", pa.string()), ("trial_name", pa.string()), ("episode", pa.string()), ("instance_id", pa.string()),
    ("audit_reasons", pa.string()), ("final_complete", pa.bool_())])
EP_RE = re.compile(r"/agent/episode-(\d+)/debug\.json$")


def select(rows: list[dict], heldout: set[str], seed: int, min_turns: int) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Return (slices, fallbacks). slices['rst-all'] holds each task's first-choice rollout; fallbacks[task] the rest of
    the task's rollouts in preference order (reward-1 first, then the shortest by output tokens), used when the first
    choice does not fit the packing length. The 2026-09-20 first pass lost 4,488 of 16,545 tasks to the 32K cap when
    only one rollout per task was ever read."""
    rng = random.Random(seed)
    bytask: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["task_id"] in heldout or (r["turn_count"] or 0) < min_turns:
            continue
        bytask[r["task_id"]].append(r)
    sel = {"rst-all": [], "rst-succ": [], "rst-succ8": []}
    fallbacks: dict[str, list[dict]] = {}
    for tid in sorted(bytask):
        cands = sorted(bytask[tid], key=lambda r: r["execution_id"])
        succ = [r for r in cands if r["reward"] == 1]
        rest = [r for r in cands if r["reward"] != 1]
        rng.shuffle(succ); rng.shuffle(rest)
        pick = (succ + rest)[0]
        sel["rst-all"].append(pick)
        others = [r for r in succ if r is not pick] + sorted((r for r in rest if r is not pick), key=lambda r: r["output_tokens"] or 0)
        fallbacks[tid] = others
        sel["rst-succ8"].extend(succ)
    return sel, fallbacks


def conversation_from_trial(tf: tarfile.TarFile, members: dict[str, tarfile.TarInfo], prefix: str) -> list[dict] | None:
    eps = {int(m.group(1)): n for n, m in ((n, EP_RE.search(n)) for n in members) if m and n.startswith(prefix + "/")}
    if not eps:
        return None
    # the last episode may hold an error response (timeout mid-request: no "choices"); walk back to the last complete one
    dbg = None
    for e in sorted(eps, reverse=True):
        fh = tf.extractfile(members[eps[e]])
        cand = json.loads(fh.read()) if fh is not None else None
        if cand and (cand.get("response") or {}).get("choices"):
            dbg = cand; break
    if dbg is None:
        return None
    conv = []
    for m in dbg["request"]["messages"]:
        if m["role"] == "user":
            conv.append({"role": "user", "content": m["content"] or ""})
        elif m["role"] == "assistant":
            conv.append({"role": "assistant", "content": assistant_text(m.get("reasoning_content"), m.get("content"))})
        else:
            return None  # system turns never appear in this bundle; refuse rather than guess
    resp = dbg["response"]["choices"][0]["message"]
    think = resp.get("reasoning_content") or resp.get("reasoning") or ""
    conv.append({"role": "assistant", "content": assistant_text(think, resp.get("content"))})
    return conv


def assistant_text(think: str | None, content: str | None) -> str:
    content = (content or "").strip()
    think = (think or "").strip()
    return f"<think>\n{think}\n</think>\n\n{content}" if think else content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="local copy of the HF bundle (logical_attempts.parquet + payload/)")
    ap.add_argument("--heldout", required=True, type=Path, help="json list of held-out task ids")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jupiter-root", required=True)
    ap.add_argument("--min-turns", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=0, help="debug: only read the first N shards")
    ap.add_argument("--fallback-passes", type=int, default=3, help="how many further rollouts to try for a task whose pick did not fit")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    assert tok.convert_tokens_to_ids("<|start_think|>") == 128002, "wrong tokenizer"

    heldout = set(json.load(open(a.heldout)))
    rows = pq.read_table(a.raw / "logical_attempts.parquet").to_pylist()
    sel, fallbacks = select(rows, heldout, a.seed, a.min_turns)
    print(f"selected: " + ", ".join(f"{k}={len(v)}" for k, v in sel.items()), flush=True)

    stats: collections.Counter = collections.Counter()
    built: dict[str, dict] = {}
    exc_kept: collections.Counter = collections.Counter()

    def build_one(tf: tarfile.TarFile, members: dict, shard: str, r: dict) -> int | None:
        """Convert one trial; store it in built[] and return its token length, or None (reason counted in stats)."""
        prefix = r["member_prefix"]
        stats["rows_in"] += 1
        conv = conversation_from_trial(tf, members, prefix)
        if conv is None:
            stats["drop:no_episode_log"] += 1; return None
        out_conv, bad = [], False
        for m in conv:
            if m["role"] != "assistant":
                out_conv.append(m); continue
            new = rewrite_assistant(m["content"], stats)
            if new is None:
                bad = True; break
            out_conv.append({"role": "assistant", "content": new})
        if bad:
            stats["drop:malformed_think"] += 1; return None
        if len(out_conv) < 2 or out_conv[-1]["role"] != "assistant":
            stats["drop:shape"] += 1; return None
        text = tok.apply_chat_template(out_conv, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > PACK_LEN:
            stats["drop:over_len"] += 1; return None
        assert ids.count(128002) == sum(1 for m in out_conv if m["role"] == "assistant")
        reward = r["reward"]
        result = "" if reward is None else f"{float(reward):.1f}"
        if reward is None:
            exc_kept[r["exception_type"]] += 1
        built[prefix] = {"task": r["task_id"], "base_task": r["task_group_id"], "row_id": f"{shard}:{prefix}", "slice": "rst",
            "kind": "action", "trace_source": "rst-glm-5.3", "result": result, "conversations": out_conv,
            "model": "glm-5.3", "agent": "terminus-2", "trial_name": prefix, "episode": str(r["attempt"]),
            "instance_id": r["task_id"], "audit_reasons": r["exception_type"] or "", "final_complete": reward is not None,
            "_tokens": len(ids)}
        return len(ids)

    def run_pass(wanted: dict[str, dict], label: str) -> None:
        shards = sorted({r["shard"] for r in wanted.values()})
        if a.limit_shards:
            shards = shards[: a.limit_shards]
        for i, shard in enumerate(shards):
            need = {p for p, r in wanted.items() if r["shard"] == shard}
            with tarfile.open(a.raw / shard) as tf:
                members = {m.name: m for m in tf.getmembers() if m.isfile() and m.name.split("/")[0] in need and EP_RE.search(m.name)}
                for prefix in sorted(need):
                    build_one(tf, members, shard, wanted[prefix])
            print(f"[{label} {i + 1}/{len(shards)}] {shard}: built {len(built)} so far, stats {dict(stats)}", flush=True)

    # pass 1: every first-choice rollout (rst-all) plus every success (rst-succ8)
    wanted = {r["member_prefix"]: r for k in ("rst-all", "rst-succ8") for r in sel[k]}
    run_pass(wanted, "pass1")
    # passes 2..: tasks whose first choice did not fit get their next candidate, up to --fallback-passes times
    choice = {r["task_id"]: r for r in sel["rst-all"]}
    for p in range(a.fallback_passes):
        missing = [t for t, r in choice.items() if r["member_prefix"] not in built and fallbacks[t]]
        if not missing:
            break
        wanted = {}
        for t in missing:
            nxt = fallbacks[t].pop(0); choice[t] = nxt
            if nxt["member_prefix"] in built:  # a success already read in pass 1 that fits
                continue
            wanted[nxt["member_prefix"]] = nxt
        stats[f"fallback_pass{p + 2}_tasks"] = len(missing)
        print(f"fallback pass {p + 2}: {len(missing)} tasks, {len(wanted)} trials to read", flush=True)
        run_pass(wanted, f"pass{p + 2}")
    sel["rst-all"] = [choice[t] for t in sorted(choice)]
    sel["rst-succ"] = [r for r in sel["rst-all"] if r["reward"] == 1]
    lengths: dict[str, list[int]] = {k: [built[r["member_prefix"]]["_tokens"] for r in rs if r["member_prefix"] in built] for k, rs in sel.items()}
    for rec in built.values():
        rec.pop("_tokens", None)

    a.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for k, rs in sel.items():
        recs = [built[r["member_prefix"]] for r in rs if r["member_prefix"] in built]
        name = f"{k}-train-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(sorted(recs, key=lambda x: x["row_id"]), schema=SCHEMA), a.out / name,
                       compression="zstd", row_group_size=1000)
        written[name] = hashlib.sha256((a.out / name).read_bytes()).hexdigest()
    ho_rows = [r for r in rows if r["task_id"] in heldout]
    with open(a.out / "rst-heldout-tasks.jsonl", "w") as f:
        for tid in sorted(heldout):
            rs = [r for r in ho_rows if r["task_id"] == tid]
            f.write(json.dumps({"task": tid, "family": rs[0]["task_group_id"] if rs else None,
                                "teacher_rewards": [r["reward"] for r in rs], "teacher_exceptions": [r["exception_type"] for r in rs]}) + "\n")
    (a.out / "parquet.list").write_text("".join(f"{a.jupiter_root.rstrip('/')}/{n}\n" for n in sorted(written)))
    (a.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(written.items())))

    def q(v, p):
        v = sorted(v); return v[int(p * len(v))] if v else 0
    census = {"source": "open-athena/recursive-task-synthesis-glm-5.3-rollouts (payload read locally)", "min_turns": a.min_turns,
        "seed": a.seed, "heldout_tasks": len(heldout), "counts": dict(stats), "null_reward_exceptions_kept": dict(exc_kept),
        "slices": {k: {"selected": len(sel[k]), "train_rows": len(lengths[k]), "train_tokens": sum(lengths[k]),
                       "tokens_median": st.median(lengths[k]) if lengths[k] else 0, "tokens_p90": q(lengths[k], 0.9),
                       "tasks": len({r["task_id"] for r in sel[k] if r["member_prefix"] in built}),
                       "reward1": sum(1 for r in sel[k] if r["member_prefix"] in built and r["reward"] == 1),
                       "reward0": sum(1 for r in sel[k] if r["member_prefix"] in built and r["reward"] == 0),
                       "null": sum(1 for r in sel[k] if r["member_prefix"] in built and r["reward"] is None)} for k in sel},
        "sha256": written}
    (a.out / "census.json").write_text(json.dumps(census, indent=1)); print(json.dumps(census, indent=1))


if __name__ == "__main__":
    main()
