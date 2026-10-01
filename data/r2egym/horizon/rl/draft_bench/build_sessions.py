#!/usr/bin/env python3
"""build_sessions.py — turn a model's harbor eval trajectories (terminus-2, ATIF trajectory.json) into replayable chat
sessions for the draft on/off serve bench (bench_client.py).

One session = one trial's first context segment: the initial user prompt, then for every agent step the recorded
assistant message (raw content, think span included, as harbor sent it back) and the observation as the next user
message. Steps after the first context summarization (a 'system' step) are dropped: their real prompt holds a summary
preamble this replay cannot rebuild. Turn k of a session is the request messages[:2k+1].

--check N renders N sessions with the model's chat_template.jinja + tokenizer.json (jinja2 + tokenizers only, no torch)
and compares each turn's prompt length with the prompt_tokens harbor recorded, to prove the replay is the real prompt.

  python build_sessions.py --glob '<jobs>/swe_s*_h9acont_r*/*/attempts/*/agent/trajectory.json' --out h9.jsonl \
      --model <hf dir> --check 20
"""
import argparse, ast, glob, json, random, statistics


def load_session(path):
    t = json.load(open(path))
    steps = t["steps"]
    if not steps or steps[0].get("source") != "user":
        return None
    msgs = [{"role": "user", "content": steps[0]["message"]}]
    ptoks, ctoks = [], []
    for s in steps[1:]:
        src = s.get("source")
        if src == "system":
            break  # context summarization: the next prompt is not a replay of this history
        if src != "agent":
            break
        met = s.get("metrics")
        met = ast.literal_eval(met) if isinstance(met, str) else (met or {})
        obs = s.get("observation")
        obs = ast.literal_eval(obs) if isinstance(obs, str) else obs
        if not met or not obs or not obs.get("results") or "content" not in obs["results"][0]:
            break
        ptoks.append(int(met.get("prompt_tokens", 0)))
        ctoks.append(int(met.get("completion_tokens", 0)))
        msgs.append({"role": "assistant", "content": s["message"]})
        msgs.append({"role": "user", "content": obs["results"][0]["content"]})
    if len(ptoks) < 2:
        return None
    # msgs ends with the last observation (a user turn nobody answered); turn k = msgs[:2k+1], k < len(ptoks)
    return {"id": path.split("/")[-6] + "/" + path.split("/")[-5], "messages": msgs[: 2 * len(ptoks) - 1],
            "prompt_tokens": ptoks, "completion_tokens": ctoks}


def check(sessions, model_dir, n):
    import jinja2
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    src = open(f"{model_dir}/chat_template.jinja").read()
    src = src.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    tmpl = env.from_string(src)
    diffs = []
    for s in random.Random(0).sample(sessions, min(n, len(sessions))):
        for k in sorted({0, 1, len(s["prompt_tokens"]) // 2, len(s["prompt_tokens"]) - 1}):
            text = tmpl.render(messages=s["messages"][: 2 * k + 1], bos_token="<|begin_of_text|>",
                               add_generation_prompt=True)
            got = len(tok.encode(text, add_special_tokens=False).ids)
            diffs.append(got - s["prompt_tokens"][k])
    print(f"check: {len(diffs)} turns, rendered-minus-recorded prompt tokens: min {min(diffs)} max {max(diffs)} "
          f"exact {sum(d == 0 for d in diffs)}/{len(diffs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    sessions = [s for s in (load_session(p) for p in sorted(glob.glob(a.glob))) if s]
    with open(a.out, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    pt = [p for s in sessions for p in s["prompt_tokens"]]
    ct = [c for s in sessions for c in s["completion_tokens"]]
    q = lambda x, p: sorted(x)[int(p * (len(x) - 1))]
    print(f"{len(sessions)} sessions, {len(pt)} turns; prompt p50 {q(pt, .5)} mean {statistics.mean(pt):.0f} "
          f"p90 {q(pt, .9)} max {max(pt)}; completion p50 {q(ct, .5)} mean {statistics.mean(ct):.0f} p90 {q(ct, .9)} "
          f"max {max(ct)} -> {a.out}")
    if a.check and a.model:
        check(sessions, a.model, a.check)


if __name__ == "__main__":
    main()
