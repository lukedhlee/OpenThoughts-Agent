import json, glob, sys, collections, random
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M); ST, EN = 128002, 128003
for S, out in [(sys.argv[1], 8192), (sys.argv[2], 2048)]:
    ps = sorted(glob.glob(f"{S}/*/attempts/*/agent/trajectory.json")); random.seed(2); random.shuffle(ps)
    tails = collections.Counter(); st_in_prompt = collections.Counter(); ctmax = 0; c2048 = 0; cgt = 0; n = 0; hist = None
    for p in ps[:60]:
        ag = [s for s in json.load(open(p))["steps"] if s.get("source") == "agent"]
        for i, s in enumerate(ag):
            m = s.get("metrics") or {}; pids = m.get("prompt_token_ids") or []; cids = m.get("completion_token_ids") or []
            if not pids: continue
            tails[tok.decode(pids[-6:], skip_special_tokens=False)] += 1
            st_in_prompt[(pids.count(ST), pids.count(EN), i)] += 1 if i < 4 else 0
            ct = m.get("completion_tokens") or len(cids); n += 1; ctmax = max(ctmax, ct); c2048 += ct == out; cgt += ct > out
            if i == 2 and hist is None:
                txt = tok.decode(pids, skip_special_tokens=False); k = txt.find("<|start_header_id|>assistant<|end_header_id|>")
                hist = txt[k:k + 260]
    print(f"== {S.split('/')[-1]}: prompt tails: {tails.most_common(3)}")
    print(f"   (n_ST, n_EN, turn_idx) in prompt ids for turns 0-3: {sorted((k, v) for k, v in st_in_prompt.items() if v)[:8]}")
    print(f"   completions: n={n} max_ct={ctmax} ct=={out}: {c2048} ({c2048 / n:.1%}) ct>{out}: {cgt} ({cgt / n:.1%})")
    print(f"   turn-3 prompt, first assistant turn as rendered: {hist!r}")
    c = json.load(open(ps[0].split('/agent/')[0] + "/config.json")); print("   agent kwargs:", {k: v for k, v in c["agent"]["kwargs"].items() if k not in ("store_all_messages",)})
