import json, glob, os, re, random, collections, sys
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M)
S = sys.argv[1]; ST, EN = 128002, 128003
ps = glob.glob(f"{S}/*/attempts/*/agent/trajectory.json"); random.seed(1); random.shuffle(ps); ps = ps[:400]
KIND = [("test", r"(pytest|python3? -m unittest|nosetests|tox\b)"), ("view_file", r"(^|[;&|]\s*)(cat |head |tail |sed -n|nl |less |more )"),
        ("search", r"(^|[;&|]\s*)(grep|rg|ag )"), ("listing", r"(^|[;&|]\s*)(find |ls |tree )"), ("run_python", r"python3? "), ("git", r"git ")]
def obs_content(o):
    if isinstance(o, dict):
        try: return o["results"][0]["content"]
        except Exception: return json.dumps(o)
    try: return json.loads(o)["results"][0]["content"]
    except Exception: return o if isinstance(o, str) else json.dumps(o)
share = collections.Counter(); n_big = 0; tot = 0; pos = collections.Counter(); head = collections.Counter(); ex = collections.Counter()
for p in ps:
    try: ag = [s for s in json.load(open(p))["steps"] if s.get("source") == "agent"]
    except Exception: continue
    for s in ag:
        ids = (s.get("metrics") or {}).get("completion_token_ids") or []
        if ids:
            pos[ids.index(ST) if ST in ids else -1] += 1; head[tok.decode(ids[:4], skip_special_tokens=False)[:24]] += 1
        c = obs_content(s.get("observation"))
        if len(c) < 6000: continue
        n = len(tok.encode(c, add_special_tokens=False))
        if n <= 2000: continue
        n_big += 1; tot += n
        msg = s.get("message", ""); kinds = set()
        try:
            a, b = msg.index("{"), msg.rindex("}"); j = json.loads(msg[a:b + 1])
            ks = "\n".join(c_.get("keystrokes", "") for c_ in (j.get("commands") or []) if isinstance(c_, dict))
            for k, pat in KIND:
                if re.search(pat, ks, re.M): kinds.add(k)
        except Exception: kinds.add("PARSE_FAIL")
        key = "+".join(sorted(kinds)) or "other"; share[key] += n
        if len(ex) < 6 and key not in ex: ex[key] = ks.strip()[:150].replace("\n", " ; ") if kinds and "PARSE_FAIL" not in kinds else msg[:100]
print(f"sample attempts={len(ps)}; observations >2k tokens: {n_big}, tokens {tot}")
print("dump tokens by command kind:", ", ".join(f"{k}={v / tot:.0%}" for k, v in share.most_common(10)))
print("examples:", dict(ex))
print("position of <|start_think|> in completion ids:", pos.most_common(5)); print("first-4-token heads:", head.most_common(5))
