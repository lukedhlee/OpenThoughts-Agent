import json, glob
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M)
S = "/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
for task in ["r2egym-0003__9XGriDa", "r2egym-0003__GeKs69d"]:
    d = json.load(open(f"{S}/{task}/attempts/000/agent/trajectory.json"))
    ag = [s for s in d["steps"] if s.get("source") == "agent"]
    pids = ag[2]["metrics"]["prompt_token_ids"]
    txt = tok.decode(pids, skip_special_tokens=False)
    HDR = "<|start_header_id|>"
    idx = [i for i in range(len(txt)) if txt.startswith(HDR, i)]
    print("=" * 80, task)
    for j, i in enumerate(idx[1:], 1):
        role = txt[i+len(HDR):i+len(HDR)+12].split('<')[0]
        seg_end = idx[j+1] if j+1 < len(idx) else len(txt)
        if role == "user":
            print(f"USER turn {j}: HEAD {txt[i:i+300]!r}")
    # ids after each <|eot_id|>
    pos = [k for k, t in enumerate(pids) if t == 128009]
    print("ids following each <|eot_id|> (3 ids):", [(pids[k+1:k+4]) for k in pos if k+1 < len(pids)])
    print("id before each <|eot_id|>:", [pids[k-1] for k in pos])
    # assistant turn 1 exact ids around reasoning->json junction: find 'task_complete' region? Instead find the first '\n\n\n{' occurrence
    a0 = txt.find("<|start_header_id|>assistant<|end_header_id|>")
    seg = txt[a0:a0+2000]
    k = seg.find("\n\n\n{")
    print("assistant turn1 junction reasoning->json:", repr(seg[max(0,k-40):k+12]) if k >= 0 else "NO '\\n\\n\\n{' found; first '{' ctx: " + repr(seg[max(0,seg.find('{')-40):seg.find('{')+12]))
    # exact ids at junction
    enc = tok(seg[:k+12], add_special_tokens=False)["input_ids"] if k >= 0 else []
    print("junction tail ids:", enc[-8:], [tok.decode([t]) for t in enc[-8:]])
print("\nHF apply_chat_template render check (export tokenizer):")
msgs = [{"role":"user","content":"U1"},{"role":"assistant","content":"\nR\n\n\n{\"a\":1}"},{"role":"user","content":"U2"}]
r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(r))
print("ids:", tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
import transformers, vllm, sys
print("transformers", transformers.__version__, "vllm", vllm.__version__)
