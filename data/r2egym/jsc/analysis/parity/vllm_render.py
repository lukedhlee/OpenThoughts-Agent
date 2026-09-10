import json
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M)
msgs = [{"role":"user","content":"U1"},{"role":"assistant","content":"\nR\n\n\n{\"a\":1}"},{"role":"user","content":"U2"}]
print("HF:", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
try:
    from vllm.entrypoints.chat_utils import apply_hf_chat_template, resolve_hf_chat_template
    from vllm.config import ModelConfig
    import inspect
    print("apply_hf_chat_template sig:", inspect.signature(apply_hf_chat_template))
    src = inspect.getsource(apply_hf_chat_template)
    print(src[:3000])
except Exception as e:
    print("vllm import/inspect failed:", repr(e))
# served prompt: check harbor-0.8.1 vs corpus wording of the newline rule
S = "/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
d = json.load(open(f"{S}/r2egym-0003__9XGriDa/attempts/000/agent/trajectory.json"))
ag = [s for s in d["steps"] if s.get("source") == "agent"]
txt = tok.decode(ag[2]["metrics"]["prompt_token_ids"], skip_special_tokens=False)
print("served has harbor081 line 'You must end every command with a newline':", "You must end every command with a newline" in txt)
print("served has corpus line 'Most bash commands should end with a newline':", "Most bash commands should end with a newline" in txt)
k = txt.find("Task Description:"); print("served prefix len before 'Task Description:':", k)
open("/e/project1/transfernetx/lee27/code/snowball/analysis/parity/served_prefix.txt","w").write(txt[:k])
# first user turn tail: 'Current terminal state:' block
k2 = txt.find("Current terminal state:"); print("first user turn 'Current terminal state' block:", repr(txt[k2:k2+200]))
