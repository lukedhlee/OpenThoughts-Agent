# Emit "name==version" for packages installed in rl-fa but absent from snowball, excluding vllm-owned families.
import json, re, subprocess
C = "/e/project1/transfernetx/lee27/code"; UV = "/e/home/jusers/lee27/jupiter/.local/bin/uv"
def lst(env):
    return {p["name"].lower().replace("_", "-"): p["version"] for p in json.loads(subprocess.check_output([UV, "pip", "list", "-p", f"{C}/envs/{env}/bin/python", "--format=json"]))}
rf, sb = lst("rl-fa"), lst("snowball")
skip = re.compile(r"^(torch|torchvision|torchaudio|vllm|flashinfer|triton|pytorch-triton|transformers|numpy|xformers|nvidia|cuda|flash-attn|tokenizers|safetensors|huggingface|skyrl|harbor|dynamic-semaphore|torchtitan|transformer-engine|marinskyrl|torchdata)")
out = [f"{n}=={v}" for n, v in sorted(rf.items()) if n not in sb and not skip.match(n)]
open(f"{C}/envs/snowball.delta.in", "w").write("\n".join(out) + "\n")
print("delta:", len(out)); print(" ".join(out))
