p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"
s = open(p).read()
old = "f'export DCFT_RL_ENV={VENV}\\nRL_ENV_DIR="
new = ("f'export DCFT_RL_ENV={VENV}\\n"
       "export VLLM_CACHE_ROOT=/e/fscratch/reformo/lee27/cache/vllm XDG_CACHE_HOME=/e/fscratch/reformo/lee27/cache/xdg "
       "TRITON_CACHE_DIR=/e/fscratch/reformo/lee27/cache/triton TORCHINDUCTOR_CACHE_DIR=/e/fscratch/reformo/lee27/cache/inductor\\n"
       "RL_ENV_DIR=")
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
print("patched")
