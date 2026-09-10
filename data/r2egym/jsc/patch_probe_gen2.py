p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"
s = open(p).read()
old = "f'export DCFT_RL_ENV={VENV}\\n"
new = ("f'export DCFT_RL_ENV={VENV}\\n"
       "export RL_PYTHON={VENV}/bin/python\\n"
       "export LD_LIBRARY_PATH=\"${{LD_LIBRARY_PATH//envs\\\\/rl-fa/envs\\\\/snowball}}\"\\n")
assert s.count(old) == 1 and "RL_PYTHON={VENV}" not in s
s = s.replace(old, new)
# the sbatch template later asserts DCFT_RL_ENV is present; keep. Also assert our new lines land.
s = s.replace('assert f"DCFT_RL_ENV={VENV}" in sb', 'assert f"DCFT_RL_ENV={VENV}" in sb and f"RL_PYTHON={VENV}/bin/python" in sb')
open(p, "w").write(s); print("generator patched")
