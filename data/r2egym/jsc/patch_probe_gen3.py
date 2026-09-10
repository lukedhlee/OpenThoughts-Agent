p = "/e/project1/transfernetx/lee27/code/snowball/make_snowball_probe.py"; s = open(p).read()
old = "export RL_PYTHON={VENV}/bin/python\\n"
new = old + "export NCCL_PXN_DISABLE=1\\n"
assert s.count(old) == 1 and "NCCL_PXN_DISABLE" not in s
open(p, "w").write(s.replace(old, new)); print("generator patched (NCCL_PXN_DISABLE=1)")
