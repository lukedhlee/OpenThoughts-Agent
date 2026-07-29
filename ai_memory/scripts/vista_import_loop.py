#!/usr/bin/env python3
"""Install missing light deps into Luke's PYTHONUSERBASE until main_base imports."""
import os
import re
import subprocess
import sys

os.environ["PYTHONUSERBASE"] = "/scratch/11584/lukedhlee/python_user"
os.environ["PYTHONPATH"] = (
    "/scratch/11584/lukedhlee/python_user/lib/python3.12/site-packages:"
    "/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train:"
    "/scratch/11584/lukedhlee/MarinSkyRL/skyrl-gym:"
    "/scratch/11584/lukedhlee/harbor/src:"
    "/scratch/11584/lukedhlee/OpenThoughts-Agent"
)
py = "/scratch/10635/penfever/miniconda3/envs/otagent/bin/python"
MAP = {
    "hydra": "hydra-core",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skyrl_gym": None,
    "flash_attn": None,
    "torch": None,
    "ray": None,
    "vllm": None,
}

for i in range(25):
    p = subprocess.run(
        [py, "-c", "from skyrl_train.entrypoints.main_base import *"],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    if p.returncode == 0:
        print(f"IMPORT CLEAN after {i}")
        sys.exit(0)
    err = p.stderr + p.stdout
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", err)
    if not m:
        print("NON-MODULE:\n", err[-3000:])
        sys.exit(1)
    mod = m.group(1).split(".")[0]
    pkg = MAP.get(mod, mod)
    if pkg is None:
        print(f"BLOCKED on {mod}:\n", err[-2500:])
        sys.exit(2)
    print(f"[{i}] install {pkg} (import {mod})")
    subprocess.check_call([py, "-m", "pip", "install", "--user", pkg])

print("gave up")
sys.exit(3)
