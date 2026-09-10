import sys
path, overlay, mode = sys.argv[1:4]
lines = open(path).read().split("\n")
launch = [i for i, l in enumerate(lines) if l.startswith('"$RL_PYTHON" -m hpc.rl_launch_utils --config')]
assert len(launch) == 1, f"expected one launch line, found {len(launch)}"
pp = [i for i, l in enumerate(lines) if l.startswith("export PYTHONPATH=")]
assert pp and max(pp) < launch[0], "PYTHONPATH export must precede the launch line"
name = overlay.rsplit("/", 1)[-1]
block = [
    "# history-think contract: harbor overlay " + name + " + HARBOR_TERMINUS2_HISTORY_THINK=" + mode + " (2026-09-06 sync overfit)",
    "export PYTHONPATH=" + overlay + "${PYTHONPATH:+:$PYTHONPATH}",
    "export HARBOR_TERMINUS2_HISTORY_THINK=" + mode,
    'echo "harbor overlay: $("$RL_PYTHON" -c \'import os, harbor.agents.terminus_2.terminus_2 as t; print(t.__file__, t.parse_history_think_mode(os.environ.get(t.HISTORY_THINK_ENV)))\')"',
    "",
]
lines[launch[0]:launch[0]] = block
open(path, "w").write("\n".join(lines))
print("overlay block inserted before line", launch[0] + 1)
