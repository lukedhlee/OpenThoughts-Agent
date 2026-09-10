import re, json, sys
L = sys.argv[1]
ansi = re.compile(r"\x1b\[[0-9;]*m")
for line in open(L, errors="replace"):
    if "WANDB_MIRROR kind=" not in line: continue
    line = ansi.sub("", line)
    kind = re.search(r"kind=(\w+) step=(\d+)", line)
    js = line[line.index("metrics=") + len("metrics="):].strip()
    try: d = json.loads(js)
    except Exception as e:
        print("PARSE FAIL", kind.group(0) if kind else "?", e, js[:200]); continue
    print(f"=== {kind.group(0)} ({len(d)} keys)")
    for k, v in d.items():
        k2 = k.replace("_e_fscratch_reformo_lee27_tasks_", "")
        if isinstance(v, float): v = round(v, 6)
        print(f"  {k2}: {v}")
