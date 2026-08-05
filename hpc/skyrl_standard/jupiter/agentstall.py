import json, glob, os, sys, re

D = sys.argv[1]
rows = []
PAT = re.compile(r'"name"\s*:\s*"(bash|edit|write|read|glob|grep|patch)"')

for f in sorted(glob.glob(os.path.join(D, "*", "agent", "opencode.txt"))):
    trial = os.path.basename(os.path.dirname(os.path.dirname(f)))
    tool = 0
    rg = 0
    texts = []
    last = None
    edits = 0
    for line in open(f, errors="replace"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        part = e.get("part") or {}
        if t == "tool_use":
            tool += 1
            name = part.get("tool")
            st = part.get("state") or {}
            if "ripgrep" in str(st.get("error", "")):
                rg += 1
            if name in ("edit", "write", "patch") and st.get("status") != "error":
                edits += 1
        elif t == "text":
            texts.append(part.get("text") or "")
        elif t == "step_finish":
            last = part.get("reason")
    unparsed = sum(1 for x in texts if PAT.search(x))
    rows.append((trial, tool, rg, edits, len(texts), unparsed, last))

print(f"{'trial':26} {'tools':>5} {'rgerr':>5} {'edits':>5} {'texts':>5} {'rawjson':>7} last_reason")
for r in rows:
    print(f"{r[0]:26} {r[1]:5} {r[2]:5} {r[3]:5} {r[4]:5} {r[5]:7} {r[6]}")

n = len(rows)
print()
print("trials analysed                          :", n)
print("with a raw-JSON tool call left AS TEXT   :", sum(1 for r in rows if r[5] > 0))
print("with a ripgrep tool error                :", sum(1 for r in rows if r[2] > 0))
print("that made >=1 successful edit/write      :", sum(1 for r in rows if r[3] > 0))
print("ended with reason=stop                   :", sum(1 for r in rows if r[6] == "stop"))
