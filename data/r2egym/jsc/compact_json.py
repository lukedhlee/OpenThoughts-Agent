import json, sys, os
for p in sys.argv[1:]:
    try:
        d = json.load(open(p)); tmp = p + ".tmp"
        with open(tmp, "w") as f: json.dump(d, f, separators=(",", ":"))
        os.replace(tmp, p)
    except Exception as e:
        print("skip", p, e)
