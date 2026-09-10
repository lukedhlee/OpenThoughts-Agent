#!/usr/bin/env python3
"""set_hydra.py <rl_config.json> key=value ... — set/replace hydra args in a run's rl_config.json (a key is matched ignoring a leading '++'/'+')."""
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    i = [j for j, x in enumerate(a) if x.lstrip("+").split("=", 1)[0] == k.lstrip("+")]
    assert len(i) <= 1, (k, i)
    if i:
        pre = a[i[0]][: len(a[i[0]]) - len(a[i[0]].lstrip("+"))]; a[i[0]] = pre + k.lstrip("+") + "=" + v
    else:
        a.append(kv)
    print(a[i[0]] if i else kv)
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
