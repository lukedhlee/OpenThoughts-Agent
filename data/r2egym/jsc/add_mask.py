#!/usr/bin/env python3
"""add_mask.py <rl_config.json> <ExceptionName>... — append exception class names to the run's mask_exceptions hydra arg."""
import json, sys
p = sys.argv[1]; names = sys.argv[2:]
c = json.load(open(p)); a = c["skyrl_hydra_args"]
i = [k for k, x in enumerate(a) if "mask_exceptions=" in x]; assert len(i) == 1, i
x = a[i[0]]; assert x.endswith("]"), x
for e in names:
    if '"%s"' % e not in x:
        x = x[:-1] + ',"%s"]' % e
a[i[0]] = x; c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print(x[x.index("mask_exceptions="):][-160:])
