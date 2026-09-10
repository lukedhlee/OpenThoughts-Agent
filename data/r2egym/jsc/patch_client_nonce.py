p = "/e/project1/transfernetx/lee27/code/snowball/smoke_client.py"; s = open(p).read()
if "_NONCE" not in s:
    s = s.replace('import argparse, json, statistics, sys, threading, time', 'import argparse, json, statistics, sys, threading, time, itertools\n_NONCE = itertools.count()')
    # unique first line per request so the prefix cache cannot share the padding
    s = s.replace('"messages": [{"role": "user", "content": PAD + prompt}]',
                  '"messages": [{"role": "user", "content": (("request-id %d\\n" % next(_NONCE)) if PAD else "") + PAD + prompt}]')
    open(p, "w").write(s); print("nonce patched")
else:
    print("already")
