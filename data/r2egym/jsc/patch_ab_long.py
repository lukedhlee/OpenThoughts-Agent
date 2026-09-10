import re
# 1) smoke_client.py: --pad-tokens N prepends ~N tokens of filler to every prompt (prefill-heavy A/B)
p = "/e/project1/transfernetx/lee27/code/snowball/smoke_client.py"; s = open(p).read()
if "--pad-tokens" not in s:
    s = s.replace('ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--conc", default="16,64")',
                  'ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--conc", default="16,64"); ap.add_argument("--pad-tokens", type=int, default=0)')
    s = s.replace('a = ap.parse_args()\n',
                  'a = ap.parse_args()\n'
                  'if a.pad_tokens:\n'
                  '    _filler = ("def f%d(x):\\n    return x * %d + 1\\n\\n" % (0, 0))\n'
                  '    _blob = "".join("def f%d(x):\\n    return x * %d + 1\\n\\n" % (i, i) for i in range(1, a.pad_tokens))\n'
                  '    _blob = _blob[: a.pad_tokens * 3]  # ~3 chars/token for this code-like filler\n'
                  '    PAD = "Here is a large source file for context; ignore it unless asked.\\n\\n" + _blob + "\\n\\nNow the actual task:\\n"\n'
                  'else:\n    PAD = ""\n', 1)
    s = s.replace('"messages": [{"role": "user", "content": prompt}]', '"messages": [{"role": "user", "content": PAD + prompt}]')
    open(p, "w").write(s); print("client patched")
else:
    print("client already patched")
# 2) serve_ab.sbatch: PAD_TOKENS / MAXTOK env -> client args
q = "/e/project1/transfernetx/lee27/code/snowball/serve_ab.sbatch"; t = open(q).read()
if "PAD_TOKENS" not in t:
    t = t.replace('CONC=${CONC:-"16,64,128"}', 'CONC=${CONC:-"16,64,128"}\nPAD_TOKENS=${PAD_TOKENS:-0}; MAXTOK=${MAXTOK:-512}')
    t = t.replace('label="ab_${LAYOUT}_${be,,}"', 'label="ab_${LAYOUT}_${be,,}_p${PAD_TOKENS}"')
    t = t.replace('--label $label --conc $CONC', '--label $label --conc $CONC --pad-tokens $PAD_TOKENS --max-tokens $MAXTOK')
    open(q, "w").write(t); print("sbatch patched")
else:
    print("sbatch already patched")
