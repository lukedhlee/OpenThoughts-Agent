#!/usr/bin/env python3
"""edit_target.py — WHERE did a command write?  (repo source / scratch file / patch file)

The history probe's published classifier (`experiments/hist_analysis/window.py`, `classify`) answers *what kind*
of thing a command is and throws the write TARGET away. The 2026-09-07 in-place-clause probe needs the target: its
whole question is whether naming "there is no patch file, edit in place" moves edits out of files nothing reads.

Two deliberate differences from `window.py`, both applied identically to every arm, so an A/B stays fair:

1. **Every line of a keystroke blob is scanned**, not only the first. `window.py:classify` splits on `&&`/`;` inside
   `s.split("\\n")[0]` because it needs one kind per command; a blob that explores on line 1 and rewrites a file on
   line 4 would be scored by its first line alone.
2. **The shell's working directory is tracked across `cd`**, and relative paths are resolved against it. Without this,
   `cd /testbed/src/PIL && sed -i '62s/.../' BufrStubImagePlugin.py` resolves to the bare name `BufrStubImagePlugin.py`,
   whose dirname is empty, and `_is_scratch` calls it a scratch file "at the repo root where source never lives". That
   is the single biggest error in the inherited predicate: it reclassifies a real source edit as a throwaway, and it
   fires on any model that navigates into a package before editing. Seen in `r2egym-v1-*` Pillow winners that the
   uncorrected classifier scored as "never edited source" while they were passing the graded tests.

3. **Every heredoc body is read for python write calls**, not only bodies whose command head is `python`. The common
   shape is `cat > /tmp/fix.py << EOF ... open("pandas/core/algorithms.py", "w") ... EOF` followed by
   `python3 /tmp/fix.py` a turn later: the write is declared in a body that a head-based rule reads as a scratch write.

4. **Python heredocs and `python -c` bodies are read for their write calls** (`open(p, "w")`, `Path(p).write_text`,
   `p.write_text`, `shutil.copy`/`move`, `json.dump(..., open(p, "w"))`). `window.py` cannot see these: `python3 - <<EOF`
   carries no `>` so no redirect regex matches and the segment falls through to "other". This matters here more than
   anywhere else, because the clause under test *recommends* "a short Python script that reads it and writes it back" —
   without this, the treatment arm would appear to edit source less simply by moving into the blind spot.

The path predicates (`_norm`, `_is_repo_test`, `_is_scratch`) and the write regexes are lifted verbatim from
`window.py` so "edit_src" here means what it meant in the 09-07 loser taxonomy. `selftest` re-checks that.

Library:  write_targets(keystrokes) -> [(path|None, how)]      how: redirect|tee_pipe|heredoc|sed_i|cp_mv|py_write|patch_apply|git_write
          attempt_flags(keystroke_strings, cwd="/testbed") -> ({"edit_src","edit_test","write_scratch","patch_file","unknown_write"}, cwd_after)
CLI:      edit_target.py selftest        agreement against window.py:classify, plus the worked cases below
"""
import os, re

# ---- verbatim from experiments/hist_analysis/window.py -------------------------------------------------
HEREDOC_W = re.compile(r"\b(?:cat|tee)\b[^\n<>|]*?>{1,2}\s*([^\s;&|]+)[^\n]*?<<")
PLAIN_W = re.compile(r"\b(?:cat|tee|printf|echo)\b[^\n<>|]*?>{1,2}\s*([^\s;&|]+)")
BARE_REDIR = re.compile(r"(?:^|&&|\|\|)\s*>{1,2}\s*([^\s;&|]+)")
SED_I = re.compile(r"\bsed\b[^\n|]*\s-i\b")
REPO_TEST_PATH = re.compile(r"(^|/)tests?(/|$)")
BACKUP_DST = re.compile(r"\.(bak|backup|orig|old|save|copy|prev)\d*$", re.I)


# ---- ours: resolve a path the way the shell would, before handing it to the verbatim predicates ------------
CD = re.compile(r"^cd\s+([^\s;&|]+)\s*$")


def resolve(path, cwd):
    """the absolute path a shell sitting in `cwd` would write to."""
    p = (path or "").strip("'\"")
    if not p or p.startswith(("$", "`")):
        return p
    if p.startswith("~"):
        return p
    return p if p.startswith("/") else os.path.normpath(os.path.join(cwd, p))


def next_cwd(line, cwd):
    """apply any `cd` in this line; unresolvable targets (`cd -`, `cd $X`) leave the cwd unknown-but-unchanged."""
    for sg in split_segments(line):
        m = CD.match(sg.strip())
        if not m:
            continue
        t = m.group(1).strip("'\"")
        if t in ("-", "~") or t.startswith(("$", "`")):
            continue
        cwd = t if t.startswith("/") else os.path.normpath(os.path.join(cwd, t))
    return cwd


def _norm(p):
    p = p.strip("'\"")
    if p.startswith("/testbed/"):
        p = p[len("/testbed/"):]
    return p


def _is_repo_test(path):
    """a path with a tests/ or test/ directory component belongs to the repo suite."""
    return bool(REPO_TEST_PATH.search(os.path.dirname(_norm(path))))


def _is_scratch(path):
    """model-authored file: under /tmp, or sitting at the repo root where source never lives."""
    if not path:
        return True
    p = path.strip("'\"")
    if p.startswith("/tmp/") or p.startswith("/var/tmp"):
        return True
    p = _norm(p)
    d = os.path.dirname(p)
    if REPO_TEST_PATH.search(d):
        return False
    return d in ("", ".", "/")
# ---- end verbatim lift ---------------------------------------------------------------------------------

# window.py's three redirect regexes only fire when the line STARTS with the redirect or is headed by cat/tee/printf/echo,
# so `git diff > fix.patch` -- the exact command this probe is about -- matches none of them. ANY_REDIR closes that, restricted
# to path-shaped targets so a ">" inside a one-line `python -c` body is not read as a redirect (heredoc BODIES are skipped
# outright by the scanner, which is where multi-line python code lives).
ANY_REDIR = re.compile(r">{1,2}\s*([\w./~@+-]*[./][\w./~@+-]*)")

# a file the model wrote a DIFF into: the SFT habit the clause is aimed at. Either the name says so, or the
# command redirects a diff generator into a file.
PATCH_NAME = re.compile(r"\.(patch|diff)$|(^|/)(patch|diff|fix|changes)\.(txt|out)$", re.I)
DIFF_PRODUCER = re.compile(r"\bgit\s+diff\b|\bdiff\s+(-\w+\s+)*-\w*[uU]\w*\b|\bgit\s+format-patch\b")

# python write calls inside a heredoc / -c body
PY_OPEN_W = re.compile(r"""\bopen\(\s*(?:file\s*=\s*)?['"]([^'"]+)['"]\s*,\s*['"][waxr]?[+bt]*[wax][+bt]*['"]""")
PY_OPEN_VAR = re.compile(r"""\bopen\(\s*[A-Za-z_]\w*\s*,\s*['"][^'"]*[wax][^'"]*['"]""")
PY_WRITE_TEXT = re.compile(r"""(?:Path\(\s*['"]([^'"]+)['"]\s*\)|['"]([^'"]+)['"])\s*\)?\s*\.write_text\(""")
PY_WRITE_TEXT_VAR = re.compile(r"\.write_text\(|\.write_bytes\(")
PY_SHUTIL = re.compile(r"""\bshutil\.(?:copy2?|copyfile|move)\(\s*[^,]+,\s*['"]([^'"]+)['"]""")
PY_HEREDOC = re.compile(r"\bpython[0-9.]*\b[^\n]*<<|\bpython[0-9.]*\b\s+-c\b")
SED_I_INLINE = re.compile(r"\bsed\b[^\n|]*?\s-i(?:\.\w+)?\b")


def _strip_comments(s):
    return "\n".join(l for l in s.split("\n") if not l.strip().startswith("#"))


def _sed_i_targets(line):
    parts = line.split()
    return [a for a in parts[1:] if not a.startswith("-") and ("/" in a or a.endswith(".py"))]


def split_segments(line):
    """split a command line on && || ; that are OUTSIDE quotes.  `sed -i 's/a;b/c/' f` must stay one segment;
    window.py's plain re.split breaks it into two and loses the -i."""
    segs, cur, q, i = [], [], None, 0
    while i < len(line):
        ch = line[i]
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            cur.append(ch)
        elif ch == ";":
            segs.append("".join(cur)); cur = []
        elif ch in "&|" and i + 1 < len(line) and line[i + 1] == ch:
            segs.append("".join(cur)); cur = []; i += 1
        else:
            cur.append(ch)
        i += 1
    segs.append("".join(cur))
    return [x for x in segs if x.strip()]


def _py_writes(body):
    """write targets a python body declares (heredoc body, or a `python -c` string)."""
    got = []
    for m in PY_OPEN_W.finditer(body):
        got.append(m.group(1))
    for m in PY_WRITE_TEXT.finditer(body):
        got.append(m.group(1) or m.group(2))
    for m in PY_SHUTIL.finditer(body):
        got.append(m.group(1))
    if not got and (PY_OPEN_VAR.search(body) or PY_WRITE_TEXT_VAR.search(body)):
        got.append(None)
    return got


def write_targets(keystrokes, cwd="/testbed"):
    """every file this keystroke blob writes to -> ([(abs_path|None, how)], cwd_after).

    `cwd` is the directory the shell is sitting in when the blob runs (tmux keeps it across turns, so an attempt
    threads it through). Targets come back resolved against it; None = a write we can see but cannot place.
    """
    s = (keystrokes or "").replace("\\n", "\n")
    s = _strip_comments(s)
    out, hd_cwd = [], cwd
    if not s.strip():
        return out, cwd

    if re.search(r"\bpython[0-9.]*\b\s+-c\b", s):          # inline python, no heredoc body to collect
        for t in _py_writes(s):
            out.append((resolve(t, cwd) if t else None, "py_write"))

    term, body = None, []
    for line in s.split("\n"):
        if term is not None:                                 # heredoc body: DATA, not commands
            if line.strip() == term:
                for t in _py_writes("\n".join(body)):        # ... but it declares its own writes
                    out.append((resolve(t, hd_cwd) if t else None, "py_write"))
                term, body = None, []
            else:
                body.append(line)
            continue
        hd = re.search(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?\s*$", line)
        line = line.strip()
        if not line:
            continue
        # segments run left to right, so a `cd` earlier in the line applies to a write later in it
        for sg in split_segments(line):
            sg = sg.strip()
            if not sg:
                continue
            if CD.match(sg):
                cwd = next_cwd(sg, cwd)
                continue
            for rx, how in ((HEREDOC_W, "heredoc"), (PLAIN_W, "redirect"), (BARE_REDIR, "redirect"), (ANY_REDIR, "redirect")):
                m = rx.search(sg)
                if m and ">" in sg:
                    t = m.group(1)
                    if t not in ("/dev/null", "/dev/stderr", "/dev/stdout") and not t.startswith("&"):
                        out.append((resolve(t, cwd), how))
                    break
            else:
                m = re.search(r"\|\s*tee\s+(?:-a\s+)?([^\s;&|]+)", sg)
                if m:
                    out.append((resolve(m.group(1), cwd), "tee_pipe"))
            if SED_I_INLINE.search(sg):
                t = _sed_i_targets(sg)
                out.append((resolve(t[-1], cwd) if t else None, "sed_i"))
            pp = sg.split()
            if pp and os.path.basename(pp[0]) in ("cp", "mv"):
                args = [x for x in pp[1:] if not x.startswith("-")]
                if len(args) >= 2 and not BACKUP_DST.search(args[-1]):
                    out.append((resolve(args[-1], cwd), "cp_mv"))
            if re.search(r"\bpatch\s+-p\d\b|\bgit\s+apply\b", sg):
                out.append((None, "patch_apply"))
            elif re.search(r"\bgit\s+(checkout|restore|stash|reset)\b", sg):
                out.append((None, "git_write"))
        if hd:
            term = hd.group(1)
            hd_cwd = cwd
    if term is not None and body:                            # blob ended inside a heredoc (no terminator emitted)
        for t in _py_writes("\n".join(body)):
            out.append((resolve(t, hd_cwd) if t else None, "py_write"))
    return out, cwd


def is_patch_file(path, keystrokes=""):
    """the write landed in a file holding a DIFF rather than source."""
    if path and PATCH_NAME.search(os.path.basename(path.strip("'\""))):
        return True
    return bool(path and DIFF_PRODUCER.search(keystrokes or ""))


def attempt_flags(cmds, cwd="/testbed"):
    """cmds = the keystroke strings, in order -> (flags, cwd_after).

    The shell is a tmux pane that OUTLIVES the turn, so `cd src/PIL` in turn 12 still governs `sed -i ... x.py` in
    turn 15. Callers that classify turn by turn must thread the returned cwd into the next call.

    `edit_src` is repo source PROPER: non-scratch and not under tests/. `edit_test` is the rest of the non-scratch
    writes -- files under tests/. window.py's `edit_src` kind is the union of the two (its `_write_kind` returns
    edit_src for any non-scratch target, repo tests included), so `edit_src or edit_test` is what compares against
    the 09-07 loser taxonomy, and `edit_src` alone is what this probe's question is about.
    """
    f = {"edit_src": False, "edit_test": False, "write_scratch": False, "patch_file": False, "unknown_write": False}
    for c in cmds or []:
        targets, cwd = write_targets(c, cwd)
        for path, how in targets:
            if path is None:
                f["unknown_write"] = True
                if how == "patch_apply":
                    f["edit_src"] = True          # applying a patch mutates the tree
                continue
            if is_patch_file(path, c):
                f["patch_file"] = True
                # a .patch file is itself a scratch write unless it sits over source
            if _is_scratch(path):
                f["write_scratch"] = True
            elif _is_repo_test(path):
                f["edit_test"] = True
            else:
                f["edit_src"] = True
    return f, cwd


CASES = [
    ("cat > /testbed/sympy/core/expr.py << 'EOF'\nx = 1\nEOF", "edit_src"),
    ("sed -i 's/a/b/' /testbed/pandas/core/frame.py", "edit_src"),
    ("python3 - << 'EOF'\nimport re\ns = open('/testbed/PIL/Image.py').read()\nopen('/testbed/PIL/Image.py','w').write(s)\nEOF", "edit_src"),
    ("python3 - <<'EOF'\nfrom pathlib import Path\nPath('/testbed/aiohttp/client.py').write_text('x')\nEOF", "edit_src"),
    ("git diff > /testbed/fix.patch", "patch_file"),
    ("cat > /tmp/fix.diff << 'EOF'\n--- a/x\nEOF", "patch_file"),
    ("cat > /testbed/reproduce_issue.py << 'EOF'\nprint(1)\nEOF", "write_scratch"),
    ("grep -rn 'foo' /testbed | head -50", None),
    ("python3 /testbed/reproduce_issue.py", None),
    ("cat /testbed/sympy/core/expr.py | head -50", None),
    ("patch -p1 < /tmp/fix.patch", "edit_src"),
    ("echo 'x = 2' >> /testbed/numpy/core/foo.py", "edit_src"),
    ("python3 -c \"print(1 > 0)\"", None),
    ("python3 - << 'EOF'\nfor a in xs:\n    if a > 0:\n        print(a)\nEOF", None),
    ("git diff", None),
    ("cat /testbed/setup.py > /dev/null", None),
    ("pytest /testbed/tests/test_core.py > /tmp/out.txt 2>&1", "write_scratch"),
    ("cat > /testbed/tests/test_new.py << 'EOF'\nx\nEOF", "edit_test"),
    # the two corrections, taken from real keep-arm winners the uncorrected predicate scored "never edited source"
    ("cd /testbed/src/PIL\nsed -i '62s/\"_handler\"/_handler/' BufrStubImagePlugin.py", "edit_src"),
    ("cat > /tmp/fix.py << 'EOF'\nwith open('pandas/core/algorithms.py','w') as f:\n    f.write(new)\nEOF", "edit_src"),
    ("cd /testbed && cat > reproduce_issue.py << 'EOF'\nprint(1)\nEOF", "write_scratch"),
    ("cd /testbed/pandas/core\ncat > frame.py << 'EOF'\nx\nEOF", "edit_src"),
]

if __name__ == "__main__":
    bad = 0
    for c, want in CASES:
        f, _ = attempt_flags([c])
        got = [k for k in ("edit_src", "edit_test", "write_scratch", "patch_file") if f[k]]
        ok = (want is None and not got) or (want is not None and want in got)
        if not ok:
            bad += 1
            print("FAIL want=%s got=%s :: %s" % (want, got, c.replace("\n", "\\n")[:90]))
    print("cases: %d, failures: %d" % (len(CASES), bad))
    try:
        import sys
        sys.path.insert(0, "/e/data1/mmlaion/lee27/experiments/hist_analysis")
        import window
        agree = tot = 0
        for c, _ in CASES:
            f, _ = attempt_flags([c])
            w = window.classify(c)
            mine = "edit_src" if (f["edit_src"] or f["edit_test"]) else ("write_scratch" if f["write_scratch"] else "-")
            tot += 1
            agree += (w == mine) or (w not in ("edit_src", "write_scratch") and mine == "-")
            print("  window=%-14s mine=%-14s :: %s" % (w, mine, c.replace("\n", "\\n")[:70]))
        print("window.py agreement on the worked cases: %d/%d (disagreements are the two documented extensions)" % (agree, tot))
    except Exception as e:
        print("window.py not importable here (%s) -- agreement check skipped" % e)
    raise SystemExit(1 if bad else 0)
