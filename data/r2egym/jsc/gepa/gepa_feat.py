#!/usr/bin/env python3
"""gepa_feat.py — per-trial reward + deterministic behaviour features for GEPA probes. Imported, not run.

Ported by diff from three existing readers rather than rewritten:
  * the command/think parser from p2o/p2o_adherence.py (`parse_response`) -- these trajectories have NO `tool_calls`
    key, the commands live inside the agent message as terminus-2 JSON, so the swe scan's extraction does not apply;
  * the write-target / edit-mode / repeat / rm / json-warning detectors from
    /e/data1/mmlaion/lee27/experiments/behaviour_scans_20260919/swe_evals/scan.py (regexes verbatim);
  * the self-check detector from .../behaviour_scans_20260919/tb2_why/selfcheck.py.

Two substitutions the originals force, both deliberate:
  1. scan.py reads `step["tool_calls"]`; here the keystrokes come from parse_response(message). The regex layer below
     it transfers unchanged -- only the I/O layer differs.
  2. selfcheck.py's `read_back_required` needs `required_paths` from tb2_why/task_meta.json. R2E-Gym has no such
     manifest, so the required set is replaced by the repo paths the agent actually edited: "did it read back what it
     just wrote". Same intent, evaluated against the trial's own edits.

Reward rules (confirmed against p2o6all_s0, 2026-09-20): the graded file is `verifier/reward.txt` in the LAST attempt
that has a trajectory; `result.json -> verifier_result.rewards.reward` agrees and is the fallback. ~6 % of last
attempts have neither because the attempt died before the verifier ran (ConnectTimeout, BridgeOperationError): that is
a DROPPED SAMPLE, not a zero, and it must stay out of pass-rate denominators. One reward.txt in that run is empty, so
float() is guarded and the failure is recorded rather than swallowed.

Python 3.9 / stdlib (Jupiter login node).
"""
import glob, json, os, re, statistics as st

# ---------------------------------------------------------------- terminus-2 message parser (p2o_adherence.py)
def parse_response(msg):
    """terminus-2 JSON response -> (analysis, plan, [keystrokes], task_complete, had_think); tolerant of think spans."""
    m = msg or ""
    had_think = "<|start_think|>" in m
    if "<|end_think|>" in m:
        m = m.split("<|end_think|>", 1)[1]
    i, j = m.find("{"), m.rfind("}")
    if i < 0 or j < 0:
        return "", "", [], False, had_think
    try:
        d = json.loads(m[i:j + 1])
    except Exception:
        return "", "", [], False, had_think
    cmds = [c.get("keystrokes", "") for c in (d.get("commands") or []) if isinstance(c, dict)]
    return d.get("analysis") or "", d.get("plan") or "", cmds, bool(d.get("task_complete")), had_think


# ---------------------------------------------------------------- detectors (swe_evals/scan.py, verbatim)
SRC_EXT = r"(?:py|pyx|pyi|txt|rst|md|cfg|toml|json|yaml|yml|html|c|h|cpp|sh|in|ini|template|tpl|po)"
QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
PATHTOK = re.compile(r"(?:\./|/|~/)?[\w][\w./+-]*\." + SRC_EXT + r"\b|/[\w./+-]+")
REDIR = re.compile(r"(?<![0-9&=<>!])>>?(?!=)\s*([^\s;|&<>()\"']+)")
TEE = re.compile(r"\btee\s+(?:-a\s+)?([^\s;|&<>]+)")
PYWRITE = re.compile(r"open\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"'][^\"']*[wax+][^\"']*[\"']")
PYWRITE2 = re.compile(r"(?:Path\(\s*[\"']([^\"']+)[\"']\s*\)\s*\.write_(?:text|bytes))")
CPMV = re.compile(r"\b(?:cp|mv)\s+(?:-[a-zA-Z]+\s+)*[^\s;|&]+\s+([^\s;|&<>]+)")
HEREDOC_TGT = re.compile(r"<<-?\s*[\"']?\w+[\"']?\s*>>?\s*([^\s;|&]+)")
APPLY = re.compile(r"\bgit\s+apply\b|\bpatch\s+-p\d|\bapply_patch\b")
SEARCHBLK = re.compile(r"<<<<<<<\s*SEARCH")
SEDI_CMD = re.compile(r"\bsed\b[^\n;|&]*?(?:^|\s)-[a-zA-Z]*i[a-zA-Z]*(?:\s|\.|$)")
TEST_TOOL = re.compile(r"\b(pytest|py\.test|nosetests|tox)\b|python3?\s+-m\s+(pytest|unittest|nose)\b|\bunittest\b|\brun_tests(?:\.py|\.sh)?\b|\bruntests\.py\b|\bmanage\.py\s+test\b")
TEST_SCRIPT = re.compile(r"\bpython3?\s+[^\s;|&]*\b(?:test|repro|reproduce|check|verify|debug|test_fix)[\w./-]*\.py\b", re.I)
REPRO_SCRIPT = re.compile(r"\bpython3?\s+[^\s;|&]*\b(?:repro|reproduce|test_fix|check|verify|debug)[\w./-]*\.py\b", re.I)
GITRESTORE = re.compile(r"\bgit\s+(?:checkout|reset|restore|stash|revert)\b")
GITDIFF = re.compile(r"\bgit\s+(?:diff|status)\b")
PREAMBLE_WARN = re.compile(r"Extra text detected before JSON object|AUTO-CORRECTED: Extracted JSON from mixed content")
SCHEMA_WARN = re.compile(r"Unknown fields|missing required|should end with newline|Failed to parse|No valid JSON|Invalid JSON")
BACKUP = re.compile(r"\.(?:bak|orig|backup|save|old|copy)\d*$|~$")
RM_RF = re.compile(r"\brm\s+(?:-[a-zA-Z]+\s+)*(-[a-zA-Z]*r[a-zA-Z]*)?\s*([^\s;|&]+)")
NEWFILE_HINT = re.compile(r"(?:^|/)(?:repro|reproduce|test_fix|check|verify|debug|tmp|scratch|demo|minimal|bug|issue|example|run)[\w-]*\.py$", re.I)

# ---------------------------------------------------------------- detectors (tb2_why/selfcheck.py, verbatim)
CHECKNAME = re.compile(r"(?:^|/)[\w.-]*(?:test|check|verify|validat|assert|sanity|smoke)[\w.-]*"
                       r"\.(?:py|sh|js|rb|pl)$", re.I)
RUNNER = re.compile(r"\b(?:python3?|bash|sh|node|ruby|perl|pytest|py\.test)\s+(?:-\S+\s+)*([^\s;|&<>]+)")
DOTSLASH = re.compile(r"(?:^|[;|&]\s*)(\./[^\s;|&<>]+)")
READVERB = re.compile(r"\b(?:cat|head|tail|less|more|od|xxd|hexdump|wc|grep|egrep|rg|diff|cmp|ls|"
                      r"stat|file|md5sum|sha1sum|sha256sum|jq|openssl|unzip|tar|readelf|objdump|"
                      r"sqlite3|python3?)\b")
OPENREAD = re.compile(r"open\(\s*[\"'][^\"']+[\"']\s*(?:,\s*[\"']r)?")
ASSERTX = re.compile(r"\bassert\s|\bassertEqual\b|\bnp\.testing\b")
DIFFX = re.compile(r"\bdiff\s+[^\s;|&]+\s+[^\s;|&]+|\bcmp\s+[^\s;|&]+\s+[^\s;|&]+")


def is_tmp(p):
    return p.startswith("/tmp") or p.startswith("tmp/") or p.startswith("./tmp/") or p.startswith("/dev/") or p == "/dev/null"


def is_patchfile(p):
    return p.endswith(".patch") or p.endswith(".diff")


def looks_like_path(p):
    if not p or p[0] in "$=({":
        return False
    if p in ("&1", "&2"):
        return False
    if re.match(r"^[\d.]+$", p):
        return False
    if "." not in p and "/" not in p:
        return False
    if re.match(r"^\d", p):
        return False
    return True


def sed_targets(ks):
    """File args of a `sed -i` invocation: strip quoted scripts, take path-like tokens."""
    out = []
    for seg in re.split(r"[;|&\n]", ks):
        if not SEDI_CMD.search(seg):
            continue
        rest = QUOTED.sub(" ", seg)
        rest = re.sub(r"\bsed\b", " ", rest)
        rest = re.sub(r"(?:^|\s)-[a-zA-Z.]+", " ", rest)
        for m in PATHTOK.finditer(rest):
            t = m.group(0)
            if looks_like_path(t):
                out.append(t)
    return out


def write_targets(ks):
    tg = []
    tg.extend(sed_targets(ks))
    for rx in (TEE, PYWRITE, PYWRITE2, CPMV, HEREDOC_TGT):
        for m in rx.finditer(ks):
            g = m.group(1)
            if g:
                tg.append(g)
    for m in REDIR.finditer(ks):
        tg.append(m.group(1))
    ap = bool(APPLY.search(ks)) or bool(SEARCHBLK.search(ks))
    tg = [t.strip("'\"") for t in tg]
    tg = [t for t in tg if looks_like_path(t)]
    return tg, ap


def classify_targets(tg, ap):
    kinds = set()
    repo = []
    for p in tg:
        if BACKUP.search(p):
            kinds.add("backup")
            continue
        if is_tmp(p):
            kinds.add("tmp")
        elif is_patchfile(p):
            kinds.add("patch")
        else:
            kinds.add("repo")
            repo.append(p)
    if ap:
        kinds.add("patch")
    return kinds, repo


def cmd_tokens(k):
    """Candidate file arguments of a command (selfcheck.py)."""
    out = []
    for m in RUNNER.finditer(k):
        out.append(m.group(1))
    for m in DOTSLASH.finditer(k):
        out.append(m.group(1))
    return [t.strip("'\"") for t in out]


# ---------------------------------------------------------------- trial reader
AXES = ["self_check", "ran_test_after_edit", "in_place", "no_sed_patch",
        "no_repeat3", "no_json_reject", "no_input_delete", "no_ctx_death"]


def last_attempt(trial_dir):
    """The attempt that was graded: the last one holding a trajectory (000-003 are retries of ONE sample)."""
    atts = sorted(glob.glob(os.path.join(trial_dir, "attempts", "*", "agent", "trajectory.json")))
    return os.path.dirname(os.path.dirname(atts[-1])) if atts else None


def read_reward(att):
    """(reward, source, error). None reward = the attempt died before the verifier: a dropped sample, not a zero."""
    p = os.path.join(att, "verifier", "reward.txt")
    if os.path.exists(p):
        raw = open(p).read().strip()
        if raw:
            try:
                return float(raw), "reward.txt", None
            except ValueError:
                return None, "reward.txt", "unparseable:%r" % raw[:20]
        return None, "reward.txt", "empty"
    try:
        d = json.load(open(os.path.join(att, "result.json")))
        v = d.get("verifier_result")
        r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
        if r is not None:
            return float(r), "result.json", None
    except Exception as e:
        return None, "none", "result:%s" % type(e).__name__
    return None, "none", None


def exception_type(att):
    try:
        d = json.load(open(os.path.join(att, "result.json")))
        return ((d.get("exception_info") or {}).get("exception_type")) or ""
    except Exception:
        pass
    p = os.path.join(att, "exception.txt")
    if os.path.exists(p):
        head = open(p, errors="replace").read(4000)
        m = re.search(r"(\w*Error|\w*Timeout|\w*Exception)\b", head)
        return m.group(1) if m else "unknown"
    return ""


def turns_of(att):
    """[(i, analysis, plan, [keystrokes], task_complete, had_think, observation_text)] over the agent steps."""
    d = json.load(open(os.path.join(att, "agent", "trajectory.json")))
    out = []
    for i, s in enumerate([x for x in (d.get("steps") or []) if x.get("source") == "agent"]):
        an, pl, cmds, tc, th = parse_response(s.get("message") or "")
        o = s.get("observation") or {}
        try:
            obs = "\n".join(x.get("content", "") for x in (o.get("results") or []))
        except Exception:
            obs = ""
        out.append((i, an, pl, cmds, tc, th, obs))
    return out


def trial_features(trial_dir):
    """Reward + the eight Pareto axes + the supporting counters, for one trial dir."""
    f = {"trial": os.path.basename(trial_dir)}
    att = last_attempt(trial_dir)
    if not att:
        f["parse_error"] = "no_trajectory"
        return f
    f["attempt"] = os.path.basename(att)
    f["reward"], f["reward_src"], f["reward_error"] = read_reward(att)
    f["exception_type"] = exception_type(att)
    try:
        turns = turns_of(att)
    except Exception as e:
        f["parse_error"] = "bad_trajectory:%s" % type(e).__name__
        return f

    all_cmds = []           # (turn index, keystrokes)
    step_tuples = []        # per-turn command tuple, for the repeat detector
    declared = False
    json_reject = 0         # SCHEMA-level rejection: the harness could not use the response
    preamble = 0            # benign: text before the JSON object, auto-corrected
    obs_n = 0
    no_cmd = 0
    think_turns = 0
    for i, an, pl, cmds, tc, th, obs in turns:
        step_tuples.append(tuple(cmds))
        for k in cmds:
            all_cmds.append((i, k))
        if not cmds:
            no_cmd += 1
        if th:
            think_turns += 1
        declared = declared or tc
        if obs:
            obs_n += 1
            # PREAMBLE_WARN fires on essentially every turn of a think-span model under the parity setup
            # (40/40 sampled p2oAc_s0 trials, 2026-09-20), so it carries no signal and is counted separately.
            # Only SCHEMA_WARN ("Invalid JSON", "No valid JSON", "should end with newline", "missing required")
            # marks a turn the harness actually rejected.
            if PREAMBLE_WARN.search(obs):
                preamble += 1
            if SCHEMA_WARN.search(obs):
                json_reject += 1

    # --- edit mode (scan.py)
    kinds_any, repo_targets = set(), []
    first_edit = None
    n_edit_cmds = 0
    for i, k in all_cmds:
        tg, ap = write_targets(k)
        kk, repo = classify_targets(tg, ap)
        eff = kk - {"backup"}
        if eff:
            n_edit_cmds += 1
            kinds_any |= eff
            if first_edit is None:
                first_edit = i
        for p in repo:
            if p not in repo_targets:
                repo_targets.append(p)
    used_sed_i = any(SEDI_CMD.search(k) for _, k in all_cmds)
    used_apply = any(APPLY.search(k) or SEARCHBLK.search(k) for _, k in all_cmds)

    # --- tests after the first edit (scan.py)
    ran_after = ran_suite = ran_repro = False
    for i, k in all_cmds:
        if TEST_TOOL.search(k):
            ran_suite = True
        if REPRO_SCRIPT.search(k):
            ran_repro = True
        if first_edit is not None and i > first_edit and (TEST_TOOL.search(k) or TEST_SCRIPT.search(k)):
            ran_after = True

    # --- repeats (scan.py): consecutive identical command tuples
    best = cur = 1
    for i in range(1, len(step_tuples)):
        cur = cur + 1 if (step_tuples[i] and step_tuples[i] == step_tuples[i - 1]) else 1
        best = max(best, cur)

    # --- deletion of inputs (scan.py rm_repo): rm of a non-tmp, non-backup target
    rm_repo = 0
    for _, k in all_cmds:
        for m in RM_RF.finditer(k):
            tgt = (m.group(2) or "").strip("'\"")
            if tgt and not is_tmp(tgt) and not tgt.startswith("-") and not BACKUP.search(tgt):
                rm_repo += 1

    # --- self check (selfcheck.py, with `required_paths` replaced by the trial's own repo edits)
    check_written, req_written = {}, {}
    for i, k in all_cmds:
        tg, _ = write_targets(k)
        for p in tg:
            p = p.strip("'\"").rstrip("/")
            if CHECKNAME.search(p) and p not in check_written:
                check_written[p] = i
            if p in repo_targets and p not in req_written:
                req_written[p] = i
    ran_check = False
    for i, k in all_cmds:
        for tok in cmd_tokens(k):
            if not CHECKNAME.search(tok):
                continue
            base = tok.lstrip("./")
            for w, wi in check_written.items():
                if (w.endswith(base) or base.endswith(os.path.basename(w))) and i > wi:
                    ran_check = True
        if ran_check:
            break
    read_back = False
    for p, wi in req_written.items():
        for i, k in all_cmds:
            if i <= wi or p not in k:
                continue
            if READVERB.search(k) or OPENREAD.search(k):
                tg, _ = write_targets(k)
                if p not in [x.strip("'\"").rstrip("/") for x in tg]:
                    read_back = True
                    break
        if read_back:
            break
    assert_or_diff = any(ASSERTX.search(k) or DIFFX.search(k) for _, k in all_cmds)

    ctx_death = int("ContextLength" in (f["exception_type"] or "") or "ContextBudget" in (f["exception_type"] or ""))
    f.update({
        "turns": len(turns), "n_commands": len(all_cmds), "no_cmd_steps": no_cmd,
        "think_turns": think_turns, "declared_done": int(declared),
        "obs_steps": obs_n, "json_reject_steps": json_reject, "preamble_steps": preamble,
        "edit_mode": "in_place" if "repo" in kinds_any else ("patch_file" if "patch" in kinds_any else
                     ("tmp_only" if "tmp" in kinds_any else "no_write")),
        "n_edit_cmds": n_edit_cmds, "first_edit_turn": first_edit, "n_repo_targets": len(repo_targets),
        "used_sed_i": int(used_sed_i), "used_apply": int(used_apply),
        "ran_repo_suite": int(ran_suite), "ran_repro_script": int(ran_repro),
        "max_repeat_run": best, "rm_repo_cmds": rm_repo,
        "wrote_check_script": int(bool(check_written)), "ran_check_script": int(ran_check),
        "read_back_edited": int(read_back), "assert_or_diff": int(assert_or_diff),
        "ctx_death": ctx_death,
        # the eight Pareto axes, all oriented so 1 is the behaviour we want
        "self_check": int(ran_check or (read_back and assert_or_diff)),
        "ran_test_after_edit": int(ran_after),
        "in_place": int("repo" in kinds_any),
        "no_sed_patch": int(not (used_sed_i or used_apply or "patch" in kinds_any)),
        "no_repeat3": int(best < 3),
        # the one continuous axis: fraction of observed turns the harness did NOT reject. Binary would be constant-0
        # here (most trials have at least one rejection), which is why it is a rate.
        "no_json_reject": (1.0 - json_reject / float(obs_n)) if obs_n else 1.0,
        "no_input_delete": int(rm_repo == 0),
        "no_ctx_death": int(not ctx_death),
    })
    return f


# ---------------------------------------------------------------- trial discovery
def iter_trials(run_dirs):
    """Yield (task, cand, trial_dir) for every probe trial under the given run dirs.

    Probe trials live at <run>/<run>/trace_jobs/eval_sessions/<session>/<task>-p<cand>__<hash>/; the sibling
    <run>/<run>/trace_jobs/<task>__<hash>/ dirs are TRAINING rollouts and are excluded by requiring eval_sessions."""
    seen = set()
    for r in run_dirs:
        pats = [os.path.join(r, os.path.basename(r.rstrip("/")), "trace_jobs", "eval_sessions", "*", "*__*"),
                os.path.join(r, "trace_jobs", "eval_sessions", "*", "*__*")]
        hits = []
        for p in pats:
            hits += glob.glob(p)
        if not hits:  # last resort, the p2o_adherence.py recursive form
            hits = glob.glob(os.path.join(r, "**", "eval_sessions", "*", "*__*"), recursive=True)
        for td in sorted(hits):
            if td in seen:
                continue
            seen.add(td)
            name = os.path.basename(td).split("__", 1)[0]
            m = re.match(r"^(.*)-p([A-Za-z0-9]+)$", name)
            task, cand = (m.group(1), m.group(2)) if m else (name, "ctl")
            yield task, cand, td


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None
