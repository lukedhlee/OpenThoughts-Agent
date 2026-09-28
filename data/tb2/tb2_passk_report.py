#!/usr/bin/env python3
"""Mechanical statistics and per-trial digests for a Terminal-Bench 2 pass@k study (n harbor runs of the same checkpoint).

    python3 tb2_passk_report.py --runs <job_dir> [<job_dir> ...] --tasks <tb2 task tree> --tokenizer <tokenizer.json> \
        --parser <harbor>/src/harbor/agents/terminus_2/terminus_json_plain_parser.py --out <dir> [--digests <dir>] \
        [--expected-tasks 89] [--budget 1800]

Inputs are local copies of harbor job dirs (per trial: result.json, attempts/*/agent/trajectory*.json, verifier/test-stdout.txt)
from runs whose agent kwargs set trajectory_config.raw_content=true, so every turn's raw completion (think span included) is in
trajectory.json. Commands are re-parsed from that raw text with harbor's own Terminus JSON parser (the file given by --parser),
which is what the agent executed; a turn whose parse had an ERROR executed nothing.

Writes to --out: per_trial.jsonl, per_task.csv, summary.json and tables.md. With --digests, one JSONL per task (one line per
trial): instruction head, then per turn the context tokens, think tokens and a 300-char think head, the commands, a 400-char
head+tail of the terminal output and any completion claim; final outcome and verifier output.

Definitions (also printed in tables.md):
- outcome: pass (reward 1); fail_declared_complete (reward 0, the agent ended by a confirmed task_complete); fail_timeout
  (AgentTimeoutError); fail_context (ContextLengthExceededError); fail_turncap (TurnCapExhaustedError); fail_nonzero_exit
  (NonZeroAgentExitCodeError); fail_other:<type>; infra:<type> (no reward: the verifier never scored it) or infra:not_run.
- verification before the claim: a run/test/check command (tests, running a script or binary, compiling, diff/cmp/curl/test)
  executed after the last file-editing command and no later than the first completion claim. Heredoc bodies are ignored.
- think loop: within one turn's think span, some 50-token window occurs 3 or more times; plus unclosed think span, and
  turns that generated >= max_output (8,192) tokens or had a truncated attempt re-asked inside the step. harbor 761fb516's
  chat path does not send max_tokens, so single responses can run past 8,192 (seen up to 62k); the 8,192 is the
  policy's nominal per-turn cap and harbor's budget arithmetic, not a hard stop.
- wall shares: gen_share = sum of harbor's per-request LLM latencies / agent wall time (the rest is command execution and
  harness overhead). think_wall_share = gen_share x (think + truncated-overflow tokens / generated tokens);
  degen_wall_share = gen_share x (tokens of degenerate turns / generated tokens), where a degenerate turn has a think loop,
  an unclosed think span, >= 8,192 generated tokens or a truncated attempt. Both assume decode time ~ tokens.
- thinking plausibly consumed the budget (failures only): the trial failed by timeout or context, and more than half of the
  agent's wall went to generating think tokens (think_wall_share > 0.5) or more than a third went to degenerate turns
  (degen_wall_share > 1/3).
"""
import argparse, collections, csv, glob, importlib.util, json, math, os, random, re, statistics, sys
from datetime import datetime

START, END = "<|start_think|>", "<|end_think|>"
MODEL_ERRORS = {"AgentTimeoutError": "fail_timeout", "ContextLengthExceededError": "fail_context",
                "TurnCapExhaustedError": "fail_turncap", "NonZeroAgentExitCodeError": "fail_nonzero_exit"}

# ---------------------------------------------------------------- command classification (heuristic, stated in the docstring)
# A keystroke string is cut into command segments (newlines, &&, ||, ;, |; heredoc bodies dropped) and each segment is judged
# by its first word. edit = writes/deletes files as text: cat/echo/printf/tee into a file (redirect or heredoc), sed -i /
# perl -i, patch / git apply, editors, cp/mv/rm/touch/ln/install, python that writes files (open(..., 'w'), write_text).
# verify = runs code or checks: test runners, an interpreter on a script / -c / -m module (not pip or venv), ./x or /abs/x
# at command position, compilers and build tools, diff/cmp/curl/jq/sha256sum/md5sum, test and [ ]. Pure inspection
# (cat, head, tail, ls, grep, wc, stat, find, sed -n) is neither.
HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
SPLIT = re.compile(r"\n|&&|\|\||;|\|")
PREFIX = re.compile(r"^(\s*(sudo|time|nohup|env|timeout\s+\S+|[A-Za-z_][A-Za-z0-9_]*=\S*)\s+)+")
INTERP = re.compile(r"^(python[0-9.]*|node|ruby|Rscript|julia|java|php|lua|ocaml|runghc|swift|deno|bun|tclsh|sqlite3|perl|octave)$")
RUNNERS = {"pytest", "py.test", "tox", "ctest", "nosetests", "make", "cmake", "ninja", "cargo", "go", "mvn", "gradle", "dotnet",
           "npm", "yarn", "pnpm", "gcc", "g++", "clang", "clang++", "cc", "rustc", "javac", "tsc", "mypy", "ruff", "flake8",
           "pylint", "shellcheck", "gfortran", "nvcc", "diff", "cmp", "curl", "jq", "sha256sum", "md5sum", "test", "[", "[["}
EDITORS = {"vi", "vim", "nano", "emacs", "ed", "patch", "cp", "mv", "rm", "touch", "ln", "install", "tee", "truncate", "dd"}
REDIR = re.compile(r"(?<![0-9&<>])>>?\s*(?!&|/dev/null)[\w./~\"'$-]")


def strip_heredocs(ks):
    out, lines, i, bodies = [], ks.split("\n"), 0, []
    while i < len(lines):
        line = lines[i]; out.append(line); m = HEREDOC.search(line); i += 1
        if m:
            tag, body = m.group(1), []
            while i < len(lines) and lines[i].strip() != tag: body.append(lines[i]); i += 1
            i += 1; bodies.append("\n".join(body))
    return "\n".join(out), bodies


def classify(ks):
    s, bodies = strip_heredocs(ks)
    is_edit = is_verify = False
    for seg in SPLIT.split(s):
        seg = PREFIX.sub("", seg.strip())
        if not seg: continue
        w = seg.split()[0]; rest = seg[len(w):].strip()
        if w in ("cat", "echo", "printf") and (REDIR.search(seg) or ("<<" in seg and ">" in seg)): is_edit = True
        elif w in ("sed", "perl") and re.search(r"(^|\s)-[A-Za-z]*i", rest): is_edit = True
        elif w in EDITORS or (w == "git" and re.match(r"(apply|checkout\s+--|restore)\b", rest)): is_edit = True
        if INTERP.match(w) and not re.match(r"-m\s+(pip|venv|ensurepip)\b|--version|-V\b", rest) and rest[:1] not in ("",): is_verify = True
        elif INTERP.match(w) and "<<" in seg: is_verify = True
        elif w in ("bash", "sh", "zsh") and re.match(r"[\w./~-]+\.(sh|bash)\b", rest): is_verify = True
        elif w in RUNNERS and not (w in ("make", "cargo", "go", "npm", "yarn", "pnpm") and re.match(r"(install|--version|-v\b)", rest)): is_verify = True
        elif re.match(r"(\./|/)[\w./-]+$", w) and not w.endswith("/"): is_verify = True
    if re.search(r"(^|[\s;&|(])python[0-9.]*\s", ks) and re.search(r"open\([^)]*['\"][wa]b?['\"]|\.write_text\(|\.write_bytes\(", ks):
        is_edit = True
    return is_edit, is_verify


# ---------------------------------------------------------------- helpers
def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def dur(block):
    if not block or not block.get("started_at") or not block.get("finished_at"): return None
    return (ts(block["finished_at"]) - ts(block["started_at"])).total_seconds()


def head_tail(s, n=400):
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + " [...] " + s[-n // 2:]


def think_spans(text):
    """-> (think text joined, n_spans, unclosed)"""
    text = text or ""; spans, pos, unclosed = [], 0, False
    while True:
        a = text.find(START, pos)
        if a < 0: break
        b = text.find(END, a + len(START))
        if b < 0: spans.append(text[a + len(START):]); unclosed = True; break
        spans.append(text[a + len(START):b]); pos = b + len(END)
    return "\n".join(spans), len(spans), unclosed


def ngram_max_repeat(ids, n=50):
    if len(ids) < n * 3: return 0
    c = collections.Counter(hash(tuple(ids[i:i + n])) for i in range(0, len(ids) - n + 1))
    return max(c.values())


def mean(x): x = [v for v in x if v is not None]; return sum(x) / len(x) if x else None
def median(x): x = [v for v in x if v is not None]; return statistics.median(x) if x else None


def boot_ci(values, stat, B=10000, seed=0):
    rng = random.Random(seed); n = len(values); out = []
    for _ in range(B):
        out.append(stat([values[rng.randrange(n)] for _ in range(n)]))
    out.sort(); return out[int(0.025 * B)], out[int(0.975 * B) - 1]


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pass_at_k(n, c, k):
    if n < k: return None
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------- one trial
def load_trial(tdir, run, parser, tok, instr, max_out, budget_default, want_digest):
    rj = os.path.join(tdir, "result.json")
    r = json.load(open(rj))
    task = r.get("task_name") or os.path.basename(tdir).split("__")[0]
    rel = r.get("trial_relpath") or ""
    adir = os.path.join(os.path.dirname(tdir), rel) if rel and os.path.isdir(os.path.join(os.path.dirname(tdir), rel)) else None
    if adir is None:
        cands = sorted(glob.glob(os.path.join(tdir, "attempts", "*")))
        adir = cands[-1] if cands else tdir
    n_attempts = len(glob.glob(os.path.join(tdir, "attempts", "*")))
    exc = (r.get("exception_info") or {}).get("exception_type")
    vr = r.get("verifier_result") or {}
    reward = (vr.get("rewards") or {}).get("reward")
    ar = r.get("agent_result") or {}; md = ar.get("metadata") or {}
    budget = ((r.get("config") or {}).get("resolved_timeouts") or {}).get("agent") or budget_default
    agent_s = dur(r.get("agent_execution")); gen_s = sum(md.get("api_request_times_msec") or []) / 1000.0
    # trajectory
    traj_p = os.path.join(adir, "agent", "trajectory.json")
    steps = json.load(open(traj_p)).get("steps", []) if os.path.exists(traj_p) else []
    n_summ = len({m.group(1) for f in glob.glob(os.path.join(adir, "agent", "trajectory.summarization-*"))
                  for m in [re.search(r"summarization-(\d+)", f)] if m})
    turns, cmds_flat, claims = [], [], []
    think_tok_total = comp_total = overflow_total = degen_tokens = 0; think_per_turn = []; loop_turns = unclosed_turns = maxtok_turns = 0
    max_ctx = 0; seen_heads = set(); dup_think = 0; parse_errors = 0; toolcall_err = 0; pending_summary = False
    for st in steps:
        if st.get("source") != "agent":
            if turns and st.get("source") == "user": pending_summary = True
            continue
        msg = st.get("message") or ""
        if not isinstance(msg, str): msg = json.dumps(msg)
        m = st.get("metrics") or {}; pt = m.get("prompt_tokens") or 0; ct = m.get("completion_tokens") or 0
        max_ctx = max(max_ctx, pt); comp_total += ct   # input context of the turn (a step with several LLM calls sums them; rare)
        think, nspan, unclosed = think_spans(msg)
        tt = len(tok.encode(think, add_special_tokens=False).ids) if think else 0
        think_tok_total += tt; think_per_turn.append(tt)
        rep = ngram_max_repeat(tok.encode(think, add_special_tokens=False).ids) if tt >= 150 else 0
        msg_tok = len(tok.encode(msg, add_special_tokens=False).ids)
        overflow = max(0, ct - msg_tok - 32)   # tokens of truncated attempts harbor re-asked within this step (not in the record)
        overflow_total += overflow
        looped = rep >= 3; maxtok = ct >= max_out - 1 or overflow >= max_out // 2
        loop_turns += looped; unclosed_turns += unclosed; maxtok_turns += maxtok
        if looped or unclosed or maxtok or overflow > 0: degen_tokens += ct
        hd = re.sub(r"\s+", " ", think[:300]).strip()
        if hd and hd in seen_heads: dup_think += 1
        seen_heads.add(hd)
        obs = "\n".join(x.get("content") or "" for x in ((st.get("observation") or {}).get("results") or []) if isinstance(x.get("content"), str))
        err = obs.startswith("Previous response had parsing errors")
        pr = parser.parse_response(msg)
        executed = [] if (err or pr.error) else [c.keystrokes for c in pr.commands]
        claim = bool(pr.is_task_complete) and not (err or pr.error)
        if st.get("tool_calls"):   # runs without raw_content: harbor recorded the parsed actions itself
            tc = st["tool_calls"]; pr.error = ""
            executed = [c["arguments"].get("keystrokes", "") for c in tc if c.get("function_name") == "bash_command"]
            claim = any(c.get("function_name") == "mark_task_complete" for c in tc)
        parse_errors += bool(err or pr.error); toolcall_err += bool((err or pr.error) and "<tool_call>" in msg)
        ti = len(turns)
        for j, ks in enumerate(executed):
            e, v = classify(ks); cmds_flat.append((ti, j, e, v))
        if claim: claims.append(ti)
        turns.append({"i": ti, "ctx": pt, "out": ct, "think_tok": tt, "think_spans": nspan, "unclosed": unclosed,
                      "think_loop": looped, "maxtok": maxtok, "claim": claim, "parse_error": bool(err or pr.error),
                      "summarized_before": pending_summary,
                      **({"think_head": think[:300], "commands": [head_tail(k, 600) for k in executed],
                          "output": (re.search(r"ERROR:[^\n]*", obs).group(0)[:160] if (err and re.search(r"ERROR:[^\n]*", obs)) else head_tail(obs, 400)),
                          "answer_head": (msg.split(END, 1)[1] if END in msg else ("" if unclosed else msg))[:200] if (err or pr.error) else None}
                         if want_digest else {})})
        pending_summary = False
    # verification before the first claim
    verified = None; edits_before = None
    if claims:
        c0 = claims[0]; before = [x for x in cmds_flat if x[0] <= c0]
        last_edit = max(((t, j) for t, j, e, v in before if e), default=None)
        edits_before = last_edit is not None
        verified = any(v for t, j, e, v in before if last_edit is None or (t, j) > last_edit)
    ended_by_claim = bool(claims) and exc is None and len(claims) >= 2 and claims[-1] == len(turns) - 1
    if md.get("stop_reason"): ended_by_claim = md["stop_reason"] == "task_complete"   # harbor's own record, when present
    if reward is None:
        outcome = f"infra:{exc or 'no_reward'}"
    elif reward >= 1:
        outcome = "pass"
    elif exc in MODEL_ERRORS:
        outcome = MODEL_ERRORS[exc]
    elif exc:
        outcome = f"fail_other:{exc}"
    elif ended_by_claim:
        outcome = "fail_declared_complete"
    else:
        sr = md.get("stop_reason")
        outcome = "fail_session_ended" if sr == "session_ended" else f"fail_other:{sr or 'no_claim'}"
    think_share = think_tok_total / comp_total if comp_total else None
    vout = vr.get("stdout") if isinstance(vr.get("stdout"), str) else None
    vp = os.path.join(adir, "verifier", "test-stdout.txt")
    if vout is None and os.path.exists(vp): vout = open(vp, errors="replace").read()
    vout = vout or ""
    k = max(vout.find("test session starts"), vout.find("=== RUN"), vout.find("PASSED"), vout.find("FAILED"), -1)
    vstart = max(0, vout.rfind("\n", 0, max(k - 200, 0)) if k > 0 else 0)
    row = {"run": run, "trial": os.path.basename(tdir), "task": task, "reward": reward, "exception": exc, "outcome": outcome,
           "n_attempts": n_attempts, "declared_complete": bool(claims), "ended_by_claim": ended_by_claim,
           "n_claims": len(claims), "verified_before_claim": verified, "edits_before_claim": edits_before,
           "turns": len(turns), "parse_error_turns": parse_errors, "parse_error_share": parse_errors / len(turns) if turns else None,
           "toolcall_format_error_turns": toolcall_err, "n_commands": len(cmds_flat),
           "n_edit_cmds": sum(1 for x in cmds_flat if x[2]), "n_verify_cmds": sum(1 for x in cmds_flat if x[3]), "n_input_tokens": ar.get("n_input_tokens"),
           "n_output_tokens": ar.get("n_output_tokens"),
           "total_tokens": (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0) if ar else None,
           "gen_tokens_main": comp_total, "max_context": max_ctx, "think_tokens": think_tok_total, "think_share": think_share,
           "overflow_tokens": overflow_total, "think_or_overflow_share": ((think_tok_total + overflow_total) / comp_total) if comp_total else None,
           "think_per_turn_mean": mean(think_per_turn), "think_per_turn_max": max(think_per_turn) if think_per_turn else None,
           "think_loop_turns": loop_turns, "unclosed_think_turns": unclosed_turns, "maxtok_turns": maxtok_turns,
           "dup_think_turns": dup_think, "summarizations": md.get("summarization_count", n_summ),
           "stop_reason": md.get("stop_reason"), "agent_s": agent_s, "budget_s": budget,
           "budget_used": (agent_s / budget) if (agent_s and budget) else None, "gen_s": gen_s,
           "gen_share": (gen_s / agent_s) if agent_s else None,
           "verifier_s": dur(r.get("verifier")), "setup_s": dur(r.get("environment_setup")),
           "verifier_head": vout[vstart:vstart + 800], "verifier_tail": vout[-800:] if len(vout) > vstart + 1600 else ""}
    row["think_flag"] = bool(loop_turns or unclosed_turns or maxtok_turns)
    row["degen_token_share"] = degen_tokens / comp_total if comp_total else None
    g = row["gen_share"] or 0
    row["think_wall_share"] = g * (row["think_or_overflow_share"] or 0)   # ~share of the agent's wall spent generating think/overflow tokens
    row["degen_wall_share"] = g * (row["degen_token_share"] or 0)       # ~share spent generating loop / unclosed / runaway turns
    row["unclosed_turn_frac"] = unclosed_turns / len(turns) if turns else None
    row["thinking_consumed_budget"] = (outcome in ("fail_timeout", "fail_context")) and (
        row["think_wall_share"] > 0.5 or row["degen_wall_share"] > 1 / 3)
    digest = None
    if want_digest:
        digest = {"task": task, "run": run, "trial": row["trial"], "instruction": (instr.get(task) or "")[:1500],
                  "turns": turns, "final": {k2: row[k2] for k2 in ("outcome", "reward", "exception", "declared_complete",
                  "verified_before_claim", "turns", "max_context", "think_share", "summarizations", "agent_s", "gen_share")},
                  "verifier_head": row["verifier_head"], "verifier_tail": row["verifier_tail"]}
    return row, digest


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--parser", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--digests"); ap.add_argument("--expected-tasks", type=int, default=89)
    ap.add_argument("--budget", type=float, default=1800.0); ap.add_argument("--max-out", type=int, default=8192)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    spec = importlib.util.spec_from_file_location("tjp", a.parser); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    parser = mod.TerminusJSONPlainParser()
    all_tasks = sorted(d for d in os.listdir(a.tasks) if os.path.isfile(os.path.join(a.tasks, d, "task.toml")))
    instr = {t: open(os.path.join(a.tasks, t, "instruction.md"), errors="replace").read() for t in all_tasks
             if os.path.exists(os.path.join(a.tasks, t, "instruction.md"))}
    os.makedirs(a.out, exist_ok=True)
    rows, digests = [], collections.defaultdict(list)
    for run in a.runs:
        rname = os.path.basename(run.rstrip("/"))
        chosen = {}   # task -> (row, digest); a <run>_rec recovery dir replaces this run's unscored trial of the same task
        for src in [run.rstrip("/"), run.rstrip("/") + "_rec"]:
            for tdir in sorted(glob.glob(os.path.join(src, "*__*"))):
                if not os.path.exists(os.path.join(tdir, "result.json")): continue
                row, dg = load_trial(tdir, rname, parser, tok, instr, a.max_out, a.budget, bool(a.digests))
                prev = chosen.get(row["task"])
                if src.endswith("_rec"):
                    if prev and prev[0].get("reward") is not None: continue
                    row["recovered_from"] = prev[0]["exception"] if prev else "not_run"
                    if dg: dg["recovered_from"] = row["recovered_from"]
                elif prev and prev[0].get("reward") is not None:
                    continue
                chosen[row["task"]] = (row, dg)
        for t, (row, dg) in sorted(chosen.items()):
            rows.append(row)
            if dg: digests[t].append(dg)
        seen = {r["task"] for r in rows if r["run"] == rname}
        for t in all_tasks:   # planned trials that never produced a result (excluded or never ran) are infra losses
            if t not in seen:
                rows.append({"run": rname, "trial": None, "task": t, "reward": None, "exception": "not_run", "outcome": "infra:not_run"})
    with open(os.path.join(a.out, "per_trial.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    if a.digests:
        os.makedirs(a.digests, exist_ok=True)
        for t, lst in digests.items():
            with open(os.path.join(a.digests, f"{t}.jsonl"), "w") as f:
                for d in sorted(lst, key=lambda d: d["run"]): f.write(json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n")
    summarize(rows, all_tasks, a)


def summarize(rows, all_tasks, a):
    scored = [r for r in rows if r.get("reward") is not None]
    by_task = collections.defaultdict(list)
    for r in scored: by_task[r["task"]].append(1 if r["reward"] >= 1 else 0)
    per_task = []
    for t in all_tasks:
        v = by_task.get(t, []); k, n = sum(v), len(v)
        infra = [r["outcome"] for r in rows if r["task"] == t and r.get("reward") is None]
        bucket = None if n == 0 else ("always-pass" if k == n else "always-fail" if k == 0 else "sometimes-pass")
        per_task.append({"task": t, "k": k, "n": n, "bucket": bucket, "infra": len(infra), "infra_types": ";".join(sorted(set(infra)))})
    with open(os.path.join(a.out, "per_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0].keys())); w.writeheader(); w.writerows(per_task)
    scored_tasks = [p for p in per_task if p["n"] > 0]
    means = [p["k"] / p["n"] for p in scored_tasks]
    p1 = mean(means); p1_ci = boot_ci(means, lambda x: sum(x) / len(x)) if means else (None, None)
    full = [p for p in scored_tasks if p["n"] >= 4]
    p4_vals = [pass_at_k(p["n"], p["k"], 4) for p in full]
    p4 = mean(p4_vals); p4_ci = boot_ci(p4_vals, lambda x: sum(x) / len(x)) if p4_vals else (None, None)
    anypass = sum(1 for p in scored_tasks if p["k"] > 0)
    buckets = collections.Counter(p["bucket"] for p in scored_tasks)
    outc = collections.Counter(r["outcome"] for r in rows)
    n_planned = len(rows); n_infra = sum(1 for r in rows if r.get("reward") is None)
    full_rows = [r for r in rows if r.get("trial")]
    # flags
    flags = []
    for p in per_task:
        tr = [r for r in rows if r["task"] == p["task"]]
        why = []
        if p["n"] == 0: why.append("no scored trial (all infra)")
        elif p["infra"] >= 2: why.append(f"{p['infra']} of {p['infra'] + p['n']} trials unscored ({p['infra_types']})")
        vt = [r for r in tr if (r.get("exception") or "").startswith("Verifier") or (r.get("exception") or "") in ("RewardFileNotFoundError", "RewardFileEmptyError")]
        if vt: why.append(f"verifier errors x{len(vt)} ({', '.join(sorted({r['exception'] for r in vt}))})")
        lazy = [r for r in tr if r.get("reward") == 1 and (r.get("turns") or 0) <= 2]
        noedit = [r for r in tr if r.get("reward") == 1 and (r.get("edits_before_claim") is False or r.get("n_edit_cmds") == 0)]
        if lazy: why.append(f"pass in <=2 turns x{len(lazy)}")
        if noedit: why.append(f"pass with no file-editing command (before the claim, or at all) x{len(noedit)}")
        vnoise = [r for r in tr if re.search(r"Temporary failure in name resolution|Could not resolve host|429 Too Many|No space left|Read timed out", (r.get("verifier_head") or "") + (r.get("verifier_tail") or ""))]
        if vnoise: why.append(f"verifier output shows external network or disk errors (DNS, 429, ENOSPC, read timeout) x{len(vnoise)}")
        if p["task"] == "train-fasttext": why.append("known: Daytona image gone and org snapshot cap full (not run)")
        if p["task"] == "filter-js-from-html": why.append("known: Selenium verifier can time out at 1 vCPU (tb2_eval.md)")
        if why: flags.append({"task": p["task"], "k": p["k"], "n": p["n"], "why": why})
    # success vs failure
    metrics = ["turns", "total_tokens", "gen_tokens_main", "max_context", "think_share", "think_or_overflow_share", "overflow_tokens", "think_per_turn_mean", "think_per_turn_max",
               "think_wall_share", "degen_wall_share", "degen_token_share", "unclosed_turn_frac", "think_loop_turns",
               "summarizations", "budget_used", "gen_share", "parse_error_turns", "parse_error_share"]
    def stats(rs):
        d = {"n": len(rs)}
        for m in metrics:
            v = [r.get(m) for r in rs if r.get(m) is not None]
            d[m] = {"mean": mean(v), "median": median(v)}
        for lab, f in [("timeout", lambda r: r["outcome"] == "fail_timeout" or r.get("exception") == "AgentTimeoutError"),
                       ("context_exceeded", lambda r: r.get("exception") == "ContextLengthExceededError"),
                       ("summarized", lambda r: (r.get("summarizations") or 0) > 0),
                       ("declared_complete", lambda r: r.get("declared_complete")),
                       ("verified_before_claim", lambda r: r.get("verified_before_claim") is True),
                       ("think_loop", lambda r: (r.get("think_loop_turns") or 0) > 0),
                       ("unclosed_think", lambda r: (r.get("unclosed_think_turns") or 0) > 0),
                       ("maxtok_turn", lambda r: (r.get("maxtok_turns") or 0) > 0)]:
            d[lab + "_frac"] = (sum(1 for r in rs if f(r)) / len(rs)) if rs else None
        return d
    passes = [r for r in full_rows if r.get("reward") == 1]; fails = [r for r in full_rows if r.get("reward") == 0]
    sp_tasks = {p["task"] for p in per_task if p["bucket"] == "sometimes-pass"}
    sp_pass = [r for r in passes if r["task"] in sp_tasks]; sp_fail = [r for r in fails if r["task"] in sp_tasks]
    paired = collections.defaultdict(list)   # within-task pass minus fail, averaged over sometimes-pass tasks
    for t in sp_tasks:
        pr = [r for r in passes if r["task"] == t]; fr = [r for r in fails if r["task"] == t]
        for m in metrics:
            a1 = mean([r.get(m) for r in pr]); b1 = mean([r.get(m) for r in fr])
            if a1 is not None and b1 is not None: paired[m].append(a1 - b1)
    fail_timeouts = [r for r in fails if r["outcome"] == "fail_timeout"]
    fail_declared = [r for r in fails if r["outcome"] == "fail_declared_complete"]
    summary = {
        "n_runs": len({r["run"] for r in rows}), "tasks": len(all_tasks), "planned_trials": n_planned,
        "scored_trials": len(scored), "infra_trials": n_infra, "infra_frac": n_infra / n_planned if n_planned else None,
        "pass@1": p1, "pass@1_ci95_boot_tasks": p1_ci, "tasks_scored": len(scored_tasks),
        "pass@4_unbiased_tasks_n>=4": p4, "pass@4_ci95_boot_tasks": p4_ci, "tasks_with_n>=4": len(full),
        "tasks_any_pass": anypass, "tasks_any_pass_wilson": wilson(anypass, len(scored_tasks)),
        "buckets": dict(buckets), "outcomes": dict(outc), "flags": flags,
        "pass": stats(passes), "fail": stats(fails), "sometimes_pass_pass": stats(sp_pass), "sometimes_pass_fail": stats(sp_fail),
        "sometimes_pass_within_task_diff_mean": {m: mean(v) for m, v in paired.items()},
        "timeout_fails_gen_majority_frac": (sum(1 for r in fail_timeouts if (r.get("gen_share") or 0) > 0.5) / len(fail_timeouts)) if fail_timeouts else None,
        "fails_thinking_consumed_budget": sum(1 for r in fails if r.get("thinking_consumed_budget")), "n_fails": len(fails),
        "timeout_fails": stats(fail_timeouts), "declared_fails": stats(fail_declared),
        "timeout_fails_think_wall_gt_half": sum(1 for r in fail_timeouts if (r.get("think_wall_share") or 0) > 0.5),
        "timeout_fails_degen_wall_gt_third": sum(1 for r in fail_timeouts if (r.get("degen_wall_share") or 0) > 1 / 3),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    write_tables(summary, per_task, a)
    print(json.dumps({k: summary[k] for k in ("pass@1", "pass@1_ci95_boot_tasks", "pass@4_unbiased_tasks_n>=4", "pass@4_ci95_boot_tasks",
                                             "buckets", "outcomes", "infra_frac")}, indent=1, default=str))


def fmt(v, pct=False, nd=1):
    if v is None: return "–"
    if pct: return f"{100 * v:.{nd}f} %"
    if isinstance(v, float): return f"{v:,.{nd}f}" if abs(v) >= 10 else f"{v:.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def write_tables(s, per_task, a):
    L = []
    L.append("| metric (mean / median) | pass | fail | sometimes-pass: pass | sometimes-pass: fail | fail: timeout | fail: declared done |")
    L.append("|---|---|---|---|---|---|---|")
    cols = [s["pass"], s["fail"], s["sometimes_pass_pass"], s["sometimes_pass_fail"], s["timeout_fails"], s["declared_fails"]]
    L.append("| trials | " + " | ".join(str(c["n"]) for c in cols) + " |")
    for m, lab in [("turns", "turns"), ("total_tokens", "total tokens (in+out)"), ("gen_tokens_main", "generated tokens"),
                   ("max_context", "max input context (tokens)"), ("think_share", "think share of generated"), ("think_per_turn_mean", "think tokens / turn"),
                   ("think_per_turn_max", "max think tokens in a turn"), ("summarizations", "summarizations"),
                   ("budget_used", "share of 30-min budget used"), ("gen_share", "share of wall in generation"),
                   ("think_wall_share", "share of wall generating think tokens"), ("degen_wall_share", "share of wall generating degenerate turns"),
                   ("unclosed_turn_frac", "share of turns with an unclosed think span"), ("think_loop_turns", "think-loop turns"),
                   ("parse_error_turns", "parse-error turns"), ("parse_error_share", "share of turns rejected by the parser")]:
        pct = m in ("think_share", "budget_used", "gen_share", "think_wall_share", "degen_wall_share", "unclosed_turn_frac", "parse_error_share")
        L.append(f"| {lab} | " + " | ".join(f"{fmt(c[m]['mean'], pct)} / {fmt(c[m]['median'], pct)}" for c in cols) + " |")
        if m == "think_share":
            L.append("| think + truncated-overflow share | " + " | ".join(f"{fmt(c['think_or_overflow_share']['mean'], True)} / {fmt(c['think_or_overflow_share']['median'], True)}" for c in cols) + " |")
    for m, lab in [("timeout_frac", "hit the 30-min budget"), ("context_exceeded_frac", "context exceeded"), ("summarized_frac", "summarized at least once"),
                   ("declared_complete_frac", "declared task_complete"), ("verified_before_claim_frac", "verified after last edit, before claim"),
                   ("think_loop_frac", "think loop (50-token span x3)"), ("unclosed_think_frac", "unclosed think span"), ("maxtok_turn_frac", "a turn generated >= 8,192 tokens")]:
        L.append(f"| {lab} | " + " | ".join(fmt(c[m], True) for c in cols) + " |")
    L.append(""); L.append("| task | k/n | bucket | infra |"); L.append("|---|---|---|---|")
    for p in sorted(per_task, key=lambda p: (-(p["k"] / p["n"]) if p["n"] else 1, p["task"])):
        L.append(f"| {p['task']} | {p['k']}/{p['n']} | {p['bucket'] or 'unscored'} | {p['infra_types'] or ''} |")
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
