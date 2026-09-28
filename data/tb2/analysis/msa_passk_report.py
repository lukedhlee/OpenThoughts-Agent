#!/usr/bin/env python3
"""Statistics and condensed per-trial views for Terminal-Bench 2 runs of mini-swe-agent-host in TOOL mode.

    python3 msa_passk_report.py --runs <job_dir> [<job_dir> ...] --tasks <tb2 task tree> --tokenizer <tokenizer.json> \
        --out <dir> [--views <dir>] [--attempts 4] [--exclude train-fasttext] [--context 65536] [--head 30 --tail 12]

The mini-swe counterpart of tb2_passk_report.py (Terminus-2): the pass@k statistics, buckets, outcome classes, verification
rule and think-loop rule are imported from it unchanged, so the two reports compare line by line. Inputs are local copies of
harbor job dirs (per trial: result.json, attempts/NNN/agent/{trajectory.json, mini-swe-agent.trajectory.json},
attempts/NNN/verifier/). Every job dir is a slice of one study: trials are pooled by task, and a task with fewer scored or
unscored trials than --attempts (minus nothing for tasks in --exclude, which count as infra:not_run) is filled with
infra:not_run rows.

What mini-swe tool mode records, and how it is read:
- One model call per ATIF agent step (trajectory.json), in order, with the raw sampled text (think span included, because the
  Grug serve runs no reasoning parser), prompt/completion tokens and finish_reason. mini-swe-agent's own message list
  (mini-swe-agent.trajectory.json) says what each call became: an assistant message with actions (then one tool message per
  executed action, with a timestamp), or a FormatError user message (the reply is dropped; nothing ran).
- Parse failure kinds: ctx_cut (finish_reason length with prompt + completion at the served window: the context filled, not
  a format mistake), length (other truncation), ended_in_thinking, no_tool_call, bad_json, unclosed_call, bad_call (unknown
  tool / missing command / no name), other.
- Outcome (tb2_passk_report's classes plus mini-swe's own ends): pass; fail_declared_complete (submitted, reward 0);
  fail_timeout; fail_context (ContextLengthExceededError); fail_format_exit_ctx (3 format errors in a row, the last one a
  ctx_cut: the context filled); fail_format_exit (3 format errors in a row otherwise); fail_other:<type>; infra:<type>.
- Time: generation = agent wall time not spent executing commands; execution = from an assistant reply to its last tool
  result (mini-swe executes after the reply is parsed). A trailing gap counts as execution only if the last reply's actions
  never returned.
- Thinking re-fed: for two successive parsed calls with nothing between them but tool results, the next prompt must grow by
  at least 90 % of the previous reply's completion tokens (the reply, reasoning included, is re-rendered into the prompt).
Stdlib + tokenizers.
"""
import argparse, collections, glob, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tpr", os.path.join(HERE, "..", "tb2_passk_report.py"))
tpr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tpr)
START, END = tpr.START, tpr.END
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MODEL_ERRORS = dict(tpr.MODEL_ERRORS)


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def one(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def headtail(s, h, t):
    s = re.sub(r"\n\s*\n+", "\n", s or "").strip()
    return s if len(s) <= h + t + 20 else s[:h] + f" [...{len(s) - h - t} ch...] " + s[-t:]


def fmt_kind(msg_content, raw, finish, pt, ct, context):
    if finish == "length":
        return "ctx_cut" if (pt or 0) + (ct or 0) >= context - 32 else "length"
    _, _, unclosed = tpr.think_spans(raw)
    if unclosed:
        return "ended_in_thinking"
    m = re.search(r"<error>\s*(.*?)\s*</error>", msg_content or "", re.S)
    err = m.group(1) if m else (msg_content or "")
    if "No tool calls found" in err:
        return "no_tool_call"
    if "not valid JSON" in err or "Error parsing tool call arguments" in err:
        return "bad_json"
    if "opened but never closed" in err:
        return "unclosed_call"
    if "Unknown tool" in err or "Missing 'command'" in err or 'has no "name"' in err:
        return "bad_call"
    return "other"


def trial_calls(adir, context):
    """-> (calls, messages, exit_extra). One record per model call, joined from both trajectory files."""
    atif = load(os.path.join(adir, "agent", "trajectory.json")) or {"steps": []}
    up = load(os.path.join(adir, "agent", "mini-swe-agent.trajectory.json")) or {"messages": []}
    steps = [s for s in atif.get("steps", []) if s.get("source") == "agent"]
    msgs = up.get("messages", [])
    exit_extra = (msgs[-1].get("extra") or {}) if msgs and msgs[-1].get("role") == "exit" else {}
    calls, i = [], 2
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "exit":
            i += 1; continue
        if m.get("role") == "assistant":
            tools = []; i += 1
            while i < len(msgs) and msgs[i].get("role") == "tool":
                tools.append(msgs[i]); i += 1
            calls.append({"ok": True, "msg": m, "tools": tools})
        elif (m.get("extra") or {}).get("interrupt_type") == "FormatError":
            calls.append({"ok": False, "msg": m, "tools": []}); i += 1
        else:
            calls.append({"ok": None, "msg": m, "tools": []}); i += 1   # unexpected role: kept, flagged
    aligned = len(calls) == len(steps)
    for k, c in enumerate(calls):
        st = steps[k] if k < len(steps) else {}
        mt = st.get("metrics") or {}
        c["raw"] = st.get("message") or (c["msg"].get("extra") or {}).get("model_response") or ""
        c["pt"], c["ct"] = mt.get("prompt_tokens"), mt.get("completion_tokens")
        c["finish"] = (st.get("extra") or {}).get("finish_reason")
        c["i"] = k + 1
        if c["ok"]:
            c["actions"] = [a.get("command", "") for a in ((c["msg"].get("extra") or {}).get("actions") or [])]
            c["ts"] = (c["msg"].get("extra") or {}).get("timestamp")
            c["tool_ts"] = [(t.get("extra") or {}).get("timestamp") for t in c["tools"]]
            c["rcs"] = [(t.get("extra") or {}).get("returncode") for t in c["tools"]]
            c["timeouts"] = sum(1 for t in c["tools"] if "timed out after" in ((t.get("extra") or {}).get("exception_info") or ""))
            c["submit"] = any(a.strip() == f"echo {SUBMIT}" or (SUBMIT in a and a.strip().startswith("echo")) for a in c["actions"])
        else:
            c["actions"], c["ts"], c["tool_ts"], c["rcs"], c["timeouts"], c["submit"] = [], None, [], [], 0, False
            c["kind"] = fmt_kind(c["msg"].get("content"), c["raw"], c["finish"], c["pt"], c["ct"], context)
    return calls, msgs, exit_extra, aligned


def time_split(calls, t0, t1):
    """(generation_s, execution_s) from mini-swe's timestamps; unix seconds; t0/t1 datetimes."""
    if not t0 or not t1:
        return None, None
    a, b = t0.timestamp(), t1.timestamp()
    execu, prev = 0.0, a
    for c in calls:
        if not c["ok"] or c["ts"] is None:
            continue
        prev = max(prev, c["ts"])
        for tt in c["tool_ts"]:
            if tt is not None:
                execu += max(0.0, tt - prev); prev = max(prev, tt)
    last = next((c for c in reversed(calls) if c["ok"] is not None), None)
    if last and last["ok"] and last["actions"] and len(last["tool_ts"]) < len(last["actions"]) and not last["submit"]:
        execu += max(0.0, b - prev)
    total = max(0.0, b - a)
    return max(0.0, total - execu), min(execu, total)


def load_trial(tdir, run, tok, instr, a, want_view):
    r = load(os.path.join(tdir, "result.json")) or {}
    task = r.get("task_name") or os.path.basename(tdir).split("__")[0]
    cands = sorted(glob.glob(os.path.join(tdir, "attempts", "*")))
    adir = cands[-1] if cands else tdir
    exc = (r.get("exception_info") or {}).get("exception_type")
    vr = r.get("verifier_result") or {}
    reward = (vr.get("rewards") or {}).get("reward")
    ar = r.get("agent_result") or {}; md = ar.get("metadata") or {}
    budget = ((r.get("config") or {}).get("resolved_timeouts") or {}).get("agent") or a.budget
    ae = r.get("agent_execution") or {}
    t0, t1 = tpr.ts(ae.get("started_at")), tpr.ts(ae.get("finished_at"))
    agent_s = (t1 - t0).total_seconds() if t0 and t1 else None
    calls, msgs, exit_extra, aligned = trial_calls(adir, a.context)
    exit_status = exit_extra.get("exit_status") or md.get("exit_status")
    # per call: thinking, loops, re-feed
    think_tot = comp_tot = 0; think_per = []; loops = unclosed = 0; max_ctx = 0
    for c in calls:
        th, nsp, unc = tpr.think_spans(c["raw"])
        ids = tok.encode(th, add_special_tokens=False).ids if th else []
        c["think"], c["think_tok"], c["unclosed"] = th, len(ids), unc
        c["loop"] = (tpr.ngram_max_repeat(ids) >= 3) if len(ids) >= 150 else False
        c["answer"] = c["raw"].split(END, 1)[1] if END in c["raw"] else ("" if unc else c["raw"])
        loops += c["loop"]; unclosed += bool(unc and c["finish"] == "length")
        think_tot += len(ids); comp_tot += c["ct"] or 0; think_per.append(len(ids))
        max_ctx = max(max_ctx, (c["pt"] or 0) + (c["ct"] or 0))
    refeed_checked = refeed_ok = 0
    for c1, c2 in zip(calls, calls[1:]):
        if c1["ok"] and c2["ok"] is not None and c1["tools"] and c1["ct"] and c1["pt"] is not None and c2["pt"] is not None:
            refeed_checked += 1; refeed_ok += (c2["pt"] - c1["pt"]) >= 0.9 * c1["ct"]
    reasoning_kept = sum(1 for c in calls if c["ok"] and c["think_tok"] > 0 and (c["msg"].get("reasoning_content") or "").strip())
    reasoning_expected = sum(1 for c in calls if c["ok"] and c["think_tok"] > 0)
    # parse / execution
    ok = [c for c in calls if c["ok"]]; bad = [c for c in calls if c["ok"] is False]
    kinds = collections.Counter(c["kind"] for c in bad)
    executed_calls = sum(1 for c in ok if c["actions"] and (len(c["tools"]) == len(c["actions"]) or c["submit"]))
    # verification before the completion claim (tb2_passk_report.classify on each executed action)
    flat = []   # (call idx, action idx, is_edit, is_verify)
    for c in ok:
        for j, cmd in enumerate(c["actions"][:len(c["tools"])] if not c["submit"] else c["actions"]):
            if SUBMIT in cmd:
                continue
            e, v = tpr.classify(cmd); flat.append((c["i"], j, e, v))
    claim = next((c["i"] for c in ok if c["submit"]), None)
    verified = edits_before = None
    if claim is not None:
        before = [x for x in flat if x[0] <= claim]
        last_edit = max(((t, j) for t, j, e, v in before if e), default=None)
        edits_before = last_edit is not None
        verified = any(v for t, j, e, v in before if last_edit is None or (t, j) > last_edit)
    # outcome
    submitted = exit_status == "Submitted"
    last_bad = bad[-1] if bad else None
    if reward is None:
        outcome = f"infra:{exc or 'no_reward'}"
    elif reward >= 1:
        outcome = "pass"
    elif exc in MODEL_ERRORS:
        outcome = MODEL_ERRORS[exc]
    elif exc:
        outcome = f"fail_other:{exc}"
    elif submitted:
        outcome = "fail_declared_complete"
    elif exit_status == "RepeatedFormatError":
        outcome = "fail_format_exit_ctx" if (last_bad and last_bad["kind"] == "ctx_cut") else "fail_format_exit"
    else:
        outcome = f"fail_other:{exit_status or md.get('stop_reason') or 'no_claim'}"
    gen_s, exec_s = time_split(calls, t0, t1)
    n_calls = len(calls)
    row = {
        "run": run, "trial": os.path.basename(tdir), "task": task, "reward": reward, "exception": exc, "outcome": outcome,
        "exit_status": exit_status, "aligned": aligned, "n_attempts": len(cands),
        "declared_complete": claim is not None, "claim_call": claim, "verified_before_claim": verified,
        "edits_before_claim": edits_before, "turns": n_calls, "parsed_calls": len(ok), "parse_error_turns": len(bad),
        "parse_fail_rate": len(bad) / n_calls if n_calls else None,
        "parse_fail_rate_excl_ctx": (len(bad) - kinds.get("ctx_cut", 0)) / (n_calls - kinds.get("ctx_cut", 0)) if n_calls - kinds.get("ctx_cut", 0) else None,
        "executed_calls": executed_calls, "parse_kinds": dict(kinds), "n_commands": len(flat),
        "multi_action_calls": sum(1 for c in ok if len(c["actions"]) > 1), "command_timeouts": sum(c["timeouts"] for c in ok),
        "n_edit_cmds": sum(1 for x in flat if x[2]), "n_verify_cmds": sum(1 for x in flat if x[3]),
        "n_input_tokens": ar.get("n_input_tokens"), "n_output_tokens": ar.get("n_output_tokens"),
        "total_tokens": (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0) if ar else None,
        "gen_tokens_main": comp_tot, "max_context": max_ctx, "think_tokens": think_tot,
        "think_share": think_tot / comp_tot if comp_tot else None,
        "think_per_turn_mean": tpr.mean(think_per), "think_per_turn_median": tpr.median(think_per),
        "think_per_turn_max": max(think_per) if think_per else None,
        "think_loop_turns": loops, "unclosed_think_turns": sum(1 for c in calls if c["unclosed"]),
        "unclosed_at_cap_turns": unclosed, "maxtok_turns": sum(1 for c in calls if c["finish"] == "length"),
        "summarizations": 0, "stop_reason": md.get("stop_reason"), "agent_s": agent_s, "budget_s": budget,
        "budget_used": (agent_s / budget) if (agent_s and budget) else None, "gen_s": gen_s, "exec_s": exec_s,
        "gen_share": (gen_s / agent_s) if (agent_s and gen_s is not None) else None,
        "exec_share": (exec_s / agent_s) if (agent_s and exec_s is not None) else None,
        "refeed_checked": refeed_checked, "refeed_ok": refeed_ok,
        "reasoning_kept": reasoning_kept, "reasoning_expected": reasoning_expected,
        "verifier_s": tpr.dur(r.get("verifier")), "setup_s": tpr.dur(r.get("environment_setup")),
    }
    row["_t0"] = t0.timestamp() if t0 else None
    row["think_flag"] = bool(loops or unclosed)
    row["thinking_consumed_budget"] = (outcome in ("fail_timeout", "fail_context", "fail_format_exit_ctx")) and bool(
        ((row["gen_share"] or 0) > 0.5 and (row["think_share"] or 0) > 0.5) or row["think_flag"])
    vout = vr.get("stdout") if isinstance(vr.get("stdout"), str) else None
    vp = os.path.join(adir, "verifier", "test-stdout.txt")
    if vout is None and os.path.exists(vp):
        vout = open(vp, errors="replace").read()
    vout = vout or ""
    row["verifier_head"], row["verifier_tail"] = vout[:800], vout[-800:] if len(vout) > 1600 else ""
    view = render(row, calls, instr.get(task, ""), vout, adir, a) if want_view else None
    return row, view


def render(row, calls, instruction, vout, adir, a):
    L = []
    L.append(f"=== {row['task']} | {row['trial']} | reward {row['reward']} | outcome {row['outcome']} | exit {row['exit_status']} | exc {row['exception']}")
    L.append(f"budget {row['budget_s']}s, agent ran {round(row['agent_s'] or 0)}s (generation {round(row['gen_s'] or 0)}s = "
             f"{100 * (row['gen_share'] or 0):.0f} %, execution {round(row['exec_s'] or 0)}s) | calls {row['turns']} (parsed "
             f"{row['parsed_calls']}, parse errors {row['parse_error_turns']} {row['parse_kinds']}) | commands {row['n_commands']}, "
             f"timeouts {row['command_timeouts']}")
    L.append(f"tokens: out {row['gen_tokens_main']}, think {row['think_tokens']} ({100 * (row['think_share'] or 0):.0f} %), "
             f"think/call median {row['think_per_turn_median']} max {row['think_per_turn_max']}, max context {row['max_context']} | "
             f"think loops {row['think_loop_turns']}, unclosed at cap {row['unclosed_at_cap_turns']} | thinking re-fed "
             f"{row['refeed_ok']}/{row['refeed_checked']}")
    L.append(f"claim at call {row['claim_call']} | verified after last edit, before claim: {row['verified_before_claim']} "
             f"(edits before claim: {row['edits_before_claim']})")
    if instruction:
        L.append("--- INSTRUCTION ---")
        ins = re.sub(r"\n\s*\n+", "\n", instruction).strip()
        L.append(ins[:4000] + (" [...]" if len(ins) > 4000 else ""))
    L.append("--- CALLS (C#, +elapsed s, ctx = prompt tokens, out = completion tokens, think = think tokens; $ = executed command, "
             "> = output [rc]) ---")
    n = len(calls); detail = set(range(1, a.head + 1)) | set(range(n - a.tail + 1, n + 1))
    if row["claim_call"]:
        detail |= set(range(row["claim_call"] - 2, row["claim_call"] + 1))
    t0 = row.get("_t0") or min((c["ts"] for c in calls if c["ts"]), default=0.0)
    k = 0
    while k < n:
        c = calls[k]
        el = f"+{c['ts'] - t0:.0f}s " if c["ts"] else ""
        if c["i"] not in detail:
            if c["ok"]:
                outs = []
                for t, rc in zip(c["tools"], c["rcs"]):
                    raw = (t.get("extra") or {}).get("raw_output") or ""
                    last = [ln for ln in raw.strip().splitlines() if ln.strip()][-1:] or [""]
                    outs.append(f"[{rc}] " + one(last[0], 90))
                L.append(f"C{c['i']:03d} {el}out {c['ct']} think {c['think_tok']}" + (" LOOP" if c["loop"] else "") + ": "
                         + " ; ".join("$ " + one(x, 90) for x in c["actions"]) + " | > " + " ; ".join(outs))
                k += 1; continue
            j = k; streak = []
            while j < n and calls[j]["ok"] is False and calls[j]["i"] not in detail:
                streak.append(calls[j]); j += 1
            if not streak:
                streak, j = [c], k + 1
            L.append(f"C{streak[0]['i']:03d}-C{streak[-1]['i']:03d} {len(streak)} PARSE-ERR "
                     f"{dict(collections.Counter(s['kind'] for s in streak))} out {sum(s['ct'] or 0 for s in streak)}")
            k = j; continue
        hdr = f"C{c['i']:03d} {el}ctx {c['pt']} out {c['ct']} think {c['think_tok']}" + (" LOOP" if c["loop"] else "")
        if c["ok"]:
            L.append(hdr + (" SUBMIT" if c["submit"] else ""))
            if c["think"]:
                L.append("   think: " + one(headtail(c["think"], int(500 * a.scale), int(300 * a.scale)), 10 ** 6))
            ans = re.sub(r"<tool_call>.*?</tool_call>", "", c["answer"], flags=re.S).strip()
            if ans:
                L.append("   says: " + one(ans, int(300 * a.scale)))
            for x, t, rc in zip(c["actions"], c["tools"] + [None] * len(c["actions"]), c["rcs"] + [None] * len(c["actions"])):
                L.append("   $ " + one(x, int(300 * a.scale)))
                if t is not None:
                    raw = (t.get("extra") or {}).get("raw_output") or ""
                    exi = (t.get("extra") or {}).get("exception_info") or ""
                    L.append(f"   > [{rc}] " + headtail(raw, int(250 * a.scale), int(350 * a.scale)).replace("\n", "\n     ")
                             + (f"  !! {one(exi, 120)}" if exi else ""))
        else:
            L.append(hdr + f" PARSE-ERR[{c['kind']}] finish {c['finish']}")
            if c["think"]:
                L.append("   think: " + one(headtail(c["think"], int(400 * a.scale), int(300 * a.scale)), 10 ** 6))
            if c["answer"].strip():
                L.append("   reply: " + one(c["answer"], int(400 * a.scale)))
        k += 1
    L.append("--- VERIFIER ---")
    ctrf = load(os.path.join(adir, "verifier", "ctrf.json")) or {}
    tests = ((ctrf.get("results") or {}).get("tests")) or []
    fails = [(x.get("name", "").split("::")[-1], one(x.get("message") or "", 160)) for x in tests if x.get("status") != "passed"]
    tsum = (ctrf.get("results") or {}).get("summary") or {}
    L.append(f"   tests {tsum.get('passed')}/{tsum.get('tests')} passed")
    for nm, m in fails[:12]:
        L.append(f"   FAIL {nm}: {m}")
    tail = [ln for ln in (vout or "").splitlines() if re.search(r"(passed|failed|error|Error|assert|FAILED|PASSED)", ln)]
    if tail:
        L.append("   stdout: " + " | ".join(one(x, 200) for x in tail[-8:]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True); ap.add_argument("--views")
    ap.add_argument("--attempts", type=int, default=1); ap.add_argument("--exclude", default="train-fasttext")
    ap.add_argument("--only-tasks", default="", help="comma list: restrict the task universe (smoke)")
    ap.add_argument("--budget", type=float, default=1800.0); ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--head", type=int, default=30); ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--max-out", type=int, default=65536)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    all_tasks = sorted(d for d in os.listdir(a.tasks) if os.path.isfile(os.path.join(a.tasks, d, "task.toml")))
    if a.only_tasks:
        all_tasks = [t for t in all_tasks if t in a.only_tasks.split(",")]
    instr = {t: open(os.path.join(a.tasks, t, "instruction.md"), errors="replace").read() for t in all_tasks
             if os.path.exists(os.path.join(a.tasks, t, "instruction.md"))}
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run in a.runs:
        rname = os.path.basename(run.rstrip("/"))
        for tdir in sorted(glob.glob(os.path.join(run, "*__*"))):
            if not os.path.exists(os.path.join(tdir, "result.json")):
                continue
            row, view = load_trial(tdir, rname, tok, instr, a, bool(a.views))
            if row["task"] not in all_tasks:
                continue
            rows.append(row)
            if view:
                os.makedirs(a.views, exist_ok=True)
                with open(os.path.join(a.views, f"{row['trial']}.txt"), "w") as f:
                    f.write(view)
    have = collections.Counter(r["task"] for r in rows)
    for t in all_tasks:   # planned trials that never produced a result (excluded or never ran) are infra losses
        for _ in range(max(0, a.attempts - have.get(t, 0))):
            rows.append({"run": "planned", "trial": None, "task": t, "reward": None, "exception": "not_run", "outcome": "infra:not_run"})
    with open(os.path.join(a.out, "per_trial.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tpr.summarize(rows, all_tasks, a)
    # mini-swe specific totals
    full = [r for r in rows if r.get("trial")]
    tot_calls = sum(r["turns"] for r in full); tot_bad = sum(r["parse_error_turns"] for r in full)
    kinds = collections.Counter()
    for r in full:
        kinds.update(r["parse_kinds"])
    ctx = kinds.get("ctx_cut", 0)
    extra = {
        "trials": len(full), "calls": tot_calls, "parse_errors": tot_bad,
        "parse_rate": 1 - tot_bad / tot_calls if tot_calls else None,
        "parse_rate_excl_ctx_cut": 1 - (tot_bad - ctx) / (tot_calls - ctx) if tot_calls - ctx else None,
        "executed_rate": sum(r["executed_calls"] for r in full) / tot_calls if tot_calls else None,
        "parse_kinds": dict(kinds),
        "per_trial_parse_rate_median": tpr.median([1 - r["parse_fail_rate"] for r in full if r["parse_fail_rate"] is not None]),
        "refeed": [sum(r["refeed_ok"] for r in full), sum(r["refeed_checked"] for r in full)],
        "reasoning_kept": [sum(r["reasoning_kept"] for r in full), sum(r["reasoning_expected"] for r in full)],
        "misaligned_trials": [r["trial"] for r in full if not r["aligned"]],
        "exit_status": dict(collections.Counter(r["exit_status"] for r in full)),
        "command_timeouts": sum(r["command_timeouts"] for r in full),
        "gen_share_median": tpr.median([r["gen_share"] for r in full]),
        "think_share_median": tpr.median([r["think_share"] for r in full]),
        "think_loop_trials": sum(1 for r in full if r["think_loop_turns"]),
        "unclosed_at_cap_trials": sum(1 for r in full if r["unclosed_at_cap_turns"]),
    }
    s = json.load(open(os.path.join(a.out, "summary.json"))); s["mini_swe"] = extra
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    print(json.dumps(extra, indent=1, default=str))


if __name__ == "__main__":
    main()
