#!/usr/bin/env python3
"""Summarize a harbor Terminal-Bench 2 job directory: pass rate, marin's quality gate, and what to rerun.

    python3 summarize_tb2.py <jobs_dir>/<run> [--json]

Reads every trial's result.json. A trial counts as scored when the verifier returned a reward; an AgentTimeoutError
is benign (scored as it stands, marin policy). Everything else with an exception is an infrastructure loss and is
listed by type so it can be rerun with `harbor jobs resume -p <run> -f <ExceptionType>`. The pass rate is the mean
over scored attempts (n_attempts per task), with a per-task mean first, plus a normal 95 % interval over tasks.
marin's gate: valid-complete >= 90 % of attempts and non-benign errors <= 10 %.
"""
import json, sys, math, collections
from pathlib import Path

run = Path(sys.argv[1]); as_json = "--json" in sys.argv
rows = []
for rj in sorted(run.glob("*/result.json")):
    if rj.parent == run: continue
    r = json.load(open(rj))
    task = r.get("task_name") or rj.parent.name.split("__")[0]
    exc = (r.get("exception_info") or {}).get("exception_type")
    rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    rows.append((task, exc, rew))
planned = None
try: planned = len(json.load(open(run / "config.json"))["tasks"]) if (run / "config.json").exists() else None
except Exception: pass
n = len(rows); scored = [(t, w) for t, e, w in rows if w is not None]
benign = sum(1 for t, e, w in rows if e == "AgentTimeoutError" and w is not None)
infra = collections.Counter(e for t, e, w in rows if w is None and e)
by_task = collections.defaultdict(list)
for t, w in scored: by_task[t].append(float(w))
task_means = [sum(v) / len(v) for v in by_task.values()]
pass1 = sum(task_means) / len(task_means) if task_means else float("nan")
se = (math.sqrt(sum((m - pass1) ** 2 for m in task_means) / max(len(task_means) - 1, 1) / max(len(task_means), 1))) if len(task_means) > 1 else float("nan")
out = {"run": str(run), "trials_seen": n, "scored": len(scored), "benign_agent_timeouts": benign,
       "infra_losses": dict(infra), "tasks_scored": len(by_task), "pass@1_mean_of_task_means": round(pass1, 4),
       "ci95_over_tasks": [round(pass1 - 1.96 * se, 4), round(pass1 + 1.96 * se, 4)] if se == se else None,
       "valid_complete_frac": round(len(scored) / n, 3) if n else None,
       "gate_valid_complete_ge_90": (len(scored) / n >= 0.9) if n else None,
       "gate_nonbenign_le_10": ((n - len(scored)) / n <= 0.1) if n else None,
       "tasks_all_zero": sorted(t for t, v in by_task.items() if max(v) == 0),
       "tasks_all_pass": sorted(t for t, v in by_task.items() if min(v) == 1)}
if as_json: print(json.dumps(out, indent=2))
else:
    print(f"{run.name}: {n} trials, {len(scored)} scored ({out['valid_complete_frac']}), {benign} benign agent timeouts, infra {dict(infra)}")
    print(f"pass@1 = {pass1:.3f} over {len(by_task)} tasks  95% CI {out['ci95_over_tasks']}  gate: valid>=90% {out['gate_valid_complete_ge_90']}, nonbenign<=10% {out['gate_nonbenign_le_10']}")
    print(f"all-pass {len(out['tasks_all_pass'])}, all-zero {len(out['tasks_all_zero'])}")
    if infra: print("rerun: harbor jobs resume -p", run, " ".join(f"-f {e}" for e in infra))
