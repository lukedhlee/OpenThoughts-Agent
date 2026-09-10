"""patch_arm.py <rl_config.json> [staleness=3] [epochs=10] — sets trainer.epochs (the trainer runs epochs x len(dataloader) steps, capped by max_steps; the generator leaves epochs=1 which stops a 1,332-task/64-batch arm after ~21 steps — found 2026-09-03 19:12 when overfit_b ended after one step); for the band GRPO arms: set max_staleness_steps, and add the two unlisted
infrastructure classes (ConnectionResetError, BridgeOperationTimeoutError) to harbor.mask_exceptions (audit 2026-09-03),
plus RuntimeError (2026-09-03: harbor raises it only for tmux-session loss - "failed to send batched keys: no server running" /
"batched send/capture produced no markers"; 10/10 sampled exception.txt files were this class. Model-killed tmux is a harness
fragility, not a task failure, so the sample leaves the group instead of scoring 0 and teaching the policy to avoid C-c),
and switch the advantage estimator to rloo_n (masked samples leave the baseline; zero-variance groups filtered) with group_advantage_min_size=4 (Luke, 09-03)."""
import json, sys, re
cp = sys.argv[1]; st = sys.argv[2] if len(sys.argv) > 2 else "3"; ep = sys.argv[3] if len(sys.argv) > 3 else "10"
c = json.load(open(cp)); out = []; done = set()
for a in c["skyrl_hydra_args"]:
    if a.startswith("trainer.fully_async.max_staleness_steps="): a = f"trainer.fully_async.max_staleness_steps={st}"; done.add("st")
    elif a.startswith("++terminal_bench_config.harbor.mask_exceptions="):
        lst = json.loads(a.split("=", 1)[1])
        for x in ("ConnectionResetError", "BridgeOperationTimeoutError", "RuntimeError", "TmuxSessionLostError"):
            if x not in lst: lst.append(x)
        a = "++terminal_bench_config.harbor.mask_exceptions=" + json.dumps(lst, separators=(",", ":")); done.add("mask")
    elif a.startswith("trainer.algorithm.advantage_estimator="): a = "trainer.algorithm.advantage_estimator=rloo_n"; done.add("est")
    elif a.startswith("trainer.epochs="): a = f"trainer.epochs={ep}"; done.add("ep")
    elif a.startswith("trainer.algorithm.group_advantage_min_size="): continue
    elif a.startswith("++terminal_bench_config.harbor.verifier_override_timeout_sec="): a = "++terminal_bench_config.harbor.verifier_override_timeout_sec=600"; done.add("vt")
    out.append(a)
out.append("trainer.algorithm.group_advantage_min_size=4")
assert done == {"st", "mask", "est", "vt", "ep"}, done
c["skyrl_hydra_args"] = out; json.dump(c, open(cp, "w"), indent=2)
print("patched:", [a for a in out if "staleness" in a or "advantage" in a or "verifier_override" in a or "mask_exceptions" in a or "eval_before" in a or "placement" in a or "n_concurrent" in a or a.startswith("trainer.epochs=")])
