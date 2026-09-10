import re, sys, json
R, mf, pf = sys.argv[1:4]; txt = open(mf).read(); prof = {}
for l in open(pf):
    try: r = json.loads(l); prof[str(r["step"])] = r
    except Exception: pass
print(f"== {R}")
for b in re.split(r"^=== ", txt, flags=re.M):
    if not b.startswith("kind=train"): continue
    step = re.search(r"step=(\d+)", b).group(1)
    def g(k):
        m = re.search(re.escape(k) + r": ([-0-9.e+]+)", b); return m.group(1) if m else "?"
    p = prof.get(step, {}); pr = lambda k: p.get(k, "?")
    print(f"  step {step}: rew={g('reward/avg_raw_reward')} pass8={g('reward/avg_pass_at_8')} ent={g('policy/policy_entropy')} "
          f"lr={g('policy/tis/log_ratio_abs_mean')} masked={g('generate/num_masked_trajectories')} rej={g('async/rejected_count/fully_masked')} "
          f"grad={g('policy/raw_grad_norm')} step_s={g('timing/step')} | turns={pr('turns_p50')}/{pr('turns_p90')} tok/turn={pr('tok_per_turn_p50')} "
          f"reason={pr('reasoning_share')} never_edit={pr('never_edit')} tc={pr('task_complete')} pfail={pr('parse_fail_turns')} stop={pr('stop')}")
