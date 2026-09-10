#!/usr/bin/env python3
"""step_table.py <log> [last_n] — one row per WANDB_MIRROR train step: step time, generation wait, vLLM per-turn latency,
throughput, running requests, bridge lag. Compares seat settings across runs."""
import json, re, sys
keys = ["timing/step", "timing/wait_for_generation_buffer", "timing/run_training", "vllm/latency_e2e_mean", "vllm/median_generation_throughput",
        "vllm/median_running_reqs", "vllm/peak_running_reqs", "vllm/num_engines", "vllm/total_finished_requests", "generate/avg_num_tokens",
        "inference_bridge/event_loop_lag_seconds/p95", "reward/avg_raw_reward", "policy/policy_entropy", "async/rejected_count"]
rows = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"WANDB_MIRROR kind=train step=(\d+) metrics=(\{.*\})", line)
    if m:
        try: d = json.loads(m.group(2)); rows.append((int(m.group(1)), d))
        except Exception: pass
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
print("step  t_step  wait  train  e2e/turn  tok/s(med)  run(med)  run(peak)  eng  finished_req  avg_tok  bridge_p95  reward  ent   rej")
for s, d in rows[-n:]:
    v = [d.get(k) for k in keys]
    print(f"{s:4d} {v[0]:7.0f} {v[1]:5.0f} {v[2]:6.0f} {v[3]:8.1f} {v[4]:10.0f} {v[5]:8.1f} {v[6]:9.0f} {v[7]:4.0f} {v[8]:12.0f} {v[9]:8.0f} {v[10]:10.2f} {v[11]:6.3f} {v[12]:5.3f} {v[13]:4.0f}")
