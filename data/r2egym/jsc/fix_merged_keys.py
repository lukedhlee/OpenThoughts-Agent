"""fix_merged_keys.py <rl_config.json> — rewrite hydra args for the MarinSkyRL schema after the 2026-09-03 upstream merge (#485 etc.):
generator.vllm_stats_interval -> generator.inference_stats_interval; rollout.fanout.enabled dropped (process pool is the only path);
rollout.fanout.{num_coordinators,cpus_per_coordinator} -> trajectory_runner.process_pool.*. Idempotent. Session 2189af97, 2026-09-04."""
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = c["skyrl_hydra_args"]; out = []; changed = []
ren = {"generator.vllm_stats_interval=": "generator.inference_stats_interval=",
       "rollout.fanout.num_coordinators=": "trajectory_runner.process_pool.num_coordinators=",
       "rollout.fanout.cpus_per_coordinator=": "trajectory_runner.process_pool.cpus_per_coordinator=",
       "rollout.fanout.coordinator_rpc_timeout=": "trajectory_runner.process_pool.rpc_timeout_seconds=",
       "rollout.fanout.coordinator_executor_workers=": "trajectory_runner.process_pool.executor_workers="}
drop = ("rollout.fanout.enabled=",)
for x in a:
    if x.startswith(drop): changed.append(("drop", x)); continue
    for k, v in ren.items():
        if x.startswith(k): changed.append((x, v + x[len(k):])); x = v + x[len(k):]; break
    out.append(x)
c["skyrl_hydra_args"] = out; json.dump(c, open(p, "w"), indent=2)
print("fix_merged_keys:", changed if changed else "nothing to change")
