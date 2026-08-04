#!/usr/bin/env python3
"""Generate learnable-band p@4 probe configs from the 8B r2egym GRPO config.

The probe is PURE ROLLOUT: `main_tbench_generate` builds vLLM engines and calls
generator.generate() once over every prompt. No policy/ref model, no optimizer,
no FSDP, no weight sync, no checkpoint -- which removes most of the failure
chain that has cost us nine runs. `policy_strict_spread_pg` is forced off so
get_policy_pg() returns None and no GPUs are reserved away from the engines.

Sharded because generate() has NO resume: one shard = one allocation's worth of
work, and a shard that dies only loses its own tasks. Completed trials are
already durable per-trial as trace_jobs/<task>/result.json, so a re-run skips
nothing but costs only the lost shard.

usage: gen_band_yaml.py <out_dir> <task_names_file> <tasks_root> <n_shards> \
           <rollout_nodes> <endpoint_port_base> [shard_index_to_emit ...]
"""
import sys, os, copy, yaml

OUT, NAMES, ROOT, NSHARD, NODES, PORTBASE = sys.argv[1:7]
NSHARD, NODES, PORTBASE = int(NSHARD), int(NODES), int(PORTBASE)
only = set(int(x) for x in sys.argv[7:]) if len(sys.argv) > 7 else None

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base8b.yaml")
with open(SRC) as f:
    base = yaml.safe_load(f)

names = [l.strip() for l in open(NAMES) if l.strip()]
shards = [names[i::NSHARD] for i in range(NSHARD)]  # round-robin: even mix of envs

FSCRATCH = "/e/fscratch/reformo/lee27"

for i, shard in enumerate(shards):
    if only is not None and i not in only:
        continue
    c = copy.deepcopy(base)
    c["entrypoint"] = "examples.terminal_bench.entrypoints.main_tbench_generate"

    # The launcher accepts exactly ONE context declaration and derives the seven
    # downstream fields itself; declaring any of them is a hard error. These values
    # reproduce the 8B config's previous behaviour exactly:
    #   max_input_tokens = 32768 - 4096 = 28672
    # client_window_tokens is omitted deliberately: it exists to pull OpenCode's
    # proactive compaction threshold down below the prompt ceiling, and terminus-2
    # does not compact.
    c["context_budget"] = {
        "request_window_tokens": 32768,
        "max_new_tokens_per_turn": 4096,
        "max_turns": 999999,
    }
    for path in (("trainer", "max_prompt_length"),
                 ("generator", "max_turns"),
                 ("generator", "sampling_params", "max_generate_length"),
                 ("generator", "engine_init_kwargs", "max_model_len"),
                 ("terminal_bench", "harbor", "max_episodes"),
                 ("terminal_bench", "model_info", "max_input_tokens"),
                 ("terminal_bench", "model_info", "max_output_tokens")):
        node = c
        for k in path[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)

    m = f"{FSCRATCH}/models/g1_diverse_tezos_100k_8b"
    c["trainer"]["policy"]["model"]["path"] = m
    c["trainer"]["ref"]["model"]["path"] = m
    # No policy placement group: the generate path has no policy worker, and a
    # strict-spread PG would reserve whole nodes away from the vLLM engines.
    c["trainer"]["placement"]["policy_strict_spread_pg"] = False
    c["trainer"]["train_batch_size"] = min(32, len(shard))
    c["trainer"]["policy_mini_batch_size"] = min(32, len(shard))
    c["trainer"]["eval_before_train"] = False
    c["trainer"]["eval_interval"] = 999999
    c["trainer"]["ckpt_interval"] = 999999
    c["trainer"]["hf_save_interval"] = 999999
    c["trainer"]["resume_mode"] = "none"
    c["trainer"]["project_name"] = "jupiter-r2egym-band-probe-8b"

    g = c["generator"]
    g["n_samples_per_prompt"] = 4          # p@4 -- the band filter itself
    g["num_inference_engines"] = 4 * NODES  # TP1, one engine per GH200
    # 8B in bf16 is ~16GB of a 96GB GH200, so KV has room the 35B never had.
    # This is the lever that lifts concurrency from 32 to the hundreds.
    g["max_num_seqs"] = 64

    tb = c["terminal_bench"]["harbor"]
    # Hardcode, do NOT leave as ${oc.env:APPTAINER_BRIDGE_URL,...:9920}: if the
    # env var is unset at hydra-resolution time the default silently attaches the
    # probe to the 9920 bridge serving the 35B run. Same accepted-but-ignored
    # failure class as strict_json_parser / compaction.reserved.
    tb["bridge_url"] = "http://10.128.1.2:9921"
    # Sandbox supply is 48/node on the JURECA dc-cpu fleet; keep the driver's
    # concurrency at the engine working set so trials are not queued behind vLLM.
    tb["n_concurrent_trials"] = 4 * NODES * 16
    # Learned on 1229488: one unenumerated exception aborted 68 honest trials.
    tb["fail_on_infrastructure_error"] = False
    if "AddTestsDirError" not in tb["mask_exceptions"]:
        tb["mask_exceptions"].append("AddTestsDirError")
    if "NonZeroAgentExitCodeError" not in tb["passthrough_exceptions"]:
        tb["passthrough_exceptions"].append("NonZeroAgentExitCodeError")

    e = c["container"]["extra_env"]
    # Each shard gets its own reverse-forward port so shards can run concurrently
    # without fighting over one tunnel.
    e["SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST"] = "jrlogin05i"
    e["SKYRL_ROLLOUT_HTTP_ENDPOINT_PORT"] = str(PORTBASE + i)
    # Probe bridge, NOT the 9920 bridge serving the 35B milestone run.
    e["APPTAINER_BRIDGE_URL"] = "http://10.128.1.2:9921"
    # hpc.py points this at /e/data1/.../ot-baf/experiments/_ray_logs, another
    # project's tree that we cannot write: ray_utils._start_node mkdir's it and
    # dies with EACCES 52s into the job. /tmp is node-local, which is correct --
    # Ray logs are per-node anyway.
    e["OT_AGENT_RAY_LOG_DIR"] = "/tmp/ray_logs"
    # MUST be set. The default is 600s, which is SHORTER than the 1800s agent
    # cap, so the bridge kills the exec mid-task. Measured on the 35B canary:
    # 76% of trials cut off mid-task, which flattens reward to 0 and would make
    # a saturated-looking band out of tasks the model never got to finish.
    e["BRIDGE_EXEC_TIMEOUT"] = "2100"
    e["WANDB_DIR"] = f"{FSCRATCH}/wandb"
    e["WANDB_MODE"] = "offline"
    # /e/scratch is inode-exhausted and cannot create files; it killed 1229643
    # via exactly this path.
    e["RAY_object_spilling_config"] = (
        '{"type":"filesystem","params":{"directory_path":"%s/ray_spill"}}' % FSCRATCH
    )

    c["data"]["train_data"] = [os.path.join(ROOT, n) for n in shard]
    c["data"]["val_data"] = []

    p = os.path.join(OUT, f"band_probe_8b_p4_shard{i:02d}of{NSHARD:02d}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False, width=10**6)
    print(f"{p}  tasks={len(shard)} trials={len(shard)*4} engines={4*NODES} conc={tb['n_concurrent_trials']} port={PORTBASE+i}")
