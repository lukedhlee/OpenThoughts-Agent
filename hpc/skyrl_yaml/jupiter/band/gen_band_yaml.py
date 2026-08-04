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
    # PROVEN PATH, deliberately not main_tbench_generate.
    # main_tbench_generate is unexercised in this fork: on smoke 1235333 the
    # AsyncVLLMInferenceEngine actors crash-looped inside
    # create_ray_wrapped_inference_engines (GPUs held 4/4 in placement groups, a
    # fresh ~1-minute-old engine process on every inspection, driver parked in
    # ray.get forever) with no traceback in the worker logs. main_tbench is the
    # path running the 35B right now, so the probe rides it instead.
    #
    # lr = 0 is what turns a trainer into a probe: updates fire (which flushes the
    # generation buffer and lets the run walk the whole shard), but the weights
    # never move, so every rollout comes from the base policy and p@4 stays a
    # measurement OF THAT MODEL. Without it, later steps would sample a drifted
    # policy and the band would be measured against a moving target.
    c["entrypoint"] = "examples.terminal_bench.entrypoints.main_tbench"

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
    # main_tbench DOES create a policy worker, so keep the working config's
    # placement: 1 policy node (FSDP4) + 1 rollout node per shard.
    c["trainer"]["placement"]["policy_strict_spread_pg"] = True
    TBS = 32
    c["trainer"]["train_batch_size"] = TBS
    c["trainer"]["policy_mini_batch_size"] = TBS   # fully-async asserts equality
    # Walk the entire shard: ceil(tasks / prompts-per-step).
    c["trainer"]["max_steps"] = -(-len(shard) // TBS)
    c["trainer"]["policy"]["optimizer_config"]["lr"] = 0.0
    # use_tis needs sampling_params.logprobs, which is None here, and TIS is
    # meaningless at lr=0 anyway.
    c["trainer"]["algorithm"]["use_tis"] = False
    c["trainer"]["eval_before_train"] = False
    c["trainer"]["eval_interval"] = 999999
    c["trainer"]["ckpt_interval"] = 999999
    c["trainer"]["hf_save_interval"] = 999999
    c["trainer"]["resume_mode"] = "none"
    c["trainer"]["project_name"] = "jupiter-r2egym-band-probe-8b"

    g = c["generator"]
    # Drop custom_chat_template_chat_completion_path. The 8B config points it at
    # ./chat_templates/qwen3_thinking_acc.jinja2 relative to skyrl-train, and NO
    # chat_templates directory exists anywhere in the MarinSkyRL checkout -- vLLM
    # engine init then stalls with no error in the job log (smoke 1235178 sat 26
    # min at "Executing command with srun" and never served /health). g1_8b is a
    # Qwen3-8B finetune and carries its own template in tokenizer_config.json,
    # which is what the 35B canary relies on too.
    g.get("engine_init_kwargs", {}).pop("custom_chat_template_chat_completion_path", None)
    # REQUIRED for OpenCode, per the canary: OpenCode always sends
    # tool_choice="auto", and this checkpoint's template emits Qwen3 XML tool
    # calls, which vLLM 0.22 parses with qwen3_coder. WITHOUT BOTH FIELDS vLLM
    # REJECTS EVERY AGENT REQUEST BEFORE GENERATION -- it is part of the rollout
    # transport contract, not a tuning knob.
    g.setdefault("engine_init_kwargs", {})["enable_auto_tool_choice"] = True
    g.setdefault("engine_init_kwargs", {})["tool_call_parser"] = "qwen3_coder"
    g["n_samples_per_prompt"] = 4          # p@4 -- the band filter itself
    g["num_inference_engines"] = 4 * (NODES - 1)  # rollout nodes only; node 1 is policy
    # 8B in bf16 is ~16GB of a 96GB GH200, so KV has room the 35B never had.
    # This is the lever that lifts concurrency from 32 to the hundreds.
    g["max_num_seqs"] = 64

    # ===== PROVEN AGENT BLOCK, lifted from the 35B canary =====
    # terminus-2 could not be made to work through this tunnel tonight: after
    # fixing overlay staging and endpoint-by-IP, every trial still died with
    # "Request timed out." while a standalone completion on the same port took
    # 10.5s (434 tokens, thinking off, HTTP 200) and /tokenize + /v1/models both
    # 404'd in ~1ms. No short completion timeout is set anywhere in harbor/llms.
    # OpenCode 1.18.8 is the configuration that ran for HOURS against this exact
    # bridge and tunnel on the 35B, so the probe adopts it verbatim rather than
    # continuing to debug an unproven agent unattended.
    import yaml as _y, os as _os
    _hb = _y.safe_load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "canary_harbor.yaml")))
    c["terminal_bench"]["harbor"] = _hb
    tb = c["terminal_bench"]["harbor"]
    tb["bridge_url"] = "http://10.128.1.2:9921"
    # 32 per shard = 256 total. The canary ran 32 concurrent trials for hours;
    # do not exceed a figure this stack has actually sustained.
    tb["n_concurrent_trials"] = 32
    tb["fail_on_infrastructure_error"] = False
    if "AddTestsDirError" not in tb.get("mask_exceptions", []):
        tb.setdefault("mask_exceptions", []).append("AddTestsDirError")
    # client_window_tokens 20480 comes with the canary budget: it pulls OpenCode's
    # proactive compaction threshold far enough below the 28,672 prompt ceiling
    # that one large tool observation still lands.
    c["context_budget"] = _y.safe_load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "canary_ctx.yaml")))

    e = c["container"]["extra_env"]
    # Each shard gets its own reverse-forward port so shards can run concurrently
    # without fighting over one tunnel.
    # IP LITERAL, NOT the hostname. The reverse forward binds ONLY 10.14.0.46
    # (jrlogin05i's internal iface). Inside the Apptainer sandbox, curl resolves
    # jrlogin05i to that address and gets 200, but PYTHON's getaddrinfo resolves it
    # (via `search jureca`) to a different address and gets ECONNREFUSED. Measured
    # in one sandbox, seconds apart:
    #   curl   http://jrlogin05i:PORT/health   -> 200
    #   python http://jrlogin05i:PORT/health   -> [Errno 111] Connection refused
    #   python http://10.14.0.46:PORT/health   -> 200
    # There is no IPv6 record, so this is A-record selection, not v6-first.
    # It never bit the 35B canary because OpenCode is Node and its resolver picks
    # the working address; terminus-2 is Python/httpx and does not. Cost of missing
    # it: 512 trials, every one APIConnectionError with a null reward.
    e["SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST"] = "10.14.0.46"
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
