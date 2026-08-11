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

# ===== SUBSET / RAMP OVERRIDES =====
# Every one of these defaults to the value the full-band shards already used, so
# with none of them set the emitted YAML is byte-identical to the proven config.
# They exist because the goal is no longer the full band: it is the band on a
# SUBSET, measured cheaply, then extrapolated. Two of them are correctness fixes
# rather than tuning:
#   BAND_TOOL_PARSER{,_PLUGIN} -- g1_diverse_tezos_100k_8b emits tool calls as
#     BARE JSON (182/182 captured steps, zero XML), so the inherited
#     `qwen3_coder` XML parser never matches and never logs; OpenCode then sees
#     plain text, takes one no-op turn, and reward collapses to f(task) with no
#     within-group variance. Measuring a band that way re-derives the retracted
#     0/197. The plugin must be IMPORTED (its @register_module decorator runs),
#     which MarinSkyRL d8bdc79 does in pop_openai_kwargs.
#   BAND_BRIDGE_PORT -- the fleet's workers poll whichever bridge the FLEET was
#     launched against. Pointing the job at the other bridge yields a passing
#     route gate and 100% rollout timeouts. Read the fleet log, do not guess.
ENV_MAX_TASKS = int(os.environ.get("BAND_MAX_TASKS", "0"))       # 0 = whole shard
ENV_CONC = int(os.environ.get("BAND_CONC", "32"))
ENV_MAX_NUM_SEQS = int(os.environ.get("BAND_MAX_NUM_SEQS", "64"))
ENV_BRIDGE_PORT = os.environ.get("BAND_BRIDGE_PORT", "9921")
ENV_TOOL_PARSER = os.environ.get("BAND_TOOL_PARSER", "qwen3_coder")
ENV_TOOL_PARSER_PLUGIN = os.environ.get("BAND_TOOL_PARSER_PLUGIN", "")
ENV_NO_THINK = os.environ.get("BAND_SERVER_NO_THINK", "") not in ("", "0", "false", "False")

# ===== MARIANNA-PARITY OVERRIDES =====
# Values taken from the co-lead's working r2egym script. The point of the subset
# run is to REPRODUCE her band rate (~358 learnable of 4,578 ~= 8% at 0<pass@4<1),
# so anything that changes agent behaviour has to match hers, not ours.
#
#   BAND_MAX_EPISODES  -- she caps agent steps at 50. Our canary_ctx sets
#     max_turns=999999, i.e. UNBOUNDED, so our trials only ever stop by hitting
#     the 1800s wall. Measured: 63 of 64 trials still running at 38 min, 5
#     completions. The episode cap, not thinking, is what bounds trial duration.
#   BAND_CPUS/MEM/STORAGE -- hers are 1 cpu / 1024 MB / 1024 MB against our
#     2 / 4096 / 4096. Lighter sandboxes = far more of them per fleet node.
#   BAND_MAX_MODEL_LEN -- 40960 is g1's NATIVE window (config.json
#     max_position_embeddings). We were serving 32768 and discarding 8k.
#   BAND_TP / BAND_ENGINES -- she runs TP=4 with 8 engines (one engine per
#     4-GPU node) rather than our TP=1 x 4 engines on one node. Same GPUs, but
#     TP=4 shards each sequence's KV across 4 GPUs, which is what makes
#     max_num_seqs=1024 at 40960 context feasible.
ENV_MAX_EPISODES = int(os.environ.get("BAND_MAX_EPISODES", "0"))       # 0 = leave as-is
ENV_CPUS = int(os.environ.get("BAND_CPUS", "0"))
ENV_MEM_MB = int(os.environ.get("BAND_MEM_MB", "0"))
ENV_STORAGE_MB = int(os.environ.get("BAND_STORAGE_MB", "0"))
ENV_MAX_MODEL_LEN = int(os.environ.get("BAND_MAX_MODEL_LEN", "0"))
ENV_TP = int(os.environ.get("BAND_TP", "0"))
ENV_ENGINES = int(os.environ.get("BAND_ENGINES", "0"))
ENV_GMU = float(os.environ.get("BAND_GMU", "0") or 0)
# Thinking ON is the co-lead's configuration (interleaved_thinking=true,
# extra_body.chat_template_kwargs.enable_thinking=true), so band parity wants
# true. These two keys are INERT for OpenCode (harbor implements extra_body for
# terminus_2/openhands/mini_swe_agent only), but leaving them at false while the
# model actually thinks is a lie in the config that will mislead the next reader.
ENV_HARBOR_THINK = os.environ.get("BAND_HARBOR_THINK", "")
# BAND_GENERATE=1 -> TRAINER-FREE sweep via main_tbench_generate: no policy/ref
# model, no optimizer, no weight sync, no lr=0 training-step OOM pass-ender, and
# no batch=mini=workers concurrency pin. Requires MarinSkyRL branch
# lukedhlee/engine-init-batch (SKYRL_ENGINE_INIT_BATCH, ported from
# marianna13/SkyRL b07f04a) — the generate path crash-looped on smoke 1235333
# from concurrent engine-constructor contention, which that knob serializes.
ENV_GENERATE = os.environ.get("BAND_GENERATE", "") not in ("", "0", "false", "False")
ENV_ENGINE_INIT_BATCH = int(os.environ.get("BAND_ENGINE_INIT_BATCH", "4"))
# BAND_LR: 0.0 = frozen probe (the band-census default). Set >0 for a REAL
# training run (pilot GRPO on the band); 8e-6 is Marianna's r2egym value.
ENV_LR = float(os.environ.get("BAND_LR", "0") or 0)
if ENV_MAX_TASKS:
    shards = [s[:ENV_MAX_TASKS] for s in shards]

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
    c["entrypoint"] = (
        "examples.terminal_bench.entrypoints.main_tbench_generate"
        if ENV_GENERATE
        else "examples.terminal_bench.entrypoints.main_tbench"
    )

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
    # generate mode: no policy worker exists, so force the spread PG off —
    # otherwise whole nodes get reserved away from the vLLM engines.
    c["trainer"]["placement"]["policy_strict_spread_pg"] = not ENV_GENERATE
    TBS = 32
    if ENV_GENERATE:
        # main_tbench_generate has no trainer, but get_train_dataset keeps the
        # trainer's `dataset >= train_batch_size` assert — clamp to the shard so
        # small smokes pass. Batch size has no other meaning on this path
        # (generate() takes the whole dataset in one call).
        TBS = min(TBS, len(shard))
    c["trainer"]["train_batch_size"] = TBS
    c["trainer"]["policy_mini_batch_size"] = TBS   # fully-async asserts equality
    # Walk the entire shard: ceil(tasks / prompts-per-step). BAND_MAX_STEPS
    # overrides for multi-epoch training runs (the dataloader cycles).
    ENV_MAX_STEPS = int(os.environ.get("BAND_MAX_STEPS", "0"))
    c["trainer"]["max_steps"] = ENV_MAX_STEPS or -(-len(shard) // TBS)
    c["trainer"]["policy"]["optimizer_config"]["lr"] = ENV_LR
    # use_tis needs sampling_params.logprobs, which is None here, and TIS is
    # meaningless at lr=0 anyway.
    c["trainer"]["algorithm"]["use_tis"] = False
    c["trainer"]["eval_before_train"] = False
    c["trainer"]["eval_interval"] = 999999
    c["trainer"]["ckpt_interval"] = 999999
    # BAND_HF_SAVE_INTERVAL: periodic HF-format checkpoint dumps for real
    # training runs (band census probes never save).
    c["trainer"]["hf_save_interval"] = int(os.environ.get("BAND_HF_SAVE_INTERVAL", "999999"))
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
    g.setdefault("engine_init_kwargs", {})["tool_call_parser"] = ENV_TOOL_PARSER
    if ENV_TOOL_PARSER_PLUGIN:
        g["engine_init_kwargs"]["tool_parser_plugin"] = ENV_TOOL_PARSER_PLUGIN
    # SERVER-SIDE thinking-off. The harbor-level interleaved_thinking /
    # extra_body.chat_template_kwargs settings are implemented for terminus_2,
    # openhands and mini_swe_agent ONLY -- there is no OpenCode path, so for the
    # OpenCode agent they are accepted and ignored, and the model thinks anyway.
    # vLLM's renderer merges this UNDER any request-level chat_template_kwargs,
    # so it is a default rather than an override, and it reaches every request
    # regardless of which agent issued it. Requires MarinSkyRL 9904058.
    if ENV_NO_THINK:
        g["engine_init_kwargs"]["default_chat_template_kwargs"] = {"enable_thinking": False}
    g["n_samples_per_prompt"] = 4          # p@4 -- the band filter itself
    # trainer mode: node 1 is the policy node, rollout nodes are the rest.
    # generate mode: there is no policy node — every node serves engines.
    g["num_inference_engines"] = 4 * NODES if ENV_GENERATE else 4 * (NODES - 1)
    # 8B in bf16 is ~16GB of a 96GB GH200, so KV has room the 35B never had.
    # This is the lever that lifts concurrency from 32 to the hundreds.
    g["max_num_seqs"] = ENV_MAX_NUM_SEQS
    if ENV_TP:
        g["inference_engine_tensor_parallel_size"] = ENV_TP
    if ENV_ENGINES:
        # Overrides the 4*(NODES-1) default: with TP=4 an engine occupies a whole
        # 4-GPU node, so engines == rollout nodes, not 4x rollout nodes.
        g["num_inference_engines"] = ENV_ENGINES
    if ENV_GMU:
        g["gpu_memory_utilization"] = ENV_GMU
    # NOTE: max_model_len is NOT settable here. The launcher accepts exactly ONE
    # context declaration (context_budget) and derives max_model_len,
    # max_prompt_length, max_input_tokens, max_episodes and three more from it;
    # declaring any derived field is a HARD ERROR. The generator pops them for
    # that reason. So the window and the episode cap are set on context_budget
    # below, after canary_ctx is loaded.

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
    tb["bridge_url"] = f"http://10.128.1.2:{ENV_BRIDGE_PORT}"
    # 32 per shard = 256 total. The canary ran 32 concurrent trials for hours;
    # do not exceed a figure this stack has actually sustained.
    tb["n_concurrent_trials"] = ENV_CONC
    tb["fail_on_infrastructure_error"] = False
    if "AddTestsDirError" not in tb.get("mask_exceptions", []):
        tb.setdefault("mask_exceptions", []).append("AddTestsDirError")
    # client_window_tokens 20480 comes with the canary budget: it pulls OpenCode's
    # proactive compaction threshold far enough below the 28,672 prompt ceiling
    # that one large tool observation still lands.
    c["context_budget"] = _y.safe_load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "canary_ctx.yaml")))
    # max_episodes is DERIVED from max_turns, so the episode cap is set here.
    # canary_ctx ships max_turns=999999 (unbounded), which is why our trials only
    # ever terminated on the 1800s wall instead of after a bounded number of steps.
    if ENV_MAX_EPISODES:
        c["context_budget"]["max_turns"] = ENV_MAX_EPISODES
    if ENV_MAX_MODEL_LEN:
        c["context_budget"]["request_window_tokens"] = ENV_MAX_MODEL_LEN
        # client_window_tokens must stay below the derived input ceiling
        # (request_window - max_new_tokens_per_turn) or OpenCode stops compacting
        # early enough for one large tool observation to land.
        newtok = c["context_budget"].get("max_new_tokens_per_turn", 4096)
        c["context_budget"]["client_window_tokens"] = max(8192, ENV_MAX_MODEL_LEN - newtok - 4096)
    if ENV_HARBOR_THINK != "":
        _think = ENV_HARBOR_THINK not in ("0", "false", "False")
        tb["interleaved_thinking"] = _think
        tb.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = _think
    if ENV_CPUS:
        tb["override_cpus"] = ENV_CPUS
    if ENV_MEM_MB:
        tb["override_memory_mb"] = ENV_MEM_MB
    if ENV_STORAGE_MB:
        tb["override_storage_mb"] = ENV_STORAGE_MB

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
    # 10.14.0.46 = JURECA jrlogin05i (fleet 155xxxxx era). For a JUWELS fleet
    # the listeners bind 10.13.0.158 (jwlogin08i) — set BAND_ENDPOINT_HOST.
    e["SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST"] = os.environ.get("BAND_ENDPOINT_HOST", "10.14.0.46")
    e["SKYRL_ROLLOUT_HTTP_ENDPOINT_PORT"] = str(PORTBASE + i)
    # Probe bridge, NOT the 9920 bridge serving the 35B milestone run.
    e["APPTAINER_BRIDGE_URL"] = f"http://10.128.1.2:{ENV_BRIDGE_PORT}"
    # hpc.py points this at /e/data1/.../ot-baf/experiments/_ray_logs, another
    # project's tree that we cannot write: ray_utils._start_node mkdir's it and
    # dies with EACCES 52s into the job. /tmp is node-local, which is correct --
    # Ray logs are per-node anyway.
    e["OT_AGENT_RAY_LOG_DIR"] = "/tmp/ray_logs"
    # Chunked gathered log-softmax for the training step (bounds the ~23 GiB
    # seq×vocab logits spike; MarinSkyRL 637a764). MUST ride container.extra_env:
    # the band512r census exported it only in the submit shell and it never
    # reached the workers — zero 'ACTIVE' markers in any pass log, and every
    # pass died at the step-1 OOM this flag exists to prevent. Harmless in
    # generate mode (no training step ever runs).
    e["SKYRL_CHUNKED_LOGPROBS"] = "1"
    if ENV_GENERATE:
        # Serialize engine constructor bringup in batches of N — the fix for the
        # generate-path crash-loop (concurrent TP NCCL/TCPStore bringup contends
        # across engines at 8+ nodes). Needs MarinSkyRL lukedhlee/engine-init-batch.
        e["SKYRL_ENGINE_INIT_BATCH"] = str(ENV_ENGINE_INIT_BATCH)
    # MUST be set. The default is 600s, which is SHORTER than the 1800s agent
    # cap, so the bridge kills the exec mid-task. Measured on the 35B canary:
    # 76% of trials cut off mid-task, which flattens reward to 0 and would make
    # a saturated-looking band out of tasks the model never got to finish.
    e["BRIDGE_EXEC_TIMEOUT"] = "4000"  # must exceed the 3600s agent wall; 2100 truncated 96/128 trials in smoke 1270405
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
