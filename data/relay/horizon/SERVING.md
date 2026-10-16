# Relay serving on Horizon

The relay's model servers run on Horizon (4x GB200 per node) with the same scripts and the same vLLM arguments as on
Jupiter. The Qwen3.8-27B teacher works on Blackwell with the marin vLLM fork build. It needed two fixes, both on the
Horizon side: a compiler variable and a repaired package in the venv. A 2-node test (1 student node + 1 teacher node)
came up in 3 minutes with warm caches. It wrote its endpoint files and passed real completions sent from the login node.

## What had to change

- **`CC=gcc CXX=g++`.** TACC's login environment sets `CC=nvc`, and sbatch passes it on. Triton compiles a small C
  helper at engine start, nvc rejects `-Wno-psabi`, and every engine dies. `serve_env.sh` exports gcc.
- **The venv's CuTe DSL package was half-overwritten.** The venv has two 4.5.3 wheels, `nvidia-cutlass-dsl-libs-base`
  and `nvidia-cutlass-dsl-libs-cu13`, and both write the same files. uv installs wheels in parallel. On Horizon the
  compiled `_cutlass_ir` library came from `base` and the Python bindings from `cu13`. Jupiter's venv has both from
  `cu13`. Qwen3.8 is a vision-language model. vLLM's memory check runs its vision encoder once, and on sm_100 that runs
  FlashAttention 4, which is a CuTe DSL kernel. So the teacher died at start with
  `TypeError: __init__(): incompatible function arguments` in `GPUModuleOp`. Fixed on 2026-09-29 by reinstalling the
  cu13 wheel. All 200 files now match its RECORD:
  `~/snowball/bin/uv pip install -p ~/snowball/envs/snowball/bin/python --reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==4.5.3`.
  A rebuilt venv can hit the same race. Run that command again after any rebuild. A second sign of a broken install is
  the vLLM log line "FlashInfer Blackwell GDN requires an intact nvidia-cutlass-dsl-libs-cu13 install", after which
  GDN prefill falls back to Triton. The fallback flag, which is not needed now, is
  `TEACHER_EXTRA_ARGS="--mm-encoder-attn-backend TORCH_SDPA"`.

With the venv fixed, the teacher runs Jupiter's exact arguments. It uses FlashInfer attention, FlashInfer GDN prefill,
and FA4 for the vision encoder's one profiling pass.

## How to launch

First build the snowball venv (`data/r2egym/horizon/build_env.sbatch`), then run
`bash data/relay/horizon/build_relay_env.sh` on the login node. It adds `envs/relay-extra` (torchvision 0.26.0 for the
teacher), python-socks for the driver's bridge and `envs/msa-2.4.6`, and downloads into `$SCRATCH/hf_hub` the teacher
(Qwen3.8-27B @ 1d4bf0f2), the EAGLE-3 draft and the Stage-3 student that `serve_env.sh` looks up, about 150 GB. It does
not download the 09-21 base; `../README.md` section 1 has that command.

On the Horizon login node, from the OTA clone (`git -C ~/snowball/ota pull` first):

```bash
cd ~/snowball/ota/data/relay/horizon
STUDENT_MODEL=$SCRATCH/models/grug-datakit-sft-20260921 bash serve_submit.sh 2 1 01:00:00 relay_srv_2n   # NODES N_STUDENT TIME [NAME]; prints the job id
python3 serve_check.py <jobid>                        # optional: real thinking completions to every server
```

`serve_submit.sh` sources `serve_env.sh` and submits `data/relay/pilot/serve_relay.sbatch` with Horizon's flags
(`-A CCR24067 -p debug --exclude= -o $RELAY_EXP_DIR/logs/%x_%j.out`). Every Horizon path is an environment knob with
Jupiter's value as its default. Jupiter runs are unchanged. The Horizon values are the venv, overlay, relay-extra,
cache, experiment dir, models and `NODE_SUFFIX=""`. **The student defaults to the Stage-3 base, so always set
`STUDENT_MODEL`** to the student's HF dir before calling `serve_submit.sh`: the router reads the tokenizer of whatever was
served, so a missing `STUDENT_MODEL` runs the wrong student with no error. `student_model=` in the endpoints `.meta`
shows what was served. Teacher defaults match the full runs: `PER_GPU=1 TEACHER_MAXLEN=131072`.

## What the driver needs

- **Endpoint files** are in `$SCRATCH/experiments/relay/pilot/endpoints/<jobid>.{student,teacher,meta}`
  (`$RELAY_EXP_DIR`). Each holds comma-separated URLs on one line, the same format as on Jupiter. On a failure the job
  writes `<jobid>.DEAD` and cancels itself. Server logs are in `.../logs/serve_{student,teacher}<k>_<jobid>.log`.
- **Hostnames:** the URLs use short node names (`http://c102-007:8000/v1`), which resolve on the login node and on
  compute nodes. The login node can reach ports 8000–8003 on compute nodes directly. `serve_check.py` ran from login.
- **Teacher layout:** each teacher node runs four single-GPU servers on ports 8000–8003, served name `qwen38`. The
  student is one DP4+EP server on port 8000, served name `snowball`.
- **After `scancel`:** the job usually writes `<jobid>.DEAD` ("a server exited mid-run") and deletes `.student` and
  `.teacher`. That DEAD file is expected after a deliberate cancel. Once (job 35820) the files survived the cancel
  instead, so check `squeue` rather than trusting an old file.
- **Running srun from the login node** into a job needs `-p debug`, or the TACC filter refuses it.

## Measured on 2026-09-29

| | Job | Start → endpoints written |
|---|---|---|
| Teacher only, cold compile cache | 35820 | 410 s (torch.compile 122 s per engine) |
| Teacher only, warm cache, exact Jupiter args | 35833 | 180 s |
| 1 student + 1 teacher, warm cache | 35834 | 180 s (student engine init 23 s, teacher 38 s; the rest is imports and weight loading from NFS) |

- **Teacher, one request per server:** about 143 tokens/s. A 10–19k-token thought, then a correct answer. Reasoning
  was split into `reasoning` by the qwen3 parser, with no think tags left in the content.
- **Teacher, 16 requests per server:** about 1,500 tokens/s per GPU, about 6,000 per node.
- **MTP-2:** mean acceptance length 2.1–2.3, and about 0.70 then 0.50 acceptance at the two draft positions.
- **Teacher KV cache:** 1.59M tokens per GPU, which is 12 full 128k contexts per server.
- **Student (Stage-3 base):** KV cache 5.25M tokens per engine. The smoke passed with think markers kept. The Stage-3
  base emits `<|start_think|>` twice, as it does in the main serve smoke. This is the base model's behavior, not
  something the port introduced.
