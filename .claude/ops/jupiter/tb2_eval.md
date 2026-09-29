# Terminal-Bench 2 on Jupiter (Daytona sandboxes, marin eval policy) — set up 2026-09-16

**The short version.** TB2 needs no snapshots of ours. Daytona holds a global snapshot for 88 of the 89 tasks under
the exact name harbor derives from each task's environment directory (`harbor__<hash>__snapshot`), so
`auto_snapshot: true` resolves them with any key and creates nothing; only `train-fasttext` has no global image and is
built once by our key on first use (one of the org's 40 slots, reused afterwards). Creating a sandbox straight from a
Docker Hub image is refused in our org ("declarative builds are not allowed"), which is why a key that cannot create
snapshots still runs 88 tasks and fails that one. Verified from the Mac 09-17 with our key: 88 GETs succeed for both
the HF dataset `DCAgent2/terminal_bench_2` and `laude-institute/terminal-bench-2` at the registry pin 69671fba (the two
trees are byte-identical, 89 tasks, 7 base images, 89 distinct Dockerfiles).

**Shape (the 09-16 runbook, adapted to Snowball).** vLLM on one GH200 node; harbor on a login node in tmux, reaching
the node directly at `http://jpbo-XXX-YY.jupiter.internal:8000/v1` (no tunnel, no SOCKS gateway: login nodes have
internet, so Daytona is reached directly too); task containers on Daytona. Nothing runs through MarinSkyRL.

**Policy (Marin Eval Policy v0.1, marin-community/marin#9193 of 2026-09-16, which supersedes #8151; its pinned
`tb2-recovery.yaml` at marin 8aa9a354):** terminus-2, Daytona, 16 concurrent trials, one trial per task, 32,768 input /
8,192 output tokens, the v0.1 retry list (API errors, artifact/context-management infra errors, sandbox build failures,
trial timeouts), the HF dataset, no sampler override (the server's defaults); reportable only with < 10 % infra-error
trials and traces attached. Snowball-specific from marin's model entries: 65,536 server context, TP1 × DP × EP,
`skip_special_tokens=false` (its think delimiters are special tokens; without it the model scores 0). **Harbor = marin-community/harbor at the pinned 7b18505a** (Luke 2026-09-16: policy numbers use upstream harbor, never
our fork), cloned at `/e/fscratch/reformo/lee27/code/harbor-v01` and imported through PYTHONPATH by the snowball-v2
interpreter. The agent kwargs are exactly the policy's `model_info` plus `api_base` and one Snowball setting,
`extra_body.skip_special_tokens=false` (its think delimiters are special tokens; marin's snowball entries keep them);
everything else, including summarization (on, 8,000 free tokens) and terminal recording, is 7b18505a's default.
Our deviations, on purpose: the EAGLE-3 draft is on (Luke's 2026-09-12 policy) and DP4 on one node instead of marin's
DP8. A first base run on our fork was killed and discarded on Luke's instruction (2026-09-16 22:00 PT): only upstream-harbor runs exist. `POLICY=marin` on the serve job applies
the August #8151 sampler (0.7 / top_p 1 / repetition 1.1) if ever needed; `POLICY=trained` = server defaults = v0.1.

## Files (`data/tb2/jupiter/`, deployed by copy to `/e/project1/transfernetx/lee27/code/tb2/`)

| file | does |
|---|---|
| `serve_snowball.sbatch` | 1 node, 12 h, marin vLLM + EAGLE-3 overlay, served name `snowball`, sampler by `POLICY` (`trained` = v0.1), writes `$E/tb2/endpoints/<jobid>` after a real completion |
| `tb2_marin_policy.yaml` | harbor JobConfig = marin tb2.yaml + our Daytona/agent kwargs; `__JOB_NAME__`/`__API_BASE__` placeholders |
| `run_tb2.sh <serve-jobid> <name> [smoke]` | login node: checks server, key, tree, harbor commit; renders + validates the policy; `harbor run` in tmux `tb2_<name>` |
| `summarize_tb2.py <jobs_dir>/<name>` | pass@1 over task means with a 95 % CI, marin's gate, infra losses by type + the resume command |

Paths: tasks `/e/fscratch/reformo/lee27/tasks/terminal_bench_2` (HF snapshot, 861 files), jobs
`/e/data1/mmlaion/lee27/experiments/tb2_jobs` (many small files → mmlaion), endpoints/logs
`/e/fscratch/reformo/lee27/experiments/tb2/`, harbor = `code/harbor-v01` @ 7b18505a via PYTHONPATH with the
`envs/snowball-v2` interpreter (has daytona SDK + the `harbor` entry point), key `keys/daytona_eval.env`.

## SWE-bench Verified random-100 (added 2026-09-17)

Same stack, dataset swap: Marin's `swebench-recovery.yaml` differs from the TB2 one only in the dataset line
(`hf://DCAgent2/swebench-verified-random-100-folders` @ 0c553bd6, = its `main`). All 100 tasks have global Daytona snapshots
(each Dockerfile is `FROM swebench/sweb.eval.x86_64.<instance>`); agent timeout 3,000 s per task, 1 vCPU. Staged at
`/e/fscratch/reformo/lee27/tasks/swebench_verified_random100`; run with `TASKS=<that dir> NTASKS=100 bash run_tb2.sh <serve-jobid> <name>`.
Budget 100 × up to 50 min at 16 concurrent ≈ 4–6 h per checkpoint on one server.

## Sequence

1. `MODEL=<hf dir> POLICY=marin sbatch serve_snowball.sbatch` → wait for `ENDPOINT` in the log.
2. `bash run_tb2.sh <jobid> <model>_<policy>_<date> smoke` (2 tasks × 1, marin's tb2-lite shape) → results in ~15 min.
3. `bash run_tb2.sh <jobid> <name>` (89 × 1 at 16 concurrent, ~2 h) → `summarize_tb2.py`; rerun infra losses with
   `harbor jobs resume -p <run> -f <Type>` after `set -a; source keys/daytona_eval.env; set +a`.
4. Never launch the same run name twice (corrupts the job dir); the serve job's wall (12 h) bounds everything.

## Known traps

- Never `pkill -f <pattern>` inside an `ssh jupiter '...'` command whose text contains the pattern: it kills the session
  (bit again 09-16). Put such commands in a script file and run `bash file`.
- `train-fasttext`: first use builds its snapshot from the task Dockerfile (100–500 s); the runbook's "Access denied"
  was a sandbox-only key. `filter-js-from-html`'s Selenium verifier can time out at 1 vCPU (runbook) — infra, not model.
- Login-node pid cap 4,096: harbor at 32 concurrent is fine; keep `OMP_NUM_THREADS=1`; no torch imports there.
- The Mac → Jupiter ssh dies after 8 h (TOTP); the tmux on login02 owns the run.
- Do not pass `--override-cpus/--memory`: sandboxes get task.toml's vCPUs from the global snapshot (runbook + marin).

### One-shot SWE-bench run on a fresh node
`swe_chain.sh <serve-jobid> <name>` (tmux `tb2_chain_<name>`): waits for the endpoint, runs the 100 tasks under the
v0.1 policy with no smoke, runs `post_run.sh` once, then `scancel`s its own serve job so the node is not held to wall.
Submit the server with `--job-name=tb2_serve_<x> --time=08:00:00` (4–6 h of trials + one recovery pass).

## OpenThoughts-TBLite 2.0 (added 2026-09-28)

**One policy for every benchmark: TBLite runs exactly the 09-24 Marin eval policy we use for TB2.0; only the task tree
differs** (Marin's own `ot-tblite-recovery.yaml` differs from its `tb2-recovery.yaml` only in the dataset line). Validated
2026-09-28 on `newstack-step24`: **29.5 % pass@1** (95 % CI 20.3–38.7, 93 scored of 100, 7 infra losses) vs Ben's 24.3 under
the same policy, within single-run noise. **One run = 1.9 node-h and ~1 h wall on two servers** (sharded, 16 concurrent
each, both servers ~57 min including ~10 min startup). Trials are short: median 4.9 min, 19/100 hit the 1,800 s budget.

What it is: harbor registry `openthoughts-tblite` v2.0 = 100 tasks from `open-thoughts/OpenThoughts-TBLite` @ `f075e463`,
staged at `/e/fscratch/reformo/lee27/tasks/openthoughts_tblite_2_0` (README beside it). Not `DCAgent/dev_set_v2` (the older
"ot-tblite", opencode agent).

Snapshots: 92 tasks resolve to Daytona global snapshots under harbor's names; the other 8 (iot-device-registration-server,
tsl-test-case-generation, sympy-bug-fix, prediction-model-evaluation, book-portfolio-analysis, image-tile-identification,
git-repo-forensics, okhttp-trailers-crash) were built once by our key (`…b61`) and hold 8 of the eval org's 40 slots. If
they are gone, rebuild before a run with `tblite_build_snapshots.py <task> ...` (in `code/tb2`, run with `PYTHONPATH=<harbor-p0924>/src` and the key sourced)
(harbor 761fb516's own create path; 1–4 min each; needs free slots), or pass them in `EXCLUDE_TASKS` (infra losses; >10 %
lost voids a policy score). Every task's snapshot name: `/e/fscratch/reformo/lee27/experiments/tblite/snapshot_names.tsv`.

### One command (Terminus-2, policy score)
```
cd /e/project1/transfernetx/lee27/code/tb2
M=<hf-format checkpoint dir>; N=tblite_<model>_p0924_$(date +%Y%m%d)
A=$(MODEL=$M POLICY=trained sbatch --parsable -A <acct> --time=02:00:00 --job-name=tb2_tblite_serve_a serve_snowball.sbatch)
B=$(MODEL=$M POLICY=trained sbatch --parsable -A <acct> --time=02:00:00 --job-name=tb2_tblite_serve_b serve_snowball.sbatch)
tmux new -d -s tblite_${N}_s0 "SHARD=0/2 bash tblite_chain.sh $A ${N}_sh0"
tmux new -d -s tblite_${N}_s1 "SHARD=1/2 bash tblite_chain.sh $B ${N}_sh1"
# when both $E/final_${N}_sh{0,1}.DONE exist (each chain releases its own server):
J=/e/data1/mmlaion/lee27/experiments/tb2_jobs
python3 tblite_merge.py $J/${N}_merged $J/${N}_sh0 $J/${N}_sh1 && python3 summarize_tb2.py $J/${N}_merged
```
Unsharded is `bash tblite_chain.sh <jobid> <name>` on one server (same node-h, ~2 h wall). `tblite_chain.sh <jobid> <name>
smoke` = the first two tasks at 2 concurrent; `RELEASE=0` keeps the server for a following run. Each chain reruns the
unscored trials once only if they exceed 10 % of its planned trials.

### mini-swe-agent tool mode (TB2 focus harness)
`tblite_msa_toolmode_0924.yaml` = `tb2_msa_toolmode_0924.yaml` with the TBLite tree; run through `msa4/msa4_chain.sh`:
```
TASKS=/e/fscratch/reformo/lee27/tasks/openthoughts_tblite_2_0 NTASKS=100 EXCLUDE_TASKS= \
  POLICY_FILE=$PWD/msa4/tblite_msa_toolmode_0924.yaml EXPECT_MODEL=<substring of the model path> \
  bash msa4/msa4_chain.sh <jobid> <name> full            # or: smoke with SMOKE_TASKS=a,b
```
Smoke on the 09-21 SFT student passed (real tool loops, 0 infra). **The RL exports (newstack-step24) cannot use this
harness**: they answer in Terminus-2 JSON, never `<tool_call>`, so every episode ends in RepeatedFormatError (2/2 in the
smoke). Score RL checkpoints on Terminus-2 only.

### Traps seen
- Infra losses are tmux protocol/session errors and one sandbox-start timeout (7/100); no recovery pass needed at <10 %.
- The 8 org-built snapshots compete for the shared 40 slots; the 09-28 setup freed slots by deleting the MiMo probe pool
  (idle, ours) and one idle harbor snapshot. Keep CalibForge and TB2.1 (train-fasttext `harbor__9d3e262109f7`) snapshots.
