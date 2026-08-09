---

## ★★★★ 0.0-DONE 2026-08-09 ~07:00 KST — BAND512 CENSUS IS FINAL. THE MILESTONE IS DELIVERED.

**The 512-task pass@4 probe of `g1_diverse_tezos_100k_8b` (frozen, lr=0, OpenCode agent) is COMPLETE
after 7 harvest passes. Do not re-run it. Artifact: `/e/fscratch/reformo/lee27/band512_census_final.json`
(copy committed at `ai_memory/artifacts/band512_census_final.json` — includes the band task lists).**

**Headline numbers (state the denominator, always):**
- Coverage: **500/512 tasks with ≥4 clean samples** (all 512 have ≥1; the 12 stragglers have 1–3 clean
  samples after seven passes — effectively un-runnable at full depth in this harness, excluded with note).
- **Strict band (0 < pass@4 < 1, first-4 samples): 44/500 = 8.8%.**
- **all_pass = 0. zero_pass = 456/500 (91.2%).** Headroom band (pass<1) = 500/500 = **vacuous** for us.
- All-samples view (any # samples, 512 denom): 50 tasks ever-passed-but-not-always.
- Raw material: 6,427 results, 2,962 clean trials; dominant exceptions BridgeOperationError 2,492
  (mostly the quota era) + BridgeOperationTimeout 439 + NonZeroAgentExit 349.
- Band task list (44): r2egym-0212 0469 0554 0644 0653 0695 0984 1201 1385 1507 1515 1523 1631 1679
  1711 1727 1735 1784 1921 2039 2047 2071 2103 2167 2175 2183 2191 2215 2223 2291 2305 2321 2354
  2370 2387 2414 2771 2819 2940 2996 3405 3912 4121 4193.

**vs Marianna:** her band = ~1.6k/4.5k ≈ **32–36%** (terminus-structured agent, p@4). Ours is 8.8%
with **zero saturated tasks** — the gap is the OpenCode-vs-terminus absolute pass-rate gap, not a harness
defect (within-group variance is real; band fraction was stable 8.6–10.5% across coverage 257→500).
Extrapolated over the 4,469 allowlist: **~390 band tasks** — a non-degenerate GRPO pool. NEXT: decide
band-filtered GRPO (her result: band converges faster, same final SWB p@1, 48k vs 60k rollouts).

**Wall-clock honesty + the <10h recipe (Luke asked; hers took <10h, ours ~3 days):** the days went to
one-time debugging (meta-crash race root-caused+fixed, quota sweeps, JUWELS pivot) and ONE recurring
structural flaw: harvesting through the RL trainer means each job dies at its (useless) lr=0 training-step
OOM after ~32 groups → 7 passes. Fix for the next sweep: make the training step survivable
(micro_train_batch 1 / cap train seq-len — rollouts unaffected) or skip the update entirely; then the
512-census is ONE ~5–6h submission and the full-pool sweep shards wide, comfortably <10h.

**Infra state at close:** JUWELS worker fleets 14187062/63 CANCELLED by me (harvest done). Jupiter→JUWELS
CM + all forwards (9923, 18300-03) still UP on jwlogin08 (10.13.0.158). **JURECA backup fleets 15506490 /
15506501 still PENDING — cancelling needs Luke (TOTP; the old JURECA CM is dead).** Staging swept (0 stale).
Monitors stopped. Probe gotcha: the JUWELS forward listeners bind **10.13.0.158, not 127.0.0.1** — a
127.0.0.1 curl returns 000 and looks like a dead tunnel when everything is healthy.

**NEXT ACTIONS (fresh session — in order):**
1. **Operator decision first:** train band-filtered GRPO on the 44 (512-scale pilot) OR sweep the full
   4,469 allowlist to get the ~390-task band before training. Her evidence (verbatim ref): band buys
   convergence speed + ~20% compute, NOT final performance — so the full sweep is only worth it if we
   want the bigger training pool, not for the comparison itself.
2. **Before ANY next sweep, kill the pass-ender:** make the lr=0 training step survivable
   (micro_train_batch 1 / cap train seq-len; rollouts unaffected) or skip the update. Then 512-scale
   census = ONE ~5–6h submission; 4.5k ≈ 9× rollouts ⇒ ~2 days at 4 shards (36 Jupiter GPU nodes),
   <10h needs ~12 shards. CPU workers are NOT the constraint (JUWELS had ~744 idle batch nodes).
3. Reuse as-is: sync-gated `arm_rollout_forward_juwels.sh`, `mk_pass3.py` (fill-in builder),
   `band_census_final.py`, the 32/32/32 trio, submission recipe (gotchas 08-08: submit from
   `OpenThoughts-Agent-r2egym-bridge-next`, `WANDB_API_KEY=offline-dummy`, sweep staging BETWEEN passes).
4. Housekeeping owed: Luke cancels JURECA 15506490/501 (TOTP); revert canary_harbor.yaml DEBUG→INFO
   (deferred since 08-08); optionally ask Marianna what her **358** figure actually is (best guess:
   `r2egym_learnable_heldout` eval-split size — see marianna_parity.md).
5. If the scaffold gap itself becomes the target: the band would widen by fixing OpenCode-side
   pass rate (tool-schema SFT mismatch, §0.3 of the 08-06 sections) — separate workstream, operator call.

---

## ★★★ 0.0-QUOTA 2026-08-08 ~02:30 KST — BAND PROBE #1 DIED ON INODE QUOTA; RETRY STAGED AT CONC 64

**band512_s0..3 (1271527-30) crashed 15 min after start (02:15-02:29 UTC 08-07):
1,562/2,048 trials = `BridgeOperationError: [Errno 122] Disk quota exceeded` on
`/p/scratch/synthlaion`.** Cause is the **INODE quota, not bytes**: project at
3.887M/4.0M soft (4.4M hard); 512 concurrent staged sandboxes × ~2.3k files each
punched the hard limit. Data quota was fine (64TB/97TB). Check quota with
`jutil project dataquota -p synthlaion` (login shell only — `bash -lc`).

**Partial signal from the 160 clean trials:** 4 mixed (band) tasks found —
r2egym-1711 1/4, r2egym-2215 2/4, r2egym-2414 1/2, r2egym-2996 1/2; of 16 tasks
with full 4 trials, 2 in band (12.5%, n far too small to quote). Clean-trial pass
rate 5/160. Census artifact: `/e/fscratch/reformo/lee27/band512_census.json`;
census script `/e/fscratch/reformo/lee27/band_census.py` (reusable — glob
`experiments/<name>/<name>/trace_jobs/*/result.json`).

**Retry staged (08-08 night):** `band512r_s0..3` under
`/e/fscratch/reformo/lee27/experiments/` — byte-identical clones of the originals
(sed name swap in launcher JSON + sbatch) except `n_concurrent_trials=64` (peak
staging ≈ 256×2.3k ≈ 590k inodes, fits post-cleanup headroom ~1.2M). Conc lives in
`configs/<name>_rl_config.json` → `skyrl_hydra_args` — NOT in any YAML on disk
(the of04 shard YAMLs were never saved; experiments/*/sbatch+configs are the
source of truth for resubmission). Ports 18300-18303 preserved per shard.

**Rebuilt stack (all verified): bridge tmux on login02 (BRIDGE_STALE_READY_SEC=3600,
`curl 10.128.1.2:9920/status` OK); Jupiter→JURECA CM re-established (TOTP by Luke,
ControlPersist=yes + ServerAliveInterval=60, keepalive tmux `cmkeep`); 9923 forward
armed + verified from JURECA; fleets queued: 15506490 (32n/24h, est start ~19:20
08-08 cluster) + 15506501 (16n/12h backfill). Staging cleanup of 651 stale
apt_env dirs running 8-way parallel on login02.** Mac-side watcher auto-submits
the 4 retry shards when a fleet is RUNNING + workers_alive. After shard start:
arm 18300-18303 rollout forwards (arm_rollout_forward.sh, end-to-end verify) and
watch first 15 min of trace_jobs for quota errors.

**JURECA→Jupiter auth fact: the dedicated key gives only "partial success" —
JURECA chains publickey+TOTP. The CM can NEVER be re-established without Luke.**

---

## ★★★ 0.0-NIGHT 2026-08-07 ~11:00 KST — INFRA CHAIN CLOSED (reaper was the killer); NOW MEASURING THE MODEL

**The ~900s mass-masking mystery is SOLVED and fixed** (full chain in gotchas 08-07):
bridge zombie reaper executed every sandbox whose agent thought >15 min
(`BRIDGE_STALE_READY_SEC=900`, last_used only ticked on job submit). Fixes deployed:
harbor `a12ca6bc` (reaper busy-guard), `2bb65ce2` (staging sweep age-gate),
`60703b26` (flushed worker receipts), band `ce8e7ff3` (BRIDGE_EXEC_TIMEOUT 4000),
`061fe2cb` (canary_harbor carries the timeout values — gen REPLACES base8b's harbor
section, edits to base8b harbor fields DO NOTHING). Bridge restarted with fix +
BRIDGE_STALE_READY_SEC=3600.

**Smoke 32f (1270405, first unbiased n=128):** masking 119→0; all 128 transcripts;
agents run 25-40 min. Model-side reality: edit rate 6/128 (4.7%); raw-JSON-as-text
29/128; `invalid` tool calls 64 (model hallucinates `ls`/`cat` — not trained on
OpenCode schema); hallucinated read paths; glob/grep path:null SchemaErrors;
rg exec fails ~2% (binary itself OK on nodes); rewards 0.0×32 completed (plausible
at n=32 given her band density). **Dominant structural issue: TURN STARVATION —
~4 min/turn under 64 trials on 4 TP=1 engines ⇒ ~8 turns/trial vs the 50-step cap.**

**RUNNING NOW: smoke 32g (1270830, 9 nodes = 32 TP=1 engines, conc 64 ⇒ 2 trials/engine,
exec 4000):** measures edit rate + turns/trial with serving 8× less starved. Census it
first (same finalcensus/agentstall flow as 32f in the transcript).

**Marianna parity sheet: `ai_memory/marianna_parity.md`** (her script verbatim in
ai_memory/reference/ — TIMEOUT_SEC=1800, 50 steps, terminus-structured, TP=4×8 engines,
**her band = 358 tasks (~8%), NOT 36%**).

**Live state:** fleet-1 (in-alloc 15502693) DIES 07:00 CEST 08-07; fleet-2 15503116
until ~17:00 CEST; fleet-3 15504529 queued (has all fixes). CM keepalive tmux `cmkeep`
on jupiter login02; bridge tmux `bridge` (restarted w/ fix); observer tmux `obskill`
on JURECA login (kill when done). Band probe over the 4,469 remains GATED on an edit
rate that actually moves.

# NEXT SESSION — takeover note

---

## ★★★ 0.0-FINAL 2026-08-06 ~17:30 KST — TASK-POOL VALIDATION COMPLETE. NEXT: TOTP → RUNBOOK.

**The r2egym pool question is CLOSED. Do not re-sweep.** Terminal state:
- **Allowlist v3: 4,469 of 4,568 solvable (97.8%)** — `allowlist_r2egym_v3.txt` +
  `gatefail_r2egym_v3.txt` (99 names) at `/p/scratch/synthlaion/lee27/` AND
  `/e/fscratch/reformo/lee27/` (v3 supersedes v1/v2; gzipped copies in `ai_memory/artifacts/`).
- The 99 exclusions are DIAGNOSED (census in §0.0-UPDATE): 31 grader-expectation mismatch (all tests
  pass, frozen expected-output disagrees), ~47 environment-hostile tests (network on air-gapped
  nodes, wall-clock asserts, GUI), 21 numpy hypothesis-suite fragility. A quiet-node rerun recovered
  only 9 of the original 108 — the 99 fail persistently. **"Unsolvable" = the human gold fix itself
  cannot score 1.0 in our sandbox, so no agent can. Skip them in ALL training/probing.**
- **Zero free-pass tasks in the entire pool** (only (0,1)/(0,0) reward pairs). The old ~22%
  free-floor was a patched-dataset artifact.
- Operator decision (told to me directly): **skip the 99, train/probe on the 4,469 only.**

**LIVE STATE at handover (re-probe, don't trust):**
- JURECA allocation **`15502693`** (32× dc-cpu, envgate_alloc) idles for the sweep work — submitted
  ~07:00 CEST 08-06 with 24h wall ⇒ **dies ~07:00 CEST Aug 7 (14:00 KST)**. Still useful for any gate
  probes; NOT a worker fleet (no workers running in it).
- **NO worker fleet exists. The Jupiter→JURECA ControlMaster is still DOWN** unless Luke has run the
  clipboard command (`ssh -t jupiter 'mkdir -p ~/.ssh/cm_jureca && ssh -4 -M -S ~/.ssh/cm_jureca/qwen36
  -fNT -o ControlPersist=8h lee27@jureca05.fz-juelich.de && ssh -S ~/.ssh/cm_jureca/qwen36 -O check
  lee27@jureca05.fz-juelich.de'`). **First action of the new session: check
  `jup 'ssh -S ~/.ssh/cm_jureca/qwen36 -O check lee27@jureca05.fz-juelich.de'` — if "Master running",
  execute the RUNBOOK below immediately** (bridge → forward → fleet → smoke). My background watchers
  died with the old session; re-arm your own.
- Mac→JURECA direct mux (`ssh jureca '<cmd>'`) was alive all day and carried the whole campaign —
  don't kill it; new connections need TOTP.
- tmux sessions left on cluster: `egbig`/`egpandas` (JURECA login, dead/idle — safe to kill),
  `egstage` (Jupiter login, done).

**What changed today (all pushed): See §RUNBOOK preconditions + gotchas 08-06.** Highlights:
bare_json parser repair pass (`d1e3ecb8`, deployed to the fscratch checkout); harbor `55c416cd`
pandas config-strip (deployed to `/p/project1/synthlaion/lee27/harbor`, rides next fleet start with
the `rg` bind `1019a36`); envgate narrow oracle (`fa5acdd6`); the fleet-env / heredoc / broad-oracle
gotchas.

**Marianna handoff draft (operator asked how to share — approved shape, paste-ready):**
> We gate-checked your full r2egym pool model-free (pristine vs gold-commit oracle, your SIFs, our
> JURECA workers): 4,469/4,568 verifiably solvable, 99 where even the gold fix can't score 1.0
> (env-sensitive grading: network/timing tests, frozen expected-outputs, hypothesis flakes). Lists:
> `/p/scratch/synthlaion/lee27/{allowlist,gatefail}_r2egym_v3.txt`. Two free cross-checks from your
> 18k-rollout history: (a) did any of the 99 ever score 1.0 for you? (b) is your ~1.6k band fully
> inside our 4,469? FYI: a verifier applying gold via `git checkout <base> -- .` resurrects scrubbed
> working-tree configs on newer pandas/aiohttp images — checkout only the commit's own files.

**Sequence for the new session:** ① CM check (above) → ② RUNBOOK steps 2-6 (bridge, forward, fleet
with EXPLICIT env, rg-verify by behaviour, ~32-task edit-rate smoke from the ALLOWLIST) → ③ band
probe over the 4,469 (quote band %% against both denominators). The edit-rate smoke gates the band
probe: `agentstall.py` — ripgrep errors 0 WITH tools called, edit rate off 14%, and the vLLM log line
`bare_json: repaired a malformed tool call` proving the parser repair fires live.

---

Written **2026-08-06 ~02:40 KST**, revised **~13:00 KST** (§0.0, §1, §2, §5 rewritten — read those first;
later sections still carry the 02:40 framing). Supersedes all earlier versions
(previous one archived at `ai_memory/logs/2026-08-06_NEXT_SESSION_superseded.md`).
Report times in **KST** (cluster clocks are CEST = KST − 7h).

---

## ★★ 0.0-UPDATE 2026-08-06 ~14:30 KST — 260-TASK SWEEP DONE: CEILING 70.0% (n=260); FULL SWEEP IN FLIGHT

Supersedes the fleet/live-state claims in §0.0 below. The task-pool question §0.0 posed is now measured.

**Sample sweep result (COMPLETE 14:12 KST, `envgate_results_envgate_big.jsonl`):**
- **Harness: clean.** 520/520 records, 0 timeouts, 0 nulls, offline patch ok 520/520.
- **Ceiling: 182/260 = 70.0%** (95% CI ≈ 64–76%). Consistent with the n=32 estimate (78.1%, CI 61–89).
  Pooled with the 32 (3-task overlap, outcomes reproduce exactly): 206/289 ≈ 71.3%.
- **Zero free-pass tasks in 260** — no `pristine=1.0` anywhere. The patched-era "~22% pass with zero
  work" pathology does NOT exist on the raw path. Only two reward patterns occur: (0,1)×182, (0,0)×78.
- **Failures are repo-concentrated and diagnosed:** pandas **63/66 dead** (`pytest: unrecognized
  arguments: --strict-data-files`), aiohttp **11/23** (`asyncio.async(` SyntaxError), tornado 3/16
  (**all tests PASS yet reward 0.0** — grader-expectation mismatch vs `expected_output_json`; 2296/2400
  identical tails), numpy 1/40 (4 hypothesis-library errors at gold state, 571 pass).
  Everything else (pillow 40, orange3 23, scrapy 18, datalad 17, pyramid 11, coveragepy 6): **0 failures**.
  Excluding pandas+aiohttp, the rest of the pool is 167/171 = **97.7% solvable**.

**Fleet correction (supersedes §0.0 "RUNNING"):** `15502687` **FAILED at t=1s** (06:41:49 CEST) —
submitted without `HARBOR_SRC` (the sbatch requires it; the old fleet's env was inherited via
`--export=ALL` and never captured). **The sweep does not need the fleet, workers, or bridge at all** —
it only needs an allocation for `srun --overlap`. Replaced by plain allocation **`15502693`**
(32× dc-cpu, 24h, from ~07:00 CEST), which ran the sample sweep in 19 min at `EG_CONC=64`.

**Tunnel state:** the Jupiter→JURECA ControlMaster is DEAD (socket refused; no bridge process, no
`ssh -R` on login02). All reverse forwards (18300 vLLM, 9923 workers→bridge) died with it.
**Re-establishing needs operator TOTP** (`ssh -4 jureca05.fz-juelich.de` from Jupiter login) — required
before any model re-probe, NOT for the sweeps. The live listener `jrlogin05i:9920` is NOT ours
(our bridge was Jupiter-side and is dead) — do not touch it, do not run `start_bridge_jureca.sh` at 9920.

**FULL SWEEP DONE (16:38 KST): baseline ceiling 66.1%** — 3,015/4,564 usable tasks gate-pass
(9,214 records; 4 timeouts + 4 nulls + 2 `no_sif` excluded as unknown). Per-repo fails/total:
**pandas 1,332/1,443 (92%)**, aiohttp 139/299, numpy 31/776, tornado 22/259, orange3 12/482,
datalad 9/179, scrapy 4/215, coveragepy 5/108, **pillow 0/619, pyramid 0/189**. Only two reward
patterns exist at scale: (0,1)×3,015 and (0,0)×1,549 — **zero free-pass tasks in the whole pool**.
Artifacts: `allowlist_r2egym_v1.txt` (3,015) + `gatefail_r2egym_v1.txt` (1,554) in
`/p/scratch/synthlaion/lee27/`.

**★★ FINAL (17:15 KST): RETRY SWEEP DONE — CEILING 97.6% (4,460/4,568). THE ALLOWLIST EXISTS.**
Fixed mechanics: harbor `55c416cd` (worker.py `pandas_compat` strips `--strict-data-files`; envgate
inherits it by extracting the live function) + envgate `fa5acdd6` (**narrow oracle** — checkout only
the gold commit's files). Harness on the retry: 0 timeouts, 0 nulls, patch ok 9,207/9,207.
- Recovery: pandas 1,332→**8** fails; aiohttp 139→**24**; total fails 1,549→**108** (2.4%).
- ⚠ **RETRACTION: "aiohttp is genuinely unfixable (asyncio.async SyntaxError)" was over-general** —
  115 of its 139 failures were the broad-oracle resurrection artifact, same as pandas. The
  SyntaxError diagnosis holds only for (at most) the residual 24.
- Residual fails (park; diminishing returns): numpy 31, aiohttp 24, tornado 20, orange3 11, pandas 8,
  datalad 7, coveragepy 5, scrapy 2. Plus 1 task with no SIF (`r2egym-3711`).
- **Zero free-pass tasks in the entire 4,568-task pool** (reward pairs are only (0,1) and (0,0)).
- Artifacts: `allowlist_r2egym_v2.txt` (4,460 names) at `/p/scratch/synthlaion/lee27/` AND
  `/e/fscratch/reformo/lee27/` (for gen_band_yaml), gzipped copies in `ai_memory/artifacts/`.
  v1 files are superseded.
- Consequence for the band probe: run it over the allowlist; quote band % against BOTH denominators
  (allowlist and full pool) so the number is comparable to Marianna's ~36%-of-4.5k.
⚠ shim-staging trap recorded in gotchas: an UNQUOTED heredoc expanded `$_cfg` to `""` and staged a
no-op sed — verify staged shims by grep before running the gate.

---

## ★ RUNBOOK — model-side bring-up, gated ONLY on the operator's TOTP (written ~16:00 KST 08-06)

Everything below is verified staged; the single missing piece is the interactive CM re-auth.

**Preconditions already in place (verified by inspection this session):**
- harbor cluster checkout `/p/project1/synthlaion/lee27/harbor` @ `1019a36` — `rg` bind in
  `worker.py:106` + tool loop `:710`; binary `agent_tools/bin/rg` is ELF x86-64 **static-pie**.
- `bare_json` parser now carries a **repair pass** (`d1e3ecb8`, deployed to the
  `/e/fscratch/.../OpenThoughts-Agent-r2egym-bridge-next` checkout by fetch+hard-reset, md5-verified):
  repairs 8/10 real malformed tool calls from `1251403`. See gotchas 08-06 (malformed-JSON taxonomy).
- Sweep allowlist: see §0.0-UPDATE; file path recorded there when the full sweep lands.

**Sequence (operator does step 1; the rest is scriptable):**
1. **TOTP (Luke, ~2 min):** on the Jupiter login02 shell:
   `ssh -4 -M -S ~/.ssh/cm_jureca/qwen36 -fNT -o ControlPersist=8h lee27@jureca05.fz-juelich.de`
   `-4` mandatory (IPv6 resolution bypasses the key's `from=` clause → publickey denied, no TOTP
   prompt). Pin `jureca05` (10.14.0.46 is jrlogin05's; round-robin lands elsewhere and the forward
   binds nothing).
2. **Bridge** on Jupiter login02 (internal bind per guardrail — NOT 0.0.0.0):
   `tmux new -d -s bridge "python3 /p/project1/synthlaion/lee27/harbor/src/harbor/environments/apptainer/server.py --port 9920 --host 10.128.1.2 >> ~/bridge_9920.log 2>&1"`
3. **Workers→bridge forward** on the CM:
   `ssh -S ~/.ssh/cm_jureca/qwen36 -O forward -R 10.14.0.46:9923:10.128.1.2:9920 lee27@jureca05.fz-juelich.de`
   Verify from JURECA: `curl http://jrlogin05i:9923/status` → 200.
4. **Fleet, with FULL EXPLICIT env** (the 15502687 lesson — never rely on `--export=ALL` inheritance
   of an interactive shell):
   `sbatch --nodes=32 --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,BRIDGE_URL=http://jrlogin05i:9923 /p/project1/synthlaion/lee27/harbor/src/harbor/environments/apptainer/jureca_workers.sbatch`
5. **Edit-rate smoke** (~32 tasks from the ALLOWLIST, thinking ON — leave `BAND_SERVER_NO_THINK`
   unset): §8 recipe + `arm_rollout_forward.sh <job> 18300 <fleetid>`; probe the route only AFTER
   vLLM logs startup. Gate with `agentstall.py`: ripgrep errors must be 0 **with tools actually
   called** (state the denominator), edit rate must move off 14%, and grep the vLLM log for
   `bare_json: repaired a malformed tool call` — that line is the repair pass proving itself live.
6. Only on a passed smoke: **band probe** over the allowlist, pass@4, sharded
   (`gen_band_yaml.py` + the 8-shard precedent; ~200 concurrency against 512 fleet slots).
   Success criterion: within family of Marianna's ~36% (her denominator = full 4.5k pool).

---

## ★★ 0.0 STATUS 2026-08-06 ~13:00 KST — HARNESS VALIDATED; NOW VERIFYING THE TASK POOL

**The no-model gate passes on both task sets. Do not re-litigate whether the harness works.**
**But read the sample sizes below before quoting any rate.**

| | gate result | tasks gated |
|---|---|---|
| **r2egym** | **25/32** give pristine `0.0` **and** oracle `1.0`. Harness clean: 64/64 records, **0 timeouts, 0 nulls**. | **32 of 4,569 = 0.7%** |
| **SWE-bench Verified** | **8/8** give nop `0.0` and oracle `1.0` — one instance per repo, from the 8 largest repos. | **8 of 500 = 1.6%** |
| **SWE-bench SIFs** | build COMPLETE | **500/500, 577 GB**, `/p/scratch/synthlaion/lee27/swebench_sif` |

⚠ **The gate proves the HARNESS is sound — that the grader measures code state rather than returning a
constant. That is an EXISTENCE claim and 32 tasks suffice for it. It does NOT establish a pass rate over
either task set.** Keep those three claims separate (harness health / measurement / dataset ceiling).

⚠ **The "78.1% ceiling" is a point estimate from n=32, 95% CI ≈ 61–89%.** Earlier versions of this doc
stated it as a hard number — that was wrong. Quote it as a range until the sweep below lands. Pinning it
to ±5 pp needs ~260 tasks gated.
The 7 known failures are **broken tasks, not a broken harness** (diagnosed): aiohttp dies on
`asyncio.async(` -> `SyntaxError` (`async` reserved since Py3.7); pandas dies on pytest
`unrecognized arguments: --strict-data-files`.

⚠ **SWE-bench's "479/500 = 96%" is REPO coverage, not instances gated.** 8 instances were gated; their
repos account for 96% of the benchmark. That the other instances work is a reasonable inference from the
shared base-image family and harness path — **not a measurement.**

**Getting here required fixing FOUR harness bugs, each of which faked "the environment is broken":**
missing `agent_tools` bind + `_patch_test_sh_for_offline_pip`; missing `--no-home`; **missing `--cleanenv`
on `instance start` AND on every `exec`** (the big one — runs went from hanging past 25 min to finishing in
~2 min); and no `PATH` / wrong cwd / wrong exec form. See `gotchas.md`. **A gate that diverges from
`worker.py` measures the gate.**

### THE CURRENT PRIORITY (operator's call, 2026-08-06 ~12:30 KST): verify the TASK POOL before the model

Rationale, in the order that matters:
1. **Broken tasks are specifically toxic to GRPO.** It learns from spread between attempts at the SAME
   task. An unsolvable task returns all zeros, contributes zero gradient, and still consumes its full
   share of every batch. 20% broken = paying full GPU price for batches 20% guaranteed-dead.
2. **The output is a permanent asset** — a verified solvable-task list improves every future run. The `rg`
   fix improves one run.
3. **Zero GPU cost**, runs on the idle sandbox fleet.
4. **Without it the ambiguity returns one level up**: a bad reward number means "weak model" OR "impossible
   task pool" — the same ambiguity the gate was built to kill.

Sequence: fleet restart (done) → ~260-task sample sweep → decide on full 4,569 sweep → only then re-probe
the model with thinking ON + `rg` live.

**Live state as of ~13:00 KST:**
- **Fleet `15500584` CANCELLED (idle, ours); replaced by `15502687`** — 32 nodes, dc-cpu, fresh 24 h wall.
  Restart was free (fleet idle since its only client died 00:40 KST-cluster) and does double duty:
  activates `rg` AND gives the sweep a full-length allocation.
- **`rg` should now be LIVE** (harbor `1019a36f` was pulled onto `/p/project1/synthlaion/lee27/harbor`
  at 20:37 CEST, before this fleet started). Binary: `/p/scratch/synthlaion/lee27/agent_tools/bin/rg`,
  x86-64 **static-pie**. Path agreement confirmed: the sbatch defaults `BRIDGE_AGENT_TOOLS` to that dir.
  **Verify by behaviour inside a task container before believing it.**
- **`envgate.sh` on the cluster was STALE and is now fixed** — it defaulted to the pinned
  `apptainer_bridge/9c31e931/worker.py` (Jul 29, 80,793 B) instead of the live harbor worker
  (Aug 5, 86,365 B, +5.5 KB incl. the astropy shim). The fix existed locally and was deliberately not
  deployed mid-sweep, then never redeployed. **Redeployed 2026-08-06 ~12:50 KST and verified.** A 260-task
  sweep against the stale copy would have measured the wrong harness.
- **Staging is the sweep's blocker**: the gate reads `/p/scratch/synthlaion/lee27/envgate/<task>/{test.sh,
  metadata.json}` and only the original **32** are staged. A subagent is tracing the canonical source and
  staging 260 into `envgate_big/`, with a **byte-comparison of 3 freshly-staged tasks against the
  validated 32 as a hard gate** — divergent staging would silently invalidate the sweep.
- **Training job `1252276` ENDED `TIMEOUT` at 04:00:24** = its 4 h wall, the intended end, not a crash.
  **FINAL n=14: thinking-OFF is HARMFUL.** Edit rate 14% (1/7) → 7% (1/14); raw-JSON-as-text 71% → 50%
  but bought nothing because **zero-tool trajectories went 0% → ~75%**; rewards `{1.0: 1, 0.0: 13}`;
  **groups with variance: 0** → no gradient at all. **Keep thinking ON.**
  ⚠ Its "ripgrep errors 0/14" is a **denominator artefact** (no tools called at all), NOT the `rg` fix.
- SWE-bench repos need **build deps** in `$BRIDGE_AGENT_TOOLS/wheels` (`flit_core`, `setuptools_scm`,
  `cython==0.29.30`, `oldest-supported-numpy`). Staged 87 -> 95; **expect to extend per new repo family**
  when gate coverage widens past the 8 pilot repos.
- Operator-facing status page (progressive-disclosure briefing, kept current):
  `https://claude.ai/code/artifact/9434f46b-46cd-44fa-a02d-9e248607bdd4`

---

## 0. Read this first: the finding has MOVED

Yesterday's headline was "the r2egym reward is a constant per task and never measures the model." That was
correct, and **it is now fixed and verified.** The reward measures code state. Do not re-litigate it.

**★★ THE GATE PASSED at 02:30 KST on 2026-08-06. A group now contains both a `0.0` and a `1.0`.**
This is the first genuine within-group reward variance in the project's history, and it is causally clean:

```
task r2egym-2514, two of four samples (task_name confirmed in BOTH result.json files)
  dCmGsKH  reward 1.0   "3 passed"                4x edit tool calls + read/ls/bash
  uGPuwvX  reward 0.0   "2 failed, 1 passed"      bash + glob only -- NO edits
```

The sample that **edited files** turned two failing tests into passing; the sample that did not, did not.
That is the exact **inverse** of the old pathology, in which passing trials edited *less* often than failing
ones (24% vs 49%). **The environment measures the model. Stop re-verifying this and go get a band number.**

⚠ **Be precise about which claim this supports, or you will repeat the pass-rate-vs-band error.**
The gate as posed — *"does any group contain both a `0.0` and a `1.0`?"* — is **MET**. But `band.py`'s
`IN BAND` counter only considers **fully-sampled** groups (≥4 trials), and `r2egym-2514` had 2 of 4 samples
in when this was observed, so `band.py` still reports `0 fully sampled` and cannot yet print a band %.
**Within-group variance is PROVEN; the band PERCENTAGE is not yet measured.** Those are different claims —
do not upgrade one into the other. The rest of the ladder passes cleanly:

```
[1] non-null reward : 7/7 -> PASS
[2] steps           : median 3  max 8   -> PASS (multi-turn)
[2b] TOOL CALLS     : median 2  max 11  | ZERO-tool trajectories 0 (0%) -> PASS
[3] trial duration  : median 3.4 min  p90 13.9 -> PASS (real work)
[4] groups: 6 seen, 0 fully sampled
```

**Throughput is the next real constraint, and it is severe:** 0.29 trials/min measured, peak concurrency 5
(completed trials only), ⇒ the full band of 13,312 trials projects to **~759 h** at this rate. That number is
understated-in-our-favour (the run lost its first 20 min) but it is the right order of magnitude to plan
against. Raising the edit rate (§0.3) and concurrency both feed this; scaling is now sanctioned, because the
gate that gated it has passed.

The remaining problem is a **rate** problem, not a correctness one: the agent succeeds too rarely
(1 of 7 trials made any edit), for two identified and fixable reasons — see §0.3. Those are now throughput /
quality levers on a working pipeline, no longer blockers.

### 0.1 The environment is VALIDATED — model-independently

All three documented free-pass mechanisms are refuted by direct measurement on Marianna's SIFs, **with no
model in the loop**. Do not redo these.

| check | method | result |
|---|---|---|
| `/testbed` populated? | `apptainer exec <sif> ls /testbed` | **yes** — real repo (Orange3) inside the SIF, no clone, no network |
| at the BUGGY state? | `rev-parse HEAD` vs `rev-parse <base_commit>^` | **yes, HEAD == `base_commit^` exactly** |
| do agent edits reach the tests? | `Orange.__file__` + `Orange3.egg-link` / `easy-install.pth` | **yes** — develop/editable install pointing at `/testbed` |
| `/testbed` writable by the agent? | write test on fleet node `jrc0553` | **`WRITE_OK`** (fakeroot is unavailable on JURECA compute and is NOT needed) |
| does `chardet` abort `test.sh`? | real SIF + writable overlay | **no** — `Audited 1 package`, exit 0. **§4.1 RETIRED** |

The third mechanism deserves emphasis because it was thought to doom 55% of the pool: `TESTS_DIR=/r2e_tests`
lives *outside* `/testbed`, but the **code under test** still resolves to `/testbed` through the egg-link. Tests
living outside the repo is fine — arguably desirable, since the agent cannot edit them.

### 0.2 The reward is now a real measurement — proven

Three completed trials on `1251403`, three different repos. Each ran a genuine pytest with **exactly one
failing test — the specific bug named in that task** — and everything else passing:

```
r2egym-0000  1 failed,  9 passed   reward=0.0
r2egym-2040  1 failed, 12 passed   reward=0.0     (scrapy HttpCompression)
r2egym-1742  1 failed, 13 passed   reward=0.0
```

Reward `0.0` is **correct**: the agent never fixed the bug. Flipping that one test to PASSED yields `1.0`.
This is the inverse of the old failure, where the reward was ~61% `1.0` no matter what the model did.

### 0.3 The rate problem: the agent succeeds too rarely (measured)

`agentstall.py` (`~/ota-band/hpc/skyrl_standard/jupiter/agentstall.py`, deployed at
`/e/fscratch/reformo/lee27/agentstall.py`) over the completed trials:

```
trials analysed                        : 7
with a raw-JSON tool call left AS TEXT : 5   (71%)
with a ripgrep tool error              : 7   (100%)
that made >=1 successful edit/write    : 1   (14%)  <-- and that one scored 1.0
ended with reason=stop                 : 7
median steps                           : 2
```

**The 1-in-7 edit rate is the number to move.** It is also roughly what a band needs: with pass@4, a
per-trial success probability anywhere in the mid range puts most groups in band, so lifting the edit rate is
the direct lever on the band %.

**Two independent causes, both concrete:**

1. **`rg` (ripgrep) is not in the sandbox.** OpenCode's `glob` *and* `grep` tools shell out to ripgrep, so the
   agent's two primary code-search tools fail on its very first action, in **100%** of trials
   (`"error":"ripgrep execution failed"`). The tool map in harbor
   `src/harbor/environments/apptainer/worker.py` (~line 91) is a **hardcoded dict** — `opencode`, `tmux`,
   `asciinema`, `uv` — and each entry is bound from `$BRIDGE_AGENT_TOOLS/bin/<name>` to
   `/usr/local/bin/<name>`. `$BRIDGE_AGENT_TOOLS` is `/p/scratch/synthlaion/lee27/agent_tools`, which
   contains exactly `asciinema grading-python311 opencode tmux uv`. Fix = stage a **static x86_64 musl**
   `rg` into `.../agent_tools/bin/rg` **and** add `"rg"` to that dict. Needs a **fleet restart** to take
   effect (workers are long-running), so it is not free.
2. **Thinking is ON, and the 8B then emits unparseable tool calls.** Verified by BEHAVIOUR from the trial
   `config.json`: `agent.kwargs.interleaved_thinking=True` **and**
   `agent.kwargs.extra_body.chat_template_kwargs.enable_thinking=True`.
   ⚠ **But `BAND_HARBOR_THINK` is NOT what sets the model's behaviour — it is INERT for OpenCode.** Those
   two keys are implemented for `terminus_2` / `openhands` / `mini_swe_agent` only. OpenCode never touches
   harbor's LLM client; it shells out (`opencode.py:876`) to
   `opencode --model=… run --format=json --thinking --auto`, where `--thinking` merely includes thinking
   blocks in the JSON output. Unknown kwargs are silently swallowed (`BaseAgent.__init__` takes `**kwargs`
   and never reads them), so the config *looks* applied and does nothing. `gen_band_yaml.py:80-84` says so
   in a comment. The model thinks anyway because Qwen3's chat template defaults `enable_thinking=true`.
   The parser is **not** broken — real `tool_use` events do appear. What happens is
   the model emits `<think>` prose plus **malformed JSON**: an unterminated string, two concatenated JSON
   objects, and a stray `</think>`. vLLM cannot parse that into `tool_calls`, so OpenCode records it as a
   `text` part and the step finishes `reason:"stop"` — the trial ends after ~2 steps with no edit.

   **Cause 2 is still the cheaper one, but use the RIGHT knob:** `BAND_SERVER_NO_THINK=1`, which sets
   `engine_init_kwargs.default_chat_template_kwargs={"enable_thinking": False}` (`gen_band_yaml.py:179-187`)
   and reaches every request regardless of agent. That is vLLM **engine-construction** time, so it needs a
   **new RL job** — but still **no fleet restart**. Then re-run `agentstall.py` and gate on behaviour
   (no `<think>` in trajectory text), never on the config echo.
   Unverified link in this chain: the MarinSkyRL `9904058` plumbing of `engine_init_kwargs` into vLLM.
   Task #1 was once marked "thinking-off (DONE)" — wrong, reopened — and then the reopened version told
   you to set `BAND_HARBOR_THINK=0`, which was **also wrong** (a no-op). Two errors on the same knob.

## 1. DO THIS FIRST — sweep the TASK POOL with NO model (operator's call, 08-06 ~12:30 KST)

**The harness question is CLOSED (§0.0). The open question is how much of the 4,569-task pool is solvable.**

Luke's original instruction: *"validate the environments first without running the models by actually
submitting the oracles and the non-oracle answers."* Done — that's what §0.0 records. His follow-up call
after seeing the 32/4,569 denominator: **verify the tasks before spending more GPU on the model.** Rationale
is in §0.0; the short version is that unsolvable tasks are zero-gradient dead weight in every GRPO batch,
and the sweep's output is a permanent solvable-task filter.

**Concrete next steps, in order:**
1. Confirm staging validated (byte-match vs the original 32) → `envgate_big/` + `envgate_big_tasklist.txt`.
2. Run `envgate_par.sh` over the ~260-task sample against fleet `15502687`. **Set `FLEET=15502687`** — the
   script's default is hardcoded to the now-dead `15500584`. Raise concurrency past one-task-per-node if
   the wall is too long: the fleet runs **16 workers/node × 32 nodes**, so 512-way is available; the
   original one-per-node choice was conservatism, not a constraint.
3. `envgate_report.py` for HARNESS / GATE / CEILING, with the null-reward exclusion.
4. Decide on the full 4,569 sweep; its artefact is the verified task allowlist.
5. Only then re-probe the model: thinking **ON**, `rg` live.

**Cheap parallel item, no GPU, no fleet:** the tool-format hypothesis — compare the checkpoint's SFT tool
syntax against the `bare_json` parser. The invented `<argby>` wrapper is the hint. If they disagree, `rg`
was never going to matter.

**Do NOT re-run the 32 already-gated tasks expecting new information** — but DO note whether the new sample
overlaps them, because that changes how the two results combine.

**The oracle is clean and free here:** `/testbed` sits at `base_commit^`, so `git checkout <base_commit> -- .`
**is** the gold patch. No `solve.sh`, no `solution/patched_files` needed (those belong to the patched dataset).

```bash
# on JURECA compute, via the existing fleet -- no new allocation, no GPU:
bash /p/scratch/synthlaion/lee27/envgate.sh <task> pristine   # expect reward 0.0
bash /p/scratch/synthlaion/lee27/envgate.sh <task> oracle     # expect reward 1.0
```

`envgate.sh` mirrors harbor's worker (instance start + `exec instance://`, writable ext3 overlay,
`/tests` + `/logs/verifier` + `/workspace` binds) and emits one JSON line per run. 32 repo-diverse tasks are
already staged at `/p/scratch/synthlaion/lee27/envgate/<task>/{test.sh,metadata.json}`
(`/e/fscratch` is **not** visible from JURECA — that staging step is mandatory).

**A pristine `0.0` and an oracle `1.0` on the same task is THE GATE, with no model involved.** Scale it across
tasks to get the solvable-task denominator a band number is meaningless without. Status at handover: the
single-task bracket for `r2egym-0000` was **still running** — re-run it, do not assume.

**Known trap inside `envgate.sh`:** plain `apptainer exec` on the SIF cannot `cd /root` — `chdir` returns
**EPERM** even though `/root` is owned by us and writes succeed. That is why the script uses
`instance start` + `exec instance://`, as harbor does. If you rewrite it, keep that.

## 2. Live state at handover (RE-PROBE, do not trust)

| id | what | state at ~13:00 KST 08-06 |
|---|---|---|
| `15502687` | JURECA sandbox fleet, 32 nodes × 16 workers, dc-cpu | **RUNNING**, fresh 24 h. Started AFTER harbor `1019a36f` → should have `rg`. **Verify by behaviour.** |
| `15500584` | previous fleet | **CANCELLED** 08-06 ~12:40 KST — ours, idle, deliberate. Started 4 h before the `rg` fix landed, so it could never have had it. |
| `1252276` | 8B band probe, thinking OFF, 2 nodes, TP=1 | **TIMEOUT at 04:00:24** = its 4 h wall, intended end. n=14, conclusion in §0.0. |
| `1251403` | 8B × 32 tasks × pass@4 | ENDED `TIMEOUT` 01:30:02. Source of the n=7 thinking-ON baseline. |
| `1250164` | 8B mirrorfix | ENDED `TIMEOUT` 01:30:03. |

**Jupiter queue is EMPTY.** No GPU job is running. The fleet is idle and therefore free capacity for the
model-free sweep — that is a large part of why the sweep is the right next move rather than a detour.

**`1251403` lost its entire first wave of 64 rollouts to a dead reverse forward** (see §3.1). The environment
and reward conclusions above are unaffected — they come from trials that ran *after* the repair, plus
model-free SIF probes.

The band from this job will be **small but non-zero** — it only had ~35 min of working route and a 1-in-7
edit rate. Do not read a low band % here as a defect; read it as the edit rate, which §0.3 tells you how to
raise. **Distinguish the failure modes by the verifier output:** constant `1.0` with no edits was the old
free-pass bug; `0.0` with exactly one failing test is a correct measurement of an idle agent; and a `1.0`
alongside 4 `edit` calls is the pipeline working.

## 3. What broke tonight, and what now prevents it

### 3.1 A dead rollout forward that every existing check called healthy
`1251403` produced nothing for 20 minutes. The job log simply **froze at 868 lines with no error**.

- `arm_rollout_forward.sh` logged a clean install, and `ss` on JURECA showed `10.14.0.46:18300` **LISTENING**.
- But `curl` from a **fleet compute node** returned **`HTTP=000` in 1.5 ms** — refused.
- Meanwhile the *bridge* forward (`9923`) was HTTP 200, which is why envs still started and the bridge looked
  perfectly healthy. **This is rule #1: one forward up, one down ⇒ healthy bridge, 100% rollout timeouts.**
- A `-O cancel` + `-O forward` of the **byte-identical spec** fixed it instantly.

`ssh -R` binds the listener whether or not the forwarded-to target is reachable, and a stale registration on
the same port is indistinguishable by `ss`. **Fixed structurally** (commit `003a7b94`):
`arm_rollout_forward.sh` now issues a real `/v1/chat/completions` **from a JURECA compute node** (via
`srun --overlap` against the fleet, arg 3) and **exits non-zero on `HTTP=000`**. Any HTTP status from the
server is a PASS — a `400 Model name mismatch: loaded model name g1_diverse_tezos_100k_8b` is the proof you
reached the real vLLM.

⚠ **Unexplained, recorded as observation not mechanism:** why an identical spec was dead and then worked.
Also `~/.rollout_forwards` **did not exist** even though the 18:25 arm log said "recorded in" it — so the
recorded-spec safety net was absent. Find out what removes it.

### 3.2 `jrc` ran HALF your command on the WRONG CLUSTER and exited 0
This one invalidates evidence, so read it. `jrc` was `jup "ssh ... \"\$@\"" _ "$@"`, and **`ssh` joins all
its arguments into ONE remote command string**. So `jrc 'hostname; uname -m'` became, on Jupiter:

```sh
ssh -S cm jureca05 "$@" _ hostname; uname -m
```

`_ hostname` ran on **JURECA** (printing `_: command not found`, eating the hostname) and `uname -m` ran on
**JUPITER**, printing `aarch64`. Any payload containing `;`, `&&`, `||`, `|` or a newline silently executed
part of itself on the wrong cluster **and still exited 0**. Fixed by base64-encoding the payload (`42bd89e9`).
**A one-word payload is the only case the broken version got right — so distrust any pre-08-06 cross-cluster
claim that was verified with a single-word `jrc` command.**

Same commit: `trap ... RETURN` is **bash-only**, and zsh rejects it (`undefined signal: RETURN`), so it
printed two noise lines per call, never installed, and left `$TMPDIR/.jup.lock.d` behind on any interrupted
call — making every later call block the full 300 s. Now an age-based stale-lock break, and `_jup_locked`
propagates the real exit code instead of `rmdir`'s.

### 3.3 The false "auth outage", third recurrence — and it was self-inflicted
`Session open refused by peer` then `lee27@login02...: Permission denied (keyboard-interactive)`.
**Cause: my own background pollers exhausted the ControlMaster's session cap** — including one inherited
monitor calling **raw `ssh jupiter`**, bypassing `jup.sh`'s lockfile entirely.

- **DO NOT kill the Jupiter master.** It was established interactively with TOTP and `ControlPersist 8h`;
  killing it risks needing Luke for a re-auth. That is exactly the documented false "JuDoor key is rejected".
- **DO** stop every background poller. Access came back on the **first retry**, in under a minute.
- **Run at most ONE cluster poller, and only through `jup`/`jrc`.** `BatchMode=yes` did its job here: the
  refusal failed cleanly instead of burning TOTP tries.

### 3.4 Marianna's SIFs are amd64 — probe them ONLY from JURECA
From a Jupiter login node you get
`FATAL: While checking container encryption: could not open image ...: the image's architecture (amd64) could
not run on the host's (arm64)`. The "container encryption" prefix reads like a corrupt SIF; it is only an
arch mismatch. Her SIFs derive from the prebuilt `namanjain12/<repo>_final` images, which are x86_64-only,
so the JURECA `dc-cpu` fleet is **the only place they can run** — not a preference. (The *patched* dataset's
`python:X.Y` base was multi-arch and did run on Jupiter aarch64; do not carry that assumption over.)

### 3.5 `mirrorgate.py`'s CLONE check is INVERTED on the raw path
It prints `[1] CLONE ... -> FAIL (clone still failing -- mirrors not bound?)` when `attempted=0`. On the raw
path **`attempted=0` is the CORRECT outcome** — nothing should clone, the repo is already in the SIF
(`testbedcheck.py` says so explicitly: "should be 0 now"). Ignore that line, or fix the gate. Likewise, at
tiny n its `[3]`/`[4]` verdicts print `FAIL` off `n/a` denominators — **`FAIL` at n=1 is not a gate reading.**

## 4. Open risks, unmeasured

1. **We depend on another user's read-only dir** (`/p/scratch/transfernetx/nezhurina1`). Stable since March;
   copy the 4,568 SIFs (~3.7 TB against 36 TB free) if this becomes the main pool.
2. **Sandbox sizing** is 2 CPU / 4 GB / 8 GB, raised for the abandoned clone workaround. Nothing clones or
   compiles now, so it can likely come down — free throughput, but **not before the §1 gate passes.**
3. **10 tasks have no SIF** — already excluded from the name lists.
4. **Throughput unmeasured on the raw path.** Trials that stop early finish in ~3 min, which tells you
   nothing about a working agent. Do not extrapolate from it.

## 5. Retractions — do not re-derive

- **"358 of 4,578 ≈ 8%" is WRONG.** Marianna: the band is **~1.6k of 4.5k ≈ 36%** at `0 < pass@4 < 1`, and
  her filtering pass cost `18k rollouts` (= 4.5k × 4). 0-in-band reads as "consistent with 8%" (p≈0.47) but
  is ~1-in-800,000 against 36%.
- **"33% pass ≈ her 35%, so a learnable band EXISTS"** — compared GROUPS against TRIALS. A pass rate is not a
  band. The band is per-group: `0 < passes < n_samples`.
- **"Concurrency is the bottleneck ⇒ ~6× with no new nodes"** — we hit the configured cap instantly at
  32/64/128. What made trials slow was `max_turns: 999999` (she caps 50) plus heavy sandboxes.
- **"`1244916` died to a node failure"** — `sacct`: `CANCELLED by 34902`. Always `sacct -j <id> -X` first.
- **"Our JuDoor key is rejected"** — false, three times now. See §3.3.
- **"The bare_json parser is validated"** — over-stated. It **does** fire, but "median 2 tool calls/trial" was
  never healthy for a SWE task; it is the same 2-step stall documented in §0.3.
- **"thinking-off (DONE)"** — false. It is ON. See §0.3.
- **"`BAND_HARBOR_THINK=0` is the cheap test"** — false, it is a **no-op** for the OpenCode agent
  (`interleaved_thinking`/`extra_body` are wired only for terminus_2 / openhands / mini_swe_agent; OpenCode
  shells out to `opencode ... run --format=json --thinking --auto`, `opencode.py:876`, and unknown kwargs
  are silently swallowed). Acting on it would have burned a job on a confident false negative. Correct knob:
  `BAND_SERVER_NO_THINK=1`.
- **"78.1% is a hard ceiling"** — over-stated. It is a **point estimate from n=32 of 4,569 (0.7%)**, 95% CI
  ≈ **61–89%**. Quote the range. Likewise **"SWE-bench 479/500 = 96%"** is repo coverage, not instances
  gated (8 were). Both were **numerators reported without their denominators** — the same failure as the
  `ripgrep 0/14` empty-denominator trap, pointed the other way. State the denominator, always.
- **"the envgate WORKER default was fixed"** — it was fixed **locally only**. The cluster copy still ran the
  pinned Jul-29 `worker.py` until it was redeployed 08-06 ~12:50 KST. A "fixed" claim means **deployed and
  verified on the machine that runs it**, not committed locally.
- **"the fleet log went quiet, something is wedged"** — false. Five hours of silence was an **idle** fleet;
  its only client (`1252276`) had ended. Cross-check the client's `sacct` end time against the last log
  write before calling anything wedged.

## 6. Hard-won operational rules (unchanged, still binding)

- **TP=1 ONLY.** `inference_engine_tensor_parallel_size > 1` makes vLLM build a distributed executor that
  copies the parent actor's `os.environ` into a nested Ray `runtime_env`; Ray then asserts on its own
  `__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR` and **no engine starts**. SkyRL mislabels it
  `port collision (EADDRINUSE)` and burns 5×120 s. Gate on
  `grep -c RAY_WORKER_PROCESS_SETUP_HOOK <joblog>`. Cost: all 8 nodes of `1247578`.
- **Verify by BEHAVIOUR, never config.** Seven keys have been accepted, echoed and ignored
  (`ai_memory/DEAD_KEYS.md`); `strict_json_parser: False` is in the live config right now. A key is not set
  until a **log line or a trajectory** proves the consumer acted on it. Tonight's `interleaved_thinking=True`
  was caught this way.
- **Anchor your grep patterns.** The log echoes the whole hydra command line, so `FileNotFoundError` matches
  `RewardFileNotFoundError` inside `mask_exceptions=[...]`, and `grad_norm=1.0` matches `max_grad_norm=1.0`.
  Both produced phantom findings. Exclude the echo line or anchor on a line prefix.
- **Never probe the vLLM endpoint during weight sync** — params sit on the meta device; an inbound request
  kills an EngineCore and hangs the driver (cost: `1246702`). Wait for `Starting batch generation` or the
  first `trace_jobs/*/`. After that it is safe.
- **`/e/fscratch` for anything a job touches.** `/e/scratch` has an inode cap shared by 26 users.
  `/p/scratch` is **not mounted on Jupiter compute** but IS visible from the Jupiter **login** node and from
  JURECA — the login node is the only host that sees both, which is why every cross-cluster copy runs there.
- **The JURECA ControlMaster lives ON the Jupiter login node**, not the laptop. `ssh jureca` from a Mac
  failing is expected, not an outage.
- **`ssh -O cancel -R` matches the FULL spec including the connect address.** A wildcard or guessed IP
  silently no-ops and returns success.
- **Reward is at `verifier_result.rewards.reward`** — not top-level `reward`, which does not exist. What
  matters is per-group `len(set(rewards)) > 1`, never `avg_raw_reward`.
- **Do NOT rename a task dir or touch a Dockerfile** under `tasks/r2egym-raw` — SIFs are keyed
  `build_${task_name}-${sha256(Dockerfile):12}.sif`, so that breaks the free cache reuse (4,568/4,578 resolve).

## 7. Tooling — use it, don't re-derive it

| tool | what it does / enforces |
|---|---|
| `hpc/skyrl_standard/jupiter/jup.sh` | `source` it. `jup`/`jrc` serialise every cluster call through a lockfile and force `BatchMode=yes`. **Fixed 08-06** — see §3.2. `jup_bg`/`jrc_bg` detach into tmux ON the cluster. |
| `arm_rollout_forward.sh <job> <port> <fleet_jobid>` | arms the rollout forward and **verifies the route end-to-end from a compute node**. Pass arg 3 or it warns UNVERIFIED. |
| `band_preflight.sh <yaml> <wall> <fleet_jobid>` | hard pre-submit gate. Non-zero exit = DO NOT SUBMIT. **TODO: fold in the §3.1 end-to-end route probe.** |
| `/p/scratch/synthlaion/lee27/envgate.sh <task> <pristine\|oracle>` | **NEW** — model-free environment gate (§1). |
| `agentstall.py <trace_jobs>` | **NEW** — per-trial tools / ripgrep errors / edits / unparsed-JSON-as-text / last stop reason. This is how §0.3 was measured. |
| `band.py` / `band_report.py` | the gate ladder + band. **Never eyeball a band.** |
| `mirrorgate.py` | clone → `/testbed` → did passes edit a file → band. **Its CLONE verdict is inverted on the raw path (§3.5).** |
| `testbedcheck.py` | earliest signal: is `/testbed` non-empty, did anything try to clone. |

## 8. Launch recipe

```bash
source hpc/skyrl_standard/jupiter/jup.sh          # then use jup / jrc, never bare ssh

W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next ; D=$W
export BAND_MAX_TASKS=0 BAND_CONC=64 BAND_BRIDGE_PORT=9920
export BAND_SERVER_NO_THINK=1     # <-- THE knob that actually works for OpenCode (see §0.3).
export BAND_HARBOR_THINK=0        # <-- config honesty ONLY; this key is INERT for OpenCode.
export BAND_MAX_EPISODES=50 BAND_CPUS=2 BAND_MEM_MB=4096 BAND_STORAGE_MB=8192
export BAND_MAX_MODEL_LEN=40960 BAND_MAX_NUM_SEQS=1024 BAND_GMU=0.92
export BAND_TOOL_PARSER=bare_json BAND_TOOL_PARSER_PLUGIN=$D/rl/tool_parsers/bare_json_tool_parser.py
# do NOT set BAND_TP / BAND_ENGINES -- TP=1 is mandatory (§6)
python3 $D/hpc/skyrl_yaml/jupiter/band/gen_band_yaml.py \
  $D/hpc/skyrl_yaml/jupiter/band \
  /e/fscratch/reformo/lee27/band_raw_32_names.txt \
  /e/fscratch/reformo/lee27/tasks/r2egym-raw \
  1 <NODES> 18300

bash hpc/skyrl_standard/jupiter/band_preflight.sh \
  $D/hpc/skyrl_yaml/jupiter/band/band_probe_8b_p4_shard00of01.yaml <WALL> <FLEET_JOBID>
# non-zero exit => DO NOT SUBMIT

source /e/scratch/reformo/lee27/keys/secrets.env   # NOT ~/secrets.env
export SHARD=band_probe_8b_p4_shard00of01 NUM_NODES=<N> TIME_LIMIT=<WALL>
export APPTAINER_BRIDGE_URL=http://10.128.1.2:9920 SKYRL_CHUNKED_LOGPROBS=1
export JOB_NAME=<name>
bash hpc/skyrl_standard/jupiter/run_r2egym_band_probe_8b.sh

# immediately after submit -- NOTE arg 3, which is what makes the route verified:
jup "tmux new-session -d -s armfwd 'bash $W/hpc/skyrl_standard/jupiter/arm_rollout_forward.sh <JOBID> 18300 15500584'"
# then CONFIRM: grep -E 'ROUTE OK|FATAL' ~/arm_forward_<JOBID>.log
```

Keep `SKYRL_CHUNKED_LOGPROBS=1` (fixes the large-vocab backward OOM; 33.41 GiB peak vs stock OOM at the
identical 30.57 GiB, and 2.4× more accurate vs fp64). Confirm via `chunked gathered log-softmax ACTIVE`,
which only fires during a training step.

`TIME_LIMIT` ≤ `04:00:00` — a 6 h job can be deferred a day by a hidden maintenance window. Note
`scontrol update TimeLimit=` does **not** help a job whose reason is `Priority` rather than `Resources`.

## 9. Open items for Luke

1. **Decide the order:** (a) `BAND_SERVER_NO_THINK=1` relaunch — needs a NEW job (vLLM engine-construction
   time) but no fleet restart; tests the
   likelier cause; or (b) stage `rg` + patch harbor's tool map — needs a **fleet restart**. §1's model-free
   oracle sweep is independent of both and should run regardless.
2. Ask Marianna for her **~1.6k band task IDs** + band script as a cross-check (likely
   `/e/project1/jureap59/marianna/ot/dc-agent`, JSC-local; `marianna13/dc-agent` is 404).
3. Whether to copy her SIF cache into our scratch for durability.
4. **Agent choice:** Luke chose OpenCode; her band used **terminus-structured**. Our absolute band % may
   legitimately differ, so the criterion is "is there real within-group variance at all", not "= 36%".
   ⚠ Worth revisiting now that OpenCode's `glob`/`grep` are known broken in 100% of trials.

## 10. Branch / repo map

| repo | branch | holds |
|---|---|---|
| `OpenThoughts-Agent` fork `lukedhlee` | `lukedhlee/vista-moe-grpo-30b` | `ai_memory/` — this file, `gotchas.md`, `DEAD_KEYS.md` |
| `OpenThoughts-Agent` worktree `~/ota-band` | `lukedhlee/band-8b-subset` | `jup.sh` (`42bd89e9`), `arm_rollout_forward.sh` (`003a7b94`), `band_preflight.sh`, `gen_band_yaml.py`, `build_raw.py` |
| `harbor` fork `lukedhlee` (clone `~/harbor`) | `lukedhlee/apptainer-opencode-bridge` | the apptainer worker. **The `rg` fix goes here** (~line 91). Offline git mirrors are **inert on the raw path**. |

`origin` on `OpenThoughts-Agent` is `open-thoughts/` and we have **no push rights** — push to `fork`.

---

# SWE-BENCH VERIFIED — staging facts settled 2026-08-06 (read before touching it)

## What exists now
`/Users/lukedhlee/swebench-verified-tasks` on the Mac: **500 task dirs, 0 failures, 21 MB total**,
generated with `cd ~/harbor/adapters/swebench && uv run swebench --all --task-dir <out>`.
Per task: `instruction.md`, `task.toml`, `tests/{test.sh,config.json}`, `solution/solve.sh`,
`environment/Dockerfile`. Small enough to rsync anywhere; regenerating takes ~8 min.

## Facts that change the plan
- **There is no "validation split".** `princeton-nlp/SWE-bench_Verified` ships **only `test`, 500
  instances**, and the dataset id is **hardcoded** at `adapters/swebench/.../adapter.py:46` — there is no
  `--dataset`/`--split` flag. "SWE-bench val" = those 500. Do not go looking for another split.
- **The oracle is free.** Each task has `solution/solve.sh` = `patch --fuzz=5 -p1` of the dataset `patch`.
  So the model-free gate is `nop` vs `oracle`, with **no git trickery** — unlike r2egym, where the gold fix
  is `git checkout <base_commit> -- .` because `/testbed` sits at `base_commit^`.
- **500 tasks -> 500 DISTINCT base images** (`swebench/sweb.eval.x86_64.<id>:latest`, `__`->`_1776_`),
  verified: `grep -h ^FROM */environment/Dockerfile | sort -u | wc -l` = 500. **No base-image reuse trick
  exists here** (contrast SweSmith: 2500 tasks -> 38 SIFs). Images are **x86_64-only**, so this can only run
  on the JURECA/JUWELS x86 fleet — never on Jupiter (GH200/aarch64).
- **`worker.py` already carries SWE-bench-specific shims** (make_test_spec's GitHub fetch monkey-patched to
  no-ops, Django `chdir` EPERM, Astropy DeprecationWarning). Somebody debugged this path before — read
  `_patch_test_sh_for_offline_pip()` before writing anything new.
- All three verifier wheels are already staged in `$BRIDGE_AGENT_TOOLS/wheels`:
  `swebench-4.0.3`, `datasets-2.16.1`, `fastcore-1.10.5`. The offline verifier path is provisioned.

## Do NOT use harbor's `prebuild_sifs.sh` unmodified for this
It pre-pulls every `FROM` into its own `base_*.sif` **and** builds a second SIF from a generated `.def` —
with 1:1 bases that is **~2x disk for zero reuse** (~1-2 TB, not ~1 TB). Its `.def` also has a `%post`
(apt-get, pip) needing root/`--fakeroot`, and it hides every build error behind `2>/dev/null`.
Its sbatch header is Marianna's JUWELS `projectnucleus` account, and its proxy block sources
`/e/project1/jureap59/marianna/...` (a **Jupiter** path — verify reachability from JURECA before relying on it).

`hpc/skyrl_standard/jupiter/swebench_build_sifs.sh` (committed) does a **pull-only** build instead: those
Dockerfiles only add `uv` and `/logs` on top of the base, and harbor already puts `uv` on PATH from
`agent_tools` and bind-mounts `/logs/verifier`. **This is a hypothesis — verify on a small sample before
committing to 500.**

## Unverified cost risks — measure on a sample of ~5-10 FIRST, then extrapolate and report
1. **Disk**: 500 x ~1-2 GB. Check the `/p/scratch/synthlaion` quota before starting.
2. **Docker Hub rate limits**: anonymous pulls are throttled per-IP. 500 pulls may need auth or serialising
   over many hours. This, not compute, is likely the wall.
3. **Whether pull-only SIFs actually run** (`uv` on PATH inside the container).
Build a sample, time it, measure the SIF size, run the gate on it — only then launch the 500.
