# TMax relay pool: 3,119 tasks strict, 8,068 lenient, with TB-Hard removed

`tmax_strict.txt` and `tmax_lenient.txt` are the TMax tasks (`laion/TMax-15K-Harbor`, 14,601) left after the filter
in `notes/relay/scaleup_plan.md` ("TMax filter") and after removing every task that is a TB-Hard task. Both lists
hold task ids, one per line, sorted by id: every TMax budget is the same, so budget order means nothing here, and a
relay run should pass `SHUFFLE_SEED` so that a run cut early is still a uniform sample. Neither list has been through
the no-op gate yet. That gate (`data/tmax/daytona/README.md`) removes the tasks whose setup no longer builds or whose
verifier passes an untouched sandbox. Run the relay on the gate's `gate_pass.txt`, not on these lists.

## How the lists were cut (2026-09-29)

| step | lenient | strict |
|---|---:|---:|
| TMax, all tasks | 14,601 | 14,601 |
| minus the outside audit's 724 structural defects (instruction/verifier or instruction/environment mismatch, ambiguous, other) | 13,877 | 13,877 |
| minus its 1,592 answer leaks | 12,285 | 12,285 |
| strict only: minus weak verifiers that never check the answer (the audit's `keep_policy_b`) | | 5,082 |
| minus tasks Qwen3.5-9B solved on every try (at least 2 tries, all passed) | 8,130 | 3,136 |
| **minus TB-Hard's tasks** | **8,068** (62 dropped) | **3,119** (17 dropped) |
| minus instruction overlap with TB2.1 or CalibForge | 8,068 (0) | 3,119 (0) |
| minus tasks the relay router could not tell apart (first 400 instruction characters found in another task) | 8,068 (0) | 3,119 (0) |

The first four rows reproduce the plan's numbers exactly. The strict list equals the earlier `tmax_S4_ids.json` on
Jupiter minus the 17 TB-Hard tasks.

- **The audit.** `wAI-org/swerl-tmax-15k-rubric-gpt-5-6-sol` has one GPT-5.6 label per task, from a single pass. Strict
  keeps the labels `CLEAN` (1,125) and `VERIFIER-TOO-WEAK` where the mechanism still checks the answer (3,957).
- **Qwen3.5-9B.** From `qwen35_9b_k3.json`, a run of 3 tries per task that is about 70 % complete: 12,432 tasks got
  2 tries, 1,729 got 3 and 345 got 1. A task with one try or none stays in, even if it passed. In the strict list,
  2,221 tasks were never solved by Qwen3.5-9B, 831 once and 67 twice (of 3).

## TB-Hard is exactly 100 TMax tasks

TB-Hard (`Zhongzhi1228/Terminal-Bench-Hard` @ 9b4cddf6, 100 tasks) renames every task to `tbh_task_<16 hex>`, and the
hex is not a hash of the TMax id. So I matched by content.

- **Every TB-Hard task is a verbatim TMax task.** For all 100, the normalized instruction, `post_install.sh` and
  `test_final_state.py` are each identical to exactly one TMax task. `tmax_tbhard_map.tsv` maps each TB-Hard id to its
  TMax id.
- **No paraphrases.** Beyond those 100, no TMax instruction shares more than 2 % of its word 8-grams (or 5-grams) with a
  TB-Hard instruction. The flag threshold is 30 %, the same as the TB2.1 scout.
- **Shared fixtures are not overlap.** They are generator assets reused across hundreds of tasks (the `minicalc` C
  project is in 855). No kept task shares a rare fixture (one used by 10 or fewer TMax tasks) with a TB-Hard task.
- **Where TB-Hard sat in the filter.** It holds 65 weak-verifier, 16 answer-leak, 14 structural and 5 clean tasks, and
  Qwen3.5-9B solved 8 of them on every try. So only 62 were still in the lenient list and 17 in the strict one.
- **63 of TB-Hard's 100 are the heavy "intricate" v2 kind.** That is why only 74 v2 tasks remain in the lenient list
  and 43 in the strict one.

## Worth knowing before choosing strict or lenient

- **Strict** keeps only tasks whose verifier checks the answer. **Lenient** adds 4,949 weak-verifier tasks.
  - 3,773 of them test a fixed set of inputs whose answers could be hard-coded.
  - The rest check only the output's shape (489), grep for a keyword (272), trust a self-reported value (261) or other
    (140).
  - A relay pass on these can be a wrong or hard-coded answer that the verifier accepts.
- **Secondary answer leaks.** 216 strict tasks and 1,111 lenient tasks carry `ANSWER-LEAK` as the audit's *secondary*
  label, for example an oracle binary in the image that the agent could copy. Dropping them too leaves 2,903 strict.
  They are kept here because the documented filter uses the primary label only.
- **Outside hosts.** 25 strict and 64 lenient tasks have tests that reach outside hosts (`tests_ext_hosts.json`).
  Daytona has internet, but these verifiers can fail when the host is slow.

## Files

- `tmax_strict.txt`, `tmax_lenient.txt`: the lists.
- `tmax_tbhard_map.tsv`: TB-Hard id → TMax id.
- `tmax_strata.tsv`: every TMax task with its audit label, mechanism, secondary leak, source (legacy/v2), domain,
  Qwen tries and passes, TB-Hard flag, outside-host flag, and whether it is in each list.
- `tmax_counts.json`: the step counts above.
- `daytona/select_tasks.py` rebuilds all of these from the inputs below:
  - the audit parquet, `qwen35_9b_k3.json` and `tests_ext_hosts.json`, all in
    `/e/data1/mmlaion/lee27/tmp_overlap/tmax/` on Jupiter;
  - the TMax harbor tree (`laion/TMax-15K-Harbor` `harbor.tar.zst`, sha256 `71b01eb0…`);
  - TB-Hard's `tasks/` directory;
  - the TB2.1 and CalibForge instruction parquets in `.../tmp_overlap/scout_tb/corpora/`.
