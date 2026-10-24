# Mixed Vanilla / Relay SFT evaluation

This pipeline evaluates the two corrected-stack mixed-data models on TB2.1 (88 tasks), SWE Verified random100,
and TB-lite (99 valid tasks). Each model gets eight independent repetitions: 4,592 valid slots in total.
The broken Git-forensics fixture is excluded from both models.

The policy comes from the frozen original Vanilla repetition-one configs. Model, prompts, sampling, context/output
limits, agent/setup/verifier budgets, task instructions, task metadata and environment contents are preserved.
The deployed stdin transport, terminal-error verification, grader copies, managed Python/Java Docker startup,
and CLI pip-cache reclamation run from the first attempt. The snapshot adapter creates sandboxes only from
existing names. It cannot auto-create, delete, or rebuild snapshots, or fall back to a declarative sandbox build.

The 50:50 and 100+100 models use paired complete-case panels with eight numeric scores per task and model.
The readout estimates pass@1/3/4 without replacement and retains raw partial rewards and all unknown slots.
Historical Vanilla/Relay results are shown separately: their repairs were applied only to previously unknown slots,
so their protocol and panels differ from these first-attempt repaired-stack evaluations.

## Preparation and gates

Use an isolated OTA checkout descended from `1c40b0bf`; Harbor must be exactly `62e1c7b9`.
The data-phase inventory checks all original task instructions, metadata and environments against the selected copies.
The selected copies are the ten deployed SWE fixes, query and filter fixes, and three auxiliary fixes. Additional
undeployed fixes are intentionally absent. The Requests copy contains the restored official runner inside the
network-fixture wrapper.

Before launch, run the unit suite, the complete metadata-only plan and the three provisioning-only live probes:

```bash
PYTHONPATH="$HARBOR/src:$PWD" python -m pytest -q data/relay/horizon/mixed_eval/test_runtime.py
PYTHONPATH="$HARBOR/src:$PWD" python -m data.relay.horizon.mixed_eval.run \
  --root "$EVAL_ROOT" --training-root "$TRAIN_ROOT" --harbor "$HARBOR" --marin "$MARIN_ROOT" --dry-run
PYTHONPATH="$HARBOR/src:$PWD" python -m data.relay.horizon.mixed_eval.probe_adapter \
  --direct /path/to/query-optimize --python /path/to/prediction-model-evaluation \
  --java /path/to/okhttp-trailers-crash --harbor "$HARBOR" --output /path/to/gate-artifacts
```

The live probes need the existing eval credentials in the environment. They create and delete only their own three
sandboxes, guard against automatic snapshot resolution, and write their sandbox IDs and cleanup results. They do
not score or retry any benchmark slot. A successful live probe is a provisioning gate, not model performance.

The operator writes `$EVAL_ROOT/launch_gate.json` after reviewing the results, with `passed: true`, the exact
`ota_commit`, `harbor_commit: "62e1c7b9"`, `snapshot_operations: "none"`, and four true fields:
`adapter_unit_gate`, `live_direct_snapshot_gate`, `live_python_dind_gate`, `live_java_dind_gate`.
The watcher refuses to start without that receipt, or if its code/tasks/configs change while waiting for exports.

## Persistent watcher

Run the same command without `--dry-run` inside the experiment's own tmux session. It waits for both training
exports and their existing export-success markers. Serving uses 16 jobs of five nodes, at most 80 nodes and
1,280 concurrent trials. Drivers are overlapping steps inside those jobs; they consume no separate job slots.
Each driver has a distinct bridge port. Driver starts are paced five seconds apart.

Admission counts all of Luke's reserved nodes, keeps submitted jobs at most 20, uses the existing shared SU guard,
and conservatively adds all seats reserved by this controller's live serves to the shared org's started count
before admitting another group under 2,000. The queue may therefore use fewer than 80 nodes at once.

Completed drivers write `settled.json`; after all groups settle, the controller writes `results.json` and
`results.md`, including numeric coverage and unknown slots. Failed groups also get a partial readout. Original
results and canonical task trees are never changed. A controller restart with existing submissions stops for
explicit adoption review rather than starting duplicate model attempts.

The watcher records one infrastructure receipt per driver at 15 minutes, or earlier if the driver finishes.
`infra_health_15min.json` includes local srun liveness, the current model endpoint, task/result/trajectory counts,
and agent-log bytes. Lack of progress is flagged as ambiguous while allowing the original 600-second environment
and 360-second agent-setup budgets. Checks only record flags; they never cancel a run or launch a duplicate attempt.
