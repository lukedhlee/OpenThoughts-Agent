# Teacher interventions: write, replay and run your own triggers

In a relay episode the student works until something says "the teacher should step in now". That something is a
**trigger**. This folder and `data/relay/router/` let you plug in your own trigger and choose how the teacher steps
in, without editing the router. Each intervention is three separate choices:

- **When** (the trigger): a detector that looks at the episode and returns a decision, or nothing. It can be a rule,
  an LLM judge, a learned score or a fixed plan.
- **Where**: `rewrite` holds the student's reply before its command runs, throws it away, and has the teacher answer
  the same request, so the faulty command never runs. `next` lets the command run and has the teacher answer the next
  request, to recover from it.
- **How long** (the span): the teacher answers `k` turns (1, 2, 3, ...) and hands back, or `-1` keeps it to the end.
  A **release** trigger can end an intervention early, for example "the teacher stays until a command succeeds".

**One rule follows from how the sandbox works.** A trigger can only `rewrite` if it sees the fault in the student's
*proposed* reply. Faults that only show in a command's *result* (an error, a timeout, a loop) are seen after the
command ran, and a sandbox command can't be undone, so request-time triggers always act as `next`. The config loader
rejects a request-time `rewrite`.

Work in this order: replay offline first (minutes on a laptop, no GPUs), then shadow-test online (the trigger logs
but never acts), then let it act.

## 1. Replay a config on recorded episodes

```bash
python data/relay/triggers/replay.py --runs <run dirs or a manifest.json> \
  --config data/relay/triggers/replay_example.yaml --groups student_only_t2,student_only_msa --out out/
```

It rebuilds every turn of each recorded episode in order, asks your triggers exactly as the router would, and prints
one row per trigger: how often it fires on passing vs failing episodes, how early it fires before failures, and its
precision and recall for "fired" vs "failed". `out/fires.jsonl` has every fire with its reason, score and `info`;
`out/per_episode.jsonl` has the first fire per trigger. In a notebook: `await replay.areplay_episode(ep, 'my.yaml')`.

- **Clean labels come from student-alone episodes** (the `student_only` arm). In a relay episode, everything after
  the first real intervention reflects the teacher, so replay scores relay episodes only up to that point and marks
  later fires `valid: false`.
- **Recorded runs:** any relay run dir works (it needs `turns.jsonl` and `bodies/`). Ask Luke for the 160-episode
  sample bundle, or record your own with a `student_only` run (section 4).
- `--builtin-check` reproduces the router's own logged `would_fire` (100 % on the sample): a sanity check that replay
  sees what the router saw.

## 2. Write a trigger

A function is enough. `state` is a dict that lives for one episode:

```python
from interventions import Decision, trigger

@trigger(phases='request')                      # asked before the student answers, with the last command's result
async def two_errors(ev, state):
    r = ev.last_result
    failed = r is not None and (r.timed_out or (r.exit_code not in (None, 0)) or 'Traceback' in r.output)
    state['n'] = state.get('n', 0) + 1 if failed else 0
    if state['n'] >= 2 and ev.turn > 3:
        return Decision(reason=f'{state["n"]} failed commands in a row', score=1.0, info={'cmd': r.command})
```

Reference it from YAML as `use: path/to/file.py:two_errors` (relative to the YAML) or `use: package.module:two_errors`.
A class works too: it needs `name`, `phases` and `async observe(self, ev)`, and optionally `reset()`. See
`examples/two_errors.py`, `examples/stay_until_fixed.yaml` and `examples/judge.yaml`.

**What a trigger sees** (`Event`, built only from the past):

| field | what |
|---|---|
| `phase` | `request` (before the student answers) or `reply` (the student answered; its command has not run) |
| `turn`, `is_retry`, `owner` | the turn number, whether harbor resent the request, who answers it |
| `messages` | the harness history, as the student sees it |
| `last_result` | request phase: the previous command, `exit_code` (msa; Terminus-2 has none), `timed_out`, `output`, `truncated` |
| `reply` | reply phase: `text`, the parsed `action`, `parse_error` |
| `clock` | the student's own time used and the task's budget (teacher turns excluded) |
| `view_tokens()` | async: the student's view size in tokens |
| `scanner` | the built-in feature scanner (loops, waits, error streaks) |
| `interventions` | the episode's interventions so far |
| `extra` | `harness`, `task_id`, and router hooks such as `call_endpoint(name, body)` |

**What it returns:** `None`, or a `Decision`. Every field is optional except that a decision must mean something:
`reason`, `score`, `where`, `span`, `teacher_note` (shown to the teacher only, never trained on), `log_only`,
`kind` (`intervene` or `release`), `view`, `teacher` (an endpoint name), `info` (anything you want logged). Settings in
the YAML entry override the decision's `where` / `span` / `view` / `teacher`, so the same trigger can be swept.

## 3. Configure

```yaml
endpoints:                        # optional: extra model servers by name (the default teacher is `teacher`)
  big_teacher: {url: http://c101-003:8000/v1, model: qwen38}
policy:                           # episode-wide limits and views
  cooldown: 3                     # student turns between interventions
  max_interventions: 4
  earliest_turn: 2
  student_view_after_handback: capped    # how the teacher's turns look to the student afterwards: full | capped | stripped
  failed_turn_view: keep          # next: show the teacher the student's failed turn, or hide it
  show_rejected_reply: false      # rewrite: show the teacher the reply it replaces
triggers:                         # list order is priority; every fire is logged, the first one allowed to act acts
  - use: path/to/file.py:two_errors
    where: next
    span: -1
    release: {use: action_ok}     # hand back as soon as one of the teacher's commands succeeds
    teacher_note: "The last two commands failed. Find out why before continuing."
  - use: all_of                   # combinators nest
    params: {of: [{use: loop}, {use: after_turn, params: {turn: 10}}]}
    span: 2
    teacher: big_teacher
  - use: llm_judge
    log_only: true                # shadow: logged, never acted on
    params: {endpoint: judge, prompt_file: prompts/stuck.txt, threshold: 0.7, every: 2, max_calls: 15}
```

Per-entry keys: `use`, `name`, `params`, `where`, `span`, `phases`, `log_only`, `teacher_note`, `release`, `view`,
`teacher`, and `policy` (`cooldown`, `max_interventions`, `earliest_turn` counted over this trigger's own
interventions; `observe: always | armed`, where armed skips asking it when it could not act anyway, which saves judge
calls; `on_retry`; `owners`, to ask it on the teacher's turns too). Unknown keys are errors that name the entry.

**Built-ins**

| name | phase | fires when |
|---|---|---|
| `action_error` | request | the last command failed `streak` times: non-zero exit, timeout, or an output `patterns` match |
| `action_ok` | request | the last command succeeded (use it as a `release`) |
| `loop`, `no_progress_wait`, `edit_failed`, `error_streak` | request | the calibrated behavior detectors the router has always used |
| `context_budget`, `time_budget` | request | the student's view passes `tokens`, or it used `frac` of its time |
| `assist` | request | msa: a command timeout, a failing streak, repeated output or a context check-in (one teacher turn) |
| `parse_repair`, `repair_confirm` | reply / request | the harness can't parse the student's reply |
| `done_claim` | reply | the student claims it is done (`first_only` for the first claim only) |
| `gave_up`, `destructive`, `submit_check`, `success_claim_unchecked`, `success_contradicted` | reply | patterns in the proposed reply |
| `llm_judge` | request (or reply) | an LLM scores the student's recent turns over `threshold`; it only sees what the student saw |
| `scripted` | request | fixed `turns`, or a per-task `plan` (controlled experiments) |
| `any_of`, `all_of`, `streak`, `after_turn`, `before_turn`, `every` | any | combinators over other entries |

## 4. Run it online

Point the relay launcher at your YAML with `TRIGGER_CONFIG`. In the arms that run it, it replaces the legacy
intervention knobs: `ASSIST`, `TIME_TAKEOVER`, the context budget, the takeover-trigger knobs and, in relay arms,
`VERIFY_NOTE` (give a `done_claim` entry a `teacher_note` instead). The `relay_repair` arm keeps its parse-error
repair: when the harness can't parse the student's reply, the teacher answers that turn instead, as without a
config. Its `parse_repair` (and on Terminus-2 `repair_confirm`) goes in front of your triggers, unless your config has
its own `parse_repair` entry; then yours is used. The run dir keeps a checked copy of the config and the `path/file.py`
triggers it names in `trigger_config/`, with `SHA256SUMS`, and `run.meta` records the sha. A judge's `prompt_file` is
read where it is, so give it an absolute path.

```bash
# $S = the serve job from serve_submit.sh (data/relay/README.md, section 3)
TRIGGER_CONFIG=$PWD/my.yaml ARMS="student_only" NODES=4 CAP_NODE_H=40 \
  bash data/relay/horizon/launch_driver.sh $S my_shadow_run     # shadow: the student plays, your triggers only log
TRIGGER_CONFIG=$PWD/my.yaml ARMS="student_only relay_repair" NODES=4 CAP_NODE_H=40 \
  bash data/relay/horizon/launch_driver.sh $S my_relay_run      # shadow and acting arms on the same tasks
```

The `control` arm (the teacher alone) ignores `TRIGGER_CONFIG`, so it stays the same baseline in every run.

- **Shadow first.** In the `student_only` arm the student plays every turn and your triggers only log (`decisions` in
  `turns.jsonl`). That gives live fire rates and labels, which replay can then score.
- **Then act.** In a relay arm the router acts on your config. Every teacher turn's record carries
  `intervention = {id, trigger, where, span, i, teacher, view}`, every fire is in `decisions`, and `events.jsonl` has
  `intervention` and `handback` events. `pilot/readout.py` reports interventions per trigger, and the SFT renderers
  train every teacher turn and mask every student turn around any number of interventions.
- **Offline against the real router:** `python data/relay/router/router_replay.py run --trigger-config my.yaml ...`
  pushes recorded traffic through the live router code with your config.

## 5. Things that bite

- **A judge in the reply phase sits on every turn.** The student's reply waits for the judge before harbor sees it.
  In the request phase the router can start the judge early, so it overlaps the command's run time. Use `every`,
  `max_calls` and `observe: armed` to bound cost. Judge latency and tokens are in each fire's `info` and in the
  judge's own log.
- **The judge only sees what the student saw.** Teacher reasoning, verify notes and teacher-only text are removed
  before the prompt is built. Don't put task tests or grader files in a prompt; that would make a trigger look far
  better than it is.
- **Retries.** When harbor resends a request, the router answers it the same way as before and doesn't ask
  triggers again, so a trigger's per-episode `state` sees each turn once.
- **Clock.** The student's time budget counts only the student's own turns. The episode deadline is wall time.
- **Without a config nothing changes.** With no `TRIGGER_CONFIG` the router builds today's behavior from its flags,
  byte for byte.

Code: `data/relay/router/interventions.py` (types, controller, built-ins, loader), `router_triggers.py` (router-side
built-ins, the legacy config), `judge.py` (the LLM judge), `action_results.py` (the normalized command result),
`replay.py` here. Design and status: `notes/ot-agent/relay-triggers/`.
