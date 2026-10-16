#!/usr/bin/env python3
"""Router-side triggers and the router's trigger config (stage 2 of notes/ot-agent/relay-triggers).

The relay router asks one interventions.Controller per episode who answers each request. Without --trigger-config it
builds the Controller's config from its own flags (legacy_config): relay-router/1's hard-coded decisions re-expressed,
in today's order, so routing and every turns.jsonl record stay as they were. With --trigger-config <yaml> that file
replaces those flags (the router refuses both, LEGACY_FLAGS). --mode teacher / student never hand an episode over, so
there every trigger only logs (shadow()).

Triggers that need the router's view of an episode live here and join interventions' built-ins when this module is
imported (the router imports it), so a YAML config names them like any other built-in:

  assist   msa --assist (was Router.assist_trigger): a one-turn teacher intervention before the student answers, on a
           command timeout, a failing-command streak, a repeated output, or a context milestone (each milestone once,
           used up only when the assist acts).

    python router_triggers.py <relay_router.py flags>    # the config those flags stand for, as YAML
    python router_triggers.py stage <yaml> <dir>         # a run's copy of a config (run_pilot.sh TRIGGER_CONFIG)
"""
import dataclasses
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'triggers'))
import relay_triggers as rt  # noqa: E402
sys.path.insert(0, HERE)
import interventions as iv  # noqa: E402

# relay_triggers' takeover triggers in env_fires / dec_fires order (today's evaluation order: the first fire wins)
ENV_TRIGGERS = ('edit_failed', 'loop', 'no_progress_wait', 'error_streak')
DEC_TRIGGERS = ('submit_check', 'gave_up', 'success_claim_unchecked', 'success_contradicted', 'destructive')
# the flags a --trigger-config replaces: (argparse dest, option); --verify-note(-text) only in --mode relay, where the
# done claim's note is its teacher_note (--mode teacher's note on its own first confirmation stays a router flag)
LEGACY_FLAGS = (('takeover', '--takeover'), ('repair_on_parse_error', '--repair-on-parse-error'),
                ('context_budget_tokens', '--context-budget-tokens'),
                ('context_budget_min_turn', '--context-budget-min-turn'),
                ('time_takeover_frac', '--time-takeover-frac'), ('assist', '--assist'), ('assist_ctx', '--assist-ctx'),
                ('assist_error_streak', '--assist-error-streak'), ('assist_gap', '--assist-gap'),
                ('assist_max', '--assist-max'))
RELAY_NOTE_FLAGS = (('verify_note', '--verify-note'), ('verify_note_text', '--verify-note-text'))


def tool_results(messages):
    """[(returncode or None, output)] of every tool message, oldest first."""
    out = []
    for m in messages:
        if m.get('role') != 'tool':
            continue
        c = m.get('content')
        if isinstance(c, list):
            c = '\n'.join(p.get('text', '') for p in c if isinstance(p, dict))
        c = c or ''
        rc, o = None, c
        try:
            j = json.loads(c)
            rc, o = j.get('returncode'), str(j.get('output', ''))
        except (ValueError, AttributeError):
            mm = re.search(r'<returncode>(-?\d+)</returncode>', c)
            rc = int(mm.group(1)) if mm else None
        out.append((rc, o))
    return out


class Assist:
    """msa --assist (relay): a one-turn teacher intervention before the student answers this request; the student keeps
    the episode. Calibrated on msafin's solo MSA2 episodes (2026-10-07): MSA2 failures are drift, not loops, so the
    triggers are a command timeout (the 30 s cap), a failing-command streak or a repeated output, and one check-in at
    each context milestone (ctx, student-view tokens: Event.view_tokens). The reason is the assist kind. The router's
    legacy config runs it with span 1, cooldown --assist-gap - 1, max --assist-max, earliest turn 2, observe armed, and
    on_retry observe: a retry of an assisted request asks the budgets again first (today's order), and when none acts
    the router has the teacher answer it again."""
    name = 'assist'
    phases = frozenset({'request'})

    def __init__(self, ctx=(20000, 35000), error_streak=4):
        self.ctx, self.error_streak = list(ctx or ()), error_streak
        self.reset()

    def reset(self):
        self.milestones = set()           # context check-ins already used
        self._offered = None

    async def observe(self, ev):
        self._offered = None
        if ev.is_retry:                   # a retry never starts an assist (the router answers a retry of an assisted
            return None                   # request with the teacher again)
        msgs = ev.messages
        if msgs and msgs[-1].get('role') == 'tool':
            res = tool_results(msgs)
            rc, out = res[-1]
            if 'timed out' in out[:600].lower() or rc in (-1, 124, 137):
                return iv.Decision(self.name, 'cmd_timeout')
            k = self.error_streak
            if len(res) >= k and all(r not in (0, None) for r, _ in res[-k:]):
                return iv.Decision(self.name, 'error_streak')
            if len(out) >= 30 and sum(1 for _, o in res[-6:-1] if o == out) >= 2:
                return iv.Decision(self.name, 'same_output')
        if self.ctx:
            n = await ev.view_tokens()
            for m in self.ctx:
                if n is not None and n >= m and m not in self.milestones:
                    self._offered = m
                    return iv.Decision(self.name, f'ctx_checkin_{m // 1000}k', score=float(n))
        return None

    def on_outcome(self, rec):
        if rec.outcome == 'acted' and self._offered is not None:
            self.milestones.add(self._offered)
        self._offered = None


ROUTER_BUILTINS = (Assist,)


def register():
    """Add the router's triggers to interventions' built-ins (done on import)."""
    for cls in ROUTER_BUILTINS:
        iv.BUILTINS[cls.name] = cls


register()


def legacy_config(a, parser_loaded, note_text):
    """The trigger config today's flags stand for (see interventions' module docstring for the mapping), in today's
    evaluation order. parser_loaded: Terminus-2's parser is loaded (--repair-on-parse-error or --teacher-format-guard),
    which is when Terminus-2 repairs parse errors; note_text: the verify note (--verify-note)."""
    msa = a.harness == 'msa'
    enabled = rt.CALIBRATED_CONFIG['enabled'] if a.takeover is None else [t for t in a.takeover.split(',') if t]
    env = [n for n in ENV_TRIGGERS if n in enabled]
    dec = [n for n in DEC_TRIGGERS if n in enabled]
    done = 'done_claim' in enabled
    claim = dict(use='done_claim', params=dict(harness='msa') if msa else dict(confirm_kind='confirm'))
    if a.mode != 'relay':
        # would-fire only (shadow() makes every entry log_only on both owners' turns): the takeover triggers, the done
        # claim on the episode's first claim as the scanner fires it
        first = dict(use='done_claim', params=dict(harness='msa' if msa else 'terminus2', first_only=True))
        return dict(triggers=[dict(use=n) for n in env + dec] + ([first] if done else []))
    T = []
    if parser_loaded and not msa:
        T.append(dict(use='repair_confirm', span=1, policy=dict(earliest_turn=1, on_retry='observe')))
    T += [dict(use=n, span=-1, policy=dict(observe='armed')) for n in env]
    if a.context_budget_tokens:
        # asked from turn 1 (today's count is logged on every student request); the fire itself waits for min_turn
        T.append(dict(use='context_budget', params=dict(tokens=a.context_budget_tokens,
                                                        min_turn=a.context_budget_min_turn),
                      span=-1, policy=dict(earliest_turn=1, on_retry='observe', observe='armed')))
    if msa and a.time_takeover_frac:
        T.append(dict(use='time_budget', params=dict(frac=a.time_takeover_frac), span=-1,
                      policy=dict(earliest_turn=a.context_budget_min_turn, on_retry='observe', observe='armed')))
    if msa and a.assist:
        T.append(dict(use='assist', params=dict(ctx=list(a.assist_ctx), error_streak=a.assist_error_streak), span=1,
                      policy=dict(cooldown=max(0, a.assist_gap - 1), max_interventions=a.assist_max, earliest_turn=2,
                                  observe='armed', on_retry='observe')))
    if (a.repair_on_parse_error if msa else parser_loaded):
        T.append(dict(use='parse_repair', where='rewrite', span=1, policy=dict(earliest_turn=1, on_retry='observe')))
    note = dict(teacher_note=note_text) if a.verify_note else {}
    if msa and done:
        T.append(dict(claim, where='rewrite', span=-1, **note, policy=dict(earliest_turn=1)))
    T += [dict(use=n, where='rewrite', span=-1, policy=dict(observe='armed')) for n in dec]
    if done and not msa:
        T.append(dict(claim, where='next', span=-1, **note, policy=dict(earliest_turn=1)))
    return dict(policy={}, triggers=T)


def shadow(specs):
    """--mode teacher / student: the router never hands an episode over, so every trigger only logs, on both owners'
    turns (today's would_fire)."""
    return [dataclasses.replace(s, log_only=True, owners=frozenset(iv.OWNERS)) for s in specs]


EndpointError = iv.EndpointError          # where triggers catch it: interventions.EndpointError


def endpoints_of(path, default_model):
    """The `endpoints:` of a trigger config file: {name: (urls, served model)}. A value is a url, a list of urls, or
    {url: <url or list>, model: <served name>} (default --teacher-model). 'student' and 'teacher' are the router's own
    and cannot be redefined."""
    import yaml
    with open(path) as f:
        raw = (yaml.safe_load(f) or {}).get('endpoints') or {}
    out = {}
    for name, v in raw.items():
        if name in ('student', 'teacher') or not isinstance(name, str) or not re.fullmatch(r'[\w.-]+', name):
            raise iv.ConfigError(f"endpoints: {name!r} is not a free name ('student' and 'teacher' are the router's own)")
        if isinstance(v, (str, list)):
            v = dict(url=v)
        if not isinstance(v, dict) or not v.get('url') or set(v) - {'url', 'model'}:
            raise iv.ConfigError(f'endpoints.{name}: a url, a list of urls, or {{url, model}}')
        urls = v['url'] if isinstance(v['url'], list) else str(v['url']).split(',')
        out[name] = ([str(u).strip().rstrip('/') for u in urls if str(u).strip()], v.get('model') or default_model)
    return out


def load_trigger_config(path):
    """A --trigger-config file as the router loads it: `use: llm_judge` is the judge (judge.py) while it loads, and
    stage 0's placeholder again afterwards (nothing leaks into the shared registry)."""
    import judge  # pyright: ignore[reportMissingImports] - next to this file
    old = judge.register()
    try:
        return iv.load_config(path)
    finally:
        iv.BUILTINS['llm_judge'] = old


def endpoint_names(policy, specs):
    """(trigger name, endpoint) for every trigger of a config that names a model endpoint (an `endpoint` that is not a
    URL): main-list entries, their releases, the policy's releases, and combinators' children."""
    todo = [(s.name, s) for s in specs] + [(s.name, s) for s in policy.release]
    todo += [(f'{s.name}.release', s.release) for s in specs if s.release is not None]
    while todo:
        name, s = todo.pop(0)
        t = s.trigger
        e = getattr(t, 'endpoint', None)
        if isinstance(e, str) and e and not e.startswith(('http://', 'https://')):
            yield name, e
        todo += [(f'{name}.{c.name}', c) for c in getattr(t, 'children', None) or ()]


def wants_parser(specs):
    """A config that repairs parse errors needs Terminus-2's parser loaded."""
    return any(isinstance(s.trigger, iv.ParseRepair) for s in specs or ())


def scanner_names(specs):
    """The relay_triggers takeover triggers of a config, in CALIBRATED_CONFIG's order (they stand for --takeover in the
    logged signals and would_fire)."""
    have = {getattr(s.trigger, 'name', None) for s in specs}
    return [n for n in ('done_claim',) + ENV_TRIGGERS + DEC_TRIGGERS if n in have]


def conflicts(a, default):
    """The legacy intervention flags set alongside --trigger-config (default(dest) -> the flag's default)."""
    flags = LEGACY_FLAGS + (RELAY_NOTE_FLAGS if a.mode == 'relay' else ())
    return [opt for dest, opt in flags if getattr(a, dest) != default(dest)]


def stage_config(src, dest):
    """A run's own copy of a trigger config (run_pilot.sh TRIGGER_CONFIG): the YAML and the `path/file.py` triggers it
    names relative to itself go into dest (same relative paths), the copy must load, and dest/SHA256SUMS lists every
    file the config uses (an absolute path is recorded, not copied). A relative trigger file must resolve (after `..`
    and symlinks) inside the config's directory, and every write must land inside dest: otherwise ConfigError, checked
    before anything is written. Returns the copy's path."""
    import hashlib
    import shutil
    import yaml
    src, dest = os.path.abspath(src), os.path.abspath(dest)
    with open(src) as f:
        raw = yaml.safe_load(f) or {}
    files: dict[str, tuple[str | None, str]] = {os.path.basename(src): (None, src)}   # rel: (the entry's `use`, file)
    todo = [raw]
    while todo:                           # every `use:` (entries, releases, combinators' children)
        x = todo.pop()
        if isinstance(x, dict):
            u = x.get('use')
            if isinstance(u, str) and u.rpartition(':')[0].endswith('.py'):
                p = u.rpartition(':')[0]
                files[p] = (u, p if os.path.isabs(p) else os.path.join(os.path.dirname(src), p))
            todo += list(x.values())
        elif isinstance(x, list):
            todo += x
    root, droot = os.path.realpath(os.path.dirname(src)), os.path.realpath(dest)

    def inside(p, r):
        return os.path.commonpath([p, r]) == r

    for rel, (u, path) in files.items():
        if not os.path.isfile(path):
            raise iv.ConfigError(f'{src}: {rel} does not exist')
        if os.path.isabs(rel):
            continue
        if u is not None and not inside(os.path.realpath(path), root):
            raise iv.ConfigError(f"{src}: use: {u} resolves to {os.path.realpath(path)}, outside the config's directory "
                                 f'{root}; move the file under it, or name it by an absolute path (recorded, not copied)')
        if not inside(os.path.realpath(os.path.join(dest, rel)), droot):
            raise iv.ConfigError(f'{src}: use: {u or rel} would be copied to {os.path.realpath(os.path.join(dest, rel))},'
                                 f' outside {droot}')
    if not inside(os.path.realpath(os.path.join(dest, 'SHA256SUMS')), droot):
        raise iv.ConfigError(f'{dest}/SHA256SUMS resolves outside {droot}')
    os.makedirs(dest, exist_ok=True)
    sums = []
    for rel, (u, path) in files.items():
        if not os.path.isabs(rel):
            os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
            shutil.copyfile(path, os.path.join(dest, rel))
        sums.append(f"{hashlib.sha256(open(path, 'rb').read()).hexdigest()}  {rel}\n")
    with open(os.path.join(dest, 'SHA256SUMS'), 'w') as f:
        f.writelines(sums)
    copy = os.path.join(dest, os.path.basename(src))
    load_trigger_config(copy)             # it must load as it will in the router (plugins next to the copy)
    return copy


def main(argv=None):
    import yaml
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['stage']:
        print(stage_config(argv[1], argv[2]))
        return
    import relay_router as rr
    a = rr.parse_args(argv)
    msa = a.harness == 'msa'
    parser_loaded = bool(a.repair_on_parse_error or a.teacher_format_guard) and not msa
    note = a.verify_note_text or (rr.VERIFY_NOTE_MSA if msa else rr.VERIFY_NOTE if a.mode == 'relay'
                                  else rr.VERIFY_NOTE_OWN)
    cfg = legacy_config(a, parser_loaded, note)
    iv.load_config(cfg)                   # it must load as written
    sys.stdout.write(yaml.safe_dump(cfg, sort_keys=False, width=1000))


if __name__ == '__main__':
    main()
