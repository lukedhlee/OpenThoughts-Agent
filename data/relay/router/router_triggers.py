#!/usr/bin/env python3
"""Router-side triggers and the router's trigger config (stage 2 of notes/ot-agent/relay-triggers).

The relay router asks one interventions.Controller per episode who answers each request. Without --trigger-config it
builds the Controller's config from its own flags (legacy_config): relay-router/1's hard-coded decisions re-expressed,
in today's order, so routing and every turns.jsonl record stay as they were. With --trigger-config <yaml> that file
replaces those flags (the router refuses both, LEGACY_FLAGS). --mode teacher / student never hand an episode over, so
there every trigger only logs (shadow()).

Triggers that need the router's view of an episode live here; load_config() resolves them by name like interventions'
own built-ins (interventions still holds stage 0's placeholder for assist, so they join its registry only while a config
loads; register() installs them for good, for a tool outside the router):

  assist   msa --assist (was Router.assist_trigger): a one-turn teacher intervention before the student answers, on a
           command timeout, a failing-command streak, a repeated output, or a context milestone (each milestone once,
           used up only when the assist acts).
  done_claim_first
           done_claim as would_fire logs it (--mode teacher / student): the episode's first claim only.

    python router_triggers.py <relay_router.py flags>    # the config those flags stand for, as YAML
"""
import contextlib
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


class FirstClaim:
    """done_claim as the router's would_fire reports it (--mode teacher / student): relay_triggers' scanner fires it on
    the episode's first claim only, while interventions' DoneClaim (relay semantics) judges every claim. The reply as
    the router judged it: its parsed action, else its text; none when the harness rejected it (msa: would_fire is not
    computed then). Name done_claim, so records and --takeover read as today."""
    name = 'done_claim'
    phases = frozenset({'reply'})

    async def observe(self, ev):
        r = ev.reply
        if ev.scanner is None or r is None:
            return None
        view = r.action if r.action is not None else (None if r.parse_error else r.text)
        if view is None:
            return None
        cfg = dict(rt.CALIBRATED_CONFIG, enabled=['done_claim'])
        f = next((f for f in ev.scanner.current_fires(view, cfg) if f['trigger'] == 'done_claim'), None)
        return iv.Decision(self.name, f.get('reason') or self.name) if f else None


ROUTER_BUILTINS = {'assist': Assist, 'done_claim_first': FirstClaim}


def register():
    """Add the router's triggers to interventions' built-ins for good (replacing stage 0's placeholders)."""
    iv.BUILTINS.update(ROUTER_BUILTINS)


@contextlib.contextmanager
def _router_builtins():
    saved = {name: iv.BUILTINS.get(name) for name in ROUTER_BUILTINS}
    register()
    try:
        yield
    finally:
        for name, cls in saved.items():
            if cls is None:
                iv.BUILTINS.pop(name, None)
            else:
                iv.BUILTINS[name] = cls


def load_config(src):
    """interventions.load_config with the router's triggers among the built-ins."""
    with _router_builtins():
        return iv.load_config(src)


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
        return dict(triggers=[dict(use=n) for n in env + dec] + ([dict(use='done_claim_first')] if done else []))
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


def main(argv=None):
    import yaml
    import relay_router as rr
    a = rr.parse_args(argv)
    msa = a.harness == 'msa'
    parser_loaded = bool(a.repair_on_parse_error or a.teacher_format_guard) and not msa
    note = a.verify_note_text or (rr.VERIFY_NOTE_MSA if msa else rr.VERIFY_NOTE if a.mode == 'relay'
                                  else rr.VERIFY_NOTE_OWN)
    cfg = legacy_config(a, parser_loaded, note)
    load_config(cfg)                      # it must load as written
    sys.stdout.write(yaml.safe_dump(cfg, sort_keys=False, width=1000))


if __name__ == '__main__':
    main()
