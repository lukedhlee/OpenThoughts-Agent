"""A function trigger: the student's last two actions both failed, after turn 3. The YAML-only twin:
    {use: all_of, params: {of: [{use: action_error, params: {streak: 2}}, {use: after_turn, params: {turn: 3}}]}}
Enable: `- use: data/relay/triggers/examples/two_errors.py:two_errors` (relative paths resolve from the config
file)."""
from interventions import ActionError, trigger


@trigger(phases='request')
async def two_errors(ev, state):
    if 'errors' not in state:                    # per-episode state: the streak survives between turns
        state['errors'] = ActionError(streak=2)
    d = await state['errors'].observe(ev)        # asked on every turn, so the streak stays right
    if d is not None and ev.turn > 3:
        return d
