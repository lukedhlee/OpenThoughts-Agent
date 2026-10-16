# Relay trigger examples

The full guide is `../README.md`. These files are small starting points:

- `two_errors.py`: a function trigger (the student's last two commands failed, after turn 3).
- `stay_until_fixed.yaml`: the teacher takes over after a failed command and hands back once one of its own commands succeeds.
- `judge.yaml`: an LLM judge in shadow mode, to pick a threshold before letting it act.

Replay any of them on recorded episodes first (`../replay.py`), then shadow-test online, then let them act.
