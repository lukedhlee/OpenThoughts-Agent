# Relay trigger examples

- **Write** a trigger as a function `f(ev, state)` that returns a `Decision` or `None` (`two_errors.py`), or as a class with `name`, `phases` and `async observe(ev)`. The types are in `data/relay/router/interventions.py`.
- **Configure** it in YAML: a list of `use:` entries (a built-in, `module:func`, or `path/file.py:func`) with `where`, `span`, `release` and `policy` (`stay_until_fixed.yaml`). Built-ins combine with `any_of`, `all_of` and `streak`.
- **Replay** a config over recorded episodes offline with `data/relay/triggers/replay.py` (stage 1) before spending GPUs on it.
- **Enable** it online with `TRIGGER_CONFIG=<yaml>` on the relay launcher (see `../README.md` in this folder's parent, or `README.md` here), shadow-testing first with `log_only: true` or the `student_only` arm.
