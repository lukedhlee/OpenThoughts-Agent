

# ── Required tests (appended by tt_daytona_tree_v3.py) ──────────────────────────────────────────────────────────────
# The check above drops the test file from every id and grades only the expected tests that happen to appear, so a repo
# test with the same class and method name can stand in for a staged one, and a staged test that never runs (its module
# fails to import) is not counted against the agent. /tests/required_tests.json lists the staged tests the oracle run
# reported, by full id under r2e_tests/, with their expected status: every one must appear with that status.
REQUIRED_TESTS_PATH = "/tests/required_tests.json"


def _staged_id(key: str):
    k = _decolor(key).split(" - ")[0].strip()
    i = k.rfind("r2e_tests/")
    return k[i:] if i >= 0 and "::" in k else None


def _parse_log_staged(log):
    """Full test id (from r2e_tests/ on) -> status, from the `short test summary info` block; other paths are ignored."""
    if not log or "short test summary info" not in log:
        return {}
    status_map = {}
    for line in log.split("short test summary info", 1)[1].splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        for status in _STATUSES:
            if status in line:
                tid = _staged_id(line.split(status, 1)[1] if line.startswith(status) else line)
                if tid:
                    status_map[tid] = status
                break
    return status_map


def test_r2egym_required_tests_ran():
    req_path = Path(REQUIRED_TESTS_PATH)
    if not req_path.exists():
        return
    required = json.loads(req_path.read_text())
    assert required, "required_tests.json is empty -- task metadata malformed"
    parsed = _parse_log_staged(Path(TEST_OUTPUT_PATH).read_text(errors="replace"))
    missing = [k for k in required if k not in parsed]
    wrong = [(k, required[k], parsed[k]) for k in required if k in parsed and parsed[k] != required[k]]
    assert not missing, f"{len(missing)}/{len(required)} required staged tests did not run. First few: {missing[:5]}"
    assert not wrong, f"{len(wrong)}/{len(required)} required staged tests with wrong status. First few: {wrong[:5]}"
