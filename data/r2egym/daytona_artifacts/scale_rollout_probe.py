#!/usr/bin/env python3
"""One coordinator of the Daytona capacity run: N tasks, N concurrent seats, one attempt each.

Same agent/environment config and the same instrumentation (tokenize / terminal_exec / generation
timers) as paired_rollout_probe.py, but reads tasks straight from a task tree and writes a compact
result (stats + one row per trial with phase timestamps) instead of the multi-hundred-MB result.json.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from paired_rollout_probe import instrument


def compact_trials(job_dir):
    rows = []
    for path in sorted(job_dir.rglob('result.json')):
        if 'attempts' in path.parts or path.parent == job_dir:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception as exc:
            rows.append(dict(path=str(path), unreadable=type(exc).__name__))
            continue
        if 'trial_name' not in r:
            continue
        ex = r.get('exception_info') or {}
        ar = r.get('agent_result') or {}
        md = ar.get('metadata') or {}
        rows.append(dict(task=r.get('task_name'), trial=r.get('trial_name'),
                         reward=((r.get('verifier_result') or {}).get('rewards')),
                         exception=ex.get('exception_type'),
                         started_at=r.get('started_at'), finished_at=r.get('finished_at'),
                         environment_setup=r.get('environment_setup'), agent_setup=r.get('agent_setup'),
                         agent_execution=r.get('agent_execution'), verifier=r.get('verifier'),
                         n_episodes=ar.get('n_episodes'),
                         api_request_times_msec=md.get('api_request_times_msec') if isinstance(md, dict) else None))
    return rows


async def main(a):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    names = json.loads(Path(a.tasks).read_text())
    config = dict(job_name=a.name, jobs_dir=str(root), n_attempts=1, n_concurrent_trials=len(names),
                  quiet=True, retry=dict(max_retries=0),
                  environment=dict(type='daytona', delete=True, override_cpus=2, override_memory_mb=4096,
                                   kwargs=dict(auto_snapshot=True)),
                  verifier=dict(override_timeout_sec=a.verifier_seconds),
                  agents=[dict(name='terminus-2', model_name='openai/snowball', override_timeout_sec=a.agent_seconds,
                               kwargs=dict(api_base=a.api_base, temperature=1.0, collect_rollout_details=a.collect_rollout_details,
                                           enable_summarize=False, store_all_messages=False, record_terminal_session=False,
                                           enable_episode_logging=False, interleaved_thinking=False,
                                           model_info=dict(max_input_tokens=61440, max_output_tokens=4096, max_tokens=65536,
                                                           input_cost_per_token=0, output_cost_per_token=0, mode='chat'),
                                           extra_body=dict(skip_special_tokens=False, top_k=-1, top_p=1.0)))],
                  tasks=[dict(path=str(Path(a.task_root) / name)) for name in names])
    parsed = JobConfig.model_validate(config)
    (root / (a.name + '_config.json')).write_text(parsed.model_dump_json(indent=2))
    job = await Job.create(parsed)
    stream = instrument(root / (a.name + '_timing.jsonl'))
    t0 = time.time()
    try:
        result = await job.run()
    finally:
        stream.close()
    out = dict(name=a.name, seats=len(names), wall_seconds=round(time.time() - t0, 1),
               stats=json.loads(result.stats.model_dump_json()), trials=compact_trials(root / a.name))
    (root / (a.name + '_result.json')).write_text(json.dumps(out, indent=1))
    st = out['stats']
    print(f"done {a.name}: completed {st.get('n_completed_trials')} errored {st.get('n_errored_trials')} "
          f"wall {out['wall_seconds']} s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True, help='json list of task names')
    p.add_argument('--task-root', required=True, help='directory holding <task name>/ dirs')
    p.add_argument('--out', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--agent-seconds', type=int, default=600)
    p.add_argument('--verifier-seconds', type=int, default=300)
    p.add_argument('--collect-rollout-details', action='store_true',
                   help='RL-faithful: harbor tokenizes every turn on the coordinator (the /tokenize path); off = no tokenize calls at all')
    asyncio.run(main(p.parse_args()))
