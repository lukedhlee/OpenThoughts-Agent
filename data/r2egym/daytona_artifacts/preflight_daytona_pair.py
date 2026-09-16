#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path
from daytona import AsyncDaytona, DaytonaConfig
from harbor.agents.nop import NopAgent
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.utils.container_cache import environment_dir_hash_truncated
from paired_rollout_probe import instrument
from ramp_stress import load_key, KEY_FILE

async def check(self, instruction, environment, context):
    result = await environment.exec("set -e; git -C /testbed rev-parse HEAD; f=$(mktemp /testbed/.transport-probe.XXXXXX); rm -- \"$f\"; python --version", timeout_sec=30)
    print(json.dumps(dict(kind='repository_preflight',return_code=result.return_code,stdout=result.stdout,stderr=result.stderr)),flush=True)
    if result.return_code: raise RuntimeError('Repository preflight failed')

async def main(a):
    d=json.loads(Path(a.config).read_text())
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(KEY_FILE))) as client:
        for task in d['tasks']:
            name='harbor__'+environment_dir_hash_truncated(Path(task['path'])/'environment',truncate=12)+'__snapshot'
            snapshot=await client.snapshot.get(name)
            print(json.dumps(dict(kind='snapshot_verified',name=name,state=str(snapshot.state))),flush=True)
    d.update(job_name='daytona_repository_preflight',jobs_dir=a.out,n_attempts=1,n_concurrent_trials=8)
    d['environment']['kwargs']={'auto_snapshot':True}
    d['agents']=[dict(name='nop')];d['verifier']={'disable':True}
    NopAgent.run=check
    stream=instrument(Path(a.out)/'preflight_timing.jsonl')
    try:
        job=await Job.create(JobConfig.model_validate(d));result=await job.run()
        Path(a.out,'preflight_result.json').write_text(result.model_dump_json(indent=2))
        if result.stats.n_errored_trials:raise RuntimeError('Environment preflight errors')
    finally:stream.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True);asyncio.run(main(a))
