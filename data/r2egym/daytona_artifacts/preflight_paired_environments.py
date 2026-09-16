#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path

from harbor.agents.nop import NopAgent
from harbor.job import Job
from harbor.models.job.config import JobConfig


async def check(self, instruction, environment, context):
    result=await environment.exec("set -e; git -C /testbed rev-parse HEAD; f=$(mktemp /testbed/.transport-probe.XXXXXX); rm -- \"$f\"; python --version", timeout_sec=30)
    print(json.dumps(dict(kind='repository_preflight',return_code=result.return_code,stdout=result.stdout,stderr=result.stderr)),flush=True)
    if result.return_code:raise RuntimeError('Repository preflight failed')


async def main():
    NopAgent.run=check
    d=json.loads(Path('preflight/apptainer_create_config.json').read_text())
    d.update(job_name='apptainer_repository_preflight',jobs_dir=str(Path('environment_preflight').resolve()),n_attempts=1,n_concurrent_trials=8)
    d['agents']=[dict(name='nop')];d['verifier']={'disable':True}
    job=await Job.create(JobConfig.model_validate(d));result=await job.run()
    print(result.stats.model_dump_json(),flush=True)
    if result.stats.n_errored_trials:raise RuntimeError('Environment preflight errors')


if __name__=='__main__':asyncio.run(main())
