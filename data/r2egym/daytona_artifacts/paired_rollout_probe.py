#!/usr/bin/env python3
"""Run an instrumented, bounded real-agent cohort without modifying shared Harbor."""
import argparse
import asyncio
import functools
import json
from pathlib import Path
import time


def instrument(output):
    import harbor.llms.lite_llm as llm
    from harbor.environments.daytona.environment import DaytonaEnvironment
    from harbor.environments.apptainer.apptainer import ApptainerEnvironment
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    stream=output.open('a',buffering=1)
    original_init=DaytonaEnvironment.__init__
    @functools.wraps(original_init)
    def tracked_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        stream.write(json.dumps(dict(kind='environment_created',label=self._instance_label))+'\n')
    DaytonaEnvironment.__init__=tracked_init
    def timed(function,kind):
        @functools.wraps(function)
        async def wrapped(*args,**kwargs):
            started=time.monotonic();error=None
            try:return await function(*args,**kwargs)
            except BaseException as exc:
                error=type(exc).__name__;raise
            finally:stream.write(json.dumps(dict(kind=kind,seconds=time.monotonic()-started,error=error,epoch=time.time()))+'\n')
        return wrapped
    llm.tokenize_chat=timed(llm.tokenize_chat,'tokenize')
    DaytonaEnvironment.exec=timed(DaytonaEnvironment.exec,'terminal_exec')
    ApptainerEnvironment.exec=timed(ApptainerEnvironment.exec,'terminal_exec')
    Terminus2._query_llm=timed(Terminus2._query_llm,'generation')
    return stream


async def main(args):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    names=json.loads(Path(args.tasks).read_text())
    assert args.seats%len(names)==0
    taskroot=Path(args.tasks).parent/'tasks'/args.backend
    config=dict(job_name=args.name,jobs_dir=str(root),n_attempts=args.seats//len(names),n_concurrent_trials=args.seats,
        quiet=True,retry=dict(max_retries=0),
        environment=dict(type=args.backend,delete=True,override_cpus=2,override_memory_mb=4096,
            kwargs=dict(bridge_url=args.bridge) if args.backend=='apptainer' else dict(auto_snapshot=True)),
        verifier=dict(override_timeout_sec=300),
        agents=[dict(name='terminus-2',model_name='openai/snowball',override_timeout_sec=600,
            kwargs=dict(api_base='http://127.0.0.1:8000/v1',temperature=1.0,collect_rollout_details=True,
                enable_summarize=False,store_all_messages=True,record_terminal_session=False,
                enable_episode_logging=False,interleaved_thinking=False,
                model_info=dict(max_input_tokens=61440,max_output_tokens=4096,max_tokens=65536,
                                input_cost_per_token=0,output_cost_per_token=0,mode='chat'),
                extra_body=dict(skip_special_tokens=False,top_k=-1,top_p=1.0)))],
        tasks=[dict(path=str(taskroot/name)) for name in names])
    parsed=JobConfig.model_validate(config)
    (root/(args.name+'_config.json')).write_text(parsed.model_dump_json(indent=2))
    job=await Job.create(parsed)
    if args.validate_only:
        stream=instrument(root/(args.name+'_instrumentation_check.jsonl'));stream.close()
        print('Job construction and instrumentation valid:',args.name,flush=True);return
    stream=instrument(root/(args.name+'_timing.jsonl'))
    try:
        result=await job.run()
        (root/(args.name+'_result.json')).write_text(result.model_dump_json(indent=2))
    finally:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['daytona','apptainer'],required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--name',required=True)
    p.add_argument('--seats',type=int,default=16);p.add_argument('--bridge',default='http://10.128.1.2:9936')
    p.add_argument('--validate-only',action='store_true');asyncio.run(main(p.parse_args()))
