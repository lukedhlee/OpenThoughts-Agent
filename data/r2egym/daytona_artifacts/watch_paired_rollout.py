#!/usr/bin/env python3
"""Release only this experiment's CPU allocation when its GPU allocation ends."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--gpu-job',required=True);p.add_argument('--cpu-job',required=True);p.add_argument('--run',required=True);a=p.parse_args()
root=Path(a.run)
infra={'ConnectTimeout','ReadTimeout','ReadError','PoolTimeout','APIConnectionError','EnvironmentStartTimeoutError','DaytonaError','DaytonaTimeoutError','NetworkError','ConnectionError'}
tripped=False
cpu_released=False
def release_cpu():
    subprocess.run(['ssh','-S',str(Path.home()/'.ssh/cm_juwels/bridge'),'juwels01.fz-juelich.de','scancel',a.cpu_job],check=True)
while True:
    result=subprocess.run(['squeue','-h','-j',a.gpu_job,'-o','%T'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Could not read GPU scheduler state')
    state=result.stdout.strip()
    if not state:break
    if not cpu_released and (root/'apptainer_16_result.json').exists():
        release_cpu();cpu_released=True
        print('Apptainer cohort complete; own CPU allocation released.',flush=True)
    errors=[];count=0
    if root.exists():
        for path in root.glob('**/result.json'):
            if 'attempts' in path.relative_to(root).parts:continue
            try:data=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):continue
            if 'exception_info' not in data:continue
            count+=1;info=data.get('exception_info') or {};kind=info.get('exception_type')
            if kind in infra:errors.append(kind)
    print(json.dumps(dict(epoch=time.time(),state=state,finished_records=count,infra_errors=errors)),flush=True)
    if len(errors)>=4 and len(errors)/max(1,count)>=.25 and not tripped:
        print('Infrastructure error tripwire: stopping own GPU job',flush=True)
        subprocess.run(['scancel','--signal=INT','--batch',a.gpu_job],check=True)
        time.sleep(30)
        subprocess.run(['scancel',a.gpu_job],check=True)
        tripped=True
    time.sleep(20)
if not cpu_released:release_cpu()
print('GPU allocation ended; own CPU allocation released. Verify sandbox and bridge cleanup.',flush=True)
