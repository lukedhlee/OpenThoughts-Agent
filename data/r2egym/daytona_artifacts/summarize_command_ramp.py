#!/usr/bin/env python3
"""Validate and tabulate one command-ramp output directory, without network calls."""
import argparse
import collections
import json
from pathlib import Path


def q(xs, p):
    xs=sorted(xs)
    return round(xs[int((len(xs)-1)*p)],4) if xs else None


def summarize(root):
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    stages=[r for r in events if r['kind']=='plateau_summary']
    result=[]
    for stage in stages:
        rows=[r for r in events if r['kind']=='command' and r['stage']==stage['stage']]
        assert len(rows)==stage['attempts'], 'command count differs from summary'
        assert len({r['seat'] for r in rows})==stage['seats'], 'not every advertised seat ran commands'
        bad=[r for r in rows if not r['ok']]
        assert len(bad)==stage['errors'], 'error count differs from summary'
        good=[r['seconds'] for r in rows if r['ok']]
        result.append(dict(stage=stage['stage'],seats=stage['seats'],coordinators=stage['coordinators'],
            commands=len(rows),errors=len(bad),error_rate=len(bad)/len(rows) if rows else None,
            success_p50=q(good,.5),success_p90=q(good,.9),success_p99=q(good,.99),
            failed_p50=q([r['seconds'] for r in bad],.5),
            commands_per_second=stage['completed_per_second'],peak_inflight=stage['peak_inflight'],
            api_calls=sum(r['api_calls'] for r in rows),
            driver_cpu_seconds=stage['driver_cpu_seconds'],loop_lag_max=stage['loop_lag_max'],
            error_types=dict(collections.Counter(r.get('error_type','nonzero_exit') for r in bad)),
            command_kinds=stage['per_kind']))
    cleanup=[r for r in events if r['kind']=='cleanup_census']
    final=cleanup[-1] if cleanup else None
    clean=bool(final and final['remaining']==0 and not final['states'].get('destroying'))
    report=dict(stages=result,cleanup_verified=clean,cleanup_final=final,
                meta=next((r for r in events if r['kind']=='meta'),None),
                censuses=[r for r in events if r['kind']=='org_census'])
    (root/'validated_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    report=summarize(a.root)
    print('| stage | seats | commands | p50 s | p90 s | p99 s | errors | success/s |')
    print('|---|---|---|---|---|---|---|---|')
    for r in report['stages']:
        print(f"| {r['stage']} | {r['seats']} | {r['commands']} | {r['success_p50']} | {r['success_p90']} | {r['success_p99']} | {r['errors']} ({r['error_rate']:.2%}) | {r['commands_per_second']:.2f} |")
    print('Cleanup verified:',report['cleanup_verified'])
