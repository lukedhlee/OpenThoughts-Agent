#!/usr/bin/env python3
"""Markdown results table of the Horizon SFT arms whose eval record is complete (every group of eval_sft.sh done),
for the relay SFT gist. Numbers come from eval_readout.py's pass@1 (per-task mean over runs) and its paired
difference against the control tag (task bootstrap, 95 %).

    python gist_table.py --control hzA --arms "hzA:A, relay (control):1,616:246" "h3kimi:H3, A + Kimi SWE-smith:6,014:462" ...
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ST = f"/scratch/11584/{os.environ.get('USER')}/experiments/sft_eval/state"
SETS = (('tb21', 'TB2.1'), ('swe', 'SWE-bench Verified r100'), ('tblite', 'TB-lite'))


def complete(tag, day):
    gs = glob.glob(f'{ST}/g_{tag}_r*_{day}') or glob.glob(f'{ST}/*_{tag}_r*_{day}')   # grouped, or one dir per run
    return bool(gs) and all(os.path.exists(f'{g}/done') for g in gs), len(gs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--control', default='hzA')
    ap.add_argument('--arms', nargs='+', required=True, help='tag:label:rows:steps')
    ap.add_argument('--day', default='20260930')
    ap.add_argument('--save', help='also write {readout, arms} JSON here (for gist_fig.py)')
    a = ap.parse_args()
    arms = [x.split(':', 3) for x in a.arms]
    done = [x for x in arms if complete(x[0], a.day)[0]]
    tags = [x[0] for x in done]
    if a.control not in tags:
        sys.exit('control has no complete record')
    out = f'/tmp/gist_readout_{os.getpid()}.json'
    subprocess.run([sys.executable, f'{HERE}/eval_readout.py', '--tags', *tags, '--day', a.day, '--out', out],
                   check=True, capture_output=True)
    r = json.load(open(out))
    if a.save:
        json.dump(dict(readout=r, arms=done, control=a.control), open(a.save, 'w'), indent=1)
    hdr = '| arm | rows | steps | ' + ' | '.join(n for _, n in SETS) + ' |'
    print(hdr)
    print('|' + '---|' * (3 + len(SETS)))
    for tag, label, rows, steps in done:
        cells = []
        for s, _ in SETS:
            v = r.get(f'{tag}/{s}') or {}
            p = v.get('pass_at_1')
            cell = f"{100 * p:.1f} %" if p is not None else 'n/a'
            d = r.get(f'{a.control}-{tag}/{s}')
            if tag != a.control and d and d.get('diff') is not None:
                lo, hi = -d['ci95'][1], -d['ci95'][0]          # readout gives control - arm; show arm - control
                sig = '**' if lo > 0 or hi < 0 else ''
                cell += f" ({sig}{-100 * d['diff']:+.1f}{sig} [{100 * lo:+.1f}, {100 * hi:+.1f}])"
            cells.append(cell)
        print(f'| {label} | {rows} | {steps} | ' + ' | '.join(cells) + ' |')
    pending = [x[1] for x in arms if x[0] not in tags]
    if pending:
        print(f"\n<sub>Still evaluating: {', '.join(x.split(',')[0] for x in pending)}.</sub>")


if __name__ == '__main__':
    main()
