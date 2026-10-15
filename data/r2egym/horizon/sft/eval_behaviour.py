#!/usr/bin/env python3
"""Behaviour of eval_sft.sh runs per model tag, next to eval_readout.py's pass rates: how the model works, not whether
it passes. Per tag and set (TB2.1, SWE random-100, TB-lite) and pooled over the three, over every trial of every rep:

  reasoning tokens per turn  the <|start_think|>..<|end_think|> span of each agent turn (to the end of the message when
                             the close is missing), tokenized with the 09-21 tokenizer; mean and median over turns
  compaction rate            share of trials where Terminus-2 summarized its context at least once
                             (agent_result.metadata.summarization_count > 0), and summarizations per trial
  context-overflow rate      share of trials ending in ContextLengthExceededError
  timeout rate               share of trials ending in AgentTimeoutError
  turns per episode          agent_result.metadata.n_episodes; mean and median

Runs are eval_sft.sh's <set>_<tag>_r<rep>_<day> job dirs; the trajectory is the last attempt's agent/trajectory.json.

    python eval_behaviour.py --tags mqwenku@20261015 mqwenk@20261015 [--jobs <tb2_jobs>] [--tokenizer-dir <09-21>] [--out b.json]
"""
import argparse
import glob
import json
import os
import statistics

SETS = {'tb21': ('tb21',), 'swe': ('swe_s0', 'swe_s1'), 'tblite': ('tblite_s0', 'tblite_s1')}
ST, ET = '<|start_think|>', '<|end_think|>'


def reasoning(msg):
    if not isinstance(msg, str) or ST not in msg:
        return None
    body = msg.split(ST, 1)[1]
    return body.split(ET, 1)[0]


def trial(d, tok):
    r = json.load(open(os.path.join(d, 'result.json')))
    md = (r.get('agent_result') or {}).get('metadata') or {}
    exc = (r.get('exception_info') or {}).get('exception_type')
    att = sorted(glob.glob(os.path.join(d, 'attempts', '*', 'agent', 'trajectory.json')))
    traj = att[-1] if att else os.path.join(d, 'agent', 'trajectory.json')
    rtok = []
    try:
        for s in json.load(open(traj))['steps']:
            if s.get('source') != 'agent':
                continue
            text = reasoning(s.get('message'))
            rtok.append(len(tok.encode(text, add_special_tokens=False).ids) if text else 0)
    except (OSError, ValueError, KeyError):
        rtok = None
    return dict(exc=exc, summ=md.get('summarization_count') or 0, turns=md.get('n_episodes'), rtok=rtok)


def summarize(ts):
    n = len(ts)
    turns = [t['turns'] for t in ts if t['turns'] is not None]
    rt = [x for t in ts if t['rtok'] for x in t['rtok']]
    rate = lambda k: round(sum(k(t) for t in ts) / n, 4) if n else None  # noqa: E731
    return dict(trials=n,
                reasoning_tokens_per_turn_mean=round(statistics.mean(rt), 1) if rt else None,
                reasoning_tokens_per_turn_median=statistics.median(rt) if rt else None,
                turns_scored=len(rt),
                compaction_rate=rate(lambda t: t['summ'] > 0),
                summarizations_per_trial=rate(lambda t: t['summ']),
                context_overflow_rate=rate(lambda t: t['exc'] == 'ContextLengthExceededError'),
                timeout_rate=rate(lambda t: t['exc'] == 'AgentTimeoutError'),
                turns_per_episode_mean=round(statistics.mean(turns), 1) if turns else None,
                turns_per_episode_median=statistics.median(turns) if turns else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', nargs='+', required=True, help='tag@day')
    ap.add_argument('--jobs', default=f"/scratch/11584/{os.environ.get('USER')}/experiments/sft_eval/tb2_jobs")
    ap.add_argument('--tokenizer-dir', default='/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921')
    ap.add_argument('--out')
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    res = {}
    for td in a.tags:
        tag, day = td.split('@')
        per, allt = {}, []
        for s, runs in SETS.items():
            ts = []
            for run in runs:
                for jd in sorted(glob.glob(os.path.join(a.jobs, f'{run}_{tag}_r*_{day}'))):
                    for d in sorted(glob.glob(os.path.join(jd, '*', 'result.json'))):
                        ts.append(trial(os.path.dirname(d), tok))
            per[s] = summarize(ts)
            allt += ts
        per['all'] = summarize(allt)
        res[tag] = per
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
