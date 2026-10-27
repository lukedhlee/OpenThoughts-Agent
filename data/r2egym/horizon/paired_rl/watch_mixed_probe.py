#!/usr/bin/env python3
"""Bounded one-shot observation and readout of this pair. Never submits or cancels jobs."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path("/scratch/11584/lukedhlee/experiments/paired-sft-rl-20261026/mixed")
ARMS = ["probe_mix_base_1026", "probe_mix_v7_1026"]
OTA = Path.home()/"snowball/ota-rl"


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def readout():
    spec = importlib.util.spec_from_file_location("screen", OTA/"data/r2egym/horizon/rl/screen_report.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    expected = set((ROOT/"train.txt").read_text().split())
    mapping = {r["task"]:r for r in csv.DictReader((ROOT/"task_map.tsv").open(), delimiter="\t")}
    out = ROOT/"readout"; out.mkdir(exist_ok=True)
    arms = {}; per_arm = {}
    for name in ARMS:
        run = ROOT/name
        masks = mod.masks_of(run)
        candidates = collections.defaultdict(list)
        # Completion time separates the completed first eval from any second eval
        # started by the old trainer's train-end callback. Keep first 8 raw outcomes,
        # including invalids, rather than silently replacing invalid attempts.
        for p in (run/"trials/eval_sessions").glob("*/*/result.json"):
            task = re.sub(r"__r\d+$", "", p.parent.name.rsplit("__", 1)[0]).split("/")[-1]
            if task in expected:
                candidates[task].append(p)
        rows = {}
        for task in sorted(expected):
            paths = sorted(candidates.get(task, []), key=lambda p:p.stat().st_mtime)
            scores = []; exceptions = collections.Counter()
            for p in paths[:8]:
                try:
                    label, reward, ex = mod.read_outcome(p)
                except (OSError, ValueError, KeyError) as exc:
                    exceptions[type(exc).__name__] += 1
                    continue
                if reward in (0,1) and ex not in masks:
                    scores.append(int(reward))
                else:
                    exceptions[ex or "unscored"] += 1
            rows[task] = dict(task=task, source=mapping[task]["source"], raw=min(8,len(paths)),
                scored=len(scores), solved=sum(scores), invalid=sum(exceptions.values()), extra=max(0,len(paths)-8),
                exceptions=dict(exceptions))
        with (out/(name+".tsv")).open("w") as f:
            w=csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()), delimiter="\t")
            w.writeheader();w.writerows(rows.values())
        per_arm[name]=rows
        summary={}
        for source in ["r2egym","tmax","all"]:
            rs=[r for r in rows.values() if source=="all" or r["source"]==source]
            complete=[r for r in rs if r["scored"]==8 and r["raw"]==8]
            summary[source]=dict(tasks=len(rs), fully_scored_tasks=len(complete),
                scored_attempts=sum(r["scored"] for r in rs), invalid=sum(r["invalid"] for r in rs),
                missing_raw_attempts=sum(8-r["raw"] for r in rs),
                pass1_complete=sum(r["solved"] for r in complete)/(8*len(complete)) if complete else None,
                mixed_complete=sum(0<r["solved"]<8 for r in complete))
        arms[name]=summary
    base,sft=[per_arm[n] for n in ARMS]
    complete=sorted(t for t in expected if base[t]["scored"]==8 and sft[t]["scored"]==8)
    band=[t for t in complete if 0<base[t]["solved"]<8 or 0<sft[t]["solved"]<8]
    (out/"common_complete.txt").write_text("\n".join(complete)+"\n")
    (out/"common_mixed_band.txt").write_text("\n".join(band)+"\n")
    summary=dict(arms=arms, common_complete_tasks=len(complete), common_mixed_band_tasks=len(band),
        band_by_source=dict(collections.Counter(mapping[t]["source"] for t in band)),
        status="complete" if len(complete)==len(expected) else "incomplete, inspect missing and invalid outcomes before training",
        interpretation="Pre-RL diagnostic only. No GRPO training launched by this watcher.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    log(json.dumps(summary))
    (ROOT/"WATCH_STATUS").write_text("READOUT_WRITTEN\n")


def main():
    deadline=time.monotonic()+36*3600
    previous=None
    while time.monotonic()<deadline:
        state=(ROOT/"STATUS").read_text().strip() if (ROOT/"STATUS").exists() else "SETUP_PENDING"
        if state.startswith("PROBE_SETUP_FAILED"):
            log(state);(ROOT/"WATCH_STATUS").write_text(state+"\n");return
        jobs={n:(ROOT/n/"jobs.txt").read_text().splitlines()[-1] for n in ARMS if (ROOT/n/"jobs.txt").exists()}
        queue={}
        if jobs:
            raw=subprocess.check_output(["squeue","-h","-j",",".join(jobs.values()),"-o","%i|%T|%M|%R"],text=True)
            queue={line.split("|",1)[0]:line for line in raw.splitlines()}
        health={}
        for name,job in jobs.items():
            p=ROOT/name/"logs"/(name+"_"+job+".out")
            if p.exists():
                with p.open("rb") as f:
                    f.seek(max(0,p.stat().st_size-131072));tail=f.read().decode(errors="replace")
                health[name]=dict(log_bytes=p.stat().st_size, age_seconds=round(time.time()-p.stat().st_mtime),
                    eval_finished="WANDB_MIRROR kind=eval step=0 " in tail,
                    tail_error_lines=[s[:300] for s in tail.splitlines() if any(x in s for x in ["Traceback", "CUDA out of memory", "FATAL:","Connection refused"])][-5:])
        snapshot=dict(setup=state,jobs=jobs,queue=queue,health=health)
        (ROOT/"watch.json").write_text(json.dumps(snapshot,indent=2)+"\n")
        key=json.dumps([state,queue])
        if key!=previous:log(key);previous=key
        if len(jobs)==2 and not queue:
            for name,job in jobs.items():
                try:
                    receipt=subprocess.check_output(["sacct","-j",job,"--format=JobID,State,Elapsed,ExitCode","-P"],text=True)
                    (ROOT/"logs"/(name+"-accounting.txt")).write_text(receipt)
                except subprocess.CalledProcessError:pass
            readout();return
        time.sleep(60)
    (ROOT/"WATCH_STATUS").write_text("WATCH_DEADLINE_REACHED, inspect queued or running jobs\n")


if __name__=="__main__":
    main()
