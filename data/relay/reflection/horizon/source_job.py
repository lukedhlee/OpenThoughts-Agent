"""Harbor config for fixed, unassisted base-student source attempts (student->teacher's source corpus).

    python -m data.relay.reflection.horizon.source_job --config <student_teacher.json> --count 1280 --shard 0/2 --out <yaml>

Tasks are collect.make_job's: the seed-shuffled training list from offset 0, so 1,280 = 20 x 64 covers exactly the
tasks ROFT's 20 updates collect on. The agent block is collect.make_job's unchanged (MSA2 tool mode, 16k reply cap,
65k context, no relay router, the student endpoint); only harbor's infrastructure retries are added (collect.infra_retry: the
benchmark policy's list plus sandbox-start and gateway errors), so a Daytona or server hiccup is retried instead of leaving a hole. Import the finished job
with `cli import-harbor --author student` at the base revision.
"""

import argparse
from pathlib import Path

import yaml

from data.relay.reflection.collect import infra_retry, make_job
from data.relay.reflection.core import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    config = load_config(a.config)
    job = make_job({"config": config, "weights_version": 0, "count": a.count, "output_dir": a.jobs_dir})
    i, n = map(int, a.shard.split("/"))
    job["tasks"] = job["tasks"][i::n]
    job["job_name"] = f"{config['name']}-sources-base-s{i}of{n}"
    job["jobs_dir"] = a.jobs_dir
    job["n_concurrent_trials"] = a.concurrency
    job["retry"] = infra_retry({**config, "runtime": {**config["runtime"], "collection": {**config["runtime"]["collection"], "infra_retries": True}}})
    Path(a.out).write_text(yaml.safe_dump(job, sort_keys=False))
    print(f"{job['job_name']}: {len(job['tasks'])} tasks")


if __name__ == "__main__":
    main()
