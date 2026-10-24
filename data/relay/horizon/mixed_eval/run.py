"""Wait for both mixed-model exports, then run the reviewed corrected-stack eval plan."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

from data.relay.horizon.mixed_eval.plan import HARNESS, build, inventory
from data.relay.horizon.mixed_eval.readout import (
    summarize,
    markdown,
    validated_historical,
)


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=60, **kwargs).strip()


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def queue():
    rows = command(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%T|%D|%L"])
    return {
        line.split("|")[0]: line.split("|")[1:] for line in rows.splitlines() if line
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--harbor", required=True, type=Path)
    parser.add_argument("--marin", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, training, harbor = args.root, args.training_root, args.harbor
    ota = Path(__file__).resolve().parents[4]
    assert (
        command(["git", "-C", str(harbor), "rev-parse", "--short=8", "HEAD"]) == HARNESS
    )
    assert not command(["git", "-C", str(harbor), "status", "--porcelain"])
    assert (
        command(
            ["git", "-C", str(ota), "merge-base", "--is-ancestor", "1c40b0bf", "HEAD"]
        )
        == ""
    )
    validated_historical(root.parent / "cf2-repaired-readout-v2-20261023/report.json")
    data = inventory()
    groups, report = build(data, root, harbor)
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        print("CF_MIXED_EVAL_DRY_RUN_OK: no jobs or output files created", flush=True)
        return
    assert not command(["git", "-C", str(ota), "status", "--porcelain"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The operator writes this after checking the adapter unit/live gates and the complete config plan.
    receipt = json.loads((root / "launch_gate.json").read_text())
    code = command(["git", "-C", str(ota), "rev-parse", "HEAD"])
    assert receipt["passed"] and receipt["ota_commit"] == code
    assert (
        receipt["harbor_commit"] == HARNESS and receipt["snapshot_operations"] == "none"
    )
    for field in [
        "adapter_unit_gate",
        "live_direct_snapshot_gate",
        "live_python_dind_gate",
        "live_java_dind_gate",
    ]:
        assert receipt[field] is True, field
    save(root / "input_inventory.json", data)
    save(root / "plan.json", dict(groups=groups, report=report, ota_commit=code))
    while True:
        ready = all(
            (training / f"{arm}.export.ok").is_file()
            and (training / f"{arm}.export.path").is_file()
            for arm in ["cfmix50", "cfconcat"]
        )
        if ready:
            break
        status = (
            (training / "STATUS").read_text() if (training / "STATUS").exists() else ""
        )
        if "FAILED" in status:
            raise RuntimeError(f"Training controller failed: {status}")
        (root / "STATUS").write_text("WAITING_FOR_BOTH_EXPORTS\n")
        time.sleep(30)
    models = {
        arm: Path((training / f"{arm}.export.path").read_text().strip())
        for arm in ["cfmix50", "cfconcat"]
    }
    for arm, path in models.items():
        assert path.resolve().is_relative_to((training / "training" / arm).resolve()), (
            path
        )
        assert (path / "config.json").is_file() and list(path.glob("*.safetensors")), (
            path
        )
    assert inventory() == data, (
        "Frozen input/task/config drift while waiting for exports"
    )
    for name in ["bin", "serve/logs", "serve/endpoints", "runs", "drivers", "jobs"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    wrapper = args.marin / "experiments/june_tpu_67b_a2b/moe/replay_sbatch.py"
    target = root / "bin/sbatch"
    if not target.exists():
        target.symlink_to(wrapper)
    assert target.resolve() == wrapper.resolve()
    env = dict(
        os.environ,
        FP32_REPLAY_ROOT=str(root),
        PATH=str(root / "bin") + ":" + os.environ["PATH"],
        RELAY_EXP_DIR=str(root / "serve"),
        RELAY_PILOT_DIR=str(ota / "data/relay/pilot"),
        OTA=str(ota),
        HARBOR_SRC=str(harbor / "src"),
        HARBOR_SHA=HARNESS,
        MSA_DIR=str(ota),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CC="gcc",
        CXX="g++",
    )
    env.pop("LD_PRELOAD", None)
    if (root / "jobs.json").exists():
        raise RuntimeError(
            "Existing eval submissions require explicit adoption review; refusing duplicate launches"
        )
    jobs, completed, tunnels = {}, {}, []
    driver_states, health_receipts, driver_processes = {}, {}, {}
    health_stop = threading.Event()
    state_lock, admission, driver_start = (
        threading.Lock(),
        threading.Lock(),
        threading.Lock(),
    )
    next_start = [0.0]

    def started():
        result = command(
            ["python3", str(ota / "data/r2egym/horizon/sft/daytona_started.py")],
            env=env,
        )
        assert result.isdigit(), "Cannot verify shared-org concurrency"
        return int(result)

    def check_driver_health(run, state, live_jobs):
        trial_root = root / "jobs" / run
        directories = [path for path in trial_root.glob("*") if path.is_dir()]
        results = sum((path / "result.json").is_file() for path in directories)
        trajectories = sum(
            (path / "agent/trajectory.json").is_file()
            and (path / "agent/trajectory.json").stat().st_size > 0
            for path in directories
        )
        agent_bytes = sum(
            path.stat().st_size
            for trial in directories
            for path in (trial / "agent").glob("*")
            if path.is_file()
        )
        finished = (root / "drivers" / run / "settled.json").is_file()
        output = root / "drivers" / run / "driver.out"
        startup_reply = output.is_file() and "server ok:" in output.read_text(
            errors="replace"
        )
        endpoint = None
        if not finished:
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    state["url"] + "/models", timeout=5
                ) as response:
                    endpoint = response.status == 200
            except Exception:
                endpoint = False
        proven_progress = results > 0 or trajectories > 0
        driver_state = (
            "finished"
            if finished
            else "active"
            if state["job"] in live_jobs
            else "allocation_missing"
        )
        with state_lock:
            process = driver_processes.get(run)
        srun_alive = process is not None and process.poll() is None
        flags = []
        if not finished and not srun_alive:
            flags.append("srun_not_alive_before_settled_receipt")
        if not finished and not endpoint:
            flags.append("model_endpoint_unreachable")
        if driver_state == "allocation_missing":
            flags.append("allocation_missing_before_driver_settled")
        if not finished and not proven_progress:
            flags.append(
                "progress_ambiguous: no completed result or saved trajectory; native build600+setup360 may exceed15min"
            )
        if finished and results != state["expected"]:
            flags.append("incomplete_result_count")
        return dict(
            run=run,
            checked_utc=time.time(),
            age_seconds=time.time() - state["started"],
            finished_before_15_minutes=finished
            and time.time() - state["started"] < 900,
            state=driver_state,
            expected=state["expected"],
            task_directories=len(directories),
            result_files=results,
            saved_trajectories=trajectories,
            agent_log_bytes=agent_bytes,
            startup_model_reply=startup_reply,
            current_model_endpoint=endpoint,
            local_srun_alive=srun_alive,
            status="flags" if flags else "progress_observed",
            flags=flags,
            automatic_action="record_only; no cancellation or duplicate model attempt",
        )

    def health_tick(final=False):
        with state_lock:
            states = dict(driver_states)
        due = [
            (run, state)
            for run, state in states.items()
            if run not in health_receipts
            and (
                final
                or time.time() - state["started"] >= 900
                or (root / "drivers" / run / "settled.json").is_file()
            )
        ]
        live_jobs = queue() if due else {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(check_driver_health, run, state, live_jobs): run
                for run, state in due
            }
            for future in concurrent.futures.as_completed(futures):
                run = futures[future]
                try:
                    health_receipts[run] = future.result()
                except Exception as error:
                    health_receipts[run] = dict(
                        run=run,
                        status="ambiguous",
                        flags=[repr(error)],
                        automatic_action="record_only",
                    )
        save(
            root / "infra_health_15min.json",
            dict(
                required_drivers=80,
                checked_drivers=len(health_receipts),
                all_drivers_checked=len(health_receipts) == 80,
                drivers=health_receipts,
            ),
        )

    def monitor_health():
        while not health_stop.wait(30):
            try:
                health_tick()
            except Exception as error:
                save(root / "infra_health_monitor_error.json", dict(error=repr(error)))

    def driver(group, item, job, url):
        run = item["run"]
        cfg = item["config"]
        cfg["agents"][0]["kwargs"]["api_base"] = url
        config = root / "runs" / f"{run}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        directory = root / "drivers" / run
        directory.mkdir()
        node = url.split("://", 1)[1].split(":", 1)[0]
        with driver_start:
            time.sleep(max(0, next_start[0] - time.monotonic()))
            next_start[0] = time.monotonic() + 5
        remaining = queue().get(job)
        assert remaining and remaining[0] == "RUNNING", job
        driver_env = dict(
            env,
            CFG=str(config),
            SERVE_JOB=job,
            RUN_NAME=run,
            JOBS=str(root / "jobs"),
            LOGD=str(directory),
            SCANCEL_SERVE="0",
            TUNNEL_PORTS="18080,18081",
            BRIDGE_PORT=str(item["bridge_port"]),
        )
        with state_lock:
            driver_states[run] = dict(
                started=time.time(), job=job, url=url, expected=len(cfg["tasks"])
            )
        with (directory / "driver.out").open("w") as output:
            process = subprocess.Popen(
                [
                    "srun",
                    "-p",
                    "debug",
                    "-A",
                    "CCR24067",
                    "-t",
                    remaining[2],
                    "--jobid=" + job,
                    "--overlap",
                    "-N1",
                    "-n1",
                    "-w",
                    node,
                    "-c",
                    "32",
                    "--export=ALL",
                    "bash",
                    str(ota / "data/tb2/horizon/tb2_driver.sbatch"),
                ],
                env=driver_env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            with state_lock:
                driver_processes[run] = process
            rc = process.wait()
        expected = len(cfg["tasks"])
        results = sorted((root / "jobs" / run).glob("*/result.json"))
        receipt = dict(
            run=run,
            driver_exit=rc,
            expected=expected,
            result_files=len(results),
            settled_driver=True,
            results=[str(path) for path in results],
        )
        save(directory / "settled.json", receipt)
        if rc != 0 or len(results) != expected:
            raise RuntimeError(f"Incomplete driver {run}: {receipt}")
        return receipt

    def group_worker(group):
        key = group["key"]
        job = None
        try:
            while job is None:
                with admission:
                    q = queue()
                    own_nodes = sum(int(value[1]) for value in q.values())
                    with state_lock:
                        active = [j for j in jobs.values() if j in q]
                    # Deliberately conservative: reserve all seats of our live serves in addition to the API count.
                    room = (
                        own_nodes + 5 <= 100
                        and len(q) < 20
                        and started() + (len(active) + 1) * 80 <= 2000
                    )
                    if room:
                        submission = subprocess.run(
                            [
                                "bash",
                                str(ota / "data/relay/horizon/serve_submit.sh"),
                                "5",
                                "5",
                                "04:30:00",
                                "cfmix_" + key,
                            ],
                            env=dict(env, STUDENT_MODEL=str(models[group["arm"]])),
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if submission.returncode == 75:
                            time.sleep(30)
                            continue
                        if submission.returncode != 0:
                            raise RuntimeError(submission.stderr or submission.stdout)
                        result = submission.stdout.strip()
                        job = result.splitlines()[-1]
                        assert job.isdigit(), result
                        with state_lock:
                            jobs[key] = job
                            save(root / "jobs.json", jobs)
                if job is None:
                    time.sleep(30)
            endpoint = root / "serve/endpoints" / f"{job}.student"
            deadline = time.monotonic() + 24 * 3600
            while not endpoint.is_file():
                if (
                    root / "serve/endpoints" / f"{job}.DEAD"
                ).exists() or job not in queue():
                    raise RuntimeError(f"Serve {job} ended before readiness")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Serve {job} queue/readiness deadline")
                time.sleep(15)
            urls = endpoint.read_text().strip().split(",")
            assert len(urls) == 5 and len(set(urls)) == 5, urls
            for port in ["18080", "18081"]:
                output = (root / "serve/logs" / f"tunnel_{job}_{port}.log").open("w")
                process = subprocess.Popen(
                    ["bash", str(ota / "data/r2egym/horizon/tunnel.sh"), job, port],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                with state_lock:
                    tunnels.append((process, output))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                futures = [
                    pool.submit(driver, group, item, job, urls[item["node_index"]])
                    for item in group["drivers"]
                ]
                receipts = [future.result() for future in futures]
            with state_lock:
                completed[key] = receipts
                save(root / "completed.json", completed)
            return key
        finally:
            if job is not None:
                # Only this controller's own serving allocation is released, after its driver futures settle.
                subprocess.run(["scancel", job], check=False)

    (root / "STATUS").write_text("EVAL_STARTING\n")
    failures = []
    health_thread = threading.Thread(target=monitor_health, daemon=True)
    health_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            future_map = {pool.submit(group_worker, group): group for group in groups}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    future.result()
                except Exception as error:
                    failures.append(
                        dict(group=future_map[future]["key"], error=repr(error))
                    )
                    save(root / "failures.json", failures)
    finally:
        health_stop.set()
        health_thread.join()
        try:
            health_tick(final=True)
        except Exception as error:
            save(
                root / "infra_health_monitor_error.json",
                dict(error=repr(error), final=True),
            )
        for process, output in tunnels:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            output.close()
    readout = summarize(
        root, root.parent / "cf2-repaired-readout-v2-20261023/report.json"
    )
    save(root / "results.json", readout)
    (root / "results.md").write_text(markdown(readout))
    if failures:
        (root / "STATUS").write_text(
            "EVAL_FINISHED_WITH_ERRORS: inspect failures.json; no score imputation\n"
        )
        raise RuntimeError(failures)
    assert inventory() == data, "Frozen input/task/config drift during eval"
    (root / "STATUS").write_text(
        "READOUT_READY: results.json/results.md; inspect numeric coverage and unknowns\n"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--dry-run" not in sys.argv and "--root" in sys.argv:
            failure_root = Path(sys.argv[sys.argv.index("--root") + 1])
            if failure_root.is_dir():
                (failure_root / "STATUS").write_text(
                    "EVAL_FAILED: " + repr(error) + "\n"
                )
        raise
