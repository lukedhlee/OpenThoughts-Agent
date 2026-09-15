#!/usr/bin/env python3
"""Daytona sandbox-creation ramp stress test: how fast can a run open ~1,056 sandboxes without start timeouts.

One run = open sandboxes from one snapshot on a schedule (burst or ramp), hold them alive, optionally churn
(delete + replace in small groups, the RL steady state), then delete everything at a capped rate and verify the org
list shows none of this run's label left. Creation goes through the same calls the SDK's `AsyncDaytona.create` makes
(create_sandbox, then `wait_for_sandbox_start` with its websocket + 1 s poll), split so that client-side queueing
(scheduled -> request sent), the create call itself, and the wait for `started` are timed separately.

Every sandbox gets label harbor.instance=<label> plus ephemeral + auto-stop + ttl as a leak guard. Resources are not
passed: like harbor's snapshot path, a sandbox inherits its snapshot's resources (recorded in the run meta).

Usage (Jupiter login node, snowball-v2 venv, one process):
  OMP_NUM_THREADS=1 $V/bin/python ramp_stress.py --run r1_burst100 --target 100 --batch 100 --interval 0
  ... --run r3_ramp50x10 --target 1056 --batch 50 --interval 10
  ... --run r8_churn --target 1056 --batch 100 --interval 20 --hold 0 --churn-minutes 10
  ... --cleanup-label ramp-stress-lee27-r3_ramp50x10      # delete leftovers of one label only
Rows: <out>/<run>.jsonl (one per sandbox), <out>/<run>.summary.json. Exit 3 = aborted on a rate/quota error.
"""
import argparse
import asyncio
import collections
import json
import os
import random
import re
import resource
import sys
import time

KEY_FILE = "/e/fscratch/reformo/lee27/keys/daytona_eval.env"
OUT_DIR = "/e/fscratch/reformo/lee27/experiments/daytona_ramp_stress"
LABEL_PREFIX = "ramp-stress-lee27-"
QUOTA_RE = re.compile(r"quota|limit|exceed|too many|insufficient|capacity", re.I)


def load_key(path: str) -> str:
    for line in open(path):
        line = line.strip()
        if line.startswith(("DAYTONA_API_KEY=", "export DAYTONA_API_KEY=")):
            return re.sub(r"\s+#.*$", "", line.split("=", 1)[1]).strip().strip("\"'")
    raise SystemExit(f"DAYTONA_API_KEY not in {path}")


def err_info(e: BaseException) -> dict:
    status = getattr(e, "status", None) or getattr(e, "status_code", None)
    headers = getattr(e, "headers", None) or {}
    rl = {k: v for k, v in dict(headers).items() if re.search(r"rate|retry", k, re.I)}
    body = getattr(e, "body", None)
    if status is not None:
        msg = f"{getattr(e, 'reason', '')} body={body}"
    else:
        msg = str(e)
    out = {"type": type(e).__name__, "status": status, "msg": msg[:400]}
    if rl:
        out["rl"] = rl
    return out


def is_limit_error(info: dict) -> bool:
    return info.get("status") == 429 or "RateLimit" in info["type"] or bool(QUOTA_RE.search(info["msg"]))


def is_create_throttle(info: dict) -> bool:
    """The per-second sandbox-create limiter (429 + Retry-After-sandbox-create), as opposed to a quota refusal."""
    return info.get("status") == 429


def enum_val(x):
    return getattr(x, "value", x)


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1)


class Stress:
    def __init__(self, a, d, api):
        self.a, self.d, self.api = a, d, api
        self.label = LABEL_PREFIX + a.run
        self.rows: list[dict] = []
        self.live: dict[str, dict] = {}  # sandbox id -> row, started and not yet deleted
        self.pending: set[asyncio.Task] = set()
        self.abort = asyncio.Event()
        self.abort_reason = None
        self.t0 = time.time()
        self.max_lag = 0.0
        self.wait_limit_errors = 0
        self.create_429 = 0
        self.census_series: list[dict] = []

    def rel(self, t):
        return None if t is None else round(t - self.t0, 2)

    async def lag_monitor(self):
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.25)
            self.max_lag = max(self.max_lag, time.perf_counter() - t - 0.25)

    async def census(self, label: str | None = None) -> dict:
        """Org sandbox count by state (all labels), or only one label's sandboxes."""
        by_state, by_label_kind, n, cursor, pages = collections.Counter(), collections.Counter(), 0, None, 0
        labels = json.dumps({"harbor.instance": label}) if label else None
        while True:
            r = await self.api.list_sandboxes(cursor=cursor, limit=200, labels=labels, _request_timeout=120.0)
            pages += 1
            for s in r.items:
                inst = (s.labels or {}).get("harbor.instance", "")
                if label and inst != label:
                    continue
                n += 1
                st = str(enum_val(s.state))
                by_state[st] += 1
                kind = "ours_ramp" if inst.startswith(LABEL_PREFIX) else "other"
                by_label_kind[f"{kind}:{st}"] += 1
            cursor = r.next_cursor
            if not cursor or pages > 200:
                break
        return {"t": round(time.time(), 1), "total": n, "by_state": dict(by_state), "by_kind_state": dict(by_label_kind)}

    async def open_one(self, idx: int, phase: str, t_sched: float) -> None:
        from daytona import AsyncSandbox
        from daytona_api_client_async import CreateSandbox

        a = self.a
        row = {"run": a.run, "idx": idx, "phase": phase, "snapshot": a.snapshot, "t_sched": t_sched}
        self.rows.append(row)
        delay = t_sched - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if self.abort.is_set():
            row["skipped"] = True
            return
        row["t_req_first"] = time.time()
        n429, first429 = 0, None
        while True:
            row["t_req"] = time.time()
            try:
                http = await self.api.create_sandbox_with_http_info(
                    CreateSandbox(
                        snapshot=a.snapshot,
                        labels={"harbor.instance": self.label, "code-toolbox-language": "python"},
                        env={},
                        auto_stop_interval=a.auto_stop,
                        auto_delete_interval=0,  # ephemeral: deleted as soon as it stops
                        ttl_minutes=a.ttl,
                    ),
                    _request_timeout=float(a.create_http_timeout),
                )
                resp = http.data
                hdr = dict(http.headers or {})
                row["rl_remaining"] = hdr.get("X-RateLimit-Remaining-sandbox-create")
                break
            except Exception as e:  # noqa: BLE001
                info = err_info(e)
                if a.retry_mode != "none" and is_create_throttle(info):
                    n429 += 1
                    first429 = first429 or info
                    self.create_429 += 1
                    if a.retry_mode == "harbor":
                        # harbor's SANDBOX_WAIT for DaytonaRateLimitError: 60 s x attempt, no jitter, < 10 attempts
                        wait = 60.0 * n429
                        give_up = n429 >= 10
                    else:
                        wait = float((info.get("rl") or {}).get("Retry-After-sandbox-create", 1) or 1)
                        wait += random.random() * a.retry_jitter
                        give_up = False
                    if give_up or time.time() + wait - t_sched >= a.start_timeout:
                        row["t_create_ret"] = time.time()
                        row["n429"] = n429
                        row["first429"] = first429
                        row["start_timeout"] = True
                        row["gave_up_in_429_backoff"] = True
                        return
                    await asyncio.sleep(wait)
                    continue
                row["t_create_ret"] = time.time()
                row["create_error"] = info
                if n429:
                    row["n429"] = n429
                if is_limit_error(info) and not self.abort.is_set():
                    self.abort_reason = f"create {info}"
                    self.abort.set()
                return
        if n429:
            row["n429"] = n429
            row["first429"] = first429
        row["t_create_ret"] = time.time()
        row["id"] = resp.id
        row["state_at_create"] = str(enum_val(resp.state))
        row["runner_id"] = resp.runner_id
        row["res"] = [resp.cpu, resp.memory, resp.disk]
        sb = AsyncSandbox(
            await self.d._ensure_toolbox_proxy_url(resp),
            self.d._toolbox_api_client,
            self.d._sandbox_api,
            "python",
            self.d._subscription_manager,
            self.d._pool_tracker,
            analytics_api_url_provider=self.d._get_analytics_api_url,
        )
        deadline = row["t_req"] + a.start_timeout
        wait_errors = []
        while True:
            remaining = deadline - time.time()
            if remaining <= 1:
                row["start_timeout"] = True
                break
            try:
                await sb.wait_for_sandbox_start(timeout=remaining)
                row["t_started"] = time.time()
                break
            except Exception as e:  # noqa: BLE001
                st = str(enum_val(getattr(sb, "state", None)))
                if st in ("error", "build_failed"):
                    row["start_error"] = {"state": st, "reason": getattr(sb, "error_reason", None), **err_info(e)}
                    break
                if time.time() >= deadline - 1:
                    row["start_timeout"] = True
                    break
                info = err_info(e)
                wait_errors.append({"t": round(time.time(), 1), **info})
                if is_limit_error(info):
                    self.wait_limit_errors += 1
                    if self.wait_limit_errors >= a.max_wait_limit_errors and not self.abort.is_set():
                        self.abort_reason = f"{self.wait_limit_errors} rate/quota errors while waiting, last {info}"
                        self.abort.set()
                await asyncio.sleep(2 + random.random() * 3)
        if wait_errors:
            row["wait_errors_n"] = len(wait_errors)
            row["wait_errors"] = wait_errors[:5]
        row["final_state"] = str(enum_val(getattr(sb, "state", None)))
        if row.get("t_started"):
            self.live[resp.id] = row
        else:
            # never started: still delete it at the end
            self.live.setdefault(resp.id, row)

    async def delete_one(self, sid: str, row: dict) -> None:
        row["t_del_req"] = time.time()
        try:
            await self.api.delete_sandbox(sid, _request_timeout=120.0)
        except Exception as e:  # noqa: BLE001
            row["del_error"] = err_info(e)
        row["t_del_ret"] = time.time()

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self.pending.add(t)
        t.add_done_callback(self.pending.discard)
        return t

    def progress(self, tag: str) -> str:
        open_rows = self.rows
        c = collections.Counter()
        for r in open_rows:
            c["sched"] += 1
            c["req"] += "t_req" in r
            c["created"] += "id" in r
            c["started"] += "t_started" in r
            c["create_err"] += "create_error" in r
            c["start_err"] += "start_error" in r
            c["timeout"] += bool(r.get("start_timeout"))
            c["deleted"] += "t_del_ret" in r
        return (f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:6.0f}s {tag}] " + " ".join(f"{k}={v}" for k, v in c.items())
                + f" live={len(self.live)} lag_max={self.max_lag:.2f}s create429={self.create_429} wait_limit_err={self.wait_limit_errors}")

    async def reporter(self, tag_ref: list):
        n = 0
        while True:
            await asyncio.sleep(10)
            n += 1
            print(self.progress(tag_ref[0]), flush=True)
            if n % 3 == 0:  # org load every 30 s: other teams share the create bucket
                try:
                    c = await self.census()
                    c["phase"], c["rel_s"] = tag_ref[0], self.rel(c["t"])
                    self.census_series.append(c)
                except Exception as e:  # noqa: BLE001
                    self.census_series.append({"t": time.time(), "error": err_info(e)})

    async def run(self) -> dict:
        a = self.a
        meta = {"run": a.run, "label": self.label, "snapshot": a.snapshot, "target": a.target, "batch": a.batch,
                "interval": a.interval, "hold": a.hold, "churn_minutes": a.churn_minutes, "auto_stop": a.auto_stop,
                "ttl": a.ttl, "start_timeout": a.start_timeout, "pool": a.pool, "retry_mode": a.retry_mode}
        try:
            snap = await self.d.snapshot.get(a.snapshot)
            meta["snapshot_info"] = {"state": str(enum_val(snap.state)), "cpu": snap.cpu, "memory": snap.mem,
                                     "disk": snap.disk, "size_gb": getattr(snap, "size", None)}
        except Exception as e:  # noqa: BLE001
            meta["snapshot_info"] = {"error": err_info(e)}
        meta["census_before"] = await self.census()
        print("meta", json.dumps(meta), flush=True)
        tag = ["open"]
        mon = asyncio.create_task(self.lag_monitor())
        rep = asyncio.create_task(self.reporter(tag))

        # open
        self.t0 = time.time()
        t_start = self.t0 + 1.0
        opens = []
        if a.interval <= 0:
            opens = [self.spawn(self.open_one(i, "open", t_start)) for i in range(a.target)]
        else:
            k = 0
            for b in range(0, a.target, a.batch):
                for i in range(b, min(b + a.batch, a.target)):
                    opens.append(self.spawn(self.open_one(i, "open", t_start + k * a.interval)))
                k += 1
        await asyncio.gather(*opens)
        t_opened = time.time()
        meta["open_done_s"] = self.rel(t_opened)
        print(self.progress("opened"), flush=True)
        meta["census_peak"] = await self.census()

        # hold
        tag[0] = "hold"
        if not self.abort.is_set() and a.hold > 0:
            await asyncio.sleep(a.hold)

        # churn: delete + replace `group` sandboxes every 60*group/per_min seconds
        if a.churn_minutes > 0 and not self.abort.is_set():
            tag[0] = "churn"
            step = 60.0 * a.churn_group / a.churn_per_min
            t_end = time.time() + a.churn_minutes * 60
            idx = a.target
            meta["churn_start_s"] = self.rel(time.time())
            while time.time() < t_end and not self.abort.is_set():
                started = [sid for sid, r in self.live.items() if r.get("t_started")]
                for sid in random.sample(started, min(a.churn_group, len(started))):
                    self.spawn(self.delete_one(sid, self.live.pop(sid)))
                now = time.time()
                for _ in range(a.churn_group):
                    self.spawn(self.open_one(idx, "churn", now))
                    idx += 1
                await asyncio.sleep(step)
            meta["churn_end_s"] = self.rel(time.time())
            churn_open = [t for t in list(self.pending)]
            if churn_open:
                await asyncio.gather(*churn_open, return_exceptions=True)
            meta["census_churn_end"] = await self.census()

        # delete everything still alive, <= delete_batch per delete_interval
        tag[0] = "delete"
        meta["delete_start_s"] = self.rel(time.time())
        items = list(self.live.items())
        self.live.clear()
        dels = []
        for b in range(0, len(items), a.delete_batch):
            for sid, row in items[b:b + a.delete_batch]:
                dels.append(self.spawn(self.delete_one(sid, row)))
            if b + a.delete_batch < len(items):
                await asyncio.sleep(a.delete_interval)
        await asyncio.gather(*dels, return_exceptions=True)
        meta["delete_done_s"] = self.rel(time.time())
        print(self.progress("deleted"), flush=True)

        # write rows now, then verify the label is empty (list lags; re-check)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{a.run}.jsonl"), "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        meta["verify"] = await verify_label_empty(self, self.label)
        meta["census_after"] = await self.census()
        rep.cancel()
        mon.cancel()
        meta["max_loop_lag_s"] = round(self.max_lag, 2)
        meta["abort_reason"] = self.abort_reason
        meta["census_series"] = self.census_series
        meta["summary"] = summarize(self.rows, self.t0, a.start_timeout)
        with open(os.path.join(a.out, f"{a.run}.summary.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("summary", json.dumps(meta["summary"]), flush=True)
        print("verify", json.dumps(meta["verify"]), flush=True)
        return meta


async def verify_label_empty(s: Stress, label: str) -> list:
    """Re-list the label until nothing but destroyed is left; delete any stragglers (ours by label only)."""
    checks = []
    for wait in (5, 30, 60, 90, 120, 180):
        await asyncio.sleep(wait)
        c = await s.census(label)
        alive = {k: v for k, v in c["by_state"].items() if k not in ("destroyed",)}
        checks.append({"after_s": wait, "total": c["total"], "by_state": c["by_state"]})
        print(f"verify {label} +{wait}s {c['by_state']}", flush=True)
        if not alive:
            if len(checks) >= 2:
                break
            continue
        # stragglers: delete the ones not already being destroyed
        cursor = None
        while True:
            r = await s.api.list_sandboxes(cursor=cursor, limit=200, labels=json.dumps({"harbor.instance": label}),
                                           _request_timeout=120.0)
            for it in r.items:
                if (it.labels or {}).get("harbor.instance") != label:
                    continue
                if str(enum_val(it.state)) not in ("destroyed", "destroying"):
                    try:
                        await s.api.delete_sandbox(it.id, _request_timeout=120.0)
                    except Exception as e:  # noqa: BLE001
                        checks[-1].setdefault("straggler_del_errors", []).append(err_info(e))
            cursor = r.next_cursor
            if not cursor:
                break
    return checks


def summarize(rows: list, t0: float, start_timeout: float) -> dict:
    out = {}
    for phase in sorted({r["phase"] for r in rows}):
        rs = [r for r in rows if r["phase"] == phase]
        req = [r for r in rs if "t_req" in r]
        started = [r for r in rs if "t_started" in r]
        lat_sched = [r["t_started"] - r["t_sched"] for r in started]
        n429 = [r.get("n429", 0) for r in req]
        lat_req = [r["t_started"] - r["t_req"] for r in started]
        create_call = [r["t_create_ret"] - r["t_req"] for r in req if "t_create_ret" in r]
        queue = [r["t_req_first"] - r["t_sched"] for r in req]
        errs = collections.Counter()
        for r in rs:
            for k in ("create_error", "start_error"):
                if k in r:
                    e = r[k]
                    errs[f"{k}:{e.get('type')}:{e.get('status')}:{(e.get('reason') or e.get('msg') or '')[:120]}"] += 1
            for w in r.get("wait_errors", []):
                errs[f"wait_error:{w.get('type')}:{w.get('status')}:{w.get('msg', '')[:120]}"] += 1
        timeouts = sum(1 for r in rs if r.get("start_timeout"))
        gave_up = sum(1 for r in rs if r.get("gave_up_in_429_backoff"))
        dels = [r for r in rs if "t_del_req" in r]
        del_lat = [r["t_del_ret"] - r["t_del_req"] for r in dels if "t_del_ret" in r]
        out[phase] = {
            "scheduled": len(rs), "skipped": sum(1 for r in rs if r.get("skipped")), "requested": len(req),
            "created": sum(1 for r in rs if "id" in r), "started": len(started),
            "time_to_all_started_s": round(max(r["t_started"] for r in started) - t0, 1) if started else None,
            "start_from_sched_p50_p90_max": [pct(lat_sched, 50), pct(lat_sched, 90), pct(lat_sched, 100)],
            "start_from_req_p50_p90_max": [pct(lat_req, 50), pct(lat_req, 90), pct(lat_req, 100)],
            "create_call_p50_p90_max": [pct(create_call, 50), pct(create_call, 90), pct(create_call, 100)],
            "client_queue_max_s": pct(queue, 100),
            "create_429_total": sum(n429), "rows_with_429": sum(1 for x in n429 if x), "max_429_per_sandbox": max(n429, default=0),
            "e2e_over_600s": sum(1 for x in lat_sched if x > 600) + timeouts, "e2e_over_60s": sum(1 for x in lat_sched if x > 60) + timeouts,
            "create_attempts": len(req) + sum(n429), "create_429_rate": round(sum(n429) / max(1, len(req) + sum(n429)), 3),
            "over_600s": sum(1 for x in lat_req if x > 600) + timeouts, "over_60s": sum(1 for x in lat_req if x > 60) + timeouts,
            "never_started_timeout": timeouts, "gave_up_in_429_backoff": gave_up, "start_timeout_cap_s": start_timeout,
            "errors": dict(errs.most_common(20)),
            "wait_error_rows": sum(1 for r in rs if r.get("wait_errors_n")),
            "deleted": len(dels), "delete_errors": sum(1 for r in dels if "del_error" in r),
            "delete_call_p50_max": [pct(del_lat, 50), pct(del_lat, 100)],
            "runners": len({r.get("runner_id") for r in rs if r.get("runner_id")}),
            "state_at_create": dict(collections.Counter(r.get("state_at_create") for r in rs if "id" in r)),
        }
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="adhoc")
    ap.add_argument("--snapshot", default="harbor__8c0fc945338b__snapshot")
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1, help="sandboxes per wave")
    ap.add_argument("--interval", type=float, default=0, help="seconds between waves; 0 = burst all at once")
    ap.add_argument("--hold", type=float, default=180)
    ap.add_argument("--churn-minutes", type=float, default=0)
    ap.add_argument("--churn-per-min", type=float, default=75)
    ap.add_argument("--churn-group", type=int, default=8)
    ap.add_argument("--delete-batch", type=int, default=100)
    ap.add_argument("--delete-interval", type=float, default=10)
    ap.add_argument("--start-timeout", type=float, default=900, help="give up waiting after this; >600 counts as timeout")
    ap.add_argument("--create-http-timeout", type=float, default=600)
    ap.add_argument("--max-wait-limit-errors", type=int, default=50)
    ap.add_argument("--retry-429", action="store_true", help="same as --retry-mode header")
    ap.add_argument("--retry-mode", choices=["none", "header", "harbor"], default="none",
                    help="create 429s: none = record + abort; header = Retry-After-sandbox-create + jitter; "
                         "harbor = 60 s x attempt, no jitter (harbor's current tenacity policy)")
    ap.add_argument("--retry-jitter", type=float, default=1.0)
    ap.add_argument("--auto-stop", type=int, default=30, help="minutes idle before stop (ephemeral: then deleted)")
    ap.add_argument("--ttl", type=int, default=30, help="minutes wall-clock before Daytona destroys it regardless")
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--cleanup-label", default=None, help=f"only delete + verify sandboxes with this label ({LABEL_PREFIX}*)")
    ap.add_argument("--census-only", action="store_true")
    a = ap.parse_args()
    if a.retry_429 and a.retry_mode == "none":
        a.retry_mode = "header"

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    from daytona import AsyncDaytona, DaytonaConfig

    async with AsyncDaytona(DaytonaConfig(api_key=load_key(KEY_FILE), connection_pool_maxsize=a.pool)) as d:
        api = d._sandbox_api
        s = Stress(a, d, api)
        if a.census_only:
            print(json.dumps(await s.census(), indent=1))
            return 0
        if a.cleanup_label:
            if not a.cleanup_label.startswith(LABEL_PREFIX):
                raise SystemExit(f"refusing to clean a label that is not {LABEL_PREFIX}*")
            print(json.dumps(await verify_label_empty(s, a.cleanup_label), indent=1))
            return 0
        meta = await s.run()
        return 3 if meta.get("abort_reason") else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
