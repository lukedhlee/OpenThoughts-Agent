#!/usr/bin/env python3
"""gepa_runner.py — the login-node controller for the GEPA loop's candidate queue.

The Opus session enqueues a candidate (gepa_queue.sh add); this loop keeps the standing serve job busy with it.
Nothing here submits a Slurm job: the servers are already up, and a candidate is just N `harbor run`s pointed at
them on the Daytona backend.

  queue/<wave>.<cand>.json   -> waiting             (written by gepa_queue.sh)
  running/<wave>.<cand>.json -> shards in flight    (holds the endpoints and tmux names it owns)
  done/<wave>.<cand>.json    -> finished, with each leg's harbor run dirs
  runs.jsonl                 -> the append-only RUN ledger: one line per finished candidate

A candidate runs as ORDERED LEGS, feedback then dev. Feedback is 64 train tasks against dev's 500, so it lands
first and the session can start reading while the dev leg is still going -- and reading is all feedback is for:
dev is scores-only, and its trajectories are closed (gepa_feat.refuse_closed_task). When a leg finishes the runner
starts the next one immediately on the endpoints that leg just freed, so a candidate never holds idle servers
between its legs.

Why runs.jsonl and not ledger.json: ledger.json is the POPULATION ledger and every record in it carries dev scores.
A finished run is not a scored candidate yet -- the session still has to run gepa_score.py. Writing unscored rows
into ledger.json would make `gepa_ledger.py front` lie. So the runner records runs, and gepa_ledger.py add records
candidates once they have numbers.

Admission control, in order:
  * at most MAX_INFLIGHT candidates at once;
  * at most one shard per live endpoint, so no two harbor runs share a server's seats. That invariant is what
    bounds Daytona: concurrent sandboxes can never exceed live_endpoints x CONC, checked every pass against
    SANDBOX_CAP (~1,000 per user, 5 creates/s). Shards per candidate are AUTO by default -- the live endpoint
    count divided between the in-flight candidates -- so a 1-node pilot and an 8-node job need no reconfiguring;
  * an ORG guard: the Daytona org is shared with other people's jobs, so before admitting anything the runner asks
    Daytona how many sandboxes are actually started and holds above ORG_SANDBOX_HOLD. Our own arithmetic only bounds
    our own share; the budget is org-wide;
  * a login-node THREAD guard. 2026-09-03 a torch import on the login node hit the 4,096 pid ceiling and took
    Jupiter down for everyone. Each harbor run is worth hundreds of threads at conc 32, so the runner refuses to
    admit another candidate while the user's thread count is above PID_GUARD.

Usage (inside the tmux that gepa_queue.sh start creates):
  gepa_runner.py                      # loop until stopped
  gepa_runner.py --once --dry-run     # one admission pass, printing what it WOULD launch
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, glob, json, os, subprocess, sys, time

E = os.environ.get("GEPA_DIR", "/e/fscratch/reformo/lee27/experiments/gepa")
G = os.environ.get("GEPA_CODE", "/e/project1/transfernetx/lee27/code/snowball/gepa")
JOBS = os.environ.get("GEPA_JOBS", "/e/data1/mmlaion/lee27/experiments/gepa_jobs")
KEYF = os.environ.get("DAYTONA_KEYF", "/e/fscratch/reformo/lee27/keys/daytona_eval.env")
ap = argparse.ArgumentParser()
ap.add_argument("--max-inflight", type=int, default=int(os.environ.get("MAX_INFLIGHT", "2")))
ap.add_argument("--shards", type=int, default=int(os.environ.get("SHARDS_PER_CAND", "0")),
                help="harbor runs per candidate, each taking one server. 0 (the default) = AUTO: read the live "
                     "endpoint count and give each in-flight candidate its share, so a 1-node pilot uses 1 shard "
                     "and an 8-node job uses 4. Set a number only to hold a candidate below its share.")
ap.add_argument("--conc", type=int, default=int(os.environ.get("CONC", "32")),
                help="concurrent trials per shard; must not exceed the server's --max-num-seqs")
ap.add_argument("--sandbox-cap", type=int, default=int(os.environ.get("SANDBOX_CAP", "900")),
                help="refuse a layout whose peak concurrent sandboxes would exceed this")
ap.add_argument("--pid-guard", type=int, default=int(os.environ.get("PID_GUARD", "2800")),
                help="do not admit while this user's thread count on the login node is above this (cap is 4096)")
ap.add_argument("--org-hold", type=int, default=int(os.environ.get("ORG_SANDBOX_HOLD", "700")),
                help="hold admission while this many sandboxes are STARTED in the shared Daytona org")
ap.add_argument("--daytona-url", default=os.environ.get("DAYTONA_LIST_URL", "https://app.daytona.io/api/sandbox"))
ap.add_argument("--fake-org-count", type=int, default=None, help="dry-run only: pretend the org has this many started sandboxes")
ap.add_argument("--poll", type=int, default=int(os.environ.get("POLL", "60")))
ap.add_argument("--retries", type=int, default=1, help="relaunch a candidate whose shards failed, this many times")
ap.add_argument("--once", action="store_true")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--serve-job", default=None,
                help="use this job id instead of looking one up by name -- to adopt a serve job submitted elsewhere, "
                     "or to exercise admission against hand-written endpoint files in a dry run")
a = ap.parse_args()
Q, R, D = E + "/queue", E + "/running", E + "/done"
for d in (Q, R, D, E + "/runs", E + "/logs"):
    os.makedirs(d, exist_ok=True)


def log(*m):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *m, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def serve_job():
    if a.serve_job:
        return a.serve_job
    out = sh("squeue -h -u $USER -n gepa_serve -o '%i %T'").stdout.split()
    if len(out) >= 2 and out[1] == "RUNNING":
        return out[0]
    return None


def endpoints(job):
    """url -> endpoint file, for every server of the serve job that has published one."""
    out = {}
    for f in sorted(glob.glob("%s/endpoints/%s.*" % (E, job))):
        try:
            u = open(f).read().strip()
        except OSError:
            continue
        if u:
            out[u] = f
    return out


def org_sandboxes():
    """Sandboxes currently STARTED in the Daytona org, or None if the count could not be read.

    The org is shared: other people's jobs and other teams use the same ~1,000-sandbox budget, so this loop's own
    arithmetic (live endpoints x conc) is not the whole picture. Before admitting anything, ask Daytona what is
    actually running and hold if the org is already busy. Listing pages with `nextCursor` (`page=` is ignored) and
    GET/list is limited to 15,000 per 30 s, so one paged walk per admission pass is well inside the budget.

    None means "could not tell" -- the caller treats that as a soft pass and logs it, because failing closed on a
    transient API error would stall a loop that is otherwise healthy."""
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        for line in (open(KEYF).read().splitlines() if os.path.exists(KEYF) else []):
            line = line.strip()
            if line.startswith("DAYTONA_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"\'').split("#")[0].strip()
                break
    if not key:
        return None
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request
    started, cursor, pages = 0, None, 0
    while pages < 40:
        q = {"limit": "100"}
        if cursor:
            q["cursor"] = cursor
        url = "%s?%s" % (a.daytona_url, urllib.parse.urlencode(q))
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key,
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = _json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            log("daytona list failed (%s: %s); not holding on an unreadable count" % (type(e).__name__, e))
            return None
        if isinstance(d, list):
            items, cursor = d, None
        else:
            items = d.get("items") or d.get("data") or d.get("sandboxes") or []
            cursor = d.get("nextCursor") or d.get("next_cursor") or None
        for it in items:
            st = it.get("state") if isinstance(it, dict) else None
            if str(st).lower() == "started":
                started += 1
        pages += 1
        if not cursor or not items:
            break
    return started


def threads_used():
    r = sh("ps -u $USER -L -o pid= 2>/dev/null | wc -l")
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def items(d):
    return sorted(glob.glob(d + "/*.json"), key=os.path.getmtime)


def read(p):
    with open(p) as f:
        return json.load(f)


def write(p, o):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=1)
    os.replace(tmp, p)


def shard_state(name):
    """(done, exit code) for one harbor run, read from the tail its tmux wrapper appends."""
    p = "%s/logs/run_%s.log" % (E, name)
    if not os.path.exists(p):
        return False, None
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False, None
    for ln in reversed(tail.splitlines()):
        if ln.startswith("RUN_DONE"):
            try:
                return True, int(ln.split("exit=")[1].split()[0])
            except (IndexError, ValueError):
                return True, None
    return False, None


def legs_of(it):
    """Back-compat: an item written before legs existed is a single unnamed leg."""
    return it.get("legs") or [{"leg": it.get("split", "dev"), "tasks": it.get("tasks", [])}]


def reap(free, nlive):
    """Finish whatever shards are done. A candidate whose leg completes starts its NEXT leg immediately, reusing the
    endpoints that leg just freed, so the servers never wait for the whole candidate."""
    for p in items(R):
        it = read(p)
        if not it.get("shards"):  # mid-candidate, waiting for endpoints to start the next leg
            if free and start_leg(it, free, nlive):
                write(p, it)
            continue
        states = [(s, shard_state(s["name"])) for s in it["shards"]]
        if not all(dn for _, (dn, _) in states):
            continue
        codes = [c for _, (_, c) in states]
        bad = [s["name"] for s, (_, c) in states if c not in (0,)]
        for s, (_, c) in states:
            s["exit"] = c
            s["run_dir"] = "%s/%s" % (JOBS, s["name"])
            s["trials"] = len(glob.glob("%s/%s/*__*" % (JOBS, s["name"])))
        legs = legs_of(it)
        name = legs[it.get("leg", 0)]["leg"]
        res = it.setdefault("results", {})
        res[name] = {"run_dirs": [s["run_dir"] for s in it["shards"]],
                     "trials": sum(s["trials"] for s in it["shards"]),
                     "exits": codes, "status": "done" if not bad else "failed",
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        for s in it["shards"]:
            if s["url"] in busy_endpoints() or s["url"] in free:
                continue
            free.append(s["url"])
        log("leg %s of %s.%s %s: %d trials over %d shards"
            % (name, it["wave"], it["cand"], res[name]["status"], res[name]["trials"], len(it["shards"])))
        it["shards"] = []

        if bad and it.get("attempt", 1) <= a.retries:
            log("candidate %s.%s: shards failed %s (exits %s) -- requeueing attempt %d"
                % (it["wave"], it["cand"], bad, codes, it.get("attempt", 1) + 1))
            it["attempt"] = it.get("attempt", 1) + 1
            it["leg"] = 0
            it.pop("results", None)
            os.remove(p)
            write("%s/%s.%s.json" % (Q, it["wave"], it["cand"]), it)
            continue

        it["leg"] = it.get("leg", 0) + 1
        if it["leg"] < len(legs):
            start_leg(it, free, nlive)   # may be a no-op if nothing is free; the next pass retries
            write(p, it)
            continue

        it["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        it["status"] = "done" if all(v["status"] == "done" for v in it["results"].values()) else "failed"
        it["run_dirs"] = [d for v in it["results"].values() for d in v["run_dirs"]]
        it["trials"] = sum(v["trials"] for v in it["results"].values())
        os.remove(p)
        write("%s/%s.%s.json" % (D, it["wave"], it["cand"]), it)
        with open(E + "/runs.jsonl", "a") as f:
            f.write(json.dumps(it) + "\n")
        log("candidate %s.%s %s: %d trials over legs %s"
            % (it["wave"], it["cand"], it["status"], it["trials"],
               ", ".join("%s(%d)" % (k, v["trials"]) for k, v in it["results"].items())))


def busy_endpoints():
    used = set()
    for p in items(R):
        for s in read(p).get("shards", []):
            used.add(s["url"])
    return used


def shards_for(nlive):
    """How many servers one candidate may take. AUTO (--shards 0) splits the live endpoints between the in-flight
    candidates, so a 1-node pilot runs 1 shard and an 8-node job runs 4 -- nothing assumes a node count."""
    if a.shards > 0:
        return a.shards
    return max(1, nlive // max(1, a.max_inflight))


def start_leg(it, free, nlive):
    """Launch the current leg's shards over `free` endpoints (mutated). True if it started."""
    legs = legs_of(it)
    i = it.get("leg", 0)
    if i >= len(legs) or not free:
        return False
    leg, tasks = legs[i]["leg"], legs[i]["tasks"]
    if not tasks:
        return False
    n = min(shards_for(nlive), len(free), len(tasks))
    shards = []
    for j in range(n):
        part = tasks[j::n]
        name = "%s_%s_%s_s%d" % (it["wave"], it["cand"], leg, j)
        if it.get("attempt", 1) > 1:
            name += "_r%d" % it["attempt"]
        lst = "%s/runs/%s.tasks" % (E, name)
        if not a.dry_run:
            with open(lst, "w") as f:
                f.write("\n".join(part) + "\n")
        shards.append({"name": name, "url": free[j], "tasks": len(part), "list": lst, "leg": leg})
    cmdline = ["bash %s/gepa_run.sh %s %s %s %s %d %d"
               % (G, s["name"], s["url"], it["tree"], s["list"], a.conc, it.get("k", 1)) for s in shards]
    if a.dry_run:
        log("WOULD start leg %s of %s.%s as %d shards (%s of %d tasks):"
            % (leg, it["wave"], it["cand"], n, "+".join(str(s["tasks"]) for s in shards), len(tasks)))
        for c in cmdline:
            log("   " + c)
        return False
    for c, s in zip(cmdline, shards):
        r = sh(c)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            log("shard %s failed to launch: %s" % (s["name"], (r.stderr or r.stdout).strip()[:300]))
            return False
    it["shards"] = shards
    it.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
    it["attempt"] = it.get("attempt", 1)
    for s in shards:
        free.remove(s["url"])
    log("started leg %s of %s.%s: %d shards x conc %d over %s"
        % (leg, it["wave"], it["cand"], n, a.conc, ", ".join(s["url"].split("//")[-1].split(".")[0] for s in shards)))
    return True


def admit(free, nlive):
    while len(items(R)) < a.max_inflight and items(Q) and free:
        th = threads_used()
        if th > a.pid_guard:
            log("HOLD: %d threads on the login node is above the guard %d (cap 4096); not admitting"
                % (th, a.pid_guard))
            return
        n_org = a.fake_org_count if a.fake_org_count is not None else org_sandboxes()
        if n_org is None:
            log("org sandbox count unavailable; admitting anyway")
        else:
            log("org sandboxes started: %d (hold above %d)" % (n_org, a.org_hold))
            if n_org > a.org_hold:
                log("HOLD: the shared Daytona org already has %d sandboxes started, above %d; not admitting"
                    % (n_org, a.org_hold))
                return
        p = items(Q)[0]
        it = read(p)
        if not any(l["tasks"] for l in legs_of(it)):
            log("candidate %s.%s has no tasks; dropping" % (it["wave"], it["cand"]))
            os.remove(p)
            continue
        if not start_leg(it, free, nlive):
            return
        os.remove(p)
        write("%s/%s.%s.json" % (R, it["wave"], it["cand"]), it)


# The real invariant is one shard per endpoint, so concurrent sandboxes can never exceed live_endpoints x conc --
# tighter and more honest than max_inflight x shards x conc, which over-counts whenever fewer servers are up.
def check_cap(nlive):
    peak = nlive * a.conc
    if peak > a.sandbox_cap:
        sys.exit("%d live servers x conc %d = up to %d concurrent Daytona sandboxes, above the cap %d -- lower CONC "
                 "or serve fewer nodes" % (nlive, a.conc, peak, a.sandbox_cap))
    return peak


log("runner up: max_inflight %d, shards/candidate %s, conc %d; peak sandboxes = live servers x %d, cap %d; pid guard %d"
    % (a.max_inflight, ("auto (live servers / max_inflight)" if a.shards <= 0 else a.shards),
       a.conc, a.conc, a.sandbox_cap, a.pid_guard))
while True:
    try:
        job = serve_job()
        if not job:
            log("no RUNNING gepa_serve job; waiting (queued: %d)" % len(items(Q)))
        else:
            live = endpoints(job)
            check_cap(len(live))
            free = [u for u in live if u not in busy_endpoints()]
            reap(free, len(live))
            admit(free, len(live))
    except Exception as e:  # a controller that dies leaves the servers idle; never let one bad pass kill it
        log("pass failed: %s: %s" % (type(e).__name__, e))
    if a.once:
        break
    time.sleep(a.poll)
