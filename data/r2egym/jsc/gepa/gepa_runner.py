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
  * at most MAX_INFLIGHT candidates at once (Daytona: ~1,000 sandboxes per user, 5 creates/s -- the ceiling here is
    MAX_INFLIGHT x SHARDS x CONC concurrent trials, printed at startup and asserted against SANDBOX_CAP);
  * at most one shard per live endpoint, so no two harbor runs share a server's seats;
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
ap = argparse.ArgumentParser()
ap.add_argument("--max-inflight", type=int, default=int(os.environ.get("MAX_INFLIGHT", "2")))
ap.add_argument("--shards", type=int, default=int(os.environ.get("SHARDS_PER_CAND", "4")),
                help="harbor runs per candidate; each takes one server")
ap.add_argument("--conc", type=int, default=int(os.environ.get("CONC", "32")),
                help="concurrent trials per shard; must not exceed the server's --max-num-seqs")
ap.add_argument("--sandbox-cap", type=int, default=int(os.environ.get("SANDBOX_CAP", "900")),
                help="refuse a layout whose peak concurrent sandboxes would exceed this")
ap.add_argument("--pid-guard", type=int, default=int(os.environ.get("PID_GUARD", "2800")),
                help="do not admit while this user's thread count on the login node is above this (cap is 4096)")
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


def reap(free):
    """Finish whatever shards are done. A candidate whose leg completes starts its NEXT leg immediately, reusing the
    endpoints that leg just freed, so the servers never wait for the whole candidate."""
    for p in items(R):
        it = read(p)
        if not it.get("shards"):  # mid-candidate, waiting for endpoints to start the next leg
            if free and start_leg(it, free):
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
            start_leg(it, free)       # may be a no-op if nothing is free; the next pass retries
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


def start_leg(it, free):
    """Launch the current leg's shards over `free` endpoints (mutated). True if it started."""
    legs = legs_of(it)
    i = it.get("leg", 0)
    if i >= len(legs) or not free:
        return False
    leg, tasks = legs[i]["leg"], legs[i]["tasks"]
    if not tasks:
        return False
    n = min(a.shards, len(free), len(tasks))
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


def admit(free):
    while len(items(R)) < a.max_inflight and items(Q) and free:
        th = threads_used()
        if th > a.pid_guard:
            log("HOLD: %d threads on the login node is above the guard %d (cap 4096); not admitting"
                % (th, a.pid_guard))
            return
        p = items(Q)[0]
        it = read(p)
        if not any(l["tasks"] for l in legs_of(it)):
            log("candidate %s.%s has no tasks; dropping" % (it["wave"], it["cand"]))
            os.remove(p)
            continue
        if not start_leg(it, free):
            return
        os.remove(p)
        write("%s/%s.%s.json" % (R, it["wave"], it["cand"]), it)


peak = a.max_inflight * a.shards * a.conc
if peak > a.sandbox_cap:
    sys.exit("layout would hold up to %d concurrent Daytona sandboxes (%d inflight x %d shards x conc %d), above the "
             "cap %d -- lower MAX_INFLIGHT, SHARDS_PER_CAND or CONC" % (peak, a.max_inflight, a.shards, a.conc, a.sandbox_cap))
log("runner up: max_inflight %d, %d shards/candidate, conc %d -> peak %d concurrent sandboxes; pid guard %d"
    % (a.max_inflight, a.shards, a.conc, peak, a.pid_guard))
while True:
    try:
        job = serve_job()
        if not job:
            log("no RUNNING gepa_serve job; waiting (queued: %d)" % len(items(Q)))
        else:
            live = endpoints(job)
            free = [u for u in live if u not in busy_endpoints()]
            reap(free)
            admit(free)
    except Exception as e:  # a controller that dies leaves the servers idle; never let one bad pass kill it
        log("pass failed: %s: %s" % (type(e).__name__, e))
    if a.once:
        break
    time.sleep(a.poll)
