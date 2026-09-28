"""Does a timed-out command wedge a Daytona process session? (mini-swe-agent-host on TB2, 2026-09-28)

Harbor's Daytona environment runs every exec in ONE persistent process session per sandbox
(`_ensure_exec_session`), as `timeout <s> env ... bash -c '<cmd>'` with run_async=True. In the msa4 smoke
two trials (kv-store-grpc, count-dataset-tokens) had every exec after one particular timed-out command
hang for good, and then the verifier timed out. This reproduces the pattern on a scratch sandbox: each case
runs in its own fresh session, first the probe command (5 s timeout), then `echo after` with a 40 s limit.

    set -a; source <daytona key file>; set +a
    python daytona_session_wedge.py <snapshot name>
"""
import asyncio
import sys
import time

from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams, SessionExecuteRequest

WRAP = "timeout 5 env PAGER=cat bash -c {q}"   # harbor's _sandbox_exec shape, 5 s instead of 30 s


def q(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


CASES = {
    "plain_sleep": "exec 2>&1\nbash -lc 'sleep 20'",
    "bg_holds_stdout": "exec 2>&1\nbash -lc 'tail -f /dev/null & sleep 20'",
    "bg_redirected": "exec 2>&1\nbash -lc 'tail -f /dev/null >/dev/null 2>&1 & sleep 20'",
    "fg_ignores_term": "exec 2>&1\nbash -lc 'python3 -c \"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)\"'",
    "bg_setsid_holds_stdout": "exec 2>&1\nbash -lc 'setsid tail -f /dev/null & sleep 20'",
    "reads_stdin": "exec 2>&1\nbash -lc 'cat'",
    "fg_output_then_timeout": "exec 2>&1\nbash -lc 'echo partial; sleep 20'",
    "subshell_bg_nohup": "exec 2>&1\nbash -lc '(nohup sleep 100 &); sleep 20'",
}
# candidate fix: the command's output goes to a file in the sandbox (children inherit the file, not the
# session's pipe), KILL 3 s after TERM, stdin from /dev/null; the file is printed after.
FIX_INNER = ("exec 2>&1\n__f=$(mktemp 2>/dev/null || echo /tmp/.msa_out.$$)\n"
             "timeout -k 3 5 {cmd} >\"$__f\" 2>&1 </dev/null\n__rc=$?\ncat \"$__f\"; rm -f \"$__f\"\nexit $__rc")
FIX = "timeout 20 env PAGER=cat bash -c {q}"   # harbor's wrapper around the fixed script (outer timeout = t + 10)


async def run(sb, sid, cmd, limit):
    t = time.time()
    r = await sb.process.execute_session_command(sid, SessionExecuteRequest(command=cmd, run_async=True))
    while time.time() - t < limit:
        c = await sb.process.get_session_command(sid, r.cmd_id)
        if c.exit_code is not None:
            logs = await sb.process.get_session_command_logs(sid, r.cmd_id)
            return round(time.time() - t, 1), c.exit_code, (logs.stdout or "")[-80:].replace("\n", " | ")
        await asyncio.sleep(0.5)
    return round(time.time() - t, 1), None, "STUCK"


async def main(snapshot):
    async with AsyncDaytona() as d:
        sb = await d.create(params=CreateSandboxFromSnapshotParams(snapshot=snapshot, auto_delete_interval=0,
                                                            auto_stop_interval=15, labels={"purpose": "msa4-wedge-test"}),
                            timeout=300)
        print("sandbox", sb.id, flush=True)
        try:
            for fix in (False, True):
                for name, inner in CASES.items():
                    sid = f"{'fix' if fix else 'raw'}-{name}".replace("_", "-")
                    await sb.process.create_session(sid)
                    if fix:   # the adapter's own script (harbor lukedhlee/mini-swe-host-exec-isolation), harbor's wrapper
                        from harbor.agents.mini_swe_agent_host.adapters import isolated_command_script
                        cmd = inner.split("\n", 1)[1].split("bash -lc ", 1)[1]
                        cmd = cmd[1:-1].replace("'\"'\"'", "'") if cmd.startswith("'") else cmd
                        probe = FIX.format(q=q(isolated_command_script(["bash", "-lc"], cmd, 5)))
                    else:
                        probe = WRAP.format(q=q(inner))
                    a = await run(sb, sid, probe, 40)
                    b = await run(sb, sid, "echo after", 40)
                    print(f"{'FIX' if fix else 'RAW'} {name:24s} probe {a}  then echo {b}", flush=True)
                    await run(sb, f"{sid}", "true", 1) if False else None
                    # clean the sandbox between cases so leftovers do not leak into the next case
                    await sb.process.create_session(sid + "-clean")
                    await run(sb, sid + "-clean", "pkill -9 -f 'tail -f' ; pkill -9 -f 'time.sleep' ; pkill -9 -f 'sleep 100'; true", 20)
        finally:
            await sb.delete()
            print("deleted", sb.id, flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
