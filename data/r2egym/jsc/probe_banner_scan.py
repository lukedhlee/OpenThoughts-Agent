#!/usr/bin/env python3
"""probe_banner_scan.py <trace_archive.tar> [n_traj] — stream a probe trace tar, parse the first n trajectory.json files, count the
harness banner inside agent-authored text vs environment/observation text. Harbor-raw, no trainer involvement."""
import sys, tarfile, json, time
B = "[... output limited to"; path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
t0 = time.time(); n = 0; agent_b = 0; other_b = 0; traj_with_agent_b = 0; agent_msgs = 0; loops = 0; bytes_read = 0
def texts(o, role=None):
    """yield (is_agent, text) for message-like dicts."""
    if isinstance(o, dict):
        r = o.get("source") or o.get("role")
        if r in ("agent", "assistant", "user", "environment", "system", "tool"):
            for k in ("message", "content", "text", "reasoning_content", "raw_response"):
                v = o.get(k)
                if isinstance(v, str): yield (r in ("agent", "assistant"), v)
        for v in o.values(): yield from texts(v)
    elif isinstance(o, list):
        for v in o: yield from texts(v)
with tarfile.open(path, "r|") as tf:
    for m in tf:
        bytes_read += m.size
        if not m.isfile() or not m.name.endswith("trajectory.json"): continue
        try: d = json.load(tf.extractfile(m))
        except Exception: continue
        n += 1; ab = 0
        for is_agent, s in texts(d):
            c = s.count(B)
            if is_agent: agent_b += c; ab += c; agent_msgs += 1; loops += int(c >= 5)
            else: other_b += c
        traj_with_agent_b += int(ab > 0)
        if n >= N: break
print(f"{path.split('/')[-3]}: trajectories={n} agent_msgs={agent_msgs} banner_in_agent_text={agent_b} (trajectories {traj_with_agent_b}, msgs with >=5: {loops}) banner_in_observations={other_b} read={bytes_read/1e9:.1f} GB in {time.time()-t0:.0f}s")
