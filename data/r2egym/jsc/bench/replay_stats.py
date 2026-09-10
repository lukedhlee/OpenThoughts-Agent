import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
pl = [p for r in rows for p in r["prompt_len"]]; cl = [c for r in rows for c in r["completion_len"]]; api = [a / 1000 for r in rows for a in r["api_ms"]]
q = lambda x, f: sorted(x)[int(f * (len(x) - 1))]
print(f"{sys.argv[1]}: traj={len(rows)} turns={len(pl)} turns/traj p50={statistics.median([r['n_turns'] for r in rows])} "
      f"prompt p50={q(pl,.5)} p90={q(pl,.9)} max={max(pl)} completion p50={q(cl,.5)} p90={q(cl,.9)} api_s p50={q(api,.5):.1f} p90={q(api,.9):.1f} "
      f"implied tok/s p50={q([c/a for c,a in zip(cl,api) if a>0],.5):.1f}")
