#!/usr/bin/env python3
"""Runboard data puller: spec JSON -> experiment JSON.

Reads a spec from specs/, pulls every series' WandB run histories (all attempts in a
group, ordered by creation), stitches them latest-attempt-wins per step, and writes
data/<id>.json — the file the dashboard renders. Presentation fields in the spec are
passed through untouched; only series_data / fleet step info / snapshot are generated.

Usage:
  python pull_wandb.py specs/base30b_gsm8k_arms.json [-o data/]
"""
import argparse
import copy
import datetime
import json
import pathlib
import zoneinfo

RL_DEFAULT_METRICS = {
    "reward": "reward/avg_raw_reward",
    "pass8": "reward/avg_pass_at_8",
    "entropy": "policy/policy_entropy",
    "kl": "policy/policy_kl",
    "gradnorm": "policy/raw_grad_norm",
    "clip": "policy/ppo_clip_ratio",
    "mixed": "diag/frac_groups_mixed",
    "allwrong": "diag/frac_groups_all_wrong",
    "allcorrect": "diag/frac_groups_all_correct",
    "len50": "diag/response_len_p50",
    "len90": "diag/response_len_p90",
    "trunc": "diag/truncated_frac",
    "tis": "policy/tis/imp_ratio_mean",
}

PASSTHROUGH = [
    "id", "title", "eyebrow", "subtitle", "status_line", "tiles", "explainer",
    "step_target", "charts", "glossary", "evals", "eval_key_label", "attempts",
    "timeline", "notes",
]


def stitch(runs, metrics):
    """Latest-attempt-wins per step across a series' (creation-ordered) runs."""
    by_step = {}
    for run in runs:
        for row in run.scan_history(page_size=500):
            step = row.get("_step")
            if step is None:
                continue
            rec = {}
            for short, key in metrics.items():
                v = row.get(key)
                if v is not None:
                    rec[short] = round(v, 4) if isinstance(v, float) else v
            if rec:
                by_step[step] = rec
    return [{"step": s, **by_step[s]} for s in sorted(by_step)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=pathlib.Path)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "data")
    args = ap.parse_args()

    import wandb  # deferred: keeps --help fast and import errors obvious

    spec = json.loads(args.spec.read_text())
    metrics = spec.get("metrics", "rl_default")
    if metrics == "rl_default":
        metrics = RL_DEFAULT_METRICS

    api = wandb.Api(timeout=120)
    exp = {k: copy.deepcopy(spec[k]) for k in PASSTHROUGH if k in spec}
    exp["series_meta"] = [
        {k: s[k] for k in ("key", "name", "label", "color_slot") if k in s}
        for s in spec["series"]
    ]
    exp["series_data"] = {}

    for s in spec["series"]:
        runs = sorted(
            api.runs(spec["wandb_project"], filters={"group": s["group"]}, per_page=200),
            key=lambda r: r.created_at,
        )
        exp["series_data"][s["key"]] = stitch(runs, metrics)
        pts = exp["series_data"][s["key"]]
        print(f"{s['key']}: {len(runs)} runs -> {len(pts)} steps"
              + (f" (1..{pts[-1]['step']})" if pts else ""))

    tz = zoneinfo.ZoneInfo(spec.get("timezone", "Europe/Berlin"))
    now = datetime.datetime.now(tz)
    exp["snapshot"] = now.strftime("%Y-%m-%d %H:%M ") + spec.get("tz_label", "CEST")

    # timeline "now" marker for still-running attempt bars
    if "timeline" in exp and exp["timeline"].get("auto_now", True):
        exp["timeline"]["now"] = now.strftime("%H:%M")

    # fleet: presentation fields from the spec, live step/metric from the pull
    primary = "reward"
    charts = spec.get("charts", "rl_default")
    if isinstance(charts, list) and charts:
        primary = charts[0].get("field", primary)
    fleet = []
    for f in spec.get("fleet", []):
        f = dict(f)
        pts = [r for r in exp["series_data"].get(f["key"], []) if primary in r]
        if pts:
            f["step"] = pts[-1]["step"]
            f["metric_label"] = f"{primary} {pts[-1][primary]:.2f}"
        fleet.append(f)
    if fleet:
        exp["fleet"] = fleet

    # "__DEATHS__" placeholder in tiles -> number of non-running attempts
    if "attempts" in exp:
        deaths = sum(1 for ats in exp["attempts"].values() for a in ats
                     if a.get("fate") not in ("run", "done"))
        for t in exp.get("tiles", []):
            if t.get("value") == "__DEATHS__":
                t["value"] = str(deaths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{exp['id']}.json"
    out.write_text(json.dumps(exp))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB), snapshot {exp['snapshot']}")


if __name__ == "__main__":
    main()
