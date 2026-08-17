# Runboard — local experiment dashboard

A self-contained, offline, interactive dashboard for our training campaigns: per-run
trend charts with crosshair tooltips, plain-language metric glossary, fleet status,
and a Slurm-attempt timeline. No server, no CDN, no build toolchain — one HTML file.

## Use it

```bash
# 1. pull fresh data for an experiment (needs a wandb-authed python)
python scripts/dashboard/pull_wandb.py scripts/dashboard/specs/base30b_gsm8k_arms.json

# 2. bundle every data/*.json into the site
python scripts/dashboard/build.py

# 3. open it
open scripts/dashboard/dist/dashboard.html
```

The page also loads experiment JSONs at runtime — **drag any `data/*.json` onto the
page** (or use the *Load JSON…* button). Loaded experiments persist in the browser's
localStorage, so you can hand someone `dist/dashboard.html` once and ship them new
experiment files after.

## Add a new experiment

1. Copy `specs/base30b_gsm8k_arms.json`, change `id`, `title`, `wandb_project`, and
   the `series` list (`group` = the WandB group of that run's attempts; `color_slot`
   = 1–8, fixed order, color follows the run everywhere).
2. Everything else is optional and degrades gracefully — omit `attempts`/`timeline`
   and the job-war section disappears; omit `tiles`, `fleet`, `explainer` likewise.
   `metrics`/`charts`/`glossary` accept `"rl_default"` (GRPO metric map baked into
   the template and puller) or explicit definitions.
3. `pull_wandb.py <spec>` then `build.py`. Done.

## Spec → page mapping

| Spec field | Section it drives |
|---|---|
| `title` `eyebrow` `subtitle` `status_line` | header |
| `tiles` (`hero: true` for the big one, `"__DEATHS__"` auto-fills) | verdict strip |
| `explainer` (`**bold**` supported) | "What is this experiment?" |
| `fleet` + `step_target` (step/metric auto-filled at pull time) | per-run status cards |
| `charts` (`"rl_default"` or `[{field,title,how,big,dp,y0,y1,yTicks}]`) | charts; omitted domains auto-scale |
| `evals` (`[{step,v,label,series}]`) | ◆ markers on the big chart |
| `glossary` (`"rl_default"` or `[{term,wkey,text}]`) | metric explanations |
| `attempts` + `timeline` (fates: `hang` `oom` `node` `fail` `done` `run`) | job gantt + table |
| `notes` | footer |

Tile deltas are green by default; set `delta_tone` to `flat` (grey) or `bad` (red) when the
change isn't good news.

**Campaigns spanning midnight.** Timeline times are minutes-from-`t0`, so express day 2 as
24+h (`09:10` on the second day is `"33:10"`). Clock labels wrap mod 24 automatically. Add
`timeline.t0_date: "YYYY-MM-DD"` so the puller places the *now* marker in the same offset
space, and `timeline.days: [{at, label}]` to draw a dashed divider where each day starts.
See `specs/currease30b_grpo_arms.json`.

`series_data` is never hand-written — `pull_wandb.py` generates it by stitching all
of a group's runs latest-attempt-wins per `_step` (so checkpoint-resume replays don't
double-plot).

## Files

- `template.html` — the renderer (palette, charts, tooltips, tables); data-agnostic.
- `pull_wandb.py` — spec → `data/<id>.json` (WandB API; run on a wandb-authed machine).
- `build.py` — inlines `data/*.json` into `dist/dashboard.html`.
- `specs/` — one JSON per experiment (hand-maintained).
- `data/` — pulled experiment JSONs (committed so the site rebuilds anywhere).
- `dist/dashboard.html` — the site. Open directly or publish as an Artifact.
- `refresh_currease.sh` — one-shot refresh for the curriculum-easy sweep (below).

## Offline-logging campaigns

The SkyRL trainer on Jupiter logs wandb **offline** (`WANDB_MODE=offline`), one run per
Slurm attempt under each experiment dir — so the cloud project holds nothing until someone
syncs it. `refresh_currease.sh` does the whole loop: rsync the offline runs off the
cluster, `wandb sync` the ones that grew (a size manifest in `~/.cache/runboard/` skips
finished attempts), then pull + build.

```bash
scripts/dashboard/refresh_currease.sh [--open]   # ~40s incremental, ~3min cold
```

Two things matter if you adapt this for another campaign. Sync **in attempt order** —
`wandb sync` stamps `created_at` at sync time, and that ordering is exactly what
`pull_wandb.py` stitches latest-attempt-wins by. And re-syncing an offline run re-uploads
it under the same run id, so live attempts just gain their new steps rather than
duplicating.
