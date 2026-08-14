#!/usr/bin/env python3
"""Diagnostics sidecar: watch a SkyRL GSM8K GRPO experiment dir, log derived metrics to W&B.

The trainer's own wandb run reports pass rates but not WHY they move. This
sidecar polls the on-disk dumps a run leaves behind and derives the
format-vs-math diagnostics (strict/flexible parse rates, truncation, length
buckets, per-group phat spectrum) into a companion wandb run named
``<run_name>-diag``. It is read-only with respect to the experiment dir except
for its own two files (state + summary), idempotent across restarts (a state
file records processed steps; the wandb run id is deterministic so restarts
resume the same run), and runs happily on a GPU-less login node.

Inputs polled (either may not exist yet; both are tolerated mid-write):
  <experiment_dir>/dumped_evals/global_step_{N}_evals/*.jsonl
      records: {"input_prompt", "output_response", "score", "stop_reason",
                "env_class", "env_extras", "data_source"} where
      env_extras carries the gsm8k parquet extra_info
      {"split","index","answer","question"} and "answer" is the raw GSM8K
      solution ending in "#### <number>".
  <experiment_dir>/diag_rollouts/step_{N}.jsonl
      records: {"step","uid","prompt","response","reward","stop_reason",
                "response_len","tool_calls"[, "oracle"]}.

Outputs:
  - wandb run f"{run_name}-diag" (job_type="diag", resume="allow"), metrics
    keyed on the custom step metric "global_step" so panels align with the
    trainer's steps. Prefixes: evaldiag/ (eval dumps), traindiag/ (per rollout
    step), traindiag_win/ (pooled over the last --window-steps rollout steps).
  - <experiment_dir>/diag_summary.json: latest snapshot of all metrics + last
    processed steps, written atomically.

Usage:
  python scripts/analysis/gsm8k_diag_sidecar.py \\
      --experiment-dir /path/to/exp --run-name base30b_gsm8k_armA
  # single scan (cron-style):
  python scripts/analysis/gsm8k_diag_sidecar.py \\
      --experiment-dir /path/to/exp --run-name base30b_gsm8k_armA --once

Dependencies: stdlib + wandb only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: this sidecar needs wandb ({exc}). stdlib+wandb is the whole budget.", file=sys.stderr)
    raise SystemExit(2) from exc

# skyrl_gym.envs.gsm8k.utils uses exactly this pattern for method="strict"; we
# mirror it byte-for-byte so evaldiag/strict_parse_rate predicts the trainer's
# scorer instead of approximating it.
STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
# ... and this one for method="flexible" (any number; we take the LAST valid one).
FLEX_RE = re.compile(r"(\-?[0-9\.\,]+)")

EVAL_DIR_RE = re.compile(r"^global_step_(\d+)_evals$")
ROLLOUT_FILE_RE = re.compile(r"^step_(\d+)\.jsonl$")

# [lo, hi) word-count buckets; hi=None means unbounded ("inf" in metric names).
LEN_BUCKETS: tuple[tuple[int, int | None], ...] = (
    (0, 128), (128, 256), (256, 512), (512, 768), (768, 1024), (1024, None),
)
TRUNCATION_STOP_REASONS = {"length", "max_tokens"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dir", required=True, help="SkyRL experiment dir (dumped_evals/, diag_rollouts/)")
    p.add_argument("--run-name", required=True, help="Base run name; the wandb run becomes <run-name>-diag")
    p.add_argument("--wandb-project", default="jupiter-base30b-gsm8k-grpo")
    p.add_argument("--wandb-group", default="base30b_gsm8k_arms_r1")
    p.add_argument("--poll-interval", type=float, default=60, help="Seconds between scans")
    p.add_argument("--once", action="store_true", help="Single scan then exit")
    p.add_argument(
        "--state-file",
        default=None,
        help="Processed-step ledger (default: <experiment_dir>/.diag_sidecar_state.json)",
    )
    p.add_argument("--group-size", type=int, default=8, help="GRPO group size G (full-group phat spectrum)")
    p.add_argument("--window-steps", type=int, default=10, help="Rollout steps pooled into traindiag_win/*")
    return p.parse_args()


# ----------------------------------------------------------------------------
# number extraction / normalization
# ----------------------------------------------------------------------------

def normalize_number(raw: str | None) -> str | None:
    """Canonicalize a matched number string: drop commas/$, trailing dots; '1,234.' -> '1234'."""
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace("$", "").rstrip(".")
    if not s or s == "-":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if math.isfinite(f) and f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def extract_strict(text: str) -> str | None:
    m = STRICT_RE.search(text)
    return normalize_number(m.group(1)) if m else None


def extract_last_number(text: str) -> str | None:
    """LAST valid number anywhere in the text (skyrl's flexible extraction)."""
    for cand in reversed(FLEX_RE.findall(text)):
        norm = normalize_number(cand)
        if norm is not None:
            return norm
    return None


def extract_ground_truth(answer_field) -> str | None:
    """GT from env_extras['answer'] (raw GSM8K solution '...#### N') or a bare number/oracle."""
    if answer_field is None:
        return None
    text = str(answer_field)
    return extract_strict(text) or normalize_number(text)


def is_truncated(stop_reason) -> bool:
    if stop_reason is None:
        return False
    return str(stop_reason).strip().lower() in TRUNCATION_STOP_REASONS


# ----------------------------------------------------------------------------
# metric computation
# ----------------------------------------------------------------------------

def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac


def bucket_index(words: int) -> int:
    for i, (lo, hi) in enumerate(LEN_BUCKETS):
        if words >= lo and (hi is None or words < hi):
            return i
    return len(LEN_BUCKETS) - 1  # unreachable; last bucket is unbounded


def compute_suite(samples: list[dict]) -> dict:
    """Parse/EM/length/truncation metrics over normalized samples.

    Each sample: {"response": str, "correct": bool, "stop_reason": any, "gt": str|None}.
    Returns unprefixed keys; the caller prepends evaldiag/ or traindiag*/.
    """
    n = len(samples)
    out: dict = {"num_samples": n}
    if n == 0:
        return out

    answer_present = strict_parse = numeric = malformed = trunc = n_correct = 0
    em_strict = em_flex = gt_avail = 0
    words_all: list[int] = []
    words_ok: list[int] = []
    words_bad: list[int] = []
    bucket_n = [0] * len(LEN_BUCKETS)
    bucket_ok = [0] * len(LEN_BUCKETS)

    for s in samples:
        resp: str = s["response"]
        correct: bool = s["correct"]
        w = len(resp.split())
        words_all.append(w)
        (words_ok if correct else words_bad).append(w)
        n_correct += correct
        bi = bucket_index(w)
        bucket_n[bi] += 1
        bucket_ok[bi] += correct

        if "#### " in resp:
            answer_present += 1
        strict_num = extract_strict(resp)
        flex_num = extract_last_number(resp)
        strict_parse += strict_num is not None
        numeric += flex_num is not None
        malformed += (not resp.strip()) or flex_num is None
        trunc += is_truncated(s["stop_reason"])

        gt = s["gt"]
        if gt is not None:
            gt_avail += 1
            em_strict += strict_num == gt
            em_flex += flex_num == gt

    out.update(
        accuracy=n_correct / n,
        answer_present_rate=answer_present / n,
        strict_parse_rate=strict_parse / n,
        numeric_parse_rate=numeric / n,
        malformed_or_empty_rate=malformed / n,
        truncation_rate=trunc / n,
        num_gt_available=gt_avail,
    )
    # EM over samples whose GT we could extract (missing env_extras/oracle
    # shrink the denominator rather than silently counting as wrong).
    if gt_avail:
        out["exact_match_strict"] = em_strict / gt_avail
        out["exact_match_flexible"] = em_flex / gt_avail

    for name, vals, full in (("", words_all, True), ("correct_", words_ok, False), ("incorrect_", words_bad, False)):
        if not vals:
            continue
        vals.sort()
        out[f"len_words_{name}mean"] = sum(vals) / len(vals)
        out[f"len_words_{name}p50"] = percentile(vals, 0.50)
        out[f"len_words_{name}p90"] = percentile(vals, 0.90)
        if full:
            out[f"len_words_{name}p99"] = percentile(vals, 0.99)
            out[f"len_words_{name}max"] = float(vals[-1])

    for (lo, hi), cnt, ok in zip(LEN_BUCKETS, bucket_n, bucket_ok):
        tag = f"{lo}_{hi if hi is not None else 'inf'}"
        out[f"count_len_{tag}"] = cnt
        if cnt:
            out[f"acc_len_{tag}"] = ok / cnt
    return out


def compute_group_metrics(records: list[dict], group_size: int) -> dict:
    """GRPO group stats keyed by uid. Records: {"uid", "reward"}. Unprefixed keys."""
    groups: dict = defaultdict(list)
    for r in records:
        if r.get("uid") is not None:
            groups[r["uid"]].append(r["reward"])
    out: dict = {"num_groups": len(groups)}
    if not groups:
        return out

    all_wrong = mixed = all_correct = 0
    stds: list[float] = []
    full_group_k: list[int] = []
    for rewards in groups.values():
        k = sum(1 for x in rewards if x > 0)
        if k == 0:
            all_wrong += 1
        elif k == len(rewards):
            all_correct += 1
        else:
            mixed += 1
        stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
        if len(rewards) == group_size:
            full_group_k.append(k)

    n_groups = len(groups)
    out.update(
        frac_groups_all_wrong=all_wrong / n_groups,
        frac_groups_mixed=mixed / n_groups,
        frac_groups_all_correct=all_correct / n_groups,
        group_reward_std_mean=sum(stds) / len(stds),
        num_full_groups=len(full_group_k),
    )
    # phat spectrum over FULL groups only: with G rollouts/group, what fraction
    # of groups landed exactly k correct? (k=0 and k=G contribute no gradient.)
    if full_group_k:
        for k in range(group_size + 1):
            out[f"phat_frac_{k}_of_{group_size}"] = sum(1 for x in full_group_k if x == k) / len(full_group_k)
    return out


# ----------------------------------------------------------------------------
# input parsing / discovery
# ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """Parse a jsonl file; bad lines are skipped and counted, never fatal."""
    records: list[dict] = []
    bad = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    bad += 1
    except OSError as exc:
        print(f"[diag] WARN: cannot read {path}: {exc}", flush=True)
    return records, bad


def coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eval_record_to_sample(rec: dict) -> dict:
    extras = rec.get("env_extras")
    gt = extract_ground_truth(extras.get("answer")) if isinstance(extras, dict) else None
    return {
        "response": str(rec.get("output_response") or ""),
        "correct": coerce_float(rec.get("score")) > 0,
        "stop_reason": rec.get("stop_reason"),
        "gt": gt,
    }


def rollout_record_to_sample(rec: dict) -> dict:
    reward = coerce_float(rec.get("reward"))
    return {
        "response": str(rec.get("response") or ""),
        "correct": reward > 0,
        "stop_reason": rec.get("stop_reason"),
        "gt": extract_ground_truth(rec.get("oracle")) if rec.get("oracle") is not None else None,
        "uid": rec.get("uid"),
        "reward": reward,
    }


def discover_eval_steps(experiment_dir: Path) -> dict[int, Path]:
    """Ready eval-dump dirs: step -> dir. Ready = holds >=1 non-empty .jsonl (mid-write guard)."""
    root = experiment_dir / "dumped_evals"
    found: dict[int, Path] = {}
    if not root.is_dir():
        return found
    for child in root.iterdir():
        m = EVAL_DIR_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        try:
            ready = any(f.stat().st_size > 0 for f in child.glob("*.jsonl"))
        except OSError:
            ready = False
        if ready:
            found[int(m.group(1))] = child
    return found


def discover_rollout_steps(experiment_dir: Path) -> dict[int, Path]:
    """Ready rollout dumps: step -> file. Ready = non-empty file."""
    root = experiment_dir / "diag_rollouts"
    found: dict[int, Path] = {}
    if not root.is_dir():
        return found
    for child in root.iterdir():
        m = ROLLOUT_FILE_RE.match(child.name)
        if not m:
            continue
        try:
            if child.stat().st_size > 0:
                found[int(m.group(1))] = child
        except OSError:
            continue
    return found


def load_rollout_records(path: Path) -> tuple[list[dict], int]:
    records, bad = read_jsonl(path)
    return [rollout_record_to_sample(r) for r in records], bad


# ----------------------------------------------------------------------------
# state / summary persistence
# ----------------------------------------------------------------------------

def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    fresh = {"eval_steps": [], "rollout_steps": [], "bad_lines_total": 0}
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        return {
            "eval_steps": sorted(int(s) for s in state.get("eval_steps", [])),
            "rollout_steps": sorted(int(s) for s in state.get("rollout_steps", [])),
            "bad_lines_total": int(state.get("bad_lines_total", 0)),
        }
    except FileNotFoundError:
        return fresh
    except (OSError, ValueError, TypeError) as exc:
        print(f"[diag] WARN: state file {path} unreadable ({exc}); starting fresh", flush=True)
        return fresh


# ----------------------------------------------------------------------------
# wandb
# ----------------------------------------------------------------------------

def derive_run_id(run_name: str) -> str:
    """Deterministic wandb id from run_name so restarts resume the same run."""
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{run_name}-diag")
    if len(base) <= 64:
        return base
    return base[:55] + "-" + hashlib.md5(base.encode()).hexdigest()[:8]


def try_wandb_init(args: argparse.Namespace, experiment_dir: Path):
    """Init the diag run; on failure return None so the poll loop retries later."""
    try:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"{args.run_name}-diag",
            id=derive_run_id(args.run_name),
            job_type="diag",
            resume="allow",
            config={
                "experiment_dir": str(experiment_dir),
                "group_size": args.group_size,
                "window_steps": args.window_steps,
                "sidecar": "gsm8k_diag_sidecar",
            },
        )
        # Align every panel on the trainer's step axis instead of wandb's
        # internal monotonic _step (eval and rollout dumps arrive interleaved).
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run
    except Exception as exc:  # wandb raises many types; never crash the sidecar
        print(f"[diag] WARN: wandb.init failed ({type(exc).__name__}: {exc}); will retry next poll", flush=True)
        return None


# ----------------------------------------------------------------------------
# scan loop
# ----------------------------------------------------------------------------

class Sidecar:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.experiment_dir = Path(args.experiment_dir).expanduser()
        self.state_path = (
            Path(args.state_file) if args.state_file else self.experiment_dir / ".diag_sidecar_state.json"
        )
        self.summary_path = self.experiment_dir / "diag_summary.json"
        self.state = load_state(self.state_path)
        self.latest_metrics: dict = {}
        self.rollout_cache: dict[int, list[dict]] = {}  # step -> parsed samples (window pooling)

    # -- persistence ---------------------------------------------------------
    def save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def save_summary(self) -> None:
        atomic_write_json(
            self.summary_path,
            {
                "run_name": self.args.run_name,
                "updated_unix": time.time(),
                "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "last_eval_step": max(self.state["eval_steps"], default=None),
                "last_rollout_step": max(self.state["rollout_steps"], default=None),
                "processed_eval_steps": self.state["eval_steps"],
                "processed_rollout_steps": self.state["rollout_steps"],
                "bad_lines_total": self.state["bad_lines_total"],
                "metrics": self.latest_metrics,
            },
        )

    # -- per-step processing -------------------------------------------------
    def process_eval_step(self, step: int, dump_dir: Path) -> None:
        samples: list[dict] = []
        bad = 0
        for f in sorted(dump_dir.glob("*.jsonl")):
            records, b = read_jsonl(f)
            bad += b
            samples.extend(eval_record_to_sample(r) for r in records)

        payload = {f"evaldiag/{k}": v for k, v in compute_suite(samples).items()}
        payload["evaldiag/skipped_bad_lines"] = bad
        payload["global_step"] = step
        wandb.log(payload)

        self.latest_metrics.update({k: v for k, v in payload.items() if k != "global_step"})
        self.state["eval_steps"] = sorted(set(self.state["eval_steps"]) | {step})
        self.state["bad_lines_total"] += bad
        self.save_state()
        self.save_summary()
        print(
            "[diag] eval step {}: n={} acc={:.3f} strict_parse={:.3f} em_strict={} trunc={:.3f} bad_lines={}".format(
                step,
                len(samples),
                payload.get("evaldiag/accuracy", 0.0),
                payload.get("evaldiag/strict_parse_rate", 0.0),
                f"{payload['evaldiag/exact_match_strict']:.3f}" if "evaldiag/exact_match_strict" in payload else "n/a",
                payload.get("evaldiag/truncation_rate", 0.0),
                bad,
            ),
            flush=True,
        )

    def _window_samples(self, upto_step: int) -> tuple[list[dict], int]:
        """Pool samples from the last --window-steps rollout steps <= upto_step."""
        steps = sorted(set(self.state["rollout_steps"]) | {upto_step})
        window = [s for s in steps if s <= upto_step][-self.args.window_steps :]
        pooled: list[dict] = []
        used = 0
        for s in window:
            samples = self.rollout_cache.get(s)
            if samples is None:
                path = self.experiment_dir / "diag_rollouts" / f"step_{s}.jsonl"
                if not path.is_file():
                    continue  # already-processed step whose dump was cleaned up
                samples, _ = load_rollout_records(path)
                self.rollout_cache[s] = samples
            pooled.extend(samples)
            used += 1
        return pooled, used

    def process_rollout_step(self, step: int, path: Path) -> None:
        samples, bad = load_rollout_records(path)
        self.rollout_cache[step] = samples

        payload = {f"traindiag/{k}": v for k, v in compute_suite(samples).items()}
        payload.update({f"traindiag/{k}": v for k, v in compute_group_metrics(samples, self.args.group_size).items()})
        payload["traindiag/skipped_bad_lines"] = bad

        pooled, used = self._window_samples(step)
        payload.update({f"traindiag_win/{k}": v for k, v in compute_suite(pooled).items()})
        payload.update(
            {f"traindiag_win/{k}": v for k, v in compute_group_metrics(pooled, self.args.group_size).items()}
        )
        payload["traindiag_win/num_window_steps"] = used

        payload["global_step"] = step
        wandb.log(payload)

        self.latest_metrics.update({k: v for k, v in payload.items() if k != "global_step"})
        self.state["rollout_steps"] = sorted(set(self.state["rollout_steps"]) | {step})
        self.state["bad_lines_total"] += bad
        # Keep the cache bounded to what the window can ever reuse.
        for old in sorted(self.rollout_cache):
            if len(self.rollout_cache) <= self.args.window_steps:
                break
            del self.rollout_cache[old]
        self.save_state()
        self.save_summary()
        print(
            "[diag] rollout step {}: n={} groups={} all_wrong={:.2f} mixed={:.2f} acc={:.3f} trunc={:.3f} "
            "win={}steps bad_lines={}".format(
                step,
                len(samples),
                payload.get("traindiag/num_groups", 0),
                payload.get("traindiag/frac_groups_all_wrong", 0.0),
                payload.get("traindiag/frac_groups_mixed", 0.0),
                payload.get("traindiag/accuracy", 0.0),
                payload.get("traindiag/truncation_rate", 0.0),
                used,
                bad,
            ),
            flush=True,
        )

    # -- one scan --------------------------------------------------------------
    def scan_once(self) -> int:
        pending: list[tuple[int, str, Path]] = []
        done_eval = set(self.state["eval_steps"])
        for step, d in discover_eval_steps(self.experiment_dir).items():
            if step not in done_eval:
                pending.append((step, "eval", d))
        done_roll = set(self.state["rollout_steps"])
        for step, f in discover_rollout_steps(self.experiment_dir).items():
            if step not in done_roll:
                pending.append((step, "rollout", f))
        pending.sort(key=lambda t: (t[0], t[1]))  # ascending steps; eval before rollout on ties

        processed = 0
        for step, kind, path in pending:
            try:
                if kind == "eval":
                    self.process_eval_step(step, path)
                else:
                    self.process_rollout_step(step, path)
                processed += 1
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Not marked processed -> retried on the next poll. Transient FS
                # or wandb hiccups on a login node shouldn't drop a step.
                print(f"[diag] WARN: {kind} step {step} failed ({type(exc).__name__}: {exc}); will retry", flush=True)
        return processed


def main() -> int:
    args = parse_args()
    sidecar = Sidecar(args)
    print(
        f"[diag] watching {sidecar.experiment_dir} (state: {sidecar.state_path.name}, "
        f"already processed: {len(sidecar.state['eval_steps'])} eval / "
        f"{len(sidecar.state['rollout_steps'])} rollout steps)",
        flush=True,
    )

    run = None
    exit_code = 0
    try:
        while True:
            if run is None:
                run = try_wandb_init(args, sidecar.experiment_dir)
                if run is not None:
                    print(f"[diag] wandb run: {run.name} (id={run.id}, project={args.wandb_project})", flush=True)
            if run is not None:
                n = sidecar.scan_once()
                if n:
                    print(f"[diag] scan complete: {n} new step(s)", flush=True)
                elif args.once:
                    print("[diag] scan complete: 0 new steps", flush=True)
            elif args.once:
                print("[diag] FATAL: wandb.init failed and --once given; nothing processed", file=sys.stderr)
                exit_code = 1
            if args.once:
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[diag] interrupted; exiting cleanly", flush=True)
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:
                print(f"[diag] WARN: wandb.finish failed: {exc}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
