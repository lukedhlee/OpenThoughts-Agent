#!/usr/bin/env python3
"""Build the CalibForge task tree for Daytona: three shared snapshots, every task's own image layers applied at start.

Each task becomes:
  environment/Dockerfile   the recipe of the base it belongs to (pool.dockerfile), byte-identical across that base's
                           tasks, so harbor's environment-dir hash gives one `harbor__<hash>__snapshot` per base.
  setup_files/setup.sh     setup_template.sh with the task image's layers above the base (digest + size): fetched,
                           sha256-checked and stacked at sandbox start by harbor's setup-files hook.
  task.toml                CalibForge's, minus docker_image (the snapshot serves it), plus [environment] workdir and
                           the task image's ENV entries that differ from the base's (PATH for venv tasks, ...).
  instruction.md, tests/   verbatim.

A task is covered when its image starts with one of the bases' layers (longest match wins), its working directory is
/app (harbor cds into the workdir before running setup.sh, so a deeper workdir does not exist yet), and its layers
above the base are at most --max-delta-gb compressed (sandbox disk).

  python build_tree.py --parquet calibforge_tasks.parquet --source task-data.tar.gz \
      --cache ~/.local/share/otagent/calibforge-work --out <tree> [--all] [--max-delta-gb 1.5]

Writes TASKS.txt, Dockerfile.<base>, pool.json (per base: tasks, snapshot name, sizes) and coverage.tsv
(every selected task with its base or its exclusion reason). Manifests and configs come from registry.py's cache
(fetched on demand, one anonymous manifest GET per image).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pool import BASES, dockerfile  # noqa: E402
from registry import Hub  # noqa: E402

TEMPLATE = (HERE / "setup_template.sh").read_text()
MIRROR = "https://huggingface.co/datasets/laion/calibforge-daytona-layers/resolve/main/blobs/sha256"


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def dump_toml(d: dict) -> str:
    top = [f"{k} = {toml_value(v)}" for k, v in d.items() if not isinstance(v, dict)]
    out = "\n".join(top) + ("\n\n" if top else "")
    for sec, body in d.items():
        if not isinstance(body, dict):
            continue
        out += f"[{sec}]\n"
        for k, v in body.items():
            out += f"{k} = {toml_value(v)}\n"
        out += "\n"
    return out


def env_dict(env: list[str] | None) -> dict[str, str]:
    return dict(e.split("=", 1) for e in (env or []))


def assign(layers: list[str]) -> tuple[str | None, int]:
    best, k = None, 0
    for label, b in BASES.items():
        n = len(b["layers"])
        if layers[:n] == b["layers"] and n > k:
            best, k = label, n
    return best, k


def source_files(source: Path, wanted: set[str]):
    """Yield (task_id, relpath, bytes, mode) for instruction.md, task.toml, tests/**."""
    keep = lambda rel: rel in ("instruction.md", "task.toml") or rel.startswith("tests/")  # noqa: E731
    if source.is_dir():
        for t in sorted(wanted):
            base = source / t
            for p in sorted(base.rglob("*")):
                rel = str(p.relative_to(base))
                if p.is_file() and keep(rel):
                    yield t, rel, p.read_bytes(), p.stat().st_mode & 0o777
        return
    with tarfile.open(source, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            t, _, rel = m.name.partition("/")
            if t in wanted and keep(rel):
                yield t, rel, tf.extractfile(m).read(), m.mode & 0o777


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--source", required=True, help="hamishivi task-data.tar.gz or its extracted directory")
    ap.add_argument("--cache", required=True, help="registry.py manifest/config cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="every task, not only recommended_2500")
    ap.add_argument("--max-delta-gb", type=float, default=1.5)
    ap.add_argument("--mirrors", default=MIRROR, help="default CF_MIRRORS baked into setup.sh (space-separated blob base URLs)")
    ap.add_argument("--no-dockerhub", action="store_true", help="bake CF_DOCKERHUB=0: mirror only, no Docker Hub fallback")
    a = ap.parse_args()

    import pandas as pd
    d = pd.read_parquet(a.parquet)
    if not a.all:
        d = d[d.recommended_2500]
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; build into a new directory")
    cache, hub = Path(a.cache), Hub()

    rows, cover = {}, []
    for r in d.itertuples():
        m = hub.manifest(r.image_digest, cache)
        cfg = hub.config(r.image_digest.split("@")[0], m, cache)["config"]
        layers = [x["digest"] for x in m["layers"]]
        label, k = assign(layers)
        delta = m["layers"][k:]
        dbytes = sum(x["size"] for x in delta)
        reason = ""
        if label is None:
            reason = f"base {r.base_image} not in the pool"
        elif (cfg.get("WorkingDir") or "/") != "/app":
            reason = f"workdir {cfg.get('WorkingDir')!r} is not /app"
        elif dbytes > a.max_delta_gb * 1e9:
            reason = f"task layers {dbytes / 1e9:.1f} GB compressed > {a.max_delta_gb} GB"
        cover.append((r.task_id, r.base_image, label or "", len(delta), dbytes, reason or "covered"))
        if reason:
            continue
        base_env = env_dict(BASES[label]["env"])
        env = {k2: v for k2, v in env_dict(cfg.get("Env")).items() if base_env.get(k2) != v}
        rows[r.task_id] = dict(label=label, image=r.image_digest, delta=delta, env=env,
                               entrypoint=cfg.get("Entrypoint"), cmd=cfg.get("Cmd"))

    out.mkdir(parents=True)
    recipes = {label: dockerfile(label) for label in BASES}
    for label, text in recipes.items():
        (out / f"Dockerfile.{label}").write_text(text)
    seen = collections.defaultdict(set)
    for t, rel, data, mode in source_files(Path(a.source), set(rows)):
        seen[t].add(rel)
        dst = out / t / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "task.toml":
            cfg = tomllib.loads(data.decode())
            env = cfg.setdefault("environment", {})
            env.pop("docker_image", None)
            env["workdir"] = "/app"
            if rows[t]["env"]:
                env["env"] = rows[t]["env"]
            md = cfg.setdefault("metadata", {})
            md["calibforge_image"] = rows[t]["image"]
            md["calibforge_daytona_base"] = rows[t]["label"]
            dst.write_text(dump_toml(cfg))
        else:
            dst.write_bytes(data)
            dst.chmod(mode | 0o600)
    missing = [t for t in rows if not {"instruction.md", "task.toml", "tests/test.sh"} <= seen[t]]
    if missing:
        raise SystemExit(f"{len(missing)} tasks lack instruction/task.toml/tests in {a.source}: {missing[:5]}")

    for t, row in rows.items():
        (out / t / "environment").mkdir(parents=True, exist_ok=True)
        (out / t / "environment" / "Dockerfile").write_text(recipes[row["label"]])
        sh = out / t / "setup_files" / "setup.sh"
        sh.parent.mkdir(parents=True, exist_ok=True)
        layers = "\n".join(f"{x['digest']} {x['size']}" for x in row["delta"])
        sh.write_text(TEMPLATE.replace("@@IMAGE@@", row["image"]).replace("@@MIRRORS@@", a.mirrors)
                      .replace("@@DOCKERHUB@@", "0" if a.no_dockerhub else "1")
                      .replace("@@LAYERS@@", layers))
        sh.chmod(0o755)

    try:
        from harbor.utils.container_cache import environment_dir_hash_truncated as h12
    except Exception:  # noqa: BLE001
        h12 = None
    pool = {}
    by_label = collections.defaultdict(list)
    for t, row in sorted(rows.items()):
        by_label[row["label"]].append(t)
    for label, tasks in sorted(by_label.items()):
        names = {h12(out / t / "environment", truncate=12) for t in tasks} if h12 else set()
        if h12 and len(names) != 1:
            raise SystemExit(f"{label}: {len(names)} environment hashes; tasks must share one snapshot")
        sizes = sorted(sum(x["size"] for x in rows[t]["delta"]) for t in tasks)
        pool[label] = dict(base_image=BASES[label]["image"], name=f"harbor__{names.pop()}__snapshot" if h12 else None,
                           dockerfile=f"Dockerfile.{label}",
                           dockerfile_sha256=hashlib.sha256(recipes[label].encode()).hexdigest(),
                           task=tasks[0], task_count=len(tasks),
                           delta_mb_median=round(sizes[len(sizes) // 2] / 1e6, 1),
                           delta_mb_p90=round(sizes[int(len(sizes) * 0.9)] / 1e6, 1),
                           delta_gb_total=round(sum(sizes) / 1e9, 1))
    (out / "pool.json").write_text(json.dumps(pool, indent=1) + "\n")
    (out / "TASKS.txt").write_text("\n".join(sorted(rows)) + "\n")
    with open(out / "coverage.tsv", "w") as f:
        f.write("task_id\tbase_image\tdaytona_base\ttask_layers\ttask_layer_bytes\tstatus\n")
        for c in cover:
            f.write("\t".join(map(str, c)) + "\n")
    why = collections.Counter(c[5] if c[5] == "covered" else c[5].split(" ")[0] + " " + c[5].split(" ")[1] for c in cover)
    print(json.dumps({"selected": len(cover), "covered": len(rows), "why": why, "pool": pool}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
