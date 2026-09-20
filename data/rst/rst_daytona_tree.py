#!/usr/bin/env python3
"""Rewrite a held-out RST task tree for Daytona under the snapshot cap: one shared image per base, the task's
Dockerfile replayed as ``setup_files/setup.sh`` at trial start (root, internet) by the harbor setup-files hook
(lukedhlee/terminus2-think-parity-bridgewait-setuphook, cloned on Jupiter at code/harbor-hook).

This is how the rollouts themselves were collected ("Task Dockerfiles ... converted into setup scripts replayed in
fresh containers sharing a minimal base image"), so the probe's sandboxes match the teacher's. Jupiter's apptainer
path was ruled out on 2026-09-20: no user namespaces on login or compute nodes, so the agent cannot be root there.

    python rst_daytona_tree.py --src /e/data1/mmlaion/lee27/tasks/rst_heldout --out /e/data1/mmlaion/lee27/tasks/rst_heldout_daytona

Per task: environment/Dockerfile becomes the canonical per-base file (identical text -> one harbor snapshot per base),
setup_files/setup.sh carries the RUN/ENV/WORKDIR/COPY/ARG/USER steps in order, COPY sources sit under setup_files/copy/,
instruction.md, task.toml, tests/ and solution/ are copied verbatim.
"""
import argparse, collections, json, re, shlex, shutil
from pathlib import Path

TMUX = ("if command -v tmux >/dev/null 2>&1 && command -v asciinema >/dev/null 2>&1; then :; "
        "elif command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*; "
        "elif command -v yum >/dev/null 2>&1; then yum install -y tmux && yum clean all; "
        "elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux; fi")


CENTOS_VAULT = ("sed -i -e 's/^mirrorlist=/#mirrorlist=/' -e 's|^#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|' /etc/yum.repos.d/CentOS-*.repo && ")


def base_dockerfile(base: str) -> str:
    fix = CENTOS_VAULT if base.startswith("centos:7") else ""  # centos 7 is EOL: its mirrorlist is gone, yum needs the vault
    return (f"FROM {base}\n\nENV DEBIAN_FRONTEND=noninteractive\n\n# shared per-base image; the task's own Dockerfile steps run from setup_files/setup.sh\n"
            f"RUN {fix}{TMUX} \\\n    && mkdir -p /app /tests /logs/verifier\n\nWORKDIR /app\n")


# The rollouts were collected with 2 CPU / 4 GB / 6 GB sandboxes and 1,800 s agent / 2,400 s verifier limits on every
# task (bundle README). The source task.toml files say 1 CPU / 2 GB and 600–1,800 s: the first probe on those lost
# 12 % of trials to killed tmux sessions (2 GB) and 5 setups to a 600 s build timeout. Parity with the teacher instead.
RESOURCES = {"cpus": "2", "memory": '"4G"', "storage": '"6G"', "memory_mb": "4096", "storage_mb": "6144", "build_timeout_sec": "1800.0"}
TIMEOUTS = {"agent": "1800.0", "verifier": "2400.0"}


def rewrite_toml(text: str) -> str:
    out, section = [], None
    for line in text.splitlines():
        m = re.match(r"\s*\[(\w+)\]", line)
        if m:
            section = m.group(1)
        kv = re.match(r"\s*(\w+)\s*=", line)
        key = kv.group(1) if kv else None
        if section == "environment" and key in RESOURCES:
            line = f"{key} = {RESOURCES[key]}"
        elif section in TIMEOUTS and key == "timeout_sec":
            line = f"timeout_sec = {TIMEOUTS[section]}"
        out.append(line)
    return "\n".join(out) + "\n"


KEYWORDS = {"FROM", "RUN", "ENV", "ARG", "WORKDIR", "USER", "COPY", "ADD", "SHELL", "CMD", "ENTRYPOINT", "EXPOSE", "LABEL",
            "HEALTHCHECK", "VOLUME", "STOPSIGNAL", "MAINTAINER", "ONBUILD"}
HEREDOC_RE = re.compile(r"<<-?(['\"]?)(\w+)\1")


def instructions(text: str):
    """Yield (instruction line, [(marker, body_lines), ...]) with BuildKit heredocs attached (118 of the 337 held-out
    Dockerfiles use `RUN <<'EOF' cat > file` / `COPY <<'EOF' /dest`). Continuation joining never touches heredoc bodies."""
    lines = text.splitlines(); i = 0
    while i < len(lines):
        s = lines[i].rstrip(); i += 1
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        cur = s
        while cur.endswith("\\") and i < len(lines):
            nxt = lines[i].rstrip(); i += 1
            if nxt.lstrip().startswith("#") or not nxt.strip():
                continue  # Docker drops comment and empty lines inside a continuation
            cur = cur[:-1] + " " + nxt
        cur = cur.strip()
        if cur.split()[0].upper() not in KEYWORDS:
            raise ValueError(f"not a Dockerfile instruction: {cur[:80]}")
        docs = []
        for m in HEREDOC_RE.finditer(cur):
            body = []
            while i < len(lines) and lines[i].rstrip() != m.group(2):
                body.append(lines[i]); i += 1
            i += 1  # terminator
            docs.append((m.group(2), body))
        yield cur, docs


def parse_env(body: str) -> list[tuple[str, str]]:
    if "=" in body.split()[0]:
        out = []
        for p in shlex.split(body):
            k, _, v = p.partition("="); out.append((k, v))
        return out
    k, _, v = body.partition(" ")
    return [(k.strip(), v.strip())]


def convert(task_dir: Path, out_dir: Path) -> dict:
    df = task_dir / "environment" / "Dockerfile"
    base, workdir, user, steps, copies, unsupported = None, "/", "root", [], [], []
    env_persist: list[tuple[str, str]] = []
    heredoc_files: list[tuple[str, str]] = []  # (rel path under setup_files, content)
    for line, docs in instructions(df.read_text()):
        kw, _, body = line.partition(" "); kw = kw.upper(); body = body.strip()
        if kw == "FROM":
            base = body.split()[0]
        elif kw == "RUN":
            if body.startswith("["):  # exec form
                body = " ".join(shlex.quote(x) for x in json.loads(body))
            if docs:
                rest = HEREDOC_RE.sub("", body).strip()
                if not rest and len(docs) == 1:
                    # `RUN <<EOF` alone: the body IS the script (a shebang picks the interpreter, else bash)
                    script = "\n".join(docs[0][1])
                    body = script if script.startswith("#!") else script
                    steps.append(("run_script", script, workdir, user)); continue
                # `RUN <<'EOF' cat > file`: bash accepts the redirection before the command, so keep the line and
                # re-attach each heredoc body and terminator exactly as the shell expects them
                body = body + "".join(f"\n" + "\n".join(b) + f"\n{w}" for w, b in docs)
            steps.append(("run", body, workdir, user))
        elif kw == "ENV":
            for k, v in parse_env(body):
                steps.append(("env", (k, v), workdir, user)); env_persist.append((k, v))
        elif kw == "ARG":
            k, _, v = body.partition("="); steps.append(("env", (k.strip(), v.strip()), workdir, user))
        elif kw == "WORKDIR":
            workdir = body if body.startswith("/") else f"{workdir.rstrip('/')}/{body}"
            steps.append(("workdir", workdir, workdir, user))
        elif kw == "USER":
            user = body.split(":")[0]; steps.append(("user", user, workdir, user))
        elif kw in ("COPY", "ADD"):
            if docs:  # `COPY <<'EOF' /dest` (or several heredocs into a directory)
                dest = HEREDOC_RE.sub("", body).split()[-1]; n = len(copies) + len(heredoc_files)
                for w, b in docs:
                    name = Path(dest).name if (len(docs) == 1 and not dest.endswith("/")) else w
                    rel = f"copy/{n}/{name}"; heredoc_files.append((rel, "\n".join(b) + "\n"))
                    d = dest if (len(docs) == 1 and not dest.endswith("/")) else f"{dest.rstrip('/')}/{w}"
                    steps.append(("copy", (rel, d, False), workdir, user))
                continue
            parts = [p for p in shlex.split(body) if not p.startswith("--")]
            if body.startswith("["): parts = json.loads(body)
            dest = parts[-1]; n = len(copies) + len(heredoc_files)
            for src in parts[:-1]:
                src_path = (task_dir / "environment" / src)
                if not src_path.exists():
                    unsupported.append(f"COPY source missing: {src}"); continue
                rel = f"copy/{n}/{Path(src).name}"; copies.append((src_path, rel))
                steps.append(("copy", (rel, dest, src.endswith("/")), workdir, user))
        elif kw in ("SHELL", "CMD", "ENTRYPOINT", "EXPOSE", "LABEL", "HEALTHCHECK", "VOLUME", "STOPSIGNAL", "MAINTAINER", "ONBUILD"):
            continue
        else:
            unsupported.append(line[:80])
    assert base, f"{task_dir.name}: no FROM"
    sh = ["#!/bin/bash", "# generated by data/rst/rst_daytona_tree.py from environment/Dockerfile of the source task",
          "set -euo pipefail", 'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', "export DEBIAN_FRONTEND=noninteractive", "export HOME=/root", "cd /"]
    for i, (kind, val, wd, usr) in enumerate(steps):
        if kind == "env":
            k, v = val; sh.append(f'export {k}="{v}"')
        elif kind == "workdir":
            sh.append(f"mkdir -p {shlex.quote(val)} && cd {shlex.quote(val)}")
        elif kind == "user":
            sh.append(f"# USER {val}: later RUN steps execute as this user")
        elif kind == "copy":
            rel, dest, src_is_dir = val
            d = dest if dest.startswith("/") else f"{wd.rstrip('/')}/{dest}"
            if src_is_dir or dest.endswith("/"):
                sh.append(f'mkdir -p {shlex.quote(d)} && cp -a "$HERE/{rel}/." {shlex.quote(d)}/ 2>/dev/null || cp -a "$HERE/{rel}" {shlex.quote(d)}')
            else:
                sh.append(f'mkdir -p "$(dirname {shlex.quote(d)})" && cp -a "$HERE/{rel}" {shlex.quote(d)}')
        elif kind in ("run", "run_script"):
            assert "RST_RUN_EOF" not in val
            # 13 of 337 held-out Dockerfiles swap apt to mirrors.aliyun.com (a China mirror); from Daytona's region those
            # setups ran past 1,800 s and were the only setup failures on 2026-09-20. Use the default archive instead.
            val = val.replace("mirrors.aliyun.com/ubuntu", "archive.ubuntu.com/ubuntu").replace("mirrors.aliyun.com/debian", "deb.debian.org/debian")
            sh.append(f"cat > /tmp/rst_run_{i}.sh <<'RST_RUN_EOF'\n{val}\nRST_RUN_EOF")
            runner = f"/tmp/rst_run_{i}.sh" if (kind == "run_script" and val.startswith("#!")) else f"bash -e /tmp/rst_run_{i}.sh"
            if usr == "root":
                sh.append(f"chmod 755 /tmp/rst_run_{i}.sh && {runner}")
            else:
                sh.append(f"chmod 755 /tmp/rst_run_{i}.sh && su {shlex.quote(usr)} -s /bin/bash -c \"cd $(pwd) && {runner}\"")
            sh.append(f"rm -f /tmp/rst_run_{i}.sh")
    # persist ENV and the final WORKDIR for the agent's interactive shell and for later execs
    if env_persist or workdir != "/":
        sh.append("mkdir -p /etc/profile.d")
        prof = ["# RST task environment (from the source Dockerfile)"] + [f'export {k}="{v}"' for k, v in env_persist]
        sh.append("cat > /etc/profile.d/rst_task_env.sh <<'RST_ENV_EOF'\n" + "\n".join(prof) + "\nRST_ENV_EOF")
        bashrc = ["", ". /etc/profile.d/rst_task_env.sh"] + ([f"cd {shlex.quote(workdir)} 2>/dev/null || true"] if workdir != "/" else [])
        sh.append("printf '%s\\n' " + " ".join(shlex.quote(l) for l in bashrc) + " >> /root/.bashrc")
    sh.append("echo RST_SETUP_OK")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment").mkdir(exist_ok=True)
    (out_dir / "environment" / "Dockerfile").write_text(base_dockerfile(base))
    sf = out_dir / "setup_files"; sf.mkdir(exist_ok=True)
    (sf / "setup.sh").write_text("\n".join(sh) + "\n")
    for src_path, rel in copies:
        dst = sf / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir(): shutil.copytree(src_path, dst, dirs_exist_ok=True)
        else: shutil.copy2(src_path, dst)
    for rel, content in heredoc_files:
        dst = sf / rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(content)
    shutil.copy2(task_dir / "instruction.md", out_dir / "instruction.md")
    (out_dir / "task.toml").write_text(rewrite_toml((task_dir / "task.toml").read_text()))
    for sub in ("tests", "solution"):
        if (task_dir / sub).is_dir(): shutil.copytree(task_dir / sub, out_dir / sub, dirs_exist_ok=True)
    return {"base": base, "runs": sum(1 for s in steps if s[0] in ("run", "run_script")), "copies": len(copies) + len(heredoc_files), "workdir": workdir, "user": user, "unsupported": unsupported}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    report = {}
    for d in sorted(p for p in a.src.iterdir() if (p / "environment" / "Dockerfile").exists()):
        report[d.name] = convert(d, a.out / d.name)
    bases = collections.Counter(r["base"] for r in report.values())
    uns = {k: v["unsupported"] for k, v in report.items() if v["unsupported"]}
    nonroot = [k for k, v in report.items() if v["user"] != "root"]
    print(f"{len(report)} tasks; bases {dict(bases)}; unsupported in {len(uns)} tasks; non-root final USER in {len(nonroot)}")
    for k, v in list(uns.items())[:10]: print("  ", k, v)
    (a.out / "_conversion_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
