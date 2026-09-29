#!/usr/bin/env python3
"""Move finished relay runs from Horizon to Jupiter through a private HF dataset repo (push_rollouts.sh / pull_rollouts.sh).

    push  --name N --run-dir E/runs/N --jobs-root JOBS_ROOT          Horizon login node
    pull  --dest /e/data1/mmlaion/lee27/relay/horizon [--name N] [--link-runs E/runs]   Jupiter login node

Repo layout (default laion/relay-rollouts-horizon, created private): per run N
  N/run.tar.zst                         the run dir (router logs, driver.log, readout, run.meta, ...) without its jobs link
  N/<job dir>/meta.tar.zst              a harbor job dir's own files (config.json, job result.json, log)
  N/<job dir>/trials_0000.tar.zst ...   CHUNK trial dirs each, in name order
  N/MANIFEST.json                       every unit: sha256, bytes, trial dirs, episodes (trials with result.json), files
  N/DONE                                the manifest's sha256; written last, so a run without DONE is incomplete
push is idempotent and resumable: a unit already in the repo is not packed again (its sha256 comes from the repo's
LFS metadata), each unit is its own commit, and the manifest + DONE go up after every unit is there. One process;
tar | zstd -T1 (gzip if zstd is missing); the wrapper runs it under nice/ionice.
pull verifies every unit's sha256 against the manifest before unpacking it, unpacks under <dest>/N/ (run/ and jobs/;
the tars are deleted after a verified unpack unless --keep-tars), links <dest>/N/run/jobs -> ../jobs so readout.py /
merge_runs.py read it as a Jupiter run dir, and with --link-runs
links <link-runs>/N -> <dest>/N/run. A unit is unpacked once (<dest>/N/.pulled/<unit>.ok); --max-new-files caps the files
one call may create (the per-user inode caps).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO = 'laion/relay-rollouts-horizon'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def compressor():
    if shutil.which('zstd'):
        return ['zstd', '-T1', '-3', '-q'], '.tar.zst'
    return ['gzip', '-6'], '.tar.gz'


def count_files(base, members):
    """Files + dirs the members hold (what unpacking them creates: the inode cost on the pull side)."""
    n = 0
    for m in members:
        p = os.path.join(base, m)
        n += 1
        if os.path.isdir(p) and not os.path.islink(p):
            n += sum(len(ds) + len(fs) for _, ds, fs in os.walk(p))
    return n


def pack(base, members, out, comp):
    """tar members (paths relative to base) | compressor > out."""
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.list') as fl:
        fl.write('\n'.join(members) + '\n')
    try:
        with open(out, 'wb') as fo:
            tar = subprocess.Popen(['tar', '-C', base, '--sort=name', '-cf', '-', '-T', fl.name], stdout=subprocess.PIPE)
            z = subprocess.Popen(comp, stdin=tar.stdout, stdout=fo)
            tar.stdout.close()
            if z.wait() or tar.wait():
                raise RuntimeError(f'packing {out} failed')
    finally:
        os.unlink(fl.name)


def remote_files(api, repo, prefix):
    try:
        items = api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type='dataset')
        return {f.path: (f.lfs.sha256 if getattr(f, 'lfs', None) else None) for f in items if hasattr(f, 'size')}
    except Exception as e:  # noqa: BLE001  (a run not pushed yet: the folder does not exist)
        if 'not found' in str(e).lower() or '404' in str(e):
            return {}
        raise


def job_dirs(run_dir, jobs_root):
    """The run's harbor job dirs, from the configs run_pilot.sh rendered (<run>/<name>_<arm>[_p2].yaml job_name)."""
    import yaml
    names = []
    for y in sorted(glob.glob(os.path.join(run_dir, '*.yaml'))):
        try:
            jn = (yaml.safe_load(open(y)) or {}).get('job_name')
        except Exception:  # noqa: BLE001
            continue
        if jn and os.path.isdir(os.path.join(jobs_root, jn)) and jn not in names:
            names.append(jn)
    return names


def push(a):
    from huggingface_hub import HfApi
    api = HfApi()
    if not a.allow_unfinished:
        dl = os.path.join(a.run_dir, 'driver.log')
        if not (os.path.exists(os.path.join(a.run_dir, 'run.meta')) and os.path.exists(dl) and 'RUN_DONE' in open(dl, errors='replace').read()):
            sys.exit(f'{a.run_dir}: no RUN_DONE in driver.log / no run.meta (a run still going? --allow-unfinished)')
    api.create_repo(a.repo, repo_type='dataset', private=True, exist_ok=True)
    comp, ext = compressor()
    stage = os.path.join(a.stage, a.name)
    os.makedirs(stage, exist_ok=True)
    have = remote_files(api, a.repo, a.name)
    if f'{a.name}/DONE' in have and not a.force:
        print(f'{a.name}: already complete in {a.repo}'); return
    units = [dict(path=f'{a.name}/run{ext}', kind='run', base=os.path.abspath(a.run_dir),
                  members=[m for m in sorted(os.listdir(a.run_dir)) if not os.path.islink(os.path.join(a.run_dir, m))])]
    for jd in job_dirs(a.run_dir, a.jobs_root):
        top = sorted(os.listdir(os.path.join(a.jobs_root, jd)))
        trials = [t for t in top if os.path.isdir(os.path.join(a.jobs_root, jd, t))]
        files = [f'{jd}/{t}' for t in top if t not in trials]
        if files:
            units.append(dict(path=f'{a.name}/{jd}/meta{ext}', kind='job_meta', jobdir=jd, base=a.jobs_root, members=files))
        for k in range(0, len(trials), a.chunk):
            units.append(dict(path=f'{a.name}/{jd}/trials_{k // a.chunk:04d}{ext}', kind='trials', jobdir=jd, base=a.jobs_root,
                              members=[f'{jd}/{t}' for t in trials[k:k + a.chunk]]))
    manifest = dict(run=a.name, repo=a.repo, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host=socket.gethostname(),
                    run_dir=os.path.abspath(a.run_dir), jobs_root=os.path.abspath(a.jobs_root), chunk=a.chunk, units=[])
    for u in units:
        m = dict(path=u['path'], kind=u['kind'], jobdir=u.get('jobdir'), files=count_files(u['base'], u['members']))
        if u['kind'] == 'trials':
            m['trials'] = len(u['members'])
            m['episodes'] = sum(1 for t in u['members'] if os.path.exists(os.path.join(u['base'], t, 'result.json')))
        if have.get(u['path']):
            m['sha256'] = have[u['path']]
            print(f'have  {u["path"]}')
        else:
            t = time.time()
            local = os.path.join(stage, u['path'].split('/', 1)[1].replace('/', '__'))
            pack(u['base'], u['members'], local, comp)
            m['sha256'], m['bytes'] = sha256(local), os.path.getsize(local)
            api.upload_file(path_or_fileobj=local, path_in_repo=u['path'], repo_id=a.repo, repo_type='dataset',
                            commit_message=f'{u["path"]} ({m.get("episodes", "-")} episodes)')
            os.unlink(local)
            print(f'push  {u["path"]} {m["bytes"] / 1e6:.1f} MB {m["files"]} files {m.get("episodes", "")} ep '
                  f'{time.time() - t:.0f} s', flush=True)
        manifest['units'].append(m)
    manifest['episodes'] = sum(u.get('episodes', 0) for u in manifest['units'])
    manifest['files'] = sum(u.get('files') or 0 for u in manifest['units'])
    mp = os.path.join(stage, 'MANIFEST.json')
    json.dump(manifest, open(mp, 'w'), indent=1)
    api.upload_file(path_or_fileobj=mp, path_in_repo=f'{a.name}/MANIFEST.json', repo_id=a.repo, repo_type='dataset',
                    commit_message=f'{a.name} manifest')
    api.upload_file(path_or_fileobj=(sha256(mp) + '\n').encode(), path_in_repo=f'{a.name}/DONE', repo_id=a.repo,
                    repo_type='dataset', commit_message=f'{a.name} done: {manifest["episodes"]} episodes')
    print(f'{a.name}: {len(units)} units, {manifest["episodes"]} episodes, {manifest["files"]} files -> {a.repo}')


def pull(a):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    runs = [a.name] if a.name else sorted({p.split('/')[0] for p in api.list_repo_files(a.repo, repo_type='dataset')
                                           if p.endswith('/DONE')})
    budget = a.max_new_files
    for name in runs:
        root = os.path.join(a.dest, name)
        tars, marks = os.path.join(root, 'tars'), os.path.join(root, '.pulled')
        os.makedirs(tars, exist_ok=True); os.makedirs(marks, exist_ok=True)
        try:
            done = open(hf_hub_download(a.repo, f'{name}/DONE', repo_type='dataset', local_dir=tars)).read().strip()
        except Exception:  # noqa: BLE001
            print(f'{name}: no DONE in the repo yet, skipped'); continue
        mp = hf_hub_download(a.repo, f'{name}/MANIFEST.json', repo_type='dataset', local_dir=tars, force_download=True)
        if sha256(mp) != done:
            sys.exit(f'{name}: MANIFEST.json sha256 does not match DONE')
        man = json.load(open(mp))
        new = 0
        for u in man['units']:
            ok = os.path.join(marks, u['path'].split('/', 1)[1].replace('/', '__') + '.ok')
            if os.path.exists(ok):
                continue
            if (u.get('files') or 0) > budget:
                print(f'{name}: stopping before {u["path"]}: {u.get("files")} files > the {budget} left of --max-new-files'); break
            p = hf_hub_download(a.repo, u['path'], repo_type='dataset', local_dir=tars)
            got = sha256(p)
            if got != u['sha256']:
                sys.exit(f'{u["path"]}: sha256 {got} != manifest {u["sha256"]}')
            out = os.path.join(root, 'run' if u['kind'] == 'run' else 'jobs')
            os.makedirs(out, exist_ok=True)
            dec = 'zstd -d -q -c' if p.endswith('.zst') else 'gzip -d -c'
            subprocess.run(f'{dec} {p!r} | tar -C {out!r} -xf -', shell=True, check=True)
            budget -= u.get('files') or 0
            new += 1
            open(ok, 'w').write(u['sha256'] + '\n')
            if not a.keep_tars:
                os.unlink(p)
            print(f'pull  {u["path"]} ok ({u.get("episodes", "-")} episodes, {u.get("files")} files)', flush=True)
        run = os.path.join(root, 'run')
        if os.path.isdir(run) and not os.path.lexists(os.path.join(run, 'jobs')):
            os.symlink('../jobs', os.path.join(run, 'jobs'))
        if a.link_runs and os.path.isdir(run):
            ln = os.path.join(a.link_runs, name)
            if not os.path.lexists(ln):
                os.symlink(os.path.realpath(run), ln)
            elif os.path.realpath(ln) != os.path.realpath(run):
                print(f'{name}: {ln} exists and is not this run; not linked')
        left = sum(1 for u in man['units'] if not os.path.exists(os.path.join(marks, u['path'].split('/', 1)[1].replace('/', '__') + '.ok')))
        print(f'{name}: {new} units unpacked now, {left} left; {man["episodes"]} episodes in the run; run dir {run}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['push', 'pull'])
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--name')
    ap.add_argument('--run-dir'); ap.add_argument('--jobs-root')
    ap.add_argument('--stage', default=os.path.join(os.environ.get('SCRATCH', '/tmp'), 'relay_push_stage'))
    ap.add_argument('--chunk', type=int, default=500, help='trial dirs per tar')
    ap.add_argument('--allow-unfinished', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--dest'); ap.add_argument('--link-runs')
    ap.add_argument('--max-new-files', type=int, default=400000)
    ap.add_argument('--keep-tars', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'push':
        if not (a.name and a.run_dir and a.jobs_root):
            sys.exit('push needs --name --run-dir --jobs-root')
        push(a)
    else:
        if not a.dest:
            sys.exit('pull needs --dest')
        pull(a)
