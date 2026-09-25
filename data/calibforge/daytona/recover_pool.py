#!/usr/bin/env python3
"""Prepare a portable CalibForge pool backup, publish immutable images, or restore and smoke-gate it.

Adapted from data/r2egym/daytona_artifacts/recover_pool.py. CalibForge ships no reference solutions, so the smoke gate
is a no-op run per base: setup.sh must apply every task layer (digest-checked), the verifier must run to completion
(ctrf report with tests collected) and the reward must be 0. Run with the hook-enabled Harbor Python environment.
No command deletes snapshots.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from build_snapshots import DEFAULT_KEY_FILE, load_secret, repo_dockerfiles, get_optional, state

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def prepare(args):
    from daytona import Image
    from harbor.environments.daytona.snapshots import bake_agent_tooling
    from harbor.utils.container_cache import environment_dir_hash_truncated
    tree, out = Path(args.tree).resolve(), Path(args.bundle).resolve()
    if out.exists():
        raise ValueError(f'{out} exists; use a new directory (backups are immutable)')
    plan = repo_dockerfiles(tree, None)
    out.mkdir(parents=True)
    (out/'dockerfiles').mkdir(); (out/'smoke').mkdir()
    tasks = (tree/'TASKS.txt').read_text().splitlines()
    rows = []
    for repo, (df, name) in plan.items():
        matches = [t for t in tasks if (tree/t/'environment/Dockerfile').read_bytes() == df.read_bytes()]
        task = smoke_task(tree, matches)
        hashes = {environment_dir_hash_truncated(tree/t/'environment', truncate=12) for t in matches}
        if hashes != {name.split('__')[1]}:
            raise ValueError(f'{repo}: more than one environment hash; refusing incomplete backup')
        baked = bake_agent_tooling(Image.from_dockerfile(str(df))).dockerfile()
        recipe = out/'dockerfiles'/f'Dockerfile.{repo}'
        recipe.write_text(baked)
        shutil.copytree(tree/task, out/'smoke'/task)
        rows.append(dict(repo=repo, name=name, dockerfile=str(recipe.relative_to(out)),
            dockerfile_sha256=sha(recipe), baked=True, task=task, task_count=len(matches), image=None))
    archive = out/'task_tree.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        # Preserve the complete task tree, not stale per-session build logs.
        for task in tasks:
            tar.add(tree/task, arcname=f'task_tree/{task}')
        for p in [tree/'TASKS.txt', tree/'pool.json', tree/'coverage.tsv', *tree.glob('Dockerfile.*')]:
            tar.add(p, arcname=f'task_tree/{p.name}')
    write_json(out/'manifest.json', dict(version=1, images=rows,
        task_tree=dict(file=archive.name, sha256=sha(archive), tasks=len(tasks))))
    for name in ['recover_pool.py', 'build_snapshots.py', 'gate_nop.yaml', 'build_tree.py', 'pool.py', 'registry.py',
                 'setup_template.sh', 'snapshot_census.py', 'gate.py', 'fidelity_check.py', 'restore.sh', 'RECOVERY.md', 'burst_test.py', 'mirror_upload.py',
                 'test_snapshot_recovery.py']:
        shutil.copy2(HERE/name, out/name)
    print(f'Prepared {len(rows)} images / {len(tasks)} tasks at {out}', flush=True)


def read_bundle(args):
    bundle = Path(args.bundle).resolve()
    data = json.loads((bundle/'manifest.json').read_text())
    if data['version'] != 1:
        raise ValueError('Unsupported manifest')
    archive = bundle / data['task_tree']['file']
    if sha(archive) != data['task_tree']['sha256']:
        raise ValueError('Task archive checksum mismatch')
    for row in data['images']:
        if sha(bundle/row['dockerfile']) != row['dockerfile_sha256']:
            raise ValueError(f"Recipe changed: {row['repo']}")
    selected = args.repos.split(',') if args.repos else [r['repo'] for r in data['images']]
    rows = [r for r in data['images'] if r['repo'] in selected]
    if set(selected) != {r['repo'] for r in rows}:
        raise ValueError('Unknown repository selection')
    return bundle, data, sorted(rows, key=lambda r: r['repo'] != 'numpy')


def publish(args):
    bundle, data, rows = read_bundle(args)
    if not args.registry or '://' in args.registry or any(c.isspace() for c in args.registry):
        raise ValueError('Use a registry namespace such as ghcr.io/owner/r2egym')
    # Uses an already authenticated Docker/buildx context; never accepts passwords on argv.
    subprocess.run(['docker', 'buildx', 'inspect', '--bootstrap'], check=True)
    for row in rows:
        if row.get('image'):
            print(f"{row['repo']}: digest already saved, skipping", flush=True)
            continue
        tag = f"{args.registry.rstrip('/')}/{row['repo']}:{row['name'].split('__')[1]}-{row['dockerfile_sha256'][:12]}"
        metadata = bundle/f"{row['repo']}.build-metadata.json"
        subprocess.run(['docker','buildx','build','--platform','linux/amd64','--provenance=false',
            '--file',str(bundle/row['dockerfile']),'--tag',tag,'--push','--metadata-file',str(metadata),
            str(bundle/'dockerfiles')], check=True)
        digest = json.loads(metadata.read_text())['containerimage.digest']
        if len(digest) != 71 or not digest.startswith('sha256:') or any(c not in '0123456789abcdef' for c in digest[7:]):
            raise ValueError('buildx returned an invalid digest')
        row['image'] = tag.rsplit(':',1)[0]+'@'+digest
        write_json(bundle/'manifest.json',data)  # Persist each completed image before the next build.
        print(f"Saved {row['repo']}: {row['image']}", flush=True)


def smoke_task(tree, matches):
    """Smallest task layers among tasks whose verifier writes a ctrf report (so completion is checkable)."""
    def delta(t):
        text = (tree/t/'setup_files/setup.sh').read_text()
        body = text.split('LAYERS="', 1)[1].split('"', 1)[0]
        return sum(int(line.split()[1]) for line in body.splitlines() if line.strip())
    ctrf = [t for t in matches if '--ctrf' in (tree/t/'tests/test.sh').read_text()]
    return min(ctrf or matches, key=lambda t: (delta(t), t))


def verifier_completed(trial_dir):
    """The verifier ran its tests to the end: a ctrf report with at least one collected test."""
    reports = sorted(trial_dir.glob('verifier/ctrf.json')) + sorted(trial_dir.glob('attempts/*/verifier/ctrf.json'))
    try:
        summary = json.loads(reports[-1].read_text())['results']['summary']
    except (IndexError, OSError, ValueError, KeyError, TypeError):
        return False
    return summary.get('tests', 0) > 0


def set_digests(args):
    """Write the GitHub Actions run's digests (its `digests` artifact) into the manifest, after checking that each
    published image was built from this bundle's recipe for this snapshot name (tag = cf1-<recipe sha256[:12]>)."""
    bundle = Path(args.bundle).resolve()
    data = json.loads((bundle/'manifest.json').read_text())
    published = {r['repo']: r for r in json.loads(Path(args.digests).read_text())['images']}
    for row in data['images']:
        pub = published.get(row['repo'])
        if pub is None:
            continue
        if pub['name'] != row['name'] or pub['tag'] != f"cf1-{row['dockerfile_sha256'][:12]}":
            raise ValueError(f"{row['repo']}: published {pub['name']}/{pub['tag']} does not match this bundle")
        ref = pub['image']
        if '@sha256:' not in ref or len(ref.split('@sha256:')[1]) != 64:
            raise ValueError(f"{row['repo']}: not a digest reference: {ref}")
        row['image'] = ref
        print(f"{row['repo']}: {ref}", flush=True)
    write_json(bundle/'manifest.json', data)


def gate_ok(job, expected, require_ctrf=True):
    result = json.loads((job/'result.json').read_text())
    stats = result['stats']
    trials = list(job.glob('*/result.json'))
    if stats['n_completed_trials'] != 1 or stats['n_errored_trials'] or len(trials) != 1:
        return False
    trial = json.loads(trials[0].read_text())
    ok = not trial.get('exception_info') and (trial.get('verifier_result') or {}).get('rewards', {}).get('reward') == expected
    return ok and (not require_ctrf or verifier_completed(trials[0].parent))



async def verify_still_present(service, results, names):
    """A shared-org purge can invalidate an earlier success while other gates run."""
    for result in results:
        if not result['passed']:
            continue
        try:
            current = await asyncio.wait_for(get_optional(service,names[result['repo']]),30)
            if current is None or state(current) != 'active' or current.id != result['snapshot_id']:
                result.update(passed=False,stage='final_snapshot_check',error='Snapshot was removed, replaced or deactivated during the gate; rerun recovery')
        except Exception as exc:
            result.update(passed=False,stage='final_snapshot_check',error=type(exc).__name__)


def restore(args):
    bundle, data, rows = read_bundle(args)
    run = bundle/'runs'/time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['DAYTONA_API_KEY'] = load_secret('DAYTONA_API_KEY',args.key_file)
    semaphore = asyncio.Semaphore(args.parallel)
    # One recovery worker per image; gate it immediately instead of waiting for numpy.
    async def worker(row):
        async with semaphore:
            repo = row['repo']; dest = run/repo; dest.mkdir()
            cmd = [sys.executable, str(HERE/'build_snapshots.py'), '--manifest',str(bundle/'manifest.json'),
                '--repos',repo,'--parallel','1','--logs',str(dest/'build'),'--key-file',args.key_file]
            if args.require_registry:
                cmd.append('--require-registry')
            with (dest/'restore.log').open('w') as log:
                process = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                if await process.wait():
                    return dict(repo=repo, passed=False, stage='restore', log=str(dest/'restore.log'))
            tree = dest/'tree'; tree.mkdir()
            if args.mirror_only:
                # prove the layer mirror alone serves the task: no Docker Hub fallback in this smoke run
                shutil.copytree(bundle/'smoke'/row['task'], tree/row['task'])
                sh = tree/row['task']/'setup_files/setup.sh'
                text = sh.read_text()
                if 'CF_DOCKERHUB="${CF_DOCKERHUB:-1}"' not in text:
                    return dict(repo=repo, passed=False, stage='mirror_only', error='setup.sh has no Docker Hub switch')
                sh.write_text(text.replace('CF_DOCKERHUB="${CF_DOCKERHUB:-1}"', 'CF_DOCKERHUB="${CF_DOCKERHUB:-0}"'))
            else:
                (tree/row['task']).symlink_to(bundle/'smoke'/row['task'], target_is_directory=True)
            for agent, expected in [('nop',0.0)]:
                # JSON is also valid YAML; avoid path interpolation/quoting problems.
                import yaml
                config = yaml.safe_load((HERE/f'gate_{agent}.yaml').read_text())
                config.update(job_name=agent, jobs_dir=str(dest/'jobs'), n_concurrent_trials=1)
                config['datasets'] = [dict(path=str(tree))]
                cfg = dest/f'{agent}.yaml'; cfg.write_text(json.dumps(config,indent=2))
                with (dest/f'{agent}.log').open('w') as log:
                    process = await asyncio.create_subprocess_exec(args.harbor,'jobs','start','--config',str(cfg),
                        env=env,stdout=log,stderr=subprocess.STDOUT)
                    rc = await process.wait()
                try:
                    passed = rc == 0 and gate_ok(dest/'jobs'/agent,expected)
                except (OSError,ValueError,KeyError):
                    passed = False
                if not passed:
                    return dict(repo=repo,passed=False,stage=agent,log=str(dest/f'{agent}.log'))
            print(f'{repo}: ACTIVE; nop 0 + verifier completed PASS',flush=True)
            built = json.loads((dest/'build/results.json').read_text())[0]
            return dict(repo=repo,passed=True,snapshot_id=built['snapshot_id'])
    async def run_all():
        results = []
        async def record(row):
            try:
                result = await worker(row)
            except Exception as exc:
                result = dict(repo=row['repo'],passed=False,error=str(exc))
            results.append(result); write_json(run/'summary.json',results)
        await asyncio.gather(*(record(row) for row in rows))
        from daytona import AsyncDaytona, DaytonaConfig
        async with AsyncDaytona(DaytonaConfig(api_key=env['DAYTONA_API_KEY'])) as daytona:
            await verify_still_present(daytona.snapshot,results,{r['repo']:r['name'] for r in rows})
        write_json(run/'summary.json',results)
        return results
    results = asyncio.run(run_all())
    print(f'Recovery results: {run}/summary.json',flush=True)
    return 0 if all(r['passed'] for r in results) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--tree',required=True); p.add_argument('--bundle',required=True)
    p = sub.add_parser('publish'); p.add_argument('--bundle',required=True); p.add_argument('--registry',required=True); p.add_argument('--repos')
    p = sub.add_parser('set-digests'); p.add_argument('--bundle',required=True); p.add_argument('--digests',required=True)
    p = sub.add_parser('restore'); p.add_argument('--bundle',required=True); p.add_argument('--repos')
    p.add_argument('--key-file',default=str(Path(DEFAULT_KEY_FILE).expanduser()))
    p.add_argument('--parallel',type=int,default=4); p.add_argument('--harbor',default='harbor')
    p.add_argument('--require-registry',action='store_true')
    p.add_argument('--mirror-only',action='store_true',help='smoke gate with the Docker Hub fallback switched off')
    args = ap.parse_args()
    if args.command == 'restore' and not 1 <= args.parallel <= 5:
        ap.error('parallel must be 1–5')
    return {'prepare':prepare,'publish':publish,'set-digests':set_digests,'restore':restore}[args.command](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
