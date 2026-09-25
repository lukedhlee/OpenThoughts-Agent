import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from build_snapshots import ensure_snapshot, load_secret
from recover_pool import gate_ok


def snap(value):
    return SimpleNamespace(state=value, id='id', size=1)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_is_reused(self):
        service = SimpleNamespace(get=AsyncMock(return_value=snap('active')), activate=AsyncMock())
        create = AsyncMock()
        await ensure_snapshot(service,'pool',create,poll=0)
        create.assert_not_awaited(); service.activate.assert_not_awaited()

    async def test_inactive_is_activated_not_misread_as_active(self):
        service = SimpleNamespace(get=AsyncMock(side_effect=[snap('inactive'),snap('active')]),activate=AsyncMock())
        create = AsyncMock()
        await ensure_snapshot(service,'pool',create,poll=0)
        service.activate.assert_awaited_once(); create.assert_not_awaited()

    async def test_pending_is_resumed_without_creation(self):
        service = SimpleNamespace(get=AsyncMock(side_effect=[snap('building'),snap('pulling'),snap('active')]))
        create = AsyncMock()
        await ensure_snapshot(service,'pool',create,poll=0)
        create.assert_not_awaited()

    async def test_timeout_after_submission_observes_server_build(self):
        service = SimpleNamespace(get=AsyncMock(side_effect=[RuntimeError('404'),snap('building'),snap('active')]))
        create = AsyncMock(side_effect=TimeoutError())
        await ensure_snapshot(service,'pool',create,poll=0)
        create.assert_awaited_once()

    async def test_auth_failure_never_creates(self):
        service = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError('403 access denied')))
        create = AsyncMock()
        with self.assertRaises(RuntimeError):
            await ensure_snapshot(service,'pool',create,poll=0)
        create.assert_not_awaited()

    async def test_failed_snapshot_is_preserved(self):
        service = SimpleNamespace(get=AsyncMock(return_value=snap('build_failed')),delete=AsyncMock())
        create = AsyncMock()
        with self.assertRaises(RuntimeError):
            await ensure_snapshot(service,'pool',create,poll=0)
        create.assert_not_awaited(); service.delete.assert_not_awaited()

    async def test_wait_timeout_does_not_delete_or_restart(self):
        service = SimpleNamespace(get=AsyncMock(return_value=snap('building')),delete=AsyncMock())
        create = AsyncMock()
        with self.assertRaises(TimeoutError):
            await ensure_snapshot(service,'pool',create,timeout=.001,poll=.001)
        create.assert_not_awaited(); service.delete.assert_not_awaited()

    def test_explicit_file_beats_environment_and_handles_comment(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'key'; p.write_text('export DAYTONA_API_KEY="right" # comment\n')
            with patch.dict(os.environ,DAYTONA_API_KEY='wrong'):
                self.assertEqual(load_secret('DAYTONA_API_KEY',str(p)),'right')

    def test_gate_checks_reward_not_just_exit_status(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d); (job/'trial').mkdir()
            (job/'result.json').write_text(json.dumps({'stats':{'n_completed_trials':1,'n_errored_trials':0}}))
            (job/'trial/result.json').write_text(json.dumps({'verifier_result':{'rewards':{'reward':0.0}},'exception_info':None}))
            self.assertTrue(gate_ok(job,0.0,require_ctrf=False)); self.assertFalse(gate_ok(job,1.0,require_ctrf=False))

    def test_gate_requires_verifier_to_have_run_tests(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d); v=job/'trial/attempts/000/verifier'; v.mkdir(parents=True)
            (job/'result.json').write_text(json.dumps({'stats':{'n_completed_trials':1,'n_errored_trials':0}}))
            (job/'trial/result.json').write_text(json.dumps({'verifier_result':{'rewards':{'reward':0.0}},'exception_info':None}))
            self.assertFalse(gate_ok(job,0.0))  # reward 0 with no report: the verifier may have died before pytest
            (v/'ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':0,'failed':0}}}))
            self.assertFalse(gate_ok(job,0.0))  # collection error: nothing ran
            (v/'ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':7,'failed':7}}}))
            self.assertTrue(gate_ok(job,0.0))

    def test_set_digests_refuses_an_image_built_from_another_recipe(self):
        from recover_pool import set_digests
        with tempfile.TemporaryDirectory() as d:
            b=Path(d); sha='ab'*32
            (b/'manifest.json').write_text(json.dumps(dict(version=1,images=[dict(repo='ubuntu2404',
                name='harbor__313e69c036ad__snapshot',dockerfile_sha256=sha,image=None)])))
            good=dict(repo='ubuntu2404',name='harbor__313e69c036ad__snapshot',tag='cf1-'+sha[:12],
                      image='ghcr.io/x/calibforge/ubuntu2404@sha256:'+'c'*64)
            (b/'d.json').write_text(json.dumps(dict(images=[dict(good,tag='cf1-000000000000')])))
            with self.assertRaises(ValueError):
                set_digests(SimpleNamespace(bundle=d,digests=str(b/'d.json')))
            (b/'d.json').write_text(json.dumps(dict(images=[good])))
            set_digests(SimpleNamespace(bundle=d,digests=str(b/'d.json')))
            self.assertEqual(json.loads((b/'manifest.json').read_text())['images'][0]['image'],good['image'])


    async def test_final_check_catches_eviction_after_successful_gate(self):
        from recover_pool import verify_still_present
        service = SimpleNamespace(get=AsyncMock(side_effect=[RuntimeError('404'),snap('active')]))
        rows = [dict(repo='numpy',passed=True,snapshot_id='id'),dict(repo='pillow',passed=True,snapshot_id='old-id')]
        await verify_still_present(service,rows,dict(numpy='n',pillow='p'))
        self.assertTrue(all(not r['passed'] for r in rows))
        self.assertTrue(all(r['stage']=='final_snapshot_check' for r in rows))

    def test_publish_keeps_completed_digest_when_next_build_fails(self):
        import hashlib
        import subprocess
        from unittest.mock import patch
        from recover_pool import publish
        with tempfile.TemporaryDirectory() as d:
            b = Path(d); (b/'dockerfiles').mkdir()
            (b/'tree.tar.gz').write_bytes(b'archive')
            rows = []
            for repo in ['numpy','pillow']:
                recipe = b/'dockerfiles'/repo; recipe.write_text('FROM ubuntu:22.04')
                rows.append(dict(repo=repo,name='harbor__abcdef123456__snapshot',dockerfile='dockerfiles/'+repo,
                    dockerfile_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),image=None))
            manifest = dict(version=1,images=rows,task_tree=dict(file='tree.tar.gz',
                sha256=hashlib.sha256(b'archive').hexdigest()))
            (b/'manifest.json').write_text(json.dumps(manifest))
            args = SimpleNamespace(bundle=d,repos=None,registry='ghcr.io/example/r2egym')
            def fake_run(cmd, **kwargs):
                if 'inspect' in cmd:
                    return
                if 'pillow' in cmd[cmd.index('--file')+1]:
                    raise subprocess.CalledProcessError(1,cmd)
                Path(cmd[cmd.index('--metadata-file')+1]).write_text(json.dumps({'containerimage.digest':'sha256:'+'a'*64}))
            with patch('recover_pool.subprocess.run',side_effect=fake_run):
                with self.assertRaises(subprocess.CalledProcessError):
                    publish(args)
            saved = json.loads((b/'manifest.json').read_text())
            self.assertEqual(saved['images'][0]['image'],'ghcr.io/example/r2egym/numpy@sha256:'+'a'*64)
            self.assertIsNone(saved['images'][1]['image'])
            with patch('recover_pool.subprocess.run',side_effect=fake_run) as runner:
                with self.assertRaises(subprocess.CalledProcessError):
                    publish(args)
                self.assertEqual(runner.call_count,2)  # builder inspection + pillow, never rebuild numpy


if __name__ == '__main__':
    unittest.main()
