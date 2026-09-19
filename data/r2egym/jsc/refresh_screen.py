#!/usr/bin/env python3
"""Two-stage s24 curriculum screen, run on Jupiter. No training submissions.

prepare freezes the eligible pool and builds stage one. run owns a dedicated
JUWELS fleet and bridge, stops each probe at eval completion, and releases by ID.
Null/masked trials remain unresolved, never become negative evidence.
"""
import collections
import csv
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

E = Path('/e/fscratch/reformo/lee27/experiments')
T = E.parent / 'tasks'
C = Path('/e/project1/transfernetx/lee27/code/snowball')
OTA = C.parent / 'OpenThoughts-Agent'
NAME = os.environ.get('REFRESH_NAME', 'refresh_s24_20260915')
D = E / NAME
SRC = 'p2oc_s24_bare'
SOCK = str(Path.home() / '.ssh/cm_juwels/bridge')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-S', SOCK, 'juwels']
PORT = 9930
WORKER = '/p/project1/synthlaion/lee27/fleet/refresh_sparse_worker.py'
GPU_NODES = int(os.environ.get('GPU_NODES', '6'))
ENGINES = int(os.environ.get('ENGINES', '2'))
CONC = int(os.environ.get('CONC', '64'))
CPU_NODES = int(os.environ.get('CPU_NODES', '4'))
WORKERS_PER_NODE = int(os.environ.get('WORKERS_PER_NODE', '16'))
TASKS_PER_SHARD = int(os.environ.get('TASKS_PER_SHARD', '500'))
CANARY_TASKS = int(os.environ.get('CANARY_TASKS', '64'))
GPU_WALL = os.environ.get('GPU_WALL', '10:00:00')
CPU_WALL = os.environ.get('CPU_WALL', '10:30:00')
CANARY_GPU_WALL = os.environ.get('CANARY_GPU_WALL', '03:00:00')
CANARY_CPU_WALL = os.environ.get('CANARY_CPU_WALL', '03:30:00')
# Stop after the canary shard: no further shards, no stage B, no combined tree.
ONLY_CANARY = os.environ.get('ONLY_CANARY') == '1'
# Stop after this many shards submitted by this run (0 = run to completion).
MAX_NEW_SHARDS = int(os.environ.get('MAX_NEW_SHARDS', '1' if ONLY_CANARY else '0'))
# Shards finished by an earlier run (comma-separated names): their results count, their
# stage-A tasks are not screened again.
PRIOR_SHARDS = [s for s in os.environ.get('PRIOR_SHARDS', '').split(',') if s]
FLEET_WAIT_MIN = int(os.environ.get('FLEET_WAIT_MIN', '20'))
NUM_COORDINATORS = int(os.environ.get('NUM_COORDINATORS', '0')) or max(4, CONC // 16)
# Run the eval session on every coordinator (MarinSkyRL a03b2773), not coordinator 0 alone.
EVAL_SPREAD = os.environ.get('EVAL_SPREAD') == '1'
# STAGES=b skips the four-attempt screen: stage B then gives every task not yet mixed enough
# entries to reach eight scored attempts (unscreened tasks get two entries up front).
STAGES = os.environ.get('STAGES', 'ab')
if not set(STAGES) <= set('ab'):   # JSC exports a global $STAGES (software module dir); never let it drive this loop
    STAGES = 'ab'
# ADOPT=<label>:<probe id>:<fleet id> takes over a shard a stopped controller already submitted
# (its tree, config, probe and fleet exist) instead of building and submitting it again.
ADOPT = dict(zip(('label', 'probe', 'fleet'), os.environ['ADOPT'].split(':'))) if os.environ.get('ADOPT') else None
# MODEL=<hf export dir> screens that checkpoint instead of the template's (policy, ref, served name, all together).
MODEL = os.environ.get('MODEL')
# POOL=all: screen every allowlisted TaskTrove task outside val441 (the old 1,003-task train pool plus the rest), for a
# checkpoint whose learnable band is unknown (a new SFT). Default: the never-solved rest only (the s24 refresh).
POOL = os.environ.get('POOL', 'rest')   # rest | all | train
# Extra attempts of one task in a shard get their own tree entry: <task>__rep<i>.
REP = '__rep'
# DAYTONA=1 (2026-09-19): rollouts in Daytona sandboxes instead of the apptainer bridge + JUWELS fleet, and the stage
# machinery becomes rounds: every task gets ROUND_ATTEMPTS attempts; a task stops as soon as it is solved once (learnable);
# a task still unsolved gets another round until it has CAP scored attempts (then keep_zero). Nothing is spent on the
# 8/8 check, which RL's zero-variance masking makes unnecessary. Task dirs come from the gated Daytona tree (v3, 3,020
# tasks); pool tasks outside it are listed in no_daytona.txt. The sbatch gets the 1,024-seat Daytona arm's backend
# (key file at run time, per-node async SOCKS gateway, environment_type daytona + auto_snapshot, no bridge).
DAYTONA = os.environ.get('DAYTONA') == '1'
ROUND_ATTEMPTS = int(os.environ.get('ROUND_ATTEMPTS', '2'))
CAP = int(os.environ.get('CAP', '4'))
MAX_ROUNDS = int(os.environ.get('MAX_ROUNDS', str(-(-CAP // ROUND_ATTEMPTS) + 1)))   # +1 round absorbs infra-lost attempts
SHARES = int(os.environ.get('SHARES', '0')) or NUM_COORDINATORS   # org-wide create budget split; 2 screens at once -> 2x
SUBMIT = os.environ.get('SUBMIT', '1') == '1'   # SUBMIT=0: build round 0 and stop (inspect config + sbatch first)
DTREE = T / 'r2egym-daytona-v3'
ALLOW = T / 'allowlist_r2egym_daytona_v3.txt'
KEYF = '/e/fscratch/reformo/lee27/keys/daytona_eval.env'
SOCKSF = '/e/fscratch/reformo/lee27/keys/socks5_currease.env'
GWPY = '/e/fscratch/reformo/lee27/experiments/daytona_rl_arm1k/async_socks_connect_proxy.py'
GWDEPS = '/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps'
DAYTONA_INFRA = ['DaytonaError', 'DaytonaRateLimitError', 'DaytonaTimeoutError', 'DaytonaNotFoundError', 'DaytonaConflictError',
                 'DaytonaSandboxStopError', 'SandboxBuildFailedError', 'SetupScriptError']

def call(args, **kw):
    return subprocess.check_output(args, text=True, **kw).strip()

def probe_in_queue(jid):
    # squeue returns exit 1 for a job that has fully left the controller; treat that as 'gone',
    # never raise (a scancelled probe's next poll used to crash the whole screen).
    r = subprocess.run(['squeue', '-h', '-j', str(jid), '-o', '%i'], capture_output=True, text=True)
    return bool(r.stdout.strip())

def log(s):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), s, flush=True)

def setarg(args, key, value):
    args[:] = [x for x in args if x.lstrip('+').split('=')[0] != key.lstrip('+')]
    args.append(key + '=' + str(value))

def build(label, tasks):
    name = NAME + '_' + label
    tree = T / name
    tree.mkdir(exist_ok=False)
    copies = collections.Counter()
    for task in tasks:
        entry = f'{task}{REP}{copies[task]}' if copies[task] else task
        copies[task] += 1
        (tree / entry).symlink_to(D / 'tasks' / task)
    out = Path('/e/data1/mmlaion/lee27/experiments') / name
    out.mkdir(exist_ok=False)
    (E / name).symlink_to(out)
    for folder in ('configs', 'sbatch', 'logs', name):
        (out / folder).mkdir()
    (out / '.compacted').touch()
    config = json.loads((E / SRC / 'configs' / (SRC + '_rl_config.json')).read_text().replace(SRC, name))
    args = config['skyrl_hydra_args']
    updates = {
        'data.val_data': json.dumps([str(tree)]),
        'trainer.eval_batch_size': min(32, len(tasks)),
        'generator.eval_n_samples_per_prompt': ROUND_ATTEMPTS if DAYTONA else 4,
        'generator.num_inference_engines': ENGINES,
        '++terminal_bench_config.harbor.n_concurrent_trials': CONC,
        'trajectory_runner.process_pool.num_coordinators': NUM_COORDINATORS,
        'generator.sampling_params.top_p': 1.0,
        'generator.sampling_params.top_k': -1,
        '++terminal_bench_config.harbor.verifier_override_timeout_sec': 2400,
        'trainer.epochs': 0,
        'trainer.seed': (42 + int(label[1:])) if DAYTONA else (42 if label.startswith('a') else 43),
    }
    if DAYTONA:
        masks = []
        for a in args:
            if a.lstrip('+').startswith('terminal_bench_config.harbor.mask_exceptions='):
                masks = json.loads(a.split('=', 1)[1])
        updates['++terminal_bench_config.harbor.environment_type'] = 'daytona'
        updates['++terminal_bench_config.harbor.auto_snapshot'] = 'true'
    if EVAL_SPREAD:
        updates['trajectory_runner.process_pool.eval_spread_coordinators'] = 'true'
    if MODEL:
        updates['trainer.policy.model.path'] = MODEL
        updates['trainer.ref.model.path'] = MODEL
        updates['++generator.engine_init_kwargs.served_model_name'] = os.path.basename(MODEL.rstrip('/'))
    for key, value in updates.items():
        setarg(args, key, value)
    config.update(num_nodes=GPU_NODES, val_data=[str(tree)], val_data_sources=[str(tree)])
    if DAYTONA:
        config['harbor_env'] = 'daytona'
    if MODEL:
        config['model_path'] = MODEL
    # The probe's eval-before-train executes; zero epochs prevents a training step.
    cf = out / 'configs' / (name + '_rl_config.json')
    cf.write_text(json.dumps(config, indent=2))
    sb = (E / SRC / 'sbatch' / (SRC + '_rl.sbatch')).read_text().replace(SRC, name)
    sb = re.sub(r'^#SBATCH --nodes=.*$', f'#SBATCH --nodes={GPU_NODES}', sb, flags=re.M)
    wall = CANARY_GPU_WALL if label == 'a0' else GPU_WALL
    sb = re.sub(r'^#SBATCH --time=.*$', f'#SBATCH --time={wall}', sb, flags=re.M)
    if DAYTONA:
        sb = daytona_sbatch(sb, out)
    else:
        sb = re.sub(r'^DAYTONA_API_KEY_OVERRIDE=.*$', 'DAYTONA_API_KEY_OVERRIDE=""', sb, flags=re.M)
        sb = sb.replace('http://10.128.1.2:9924', 'http://10.128.1.2:9930')
    sbpath = out / 'sbatch' / (name + '_rl.sbatch')
    sbpath.write_text(sb)
    call(['bash', str(C / 'draftify_probe.sh'), name])
    if DAYTONA:
        # draftify_probe.sh resets mask_exceptions from the arm template; add the Daytona infrastructure names after it
        cfg = json.loads(cf.read_text())
        masks = []
        for a in cfg['skyrl_hydra_args']:
            if a.lstrip('+').startswith('terminal_bench_config.harbor.mask_exceptions='):
                masks = json.loads(a.split('=', 1)[1])
        setarg(cfg['skyrl_hydra_args'], '++terminal_bench_config.harbor.mask_exceptions', json.dumps(masks + [m for m in DAYTONA_INFRA if m not in masks], separators=(',', ':')))
        cf.write_text(json.dumps(cfg, indent=2))
    call(['bash', '-n', str(sbpath)])
    checked = json.loads(cf.read_text())['skyrl_hydra_args']
    assert any('speculative_config=' in a for a in checked)
    assert 'trainer.epochs=0' in checked
    assert f'generator.eval_n_samples_per_prompt={ROUND_ATTEMPTS if DAYTONA else 4}' in checked
    if DAYTONA:
        assert '++terminal_bench_config.harbor.environment_type=daytona' in checked
        assert any('DaytonaRateLimitError' in a for a in checked), 'Daytona mask names'
        assert 'dtn_' not in sbpath.read_text() and 'APPTAINER_BRIDGE_URL=http' not in sbpath.read_text()
    return name

def daytona_sbatch(sb, out):
    """The 1,024-seat Daytona arm's backend (build_darm.sh, job 1826380), applied to the screen's sbatch: key read from the
    key file at run time, one async loopback SOCKS gateway per node (proxychains stalls the coordinator loop), Daytona
    reachability check, daytona container runtime, no apptainer bridge. The venv's harbor-marin / marinskyrl-marin already
    carry create pacing, create shares and eval_spread_coordinators, so no code-under-test override."""
    def sub1(old, new):
        nonlocal sb
        assert sb.count(old) == 1, ('anchor count', old[:50], sb.count(old))
        sb = sb.replace(old, new)
    key_block = ('# Daytona screen: key read from the key file at run time (a trailing "# comment" on the line is stripped)\n'
                 'DAYTONA_API_KEY_OVERRIDE="$(grep -m1 -E \'^(export )?DAYTONA_API_KEY=\' %s | cut -d= -f2- | sed -E \'s/[[:space:]]+#.*$//\' | tr -d "\\"\' ")"\n'
                 '[ -n "$DAYTONA_API_KEY_OVERRIDE" ] || { echo "FATAL: no DAYTONA_API_KEY in %s" >&2; exit 96; }') % (KEYF, KEYF)
    sb, n = re.subn(r'^DAYTONA_API_KEY_OVERRIDE=.*$', lambda m: key_block, sb, flags=re.M)
    assert n == 1 and 'dtn_' not in sb, 'key override line'
    sb, n = re.subn(r'^export APPTAINER_BRIDGE_URL=.*$', lambda m: (
        'unset APPTAINER_BRIDGE_URL   # Daytona screen: no apptainer bridge\n'
        '# --- Daytona screen: sandbox backend (harbor paces creates at this org-wide rate; each coordinator takes 1/shares of it);\n'
        '# SOCKS credentials for the per-node gateway started below, sourced here so they precede it ---\n'
        f'export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES={SHARES}\n'
        f'set -a; source {SOCKSF}; set +a'), sb, flags=re.M)
    assert n == 1, 'bridge line'
    assert sb.index(SOCKSF) < sb.index('GW_DIR=') if 'GW_DIR=' in sb else True
    gw = str(out / 'gateway')
    sub1('\n_setup_proxy\n', '\n'
         '# --- Daytona screen: transport = one async loopback gateway per node, no proxychains (research/2026-09-15_daytona_proxy_tokenization.md) ---\n'
         'unset PROXYCHAINS_BIN_OVERRIDE PROXYCHAINS_SOCKS5_PRESET_HOST PROXYCHAINS_SOCKS5_PRESET_PORT PROXYCHAINS_SOCKS5_PRESET_AUTH PROXYCHAINS_CONF_FILE LD_PRELOAD SSH_KEY HTTP_PROXY http_proxy\n'
         f'GW_DIR={gw}; mkdir -p "$GW_DIR"; rm -f "$GW_DIR"/ready-* "$GW_DIR"/*.log\n'
         'srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=1 --gres=none --export=ALL \\\n'
         f'  bash -c \'PYTHONPATH={GWDEPS} exec "$RL_PYTHON" -u {GWPY} --ready "{gw}/ready-$(hostname -s)" > "{gw}/$(hostname -s).log" 2>&1\' &\n'
         'for _i in $(seq 1 120); do [ "$(ls "$GW_DIR"/ready-* 2>/dev/null | wc -l)" -ge "$SLURM_NNODES" ] && break; sleep 1; done\n'
         'if [ "$(ls "$GW_DIR"/ready-* 2>/dev/null | wc -l)" -lt "$SLURM_NNODES" ]; then echo "FATAL: gateway ready on $(ls "$GW_DIR"/ready-* 2>/dev/null | wc -l)/$SLURM_NNODES nodes" >&2; cat "$GW_DIR"/*.log; exit 95; fi\n'
         'export HTTPS_PROXY=http://127.0.0.1:18946 https_proxy=http://127.0.0.1:18946 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1\n'
         'if ! curl -sf -o /dev/null --connect-timeout 20 -x http://127.0.0.1:18946 -H "Authorization: Bearer $DAYTONA_API_KEY_OVERRIDE" https://app.daytona.io/api/api-keys/current; then\n'
         '  echo "FATAL: Daytona API not reachable through the gateway" >&2; exit 98\n'
         'fi\n'
         'echo "daytona screen: gateways ready on $SLURM_NNODES nodes; Daytona API reachable through 127.0.0.1:18946; HTTPS_PROXY set, HTTP_PROXY unset"\n')
    sub1('setup_container_runtime "apptainer" "$WORKDIR" || exit $?\n', 'setup_container_runtime "daytona" "$WORKDIR" || exit $?\n')
    return sb

def prepare():
    D.mkdir(exist_ok=False)
    train = {p.name.split('__r')[0] for p in (T / 'r2egym-tt-v2-train-basecurr-x16').iterdir()}
    held = {p.name for p in (T / 'r2egym-tt-v2-val441').iterdir()}
    rows = list(csv.DictReader(open(E / 'tt_v2_rest.tsv'), delimiter='\t'))
    if POOL == 'all':
        selected = [r for r in rows if r['task'] not in train and r['task'] not in held]
        for r in selected:
            r['pool'] = 'excluded'
        selected.extend({'task': t, 'repo': 'train', 'stratum': 'train', 'pool': 'included'} for t in sorted(train))
    elif POOL == 'train':
        # The current 1,247-task training pool (refresh_s24_20260915r2_train_x16): the old 1,003 plus the 245 added on
        # 2026-09-15. Every task is 'included', so the report says keep_mixed / keep_zero / drop_full per task.
        cur = sorted({p.name.split('__r')[0] for p in (T / 'refresh_s24_20260915r2_train_x16').iterdir()})
        byname = {r['task']: r for r in rows}
        selected = [{'task': t, 'repo': byname.get(t, {}).get('repo', 'train'), 'stratum': 'train' if t in train else 'added', 'pool': 'included'} for t in cur]
    else:
        selected = [r for r in rows if r['stratum'] == 'zero' and r['repo'] not in ('scrapy', 'tornado') and r['task'] not in train]
        for r in selected:
            r['pool'] = 'excluded'
    if os.environ.get('CONFIRM_TRAIN_FULL') == '1':
        dump = E / 'snowball_ttband_ns_c_lr5e7_c/snowball_ttband_ns_c_lr5e7_c/exports/dumped_data/global_step_24_train_rollouts.jsonl'
        groups = collections.defaultdict(list)
        for line in dump.open():
            r = json.loads(line)
            groups[r['uid']].append(r['reward'])
        tasks = sorted({uid.split('__r')[0] for uid, v in groups.items() if len(v) == 8 and all(x == 1 for x in v)})
        assert set(tasks) <= train
        selected.extend({'task': task, 'repo': 'included', 'stratum': 'train_full_candidate', 'pool': 'included'} for task in tasks)
    assert len(train) == 1003
    assert not ({r['task'] for r in selected} & held)
    if DAYTONA:
        allow = {l.strip() for l in ALLOW.read_text().splitlines() if l.strip()}
        missing = sorted(r['task'] for r in selected if r['task'] not in allow)
        (D / 'no_daytona.txt').write_text(''.join(t + '\n' for t in missing))
        selected = [r for r in selected if r['task'] in allow]
        log(f'DAYTONA pool: {len(selected)} tasks in the gated tree, {len(missing)} outside it (no_daytona.txt)')
    (D / 'tasks').mkdir()
    for r in selected:
        task = r['task']
        if DAYTONA:
            assert (DTREE / task / 'tests' / 'required_tests.json').is_file(), task
            shutil.copytree(DTREE / task, D / 'tasks' / task, symlinks=True)
            continue
        source = T / ('r2egym-tt-v2-train-basecurr-x16' if task in train else 'r2egym-tt-v2-rest') / task
        shutil.copytree(source, D / 'tasks' / task)
        p = D / 'tasks' / task / 'task.toml'
        s = p.read_text()
        if 'hidden_paths' not in s:
            s = s.replace('[verifier]', '[verifier]\nhidden_paths = ["/r2e_tests", "/workspace/metadata.json"]')
        p.write_text(s)
    (D / 'manifest.json').write_text(json.dumps(selected, indent=2))
    (D / 'original_train.txt').write_text('\n'.join(sorted(train)) + '\n')
    assert GPU_NODES == 4 + ENGINES
    assert CONC <= 128 * ENGINES   # proven envelope: the 2026-09-15 shards ran conc 512 on 4 engines (was 32*, stale/stricter than reality)
    assert DAYTONA or CONC <= CPU_NODES * WORKERS_PER_NODE
    log(f'PREPARED {len(selected)} tasks; repos {dict(collections.Counter(r["repo"] for r in selected))}; '
        f'geometry gpu={GPU_NODES} engines={ENGINES} conc={CONC} cpu={CPU_NODES}x{WORKERS_PER_NODE} shard={TASKS_PER_SHARD}')

TAIL_BYTES = 4 << 20

def read_outcome(path):
    """Return (task_name, reward, exception_type) from a trial result.json.

    A result file carries the whole trajectory (tens of MB) between the task name at the head
    and the verifier/exception fields at the tail, so read only those two ends. Any layout the
    byte scan does not recognize falls back to a full parse.
    """
    size = path.stat().st_size
    with path.open('rb') as f:
        head = f.read(4096)
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read()
    name = re.search(rb'"task_name":\s*"([^"\\]+)"', head)
    start = tail.rfind(b'"verifier_result":')
    region = tail[start:] if start >= 0 else b''
    split = region.find(b'"exception_info":')
    exc = re.match(rb'"exception_info":\s*(?:null|\{\s*"exception_type":\s*"([^"\\]+)")', region[split:]) if split >= 0 else None
    if name and exc and (size <= TAIL_BYTES or start > 0):
        verifier = region[:split]
        reward = None
        if not re.match(rb'"verifier_result":\s*null', verifier):
            found = re.search(rb'"rewards":\s*\{\s*"reward":\s*(-?[0-9.eE+-]+)\s*[,}]', verifier)
            if not found:
                return full_outcome(path)
            reward = float(found.group(1))
        return name.group(1).decode(), reward, exc.group(1).decode() if exc.group(1) else None
    return full_outcome(path)

def full_outcome(path):
    r = json.loads(path.read_text())
    reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    return r['task_name'], reward, (r.get('exception_info') or {}).get('exception_type')

def outcomes(name):
    cfg = json.loads((E / name / 'configs' / (name + '_rl_config.json')).read_text())
    masks = set()
    for a in cfg['skyrl_hydra_args']:
        if a.lstrip('+').startswith('terminal_bench_config.harbor.mask_exceptions='):
            masks = set(json.loads(a.split('=', 1)[1]))
    root = E / name / name / 'trace_jobs/eval_sessions' / (name + '_eval_step0')
    values = collections.defaultdict(list)
    total = invalid = 0
    if not root.exists():
        return values, total, invalid
    for p in sorted(root.glob('*/result.json')):
        try:
            task, reward, ex = read_outcome(p)
        except (ValueError, OSError, KeyError):
            continue
        total += 1
        if reward not in (0, 1) or ex in masks:
            invalid += 1
            continue
        values[task.split(REP)[0]].append(int(reward))
    return values, total, invalid

def classify(values):
    if DAYTONA:
        # rounds: solved at least once = learnable (may include always-solved; RL masks zero-variance groups itself)
        if sum(values) >= 1:
            return 'candidate'
        return 'exclude_zero' if len(values) >= CAP else 'unresolved'
    if 0 in values and 1 in values:
        return 'candidate'
    if len(values) >= 8:
        return 'exclude_zero' if sum(values) == 0 else 'exclude_full'
    return 'unresolved'

def report(names):
    rows = json.loads((D / 'manifest.json').read_text())
    allvals = collections.defaultdict(list)
    for name in names:
        vals, _, _ = outcomes(name)
        for task, vs in vals.items():
            assert len(vs) <= 8, (name, task, len(vs))
            allvals[task].extend(vs)
    candidates = []
    drops = []
    counts = collections.Counter()
    with (D / 'selection.tsv').open('w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['task', 'repo', 'pool', 'scored', 'successes', 'status'])
        for r in rows:
            vs = allvals[r['task']]
            status = classify(vs)
            if r.get('pool') == 'included':
                status = {'candidate': 'keep_mixed', 'exclude_zero': 'keep_zero', 'exclude_full': 'drop_full', 'unresolved': 'unresolved'}[status]
            counts[status] += 1
            w.writerow([r['task'], r['repo'], r.get('pool', 'excluded'), len(vs), sum(vs), status])
            if status == 'candidate':
                candidates.append(r['task'])
            elif status == 'drop_full':
                drops.append(r['task'])
    (D / 'new_tasks.txt').write_text(''.join(t + '\n' for t in sorted(candidates)))
    (D / 'drop_tasks.txt').write_text(''.join(t + '\n' for t in sorted(drops)))
    (D / 'summary.json').write_text(json.dumps(dict(counts), indent=2))
    log(f'SELECTION {dict(counts)}')
    return allvals

def status():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/status', timeout=5) as r:
        return json.load(r)

def chunks(items, canary=False):
    if canary and CANARY_TASKS and len(items) > CANARY_TASKS:
        return [items[:CANARY_TASKS]] + [items[i:i + TASKS_PER_SHARD] for i in range(CANARY_TASKS, len(items), TASKS_PER_SHARD)]
    return [items[i:i + TASKS_PER_SHARD] for i in range(0, len(items), TASKS_PER_SHARD)]

def stage_b_entries(manifest, values):
    """Top every unresolved task up to eight scored attempts, four per tree entry.

    Attempts lost to infrastructure earlier are rerun here instead of leaving the task short.
    """
    todo = []
    for task in manifest:
        need = 8 - len(values[task])
        if classify(values[task]) != 'candidate' and need > 0:
            todo.extend([task] * -(-need // 4))
    return todo

def round_entries(manifest, values):
    """Tasks not solved yet and short of CAP scored attempts: enough tree entries (ROUND_ATTEMPTS each) to reach CAP."""
    todo = []
    for task in manifest:
        vs = values[task]
        if sum(vs) == 0 and len(vs) < CAP:
            todo.extend([task] * -(-(CAP - len(vs)) // ROUND_ATTEMPTS))
    return todo

def wait_probe(name, probe, expected):
    while True:
        time.sleep(30)
        vals, n, invalid = outcomes(name)
        logs = list((E / name / 'logs').glob('*.out'))
        text = logs[0].read_text(errors='replace') if logs else ''
        if n >= 30 and invalid / n > .30:
            raise RuntimeError(f'infra tripwire {invalid}/{n}')
        if 'WANDB_MIRROR kind=eval' in text:
            call(['scancel', probe])
            log(f'EVAL DONE {probe}; {n} results, {invalid} invalid')
            break
        if not probe_in_queue(probe):
            log(f'JOB EXIT {probe}; {n} results, {invalid} invalid')
            if n < .9 * expected:
                raise RuntimeError('probe exited before complete eval')
            break
    while probe_in_queue(probe):
        time.sleep(10)

def run_daytona():
    probe = None
    names = list(PRIOR_SHARDS)
    manifest = [r['task'] for r in json.loads((D / 'manifest.json').read_text())]
    try:
        for rnd in range(MAX_ROUNDS):
            label = f'r{rnd}'
            name = NAME + '_' + label
            if name in names:
                continue
            todo = manifest if rnd == 0 else round_entries(manifest, report(names))
            if not todo:
                break
            if (D / ('job_' + label)).exists() and (E / name).exists():
                # a restarted controller adopts the round an earlier one submitted (e.g. to change SCREEN_ACCOUNT for later rounds)
                probe = (D / ('job_' + label)).read_text().strip()
                names.append(name)
                log(f'ADOPT {name} {probe} ({"in queue" if probe_in_queue(probe) else "finished"})')
                if probe_in_queue(probe):
                    wait_probe(name, probe, len(todo) * ROUND_ATTEMPTS)
                probe = None
                continue
            log(f'ROUND {rnd}: {len(set(todo))} tasks, {len(todo)} tree entries x {ROUND_ATTEMPTS} attempts')
            build(label, todo)
            if not SUBMIT:
                log(f'BUILT {name}, not submitted (SUBMIT=0): {E / name / "configs"} {E / name / "sbatch"}')
                return
            probe = call(['sbatch', '--parsable', '-A', os.environ.get('SCREEN_ACCOUNT', 'laionize'), str(E / name / 'sbatch' / (name + '_rl.sbatch'))], cwd=OTA, env={**os.environ, 'DCFT': str(OTA)})
            assert probe.isdigit(), probe
            (D / ('job_' + label)).write_text(probe)
            names.append(name)
            log(f'PROBE {name} {probe}')
            wait_probe(name, probe, len(todo) * ROUND_ATTEMPTS)
            probe = None
        vals = report(names)
        learnable = sorted(t for t in manifest if classify(vals[t]) == 'candidate')
        (D / 'learnable_tasks.txt').write_text(''.join(t + '\n' for t in learnable))
        tree = T / (NAME + '_learnable_x16')
        tree.mkdir(exist_ok=False)
        for task in learnable:
            for i in range(16):
                (tree / (task if i == 0 else f'{task}__r{i}')).symlink_to(DTREE / task)
        log(f'LEARNABLE TREE {tree}: {len(learnable)} tasks x16')
        (D / 'DONE').touch()
    except Exception as e:
        log(f'FAILED {type(e).__name__}: {e}')
        (D / 'FAILED').write_text(str(e))
        if names:
            report(names)
        raise
    finally:
        if probe:
            subprocess.run(['scancel', probe])

def run():
    if DAYTONA:
        return run_daytona()
    fleet = None
    probe = None
    names = list(PRIOR_SHARDS)
    new_shards = 0
    try:
        # Dedicated port; do not restart any other session's bridge.
        server = C.parent / 'harbor-marin/src/harbor/environments/apptainer/server.py'
        # '=' makes tmux match the session name exactly, never by prefix.
        if subprocess.run(['tmux', 'has-session', '-t', '=' + NAME + '_bridge']).returncode == 0:
            # A bridge left by an earlier run (an adopted shard's sandboxes live on it): keep it.
            log(f'REUSE bridge {NAME}_bridge')
        else:
            call(['tmux', 'new-session', '-d', '-s', NAME + '_bridge',
                  f'BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_LISTEN_BACKLOG=8192 BRIDGE_MAX_HANDLER_THREADS=3000 BRIDGE_STALE_READY_SEC=2400 python3 {server} --port {PORT} --host 0.0.0.0 >> {D}/bridge.log 2>&1'])
        for _ in range(12):
            time.sleep(5)
            try:
                status()
                break
            except Exception:
                pass
        status()
        ip = call(SSH + ['getent hosts jwlogin03i']).split()[0]
        # A forward left by an earlier run on this port is reused; the fleet readiness check
        # below proves the path end to end.
        subprocess.run(['ssh', '-S', SOCK, '-O', 'forward', '-R', f'{ip}:{PORT}:10.128.1.2:{PORT}', 'juwels'])
        manifest = [r['task'] for r in json.loads((D / 'manifest.json').read_text())]
        for stage in STAGES:
            if stage == 'a':
                screened = set()
                for prior in PRIOR_SHARDS:
                    if prior.startswith(NAME + '_a'):
                        screened |= {p.name.split(REP)[0] for p in (T / prior).iterdir()}
                todo = [task for task in manifest if task not in screened]
            else:
                todo = stage_b_entries(manifest, report(names))
                if not todo:
                    break
                log(f'STAGE B {len(set(todo))} tasks, {len(todo)} tree entries')
            prefix = f'{NAME}_{stage}'
            first = sum(1 for p in E.iterdir() if re.fullmatch(re.escape(prefix) + r'\d+', p.name))
            if ADOPT and ADOPT['label'][0] == stage:
                first = int(ADOPT['label'][1:])
            for shard, shard_tasks in enumerate(chunks(todo, canary=(stage == 'a' and not PRIOR_SHARDS)), start=first):
                label = f'{stage}{shard}'
                name = NAME + '_' + label
                if ADOPT and label == ADOPT['label']:
                    tree = sorted(p.name for p in (T / name).iterdir())
                    copies = collections.Counter()
                    expected = []
                    for task in shard_tasks:
                        expected.append(f'{task}{REP}{copies[task]}' if copies[task] else task)
                        copies[task] += 1
                    assert tree == sorted(expected), f'adopted shard {label} does not match its recomputed tasks'
                    fleet, probe = ADOPT['fleet'], ADOPT['probe']
                    names.append(name)
                    log(f'ADOPT {name} probe {probe} fleet {fleet}')
                else:
                    build(label, shard_tasks)
                    fleet_wall = CANARY_CPU_WALL if label == 'a0' else CPU_WALL
                    fleet = call(SSH + [f'cd /p/project1/synthlaion/lee27/fleet && sbatch --parsable --nodes={CPU_NODES} --time={fleet_wall} --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKER_SCRIPT={WORKER},WORKERS_PER_NODE={WORKERS_PER_NODE},STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT={PORT},MAX_CHAIN=0 -J {NAME}_{label}_fleet juwels_workers.sbatch'])
                    assert fleet.isdigit(), fleet
                    (D / ('fleet_' + label)).write_text(fleet)
                    log(f'FLEET {label} {fleet}')
                    for _ in range(FLEET_WAIT_MIN * 6):
                        time.sleep(10)
                        n = call(SSH + [f'grep -c "starting {WORKERS_PER_NODE} workers" /p/scratch/synthlaion/lee27/dc_agent_eval/logs/apptainer_workers_juwels_{fleet}.out || true'])
                        if n.isdigit() and int(n) == CPU_NODES and status().get('workers_alive'):
                            break
                    else:
                        raise RuntimeError(f'fleet startup timeout {fleet}')
                    probe = call(['sbatch', '--parsable', '-A', os.environ.get('SCREEN_ACCOUNT', 'laionize'), str(E / name / 'sbatch' / (name + '_rl.sbatch'))], cwd=OTA, env={**os.environ, 'DCFT': str(OTA)})   # reformo default fairshare stalls 8-node probes; laionize starts in ~1h
                    assert probe.isdigit(), probe
                    (D / ('job_' + label)).write_text(probe)
                    names.append(name)
                    log(f'PROBE {name} {probe}')
                while True:
                    time.sleep(30)
                    vals, n, invalid = outcomes(name)
                    logs = list((E / name / 'logs').glob('*.out'))
                    text = logs[0].read_text(errors='replace') if logs else ''
                    if n >= 30 and invalid / n > .30:
                        raise RuntimeError(f'infra tripwire {invalid}/{n}')
                    if 'WANDB_MIRROR kind=eval' in text:
                        call(['scancel', probe])
                        log(f'EVAL DONE {probe}; {n} results, {invalid} invalid')
                        break
                    if not probe_in_queue(probe):
                        log(f'JOB EXIT {probe}; {n} results, {invalid} invalid')
                        # Missing results stay unresolved; only a mostly empty shard stops the run.
                        if n < .9 * len(shard_tasks) * 4 and not MAX_NEW_SHARDS:
                            raise RuntimeError('probe exited before complete eval')
                        break
                while probe_in_queue(probe):
                    time.sleep(10)
                probe = None
                subprocess.run(SSH + [f'scancel {fleet}'])
                log(f'RELEASE fleet {fleet}')
                fleet = None
                new_shards += 1
                if MAX_NEW_SHARDS and new_shards >= MAX_NEW_SHARDS:
                    report(names)
                    log(f'STOPPED after {new_shards} new shard(s); shards so far: {",".join(names)}')
                    (D / ('CANARY_DONE' if ONLY_CANARY else 'PARTIAL_DONE')).touch()
                    return
        report(names)
        # Materialize the proposed next curriculum without changing the old tree.
        combined = T / (NAME + '_train_x16')
        combined.mkdir(exist_ok=False)
        original = T / 'r2egym-tt-v2-train-basecurr-x16'
        drops = set((D / 'drop_tasks.txt').read_text().splitlines())
        for p in original.iterdir():
            if p.name.split('__r')[0] not in drops:
                (combined / p.name).symlink_to(p.resolve())
        for task in (D / 'new_tasks.txt').read_text().splitlines():
            for i in range(16):
                entry = task if i == 0 else f'{task}__r{i}'
                (combined / entry).symlink_to(D / 'tasks' / task)
        log(f'COMBINED TREE {combined}')
        (D / 'DONE').touch()
    except Exception as e:
        log(f'FAILED {type(e).__name__}: {e}')
        (D / 'FAILED').write_text(str(e))
        if names:
            report(names)
        raise
    finally:
        if probe:
            subprocess.run(['scancel', probe])
        if fleet:
            subprocess.run(SSH + [f'scancel {fleet}'])
            log(f'RELEASE fleet {fleet}')

if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'report':
        if DAYTONA:
            report(sorted(p.name for p in E.iterdir() if re.fullmatch(re.escape(NAME) + r'_r\d+', p.name)))
        else:
            report([NAME + '_' + s for s in ('a', 'b') if (E / (NAME + '_' + s)).exists()])
