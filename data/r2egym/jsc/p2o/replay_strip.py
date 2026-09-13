#!/usr/bin/env python3
"""replay_strip.py: replay REAL prompted trials through HarborTrajectoryRunner._process_trial_result, feature on/off.

For up to N prompted trials (<task>-pA) of a P2O probe session, builds the runner the way the trainer does (real Snowball
tokenizer, full-TITO assembly with behaviour logprobs required, the served per-turn ids from result.json) and processes
each trial with context distillation OFF, ON, and ON+is_eval. Asserts: response ids, loss mask and behaviour logprobs
are identical between OFF and ON (the edit touches the prompt only); ON's prompt is the OFF prompt minus the block
(exactly `removed` tokens, the same count for every trial); ON keeps the served prompt as rollout_prompt_token_ids;
is_eval strips nothing. Runs on a Jupiter login node (CPU) with the snowball-v2 venv and the feature branch on
PYTHONPATH (rsynced under $E/p2o/msr-p2o). Read-only on the traces.
"""
import argparse, glob, json, os, sys
from types import SimpleNamespace
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
S = "/e/fscratch/reformo/lee27/experiments/p2oAc_s0/p2oAc_s0/trace_jobs/eval_sessions/p2oAc_s0_eval_step0"
ap = argparse.ArgumentParser()
ap.add_argument("--session", default=S); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--model", default="/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888")
a = ap.parse_args()

import skyrl_train
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
from skyrl_train.trajectory_runners.context_distillation import ContextDistillationConfig
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
print("skyrl_train from", skyrl_train.__file__)
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)


def runner(config):
    r = object.__new__(HarborTrajectoryRunner)
    r._error_handling_config = ErrorHandlingConfig(enable_error_classification=True, passthrough_exceptions=frozenset({"ContextLengthExceededError"}))
    r._rollout_logprobs_required = True; r._tito_full = None; r._tis_splice = True
    r._reward_shaping_enabled = False; r._collect_rollout_details = True; r._moe_router_replay = False
    r._truncation_penalty = 0.0; r._enable_token_reward_channel = False
    r._chat_template_kwargs = {}; r.custom_chat_template_content = None; r.tokenizer = tok
    r._literal_log_path = None; r._literal_log_store = None
    r._tis_lcs_alert_threshold = 0.005
    r.trajectory_runner_cfg = OmegaConf.create({"sampling_params": {"max_generate_length": 16384}, "max_input_length": 49152})
    r._context_distillation = config
    return r


def load(path):
    d = json.load(open(path))
    ei = d.get("exception_info")
    return SimpleNamespace(
        verifier_result=SimpleNamespace(rewards=d["verifier_result"]["rewards"], stdout=d["verifier_result"].get("stdout", "")),
        exception_info=None if ei is None else SimpleNamespace(**ei),
        agent_result=SimpleNamespace(metadata=d["agent_result"]["metadata"], rollout_details=d["agent_result"]["rollout_details"]),
    )


OFF, ON = runner(ContextDistillationConfig.disabled()), runner(ContextDistillationConfig(enabled=True))
files = sorted(glob.glob(a.session + "/*-pA__*/attempts/*/result.json"))[: a.n]
removed = set(); n_ok = 0
for i, f in enumerate(files):
    tid = TrajectoryID(instance_id=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))), repetition_id=0)
    res = load(f)
    off = OFF._process_trial_result(res, tid)
    on = ON._process_trial_result(res, tid)
    ev = ON._process_trial_result(res, tid, is_eval=True)
    same_resp = list(off.evidence.response_token_ids) == list(on.evidence.response_token_ids)
    same_mask = list(off.loss_mask) == list(on.loss_mask)
    same_lp = off.evidence.behavior_logprobs == on.evidence.behavior_logprobs
    served_kept = list(on.rollout_prompt_token_ids or []) == list(off.evidence.prompt_token_ids)
    eval_untouched = list(ev.evidence.prompt_token_ids) == list(off.evidence.prompt_token_ids) and ev.rollout_prompt_token_ids is None
    d = len(off.evidence.prompt_token_ids) - len(on.evidence.prompt_token_ids)
    prefix_ok = list(on.evidence.prompt_token_ids[:-d - 10]) == list(off.evidence.prompt_token_ids[:-d - 10]) if d > 0 else False
    ok = same_resp and same_mask and same_lp and served_kept and eval_untouched and on.context_edit is not None and on.context_edit.stripped and d > 0 and on.disposition.loss_eligible and off.disposition.loss_eligible
    removed.add(d); n_ok += int(ok)
    print("%s resp=%s mask=%s logprobs=%s served=%s eval=%s status=%s removed=%d loss_eligible=%s/%s n_resp=%d n_mask=%d stop=%s exc=%s" % (
        tid.instance_id, same_resp, same_mask, same_lp, served_kept, eval_untouched, on.context_edit.status if on.context_edit else None, d,
        off.disposition.loss_eligible, on.disposition.loss_eligible, len(on.evidence.response_token_ids), sum(on.loss_mask), on.evidence.stop_reason,
        None if res.exception_info is None else res.exception_info.exception_type))
print("REPLAY: %d/%d trials consistent; removed-token counts %s" % (n_ok, len(files), sorted(removed)))
sys.exit(0 if n_ok == len(files) and len(removed) == 1 else 1)
