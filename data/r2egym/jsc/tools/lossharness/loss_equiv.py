#!/usr/bin/env python
"""Offline, CPU-only policy-loss equivalence harness for two MarinSkyRL revisions.

Runs ONE deterministic synthetic batch through a revision's own policy-loss code and
prints loss / gradient / metrics as JSON. Diff two runs to see exactly what a repo
migration changed in the objective.

    python loss_equiv.py --repo <path-to-a-MarinSkyRL-checkout> [--out out.json]

`<path>` is the checkout ROOT (the dir containing `skyrl-train/`), or the
`skyrl-train/` dir itself. Nothing is imported from the ambient install: the
revision's `skyrl-train` is put at sys.path[0].

No GPU, no model weights, no ray. The loss functions take LOG-PROBS, not logits, so
the batch is vocab-free.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Production recipe (frozen arm config:
#   ~/Documents/experiments/migration_baseline_2026-09-09/arm_rl_config.json)
#
#   trainer.algorithm.advantage_estimator=rloo_n
#   trainer.algorithm.use_kl_loss=true
#   trainer.algorithm.kl_loss_coef=0.01
#   trainer.algorithm.eps_clip_low=0.2  eps_clip_high=0.2
#   trainer.algorithm.loss_reduction=sequence_mean
#   trainer.algorithm.use_tis=true      tis_imp_ratio_cap=2.0
#   generator.sampling_params.temperature=1.0
#   generator.max_input_length=49152    sampling_params.max_generate_length=16384
#     -> algorithm.max_seq_len = 65536  (utils.py: max_input_length + max_generate_length)
#   policy_loss_type / kl_estimator_type / think_token_weight / use_entropy_loss
#     are left at the ppo_base_config.yaml defaults: regular / k3 / 1.0 / false
# ---------------------------------------------------------------------------
PROD_RECIPE: dict[str, Any] = {
    "policy_loss_type": "regular",
    "loss_reduction": "sequence_mean",
    "use_tis": True,
    "tis_imp_ratio_cap": 2.0,
    "use_kl_loss": True,
    "kl_loss_coef": 0.01,
    "kl_estimator_type": "k3",
    "eps_clip_low": 0.2,
    "eps_clip_high": 0.2,
    "clip_ratio_c": 3.0,
    "use_entropy_loss": False,
    "entropy_loss_coef": 0.01,
    "think_token_weight": 1.0,
    "max_seq_len": 65536,
}

# variant name -> (overrides on top of PROD_RECIPE, needs_global_denom)
VARIANTS: dict[str, dict[str, Any]] = {
    # --- the production objective ---
    "prod": {},
    # --- (a) TIS on vs off ---
    "tis_off": {"use_tis": False},
    "tis_cap_inf": {"tis_imp_ratio_cap": 1e9},
    # --- (b) temperature. NOT a loss knob on either side: the loss takes logprobs and
    #         `generator.sampling_params.temperature` is consumed by model_wrapper when
    #         it scales the LOGITS. Injecting it into the algorithm config must be a
    #         bit-identical no-op; that is the assertion this variant makes. ---
    "temperature_1p0_injected": {"temperature": 1.0},
    "temperature_0p7_injected": {"temperature": 0.7},
    # --- (c) KL coefficient ---
    "kl_coef_0": {"kl_loss_coef": 0.0},
    "kl_loss_off": {"use_kl_loss": False},
    "kl_estimator_k1": {"kl_estimator_type": "k1"},
    # --- reductions ---
    "token_mean": {"loss_reduction": "token_mean"},
    "seq_mean_token_sum_norm": {"loss_reduction": "seq_mean_token_sum_norm"},
    "seq_mean_token_sum_norm_global": {"loss_reduction": "seq_mean_token_sum_norm_global"},
    "prompt_mean": {"loss_reduction": "prompt_mean"},  # OLD-fork only
    # --- other objectives on the same batch ---
    "dual_clip": {"policy_loss_type": "dual_clip"},
    "behavior_clip": {"policy_loss_type": "behavior_clip", "use_tis": False},
    "dppo": {"policy_loss_type": "dppo", "use_tis": False},  # OLD-fork only
    # --- entropy bonus (off in production, exercises the aux-term gradient path) ---
    "entropy_on": {"use_entropy_loss": True, "entropy_loss_coef": 0.01},
    # --- think-token reweighting (off in production) ---
    "think_weight_0p25": {"think_token_weight": 0.25},
}

GLOBAL_LOSS_DENOM = 512.0
ACCUMULATION_STEPS = 4
BATCH, TOKENS = 8, 64

# Per-sequence loss-mask lengths. Row 2 is fully masked (an excluded / zero-length
# sample); row 5 is fully masked as a `mask_length_stops` length-stopped sample.
MASK_LENGTHS = [64, 40, 0, 33, 64, 0, 12, 55]
# Per-sequence advantage, both signs, one zero-advantage row (row 6) as an
# RLOO zero-variance group would produce.
SEQ_ADVANTAGES = [1.2, -0.8, 0.5, -1.5, 0.3, -0.2, 0.0, 2.0]
# Per-sequence log-prob drift scale. Rows 0-3 are the near-on-policy regime; rows 4-7
# drift far enough that the eps_clip=0.2 bounds and the tis_imp_ratio_cap=2.0 actually
# fire, so no branch of the objective is left untested by the batch.
OLD_DRIFT = [0.05, 0.05, 0.05, 0.05, 0.30, 0.30, 0.30, 0.30]
ROLLOUT_DRIFT = [0.08, 0.08, 0.08, 0.08, 0.45, 0.45, 0.45, 0.45]


def resolve_skyrl_train(repo: str) -> str:
    repo = os.path.abspath(os.path.expanduser(repo))
    cand = os.path.join(repo, "skyrl-train")
    if os.path.isdir(os.path.join(cand, "skyrl_train")):
        return cand
    if os.path.isdir(os.path.join(repo, "skyrl_train")):
        return repo
    raise SystemExit(f"no skyrl_train package under {repo}")


def git_describe(path: str) -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", path, *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {"sha": run("rev-parse", "HEAD"), "subject": run("log", "-1", "--format=%s")}


def build_batch(torch):
    """One fixed batch. Deterministic given torch.manual_seed(0) and CPU float32."""
    torch.manual_seed(0)
    B, T = BATCH, TOKENS

    # Current-policy logprobs: the differentiable leaf every gradient is taken w.r.t.
    log_probs = (torch.randn(B, T) * 0.5 - 1.0).requires_grad_(True)

    with torch.no_grad():
        base = log_probs.detach()
        old_scale = torch.tensor(OLD_DRIFT).unsqueeze(1)
        rollout_scale = torch.tensor(ROLLOUT_DRIFT).unsqueeze(1)
        # "old" logprobs = the trainer's recompute of the same tokens (staleness drift).
        old_log_probs = base + torch.randn(B, T) * old_scale
        # rollout/behavior logprobs = what the inference engine reported. Differs from
        # `old` by the train/inference numerics gap, so the TIS ratio is non-trivial.
        rollout_logprobs = old_log_probs + torch.randn(B, T) * rollout_scale
        # reference-policy logprobs for the KL term.
        base_log_probs = base + torch.randn(B, T) * 0.12

        loss_mask = torch.zeros(B, T)
        for row, n in enumerate(MASK_LENGTHS):
            if n:
                loss_mask[row, :n] = 1.0

        advantages = torch.tensor(SEQ_ADVANTAGES).unsqueeze(1).expand(B, T).contiguous()

        # THINK/ACT span tags for the think_token_weight path: first half THINK (1).
        response_span_tags = torch.zeros(B, T, dtype=torch.long)
        response_span_tags[:, : T // 2] = 1

    # Entropy is a differentiable function of the model output in production, so keep
    # it attached to the leaf; otherwise use_entropy_loss would show a null gradient.
    token_entropy = -log_probs * 0.5 + 1.0

    return {
        "action_log_probs": log_probs,
        "old_action_log_probs": old_log_probs,
        "base_action_log_probs": base_log_probs,
        "rollout_logprobs": rollout_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "response_span_tags": response_span_tags,
        "token_entropy": token_entropy,
        "leaf": log_probs,
    }


def build_config(skyrl_train_dir: str, overrides: dict[str, Any]):
    """Build the config object the worker passes to the loss: `cfg.trainer.algorithm`.

    Loaded from the REVISION'S OWN ppo_base_config.yaml so that defaults which differ
    between revisions are picked up rather than assumed. The subtree is returned from
    the full tree so `${...}` interpolations still resolve.
    """
    from omegaconf import OmegaConf

    yaml_path = os.path.join(skyrl_train_dir, "skyrl_train", "config", "ppo_base_config.yaml")
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.set_struct(cfg, False)
    algo = cfg.trainer.algorithm
    for key, value in {**PROD_RECIPE, **overrides}.items():
        algo[key] = value
    return algo


def adapt(skyrl_train_dir: str) -> dict[str, Any]:
    """Per-revision adapter: resolve the loss entry points that exist on THIS revision."""
    sys.path.insert(0, skyrl_train_dir)
    for mod in [m for m in sys.modules if m.split(".")[0] in ("skyrl_train", "skyrl_gym")]:
        del sys.modules[mod]

    pl = importlib.import_module("skyrl_train.utils.policy_losses")
    pm = importlib.import_module("skyrl_train.utils.policy_math")
    lr = importlib.import_module("skyrl_train.utils.loss_reduction")
    reg = importlib.import_module("skyrl_train.utils.algorithm_registry")

    if pl.__file__ and not pl.__file__.startswith(skyrl_train_dir):
        raise SystemExit(f"import leaked to {pl.__file__}, expected under {skyrl_train_dir}")

    # KL entry point was renamed by upstream #517 (ours: approx_kl / theirs:
    # differentiable_approx_kl). Same body; both are the DIFFERENTIABLE one.
    kl_name = "approx_kl" if hasattr(pm, "approx_kl") else "differentiable_approx_kl"

    # policy_loss_type -> function. The registry's .get() is ray-actor backed, so map
    # by the module-level registration decorators instead (identical dispatch, no ray).
    loss_fns = {
        "regular": pl.ppo_policy_loss,
        "dual_clip": pl.ppo_policy_loss,
        "behavior_clip": pl.behavior_clipped_policy_loss,
        "sapo": pl.sapo_policy_loss,
        "gspo": pl.gspo_policy_loss,
        "cispo": pl.compute_policy_loss_cispo,
    }
    if hasattr(pl, "dppo_policy_loss"):
        loss_fns["dppo"] = pl.dppo_policy_loss

    return {
        "policy_losses": pl,
        "policy_math": pm,
        "loss_reduction_mod": lr,
        "registry": reg,
        "kl_fn_name": kl_name,
        "kl_fn": getattr(pm, kl_name),
        "loss_fns": loss_fns,
        "supported_reductions": list(lr.SUPPORTED_LOSS_REDUCTIONS),
        "policy_loss_types": sorted(t.value for t in reg.PolicyLossType),
        "has_prescaled_sum_set": hasattr(lr, "PRESCALED_SUM_LOSS_REDUCTIONS"),
        "has_rollout_logprob_set": hasattr(reg, "ROLLOUT_LOGPROB_POLICY_LOSSES"),
        "signatures": {
            "compute_policy_objective": str(inspect.signature(pl.compute_policy_objective)),
            "ppo_policy_loss": str(inspect.signature(pl.ppo_policy_loss)),
            "behavior_clipped_policy_loss": str(inspect.signature(pl.behavior_clipped_policy_loss)),
            "reduce_loss": str(inspect.signature(lr.reduce_loss)),
            kl_name: str(inspect.signature(getattr(pm, kl_name))),
            "_compute_policy_auxiliary_terms": str(inspect.signature(pl._compute_policy_auxiliary_terms)),
            "_scale_policy_objective": str(inspect.signature(pl._scale_policy_objective)),
        },
    }


def run_variant(torch, api: dict[str, Any], skyrl_train_dir: str, name: str, overrides: dict[str, Any]) -> dict:
    pl = api["policy_losses"]
    batch = build_batch(torch)
    leaf = batch.pop("leaf")

    try:
        cfg = build_config(skyrl_train_dir, overrides)
        loss_type = str(cfg.policy_loss_type)
        if loss_type not in api["loss_fns"]:
            return {"variant": name, "status": "unsupported", "reason": f"policy_loss_type {loss_type!r} not in this revision"}
        needs_denom = str(cfg.loss_reduction) == "seq_mean_token_sum_norm_global"
        objective = pl.compute_policy_objective(
            **batch,
            config=cfg,
            policy_loss_fn=api["loss_fns"][loss_type],
            accumulation_steps=ACCUMULATION_STEPS,
            scaling=pl.LossScaling.CALLER,
            global_loss_denom=GLOBAL_LOSS_DENOM if needs_denom else None,
        )
        objective.optimization_loss.backward()
    except Exception as exc:  # noqa: BLE001 - a variant a revision cannot express is a RESULT
        return {"variant": name, "status": "error", "error_type": type(exc).__name__, "error": str(exc)[:400]}

    grad = leaf.grad
    flat = grad.reshape(-1)
    return {
        "variant": name,
        "status": "ok",
        "optimization_loss": float(objective.optimization_loss.item()),
        "unscaled_loss": float(objective.unscaled_loss.item()),
        "policy_loss": float(objective.policy_loss.item()),
        "entropy": float(objective.entropy.item()),
        "kl_loss": float(objective.kl_loss.item()),
        "grad_norm": float(flat.norm().item()),
        "grad_sum": float(flat.sum().item()),
        "grad_nonzero_rows": int((grad.abs().sum(dim=-1) > 0).sum().item()),
        "grad_first10": [float(v) for v in flat[:10].tolist()],
        "metrics": {k: float(v) for k, v in sorted(objective.metrics.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="MarinSkyRL checkout (root or its skyrl-train/)")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    ap.add_argument("--variants", help="comma-separated subset of variant names")
    args = ap.parse_args()

    skyrl_train_dir = resolve_skyrl_train(args.repo)
    api = adapt(skyrl_train_dir)
    import torch

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    names = args.variants.split(",") if args.variants else list(VARIANTS)
    report = {
        "repo": os.path.abspath(os.path.expanduser(args.repo)),
        "skyrl_train_dir": skyrl_train_dir,
        "git": git_describe(skyrl_train_dir),
        "torch": torch.__version__,
        "batch": {
            "B": BATCH,
            "T": TOKENS,
            "mask_lengths": MASK_LENGTHS,
            "seq_advantages": SEQ_ADVANTAGES,
            "old_drift": OLD_DRIFT,
            "rollout_drift": ROLLOUT_DRIFT,
            "seed": 0,
            "accumulation_steps": ACCUMULATION_STEPS,
            "global_loss_denom": GLOBAL_LOSS_DENOM,
        },
        "recipe": PROD_RECIPE,
        "api": {
            "kl_fn_name": api["kl_fn_name"],
            "policy_loss_types": api["policy_loss_types"],
            "supported_reductions": api["supported_reductions"],
            "has_PRESCALED_SUM_LOSS_REDUCTIONS": api["has_prescaled_sum_set"],
            "has_ROLLOUT_LOGPROB_POLICY_LOSSES": api["has_rollout_logprob_set"],
            "signatures": api["signatures"],
        },
        "results": [run_variant(torch, api, skyrl_train_dir, n, VARIANTS[n]) for n in names],
    }
    text = json.dumps(report, indent=2, sort_keys=False)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
