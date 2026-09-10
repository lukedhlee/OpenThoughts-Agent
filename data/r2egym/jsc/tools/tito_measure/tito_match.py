#!/usr/bin/env python3
"""Measure whether TITO holds: are the token ids the trainer trains on the exact
ids vLLM sampled, across a multi-turn terminus-2 rollout?

Read-only. Point it at a harbor trials_dir (or any directory of per-trial
``result.json`` files) and it reports, per rollout and in aggregate:

  * turns, sampled completion tokens, tokens the trainer would train on
  * longest common prefix per turn against the previous served stream
  * number of divergent turns and where the divergence lands
  * the exact-match fraction over rollouts

Two tiers:

  TIER A (default, no dependencies beyond the stdlib): replays MarinSkyRL's
  ``_assemble_response_ids_tito_full`` invariant checks over harbor's recorded
  per-turn ``prompt_token_ids`` / ``completion_token_ids``. Pure integer
  arithmetic, no tokenizer needed. This decides TITO: if every check passes, the
  trainer assembles the training sequence FROM the served ids, so trained ==
  sampled by construction.

  TIER B (``--marinskyrl <checkout>`` + ``--tokenizer``): imports the real
  ``trajectory_processing`` module with heavy deps stubbed and runs the actual
  assembly twice, with ``tito_full`` on and off, then compares the loss_mask==1
  ids against the sampled ids directly. Use it to confirm Tier A and to quantify
  what the #520 re-tokenizing fallback actually trains on.

Usage:
  python3 tito_match.py --trials-dir <dir> [--limit N] [--json out.json]
  python3 tito_match.py --trials-dir <dir> --tokenizer <dir-or-tokenizer.json>
  python3 tito_match.py --trials-dir <dir> --tokenizer <t> \
      --marinskyrl <path-to-a-MarinSkyRL-checkout>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

# --------------------------------------------------------------------------
# Decline reasons, in the ORDER MarinSkyRL checks them
# (skyrl_train/trajectory_runners/trajectory_processing.py::_assemble_response_ids_tito_full).
# Names match TitoFullDeclineReason so the numbers here line up with the
# generate/tis/tito_full/decline/<reason> metrics the trainer emits.
# --------------------------------------------------------------------------
MISSING_STREAMS = "missing_streams"
EMPTY_STREAMS = "empty_streams"
TURN_COUNT_MISMATCH = "turn_count_mismatch"
ASSISTANT_MESSAGE_COUNT_MISMATCH = "assistant_message_count_mismatch"
MALFORMED_TURN_STREAM = "malformed_turn_stream"
PREFIX_MISMATCH = "prefix_mismatch"
INITIAL_PROMPT_TOO_SHORT = "initial_prompt_too_short"
GENERATION_PROMPT_MISMATCH = "generation_prompt_mismatch"
COMPLETION_REGION_MISMATCH = "completion_region_mismatch"

REASONS = [
    MISSING_STREAMS,
    EMPTY_STREAMS,
    TURN_COUNT_MISMATCH,
    ASSISTANT_MESSAGE_COUNT_MISMATCH,
    MALFORMED_TURN_STREAM,
    PREFIX_MISMATCH,
    INITIAL_PROMPT_TOO_SHORT,
    GENERATION_PROMPT_MISMATCH,
    COMPLETION_REGION_MISMATCH,
]


# --------------------------------------------------------------------------
# Loading harbor artifacts
# --------------------------------------------------------------------------
@dataclass
class Rollout:
    trial: str
    path: str
    prompt_token_ids: Optional[List[List[int]]]
    completion_token_ids: Optional[List[List[int]]]
    logprobs: Optional[List[List[Any]]]
    messages: Optional[List[Dict[str, Any]]]
    n_assistant: Optional[int]
    exception_type: Optional[str] = None
    reward: Optional[float] = None
    summarization_count: Optional[int] = None
    n_rollout_details: int = 0


_CANDIDATE_NAMES = ("result.json", "trajectory.json")


def find_artifacts(root: str, include_attempts: bool = False) -> List[str]:
    """Every per-trial JSON under ``root`` that could carry rollout evidence.

    Handles a live harbor trials_dir (``<trial>/result.json``) and a flat directory
    of ``<trial>.result.json`` files copied off a cluster.

    Harbor also persists each attempt at ``<trial>/attempts/<NNN>/result.json`` and
    copies the SELECTED one to the trial root, so counting both double-counts every
    retried trial. Attempt copies are dropped unless ``include_attempts`` is set.
    """
    if os.path.isfile(root):
        return [root]
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if not include_attempts and f"{os.sep}attempts{os.sep}" in dirpath + os.sep:
            continue
        for name in filenames:
            if name in _CANDIDATE_NAMES or name.endswith(".result.json") or name.endswith(".trajectory.json"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _trial_name(path: str, obj: Dict[str, Any]) -> str:
    for key in ("trial_name", "trial"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    base = os.path.basename(path)
    for suffix in (".result.json", ".trajectory.json"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.basename(os.path.dirname(path)) or base


def load_rollout(path: str) -> Optional[Rollout]:
    """Pull the per-turn served streams out of one artifact.

    Accepts the harbor trial result (``agent_result.rollout_details`` +
    ``agent_result.metadata.all_messages``) and the flatter shapes a probe
    capture writes (top-level ``rollout_details`` + ``messages``).
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"skip {path}: {type(error).__name__}: {error}", file=sys.stderr)
        return None
    if not isinstance(obj, dict):
        return None

    agent_result = obj.get("agent_result") if isinstance(obj.get("agent_result"), dict) else {}
    details = agent_result.get("rollout_details")
    if details is None:
        details = obj.get("rollout_details")
    metadata = agent_result.get("metadata") or obj.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    messages = metadata.get("all_messages")
    if messages is None:
        messages = obj.get("messages")
    if messages is None:
        messages = obj.get("all_messages")

    # By harbor convention rollout_details[0] is the MAIN agent's linear history;
    # later entries are subagents and are NOT part of the trained sequence.
    main: Dict[str, Any] = {}
    n_details = 0
    if isinstance(details, list) and details:
        n_details = len(details)
        if isinstance(details[0], dict):
            main = details[0]
    elif isinstance(details, dict):
        n_details = 1
        main = details

    n_assistant = None
    if isinstance(messages, list):
        n_assistant = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

    exception_info = obj.get("exception_info")
    exception_type = None
    if isinstance(exception_info, dict):
        exception_type = exception_info.get("exception_type") or exception_info.get("type")

    reward = None
    verifier = obj.get("verifier_result")
    if isinstance(verifier, dict):
        value = verifier.get("reward")
        if isinstance(value, (int, float)):
            reward = float(value)

    return Rollout(
        trial=_trial_name(path, obj),
        path=path,
        prompt_token_ids=main.get("prompt_token_ids"),
        completion_token_ids=main.get("completion_token_ids"),
        logprobs=main.get("logprobs"),
        messages=messages if isinstance(messages, list) else None,
        n_assistant=n_assistant,
        exception_type=exception_type,
        reward=reward,
        summarization_count=metadata.get("summarization_count"),
        n_rollout_details=n_details,
    )


def iter_rollouts(paths: Sequence[str]) -> Iterator[Rollout]:
    for path in paths:
        rollout = load_rollout(path)
        if rollout is not None:
            yield rollout


# --------------------------------------------------------------------------
# Tier A: the invariant checks, replayed
# --------------------------------------------------------------------------
def _lcp(left: Sequence[int], right: Sequence[int]) -> int:
    n = 0
    limit = min(len(left), len(right))
    while n < limit and left[n] == right[n]:
        n += 1
    return n


def common_suffix_len(left: Sequence[int], right: Sequence[int]) -> int:
    n = 0
    limit = min(len(left), len(right))
    while n < limit and left[-1 - n] == right[-1 - n]:
        n += 1
    return n


def infer_generation_prompt(rollouts: Sequence[Rollout], max_len: int = 64) -> List[int]:
    """Derive the generation prompt from the data, no tokenizer needed.

    Every served prompt array ends with the assistant generation prompt, so its
    longest common suffix across turns and trials is that prompt (bounded, since
    a longer shared tail can appear by chance when observations repeat).
    """
    arrays: List[List[int]] = []
    for rollout in rollouts:
        for turn in rollout.prompt_token_ids or []:
            if isinstance(turn, list) and turn:
                arrays.append(turn)
    if not arrays:
        return []
    shared = arrays[0]
    for other in arrays[1:]:
        n = common_suffix_len(shared, other)
        shared = shared[len(shared) - n :] if n else []
        if not shared:
            return []
    return shared[-max_len:]


def completion_survived(prev_prompt: Sequence[int], prev_completion: Sequence[int],
                        cur_prompt: Sequence[int], slack: int = 256) -> bool:
    """Did the model's sampled ids survive VERBATIM into the next served prompt?

    The strict prefix check answers "is the whole served history byte-exact",
    which a chat template can break on its own. This looks only for the #520
    failure: the sampled tokens themselves coming back re-cut.
    """
    if not prev_completion or not cur_prompt:
        return False
    start = len(prev_prompt)
    lo = max(0, start - slack)
    hi = min(len(cur_prompt), start + slack + len(prev_completion))
    window = list(cur_prompt[lo:hi])
    needle = list(prev_completion)
    n = len(needle)
    for i in range(0, max(0, len(window) - n) + 1):
        if window[i : i + n] == needle:
            return True
    return False


@dataclass
class TurnFinding:
    turn: int
    lcp: int
    expected_len: int
    actual_len: int
    offset_into_prev_completion: int
    divergence_class: str
    expected_ids: List[int] = field(default_factory=list)
    actual_ids: List[int] = field(default_factory=list)


@dataclass
class RolloutMeasurement:
    trial: str
    path: str
    n_turns: int
    n_assistant: Optional[int]
    sampled_completion_tokens: int
    served_stream_tokens: int
    initial_prompt_tokens: int
    trained_response_tokens: int
    trained_loss_tokens: int
    divergent_turns: int
    reason: Optional[str]
    turn_findings: List[TurnFinding] = field(default_factory=list)
    exception_type: Optional[str] = None
    reward: Optional[float] = None
    summarization_count: Optional[int] = None
    n_rollout_details: int = 0
    per_turn_replay_tokens: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.reason is None


def measure_rollout(rollout: Rollout, generation_prompt_ids: Optional[Sequence[int]],
                    window: int = 6, max_findings: int = 8) -> RolloutMeasurement:
    """Replay MarinSkyRL's full-TITO assembly checks in their real order."""
    prompts = rollout.prompt_token_ids
    completions = rollout.completion_token_ids
    notes: List[str] = []
    if rollout.n_rollout_details > 1:
        notes.append(f"{rollout.n_rollout_details} rollout_details (subagents); only [0] is measured")

    base = RolloutMeasurement(
        trial=rollout.trial,
        path=rollout.path,
        n_turns=0,
        n_assistant=rollout.n_assistant,
        sampled_completion_tokens=0,
        served_stream_tokens=0,
        initial_prompt_tokens=0,
        trained_response_tokens=0,
        trained_loss_tokens=0,
        divergent_turns=0,
        reason=MISSING_STREAMS,
        exception_type=rollout.exception_type,
        reward=rollout.reward,
        summarization_count=rollout.summarization_count,
        n_rollout_details=rollout.n_rollout_details,
        notes=notes,
    )

    if prompts is None or completions is None:
        return base
    n_turns = len(completions)
    base.n_turns = n_turns
    base.sampled_completion_tokens = sum(len(c) for c in completions if isinstance(c, list))
    if n_turns == 0:
        base.reason = EMPTY_STREAMS
        return base
    if len(prompts) != n_turns:
        base.reason = TURN_COUNT_MISMATCH
        return base
    if rollout.n_assistant is not None and rollout.n_assistant != n_turns:
        base.reason = ASSISTANT_MESSAGE_COUNT_MISMATCH
        return base
    for t in range(n_turns):
        p, c = prompts[t], completions[t]
        if not p or not isinstance(p, list) or not c or not isinstance(c, list):
            base.reason = MALFORMED_TURN_STREAM
            return base

    base.per_turn_replay_tokens = sum(len(p) + len(c) for p, c in zip(prompts, completions))

    # Prefix invariant. MarinSkyRL stops at the first failure; we walk every turn
    # so the report says HOW MANY turns diverge and where, then set the reason to
    # what the trainer would have recorded.
    findings: List[TurnFinding] = []
    for t in range(1, n_turns):
        expected = list(prompts[t - 1]) + list(completions[t - 1])
        actual = list(prompts[t])
        n = _lcp(expected, actual)
        if n >= len(expected):
            continue
        offset_into_completion = n - len(prompts[t - 1])
        if offset_into_completion < 0:
            klass = "context"
        elif completion_survived(prompts[t - 1], completions[t - 1], prompts[t]):
            klass = "boundary"
        else:
            klass = "recut"
        if len(findings) < max_findings:
            findings.append(
                TurnFinding(
                    turn=t,
                    lcp=n,
                    expected_len=len(expected),
                    actual_len=len(actual),
                    offset_into_prev_completion=offset_into_completion,
                    divergence_class=klass,
                    expected_ids=expected[max(0, n - 2) : n + window],
                    actual_ids=actual[max(0, n - 2) : n + window],
                )
            )
        base.divergent_turns += 1
    base.turn_findings = findings
    if base.divergent_turns:
        base.reason = PREFIX_MISMATCH
        return base

    p0 = list(prompts[0])
    if generation_prompt_ids:
        gp = list(generation_prompt_ids)
        initial_prompt_len = len(p0) - len(gp)
        if initial_prompt_len < 0:
            base.reason = INITIAL_PROMPT_TOO_SHORT
            base.initial_prompt_tokens = 0
            return base
        if p0[initial_prompt_len:] != gp:
            base.reason = GENERATION_PROMPT_MISMATCH
            base.initial_prompt_tokens = initial_prompt_len
            return base
    else:
        initial_prompt_len = len(p0)
        notes.append("no generation prompt supplied: boundary check skipped, prompt length is an upper bound")
    base.initial_prompt_tokens = initial_prompt_len

    served_full = list(prompts[-1]) + list(completions[-1])
    base.served_stream_tokens = len(served_full)
    for t in range(n_turns):
        off = len(prompts[t])
        comp = list(completions[t])
        if served_full[off : off + len(comp)] != comp:
            base.reason = COMPLETION_REGION_MISMATCH
            return base

    base.reason = None
    base.trained_response_tokens = len(served_full) - initial_prompt_len
    base.trained_loss_tokens = base.sampled_completion_tokens
    return base


# --------------------------------------------------------------------------
# Tier B: run the real MarinSkyRL assembly
# --------------------------------------------------------------------------
class _AnyAttrModule:
    """A module whose every attribute is a fresh permissive placeholder.

    ``trajectory_processing`` imports names from packages the assembly path never
    calls (skyrl_gym verification types, torch tensors). Fabricating them lets the
    real module import; anything that actually TOUCHES one raises, so a stub can
    never silently stand in for real behaviour.
    """

    def __init__(self, name: str):
        self.__name__ = name
        self.__path__: List[str] = []
        self.__all__: List[str] = []

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        placeholder = type(f"{self.__name__}.{item}", (), {"__module__": self.__name__})
        setattr(self, item, placeholder)
        return placeholder


class _StubFinder:
    """Import hook that fabricates any submodule under the given roots."""

    def __init__(self, roots: Sequence[str]):
        self.roots = tuple(roots)

    def find_module(self, fullname: str, path=None):  # legacy API, harmless
        return self if self._owns(fullname) else None

    def _owns(self, fullname: str) -> bool:
        return any(fullname == r or fullname.startswith(r + ".") for r in self.roots)

    def find_spec(self, fullname: str, path=None, target=None):
        if not self._owns(fullname):
            return None
        import importlib.util

        spec = importlib.util.spec_from_loader(fullname, self)
        spec.submodule_search_locations = []
        return spec

    def create_module(self, spec):
        return _AnyAttrModule(spec.name)

    def exec_module(self, module):
        return None


def _install_stubs() -> None:
    """Satisfy trajectory_processing's heavy imports without installing them.

    ``torch`` is needed only for a module-level ``@torch.no_grad()`` decorator.
    ``numpy`` and ``transformers`` are REAL imports and must be present. Every
    other missing dependency is fabricated lazily.
    """
    import types

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    try:
        import numpy  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"--marinskyrl needs numpy and transformers installed: {error}")

    if "torch" not in sys.modules:
        def no_grad(fn=None):
            if fn is None:
                class _Ctx:
                    def __enter__(self_inner): return None
                    def __exit__(self_inner, *a): return False
                    def __call__(self_inner, f): return f
                return _Ctx()
            return fn
        sys.modules["torch"] = module("torch", no_grad=no_grad, Tensor=object)
    if "loguru" not in sys.modules:
        import logging
        sys.modules["loguru"] = module("loguru", logger=logging.getLogger("tito_match"))
    if "omegaconf" not in sys.modules:
        sys.modules["omegaconf"] = module("omegaconf", DictConfig=dict, OmegaConf=object,
                                          ListConfig=list, MISSING=None)
    roots = ["skyrl_gym", "ray", "jaxtyping"]
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder(roots))


def _load_module_from_file(name: str, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_trajectory_processing(marinskyrl_root: str):
    """Load the REAL ``trajectory_processing.py`` from a MarinSkyRL checkout.

    Loaded by FILE PATH, not by package import: ``skyrl_train.trajectory_runners``'s
    ``__init__`` drags in Ray, the inference engines and the whole trainer, none of
    which the assembly path touches. Only the handful of names this module imports
    are stubbed, and ``metric_names`` (pure constants, no imports of its own) is
    loaded for real so the reported metric keys are the ones the trainer emits.
    """
    train_root = os.path.join(marinskyrl_root, "skyrl-train")
    if not os.path.isdir(train_root):
        train_root = marinskyrl_root
    target = os.path.join(train_root, "skyrl_train", "trajectory_runners", "trajectory_processing.py")
    if not os.path.isfile(target):
        raise SystemExit(f"no trajectory_processing.py under {marinskyrl_root}")

    _install_stubs()
    metric_names_path = os.path.join(train_root, "skyrl_train", "metric_names.py")
    for name in ("skyrl_train", "skyrl_train.trajectory_runners", "skyrl_train.inference_engines"):
        sys.modules.setdefault(name, _AnyAttrModule(name))
    for name in (
        "skyrl_train.group_admission",
        "skyrl_train.trajectory_runners.base",
        "skyrl_train.trajectory_runners.trajectory_retention",
        "skyrl_train.trajectory_runners.trajectory_reward_shaping",
        "skyrl_train.inference_engines.base",
    ):
        sys.modules[name] = _AnyAttrModule(name)
    sys.modules["skyrl_train.metric_names"] = _load_module_from_file(
        "skyrl_train.metric_names", metric_names_path
    )
    return _load_module_from_file("_tito_trajectory_processing", target)


def normalize_token_ids(encoded: Any) -> List[int]:
    """Coerce ``apply_chat_template(..., tokenize=True)`` into a flat list of ints.

    Mirrors MarinSkyRL's ``normalize_token_ids``. Load-bearing: on transformers
    4.57+ the call can return a ``BatchEncoding``, which is a ``UserDict`` — so
    ``len()`` is the KEY COUNT (2), and naive slicing yields an empty sequence.
    """
    if hasattr(encoded, "keys") and hasattr(encoded, "__getitem__"):
        try:
            encoded = encoded["input_ids"]
        except (KeyError, TypeError):
            pass
    if hasattr(encoded, "ids"):  # tokenizers.Encoding
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    encoded = list(encoded)
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"expected a singleton batch of token ids, got {len(encoded)} rows")
        encoded = encoded[0]
    return [int(x) for x in encoded]


def generation_prompt_from_tokenizer(tokenizer, enable_thinking: bool = False,
                                     custom_chat_template: Optional[str] = None) -> List[int]:
    """The assistant generation prompt, derived exactly as MarinSkyRL derives it."""
    ctk = {"enable_thinking": True} if enable_thinking else {}
    empty = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True,
                                      chat_template=custom_chat_template, **ctk)
    )
    with_gp = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True,
                                      tokenize=True, chat_template=custom_chat_template, **ctk)
    )
    return with_gp[len(empty):]


def load_tokenizer(spec: str):
    from transformers import AutoTokenizer

    if os.path.isfile(spec) and spec.endswith(".json"):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(spec))
    return AutoTokenizer.from_pretrained(spec, trust_remote_code=False)


def measure_rollout_with_trainer(rollout: Rollout, module, tokenizer,
                                 chat_template_kwargs: Optional[Dict[str, Any]],
                                 custom_chat_template: Optional[str]) -> Dict[str, Any]:
    """Run MarinSkyRL's assembly for real, TITO on and off, and compare ids.

    Mirrors ``HarborTrajectoryRunner._build_agent_output``: system messages are
    split off, the first user message is the prompt, and everything after it is
    the response the trainer assembles.
    """
    messages = rollout.messages or []
    conversation = [m for m in messages if m.get("role") != "system"]
    if len(conversation) < 2 or conversation[0].get("role") != "user":
        return {"error": "invalid chat history"}
    response_messages = conversation[1:]
    sampled = [list(c) for c in (rollout.completion_token_ids or [])]
    flat_sampled = [tok for turn in sampled for tok in turn]

    out: Dict[str, Any] = {}
    # "tito_full": the arm's real setting (use_tis=true forces rollout logprobs,
    # which selects full TITO and masks the whole trajectory on a decline).
    # "splice_only": the pre-#521 path, no behaviour-logprob requirement, so the
    # re-tokenizing assembly actually runs and we can see what it would train on.
    modes = (("tito_full", True, True), ("splice_only", False, False))
    for label, tito, logprobs_required in modes:
        stats = module.AlignmentStats()
        try:
            result = module.get_response_ids_and_loss_mask_from_messages(
                response_messages,
                tokenizer,
                rollout.logprobs,
                custom_chat_template=custom_chat_template,
                assistant_token_ids=rollout.completion_token_ids,
                alignment_stats=stats,
                chat_template_kwargs=chat_template_kwargs,
                assistant_prompt_token_ids=rollout.prompt_token_ids,
                rollout_logprobs_required=logprobs_required,
                tito_full=tito,
                tis_splice=True,
            )
        except Exception as error:  # noqa: BLE001 - report, never guess
            out[label] = {"error": f"{type(error).__name__}: {error}"}
            continue
        response_ids, loss_mask = result[0], result[1]
        trained = [tok for tok, keep in zip(response_ids, loss_mask) if keep]
        out[label] = {
            "response_tokens": len(response_ids),
            "trained_tokens": len(trained),
            "sampled_tokens": len(flat_sampled),
            "trained_equals_sampled": trained == flat_sampled,
            "longest_common_prefix": _lcp(trained, flat_sampled),
            "tito_full_attempts": stats.n_tito_full_attempts,
            "tito_full_successes": stats.n_tito_full_successes,
            "tito_full_declines": {getattr(k, "value", str(k)): v for k, v in stats.tito_full_declines.items()},
            "exact_aligned_tokens": stats.n_exact,
            "lcs_aligned_tokens": stats.n_lcs,
            "lcs_fallback_messages": stats.n_lcs_messages,
            "unaligned_tokens": stats.n_unaligned,
        }
    return out


# --------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------
def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def aggregate(measurements: Sequence[RolloutMeasurement]) -> Dict[str, Any]:
    n = len(measurements)
    exact = [m for m in measurements if m.exact]
    reasons: Dict[str, int] = {}
    classes: Dict[str, int] = {}
    for m in measurements:
        if m.reason:
            reasons[m.reason] = reasons.get(m.reason, 0) + 1
        for finding in m.turn_findings:
            classes[finding.divergence_class] = classes.get(finding.divergence_class, 0) + 1
    total_turns = sum(m.n_turns for m in measurements)
    checked_turns = sum(max(0, m.n_turns - 1) for m in measurements)
    divergent_turns = sum(m.divergent_turns for m in measurements)
    sampled_tokens = sum(m.sampled_completion_tokens for m in measurements)
    trained_tokens = sum(m.trained_loss_tokens for m in measurements)
    turn_counts = sorted(m.n_turns for m in measurements)
    long_rollouts = [m for m in measurements if m.n_turns >= 20]
    return {
        "n_rollouts": n,
        "n_exact": len(exact),
        "exact_match_fraction": (len(exact) / n) if n else 0.0,
        "n_rollouts_ge_20_turns": len(long_rollouts),
        "exact_match_fraction_ge_20_turns": (
            sum(1 for m in long_rollouts if m.exact) / len(long_rollouts) if long_rollouts else None
        ),
        "total_turns": total_turns,
        "prefix_checked_turns": checked_turns,
        "divergent_turns": divergent_turns,
        "divergent_turn_fraction": (divergent_turns / checked_turns) if checked_turns else 0.0,
        "sampled_completion_tokens": sampled_tokens,
        "trained_loss_tokens": trained_tokens,
        "tokens_lost_to_declines": sampled_tokens - trained_tokens,
        "turns_median": statistics.median(turn_counts) if turn_counts else 0,
        "turns_min": turn_counts[0] if turn_counts else 0,
        "turns_max": turn_counts[-1] if turn_counts else 0,
        "decline_reasons": reasons,
        "divergence_classes": classes,
    }


def print_report(measurements: Sequence[RolloutMeasurement], summary: Dict[str, Any],
                 generation_prompt_ids: Optional[Sequence[int]], show: int) -> None:
    print()
    print("PER-ROLLOUT")
    header = f"{'trial':<30} {'turns':>5} {'sampled':>9} {'trained':>9} {'div':>4} {'lcp@1st':>9}  reason"
    print(header)
    print("-" * len(header))
    for m in measurements[:show]:
        first = m.turn_findings[0].lcp if m.turn_findings else ""
        print(
            f"{m.trial[:30]:<30} {m.n_turns:>5} {m.sampled_completion_tokens:>9} "
            f"{m.trained_loss_tokens:>9} {m.divergent_turns:>4} {str(first):>9}  {m.reason or 'EXACT'}"
        )
    if len(measurements) > show:
        print(f"... {len(measurements) - show} more (use --show to widen, --json for all)")

    print()
    print("AGGREGATE")
    n = summary["n_rollouts"]
    rows = [
        ("rollouts measured", str(n)),
        ("TITO exact (trained ids == sampled ids)", f"{summary['n_exact']}/{n}  {_pct(summary['n_exact'], n):.1f}%"),
        ("turns (min/median/max)", f"{summary['turns_min']}/{summary['turns_median']}/{summary['turns_max']}"),
        ("divergent turns", f"{summary['divergent_turns']}/{summary['prefix_checked_turns']}  "
                            f"{100 * summary['divergent_turn_fraction']:.2f}%"),
        ("sampled completion tokens", f"{summary['sampled_completion_tokens']:,}"),
        ("tokens the trainer would train on", f"{summary['trained_loss_tokens']:,}"),
        ("tokens lost to declines", f"{summary['tokens_lost_to_declines']:,}"),
    ]
    if summary["exact_match_fraction_ge_20_turns"] is not None:
        rows.insert(2, (
            "TITO exact, rollouts >= 20 turns",
            f"{summary['exact_match_fraction_ge_20_turns'] * 100:.1f}% of {summary['n_rollouts_ge_20_turns']}",
        ))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")

    if summary["decline_reasons"]:
        print()
        print("  decline reasons (matches generate/tis/tito_full/decline/<reason>)")
        for reason in REASONS:
            count = summary["decline_reasons"].get(reason)
            if count:
                print(f"    {reason:<38} {count}")
    if summary["divergence_classes"]:
        print()
        print("  where the divergence lands")
        legend = {
            "context": "inside the earlier context, before the sampled tokens (template drift)",
            "boundary": "sampled ids survive verbatim, the surrounding template shifted",
            "recut": "the model's own sampled tokens came back re-cut (issue #520)",
        }
        for klass, count in sorted(summary["divergence_classes"].items(), key=lambda kv: -kv[1]):
            print(f"    {klass:<10} {count:>5}   {legend.get(klass, '')}")

    if generation_prompt_ids:
        print()
        print(f"  generation prompt ids: {list(generation_prompt_ids)}")

    print()
    if summary["n_exact"] == n and n:
        print("VERDICT: TITO HOLDS. Every rollout assembles from the served ids; 0 divergent turns.")
    else:
        print(
            f"VERDICT: TITO DOES NOT HOLD. {n - summary['n_exact']}/{n} rollouts decline; "
            f"the trainer re-tokenizes those and, with use_tis=true, masks them out of the loss entirely."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials-dir", required=True,
                        help="harbor trials_dir, or any directory of per-trial result.json files")
    parser.add_argument("--limit", type=int, default=0, help="measure at most N trials (0 = all)")
    parser.add_argument("--include-attempts", action="store_true",
                        help="also read <trial>/attempts/NNN/result.json (default: selected results only)")
    parser.add_argument("--show", type=int, default=25, help="rows in the per-rollout table")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer dir, HF repo id, or tokenizer.json; enables the generation-prompt check")
    parser.add_argument("--generation-prompt-ids", default=None,
                        help="explicit comma-separated ids instead of deriving them")
    parser.add_argument("--no-infer-generation-prompt", action="store_true",
                        help="skip deriving the generation prompt from the data")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="pass chat_template_kwargs={'enable_thinking': True}, as the runner does for Qwen3.5/3.6")
    parser.add_argument("--marinskyrl", default=None,
                        help="MarinSkyRL checkout: run the REAL assembly (tier B) instead of only replaying its checks")
    parser.add_argument("--trainer-dump", default=None,
                        help="a dumped_data/global_step_N_train_rollouts.jsonl, for a decoded-text cross-check")
    args = parser.parse_args()

    paths = find_artifacts(args.trials_dir, include_attempts=args.include_attempts)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no per-trial JSON under {args.trials_dir}", file=sys.stderr)
        return 2
    print(f"reading {len(paths)} artifact(s) under {args.trials_dir}")

    rollouts = [r for r in iter_rollouts(paths)]
    if not rollouts:
        print("no rollouts loaded", file=sys.stderr)
        return 2

    generation_prompt_ids: Optional[List[int]] = None
    source = None
    if args.generation_prompt_ids:
        generation_prompt_ids = [int(x) for x in args.generation_prompt_ids.replace(" ", "").split(",") if x]
        source = "explicit"
    elif args.tokenizer:
        generation_prompt_ids = generation_prompt_from_tokenizer(
            load_tokenizer(args.tokenizer), enable_thinking=args.enable_thinking
        )
        source = f"tokenizer {args.tokenizer}"
    elif not args.no_infer_generation_prompt:
        generation_prompt_ids = infer_generation_prompt(rollouts)
        source = ("inferred from the shared suffix of the served prompts "
                  "(may be LONGER than the tokenizer's; pass --tokenizer for the real one)")
    if generation_prompt_ids:
        print(f"generation prompt ({source}): {generation_prompt_ids}")
    else:
        print("no generation prompt: the response/prompt boundary check is skipped")

    measurements = [measure_rollout(r, generation_prompt_ids) for r in rollouts]
    summary = aggregate(measurements)

    tier_b: Dict[str, Any] = {}
    if args.marinskyrl:
        module = load_trajectory_processing(args.marinskyrl)
        if not args.tokenizer:
            print("--marinskyrl needs --tokenizer", file=sys.stderr)
            return 2
        tokenizer = load_tokenizer(args.tokenizer)
        ctk = {"enable_thinking": True} if args.enable_thinking else None
        print(f"tier B: running the real assembly from {args.marinskyrl}")
        for rollout in rollouts:
            tier_b[rollout.trial] = measure_rollout_with_trainer(rollout, module, tokenizer, ctk, None)
        agree = sum(
            1 for r in rollouts
            if bool(tier_b[r.trial].get("tito_full", {}).get("trained_equals_sampled"))
        )
        print(f"tier B: trained ids == sampled ids under tito_full for {agree}/{len(rollouts)} rollouts")

    dump_note = None
    if args.trainer_dump:
        with open(args.trainer_dump, "r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        dump_note = (
            f"trainer dump keys: {sorted(first)} - decoded TEXT only "
            "(tokenizer.decode(skip_special_tokens=True)), so it cannot settle a token-id question"
        )
        print(dump_note)

    print_report(measurements, summary, generation_prompt_ids, args.show)

    if args.json_out:
        payload = {
            "trials_dir": os.path.abspath(args.trials_dir),
            "generation_prompt_ids": generation_prompt_ids,
            "generation_prompt_source": source,
            "summary": summary,
            "rollouts": [
                {
                    **{k: v for k, v in m.__dict__.items() if k != "turn_findings"},
                    "turn_findings": [f.__dict__ for f in m.turn_findings],
                }
                for m in measurements
            ],
            "trainer_assembly": tier_b or None,
            "trainer_dump_note": dump_note,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
