#!/usr/bin/env python3
"""Build target + alpha * (donor_ft - donor), retaining target router state.

Run on a compute node. Processes one target shard at a time; all math is FP32,
then each parameter is cast to the target dtype. Never loads a whole model.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
from contextlib import ExitStack
from pathlib import Path


def is_state(key):
    return key.endswith(".mlp.router.bias") or "pending_qb_betas" in key


def inventory(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    schema = {}
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("Shard paths must be basenames")
        with (root / shard).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError("Oversized safetensors header")
            header = json.loads(stream.read(length))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in schema or index["weight_map"].get(key) != shard:
                raise ValueError(f"Invalid shard index for {key}")
            schema[key] = (value["dtype"], value["shape"])
    if set(schema) != set(index["weight_map"]):
        raise ValueError("Index and tensor inventory differ")
    return index, schema


def audit(roots):
    inventories = [inventory(root) for root in roots]
    config = json.loads((roots[0] / "config.json").read_text())
    fingerprints = {}
    for root, (_, schema) in zip(roots, inventories):
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError(f"Model config differs: {root}")
        reference = inventories[0][1]
        if set(schema) != set(reference):
            raise ValueError(f"Tensor keys differ: {root}")
        for key, (dtype, shape) in schema.items():
            if shape != reference[key][1] or (
                not is_state(key) and dtype != reference[key][0]
            ):
                raise ValueError(f"Tensor schema differs: {root}: {key}")
            if not is_state(key) and not key.endswith(".weight"):
                raise ValueError(f"Unclassified tensor (weight or state): {key}")
        hashes = {}
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        fingerprints[str(root)] = hashes
    if len({json.dumps(value, sort_keys=True) for value in fingerprints.values()}) != 1:
        raise ValueError("Tokenizer or chat template differs")
    return inventories, fingerprints


def graft_tensor(target, donor, donor_ft, alpha, key):
    import torch

    for tensor in (target, donor, donor_ft):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite source: {key}")
    if is_state(key) or alpha == 0:
        return target.clone()
    if not target.is_floating_point():
        if not torch.equal(donor, donor_ft):
            raise ValueError(f"Nonfloating donor delta: {key}")
        return target.clone()
    delta = donor_ft.float() - donor.float()
    result = torch.add(target.float(), delta, alpha=alpha).to(target.dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite result: {key}")
    return result.contiguous()


def build(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(args.threads)
    roots = [args.donor, args.donor_ft, args.target]
    inventories, fingerprints = audit(roots)
    maps = [value[0]["weight_map"] for value in inventories]
    outputs = [args.output / f"graft{round(alpha * 100):03d}" for alpha in args.alphas]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Alpha output names collide")
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    required = sum(
        (args.target / name).stat().st_size for name in set(maps[2].values())
    ) * len(outputs)
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < required + 10 * 1024**3:
        raise ValueError("Insufficient scratch space")
    for output in outputs:
        output.mkdir()
        for source in args.target.iterdir():
            if (
                source.is_file()
                and source.suffix in (".json", ".jinja", ".py")
                and source.name != "model.safetensors.index.json"
            ):
                if source.stat().st_size > 64 * 1024**2:
                    raise ValueError(f"Unexpected large sidecar: {source}")
                shutil.copy2(source, output / source.name)
    norms = {}
    for shard in sorted(set(maps[2].values())):
        keys = sorted(key for key, name in maps[2].items() if name == shard)
        tensors = [{} for _ in outputs]
        with ExitStack() as stack:
            handles = {}

            def read(i, key):
                name = maps[i][key]
                if (i, name) not in handles:
                    handles[i, name] = stack.enter_context(
                        safe_open(roots[i] / name, framework="pt", device="cpu")
                    )
                return handles[i, name].get_tensor(key)

            for key in keys:
                donor, donor_ft, target = (read(i, key) for i in range(3))
                if not is_state(key):
                    delta = donor_ft.float() - donor.float()
                    size = torch.linalg.vector_norm(delta).item()
                    base_norm = torch.linalg.vector_norm(target.float()).item()
                    norms[key] = {
                        "delta_norm": size,
                        "target_norm": base_norm,
                        "relative_delta": size / base_norm if base_norm else None,
                    }
                    del delta
                for alpha, values in zip(args.alphas, tensors):
                    values[key] = graft_tensor(target, donor, donor_ft, alpha, key)
        for output, values in zip(outputs, tensors):
            temporary = output / (shard + ".partial")
            save_file(values, temporary, metadata={"format": "pt"})
            # Verify every saved tensor against the computed result, including state.
            with safe_open(temporary, framework="pt", device="cpu") as saved:
                for key in keys:
                    actual = saved.get_tensor(key)
                    if not torch.equal(actual, values[key]):
                        raise ValueError(f"Saved tensor mismatch: {key}")
            temporary.rename(output / shard)
            values.clear()
        print(f"SHARD_OK {shard}", flush=True)
    for alpha, output in zip(args.alphas, outputs):
        (output / "model.safetensors.index.json").write_text(
            json.dumps(inventories[2][0], indent=2) + "\n"
        )
        manifest = {
            "formula": "target + alpha * (donor_ft - donor)",
            "alpha": alpha,
            "donor": str(args.donor),
            "donor_ft": str(args.donor_ft),
            "target": str(args.target),
            "state_policy": "copy target router biases and pending_qb_betas exactly",
            "tokenizer_sha256": fingerprints,
            "code_revision": args.revision,
            "tensor_count": len(maps[2]),
            "state_tensor_count": sum(map(is_state, maps[2])),
            "per_tensor_norms": norms,
        }
        (output / "graft_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if inventory(output)[1] != inventories[2][1]:
            raise ValueError("Output schema differs from target")
        (output / "graft_complete.json").write_text(
            json.dumps({"alpha": alpha, "verified": True}) + "\n"
        )
    (args.output / "build_complete.json").write_text(
        json.dumps([str(output) for output in outputs]) + "\n"
    )
    print("GRAFT_BUILD_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("donor", "donor-ft", "target", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(alpha) and 0 <= alpha <= 1 for alpha in args.alphas):
        parser.error("Alphas must be finite and in [0, 1]")
    if args.audit_only:
        inventories, fingerprints = audit([args.donor, args.donor_ft, args.target])
        print(
            json.dumps(
                {
                    "tensor_counts": [len(value[1]) for value in inventories],
                    "fingerprints": fingerprints,
                }
            )
        )
    else:
        build(args)


if __name__ == "__main__":
    main()
