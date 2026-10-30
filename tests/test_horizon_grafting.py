"""Check delta arithmetic, frozen router state and sharded export round trips."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
load_file, save_file = safetensors_torch.load_file, safetensors_torch.save_file

SCRIPT = Path(__file__).parents[1] / "data/r2egym/horizon/grafting/build_grafts.py"
SPEC = importlib.util.spec_from_file_location("horizon_grafting", SCRIPT)
grafting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grafting)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_delta_arithmetic_and_zero_parity(dtype):
    target = torch.tensor([2.0, -4.0], dtype=dtype)
    donor = torch.tensor([1.0, 3.0], dtype=dtype)
    trained = torch.tensor([5.0, 1.0], dtype=dtype)
    for alpha in (0, 0.25, 0.5, 0.75, 1):
        actual = grafting.graft_tensor(target, donor, trained, alpha, "layer.weight")
        expected = (target.float() + alpha * (trained.float() - donor.float())).to(
            target.dtype
        )
        assert torch.equal(actual, expected)
    assert torch.equal(target, torch.tensor([2.0, -4.0], dtype=dtype))
    assert torch.equal(trained, torch.tensor([5.0, 1.0], dtype=dtype))


def test_router_state_and_nonfinite_rejection():
    target = torch.tensor([0.1234567], dtype=torch.float32)
    donor = torch.tensor([9.0], dtype=torch.bfloat16)
    trained = torch.tensor([-9.0], dtype=torch.bfloat16)
    for key in ("model.layers.0.mlp.router.bias", "pending_qb_betas"):
        result = grafting.graft_tensor(target, donor, trained, 1, key)
        assert result.dtype == torch.float32
        assert torch.equal(result, target)
    with pytest.raises(ValueError, match="Nonfinite"):
        grafting.graft_tensor(
            target, donor, torch.tensor([float("nan")]), 0.5, "layer.weight"
        )


def make_model(root, values, split=False):
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "fixture"}')
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / name).write_text("identical")
    groups = [{key: value} for key, value in values.items()] if split else [values]
    weight_map = {}
    for i, group in enumerate(groups):
        name = f"shard-{i}.safetensors"
        save_file(group, root / name)
        weight_map.update({key: name for key in group})
    size = sum(value.numel() * value.element_size() for value in values.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map})
    )


def test_export_across_different_sharding(tmp_path):
    key, state = "model.layer.weight", "model.layers.0.mlp.router.bias"
    donor, trained, target = [
        tmp_path / name for name in ("donor", "trained", "target")
    ]
    make_model(
        donor,
        {key: torch.tensor([1.0], dtype=torch.bfloat16), state: torch.tensor([3.0])},
    )
    make_model(
        trained,
        {key: torch.tensor([5.0], dtype=torch.bfloat16), state: torch.tensor([-3.0])},
        split=True,
    )
    make_model(
        target,
        {
            key: torch.tensor([2.0], dtype=torch.bfloat16),
            state: torch.tensor([0.1234567]),
        },
    )
    args = SimpleNamespace(
        donor=donor,
        donor_ft=trained,
        target=target,
        output=tmp_path / "outputs",
        alphas=[0, 0.5, 1],
        threads=1,
        revision="fixture",
    )
    grafting.build(args)
    for alpha, expected in ((0, 2.0), (50, 4.0), (100, 6.0)):
        output = args.output / f"graft{alpha:03d}"
        actual = load_file(output / "shard-0.safetensors")
        assert actual[key].item() == expected
        assert torch.equal(actual[state], torch.tensor([0.1234567]))
        assert (output / "graft_complete.json").is_file()
        assert grafting.inventory(output)[1] == grafting.inventory(target)[1]
    with pytest.raises(FileExistsError):
        grafting.build(args)
    (trained / "chat_template.jinja").write_text("incompatible")
    with pytest.raises(ValueError, match="Tokenizer or chat template"):
        grafting.audit([donor, trained, target])
