"""Static and CPU-only contract tests for FlashInfer AOT node-local staging."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "hpc/shell_utils/flashinfer_aot_cache.sh"
SBATCH = REPO_ROOT / "hpc/sbatch_rl/universal_rl.sbatch"
EXPECTED_VERSION = "0.6.11.post2+cu130"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _make_archive(tmp_path: Path) -> tuple[Path, str]:
    payload = tmp_path / "payload"
    package = payload / "flashinfer_jit_cache"
    so = package / "jit_cache/fused_moe_90/fused_moe_90.so"
    so.parent.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from ._build_meta import __version__\n"
    )
    (package / "_build_meta.py").write_text(
        f"__version__ = {EXPECTED_VERSION!r}\n"
    )
    so.write_bytes(b"fake but hash-checked shared object\n")

    archive = tmp_path / "flashinfer-aot.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(package, arcname="flashinfer_jit_cache")
    return archive, _sha256(so)


def _make_fake_runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    fused_moe = runtime / "flashinfer/jit/fused_moe.py"
    fused_moe.parent.mkdir(parents=True)
    (runtime / "flashinfer/__init__.py").write_text("")
    (runtime / "flashinfer/jit/__init__.py").write_text("")
    fused_moe.write_text(
        textwrap.dedent(
            """
            import pathlib
            from types import SimpleNamespace
            import flashinfer_jit_cache

            def gen_cutlass_fused_moe_sm90_module():
                package = pathlib.Path(flashinfer_jit_cache.__file__).resolve().parent
                path = package / "jit_cache/fused_moe_90/fused_moe_90.so"
                return SimpleNamespace(is_aot=path.exists(), aot_path=path)
            """
        )
    )
    return runtime


def _make_fake_slurm(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "sbcast",
        "#!/bin/bash\nset -e\n[[ $1 == --force ]]\ncp \"$2\" \"$3\"\n",
    )
    _write_executable(
        bin_dir / "srun",
        textwrap.dedent(
            """\
            #!/bin/bash
            set -e
            while [[ "$1" == --* ]]; do shift; done
            exec "$@"
            """
        ),
    )
    return bin_dir


def _run_stage(tmp_path: Path, *, overrides: dict[str, str] | None = None):
    archive, so_sha = _make_archive(tmp_path)
    runtime = _make_fake_runtime(tmp_path)
    bin_dir = _make_fake_slurm(tmp_path)
    workspace = tmp_path / "writable-jit-fallback"
    workspace.mkdir()
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHONPATH": str(runtime),
        "USER": "unit",
        "SLURM_JOB_ID": "123",
        "SLURM_NNODES": "1",
        "FLASHINFER_WORKSPACE_BASE": str(workspace),
        "FLASHINFER_AOT_ARCHIVE": str(archive),
        "FLASHINFER_AOT_ARCHIVE_SHA256": _sha256(archive),
        "FLASHINFER_AOT_SO_SHA256": so_sha,
        "FLASHINFER_AOT_CACHE_KEY": "fi0.6.11.post2-cu130-py312-sm90a",
    }
    env.update(overrides or {})
    command = (
        f"source {SCRIPT} || exit $?; "
        "printf 'PYTHONPATH=%s\\nWORKSPACE=%s\\n' "
        '"$PYTHONPATH" "$FLASHINFER_WORKSPACE_BASE"'
    )
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_unset_hook_is_noop() -> None:
    result = subprocess.run(
        ["bash", "-c", f"PYTHONPATH=sentinel; source {SCRIPT}; printf %s \"$PYTHONPATH\""],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "sentinel"


def test_partial_configuration_fails_loudly() -> None:
    result = subprocess.run(
        ["bash", "-c", f"source {SCRIPT}"],
        env=os.environ | {"FLASHINFER_AOT_ARCHIVE": "/some/archive.tar"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "FLASHINFER_AOT_ARCHIVE_SHA256 is required" in result.stderr


def test_stages_verifies_and_forces_node_local_writable_fallback(
    tmp_path: Path,
) -> None:
    result = _run_stage(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "verified version=0.6.11.post2+cu130" in result.stdout
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("PYTHONPATH=", "WORKSPACE="))
    )
    assert values["PYTHONPATH"].startswith("/tmp/flashinfer_aot_unit_123/")
    assert values["WORKSPACE"] == "/tmp/flashinfer_unit_123"


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"FLASHINFER_AOT_ARCHIVE_SHA256": "0" * 64}, "source archive hash mismatch"),
        ({"FLASHINFER_AOT_SO_SHA256": "0" * 64}, "shared-object hash mismatch"),
        ({"FLASHINFER_AOT_CACHE_KEY": "../escape"}, "not filesystem-safe"),
    ],
)
def test_invalid_artifact_contract_fails(
    tmp_path: Path, override: dict[str, str], error: str
) -> None:
    result = _run_stage(tmp_path, overrides=override)
    assert result.returncode != 0
    assert error in result.stderr


def test_universal_sbatch_orders_hook_after_writable_cache_before_runner() -> None:
    template = SBATCH.read_text()
    triton = 'source "$WORKDIR/hpc/shell_utils/triton_cache.sh"'
    aot = 'source "$WORKDIR/hpc/shell_utils/flashinfer_aot_cache.sh"'
    runner = '"$RL_PYTHON" -m hpc.rl_launch_utils'
    assert template.index(triton) < template.index(aot) < template.index(runner)
