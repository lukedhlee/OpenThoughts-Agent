#!/bin/bash
#SBATCH --job-name=ghost_probe
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/ghost_probe/logs/%x_%N_%j.out

# Ghost-GPU reproduction probe (stranded-memory evidence for JSC).
# Run on a suspected node:   sbatch --nodelist=jpbo-XXX-YY ghost_gpu_probe.sbatch
# The job allocates NOTHING on the GPUs before the nvidia-smi snapshot below, so
# any memory.used / compute process it reports belongs to NO job on this node —
# it is residue from a previous job that Slurm did not clean up.
# Step 2 then proves the residue is real capacity loss: an 80GiB cuMemAlloc that
# always succeeds on a clean 96GB GH200 fails with CUDA_ERROR_OUT_OF_MEMORY here.
# No dependencies beyond system python3 + the NVIDIA driver (libcuda).

set -u
mkdir -p /e/fscratch/reformo/lee27/experiments/ghost_probe/logs

echo "=== GHOST GPU PROBE ==="
echo "node=$(hostname)  job=$SLURM_JOB_ID  time=$(date -Is)"
echo
echo "--- [1] nvidia-smi snapshot BEFORE this job touches any GPU ---"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
echo
echo "--- [1b] GPU compute processes (every PID here is foreign to this job) ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
echo "(empty list = clean node)"
echo
echo "--- [1c] do those PIDs still exist on the node? ---"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  if [ -d "/proc/$pid" ]; then
    echo "pid $pid ALIVE: $(ps -o user=,etime=,comm= -p $pid 2>/dev/null)"
  else
    echo "pid $pid GONE (memory leaked without an owning process)"
  fi
done
echo
echo "--- [2] allocation probe: 80GiB cuMemAlloc per GPU (clean GH200 -> OK) ---"
python3 - <<'PYEOF'
import ctypes, sys

cu = None
for name in ("libcuda.so.1", "libcuda.so"):
    try:
        cu = ctypes.CDLL(name)
        break
    except OSError:
        pass
if cu is None:
    sys.exit("libcuda not found on this node")

r = cu.cuInit(0)
if r != 0:
    sys.exit(f"cuInit failed with CUDA error {r}")

n = ctypes.c_int()
cu.cuDeviceGetCount(ctypes.byref(n))
GiB = 1024 ** 3
WANT = 80 * GiB

for d in range(n.value):
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), d)
    ctx = ctypes.c_void_p()
    r = cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    if r != 0:
        print(f"GPU{d}: context creation failed (CUDA error {r})")
        continue
    cu.cuCtxSetCurrent(ctx)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    ptr = ctypes.c_void_p()
    r = cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(WANT))
    if r == 0:
        verdict = "OK"
        cu.cuMemFree_v2(ptr)
    elif r == 2:
        verdict = "FAILED: CUDA_ERROR_OUT_OF_MEMORY"
    else:
        verdict = f"FAILED: CUDA error {r}"
    print(
        f"GPU{d}: total={total.value / GiB:.1f}GiB "
        f"free={free.value / GiB:.1f}GiB "
        f"alloc(80GiB) -> {verdict}"
    )
PYEOF
echo
echo "=== END PROBE $(hostname) ==="
