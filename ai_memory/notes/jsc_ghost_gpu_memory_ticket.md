# JSC ticket draft — orphaned ("ghost") GPU memory on JUPITER Booster after killed jobs

Draft for Luke to send to JSC support / PI channel (sc@fz-juelich.de or the JUPITER
operations contact). Suggested subject:
**"JUPITER Booster: GPU memory stays allocated after job kill (no owning process) — node
list + request for health-check"**

---

## What we observe

During a multi-day GRPO training campaign (Qwen3-30B MoE, 6×GH200 nodes per job, account
`reformo`, user `lee27`), jobs occasionally deadlock in NCCL collectives (a known
software race in our stack — not a hardware complaint). When such a job is terminated —
by `scancel` OR by the torch NCCL watchdog — the GPUs on those nodes frequently keep
**~75 GB per GPU allocated with no owning process**. The nodes return to the Slurm pool,
the next job scheduled there OOMs immediately at model load.

Direct evidence (torch OOM reports at process start, before our model allocated anything):

| date (CEST) | job that OOM'd | nodes | free HBM at start |
|---|---|---|---|
| 08-14 ~21:5x | 1376820 | jpbo-003-[35,37,39-40,42,45,47] area | 39 MiB of 95 GiB |
| 08-14 ~23:0x | 1379098 | jpbo-030-[33,36,39,41,44,46] | 37 MiB of 95 GiB |
| 08-14 ~23:2x | 1379097 | jpbo-025-[17,22-23,29-30,32] | 58 MiB of 95 GiB |
| 08-15 ~00:56 | 1379307 | jpbo-003-[20,23-24,26-27,30] | 18.88 MiB of 95 GiB |

In each case the OOM message attributes only ~7-20 GB to live processes; the rest is
unattributed. Pattern reproduced across 11 OOM incidents in ~27 hours; it happens after
both `scancel` and watchdog self-aborts of jobs whose ranks spin in a hung NCCL kernel
(SIGKILL of a process inside a spinning collective appears to leak the CUDA context;
only a GPU reset / node reboot reclaims it — not something we can do from user space).

## Nodes we are currently avoiding (quarantined via `--exclude`)

Killed-while-hung job sites (ghost memory confirmed or presumed) from 2026-08-14/15:

```
jpbo-014-[18,24-25,27,31-32]  jpbo-034-[18,23-25,27-29]  jpbo-044-[07,11-15]
jpbo-060-[05-06,10,12,14,16]  jpbo-115-[42,44-48]        jpbo-003-[35,37,39-40,42,45,47]
jpbo-102-23  jpbo-103-20      jpbo-122-[02-03,20,23,26-29]
jpbo-068-[09-13,16]           jpbo-067-[06,08-12]        jpbo-030-[33,36,39,41,44,46]
jpbo-112-[05,08-12]           jpbo-025-[17,22-23,29-30,32]
jpbo-003-[20,23-24,26-27,30]  jpbo-081-[10-15]           jpbo-107-[07-12]
jpbo-054-[01-03,09,12-13]     jpbo-066-[17-18,20-22,24]
```

(Authoritative, dated, per-incident list with job IDs: `hpc/hpc.py` `node_exclusion_list`
comments + git history on branch `lukedhlee/vista-moe-grpo-30b`.)

## Questions / requests for JSC

1. Could the nodes above be health-checked / GPU-reset / rebooted to reclaim the
   orphaned HBM? (After that we will prune them from our exclude list.)
2. Does the node epilog or periodic health check detect "HBM nearly full with zero
   owning processes"? If not, could such a check be added? It would catch this class
   before the node is rehanded to the next user — we are presumably not the only ones
   hitting it.
3. Is there any user-accessible mechanism to (a) request a GPU reset at allocation time
   (prolog option / constraint), or (b) flag a node as unhealthy from user space, so we
   don't have to maintain a manual `--exclude` list?
4. What is the preferred ongoing reporting channel when we identify such nodes —
   per-node tickets, a list like this, or something automated?
5. Context question: is CUDA-context cleanup after SIGKILL of processes spinning in NCCL
   collectives a known issue on GH200 nodes (driver/firmware level)? Happy to share
   reproduction details and exact job IDs.

## Notes for our side (not part of the ticket)

- Root cause of the hangs themselves is a software race in our RL stack (EP/FSDP2
  backward collectives) — being pursued separately upstream. JSC's part is node hygiene
  after the kill, which bites ANY user whose distributed job dies hard.
- Planned self-defense regardless of JSC action: pre-flight `nvidia-smi` free-memory
  guard in our sbatch (fail fast in ~1 min + auto-exclude + resubmit) — see
  base30b_gsm8k_validation.md incident log.
- PI guidance 2026-08-15: storage default is `/e/`; `/p/` going obsolete; datasets →
  `/e/data1/datasets/playground/mmlaion/`.
