#!/bin/bash
# mix_driver.sh — arm MIX: wait for prep, verify the cache (EQUAL required), then train (lane: train + exports + NLL).
D=/e/fscratch/reformo/lee27/experiments/relay/sft_v2; S=/e/data1/mmlaion/lee27/snowball-sft
log() { echo "[$(TZ=America/Los_Angeles date +%H:%M) PT] $*" | tee -a $D/mix_driver.log; }
st() { echo "- $(TZ=America/Los_Angeles date +%H:%M) — MIX: $*" >> $D/STATUS.md; }
until grep -qE "PREP_DONE|failed|refused" $D/prep_MIX.log; do sleep 30; done
grep -q PREP_DONE $D/prep_MIX.log || { log "FAILED: prep ($(tail -n1 $D/prep_MIX.log))"; st "FAILED at prep"; exit 1; }
log "prep done: $(grep 'packs:' $D/prep_MIX.log | tail -1 | cut -c1-200)"; st "prep done: $(grep -oE 'packs": [0-9]+.*' $D/prep_MIX.log | tail -1 | cut -c1-120)"
rm -f $D/verify_mix.rc $D/verify_mix.json
V=$(sbatch --parsable $D/verify_caches.sbatch "mix:$D/final_v2_mix_A_plus_B.jsonl") || { log "FAILED: verify submit"; exit 1; }
log "verify job $V"; echo $V > $D/mix_verify.jobid
until [ -f $D/verify_mix.rc ] || ! squeue -h -j $V | grep -q .; do sleep 30; done; sleep 5
if [ "$(cat $D/verify_mix.rc 2>/dev/null)" != 0 ] || ! grep -q '"verdict": "EQUAL"' $D/verify_mix.json; then log "FAILED: cache verify not EQUAL (rc=$(cat $D/verify_mix.rc 2>/dev/null))"; st "FAILED: cache verify not EQUAL; not training"; exit 1; fi
log "verify EQUAL: $(grep -E 'trained_tokens_trainer|packs' $D/verify_mix.json | tr -d ' \n')"; st "cache verify EQUAL ($(grep -E 'trained_tokens_trainer' $D/verify_mix.json | tr -d ' \n'))"
cd /e/project1/transfernetx/lee27/code/snowball && ARM=mix bash relay_sft_arm.sh 2>&1 | tee $D/train_MIX.log
grep -q ARM_DONE $D/train_MIX.log && log "MIX_LANE_DONE" || log "FAILED: lane ended without ARM_DONE"
