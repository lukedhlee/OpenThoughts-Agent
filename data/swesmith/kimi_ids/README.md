# Kimi SWE-smith id lists

The inputs `data/swesmith/kimi_sft_convert.py` needs besides the public traces
(`open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k` @ `1b87a9cf`). Every id is a
public SWE-smith instance id.

| file | what |
|---|---|
| `taskindex_to_instance.json` | the dataset's `task` column (`swesmith-NNNNN`, an internal index) to the SWE-smith instance id, 9,439 entries |
| `kimi_train_ids.txt` | 4,501 train instances: Stage-3 overlap, pygments and verifier errors removed, three repos held out, a cap per repo |
| `kimi_heldout_ids.txt` | 643 held-out instances: every trace of sqlparse, funcy and webargs |

The converter turns them into 4,398 train rows (66,269,429 tokens) and 628 held-out rows.
