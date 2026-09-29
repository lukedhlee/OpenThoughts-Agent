**The P10 hypothesis mostly fails.** The two "outright premature claims" came from the router, not the teacher. Its synthetic `task_complete` was scored P10 = 0: `teacher_budget` fired twice in guided and never in control. In software-engineering_20260610_111012_001, 60-120 s durations on quick commands used up the 600 s budget. In model-training_20260509_020109_002, the teacher polled a slow download for about 40 replies. The facts should mark router endings.

**claim** (vs v2)
- Adds a rule for when `task_complete` is allowed: the latest output after the last edit shows everything met, and nothing is failed, missing, running or unseen. The claim lists each check's result and what went unchecked. Also adds "never loosen a check" and "state as fact only what output has shown". Motivated by security_20260604_194036_001, security_20260603_181444_004, scientific-computing_20260617_104707_003, system-administration_20260416_020953_003 and model-training_20260613_152233_002. Also, 17 of 48 guided claims echoed v2's wording (control 2 of 47).
- Adds a duration and slow-download line (the two budget endings).
- Drops the "deliverable must not need it" clause, which was never cited.

**claimnote.** Same as claim, plus a confirmation-time check. Guided already re-checked at the confirmation in 23 of 48 claims (control 8).

**lean.** Same claim text, restructured for P5b. Of 41 replies over 8k tokens, about 15 drafted the whole script in thinking and then retyped it (software-engineering_20260424_090344_022). Others traced bytes or code by hand (software-engineering_20260610_111012_016, _20260509_065529_014). Lean adds "think only as far as the next command", "code straight into keystrokes" and "small edits, not rewrites", and repeats them in the reminder.
