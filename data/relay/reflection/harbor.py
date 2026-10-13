"""Import existing MSA2 Harbor trials without changing source artifacts."""
from pathlib import Path
import json
from .core import file_digest


def import_trials(roots, *, author, model, revision, policy_version=0):
    rows=[]; rejected={}; seen=set()
    def reject(why):
        rejected[why]=rejected.get(why,0)+1
    for root in roots:
        for path in sorted(Path(root).glob('*/result.json')):
            result=json.loads(path.read_text())
            sid=result.get('id') or str(path.parent.resolve())
            if sid in seen:
                reject('duplicate_result'); continue
            seen.add(sid)
            # Result metadata is the authoritative exact MSA message stream in this Harbor version.
            msgs=(result.get('agent_result') or {}).get('metadata',{}).get('all_messages')
            if not msgs:
                traj=path.parent/'agent/trajectory.json'
                if traj.exists():
                    t=json.loads(traj.read_text()); msgs=t.get('messages') if isinstance(t,dict) else t
            if not msgs or not any(m.get('role')=='assistant' for m in msgs):
                reject('no_experience'); continue
            task=result.get('task_name')
            task_path=(result.get('task_id') or {}).get('path')
            instruction=Path(task_path)/'instruction.md' if task_path else None
            if instruction and instruction.exists():
                task_text=instruction.read_text()
            else:
                task_text=next((m.get('content','') for m in msgs if m.get('role')=='user'),'')
            # Preserve a raw verifier reward independently of termination/errors. Never treat missing as failure.
            vr=result.get('verifier_result') or {}; rewards=vr.get('rewards') or {}
            reward=rewards.get('reward'); verdict='unknown' if reward is None else ('pass' if reward==1 else 'fail')
            feedback_path=path.parent/'verifier/test-stdout.txt'
            feedback=feedback_path.read_text() if feedback_path.exists() else ''
            error=result.get('exception_info') or {}
            termination=error.get('exception_type') or 'completed'
            rows.append(dict(id=sid, task=task, author=author, model=model,revision=revision,policy_version=policy_version,
                             task_description=task_text, messages=msgs, patch='',feedback={'verdict':verdict,'raw_reward':reward,'text':feedback},
                             termination=termination, source_path=str(path.resolve()),source_sha256=file_digest(path)))
    return rows, rejected
