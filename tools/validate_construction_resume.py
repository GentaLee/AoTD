"""Resume a reviewed existing construction journal briefly, preserving its prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from keepalive_guard import gpu_pids,proc


def prefix_digest(path,size):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining=size
        while remaining:
            block=stream.read(min(remaining,1024**2))
            if not block:raise ValueError('Journal prefix became shorter')
            digest.update(block);remaining-=len(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--qa',type=Path,required=True)
    parser.add_argument('--construction-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--new-records',type=int,default=2)
    args=parser.parse_args()
    root=Path(os.environ['AOTD_ROOT']);run=args.output_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    folder=args.construction_dir.resolve();journal=folder/'construct.jsonl';manifest=folder/'construct.manifest.json'
    old_manifest=manifest.read_bytes();before=journal.stat().st_size;before_sha=prefix_digest(journal,before)
    with journal.open() as stream:old_rows=[json.loads(line) for line in stream if line.strip()]
    old_ids={row['sample_id'] for row in old_rows};baseline={pid:proc(pid) for pid in gpu_pids()}
    if len(baseline)!=8:raise RuntimeError('Eight active original GPU workers required')
    python=str(root/'env/bin/python');keep=os.environ['AOTD_KEEPALIVE_SCRIPT']
    phase=run/'phase';phase.mkdir(exist_ok=True)
    command=[python,'-B',str(root/'tools/keepalive_guard.py'),'--keepalive-script',keep,
             '--state-dir',str(root/'runtime/guard-state'),'--',python,'-B',
             str(root/'tools/with_spare_keepalive.py'),'--mode','all','--keepalive-script',keep,
             '--state-dir',str(phase),'--','env','CUDA_VISIBLE_DEVICES=0,1,2,3',python,'-B',
             str(root/'tools/build_cot.py'),'--stage','construct','--qa',str(args.qa),
             '--output-dir',str(folder)]
    print(f'[construction resume] preserved prefix={len(old_rows)} rows; reviewed migration and new real records',flush=True)
    new={};appended=[]
    with (run/'run.log').open('w') as log:
        task=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+900
            while True:
                with journal.open('rb') as stream:
                    stream.seek(before);data=stream.read()
                appended=[]
                for line in data.splitlines(keepends=True):
                    if not line.endswith(b'\n'):continue
                    row=json.loads(line);appended.append(row)
                    if row['sample_id'] not in old_ids:new[row['sample_id']]=row
                if any(row.get('status')=='error' for row in new.values()):raise RuntimeError('A new operational failure occurred during resumed construction')
                if len(new)>=args.new_records and any(row.get('status')=='ok' for row in new.values()):
                    task.send_signal(signal.SIGTERM);code=task.wait(timeout=90);break
                if task.poll() is not None:raise RuntimeError(f'Construction exited early: {task.returncode}')
                if time.monotonic()>deadline:raise TimeoutError('Resumed construction failed to advance')
                time.sleep(.25)
        finally:
            if task.poll() is None:
                task.terminate()
                try:task.wait(timeout=45)
                except subprocess.TimeoutExpired:task.kill();task.wait()
    assert code==143,code
    with journal.open('rb') as stream:
        stream.seek(before);data=stream.read()
    appended=[json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b'\n')]
    new={row['sample_id']:row for row in appended if row['sample_id'] not in old_ids}
    assert not any(row.get('status')=='error' for row in new.values())
    assert prefix_digest(journal,before)==before_sha,'Previously saved journal bytes changed'
    assert gpu_pids()==set(baseline)
    assert all((p:=proc(pid)) and p['start']==old['start'] and p['state'] not in ('T','t','Z','X') for pid,old in baseline.items())
    latest={row['sample_id']:row for row in old_rows+appended}
    reclassified=[row for row in appended if row['sample_id'] in old_ids]
    report=dict(status='PASS',original_journal_rows=len(old_rows),original_unique_samples=len(old_ids),
                original_prefix_bytes=before,original_prefix_sha256=before_sha,prefix_preserved=True,
                original_manifest_sha256=hashlib.sha256(old_manifest).hexdigest(),
                reviewed_manifest_changed=manifest.read_bytes()!=old_manifest,
                reclassified_existing_records=len(reclassified),new_real_samples=len(new),
                new_operational_errors=0,current_unique_samples=len(latest),interrupted_exit=code,
                restoration=dict(original_workers=8,same_processes=True,duplicate_workers=0))
    (run/'construction_resume_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
