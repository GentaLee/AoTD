"""Interrupt a real two-question evaluation, resume its suffix, then skip on CPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from keepalive_guard import gpu_pids,proc


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(os.environ['AOTD_ROOT']);run=args.output_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    python=str(root/'env/bin/python');keep=os.environ['AOTD_KEEPALIVE_SCRIPT']
    baseline={pid:proc(pid) for pid in gpu_pids()}
    if len(baseline)!=8:raise RuntimeError('Eight original GPU workers required')
    output=run/'evaluation.json';journal=output.with_suffix('.jsonl')
    command=[python,'-u',str(root/'evaluation/eval_nextqa.py'),'--model_path',str(args.model_path),
             '--data_dir',str(root/'data/nextqa/val.csv'),'--video_dir',str(root/'data/nextqa/NExTVideo'),
             '--split','val','--limit','2','--attn_implementation','sdpa','--save_dir',str(output)]
    def wrapped(name):
        phase=run/name;phase.mkdir(exist_ok=True)
        return [python,'-B',str(root/'tools/keepalive_guard.py'),'--keepalive-script',keep,
                '--state-dir',str(root/'runtime/guard-state'),'--',python,'-B',
                str(root/'tools/with_spare_keepalive.py'),'--mode','eval-only','--keepalive-script',keep,
                '--state-dir',str(phase),'--','env','CUDA_VISIBLE_DEVICES=0',*command]
    def restored():
        assert gpu_pids()==set(baseline)
        assert all((p:=proc(pid)) and p['start']==old['start'] and p['state'] not in ('T','t','Z','X') for pid,old in baseline.items())
        return dict(original_workers=8,same_processes=True,duplicate_workers=0)
    with (run/'interrupted.log').open('w') as log:
        task=subprocess.Popen(wrapped('interrupted'),stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+600
            while not journal.exists() or not journal.read_bytes().endswith(b'\n'):
                if task.poll() is not None:raise RuntimeError('Evaluation exited before its first durable record')
                if time.monotonic()>deadline:raise TimeoutError('First evaluation record timeout')
                time.sleep(.02)
            prefix=journal.read_bytes();task.send_signal(signal.SIGTERM);code=task.wait(timeout=90)
        finally:
            if task.poll() is None:task.terminate();task.wait(timeout=45)
    assert code==143,code
    assert len(journal.read_text().splitlines())==1,'Interruption arrived after the two-question run finished'
    first=restored()
    with (run/'resumed.log').open('w') as log:
        subprocess.run(wrapped('resumed'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    second=restored();result=json.loads(output.read_text())
    assert result['summary']['total']==2 and result['meta']['resumed_records']==1,result['meta']
    assert journal.read_bytes().startswith(prefix)
    ids=[json.loads(line)['sample_id'] for line in journal.read_text().splitlines()]
    assert len(ids)==len(set(ids))==2
    cpu=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,check=True,timeout=600)
    assert 'no GPU load' in cpu.stdout,cpu.stdout
    output.replace(run/'evaluation.before_reconstruction.json')
    cpu=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,check=True,timeout=600)
    rebuilt=json.loads(output.read_text())
    assert 'no GPU load' in cpu.stdout and rebuilt['meta']['recovered_complete_prefix'],cpu.stdout
    assert rebuilt['summary']==result['summary']
    report=dict(status='PASS',scope='two genuine evaluation questions; signal after first durable prediction',
                interrupted_exit=code,resumed_records=1,final_records=2,prefix_bytes_preserved=True,
                unique_sample_ids=True,completed_skip_with_cuda_hidden=True,
                complete_prefix_reconstruction_with_cuda_hidden=True,
                interruption_restoration=first,resume_restoration=second)
    (run/'evaluation_resume_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
