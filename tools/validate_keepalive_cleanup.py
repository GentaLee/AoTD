"""Exercise real failure and signal cleanup while preserving original GPU workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from keepalive_guard import gpu_pids, proc


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--keepalive-script',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    tools=Path(__file__).resolve().parent
    baseline={pid:proc(pid) for pid in gpu_pids()}
    if len(baseline)!=8 or any(p is None or p['state'] in ('T','t') for p in baseline.values()):
        raise RuntimeError('Expected eight active original GPU workers before validation')
    env=dict(os.environ,AOTD_TRAIN_GPUS='0,1,2,3',AOTD_EVAL_GPU='0')
    results={}

    def assert_restored():
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            same=gpu_pids()==set(baseline)
            active=all((p:=proc(pid)) and p['start']==identity['start'] and p['state'] not in ('T','t','Z','X')
                       for pid,identity in baseline.items())
            if same and active:return dict(original_workers=8,same_processes=True,all_active=True,duplicate_workers=0)
            time.sleep(.5)
        raise RuntimeError('Original eight-worker workload was not restored exactly')

    for case in ['failure_with_orphan','signal','guard_crash']:
        directory=args.output_dir/case;directory.mkdir(exist_ok=True)
        guard=[sys.executable,'-B',str(tools/'keepalive_guard.py'),'--keepalive-script',args.keepalive_script,
               '--state-dir',str(args.output_dir/'guard-state'),'--']
        if case=='failure_with_orphan':
            payload="import subprocess,sys; subprocess.Popen(['sleep','120'],start_new_session=True); sys.exit(23)"
            command=[*guard,sys.executable,'-B',str(tools/'with_spare_keepalive.py'),'--mode','all',
                     '--keepalive-script',args.keepalive_script,'--state-dir',str(directory),'--',
                     sys.executable,'-c',payload]
        else:
            command=[*guard,sys.executable,'-c',"import time; print('SIGNAL_READY',flush=True); time.sleep(120)"]
        with (directory/'run.log').open('w') as log:
            task=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                if case in ('signal','guard_crash'):
                    deadline=time.monotonic()+30
                    while task.poll() is None and 'SIGNAL_READY' not in (directory/'run.log').read_text():
                        if time.monotonic()>deadline:raise RuntimeError('Signal fixture failed to start')
                        time.sleep(.2)
                    task.send_signal(signal.SIGTERM if case=='signal' else signal.SIGKILL)
                code=task.wait(timeout=180)
            finally:
                if task.poll() is None:
                    task.terminate()
                    try:task.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        task.kill();task.wait()
        expected={'failure_with_orphan':23,'signal':143,'guard_crash':-9}[case]
        assert code==expected,(case,code,expected)
        results[case]=dict(expected_exit=expected,actual_exit=code,restoration=assert_restored())
    report=dict(status='PASS',tests=results,scope='real operational failure and signal cleanup; no training')
    (args.output_dir/'cleanup_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
