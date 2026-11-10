"""Evaluate independent models concurrently; retain each evaluator's durable journal."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run_jobs(jobs, gpus, interval=30):
    if not gpus or len(gpus)!=len(set(gpus)):
        raise ValueError('Evaluation devices must be nonempty and unique')
    pending=iter(jobs);active={};completed=[]
    try:
        for gpu in gpus:
            job=next(pending,None)
            if job is not None:
                process=subprocess.Popen(job['command'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
                active[gpu]=(process,job)
        last=time.monotonic()
        while active:
            for gpu,(process,job) in list(active.items()):
                code=process.poll()
                if code is None:continue
                if code:raise RuntimeError(f'Evaluation {job["tag"]} failed with exit code {code}')
                completed.append(job['tag']);del active[gpu]
                next_job=next(pending,None)
                if next_job is not None:
                    process=subprocess.Popen(next_job['command'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
                    active[gpu]=(process,next_job)
            if time.monotonic()-last>=interval:
                print(f'[parallel eval] completed={completed} active='+str({g:job['tag'] for g,(_,job) in active.items()}),flush=True)
                last=time.monotonic()
            time.sleep(.2)
        return completed
    finally:
        for process,_ in active.values():
            if process.poll() is None:process.terminate()
        for process,_ in active.values():
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--tags',default='author,base,instruct,aotd')
    p.add_argument('--validation-limit',type=int)
    p.add_argument('--validation-output',type=Path)
    args=p.parse_args()
    if (args.validation_limit is None)!=(args.validation_output is None):
        p.error('Validation requires both limit and a separate output directory')
    if args.validation_output is not None and args.validation_output.resolve()==args.run.resolve():
        p.error('Validation outputs must be separate from full outputs')
    destination=args.validation_output or args.run;destination.mkdir(parents=True,exist_ok=True)
    split=os.environ.get('AOTD_EVAL_SPLIT','val')
    paths={'author':ROOT/'models/AoTD-7B-author','base':ROOT/'models/LLaVA-7B-base',
           'instruct':args.run/'instruct_s42','aotd':args.run/'aotd_s42'}
    tags=args.tags.split(',')
    if len(set(tags))!=len(tags) or any(t not in paths for t in tags):raise ValueError('Invalid evaluation tags')
    jobs=[]
    for tag in tags:
        command=[sys.executable,'-B',str(ROOT/'evaluation/eval_nextqa.py'),
                 '--model_path',str(paths[tag]),'--data_dir',str(ROOT/f'data/nextqa/{split}.csv'),
                 '--video_dir',os.environ.get('AOTD_VIDEO_DIR',str(ROOT/'data/nextqa/NExTVideo')),
                 '--split',split,'--attn_implementation','sdpa','--save_dir',str(destination/f'{tag}_{split}_full.json')]
        if args.validation_limit is not None:command+=['--limit',str(args.validation_limit)]
        jobs.append(dict(tag=tag,command=command))
    def stop(signum,_):raise SystemExit(128+signum)
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
    run_jobs(jobs,os.environ.get('AOTD_TRAIN_GPUS','0,1,2,3').split(','),
             int(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS','30')))
    for tag in tags:
        result=json.loads((destination/f'{tag}_{split}_full.json').read_text())
        if args.validation_limit is None and not result['meta']['full_dataset']:
            raise RuntimeError('Incomplete evaluation was not a full result')

if __name__=='__main__':main()
