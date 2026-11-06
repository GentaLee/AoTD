"""Run AoTD alongside a supervised keep-alive on unused GPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', required=True)
    p.add_argument('--keepalive-script', required=True)
    p.add_argument('--state-dir', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    all_ids = set(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], text=True, timeout=15).split())
    train = os.environ.get('AOTD_TRAIN_GPUS', '0,1,2,3').split(',')
    eval_gpu = os.environ.get('AOTD_EVAL_GPU', '0')
    work = [eval_gpu] if a.mode == 'eval-only' else train
    if not work or len(set(work)) != len(work) or any(x not in all_ids for x in work):
        raise ValueError(f'invalid AoTD GPUs: {work}')
    if eval_gpu not in work:
        raise ValueError('Evaluation GPU must belong to the AoTD GPU selection')
    spare = sorted(all_ids - set(work), key=int)
    directory = Path(a.state_dir)
    log_path = directory/'spare_keepalive.log'
    state = dict(work_gpus=work, spare_gpus=spare, status='starting')
    def record():
        (directory/'gpu_roles.json').write_text(json.dumps(state, indent=2))
    print(f'[GPU roles] AoTD={",".join(work)} keep-alive={",".join(spare) or "none"}', flush=True)
    record()
    stopped = []
    def worker_failed():
        with log_path.open('rb') as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-8192))
            return b'worker_error' in f.read()
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, lambda s, f: stopped.append(s))
    sidecar = task = None
    def stop(child):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        # The outer guard also cleans tagged descendants, including elastic
        # workers in separate sessions, before restoring the original workload.
    try:
        with log_path.open('w') as log:
            if spare:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(spare),
                           KEEP_ALIVE_GPUS=','.join(map(str, range(len(spare)))),
                           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
                sidecar = subprocess.Popen([os.environ.get('AOTD_KEEPALIVE_PYTHON', '/opt/venv/bin/python'),
                    '-B', a.keepalive_script, '--gpus', env['KEEP_ALIVE_GPUS'], '--duration', '0',
                    '--log-interval', '10'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['keepalive_pid'] = sidecar.pid
                record()
                deadline = time.monotonic()+90
                while True:
                    text = log_path.read_text()
                    ready = set(re.findall(r'worker_ready gpu=(\d+)', text))
                    if stopped or sidecar.poll() is not None or 'worker_error' in text:
                        raise RuntimeError(f'spare GPU keep-alive failed; see {log_path}')
                    if ready == set(map(str, range(len(spare)))):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'spare GPU keep-alive startup timeout; see {log_path}')
                    time.sleep(0.5)
                state['ready_workers'] = len(ready)
                print(f'[keep-alive] {len(ready)} spare GPU workers ready; log={log_path}', flush=True)
            env = dict(os.environ, AOTD_WORK_GPUS=','.join(work), AOTD_SPARE_GPUS=','.join(spare))
            state['status'] = 'running'
            record()
            task = subprocess.Popen(command, env=env, start_new_session=True)
            while task.poll() is None:
                if stopped:
                    return 128+stopped[0]
                if sidecar is not None and (sidecar.poll() is not None or worker_failed()):
                    raise RuntimeError('spare GPU keep-alive exited during AoTD')
                time.sleep(0.5)
            state['task_exit_code'] = task.returncode
            return task.returncode if task.returncode >= 0 else 128-task.returncode
    finally:
        stop(task)
        stop(sidecar)
        state['status'] = 'temporary_keepalive_stopped'
        record()
        print('[keep-alive] temporary workers stopped; outer guard will restore original eight-GPU workload', flush=True)


if __name__ == '__main__':
    sys.exit(main())
